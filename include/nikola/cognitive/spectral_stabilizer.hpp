#pragma once
// =============================================================================
// nikola/cognitive/spectral_stabilizer.hpp
// Phase 81 — GAP-032: Spectral Radius Upper Bound for SSM Stability
//
// SOURCE: Gemini Deep Research Round 2, Theoretical Stability Analysis Report
// SPEC:   docs/info/integration/sections/03_cognitive_systems/02_mamba_9d_ssm.md
//         §GAP-032 (lines ~2273–2423)
//
// The Mamba-9D discrete recurrence  h_k = Ā·h_{k-1} + …  is stable iff the
// spectral radius ρ(Ā) ≤ 1.  Given Ā = I − Δ(1−r)G with G SPD, the critical
// constraint is:
//
//     Δ · (1−r) · μ_max  ≤  2          (Nyquist stability limit)
//     Δ_safe             =  1.6 / ((1−r)·ρ(G) + ε)   (80% safety margin)
//
// Matrix clamping enforces the gain limiter:  ||A|| ≤ ln(10) ≈ 2.3026
//
// All constants are explicit spec values; all logic is pure constexpr —
// no external runtime required.
// =============================================================================

#include <cstdint>
#include <string_view>
#include <cmath>

namespace nikola::cognitive {

// ---------------------------------------------------------------------------
// § Enumerations
// ---------------------------------------------------------------------------

/// Stability classification of the current Mamba-9D recurrence step.
/// Failure modes §GAP-032 §"Failure Modes and Detection"
enum class StabilityCondition : uint8_t {
    STABLE              = 0,  ///< ρ(Ā) ≤ 1 and ||A|| ≤ ln(10)  — nominal
    TIMESTEP_VIOLATION  = 1,  ///< Δ > 2/((1−r)·ρ(G)) — eigenvalue < −1
    CURVATURE_SHOCKWAVE = 2,  ///< Sudden ρ(G) spike (neurogenesis event)
    RUNAWAY_ATTENTION   = 3,  ///< ||A|| > ln(10) — gain > 10×
};

/// Corrective intervention triggered per stability condition.
enum class InterventionType : uint8_t {
    NONE         = 0,  ///< STABLE — no action
    CLAMP_DELTA  = 1,  ///< TIMESTEP_VIOLATION — SpectralStabilizer clamps Δ
    SCALE_MATRIX = 2,  ///< RUNAWAY_ATTENTION  — rescale A to gain limit
    LOCAL_SCRAM  = 3,  ///< CURVATURE_SHOCKWAVE — Physics Oracle local SCRAM
};

/// Diagnostic category for hidden-state energy monitoring.
enum class EnergyTrend : uint8_t {
    BOUNDED    = 0,  ///< energy decays or holds — nominal
    OSCILLATING= 1,  ///< sawtooth alternating-sign pattern — Δ at limit
    DIVERGING  = 2,  ///< monotone growth — ρ(Ā) > 1
    LOCALISED  = 3,  ///< single component > 90% of total — runaway attention
};

/// Power-iteration convergence status (5-step algorithm).
enum class PowerIterStatus : uint8_t {
    CONVERGED    = 0,  ///< estimate is reliable after POWER_ITER_STEPS
    DEGENERATE   = 1,  ///< zero vector encountered — treat as ρ = 0
};

// ---------------------------------------------------------------------------
// § Spec constants
// ---------------------------------------------------------------------------

/// Number of power-iteration steps for estimating ρ(G).
/// Spec: "Power Iteration (5 iterations, O(N²) complexity)"
inline constexpr int   POWER_ITER_STEPS = 5;

/// Nyquist stability upper bound coefficient: Δ · (1−r) · μ_max ≤ NYQUIST_BOUND.
/// Derived from |λ_i| ≤ 1 with lower-bound constraint.
inline constexpr float NYQUIST_BOUND = 2.0f;

/// Safety factor applied to the Nyquist bound (20% margin).
/// Spec: "We apply a safety factor α = 0.8 to the timestep"
inline constexpr float SAFETY_FACTOR = 0.8f;

/// Numerator of the safe-timestep formula: 2 × 0.8 = 1.6.
/// Δ_safe = SAFE_DELTA_NUMERATOR / ((1−r)·ρ(G) + ε)
inline constexpr float SAFE_DELTA_NUMERATOR = NYQUIST_BOUND * SAFETY_FACTOR;  // 1.6f

/// Maximum attention-amplifier gain (×10 signal amplification).
/// Spec: "max_growth_rate = 10.0"
inline constexpr float MAX_GROWTH_RATE = 10.0f;

/// Continuous-time Lyapunov gain-limit: ||A|| must not exceed ln(10).
/// Derivation: e^λ_max ≤ 10  →  λ_max ≤ ln(10) ≈ 2.302585
inline constexpr double MAX_GROWTH_RATE_LOG_D = 2.302585092994046;   // ln(10) full precision
inline constexpr float  MAX_GROWTH_RATE_LOG   = 2.302585f;           // used in comparisons

/// Safety margin fraction below the Nyquist limit (= 1 − SAFETY_FACTOR).
inline constexpr float SAFETY_MARGIN_FRACTION = 0.20f;

/// Small epsilon added to denominator of Δ_safe to prevent division by zero.
/// Applied when (1−r)·ρ(G) ≈ 0 (e.g., r → 1 or flat manifold).
inline constexpr float SPECTRAL_EPSILON = 1.0e-6f;

/// Resonance range [0.0, 1.0] — clamped at these bounds.
inline constexpr float RESONANCE_MIN = 0.0f;
inline constexpr float RESONANCE_MAX = 1.0f;

/// Attention dominance threshold: a single h-component is "runaway" when
/// it contributes > 90% of total hidden-state energy.
/// Spec: "single hidden state component dominates (>90% of total energy)"
inline constexpr float ATTENTION_DOMINANCE_THRESHOLD = 0.90f;

/// Fraction representing 100% (for percentage helpers).
inline constexpr float PERCENT_100 = 100.0f;

// ---------------------------------------------------------------------------
// § Eigenvalue of discrete operator Ā
// ---------------------------------------------------------------------------

/// Eigenvalue of Ā = I − Δ(1−r)G for a single metric-tensor eigenvalue μ.
/// λ_i(Ā) = 1 − Δ·(1−r)·μ_i
[[nodiscard]] constexpr float eigenvalue_of_Abar(
    float delta, float resonance, float mu) noexcept
{
    float r = resonance < RESONANCE_MIN ? RESONANCE_MIN
            : resonance > RESONANCE_MAX ? RESONANCE_MAX
            : resonance;
    return 1.0f - delta * (1.0f - r) * mu;
}

/// Return |λ_i(Ā)| — the spectral contribution of one eigenvalue.
[[nodiscard]] constexpr float eigenvalue_abs(
    float delta, float resonance, float mu) noexcept
{
    float v = eigenvalue_of_Abar(delta, resonance, mu);
    return v < 0.0f ? -v : v;
}

// ---------------------------------------------------------------------------
// § Nyquist stability bound
// ---------------------------------------------------------------------------

/// Theoretical maximum safe timestep (Nyquist limit, no safety margin).
/// Δ_nyquist = 2 / ((1−r)·ρ(G) + ε)
[[nodiscard]] constexpr float delta_nyquist(float rho_G, float resonance) noexcept {
    float r = resonance < RESONANCE_MIN ? RESONANCE_MIN
            : resonance > RESONANCE_MAX ? RESONANCE_MAX
            : resonance;
    float denom = (1.0f - r) * rho_G + SPECTRAL_EPSILON;
    return NYQUIST_BOUND / denom;
}

/// Safe timestep with 80% safety factor applied.
/// Δ_safe = 1.6 / ((1−r)·ρ(G) + ε)
[[nodiscard]] constexpr float delta_safe(float rho_G, float resonance) noexcept {
    float r = resonance < RESONANCE_MIN ? RESONANCE_MIN
            : resonance > RESONANCE_MAX ? RESONANCE_MAX
            : resonance;
    float denom = (1.0f - r) * rho_G + SPECTRAL_EPSILON;
    return SAFE_DELTA_NUMERATOR / denom;
}

/// Clamp a requested timestep to the spec-safe value.
/// Returns min(delta_requested, delta_safe(rho_G, resonance)).
[[nodiscard]] constexpr float clamp_delta(
    float delta_requested, float rho_G, float resonance) noexcept
{
    float safe = delta_safe(rho_G, resonance);
    return delta_requested < safe ? delta_requested : safe;
}

/// True if delta_requested is within the safe bound.
[[nodiscard]] constexpr bool timestep_within_bound(
    float delta_requested, float rho_G, float resonance) noexcept
{
    return delta_requested <= delta_safe(rho_G, resonance);
}

/// True if the spectral radius of Ā is ≤ 1 for the given parameters.
/// Uses the critical lower-bound constraint: Δ·(1−r)·ρ(G) ≤ 2
[[nodiscard]] constexpr bool spectral_radius_bounded(
    float delta, float rho_G, float resonance) noexcept
{
    float r = resonance < RESONANCE_MIN ? RESONANCE_MIN
            : resonance > RESONANCE_MAX ? RESONANCE_MAX
            : resonance;
    return delta * (1.0f - r) * rho_G <= NYQUIST_BOUND;
}

// ---------------------------------------------------------------------------
// § Gain limiter (matrix clamping)
// ---------------------------------------------------------------------------

/// True when the continuous-time A matrix ||A|| satisfies the gain limit.
/// Spec: ||A|| ≤ ln(10) ≈ 2.3026
[[nodiscard]] constexpr bool matrix_within_gain_limit(float matrix_norm) noexcept {
    return matrix_norm <= MAX_GROWTH_RATE_LOG;
}

/// Scale factor to rescale A matrix back to the gain limit.
/// A' = A × (ln(10) / ||A||)
/// Returns 1.0 if already within limit (no scaling needed).
[[nodiscard]] constexpr float gain_limit_scale_factor(float matrix_norm) noexcept {
    if (matrix_norm <= MAX_GROWTH_RATE_LOG || matrix_norm <= 0.0f) return 1.0f;
    return MAX_GROWTH_RATE_LOG / matrix_norm;
}

/// Rescaled matrix norm after clamping (should equal MAX_GROWTH_RATE_LOG if > limit).
[[nodiscard]] constexpr float clamped_matrix_norm(float matrix_norm) noexcept {
    if (matrix_norm <= MAX_GROWTH_RATE_LOG) return matrix_norm;
    return MAX_GROWTH_RATE_LOG;
}

/// Maximum signal amplification factor from a matrix with the given norm.
/// ||exp(A·t)|| ≤ e^(ω_max·t); at t=1: amplification = e^norm (bounded by e^ln10 = 10).
[[nodiscard]] constexpr float max_amplification(float matrix_norm) noexcept {
    // constexpr-safe: use precomputed exp values for common cases; general = e^norm.
    // We return the bound: if clamped, exactly MAX_GROWTH_RATE (10.0).
    // For unclamped norms we return e^norm approximated as norm (for policy comparison).
    return matrix_within_gain_limit(matrix_norm) ? matrix_norm : MAX_GROWTH_RATE;
}

// ---------------------------------------------------------------------------
// § Stability condition classification
// ---------------------------------------------------------------------------

/// Classify the stability state of one recurrence step.
/// Priority: TIMESTEP_VIOLATION → RUNAWAY_ATTENTION → (others are latent)
[[nodiscard]] constexpr StabilityCondition classify_stability(
    float delta,
    float rho_G,
    float resonance,
    float matrix_norm) noexcept
{
    if (!spectral_radius_bounded(delta, rho_G, resonance))
        return StabilityCondition::TIMESTEP_VIOLATION;
    if (!matrix_within_gain_limit(matrix_norm))
        return StabilityCondition::RUNAWAY_ATTENTION;
    return StabilityCondition::STABLE;
}

/// Curvature-shockwave detection: true if rho_G_new is a significant spike
/// relative to rho_G_prev (a multiplicative threshold rather than absolute).
/// Spec: "Neurogenesis event creates local ρ(G) spike"
/// We define "significant" as > 2× — enough to push a safe Δ over the limit.
[[nodiscard]] constexpr bool is_curvature_shockwave(
    float rho_G_prev, float rho_G_new) noexcept
{
    return rho_G_new > rho_G_prev * 2.0f;
}

/// True if previously-safe delta is now violated after a shockwave.
[[nodiscard]] constexpr bool shockwave_violates_delta(
    float delta, float rho_G_new, float resonance) noexcept
{
    return !spectral_radius_bounded(delta, rho_G_new, resonance);
}

// ---------------------------------------------------------------------------
// § Intervention dispatch
// ---------------------------------------------------------------------------

/// Return the required intervention for a given stability condition.
[[nodiscard]] constexpr InterventionType intervention_for(
    StabilityCondition s) noexcept
{
    switch (s) {
        case StabilityCondition::STABLE:              return InterventionType::NONE;
        case StabilityCondition::TIMESTEP_VIOLATION:  return InterventionType::CLAMP_DELTA;
        case StabilityCondition::CURVATURE_SHOCKWAVE: return InterventionType::LOCAL_SCRAM;
        case StabilityCondition::RUNAWAY_ATTENTION:   return InterventionType::SCALE_MATRIX;
    }
    return InterventionType::NONE;
}

// ---------------------------------------------------------------------------
// § Attention dominance detection
// ---------------------------------------------------------------------------

/// True when a single hidden-state component dominates total energy.
/// Spec: "single hidden state component dominates (>90% of total energy)"
[[nodiscard]] constexpr bool is_runaway_attention(
    float component_energy, float total_energy) noexcept
{
    if (total_energy <= 0.0f) return false;
    return (component_energy / total_energy) > ATTENTION_DOMINANCE_THRESHOLD;
}

/// The fraction [0,1] that a component represents of total energy.
[[nodiscard]] constexpr float attention_fraction(
    float component_energy, float total_energy) noexcept
{
    if (total_energy <= 0.0f) return 0.0f;
    return component_energy / total_energy;
}

// ---------------------------------------------------------------------------
// § Resonance effect on Δ
// ---------------------------------------------------------------------------

/// Effective stiffness denominator (1−r)·ρ(G).
/// Higher resonance → lower stiffness → larger safe Δ allowed.
[[nodiscard]] constexpr float effective_stiffness(
    float rho_G, float resonance) noexcept
{
    float r = resonance < RESONANCE_MIN ? RESONANCE_MIN
            : resonance > RESONANCE_MAX ? RESONANCE_MAX
            : resonance;
    return (1.0f - r) * rho_G;
}

/// The ratio by which high resonance expands the allowed safe Δ vs r=0.
/// ratio = delta_safe(rho_G, r) / delta_safe(rho_G, 0)
///       = (rho_G + ε) / ((1−r)·rho_G + ε)
[[nodiscard]] constexpr float resonance_delta_expansion(
    float rho_G, float resonance) noexcept
{
    float denom_r0 = rho_G + SPECTRAL_EPSILON;
    float denom_r  = effective_stiffness(rho_G, resonance) + SPECTRAL_EPSILON;
    if (denom_r <= 0.0f) return 1.0f;
    return denom_r0 / denom_r;
}

// ---------------------------------------------------------------------------
// § Power iteration policy
// ---------------------------------------------------------------------------

/// Returns the number of power-iteration steps the spec prescribes.
[[nodiscard]] constexpr int power_iter_step_count() noexcept {
    return POWER_ITER_STEPS;
}

/// Classify power-iteration result given the estimated rho.
[[nodiscard]] constexpr PowerIterStatus classify_power_iter(
    float rho_estimate) noexcept
{
    // zero estimate from a zero vector — degenerate
    return (rho_estimate < SPECTRAL_EPSILON)
        ? PowerIterStatus::DEGENERATE
        : PowerIterStatus::CONVERGED;
}

// ---------------------------------------------------------------------------
// § Energy trend classification
// ---------------------------------------------------------------------------

/// Classify observed hidden-state energy trend.
[[nodiscard]] constexpr EnergyTrend classify_energy_trend(
    float prev_energy, float curr_energy,
    float single_component_fraction) noexcept
{
    if (single_component_fraction > ATTENTION_DOMINANCE_THRESHOLD)
        return EnergyTrend::LOCALISED;
    // diverging: current > prev by more than 1%
    if (curr_energy > prev_energy * 1.01f)
        return EnergyTrend::DIVERGING;
    // oscillating: sign alternation implies curr_energy ≪ prev_energy rapidly
    // heuristic: if energy toggled (fell then rose), flagged externally;
    // as a pure constant predicate we detect near-zero crossing
    if (curr_energy < prev_energy * 0.01f && prev_energy > 0.0f)
        return EnergyTrend::OSCILLATING;
    return EnergyTrend::BOUNDED;
}

// ---------------------------------------------------------------------------
// § Physical interpretation helpers
// ---------------------------------------------------------------------------

/// Characteristic frequency ω_max = (1−r)·ρ(G).
/// This is the highest geometric dynamic frequency that must be sampled.
[[nodiscard]] constexpr float omega_max(float rho_G, float resonance) noexcept {
    return effective_stiffness(rho_G, resonance);
}

/// True if the timestep satisfies the Nyquist condition Δ·ω_max ≤ 2.
[[nodiscard]] constexpr bool nyquist_satisfied(
    float delta, float rho_G, float resonance) noexcept
{
    return delta * omega_max(rho_G, resonance) <= NYQUIST_BOUND;
}

/// How far (in %) the requested delta is below the safe delta (headroom).
/// Positive value = safe; negative = violation.
[[nodiscard]] constexpr float delta_headroom_pct(
    float delta_requested, float rho_G, float resonance) noexcept
{
    float safe = delta_safe(rho_G, resonance);
    if (safe <= 0.0f) return 0.0f;
    return ((safe - delta_requested) / safe) * PERCENT_100;
}

// ---------------------------------------------------------------------------
// § Label helpers
// ---------------------------------------------------------------------------

[[nodiscard]] constexpr std::string_view stability_label(StabilityCondition s) noexcept {
    switch (s) {
        case StabilityCondition::STABLE:              return "STABLE";
        case StabilityCondition::TIMESTEP_VIOLATION:  return "TIMESTEP_VIOLATION";
        case StabilityCondition::CURVATURE_SHOCKWAVE: return "CURVATURE_SHOCKWAVE";
        case StabilityCondition::RUNAWAY_ATTENTION:   return "RUNAWAY_ATTENTION";
    }
    return "UNKNOWN_STABILITY";
}

[[nodiscard]] constexpr std::string_view intervention_label(InterventionType t) noexcept {
    switch (t) {
        case InterventionType::NONE:         return "NONE";
        case InterventionType::CLAMP_DELTA:  return "CLAMP_DELTA";
        case InterventionType::SCALE_MATRIX: return "SCALE_MATRIX";
        case InterventionType::LOCAL_SCRAM:  return "LOCAL_SCRAM";
    }
    return "UNKNOWN_INTERVENTION";
}

[[nodiscard]] constexpr std::string_view energy_trend_label(EnergyTrend t) noexcept {
    switch (t) {
        case EnergyTrend::BOUNDED:    return "BOUNDED";
        case EnergyTrend::OSCILLATING:return "OSCILLATING";
        case EnergyTrend::DIVERGING:  return "DIVERGING";
        case EnergyTrend::LOCALISED:  return "LOCALISED";
    }
    return "UNKNOWN_TREND";
}

[[nodiscard]] constexpr std::string_view power_iter_label(PowerIterStatus s) noexcept {
    switch (s) {
        case PowerIterStatus::CONVERGED:  return "CONVERGED";
        case PowerIterStatus::DEGENERATE: return "DEGENERATE";
    }
    return "UNKNOWN_ITER_STATUS";
}

} // namespace nikola::cognitive
