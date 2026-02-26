// =============================================================================
// phase81_spectral_stabilizer_test.cpp
// Phase 81 — GAP-032: Spectral Radius Upper Bound for SSM Stability
//
// Exhaustively tests every constant, formula, and pure function in
// nikola/cognitive/spectral_stabilizer.hpp against the mathematical
// derivations in §GAP-032 of 02_mamba_9d_ssm.md.
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "nikola/cognitive/spectral_stabilizer.hpp"
#include <cmath>

using namespace nikola::cognitive;
using Catch::Approx;

// ---------------------------------------------------------------------------
// §1 — Spec constants
// ---------------------------------------------------------------------------

TEST_CASE("POWER_ITER_STEPS is 5", "[constants]") {
    CHECK(POWER_ITER_STEPS == 5);
}

TEST_CASE("NYQUIST_BOUND is 2.0", "[constants]") {
    CHECK(NYQUIST_BOUND == Approx(2.0f));
}

TEST_CASE("SAFETY_FACTOR is 0.8 (20% margin)", "[constants]") {
    CHECK(SAFETY_FACTOR == Approx(0.8f));
}

TEST_CASE("SAFE_DELTA_NUMERATOR is 1.6 = 2 * 0.8", "[constants]") {
    CHECK(SAFE_DELTA_NUMERATOR == Approx(1.6f));
    CHECK(SAFE_DELTA_NUMERATOR == Approx(NYQUIST_BOUND * SAFETY_FACTOR));
}

TEST_CASE("MAX_GROWTH_RATE is 10.0", "[constants]") {
    CHECK(MAX_GROWTH_RATE == Approx(10.0f));
}

TEST_CASE("MAX_GROWTH_RATE_LOG is ln(10) approximately 2.302585", "[constants]") {
    CHECK(MAX_GROWTH_RATE_LOG == Approx(2.302585f).epsilon(1e-5f));
}

TEST_CASE("MAX_GROWTH_RATE_LOG_D matchesln(10) double precision", "[constants]") {
    CHECK(MAX_GROWTH_RATE_LOG_D == Approx(std::log(10.0)).epsilon(1e-12));
}

TEST_CASE("MAX_GROWTH_RATE_LOG equals ln(MAX_GROWTH_RATE)", "[constants]") {
    // e^ln(10) = 10 — core derivation
    CHECK(std::exp(MAX_GROWTH_RATE_LOG_D) == Approx(10.0).epsilon(1e-9));
}

TEST_CASE("SAFETY_MARGIN_FRACTION is 0.20", "[constants]") {
    CHECK(SAFETY_MARGIN_FRACTION == Approx(0.20f));
    CHECK(SAFETY_MARGIN_FRACTION == Approx(1.0f - SAFETY_FACTOR));
}

TEST_CASE("SPECTRAL_EPSILON is 1e-6", "[constants]") {
    CHECK(SPECTRAL_EPSILON == Approx(1.0e-6f));
}

TEST_CASE("RESONANCE bounds are 0 and 1", "[constants]") {
    CHECK(RESONANCE_MIN == Approx(0.0f));
    CHECK(RESONANCE_MAX == Approx(1.0f));
}

TEST_CASE("ATTENTION_DOMINANCE_THRESHOLD is 0.90", "[constants]") {
    CHECK(ATTENTION_DOMINANCE_THRESHOLD == Approx(0.90f));
}

// ---------------------------------------------------------------------------
// §2 — Enum ordinals
// ---------------------------------------------------------------------------

TEST_CASE("StabilityCondition ordinals", "[enums]") {
    CHECK(static_cast<uint8_t>(StabilityCondition::STABLE)              == 0);
    CHECK(static_cast<uint8_t>(StabilityCondition::TIMESTEP_VIOLATION)  == 1);
    CHECK(static_cast<uint8_t>(StabilityCondition::CURVATURE_SHOCKWAVE) == 2);
    CHECK(static_cast<uint8_t>(StabilityCondition::RUNAWAY_ATTENTION)   == 3);
}

TEST_CASE("InterventionType ordinals", "[enums]") {
    CHECK(static_cast<uint8_t>(InterventionType::NONE)        == 0);
    CHECK(static_cast<uint8_t>(InterventionType::CLAMP_DELTA) == 1);
    CHECK(static_cast<uint8_t>(InterventionType::SCALE_MATRIX)== 2);
    CHECK(static_cast<uint8_t>(InterventionType::LOCAL_SCRAM) == 3);
}

TEST_CASE("EnergyTrend ordinals", "[enums]") {
    CHECK(static_cast<uint8_t>(EnergyTrend::BOUNDED)    == 0);
    CHECK(static_cast<uint8_t>(EnergyTrend::OSCILLATING)== 1);
    CHECK(static_cast<uint8_t>(EnergyTrend::DIVERGING)  == 2);
    CHECK(static_cast<uint8_t>(EnergyTrend::LOCALISED)  == 3);
}

TEST_CASE("PowerIterStatus ordinals", "[enums]") {
    CHECK(static_cast<uint8_t>(PowerIterStatus::CONVERGED)  == 0);
    CHECK(static_cast<uint8_t>(PowerIterStatus::DEGENERATE) == 1);
}

// ---------------------------------------------------------------------------
// §3 — eigenvalue_of_Abar: λ_i = 1 − Δ(1−r)μ
// ---------------------------------------------------------------------------

TEST_CASE("eigenvalue_of_Abar: Δ=1, r=0, μ=1 → 0.0", "[eigenvalue]") {
    CHECK(eigenvalue_of_Abar(1.0f, 0.0f, 1.0f) == Approx(0.0f));
}

TEST_CASE("eigenvalue_of_Abar: Δ=0, any r, any μ → 1.0", "[eigenvalue]") {
    CHECK(eigenvalue_of_Abar(0.0f, 0.0f, 5.0f) == Approx(1.0f));
    CHECK(eigenvalue_of_Abar(0.0f, 0.5f, 5.0f) == Approx(1.0f));
}

TEST_CASE("eigenvalue_of_Abar: Δ=2, r=0, μ=1 → -1 (stability limit)", "[eigenvalue]") {
    CHECK(eigenvalue_of_Abar(2.0f, 0.0f, 1.0f) == Approx(-1.0f));
}

TEST_CASE("eigenvalue_of_Abar: Δ=2.5, r=0, μ=1 → -1.5 (violation)", "[eigenvalue]") {
    CHECK(eigenvalue_of_Abar(2.5f, 0.0f, 1.0f) == Approx(-1.5f));
}

TEST_CASE("eigenvalue_of_Abar: r=0.5 halves stiffness contribution", "[eigenvalue]") {
    // λ = 1 - 1*(1-0.5)*2 = 1 - 1 = 0
    CHECK(eigenvalue_of_Abar(1.0f, 0.5f, 2.0f) == Approx(0.0f));
}

TEST_CASE("eigenvalue_of_Abar: r=1 → always 1 (frozen state — max resonance)", "[eigenvalue]") {
    CHECK(eigenvalue_of_Abar(10.0f, 1.0f, 100.0f) == Approx(1.0f));
}

TEST_CASE("eigenvalue_abs: negative eigenvalue gives correct magnitude", "[eigenvalue]") {
    // λ = 1 - 2*1*1 = -1, |λ| = 1
    CHECK(eigenvalue_abs(2.0f, 0.0f, 1.0f) == Approx(1.0f));
    // λ = 1 - 2.5*1*1 = -1.5, |λ| = 1.5
    CHECK(eigenvalue_abs(2.5f, 0.0f, 1.0f) == Approx(1.5f));
}

TEST_CASE("eigenvalue_of_Abar: resonance clamped to [0,1]", "[eigenvalue]") {
    // r=-0.5 → clamped to 0 → same as r=0
    CHECK(eigenvalue_of_Abar(1.0f, -0.5f, 1.0f) == Approx(eigenvalue_of_Abar(1.0f, 0.0f, 1.0f)));
    // r=2.0 → clamped to 1 → always 1
    CHECK(eigenvalue_of_Abar(1.0f, 2.0f, 1.0f) == Approx(1.0f));
}

// ---------------------------------------------------------------------------
// §4 — delta_nyquist and delta_safe
// ---------------------------------------------------------------------------

TEST_CASE("delta_nyquist: r=0, rho=1 → 2.0 (ignoring epsilon)", "[delta]") {
    CHECK(delta_nyquist(1.0f, 0.0f) == Approx(2.0f).epsilon(1e-4f));
}

TEST_CASE("delta_nyquist: r=0, rho=2 → 1.0", "[delta]") {
    CHECK(delta_nyquist(2.0f, 0.0f) == Approx(1.0f).epsilon(1e-4f));
}

TEST_CASE("delta_nyquist: r=0.5, rho=1 → 4.0 (stiffness halved)", "[delta]") {
    // 2 / (0.5 * 1) = 4
    CHECK(delta_nyquist(1.0f, 0.5f) == Approx(4.0f).epsilon(1e-3f));
}

TEST_CASE("delta_safe: always equals 0.8 * delta_nyquist", "[delta]") {
    float rho = 3.0f;
    float r   = 0.2f;
    CHECK(delta_safe(rho, r) == Approx(SAFETY_FACTOR * delta_nyquist(rho, r)).epsilon(1e-5f));
}

TEST_CASE("delta_safe: r=0, rho=1 → 1.6", "[delta]") {
    CHECK(delta_safe(1.0f, 0.0f) == Approx(1.6f).epsilon(1e-4f));
}

TEST_CASE("delta_safe: r=0, rho=2 → 0.8", "[delta]") {
    CHECK(delta_safe(2.0f, 0.0f) == Approx(0.8f).epsilon(1e-4f));
}

TEST_CASE("delta_safe: r=0, rho=4 → 0.4", "[delta]") {
    CHECK(delta_safe(4.0f, 0.0f) == Approx(0.4f).epsilon(1e-4f));
}

TEST_CASE("delta_safe: r=0.5, rho=1 → 3.2", "[delta]") {
    // 1.6 / 0.5 = 3.2
    CHECK(delta_safe(1.0f, 0.5f) == Approx(3.2f).epsilon(1e-3f));
}

TEST_CASE("delta_safe: high resonance allows larger Δ", "[delta]") {
    float rho = 5.0f;
    CHECK(delta_safe(rho, 0.9f) > delta_safe(rho, 0.5f));
    CHECK(delta_safe(rho, 0.5f) > delta_safe(rho, 0.0f));
}

TEST_CASE("delta_safe: resonance clamped at 0 and 1", "[delta]") {
    CHECK(delta_safe(1.0f, -1.0f) == Approx(delta_safe(1.0f, 0.0f)).epsilon(1e-5f));
    // r=1 → denom = 0*rho + epsilon → large safe delta
    CHECK(delta_safe(1.0f, 1.0f) > 1.0e5f);
}

// ---------------------------------------------------------------------------
// §5 — clamp_delta and timestep_within_bound
// ---------------------------------------------------------------------------

TEST_CASE("clamp_delta: smaller than safe → unchanged", "[clamp]") {
    CHECK(clamp_delta(0.5f, 1.0f, 0.0f) == Approx(0.5f));
}

TEST_CASE("clamp_delta: larger than safe → clamped", "[clamp]") {
    float safe = delta_safe(1.0f, 0.0f); // ~1.6
    CHECK(clamp_delta(5.0f, 1.0f, 0.0f) == Approx(safe).epsilon(1e-4f));
}

TEST_CASE("clamp_delta: exactly at safe → unchanged", "[clamp]") {
    float safe = delta_safe(2.0f, 0.0f); // ~0.8
    CHECK(clamp_delta(safe, 2.0f, 0.0f) == Approx(safe).epsilon(1e-5f));
}

TEST_CASE("timestep_within_bound: Δ = delta_safe → true", "[clamp]") {
    CHECK(timestep_within_bound(delta_safe(1.0f, 0.0f), 1.0f, 0.0f) == true);
}

TEST_CASE("timestep_within_bound: Δ < delta_safe → true", "[clamp]") {
    CHECK(timestep_within_bound(0.1f, 1.0f, 0.0f) == true);
}

TEST_CASE("timestep_within_bound: Δ > delta_safe → false", "[clamp]") {
    CHECK(timestep_within_bound(10.0f, 1.0f, 0.0f) == false);
}

// ---------------------------------------------------------------------------
// §6 — spectral_radius_bounded: Δ(1−r)ρ ≤ 2
// ---------------------------------------------------------------------------

TEST_CASE("spectral_radius_bounded: Δ*(1-r)*rho = 2 exactly → bounded (at limit)", "[spectral]") {
    // Δ=2, r=0, ρ=1 → 2*1*1=2 ≤ 2
    CHECK(spectral_radius_bounded(2.0f, 1.0f, 0.0f) == true);
}

TEST_CASE("spectral_radius_bounded: product = 1.6 (within safe zone) → bounded", "[spectral]") {
    // Δ=1.6, r=0, ρ=1
    CHECK(spectral_radius_bounded(1.6f, 1.0f, 0.0f) == true);
}

TEST_CASE("spectral_radius_bounded: product > 2 → NOT bounded", "[spectral]") {
    // Δ=2.1, r=0, ρ=1 → 2.1 > 2
    CHECK(spectral_radius_bounded(2.1f, 1.0f, 0.0f) == false);
}

TEST_CASE("spectral_radius_bounded: r=0.5 doubles the headroom", "[spectral]") {
    // Δ=3.0, r=0.5, ρ=1 → 3*0.5*1 = 1.5 ≤ 2 → bounded
    CHECK(spectral_radius_bounded(3.0f, 1.0f, 0.5f) == true);
    // Δ=4.1, r=0.5, ρ=1 → 4.1*0.5 > 2 → not bounded
    CHECK(spectral_radius_bounded(4.1f, 1.0f, 0.5f) == false);
}

TEST_CASE("spectral_radius_bounded: large r → high Δ still bounded", "[spectral]") {
    // r=0.9, rho=1, Δ=15: 15*(0.1)*1=1.5 ≤ 2
    CHECK(spectral_radius_bounded(15.0f, 1.0f, 0.9f) == true);
}

// ---------------------------------------------------------------------------
// §7 — Gain limiter: matrix_within_gain_limit, gain_limit_scale_factor
// ---------------------------------------------------------------------------

TEST_CASE("matrix_within_gain_limit: norm < ln(10) → true", "[gain]") {
    CHECK(matrix_within_gain_limit(2.0f)    == true);
    CHECK(matrix_within_gain_limit(0.0f)    == true);
    CHECK(matrix_within_gain_limit(2.3f)    == true);
}

TEST_CASE("matrix_within_gain_limit: norm = ln(10) exactly → true", "[gain]") {
    CHECK(matrix_within_gain_limit(MAX_GROWTH_RATE_LOG) == true);
}

TEST_CASE("matrix_within_gain_limit: norm > ln(10) → false", "[gain]") {
    CHECK(matrix_within_gain_limit(2.4f)  == false);
    CHECK(matrix_within_gain_limit(5.0f)  == false);
    CHECK(matrix_within_gain_limit(100.0f)== false);
}

TEST_CASE("gain_limit_scale_factor: within limit → 1.0", "[gain]") {
    CHECK(gain_limit_scale_factor(2.0f) == Approx(1.0f));
    CHECK(gain_limit_scale_factor(0.5f) == Approx(1.0f));
}

TEST_CASE("gain_limit_scale_factor: at limit → 1.0", "[gain]") {
    CHECK(gain_limit_scale_factor(MAX_GROWTH_RATE_LOG) == Approx(1.0f));
}

TEST_CASE("gain_limit_scale_factor: norm=3.0 → ln(10)/3.0", "[gain]") {
    float expected = MAX_GROWTH_RATE_LOG / 3.0f;
    CHECK(gain_limit_scale_factor(3.0f) == Approx(expected).epsilon(1e-5f));
}

TEST_CASE("gain_limit_scale_factor: scaled norm == ln(10)", "[gain]") {
    float norm   = 5.0f;
    float factor = gain_limit_scale_factor(norm);
    CHECK(norm * factor == Approx(MAX_GROWTH_RATE_LOG).epsilon(1e-4f));
}

TEST_CASE("clamped_matrix_norm: below limit → unchanged", "[gain]") {
    float n = 2.0f;
    CHECK(clamped_matrix_norm(n) == Approx(n));
}

TEST_CASE("clamped_matrix_norm: above limit → MAX_GROWTH_RATE_LOG", "[gain]") {
    CHECK(clamped_matrix_norm(5.0f)    == Approx(MAX_GROWTH_RATE_LOG).epsilon(1e-5f));
    CHECK(clamped_matrix_norm(100.0f)  == Approx(MAX_GROWTH_RATE_LOG).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §8 — classify_stability
// ---------------------------------------------------------------------------

TEST_CASE("classify_stability: good Δ, good norm → STABLE", "[classify]") {
    // Δ=0.5, rho=1, r=0, norm=1.0 — all within bounds
    CHECK(classify_stability(0.5f, 1.0f, 0.0f, 1.0f) == StabilityCondition::STABLE);
}

TEST_CASE("classify_stability: oversized Δ → TIMESTEP_VIOLATION", "[classify]") {
    // Δ=3.0, rho=1, r=0 → 3>2
    CHECK(classify_stability(3.0f, 1.0f, 0.0f, 1.0f)
        == StabilityCondition::TIMESTEP_VIOLATION);
}

TEST_CASE("classify_stability: good Δ but large matrix norm → RUNAWAY_ATTENTION", "[classify]") {
    CHECK(classify_stability(0.5f, 1.0f, 0.0f, 5.0f)
        == StabilityCondition::RUNAWAY_ATTENTION);
}

TEST_CASE("classify_stability: TIMESTEP_VIOLATION takes priority over runaway", "[classify]") {
    // Both failing: Δ > safe AND large norm
    CHECK(classify_stability(3.0f, 1.0f, 0.0f, 5.0f)
        == StabilityCondition::TIMESTEP_VIOLATION);
}

TEST_CASE("classify_stability: exactly at Nyquist limit → still STABLE", "[classify]") {
    // Δ=2, rho=1, r=0 → product exactly 2 (bounded), norm=1 (ok)
    CHECK(classify_stability(2.0f, 1.0f, 0.0f, 1.0f) == StabilityCondition::STABLE);
}

// ---------------------------------------------------------------------------
// §9 — Curvature shockwave
// ---------------------------------------------------------------------------

TEST_CASE("is_curvature_shockwave: new rho > 2x old → shockwave", "[shockwave]") {
    CHECK(is_curvature_shockwave(1.0f, 2.1f) == true);
}

TEST_CASE("is_curvature_shockwave: new rho = 2x old → no shockwave (boundary)", "[shockwave]") {
    CHECK(is_curvature_shockwave(1.0f, 2.0f) == false);
}

TEST_CASE("is_curvature_shockwave: slight increase → no shockwave", "[shockwave]") {
    CHECK(is_curvature_shockwave(5.0f, 6.0f) == false);
}

TEST_CASE("is_curvature_shockwave: 10x spike → shockwave", "[shockwave]") {
    CHECK(is_curvature_shockwave(1.0f, 10.0f) == true);
}

TEST_CASE("shockwave_violates_delta: previously safe Δ, spiked rho → violation", "[shockwave]") {
    // r=0, safe Δ for rho=1 is ~1.6; rho spikes to 5 → 1.6*5=8 > 2 → violation
    CHECK(shockwave_violates_delta(1.6f, 5.0f, 0.0f) == true);
}

TEST_CASE("shockwave_violates_delta: Δ very small → always safe even after spike", "[shockwave]") {
    CHECK(shockwave_violates_delta(0.01f, 100.0f, 0.0f) == false);
}

// ---------------------------------------------------------------------------
// §10 — intervention_for
// ---------------------------------------------------------------------------

TEST_CASE("intervention_for: STABLE → NONE", "[intervention]") {
    CHECK(intervention_for(StabilityCondition::STABLE) == InterventionType::NONE);
}

TEST_CASE("intervention_for: TIMESTEP_VIOLATION → CLAMP_DELTA", "[intervention]") {
    CHECK(intervention_for(StabilityCondition::TIMESTEP_VIOLATION)
        == InterventionType::CLAMP_DELTA);
}

TEST_CASE("intervention_for: CURVATURE_SHOCKWAVE → LOCAL_SCRAM", "[intervention]") {
    CHECK(intervention_for(StabilityCondition::CURVATURE_SHOCKWAVE)
        == InterventionType::LOCAL_SCRAM);
}

TEST_CASE("intervention_for: RUNAWAY_ATTENTION → SCALE_MATRIX", "[intervention]") {
    CHECK(intervention_for(StabilityCondition::RUNAWAY_ATTENTION)
        == InterventionType::SCALE_MATRIX);
}

// ---------------------------------------------------------------------------
// §11 — Attention dominance
// ---------------------------------------------------------------------------

TEST_CASE("is_runaway_attention: component > 90% → true", "[attention]") {
    CHECK(is_runaway_attention(9.1f, 10.0f) == true);
}

TEST_CASE("is_runaway_attention: component = 90% → false (strict >)", "[attention]") {
    CHECK(is_runaway_attention(9.0f, 10.0f) == false);
}

TEST_CASE("is_runaway_attention: component < 90% → false", "[attention]") {
    CHECK(is_runaway_attention(5.0f, 10.0f) == false);
}

TEST_CASE("is_runaway_attention: zero total energy → false (safe)", "[attention]") {
    CHECK(is_runaway_attention(0.0f, 0.0f) == false);
}

TEST_CASE("attention_fraction: 9/10 = 0.9", "[attention]") {
    CHECK(attention_fraction(9.0f, 10.0f) == Approx(0.9f));
}

TEST_CASE("attention_fraction: zero total → 0.0", "[attention]") {
    CHECK(attention_fraction(5.0f, 0.0f) == Approx(0.0f));
}

TEST_CASE("attention_fraction: full dominance = 1.0", "[attention]") {
    CHECK(attention_fraction(10.0f, 10.0f) == Approx(1.0f));
}

// ---------------------------------------------------------------------------
// §12 — effective_stiffness and resonance_delta_expansion
// ---------------------------------------------------------------------------

TEST_CASE("effective_stiffness: r=0 → rho_G", "[stiffness]") {
    CHECK(effective_stiffness(5.0f, 0.0f) == Approx(5.0f));
}

TEST_CASE("effective_stiffness: r=0.5 → rho/2", "[stiffness]") {
    CHECK(effective_stiffness(4.0f, 0.5f) == Approx(2.0f));
}

TEST_CASE("effective_stiffness: r=1 → 0", "[stiffness]") {
    CHECK(effective_stiffness(10.0f, 1.0f) == Approx(0.0f));
}

TEST_CASE("effective_stiffness: r>1 clamped to 1 → 0", "[stiffness]") {
    CHECK(effective_stiffness(5.0f, 2.0f) == Approx(0.0f));
}

TEST_CASE("resonance_delta_expansion: r=0 → ratio = 1.0", "[stiffness]") {
    CHECK(resonance_delta_expansion(3.0f, 0.0f) == Approx(1.0f).epsilon(1e-4f));
}

TEST_CASE("resonance_delta_expansion: r=0.5 → ratio ≈ 2.0", "[stiffness]") {
    // rho=10: denom_r0 = 10+ε, denom_r = 5+ε → ratio ≈ 2
    CHECK(resonance_delta_expansion(10.0f, 0.5f) == Approx(2.0f).epsilon(1e-3f));
}

TEST_CASE("resonance_delta_expansion: r=0.9 → ratio ≈ 10.0", "[stiffness]") {
    // denom_r0 = 10+ε, denom_r = 1+ε → ratio ≈ 10
    CHECK(resonance_delta_expansion(10.0f, 0.9f) == Approx(10.0f).epsilon(1e-2f));
}

// ---------------------------------------------------------------------------
// §13 — Power iteration policy
// ---------------------------------------------------------------------------

TEST_CASE("power_iter_step_count: returns 5", "[power_iter]") {
    CHECK(power_iter_step_count() == 5);
}

TEST_CASE("classify_power_iter: positive rho → CONVERGED", "[power_iter]") {
    CHECK(classify_power_iter(1.0f)    == PowerIterStatus::CONVERGED);
    CHECK(classify_power_iter(0.001f)  == PowerIterStatus::CONVERGED);
    CHECK(classify_power_iter(100.0f)  == PowerIterStatus::CONVERGED);
}

TEST_CASE("classify_power_iter: zero rho → DEGENERATE", "[power_iter]") {
    CHECK(classify_power_iter(0.0f)   == PowerIterStatus::DEGENERATE);
    CHECK(classify_power_iter(5e-7f)  == PowerIterStatus::DEGENERATE);
}

// ---------------------------------------------------------------------------
// §14 — Energy trend classification
// ---------------------------------------------------------------------------

TEST_CASE("classify_energy_trend: stable decay → BOUNDED", "[energy]") {
    CHECK(classify_energy_trend(10.0f, 9.0f, 0.1f) == EnergyTrend::BOUNDED);
}

TEST_CASE("classify_energy_trend: growing → DIVERGING", "[energy]") {
    CHECK(classify_energy_trend(10.0f, 11.0f, 0.1f) == EnergyTrend::DIVERGING);
}

TEST_CASE("classify_energy_trend: dominating component → LOCALISED (priority)", "[energy]") {
    // fraction=0.95 → LOCALISED even if overall energy grew
    CHECK(classify_energy_trend(10.0f, 11.0f, 0.95f) == EnergyTrend::LOCALISED);
}

TEST_CASE("classify_energy_trend: near-zero current with non-zero prev → OSCILLATING", "[energy]") {
    // curr drops to < 1% of prev → oscillation heuristic
    CHECK(classify_energy_trend(100.0f, 0.5f, 0.1f) == EnergyTrend::OSCILLATING);
}

TEST_CASE("classify_energy_trend: bounded energy, no dominant component → BOUNDED", "[energy]") {
    CHECK(classify_energy_trend(5.0f, 5.0f, 0.5f) == EnergyTrend::BOUNDED);
}

// ---------------------------------------------------------------------------
// §15 — Physical interpretation helpers
// ---------------------------------------------------------------------------

TEST_CASE("omega_max: equals effective_stiffness", "[physics]") {
    CHECK(omega_max(3.0f, 0.4f) == Approx(effective_stiffness(3.0f, 0.4f)));
}

TEST_CASE("nyquist_satisfied: Δ=1.0, rho=1, r=0 → 1*1*1=1 ≤ 2 → true", "[physics]") {
    CHECK(nyquist_satisfied(1.0f, 1.0f, 0.0f) == true);
}

TEST_CASE("nyquist_satisfied: Δ=2.0, rho=1, r=0 → 2 ≤ 2 → true", "[physics]") {
    CHECK(nyquist_satisfied(2.0f, 1.0f, 0.0f) == true);
}

TEST_CASE("nyquist_satisfied: Δ=2.1, rho=1, r=0 → 2.1 > 2 → false", "[physics]") {
    CHECK(nyquist_satisfied(2.1f, 1.0f, 0.0f) == false);
}

TEST_CASE("delta_headroom_pct: Δ = delta_safe → 0%", "[physics]") {
    float safe = delta_safe(2.0f, 0.0f);
    CHECK(delta_headroom_pct(safe, 2.0f, 0.0f) == Approx(0.0f).epsilon(1e-3f));
}

TEST_CASE("delta_headroom_pct: Δ = 0 → 100%", "[physics]") {
    CHECK(delta_headroom_pct(0.0f, 2.0f, 0.0f) == Approx(100.0f).epsilon(1e-3f));
}

TEST_CASE("delta_headroom_pct: Δ = half of safe → 50%", "[physics]") {
    float safe = delta_safe(2.0f, 0.0f);
    CHECK(delta_headroom_pct(safe * 0.5f, 2.0f, 0.0f) == Approx(50.0f).epsilon(0.1f));
}

TEST_CASE("delta_headroom_pct: Δ > safe → negative (violation)", "[physics]") {
    float safe = delta_safe(1.0f, 0.0f);  // ~1.6
    CHECK(delta_headroom_pct(safe + 0.5f, 1.0f, 0.0f) < 0.0f);
}

// ---------------------------------------------------------------------------
// §16 — Label helpers
// ---------------------------------------------------------------------------

TEST_CASE("stability_label: correct strings", "[labels]") {
    CHECK(stability_label(StabilityCondition::STABLE)              == "STABLE");
    CHECK(stability_label(StabilityCondition::TIMESTEP_VIOLATION)  == "TIMESTEP_VIOLATION");
    CHECK(stability_label(StabilityCondition::CURVATURE_SHOCKWAVE) == "CURVATURE_SHOCKWAVE");
    CHECK(stability_label(StabilityCondition::RUNAWAY_ATTENTION)   == "RUNAWAY_ATTENTION");
}

TEST_CASE("intervention_label: correct strings", "[labels]") {
    CHECK(intervention_label(InterventionType::NONE)         == "NONE");
    CHECK(intervention_label(InterventionType::CLAMP_DELTA)  == "CLAMP_DELTA");
    CHECK(intervention_label(InterventionType::SCALE_MATRIX) == "SCALE_MATRIX");
    CHECK(intervention_label(InterventionType::LOCAL_SCRAM)  == "LOCAL_SCRAM");
}

TEST_CASE("energy_trend_label: correct strings", "[labels]") {
    CHECK(energy_trend_label(EnergyTrend::BOUNDED)    == "BOUNDED");
    CHECK(energy_trend_label(EnergyTrend::OSCILLATING)== "OSCILLATING");
    CHECK(energy_trend_label(EnergyTrend::DIVERGING)  == "DIVERGING");
    CHECK(energy_trend_label(EnergyTrend::LOCALISED)  == "LOCALISED");
}

TEST_CASE("power_iter_label: correct strings", "[labels]") {
    CHECK(power_iter_label(PowerIterStatus::CONVERGED)  == "CONVERGED");
    CHECK(power_iter_label(PowerIterStatus::DEGENERATE) == "DEGENERATE");
}

TEST_CASE("all labels non-empty", "[labels]") {
    for (auto s : {StabilityCondition::STABLE, StabilityCondition::TIMESTEP_VIOLATION,
                   StabilityCondition::CURVATURE_SHOCKWAVE, StabilityCondition::RUNAWAY_ATTENTION})
        CHECK(!stability_label(s).empty());
    for (auto t : {InterventionType::NONE, InterventionType::CLAMP_DELTA,
                   InterventionType::SCALE_MATRIX, InterventionType::LOCAL_SCRAM})
        CHECK(!intervention_label(t).empty());
}

// ---------------------------------------------------------------------------
// §17 — Integration / scenario tests
// ---------------------------------------------------------------------------

TEST_CASE("Scenario: nominal Mamba-9D step at spec values", "[scenario]") {
    // rho_G=1.0, r=0.0 → Δ_safe = 1.6
    float rho    = 1.0f;
    float r      = 0.0f;
    float d_safe = delta_safe(rho, r);
    CHECK(d_safe == Approx(1.6f).epsilon(1e-4f));

    // request 0.8 (half of safe) → granted unchanged
    CHECK(clamp_delta(0.8f, rho, r) == Approx(0.8f));
    CHECK(spectral_radius_bounded(0.8f, rho, r) == true);

    // eigenvalue at Δ=0.8, μ=1: λ = 1 - 0.8 = 0.2 (well within |λ|<1)
    CHECK(eigenvalue_of_Abar(0.8f, r, 1.0f) == Approx(0.2f));
    CHECK(classify_stability(0.8f, rho, r, 1.5f) == StabilityCondition::STABLE);
    CHECK(intervention_for(StabilityCondition::STABLE) == InterventionType::NONE);
}

TEST_CASE("Scenario: timestep violation and auto-clamping", "[scenario]") {
    float rho = 1.0f, r = 0.0f;
    float d_requested = 3.0f;  // > Nyquist limit of 2.0

    // detect
    CHECK(classify_stability(d_requested, rho, r, 1.0f)
        == StabilityCondition::TIMESTEP_VIOLATION);
    CHECK(intervention_for(StabilityCondition::TIMESTEP_VIOLATION)
        == InterventionType::CLAMP_DELTA);

    // apply clamp
    float d_clamped = clamp_delta(d_requested, rho, r);
    CHECK(d_clamped == Approx(delta_safe(rho, r)).epsilon(1e-4f));
    CHECK(classify_stability(d_clamped, rho, r, 1.0f) == StabilityCondition::STABLE);
}

TEST_CASE("Scenario: runaway attention → matrix rescale", "[scenario]") {
    float norm = 4.0f;  // > ln(10)
    CHECK(matrix_within_gain_limit(norm) == false);
    CHECK(classify_stability(0.5f, 1.0f, 0.0f, norm)
        == StabilityCondition::RUNAWAY_ATTENTION);
    CHECK(intervention_for(StabilityCondition::RUNAWAY_ATTENTION)
        == InterventionType::SCALE_MATRIX);

    float factor     = gain_limit_scale_factor(norm);
    float new_norm   = norm * factor;
    CHECK(new_norm == Approx(MAX_GROWTH_RATE_LOG).epsilon(1e-4f));
    CHECK(matrix_within_gain_limit(new_norm) == true);
}

TEST_CASE("Scenario: neurogenesis curvature shockwave", "[scenario]") {
    float rho_prev = 1.0f;
    float rho_new  = 3.0f;   // 3× spike → shockwave
    float r        = 0.0f;

    CHECK(is_curvature_shockwave(rho_prev, rho_new) == true);

    // Δ was safe for rho=1 (~1.6); check if still safe for rho=3
    float d = delta_safe(rho_prev, r);  // ~1.6
    CHECK(shockwave_violates_delta(d, rho_new, r) == true);

    // intervention
    CHECK(intervention_for(StabilityCondition::CURVATURE_SHOCKWAVE)
        == InterventionType::LOCAL_SCRAM);

    // recalculate safe Δ for new rho
    float d_new = delta_safe(rho_new, r);
    // delta_safe(3, 0) = 1.6/3 ≈ 0.533
    CHECK(d_new == Approx(1.6f / 3.0f).epsilon(1e-3f));
    CHECK(!shockwave_violates_delta(d_new, rho_new, r));
}

TEST_CASE("Scenario: high resonance memory — frozen state", "[scenario]") {
    // r → 1 means memory is strongly persistent (high resonance)
    // The safe Δ becomes very large — system can use large steps
    float rho = 2.0f, r = 0.95f;
    float d = delta_safe(rho, r);
    // 1.6 / (0.05 * 2) = 1.6 / 0.1 = 16
    CHECK(d == Approx(16.0f).epsilon(0.1f));
    CHECK(spectral_radius_bounded(10.0f, rho, r) == true);  // 10*0.05*2 = 1 ≤ 2
}

TEST_CASE("Scenario: safety factor derivation — 20% margin below Nyquist", "[scenario]") {
    float rho = 4.0f, r = 0.0f;
    float d_nyq  = delta_nyquist(rho, r);  // 0.5
    float d_safe = delta_safe(rho, r);      // 0.4 = 80% of 0.5
    CHECK(d_safe == Approx(d_nyq * SAFETY_FACTOR).epsilon(1e-4f));
    CHECK((d_nyq - d_safe) / d_nyq == Approx(SAFETY_MARGIN_FRACTION).epsilon(1e-4f));
}

TEST_CASE("Scenario: gain limiter — max 10x amplification", "[scenario]") {
    // The spec says e^λ_max ≤ 10 → λ_max ≤ ln(10)
    // If ||A|| = ln(10), max amplification factor is exactly 10
    CHECK(std::exp(static_cast<double>(MAX_GROWTH_RATE_LOG))
        == Approx(MAX_GROWTH_RATE).epsilon(1e-4));
}

TEST_CASE("Scenario: power iteration convergence in 5 steps", "[scenario]") {
    // Policy: always use exactly 5 iterations
    CHECK(power_iter_step_count() == POWER_ITER_STEPS);

    // A nonzero rho after 5 iterations is CONVERGED
    float rho_est = 2.5f;
    CHECK(classify_power_iter(rho_est) == PowerIterStatus::CONVERGED);
    // A zero rho is DEGENERATE
    CHECK(classify_power_iter(0.0f) == PowerIterStatus::DEGENERATE);
}
