/**
 * @file   inner_monologue.hpp
 * @brief  GAP-016: Inner Monologue Recursive Reasoning Control
 *
 * Governs the Chain-of-Thought re-injection loop, preventing three
 * catastrophic failure modes:
 *   1. Epileptic Resonance   — positive feedback / energy divergence
 *   2. Teleological Deadlock — closed geodesic (circular reasoning) loops
 *   3. Coherence Degradation — signal below thermal noise floor
 *
 * Core design: a "thought" is a wavefunction Ψ(x,t) propagating through
 * Riemannian 9D manifold.  Recursion consumes metabolic ATP and is bounded
 * by both a thermodynamic energy budget and Mamba-9D context horizon.
 *
 * Namespace: nikola::cognitive
 * Spec:      docs/info/integration/sections/03_cognitive_systems/
 *                03_neuroplastic_transformer.md §GAP-016
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace nikola::cognitive {

// ═══════════════════════════════════════════════════════════════════════════
// Recursion depth limits
// ═══════════════════════════════════════════════════════════════════════════

/// Hard cap: Mamba-9D effective context horizon.
/// Beyond 12 re-injections phase coherence of Ψ₀ degrades below thermal
/// noise floor due to numerical diffusion.
constexpr int RECURSION_HARD_DEPTH_LIMIT = 12;

/// Soft cap aligning with Miller's Law (7 ± 2):
/// beyond d=7 the cost of maintaining Refractive Traps exceeds recharge rate.
constexpr int RECURSION_SOFT_LIMIT = 7;

// ═══════════════════════════════════════════════════════════════════════════
// Metabolic cost constants
// ═══════════════════════════════════════════════════════════════════════════

/// Minimum ATP reserve that must never be consumed (forces Nap state).
constexpr float RECURSION_E_RESERVE = 0.15f;

/// Base metabolic cost (ATP) of one active-reasoning step at depth 1.
constexpr float RECURSION_C_BASE = 0.05f;

/// Compound recursion tax per level: cost multiplier = (1 + λ)^d.
constexpr float RECURSION_LAMBDA_PENALTY = 0.15f;

// ═══════════════════════════════════════════════════════════════════════════
// Coherence degradation thresholds
// ═══════════════════════════════════════════════════════════════════════════

/// Absolute spectral entropy threshold above which the re-injected wavefunction
/// is indistinguishable from thermal noise → Confusion Interrupt.
constexpr float COHERENCE_ENTROPY_THRESHOLD = 0.85f;

/// Per-step entropy gradient limit: ΔH > 0.05 signals rapid phase decoherence
/// ("scrambling") → terminate branch.
constexpr float COHERENCE_ENTROPY_GRADIENT_LIMIT = 0.05f;

// ═══════════════════════════════════════════════════════════════════════════
// Teleological deadlock (loop) constants
// ═══════════════════════════════════════════════════════════════════════════

/// Boredom neurochemical spike injected when a closed geodesic is confirmed.
constexpr float LOOP_BOREDOM_SPIKE = 0.2f;

// ═══════════════════════════════════════════════════════════════════════════
// Memory / Refractive Trap constants
// ═══════════════════════════════════════════════════════════════════════════

/// Stencil size per trap cluster: central node + 18-point 9D neighbour stencil.
constexpr int TRAP_NODES_PER_CLUSTER = 19;

/// Approximate memory footprint per node (Ψ complex-double + metric + overhead).
/// Ψ: 16 B, metric tensor (45 floats): 180 B, rest ~3.2 KB overhead.
constexpr int TRAP_MEMORY_BYTES_PER_NODE = 3400;       // ~3.4 KB

/// Total memory per trap cluster: 19 nodes × ~3.4 KB ≈ 65 KB.
constexpr int TRAP_MEMORY_BYTES_PER_CLUSTER = 65'000;  // ~65 KB

/// Maximum simultaneous active traps before 1ms physics frame-time is breached.
constexpr int MAX_ACTIVE_TRAPS = 9;

// ═══════════════════════════════════════════════════════════════════════════
// Memory pruning threshold
// ═══════════════════════════════════════════════════════════════════════════

/// Neuro-Necrosis pruning: resonance patterns below this are reclaimed
/// immediately by the SoACompactor.
constexpr float PRUNING_RESONANCE_THRESHOLD = 0.3f;

// ═══════════════════════════════════════════════════════════════════════════
// Termination reason enum
// ═══════════════════════════════════════════════════════════════════════════

/// Reason a recursion branch was (or must be) terminated.
enum class RecursionTermination {
    NONE,               ///< No termination — recursion may proceed
    HARD_DEPTH_CAP,     ///< Reached RECURSION_HARD_DEPTH_LIMIT (12)
    SOFT_DEPTH_CAP,     ///< Reached RECURSION_SOFT_LIMIT (7)
    METABOLIC_FATIGUE,  ///< Insufficient ATP (below reserve + next step cost)
    COHERENCE_ALARM,    ///< Spectral entropy absolute threshold breached
    ENTROPY_GRADIENT,   ///< Per-step entropy gradient exceeded
    LOOP_DETECTED,      ///< Closed geodesic (teleological deadlock) confirmed
};

// ═══════════════════════════════════════════════════════════════════════════
// §1  Metabolic cost formulas
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief  Metabolic cost of the d-th recursion step.
 *
 * Cost(d) = c_base × (1 + λ)^d
 *
 * Depth d=1 is the first recursive re-injection; d=0 is treated as the
 * "initial" invocation (flat base cost).
 *
 * @param depth     Recursion depth (0 = initial call, 1..12 = re-injections)
 * @param c_base    Base cost per step (default RECURSION_C_BASE)
 * @param lambda    Compound tax (default RECURSION_LAMBDA_PENALTY)
 * @throws std::invalid_argument if depth < 0, c_base <= 0, or lambda < 0
 */
[[nodiscard]] inline float recursion_step_cost(
    int depth,
    float c_base  = RECURSION_C_BASE,
    float lambda  = RECURSION_LAMBDA_PENALTY)
{
    if (depth < 0)
        throw std::invalid_argument("recursion_step_cost: depth must be >= 0");
    if (c_base <= 0.0f)
        throw std::invalid_argument("recursion_step_cost: c_base must be > 0");
    if (lambda < 0.0f)
        throw std::invalid_argument("recursion_step_cost: lambda must be >= 0");
    return c_base * std::pow(1.0f + lambda, static_cast<float>(depth));
}

/**
 * @brief  Total metabolic cost to complete max_depth levels of recursion.
 *
 * Cumulative(max_depth) = Σ_{d=1}^{max_depth} Cost(d)
 *
 * Note: depth-0 (initial invocation) is excluded from recursion cost.
 * max_depth = 0 → 0 (no recursion yet).
 *
 * @throws std::invalid_argument if max_depth < 0
 */
[[nodiscard]] inline float recursion_cumulative_cost(
    int max_depth,
    float c_base = RECURSION_C_BASE,
    float lambda = RECURSION_LAMBDA_PENALTY)
{
    if (max_depth < 0)
        throw std::invalid_argument("recursion_cumulative_cost: max_depth must be >= 0");
    if (c_base <= 0.0f)
        throw std::invalid_argument("recursion_cumulative_cost: c_base must be > 0");
    if (lambda < 0.0f)
        throw std::invalid_argument("recursion_cumulative_cost: lambda must be >= 0");

    float total = 0.0f;
    for (int d = 1; d <= max_depth; ++d)
        total += c_base * std::pow(1.0f + lambda, static_cast<float>(d));
    return total;
}

/**
 * @brief  Maximum thermodynamically-affordable recursion depth.
 *
 * D_max = min(HARD_LIMIT, max d such that Σ_{i=1}^{d} Cost(i) ≤ atp − e_reserve)
 *
 * Iteratively accumulates costs until ATP budget is exhausted or the hard
 * depth limit is reached.
 *
 * @param current_atp  Available ATP at time of invocation [0.0, 1.0]
 * @param e_reserve    Minimum reserve to keep (default RECURSION_E_RESERVE)
 * @param c_base       Base step cost (default RECURSION_C_BASE)
 * @param lambda       Compound penalty (default RECURSION_LAMBDA_PENALTY)
 * @return Maximum affordable depth in [0, RECURSION_HARD_DEPTH_LIMIT]
 * @throws std::invalid_argument if current_atp < 0 or e_reserve < 0
 */
[[nodiscard]] inline int max_thermodynamic_depth(
    float current_atp,
    float e_reserve = RECURSION_E_RESERVE,
    float c_base    = RECURSION_C_BASE,
    float lambda    = RECURSION_LAMBDA_PENALTY)
{
    if (current_atp < 0.0f)
        throw std::invalid_argument("max_thermodynamic_depth: current_atp must be >= 0");
    if (e_reserve < 0.0f)
        throw std::invalid_argument("max_thermodynamic_depth: e_reserve must be >= 0");

    float budget = current_atp - e_reserve;
    if (budget <= 0.0f) return 0;

    float accumulated = 0.0f;
    for (int d = 1; d <= RECURSION_HARD_DEPTH_LIMIT; ++d) {
        accumulated += c_base * std::pow(1.0f + lambda, static_cast<float>(d));
        if (accumulated > budget) return d - 1;
    }
    return RECURSION_HARD_DEPTH_LIMIT;
}

// ═══════════════════════════════════════════════════════════════════════════
// §2  Coherence degradation detection
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief  Spectral entropy of re-injected wavefunction power spectrum.
 *
 * H_spec = −Σ_k p_k log₂(p_k)
 *
 * Input `pk` must be a probability distribution (non-negative values that
 * sum to 1.0 within floating-point tolerance).  Zero-valued elements are
 * skipped to avoid log(0).
 *
 * @param pk  Normalised power spectral density (elements ≥ 0, Σ ≈ 1)
 * @return    Spectral entropy in nats-equivalent bits [0, log₂(N)]
 * @throws std::invalid_argument if pk is empty
 */
[[nodiscard]] inline float spectral_entropy(std::span<const float> pk)
{
    if (pk.empty())
        throw std::invalid_argument("spectral_entropy: distribution must be non-empty");

    float h = 0.0f;
    for (float p : pk) {
        if (p > 0.0f)
            h -= p * std::log2(p);
    }
    return h;
}

/**
 * @brief  Per-step entropy gradient: ΔH = H_current − H_previous.
 *
 * Positive values indicate increasing disorder (decoherence).
 */
[[nodiscard]] constexpr float entropy_gradient(float h_current, float h_prev) noexcept
{
    return h_current - h_prev;
}

/**
 * @brief  True when coherence alarm conditions are met.
 *
 * Triggers when EITHER:
 *   • H_spec > COHERENCE_ENTROPY_THRESHOLD (absolute thermal-noise threshold), OR
 *   • ΔH > COHERENCE_ENTROPY_GRADIENT_LIMIT (rapid scrambling)
 *
 * On alarm: collapse stack, return last coherent state, penalise confidence.
 */
[[nodiscard]] constexpr bool is_coherence_alarm(
    float entropy,
    float gradient,
    float abs_threshold  = COHERENCE_ENTROPY_THRESHOLD,
    float grad_threshold = COHERENCE_ENTROPY_GRADIENT_LIMIT) noexcept
{
    return (entropy > abs_threshold) || (gradient > grad_threshold);
}

/**
 * @brief  Confidence score penalty on Confusion Interrupt.
 *
 * penalty = 1 − H_spec    (higher entropy → lower confidence)
 *
 * Result is clamped to [0, 1].
 */
[[nodiscard]] inline float confidence_penalty(float entropy)
{
    if (entropy < 0.0f)
        throw std::invalid_argument("confidence_penalty: entropy must be >= 0");
    return std::clamp(1.0f - entropy, 0.0f, 1.0f);
}

/**
 * @brief  Normalise a raw power spectrum to a probability distribution.
 *
 * Each element is divided by the total sum so the output sums to 1.
 * Required pre-processing before passing to spectral_entropy().
 *
 * @param power  Raw (non-negative) power values — modified in-place
 * @throws std::invalid_argument if all values are zero, or any value is negative
 */
inline void normalise_spectrum(std::span<float> power)
{
    float total = 0.0f;
    for (float v : power) {
        if (v < 0.0f)
            throw std::invalid_argument("normalise_spectrum: negative power value");
        total += v;
    }
    if (total == 0.0f)
        throw std::invalid_argument("normalise_spectrum: all-zero spectrum");
    for (float& v : power)
        v /= total;
}

// ═══════════════════════════════════════════════════════════════════════════
// §3  Circular reasoning / closed geodesic detection
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief  True if `hash` appears in the trajectory path (closed geodesic check).
 *
 * Implements O(d) Morton hash collision detection over the recorded path.
 * A collision is necessary but not sufficient for a deadlock — caller should
 * additionally verify metric tensor contraction (Tr(g) decrease).
 */
[[nodiscard]] inline bool is_loop_detected(
    std::size_t hash,
    std::span<const std::size_t> trajectory) noexcept
{
    for (std::size_t h : trajectory) {
        if (h == hash) return true;
    }
    return false;
}

/**
 * @brief  Amount of boredom spike to inject when a deadlock is confirmed.
 *
 * Returns LOOP_BOREDOM_SPIKE (0.2).  Extracted as a function so callers
 * have a single named call site matching the spec pseudocode.
 */
[[nodiscard]] constexpr float boredom_spike_amount() noexcept
{
    return LOOP_BOREDOM_SPIKE;
}

// ═══════════════════════════════════════════════════════════════════════════
// §4  Neuro-Necrosis: Refractive Trap garbage collection
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief  Refractive index decay post-return: s(t) = s₀ · e^(−λ_n · t)
 *
 * As s → 0 the trapped wavefunction releases its energy:
 *   • Constructive interference with parent thought = successful return
 *   • Dissipation as heat                           = forgetting
 *
 * @param s0       Initial refractive index of the trap
 * @param lambda_n Necrosis decay rate (must be > 0)
 * @param t        Time elapsed since trap release (must be ≥ 0)
 * @throws std::invalid_argument if lambda_n <= 0 or t < 0
 */
[[nodiscard]] inline float necrosis_decay(float s0, float lambda_n, float t)
{
    if (lambda_n <= 0.0f)
        throw std::invalid_argument("necrosis_decay: lambda_n must be > 0");
    if (t < 0.0f)
        throw std::invalid_argument("necrosis_decay: t must be >= 0");
    return s0 * std::exp(-lambda_n * t);
}

/**
 * @brief  True if a trap's resonance is below the pruning threshold.
 *
 * Low-resonance patterns (r < PRUNING_RESONANCE_THRESHOLD = 0.3) are marked
 * for immediate reclamation by the SoACompactor.
 */
[[nodiscard]] constexpr bool is_prunable(float resonance) noexcept
{
    return resonance < PRUNING_RESONANCE_THRESHOLD;
}

// ═══════════════════════════════════════════════════════════════════════════
// §5  Memory overhead helpers
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief  Total memory (bytes) for a given number of active refractive traps.
 *
 * Each trap cluster = TRAP_NODES_PER_CLUSTER (19) × TRAP_MEMORY_BYTES_PER_NODE
 * ≈ TRAP_MEMORY_BYTES_PER_CLUSTER (65 KB).
 *
 * @throws std::invalid_argument if n_traps < 0
 */
[[nodiscard]] inline std::size_t trap_memory_bytes(int n_traps)
{
    if (n_traps < 0)
        throw std::invalid_argument("trap_memory_bytes: n_traps must be >= 0");
    return static_cast<std::size_t>(n_traps)
         * static_cast<std::size_t>(TRAP_MEMORY_BYTES_PER_CLUSTER);
}

/**
 * @brief  True if adding one more trap would stay within the hardware budget.
 *
 * Returns false when n_active >= MAX_ACTIVE_TRAPS (9).
 */
[[nodiscard]] constexpr bool can_allocate_trap(int n_active) noexcept
{
    return n_active < MAX_ACTIVE_TRAPS;
}

// ═══════════════════════════════════════════════════════════════════════════
// §6  Composite recursion gate
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief  Evaluate all termination conditions and return the first violation.
 *
 * Checks in priority order:
 *   1. Hard depth cap
 *   2. Metabolic fatigue (ATP)
 *   3. Coherence alarm (absolute entropy)
 *   4. Entropy gradient alarm
 *   5. Loop collision (hash in trajectory)
 *   6. Soft depth cap (informational — caller may choose to respect)
 *
 * Returns RecursionTermination::NONE if recursion is safe to proceed.
 *
 * @param depth        Current recursion depth (0 = initial)
 * @param current_atp  Available ATP
 * @param entropy      Spectral entropy of current re-injected wavefunction
 * @param delta_h      Entropy gradient from previous step (pass 0.0 at d=0)
 * @param hash         Morton hash of current wave-packet centroid
 * @param trajectory   Previously visited hashes (empty at d=0)
 */
[[nodiscard]] inline RecursionTermination evaluate_recursion_gate(
    int depth,
    float current_atp,
    float entropy,
    float delta_h,
    std::size_t hash,
    std::span<const std::size_t> trajectory)
{
    if (depth >= RECURSION_HARD_DEPTH_LIMIT)
        return RecursionTermination::HARD_DEPTH_CAP;

    float next_cost = recursion_step_cost(depth + 1);
    if (current_atp < (RECURSION_E_RESERVE + next_cost))
        return RecursionTermination::METABOLIC_FATIGUE;

    if (entropy > COHERENCE_ENTROPY_THRESHOLD)
        return RecursionTermination::COHERENCE_ALARM;

    if (delta_h > COHERENCE_ENTROPY_GRADIENT_LIMIT)
        return RecursionTermination::ENTROPY_GRADIENT;

    if (is_loop_detected(hash, trajectory))
        return RecursionTermination::LOOP_DETECTED;

    if (depth >= RECURSION_SOFT_LIMIT)
        return RecursionTermination::SOFT_DEPTH_CAP;

    return RecursionTermination::NONE;
}

} // namespace nikola::cognitive
