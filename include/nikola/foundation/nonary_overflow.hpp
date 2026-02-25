#pragma once
/**
 * @file   nonary_overflow.hpp
 * @brief  GAP-044: Nonary Overflow Probability Distribution
 *
 * Statistical characterisation of overflow frequency, information loss, and
 * carry (Spectral Cascading) for Nikola's Balanced Nonary arithmetic
 * (digits {-4 … +4}, §02_foundations/03_balanced_nonary_logic.md §GAP-044).
 *
 * Key results
 * ──────────────────────────────────────────────────────────────────────────
 * • Information density  log₂(9) ≈ 3.170 bits per trit
 * • Addition overflow    uniform worst-case  20/81 ≈ 24.7 %
 *                        Gaussian operational estimate  < 5 %
 * • Saturation alert     > 1 % of operations triggers gain throttle
 * • Carry mechanism      A = C·9 + R  where R ∈ [−4, +4];
 *                        excess energy propagates to next higher dimension
 *                        (Spectral Cascading — §GAP-044 "Carry Mechanism")
 * • Dither injection     ε ~ Uniform(−δ, δ), δ = 0.5 (Voronoi cell radius)
 *                        prevents limit cycles and ensures ergodic exploration
 *
 * Namespace : nikola::foundation  (companion to nit.hpp)
 * C++ std   : C++23 (constexpr lambdas, [[nodiscard]])
 */

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "nikola/foundation/nit.hpp"   // Nit, NIT_MIN, NIT_MAX

namespace nikola::foundation {

// ═══════════════════════════════════════════════════════════════════════════
// §1  Base system constants
// ═══════════════════════════════════════════════════════════════════════════

/// Number of distinct digits in one Nit position {−4 … +4}.
inline constexpr int NONARY_BASE         = 9;

/// Number of distinct digit values (same as NONARY_BASE; alias for clarity).
inline constexpr int NONARY_DIGIT_COUNT  = 9;

/// Information density: log₂(9) bits per trit.
/// Exact value to 16 significant figures.
inline constexpr double NONARY_INFO_BITS_PER_TRIT = 3.169925001442312;

// ═══════════════════════════════════════════════════════════════════════════
// §2  Arithmetic range limits
// ═══════════════════════════════════════════════════════════════════════════

/// Maximum integer result of adding two Nits: NIT_MAX + NIT_MAX = +8.
inline constexpr int NONARY_ADD_RESULT_MAX = +8;

/// Minimum integer result of adding two Nits: NIT_MIN + NIT_MIN = −8.
inline constexpr int NONARY_ADD_RESULT_MIN = -8;

/// Maximum integer result of multiplying two Nits: (−4)×(−4) = +16.
inline constexpr int NONARY_MUL_RESULT_MAX = +16;

/// Minimum integer result of multiplying two Nits: (+4)×(−4) = −16.
inline constexpr int NONARY_MUL_RESULT_MIN = -16;

// ═══════════════════════════════════════════════════════════════════════════
// §3  Overflow / saturation boundary (continuous PDF integration)
// ═══════════════════════════════════════════════════════════════════════════

/// Continuous saturation clip boundary.  Voronoi cell boundary around ±4 sits
/// at ±4.5.  PDF integral beyond this boundary = information loss L.
inline constexpr double NONARY_CLIP_BOUNDARY = 4.5;

// ═══════════════════════════════════════════════════════════════════════════
// §4  Probability constants
// ═══════════════════════════════════════════════════════════════════════════

/// Uniform worst-case addition overflow: exact count of (a,b) pairs in
/// {−4…+4}² for which |a + b| > 4.
///
/// Enumeration:
///   Positive overflow (a+b > 4): (1,4),(2,3),(2,4),(3,2),(3,3),(3,4),
///                                 (4,1),(4,2),(4,3),(4,4)  = 10 pairs
///   Negative overflow (a+b < −4): 10 pairs by symmetry
///   Total = 20 out of 81
inline constexpr int NONARY_OVERFLOW_PAIR_COUNT     = 20;

/// Total addition pair count: 9 × 9 = 81.
inline constexpr int NONARY_TOTAL_PAIR_COUNT        = 81;

/// Analytical overflow probability under perfectly uniform input distribution
/// (worst case).  20/81 ≈ 0.2469 (~24.7 %).
/// Spec §GAP-044 quotes ≈ 22 % (approximate; exact analytical value is 24.7 %).
inline constexpr double NONARY_OVERFLOW_PROB_ADD_UNIFORM = 20.0 / 81.0;

/// Operational overflow probability under Gaussian thermal initialisation.
/// Spec-quoted upper bound: < 5 % per operation.
inline constexpr double NONARY_OVERFLOW_PROB_ADD_GAUSSIAN = 0.05;

// ═══════════════════════════════════════════════════════════════════════════
// §5  Saturation monitoring threshold
// ═══════════════════════════════════════════════════════════════════════════

/// Per §GAP-044 Validation: if saturation events > 1 % of total operations,
/// Input Gain is too high and must be throttled.
inline constexpr double SATURATION_RATE_ALERT_THRESHOLD = 0.01;

// ═══════════════════════════════════════════════════════════════════════════
// §6  Dither injection (Voronoi quantisation)
// ═══════════════════════════════════════════════════════════════════════════

/// Maximum dither amplitude δ for Voronoi quantisation.
/// Each integer Nit occupies a Voronoi cell of radius 0.5 in continuous space,
/// so dither ε ~ Uniform(−δ, +δ) with δ ≤ 0.5 avoids crossing cell boundaries.
/// Entropy source: Xoshiro256++ (§04_infrastructure/05_security_subsystem.md).
inline constexpr double NONARY_DITHER_AMPLITUDE_MAX = 0.5;

// ═══════════════════════════════════════════════════════════════════════════
// §7  Carry / Spectral Cascading constants
// ═══════════════════════════════════════════════════════════════════════════

/// Divisor for carry decomposition — equals NONARY_BASE.
/// A = carry × 9 + remainder,  remainder ∈ [−4, +4].
inline constexpr int NONARY_CARRY_DIVISOR = 9;

// ═══════════════════════════════════════════════════════════════════════════
// §8  Constexpr arithmetic functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Saturating addition of two Nits — clamps to [NIT_MIN, NIT_MAX].
 *
 * This is the "hard clipping" path.  Use add_with_carry() when Spectral
 * Cascading (energy conservation) is required.
 */
[[nodiscard]] inline constexpr Nit add_saturated(Nit a, Nit b) noexcept
{
    int s = static_cast<int>(a) + static_cast<int>(b);
    if (s > NIT_MAX) return NIT_MAX;
    if (s < NIT_MIN) return NIT_MIN;
    return static_cast<Nit>(s);
}

/**
 * @brief Add two Nits with Spectral Cascading carry.
 *
 * Implements §GAP-044 "Carry Mechanism":
 *   raw       = a + b  (integer, range [−8, +8])
 *   carry     = raw / 9  (truncation toward zero)
 *   remainder = raw − carry × 9
 *
 * Post-adjust ensures remainder ∈ [−4, +4] for values where C++ truncation
 * leaves |remainder| == 5 … 8.
 *
 * Energy conservation: a + b  =  carry_out × 9 + return_value
 *
 * @param[in]  a          First Nit operand
 * @param[in]  b          Second Nit operand
 * @param[out] carry_out  Integer carry to emit to next higher torus dimension
 *                        (−1, 0, or +1 for two-Nit addition)
 * @return                Remainder in [−4, +4]
 */
[[nodiscard]] inline constexpr Nit add_with_carry(Nit a, Nit b, int& carry_out) noexcept
{
    int raw      = static_cast<int>(a) + static_cast<int>(b);
    carry_out    = raw / NONARY_CARRY_DIVISOR;
    int rem      = raw - carry_out * NONARY_CARRY_DIVISOR;
    // Remainder from C++ truncation may still lie outside [−4, +4] for
    // inputs with |raw| in {5,6,7,8}: adjust one step.
    if (rem >  4) { rem -= 9; ++carry_out; }
    if (rem < -4) { rem += 9; --carry_out; }
    return static_cast<Nit>(rem);
}

/**
 * @brief Saturating multiplication — Hard Clipping / low-pass path.
 *
 * Clamps the integer product to [NIT_MIN, NIT_MAX].  Preserves sign (phase
 * direction) while truncating magnitude at ±4.  Functions analogously to
 * tanh/sigmoid nonlinear activation.
 */
[[nodiscard]] inline constexpr Nit multiply_saturated(Nit a, Nit b) noexcept
{
    int p = static_cast<int>(a) * static_cast<int>(b);
    if (p > NIT_MAX) return NIT_MAX;
    if (p < NIT_MIN) return NIT_MIN;
    return static_cast<Nit>(p);
}

/**
 * @brief Spectral Cascade carry decomposition for an arbitrary integer amplitude.
 *
 * Generalises add_with_carry() for pre-computed amplitudes outside [−4, +4].
 * After decomposition: amplitude = carry_out × 9 + remainder_out.
 *
 * §GAP-044 worked example: A = 13 → carry = 1, remainder = 4
 *   Verify: 1 × 9 + 4 = 13 ✓
 *
 * @param[in]  amplitude     Unclamped integer result (any range)
 * @param[out] carry_out     Carry value propagated to next higher dimension
 * @param[out] remainder_out Local retained value in [−4, +4]
 */
inline constexpr void carry_decompose(int amplitude, int& carry_out, int& remainder_out) noexcept
{
    carry_out     = amplitude / NONARY_CARRY_DIVISOR;
    remainder_out = amplitude - carry_out * NONARY_CARRY_DIVISOR;
    if (remainder_out >  4) { remainder_out -= 9; ++carry_out; }
    if (remainder_out < -4) { remainder_out += 9; --carry_out; }
}

// ═══════════════════════════════════════════════════════════════════════════
// §9  Statistical / information-theoretic functions  (runtime, require <cmath>)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Information loss L from saturation clipping for a Gaussian input.
 *
 *   L = ∫_{−∞}^{−4.5} P(x) dx + ∫_{4.5}^{+∞} P(x) dx
 *     = erfc( 4.5 / (σ √2) )
 *
 * L → 1 indicates severe harmonic distortion (Gibbs phenomenon).
 * Normal cognitive operation: L ≪ 0.01.
 *
 * @param sigma  Standard deviation of wavefunction amplitude distribution (σ > 0)
 * @return       Fractional information loss in [0, 1]
 * @throws std::invalid_argument if sigma ≤ 0
 */
[[nodiscard]] inline double information_loss_gaussian(double sigma)
{
    if (sigma <= 0.0)
        throw std::invalid_argument("sigma must be positive");
    return std::erfc(NONARY_CLIP_BOUNDARY / (sigma * M_SQRT2));
}

/**
 * @brief Addition overflow probability for Gaussian-distributed Nit operands.
 *
 * With a, b ~ N(0, σ²) independently, the sum a + b ~ N(0, 2σ²).
 *
 *   P( |a + b| > 4 ) = erfc( 4 / (σ√2 · √2) ) = erfc( 2 / σ )
 *
 * For σ = 1:  erfc(2) ≈ 0.00468  (4.68 %) — consistent with "< 5 %" spec.
 *
 * @param sigma  Standard deviation of each individual Nit operand (σ > 0)
 * @return       Probability of addition overflow in [0, 1]
 * @throws std::invalid_argument if sigma ≤ 0
 */
[[nodiscard]] inline double overflow_prob_gaussian(double sigma)
{
    if (sigma <= 0.0)
        throw std::invalid_argument("sigma must be positive");
    // sum variance = 2σ²  →  std dev of sum = σ√2
    // P(|sum| > 4) = erfc(4 / (σ√2 · √2)) = erfc(4 / (2σ)) = erfc(2/σ)
    return std::erfc(2.0 / sigma);
}

// ═══════════════════════════════════════════════════════════════════════════
// §10  Saturation monitoring
// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Saturation alert predicate (standalone).
 *
 * Returns true when saturation_events / total_ops > SATURATION_RATE_ALERT_THRESHOLD
 * (1 %).  Uses integer arithmetic to avoid floating-point division at runtime.
 */
[[nodiscard]] inline constexpr bool is_saturation_alert(
        uint64_t saturation_events, uint64_t total_ops) noexcept
{
    if (total_ops == 0) return false;
    // alert iff saturation_events / total_ops > 0.01
    // ↔  saturation_events * 100 > total_ops
    return saturation_events * 100u > total_ops;
}

/**
 * @brief Lightweight saturation event accumulator.
 *
 * Tracks saturation and total operation counts with integer counters.
 * Not thread-safe; wrap with external synchronisation for concurrent use.
 *
 * Usage:
 *   SaturationMonitor mon;
 *   Nit r = add_with_carry(a, b, carry);
 *   if (carry != 0) mon.record_saturated(); else mon.record_clean();
 *   if (mon.alert()) // throttle input gain
 */
class SaturationMonitor {
public:
    /// Reset all accumulators to zero.
    constexpr void reset() noexcept { saturated_ = 0; total_ = 0; }

    /// Record one operation that triggered saturation / carry emission.
    constexpr void record_saturated() noexcept { ++saturated_; ++total_; }

    /// Record one operation that completed without saturation.
    constexpr void record_clean() noexcept { ++total_; }

    /// Number of saturation events recorded.
    [[nodiscard]] constexpr uint64_t saturated_count() const noexcept { return saturated_; }

    /// Total operations recorded.
    [[nodiscard]] constexpr uint64_t total_count() const noexcept { return total_; }

    /// Saturation rate as a fraction in [0.0, 1.0].
    [[nodiscard]] double saturation_rate() const noexcept {
        if (total_ == 0) return 0.0;
        return static_cast<double>(saturated_) / static_cast<double>(total_);
    }

    /// True if saturation rate exceeds SATURATION_RATE_ALERT_THRESHOLD (1 %).
    [[nodiscard]] constexpr bool alert() const noexcept {
        return is_saturation_alert(saturated_, total_);
    }

private:
    uint64_t saturated_{0};
    uint64_t total_{0};
};

}  // namespace nikola::foundation
