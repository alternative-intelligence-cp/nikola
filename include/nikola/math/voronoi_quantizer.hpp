#pragma once
// =============================================================================
// nikola/math/voronoi_quantizer.hpp
// Phase 82 — GAP-007: Voronoi Quantization with Soft Saturation and TPDF Dithering
//
// SOURCE: Gemini Deep Research Round 2, Tasks 7-9 (December 14, 2025)
// SPEC:   docs/info/integration/sections/03_cognitive_systems/
//         01_wave_interference_processor.md §GAP-007 (lines ~2004–2065)
//
// Two-stage quantization pipeline:
//
//   Stage 1 — Hyperbolic Tangent Soft Saturation:
//       z' = A_max · tanh(z / A_scale)     (C∞, no Gibbs discontinuities)
//
//   Stage 2 — Voronoi Classification on real axis:
//       Nit = argmin_{n ∈ {-4…+4}} |z' - n|²
//
//   Optional — TPDF Dithering for audio/visual transduction:
//       z_dithered = z' + ν,    ν = U[-0.5, 0.5] + U[-0.5, 0.5]
//
// Performance spec: <5% THD, <0.01% energy drift over 10⁶ iterations.
//
// All pure functions are constexpr/noexcept. TPDF random helpers are
// decoupled via a `tpdf_sample(u1, u2)` pure function so tests never need a
// random engine.
// =============================================================================

#include <cstdint>
#include <cmath>
#include <complex>
#include <string_view>
#include "nikola/foundation/nit.hpp"   // Nit (int8_t), NIT_MIN=-4, NIT_MAX=+4

namespace nikola::math {

// Import Nit fundamentals from nikola::foundation into this namespace.
using nikola::foundation::Nit;
using nikola::foundation::NIT_MIN;
using nikola::foundation::NIT_MAX;
using nikola::foundation::NIT_ZERO;
using nikola::foundation::NIT_RADIX;

// ---------------------------------------------------------------------------
// § Enumerations
// ---------------------------------------------------------------------------

/// Dithering mode for quantize_wave.
enum class QuantizeMode : uint8_t {
    NO_DITHER   = 0,  ///< Pure two-stage pipeline (default)
    TPDF_DITHER = 1,  ///< Add triangular TPDF noise before Voronoi step
};

/// Diagnostic zone indicating which region of the soft saturator is active.
enum class SaturationZone : uint8_t {
    LINEAR_REGION    = 0,  ///< |x| ≤ A_scale (≈ linear response, slope ≈ A_max/A_scale)
    SOFT_REGION      = 1,  ///< A_scale < |x| ≤ 3*A_scale (gradual rolloff)
    SATURATED_REGION = 2,  ///< |x| > 3*A_scale (output ≈ ±A_max)
};

/// Voronoi membership classification for a post-saturation sample.
enum class VoronoiRegion : uint8_t {
    EXACT_CENTER = 0,  ///< z' is exactly an integer seed point
    INTERIOR     = 1,  ///< z' is strictly inside one Voronoi cell (unambiguous)
    BOUNDARY     = 2,  ///< z' is within VORONOI_BOUNDARY_EPSILON of a cell edge
};

// ---------------------------------------------------------------------------
// § Spec constants
// ---------------------------------------------------------------------------

/// Asymptotic saturation limit (output bound). 0.5 headroom above Nit max (4.0).
/// Spec: "A_max = 4.5: Asymptotic limit with 0.5 headroom buffer"
inline constexpr double A_MAX = 4.5;

/// Scale factor in the tanh argument — calibrated to pilot wave energy.
/// Spec: "A_scale = 2.5: Calibrated to pilot wave initialization energy"
inline constexpr double A_SCALE = 2.5;

/// Headroom between A_MAX and the highest Nit value (NIT_MAX = 4).
/// Prevents exact saturation at any representable Nit.
inline constexpr double A_HEADROOM = A_MAX - static_cast<double>(NIT_MAX);   // 0.5

/// Gain (slope) of the soft saturator at the origin: dz'/dx|_{x=0} = A_max/A_scale.
/// At small |x|, soft_saturate behaves as a linear amplifier with this slope.
inline constexpr double A_ORIGIN_SLOPE = A_MAX / A_SCALE;          // 1.8

/// Number of Nit symbols in the balanced nonary alphabet.
/// Seeds are at integer positions {-4, -3, …, 0, …, +3, +4} on the real axis.
inline constexpr int NIT_COUNT = NIT_RADIX;                        // 9

/// Half-width of one TPDF uniform component U[-TPDF_HALF, +TPDF_HALF].
/// Spec: ν = U[-0.5, 0.5] + U[-0.5, 0.5]
inline constexpr double TPDF_HALF = 0.5;

/// Full output range of the TPDF noise: [-TPDF_RANGE, +TPDF_RANGE].
inline constexpr double TPDF_RANGE = 2.0 * TPDF_HALF;             // 1.0

/// Theoretical variance of one U[-0.5, 0.5] component: (1/12)*(1.0)² = 1/12.
inline constexpr double TPDF_COMPONENT_VARIANCE = 1.0 / 12.0;

/// Total TPDF variance: Var(U1 + U2) = 2 * 1/12 = 1/6.
inline constexpr double TPDF_VARIANCE = 2.0 * TPDF_COMPONENT_VARIANCE;   // ≈ 0.16667

/// Standard deviation of TPDF noise: sqrt(1/6) ≈ 0.40825.
inline constexpr double TPDF_STDDEV = 0.408248290463863;    // sqrt(1.0/6.0)

/// TPDF noise mean is exactly 0 (symmetric triangular distribution).
inline constexpr double TPDF_MEAN = 0.0;

/// Performance specification: THD must remain below this fraction.
/// Spec: "<5% THD (Total Harmonic Distortion)"
inline constexpr double THD_LIMIT = 0.05;

/// Performance specification: energy drift per normalised iteration.
/// Spec: "<0.01% energy drift over 10^6 iterations"
inline constexpr double ENERGY_DRIFT_LIMIT = 0.0001;

/// Epsilon for classifying a sample as being on a Voronoi cell boundary.
/// A boundary is within ±VORONOI_BOUNDARY_EPSILON of a half-integer value.
inline constexpr double VORONOI_BOUNDARY_EPSILON = 1.0e-9;

/// The Voronoi cell boundary spacing (distance between adjacent seeds = 1.0).
inline constexpr double VORONOI_CELL_WIDTH = 1.0;

/// Half the cell width: boundary between seed n and n+1 is at n + 0.5.
inline constexpr double VORONOI_HALF_CELL = 0.5;

// ---------------------------------------------------------------------------
// § Stage 1: Hyperbolic Tangent Soft Saturation
// ---------------------------------------------------------------------------

/// z' = A_max · tanh(z / A_scale)
/// Maps ℝ → (−A_max, +A_max).  C∞, odd function, eliminates Gibbs artifacts.
[[nodiscard]] constexpr double soft_saturate(double x) noexcept {
    return A_MAX * std::tanh(x / A_SCALE);
}

/// Derivative of soft_saturate: dz'/dx = (A_max/A_scale) · sech²(x/A_scale).
/// Used to verify linearity at origin and compression at saturation.
[[nodiscard]] constexpr double soft_saturate_prime(double x) noexcept {
    double s = std::tanh(x / A_SCALE);          // tanh(x/A_scale)
    return A_ORIGIN_SLOPE * (1.0 - s * s);       // A_max/A_scale * sech²
}

/// Returns the saturation zone for a given input magnitude.
[[nodiscard]] constexpr SaturationZone saturation_zone(double x) noexcept {
    double ax = x < 0.0 ? -x : x;
    if (ax <= A_SCALE)         return SaturationZone::LINEAR_REGION;
    if (ax <= 3.0 * A_SCALE)   return SaturationZone::SOFT_REGION;
    return                            SaturationZone::SATURATED_REGION;
}

/// Headroom remaining before A_MAX for the saturated output.
/// Always positive (A_max is asymptotic — never equals ±A_max).
[[nodiscard]] constexpr double saturation_headroom(double x) noexcept {
    double sx = soft_saturate(x);
    return A_MAX - (sx < 0.0 ? -sx : sx);
}

// ---------------------------------------------------------------------------
// § Stage 2: Voronoi Classification
// ---------------------------------------------------------------------------

/// Voronoi seed position for Nit n (trivially equal to the integer value).
[[nodiscard]] constexpr double voronoi_seed(Nit n) noexcept {
    return static_cast<double>(n);
}

/// Squared Euclidean distance from z_real to Voronoi seed n.
[[nodiscard]] constexpr double voronoi_distance_sq(double z_real, Nit n) noexcept {
    double d = z_real - voronoi_seed(n);
    return d * d;
}

/// Find nearest Nit seed to z_real (argmin |z' - n|²  for n ∈ {-4…+4}).
/// Ties (exactly at boundary n + 0.5) round toward +∞ (higher Nit).
[[nodiscard]] constexpr Nit nearest_nit(double z_real) noexcept {
    // Hard-clamp to reachable range first (soft_saturate guarantees < A_MAX,
    // but the user might call nearest_nit directly with arbitrary values).
    if (z_real <= static_cast<double>(NIT_MIN)) return NIT_MIN;
    if (z_real >= static_cast<double>(NIT_MAX)) return NIT_MAX;

    // Floor to integer part; fractional part determines whether to round up.
    // Effective: round(z_real) clamped to [NIT_MIN, NIT_MAX].
    double rounded = std::floor(z_real + 0.5);
    int i = static_cast<int>(rounded);
    if (i < static_cast<int>(NIT_MIN)) i = static_cast<int>(NIT_MIN);
    if (i > static_cast<int>(NIT_MAX)) i = static_cast<int>(NIT_MAX);
    return static_cast<Nit>(i);
}

/// Classify whether z_real lies at an exact center, well inside a cell,
/// or near a cell boundary.
[[nodiscard]] constexpr VoronoiRegion voronoi_region(double z_real) noexcept {
    double frac = z_real - std::floor(z_real);  // fractional part in [0,1)
    // Near exact integer (center)?
    if (frac < VORONOI_BOUNDARY_EPSILON || frac > (1.0 - VORONOI_BOUNDARY_EPSILON))
        return VoronoiRegion::EXACT_CENTER;
    // Near boundary (*.5)?
    double dist_to_half = frac - 0.5;
    if (dist_to_half < 0.0) dist_to_half = -dist_to_half;
    if (dist_to_half < VORONOI_BOUNDARY_EPSILON)
        return VoronoiRegion::BOUNDARY;
    return VoronoiRegion::INTERIOR;
}

/// True when two values z_real map to the same Nit.
[[nodiscard]] constexpr bool same_voronoi_cell(double a, double b) noexcept {
    return nearest_nit(a) == nearest_nit(b);
}

// ---------------------------------------------------------------------------
// § Full pipeline: soft_saturate → nearest_nit
// ---------------------------------------------------------------------------

/// Quantize a real-valued sample through the two-stage pipeline (no dither).
/// quantize_real(x) = nearest_nit(soft_saturate(x))
[[nodiscard]] constexpr Nit quantize_real(double x) noexcept {
    return nearest_nit(soft_saturate(x));
}

/// Quantize with explicit TPDF dither already applied to the saturated value.
/// Caller pre-generates u1, u2 ∈ [-0.5, +0.5] (e.g. from a URNG).
/// Result: nearest_nit(soft_saturate(x) + u1 + u2)
[[nodiscard]] constexpr Nit quantize_real_dithered(
    double x, double u1, double u2) noexcept
{
    return nearest_nit(soft_saturate(x) + u1 + u2);
}

/// Quantize a complex wavefunction.
/// The imaginary component is projected onto the real axis (wavefunction collapse).
/// With QuantizeMode::TPDF_DITHER, u1 and u2 must be pre-generated ∈ [-0.5, 0.5].
[[nodiscard]] constexpr Nit quantize_wave(
    std::complex<double> wave,
    QuantizeMode mode = QuantizeMode::NO_DITHER,
    double u1 = 0.0, double u2 = 0.0) noexcept
{
    double sat_real = soft_saturate(wave.real());
    if (mode == QuantizeMode::TPDF_DITHER)
        sat_real += (u1 + u2);
    return nearest_nit(sat_real);
}

/// Convenience overload for plain real input.
[[nodiscard]] constexpr Nit quantize_wave(
    double real_part,
    QuantizeMode mode = QuantizeMode::NO_DITHER,
    double u1 = 0.0, double u2 = 0.0) noexcept
{
    return quantize_wave(std::complex<double>{real_part, 0.0}, mode, u1, u2);
}

// ---------------------------------------------------------------------------
// § TPDF Dithering helpers
// ---------------------------------------------------------------------------

/// Pure TPDF sample given two pre-generated uniform values.
/// u1, u2 must each lie in [-0.5, +0.5].
/// Result ν ∈ [-1.0, +1.0] with triangular distribution (mean=0, var=1/6).
[[nodiscard]] constexpr double tpdf_sample(double u1, double u2) noexcept {
    return u1 + u2;
}

/// True when |ν| ≤ TPDF_RANGE — validates a sample is within spec bounds.
[[nodiscard]] constexpr bool tpdf_sample_valid(double nu) noexcept {
    return (nu >= -TPDF_RANGE) && (nu <= TPDF_RANGE);
}

/// Midpoint of TPDF distribution (always 0.0 by symmetry).
[[nodiscard]] constexpr double tpdf_theoretical_mean() noexcept {
    return TPDF_MEAN;
}

/// Theoretical variance of TPDF (sum of two U distributions).
[[nodiscard]] constexpr double tpdf_theoretical_variance() noexcept {
    return TPDF_VARIANCE;
}

// ---------------------------------------------------------------------------
// § Performance assessment predicates
// ---------------------------------------------------------------------------

/// True if the observed THD fraction is within spec (<5%).
[[nodiscard]] constexpr bool thd_within_spec(double thd_fraction) noexcept {
    return thd_fraction < THD_LIMIT;
}

/// True if the observed relative energy drift is within spec (<0.01%).
[[nodiscard]] constexpr bool energy_drift_within_spec(double drift_fraction) noexcept {
    return drift_fraction < ENERGY_DRIFT_LIMIT;
}

/// Compute THD from harmonic RMS amplitudes.
/// fundamental_rms: amplitude of the fundamental tone.
/// harmonic_rms_sum_sq: sum of squares of all identified harmonic amplitudes.
/// Returns sqrt(sum_sq) / fundamental_rms.
[[nodiscard]] constexpr double compute_thd(
    double fundamental_rms, double harmonic_rms_sum_sq) noexcept
{
    if (fundamental_rms <= 0.0) return 0.0;
    return std::sqrt(harmonic_rms_sum_sq) / fundamental_rms;
}

/// Relative energy drift given initial and final energy values.
[[nodiscard]] constexpr double relative_energy_drift(
    double initial_energy, double final_energy) noexcept
{
    if (initial_energy <= 0.0) return 0.0;
    double diff = final_energy - initial_energy;
    if (diff < 0.0) diff = -diff;
    return diff / initial_energy;
}

// ---------------------------------------------------------------------------
// § Label helpers
// ---------------------------------------------------------------------------

[[nodiscard]] constexpr std::string_view quantize_mode_label(QuantizeMode m) noexcept {
    switch (m) {
        case QuantizeMode::NO_DITHER:   return "NO_DITHER";
        case QuantizeMode::TPDF_DITHER: return "TPDF_DITHER";
    }
    return "UNKNOWN_MODE";
}

[[nodiscard]] constexpr std::string_view saturation_zone_label(SaturationZone z) noexcept {
    switch (z) {
        case SaturationZone::LINEAR_REGION:    return "LINEAR_REGION";
        case SaturationZone::SOFT_REGION:      return "SOFT_REGION";
        case SaturationZone::SATURATED_REGION: return "SATURATED_REGION";
    }
    return "UNKNOWN_ZONE";
}

[[nodiscard]] constexpr std::string_view voronoi_region_label(VoronoiRegion r) noexcept {
    switch (r) {
        case VoronoiRegion::EXACT_CENTER: return "EXACT_CENTER";
        case VoronoiRegion::INTERIOR:     return "INTERIOR";
        case VoronoiRegion::BOUNDARY:     return "BOUNDARY";
    }
    return "UNKNOWN_REGION";
}

} // namespace nikola::math
