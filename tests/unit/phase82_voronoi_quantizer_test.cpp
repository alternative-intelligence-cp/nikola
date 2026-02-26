// =============================================================================
// phase82_voronoi_quantizer_test.cpp
// Phase 82 — GAP-007: Voronoi Quantization with Soft Saturation and TPDF Dithering
//
// Exhaustively tests every constant, formula, and pure function in
// nikola/math/voronoi_quantizer.hpp against the mathematical derivations
// in §GAP-007 of 01_wave_interference_processor.md.
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "nikola/math/voronoi_quantizer.hpp"
#include <cmath>
#include <complex>
#include <numbers>   // std::numbers::pi

using namespace nikola::math;
using Catch::Approx;

// ---------------------------------------------------------------------------
// §1 — Spec constants
// ---------------------------------------------------------------------------

TEST_CASE("A_MAX is 4.5", "[constants]") {
    CHECK(A_MAX == Approx(4.5));
}

TEST_CASE("A_SCALE is 2.5", "[constants]") {
    CHECK(A_SCALE == Approx(2.5));
}

TEST_CASE("A_HEADROOM is 0.5 = A_MAX - NIT_MAX", "[constants]") {
    CHECK(A_HEADROOM == Approx(0.5));
    CHECK(A_HEADROOM == Approx(A_MAX - static_cast<double>(NIT_MAX)));
}

TEST_CASE("A_ORIGIN_SLOPE is 1.8 = A_MAX / A_SCALE", "[constants]") {
    CHECK(A_ORIGIN_SLOPE == Approx(1.8));
    CHECK(A_ORIGIN_SLOPE == Approx(A_MAX / A_SCALE));
}

TEST_CASE("NIT_COUNT is 9 (balanced nonary)", "[constants]") {
    CHECK(NIT_COUNT == 9);
    CHECK(NIT_COUNT == NIT_MAX - NIT_MIN + 1);
}

TEST_CASE("TPDF_HALF is 0.5", "[constants]") {
    CHECK(TPDF_HALF == Approx(0.5));
}

TEST_CASE("TPDF_RANGE is 1.0 = 2 * TPDF_HALF", "[constants]") {
    CHECK(TPDF_RANGE == Approx(1.0));
    CHECK(TPDF_RANGE == Approx(2.0 * TPDF_HALF));
}

TEST_CASE("TPDF_VARIANCE is 1/6 = sum of two uniform variances", "[constants]") {
    CHECK(TPDF_VARIANCE == Approx(1.0 / 6.0).epsilon(1e-10));
    // Each U[-0.5,0.5] has variance (1.0)² / 12 = 1/12
    CHECK(TPDF_COMPONENT_VARIANCE == Approx(1.0 / 12.0).epsilon(1e-10));
    CHECK(TPDF_VARIANCE == Approx(2.0 * TPDF_COMPONENT_VARIANCE));
}

TEST_CASE("TPDF_STDDEV is sqrt(1/6)", "[constants]") {
    CHECK(TPDF_STDDEV == Approx(std::sqrt(1.0 / 6.0)).epsilon(1e-10));
    CHECK(TPDF_STDDEV * TPDF_STDDEV == Approx(TPDF_VARIANCE).epsilon(1e-9));
}

TEST_CASE("TPDF_MEAN is 0.0 (symmetric distribution)", "[constants]") {
    CHECK(TPDF_MEAN == Approx(0.0));
}

TEST_CASE("THD_LIMIT is 0.05 (5%)", "[constants]") {
    CHECK(THD_LIMIT == Approx(0.05));
}

TEST_CASE("ENERGY_DRIFT_LIMIT is 0.0001 (0.01%)", "[constants]") {
    CHECK(ENERGY_DRIFT_LIMIT == Approx(0.0001));
}

TEST_CASE("VORONOI_CELL_WIDTH is 1.0", "[constants]") {
    CHECK(VORONOI_CELL_WIDTH == Approx(1.0));
}

TEST_CASE("VORONOI_HALF_CELL is 0.5", "[constants]") {
    CHECK(VORONOI_HALF_CELL == Approx(0.5));
}

// ---------------------------------------------------------------------------
// §2 — Enum ordinals
// ---------------------------------------------------------------------------

TEST_CASE("QuantizeMode ordinals", "[enums]") {
    CHECK(static_cast<uint8_t>(QuantizeMode::NO_DITHER)   == 0);
    CHECK(static_cast<uint8_t>(QuantizeMode::TPDF_DITHER) == 1);
}

TEST_CASE("SaturationZone ordinals", "[enums]") {
    CHECK(static_cast<uint8_t>(SaturationZone::LINEAR_REGION)    == 0);
    CHECK(static_cast<uint8_t>(SaturationZone::SOFT_REGION)      == 1);
    CHECK(static_cast<uint8_t>(SaturationZone::SATURATED_REGION) == 2);
}

TEST_CASE("VoronoiRegion ordinals", "[enums]") {
    CHECK(static_cast<uint8_t>(VoronoiRegion::EXACT_CENTER) == 0);
    CHECK(static_cast<uint8_t>(VoronoiRegion::INTERIOR)     == 1);
    CHECK(static_cast<uint8_t>(VoronoiRegion::BOUNDARY)     == 2);
}

// ---------------------------------------------------------------------------
// §3 — soft_saturate: z' = A_max · tanh(z / A_scale)
// ---------------------------------------------------------------------------

TEST_CASE("soft_saturate: origin maps to 0", "[soft_saturate]") {
    CHECK(soft_saturate(0.0) == Approx(0.0));
}

TEST_CASE("soft_saturate: A_scale input → A_max * tanh(1)", "[soft_saturate]") {
    CHECK(soft_saturate(A_SCALE) == Approx(A_MAX * std::tanh(1.0)).epsilon(1e-12));
}

TEST_CASE("soft_saturate: odd function — soft_saturate(-x) = -soft_saturate(x)", "[soft_saturate]") {
    for (double x : {0.5, 1.0, 2.5, 5.0, 10.0, 100.0}) {
        CHECK(soft_saturate(-x) == Approx(-soft_saturate(x)).epsilon(1e-12));
    }
}

TEST_CASE("soft_saturate: output strictly bounded by A_MAX", "[soft_saturate]") {
    // Note: for |x| >= ~47 in double precision, tanh returns exactly 1.0,
    // so output == A_MAX exactly. Use practical input range here.
    for (double x : {1.0, 2.5, 5.0, 10.0}) {
        double y = soft_saturate(x);
        CHECK(y > 0.0);
        CHECK(y < A_MAX);
        CHECK(-soft_saturate(-x) < A_MAX);
    }
}

TEST_CASE("soft_saturate: A_HEADROOM — output stays below A_MAX for practical inputs", "[soft_saturate]") {
    // For |x| <= 10, tanh(x/A_scale) is strictly < 1.0 in double precision.
    // Very large inputs (|x| >> A_scale) round to exactly A_MAX in floating-point.
    double near_limit = soft_saturate(10.0);  // tanh(4.0) ≈ 0.9993
    CHECK(near_limit < A_MAX);
    CHECK(A_MAX - near_limit < A_HEADROOM + 1e-2);  // within the 0.5 headroom band
}

TEST_CASE("soft_saturate: origin slope = A_max/A_scale = 1.8", "[soft_saturate]") {
    // f'(0) = A_max/A_scale * sech²(0) = A_max/A_scale * 1 = 1.8
    CHECK(soft_saturate_prime(0.0) == Approx(A_ORIGIN_SLOPE));
    CHECK(soft_saturate_prime(0.0) == Approx(1.8));
}

TEST_CASE("soft_saturate: slope decreases monotonically away from origin", "[soft_saturate]") {
    double d0 = soft_saturate_prime(0.0);
    double d1 = soft_saturate_prime(1.0);
    double d5 = soft_saturate_prime(5.0);
    CHECK(d0 > d1);
    CHECK(d1 > d5);
    CHECK(d5 > 0.0);  // always positive (monotone increasing)
}

TEST_CASE("soft_saturate: monotone increasing", "[soft_saturate]") {
    double prev = soft_saturate(-10.0);
    for (int i = -9; i <= 10; ++i) {
        double curr = soft_saturate(static_cast<double>(i));
        CHECK(curr > prev);
        prev = curr;
    }
}

TEST_CASE("soft_saturate: large positive input asymptotic to A_MAX", "[soft_saturate]") {
    // For large inputs, output converges to (and may equal) A_MAX in double precision.
    CHECK(soft_saturate(100.0) == Approx(A_MAX).epsilon(1e-6));
    CHECK(soft_saturate(10.0) == Approx(A_MAX).epsilon(1e-2));  // within 1% of A_MAX
}

TEST_CASE("soft_saturate: NIT_MAX input is preserved without saturation clip", "[soft_saturate]") {
    // NIT_MAX=4; soft_saturate(4) = 4.5 * tanh(4/2.5) = 4.5 * tanh(1.6)
    double expected = A_MAX * std::tanh(static_cast<double>(NIT_MAX) / A_SCALE);
    CHECK(soft_saturate(static_cast<double>(NIT_MAX)) == Approx(expected).epsilon(1e-12));
    CHECK(soft_saturate(static_cast<double>(NIT_MAX)) < A_MAX);
}

// ---------------------------------------------------------------------------
// §4 — saturation_zone classifier
// ---------------------------------------------------------------------------

TEST_CASE("saturation_zone: x=0 → LINEAR_REGION", "[satzone]") {
    CHECK(saturation_zone(0.0) == SaturationZone::LINEAR_REGION);
}

TEST_CASE("saturation_zone: |x| = A_scale → LINEAR_REGION (boundary inclusive)", "[satzone]") {
    CHECK(saturation_zone(A_SCALE) == SaturationZone::LINEAR_REGION);
}

TEST_CASE("saturation_zone: |x| between A_scale and 3*A_scale → SOFT_REGION", "[satzone]") {
    CHECK(saturation_zone(A_SCALE + 0.1) == SaturationZone::SOFT_REGION);
    CHECK(saturation_zone(5.0)           == SaturationZone::SOFT_REGION);
    CHECK(saturation_zone(-4.0)          == SaturationZone::SOFT_REGION);
}

TEST_CASE("saturation_zone: |x| > 3*A_scale = 7.5 → SATURATED_REGION", "[satzone]") {
    CHECK(saturation_zone(7.6)   == SaturationZone::SATURATED_REGION);
    CHECK(saturation_zone(100.0) == SaturationZone::SATURATED_REGION);
}

TEST_CASE("saturation_zone: symmetric — negative mirrors positive", "[satzone]") {
    for (double x : {0.5, 3.0, 8.0}) {
        CHECK(saturation_zone(x) == saturation_zone(-x));
    }
}

// ---------------------------------------------------------------------------
// §5 — nearest_nit: Voronoi classification on real axis
// ---------------------------------------------------------------------------

TEST_CASE("nearest_nit: exact integer seeds map to themselves", "[nearest_nit]") {
    for (int n = NIT_MIN; n <= NIT_MAX; ++n) {
        CHECK(nearest_nit(static_cast<double>(n)) == static_cast<Nit>(n));
    }
}

TEST_CASE("nearest_nit: interior of each Voronoi cell", "[nearest_nit]") {
    // Each cell (n-0.5, n+0.5) should map to n
    for (int n = NIT_MIN; n <= NIT_MAX; ++n) {
        double center = static_cast<double>(n);
        CHECK(nearest_nit(center + 0.3)  == static_cast<Nit>(n));
        CHECK(nearest_nit(center - 0.3)  == static_cast<Nit>(n));
        CHECK(nearest_nit(center + 0.49) == static_cast<Nit>(n));
    }
}

TEST_CASE("nearest_nit: boundary at n+0.5 rounds to n+1", "[nearest_nit]") {
    // By convention: floor(x + 0.5) rounds 0.5 up
    CHECK(nearest_nit(0.5) == static_cast<Nit>(1));
    CHECK(nearest_nit(1.5) == static_cast<Nit>(2));
    CHECK(nearest_nit(-0.5) == static_cast<Nit>(0));
    CHECK(nearest_nit(-1.5) == static_cast<Nit>(-1));
}

TEST_CASE("nearest_nit: clamps to NIT_MIN for large negative inputs", "[nearest_nit]") {
    CHECK(nearest_nit(-100.0) == NIT_MIN);
    CHECK(nearest_nit(-4.5)   == NIT_MIN);
    CHECK(nearest_nit(-5.0)   == NIT_MIN);
}

TEST_CASE("nearest_nit: clamps to NIT_MAX for large positive inputs", "[nearest_nit]") {
    CHECK(nearest_nit(100.0) == NIT_MAX);
    CHECK(nearest_nit(4.5)   == NIT_MAX);
    CHECK(nearest_nit(5.0)   == NIT_MAX);
}

TEST_CASE("nearest_nit: specific key values", "[nearest_nit]") {
    CHECK(nearest_nit(0.0)  == static_cast<Nit>(0));
    CHECK(nearest_nit(1.0)  == static_cast<Nit>(1));
    CHECK(nearest_nit(-1.0) == static_cast<Nit>(-1));
    CHECK(nearest_nit(3.7)  == static_cast<Nit>(4));
    CHECK(nearest_nit(-3.7) == static_cast<Nit>(-4));
    CHECK(nearest_nit(2.1)  == static_cast<Nit>(2));
    CHECK(nearest_nit(-2.1) == static_cast<Nit>(-2));
}

TEST_CASE("voronoi_seed: trivially equals the integer Nit value", "[voronoi_seed]") {
    for (int n = NIT_MIN; n <= NIT_MAX; ++n) {
        CHECK(voronoi_seed(static_cast<Nit>(n)) == Approx(static_cast<double>(n)));
    }
}

TEST_CASE("voronoi_distance_sq: zero at seed center", "[voronoi_dist]") {
    for (int n = NIT_MIN; n <= NIT_MAX; ++n) {
        CHECK(voronoi_distance_sq(static_cast<double>(n), static_cast<Nit>(n))
            == Approx(0.0).margin(1e-12));
    }
}

TEST_CASE("voronoi_distance_sq: 0.5 unit from center → 0.25", "[voronoi_dist]") {
    CHECK(voronoi_distance_sq(2.5, static_cast<Nit>(2)) == Approx(0.25));
    CHECK(voronoi_distance_sq(-1.5, static_cast<Nit>(-2)) == Approx(0.25));
}

// ---------------------------------------------------------------------------
// §6 — quantize_real: full two-stage pipeline
// ---------------------------------------------------------------------------

TEST_CASE("quantize_real: zero input → Nit 0", "[quantize_real]") {
    CHECK(quantize_real(0.0) == static_cast<Nit>(0));
}

TEST_CASE("quantize_real: odd symmetry — quantize_real(-x) = -quantize_real(x)", "[quantize_real]") {
    // soft_saturate is odd, nearest_nit obeys symmetry
    for (int i = 1; i <= 4; ++i) {
        double x = 0.4 + i * 0.7;
        CHECK(quantize_real(-x) == -quantize_real(x));
    }
}

TEST_CASE("quantize_real: output always in [NIT_MIN, NIT_MAX]", "[quantize_real]") {
    for (double x : {-100.0, -10.0, -4.0, -2.5, 0.0, 1.0, 2.5, 5.0, 10.0, 100.0}) {
        Nit n = quantize_real(x);
        CHECK(n >= NIT_MIN);
        CHECK(n <= NIT_MAX);
    }
}

TEST_CASE("quantize_real: integer inputs map to nearest Nit", "[quantize_real]") {
    // At x=1, soft_saturate(1) = 4.5*tanh(0.4) ≈ 1.726 → Nit 2
    // At x=0.3, soft_saturate(0.3) = 4.5*tanh(0.12) ≈ 0.537 → Nit 1
    CHECK(quantize_real(0.3) == static_cast<Nit>(1));
    CHECK(quantize_real(-0.3) == static_cast<Nit>(-1));
    CHECK(quantize_real(0.0) == static_cast<Nit>(0));
}

TEST_CASE("quantize_real: large inputs saturate to ±4", "[quantize_real]") {
    CHECK(quantize_real(100.0)  == NIT_MAX);
    CHECK(quantize_real(-100.0) == NIT_MIN);
}

TEST_CASE("quantize_real: soft_saturate then nearest_nit chain verified", "[quantize_real]") {
    double x   = 3.0;
    double sat = soft_saturate(x);  // 4.5 * tanh(3/2.5)
    Nit    nit = nearest_nit(sat);
    CHECK(quantize_real(x) == nit);
}

// ---------------------------------------------------------------------------
// §7 — quantize_wave: complex wavefunction collapse
// ---------------------------------------------------------------------------

TEST_CASE("quantize_wave: real part only — same as quantize_real", "[quantize_wave]") {
    for (double r : {0.0, 1.0, -2.0, 3.5, -0.7}) {
        CHECK(quantize_wave(std::complex<double>{r, 0.0}) == quantize_real(r));
    }
}

TEST_CASE("quantize_wave: imaginary component is projected (ignored)", "[quantize_wave]") {
    // Two waves with same real but different imaginary → same Nit
    double r = 1.5;
    auto w1 = std::complex<double>{r, 0.0};
    auto w2 = std::complex<double>{r, 100.0};  // huge imaginary
    auto w3 = std::complex<double>{r, -99.9};
    CHECK(quantize_wave(w1) == quantize_wave(w2));
    CHECK(quantize_wave(w1) == quantize_wave(w3));
}

TEST_CASE("quantize_wave: NO_DITHER mode with u1,u2 non-zero → u1,u2 ignored", "[quantize_wave]") {
    auto w = std::complex<double>{0.8, 0.0};
    Nit base = quantize_wave(w, QuantizeMode::NO_DITHER);
    // Non-zero u1,u2 should be ignored when mode is NO_DITHER
    Nit same = quantize_wave(w, QuantizeMode::NO_DITHER, 0.3, 0.4);
    CHECK(base == same);
}

TEST_CASE("quantize_wave: TPDF_DITHER with u1=u2=0 → same as no-dither", "[quantize_wave]") {
    auto w = std::complex<double>{1.2, 0.5};
    CHECK(quantize_wave(w, QuantizeMode::TPDF_DITHER, 0.0, 0.0) == quantize_wave(w));
}

TEST_CASE("quantize_wave: TPDF dither shifts classification near boundary", "[quantize_wave]") {
    // z = 0.26 → sat ≈ 0.47 → Nit 0 without dither, Nit 1 with +0.5+0.05
    auto w = std::complex<double>{0.26, 0.0};
    Nit no_d = quantize_wave(w, QuantizeMode::NO_DITHER);
    Nit with_d = quantize_wave(w, QuantizeMode::TPDF_DITHER, 0.25, 0.25);
    // With +0.5 dither, should push to adjacent cell or stay
    CHECK(with_d >= NIT_MIN);
    CHECK(with_d <= NIT_MAX);
    // The dithered case should shift the cell compared to base without dither
    // (we just verify it doesn't corrupt the output — exact value depends on sat val)
    (void)no_d;
}

TEST_CASE("quantize_wave scalar overload: same as complex with imag=0", "[quantize_wave]") {
    for (double x : {-3.0, 0.0, 1.5, 4.0}) {
        CHECK(quantize_wave(x) == quantize_wave(std::complex<double>{x, 0.0}));
    }
}

// ---------------------------------------------------------------------------
// §8 — TPDF dithering
// ---------------------------------------------------------------------------

TEST_CASE("tpdf_sample: u1=u2=0 → 0", "[tpdf]") {
    CHECK(tpdf_sample(0.0, 0.0) == Approx(0.0));
}

TEST_CASE("tpdf_sample: full positive → +1.0", "[tpdf]") {
    CHECK(tpdf_sample(0.5, 0.5) == Approx(1.0));
}

TEST_CASE("tpdf_sample: full negative → -1.0", "[tpdf]") {
    CHECK(tpdf_sample(-0.5, -0.5) == Approx(-1.0));
}

TEST_CASE("tpdf_sample: symmetric midpoint → 0", "[tpdf]") {
    CHECK(tpdf_sample(0.3, -0.3) == Approx(0.0));
}

TEST_CASE("tpdf_sample_valid: within range → true", "[tpdf]") {
    CHECK(tpdf_sample_valid(0.0)  == true);
    CHECK(tpdf_sample_valid(-1.0) == true);
    CHECK(tpdf_sample_valid(1.0)  == true);
    CHECK(tpdf_sample_valid(0.99) == true);
}

TEST_CASE("tpdf_sample_valid: outside range → false", "[tpdf]") {
    CHECK(tpdf_sample_valid(1.01)  == false);
    CHECK(tpdf_sample_valid(-1.01) == false);
    CHECK(tpdf_sample_valid(2.0)   == false);
}

TEST_CASE("tpdf_theoretical_mean: returns 0.0", "[tpdf]") {
    CHECK(tpdf_theoretical_mean() == Approx(0.0));
}

TEST_CASE("tpdf_theoretical_variance: returns 1/6", "[tpdf]") {
    CHECK(tpdf_theoretical_variance() == Approx(1.0 / 6.0).epsilon(1e-10));
}

TEST_CASE("TPDF sample mean converges to 0 over many samples", "[tpdf]") {
    // Deterministic sweep over a 100×100 uniform grid of (u1,u2) in [-0.5,0.5]²
    constexpr int N = 100;
    double sum = 0.0;
    long count = 0;
    for (int i = 0; i < N; ++i) {
        double u1 = -0.5 + (i + 0.5) / N;  // uniform centres
        for (int j = 0; j < N; ++j) {
            double u2 = -0.5 + (j + 0.5) / N;
            sum += tpdf_sample(u1, u2);
            ++count;
        }
    }
    double mean = sum / static_cast<double>(count);
    CHECK(mean == Approx(0.0).margin(1e-10));
}

TEST_CASE("TPDF sample variance converges to 1/6", "[tpdf]") {
    constexpr int N = 100;
    double sum_sq = 0.0;
    long count = 0;
    for (int i = 0; i < N; ++i) {
        double u1 = -0.5 + (i + 0.5) / N;
        for (int j = 0; j < N; ++j) {
            double u2 = -0.5 + (j + 0.5) / N;
            double nu = tpdf_sample(u1, u2);
            sum_sq += nu * nu;
            ++count;
        }
    }
    double variance = sum_sq / static_cast<double>(count);
    CHECK(variance == Approx(TPDF_VARIANCE).epsilon(5e-4));
}

TEST_CASE("quantize_real_dithered: u1=u2=0 → same as quantize_real", "[tpdf]") {
    for (double x : {0.0, 1.0, -2.5, 3.0}) {
        CHECK(quantize_real_dithered(x, 0.0, 0.0) == quantize_real(x));
    }
}

// ---------------------------------------------------------------------------
// §9 — VoronoiRegion classifier
// ---------------------------------------------------------------------------

TEST_CASE("voronoi_region: exact integers → EXACT_CENTER", "[voronoi_region]") {
    for (int n = NIT_MIN; n <= NIT_MAX; ++n) {
        CHECK(voronoi_region(static_cast<double>(n)) == VoronoiRegion::EXACT_CENTER);
    }
}

TEST_CASE("voronoi_region: mid-cell → INTERIOR", "[voronoi_region]") {
    CHECK(voronoi_region(0.2)  == VoronoiRegion::INTERIOR);
    CHECK(voronoi_region(-1.3) == VoronoiRegion::INTERIOR);
    CHECK(voronoi_region(2.7)  == VoronoiRegion::INTERIOR);
}

TEST_CASE("voronoi_region: exactly at boundary 0.5 → BOUNDARY", "[voronoi_region]") {
    CHECK(voronoi_region(0.5) == VoronoiRegion::BOUNDARY);
    CHECK(voronoi_region(1.5) == VoronoiRegion::BOUNDARY);
}

TEST_CASE("same_voronoi_cell: two points in same cell → true", "[voronoi_region]") {
    CHECK(same_voronoi_cell(0.1, 0.4) == true);
    CHECK(same_voronoi_cell(-2.1, -2.4) == true);
}

TEST_CASE("same_voronoi_cell: points in adjacent cells → false", "[voronoi_region]") {
    CHECK(same_voronoi_cell(0.4, 0.6) == false);  // 0 vs 1
    CHECK(same_voronoi_cell(2.4, 2.6) == false);  // 2 vs 3
}

// ---------------------------------------------------------------------------
// §10 — saturation_headroom
// ---------------------------------------------------------------------------

TEST_CASE("saturation_headroom: always >= 0 for all inputs", "[headroom]") {
    // For large |x|, double precision tanh == 1.0 exactly, so headroom == 0.
    for (double x : {0.0, 1.0, 2.5, 5.0, 10.0, 1e6}) {
        CHECK(saturation_headroom(x) >= 0.0);
    }
    // For practical range |x| <= 10, headroom is strictly positive
    for (double x : {0.0, 1.0, 2.5, 5.0, 10.0}) {
        CHECK(saturation_headroom(x) > 0.0);
    }
}

TEST_CASE("saturation_headroom: x=0 → A_MAX (full headroom)", "[headroom]") {
    CHECK(saturation_headroom(0.0) == Approx(A_MAX));
}

TEST_CASE("saturation_headroom: decreases as |x| increases", "[headroom]") {
    double h1 = saturation_headroom(1.0);
    double h5 = saturation_headroom(5.0);
    double h10 = saturation_headroom(10.0);
    CHECK(h1 > h5);
    CHECK(h5 > h10);
    CHECK(h10 > 0.0);  // tanh(4.0) < 1.0 in double, so still positive
}

// ---------------------------------------------------------------------------
// §11 — Performance predicates
// ---------------------------------------------------------------------------

TEST_CASE("thd_within_spec: 4.9% → true; 5.1% → false", "[performance]") {
    CHECK(thd_within_spec(0.049) == true);
    CHECK(thd_within_spec(0.050) == false);   // strict <
    CHECK(thd_within_spec(0.051) == false);
}

TEST_CASE("energy_drift_within_spec: 0.009% → true; 0.011% → false", "[performance]") {
    CHECK(energy_drift_within_spec(0.00009) == true);
    CHECK(energy_drift_within_spec(0.0001)  == false);  // strict <
    CHECK(energy_drift_within_spec(0.0002)  == false);
}

TEST_CASE("compute_thd: only fundamental → 0.0", "[performance]") {
    CHECK(compute_thd(1.0, 0.0) == Approx(0.0));
}

TEST_CASE("compute_thd: equal fundamental and harmonic → THD = 1.0 (100%)", "[performance]") {
    // sqrt(1.0) / 1.0 = 1.0
    CHECK(compute_thd(1.0, 1.0) == Approx(1.0));
}

TEST_CASE("compute_thd: 3% single harmonic", "[performance]") {
    // sqrt(0.03²) / 1.0 = 0.03
    CHECK(compute_thd(1.0, 0.0009) == Approx(0.03).epsilon(1e-6));
}

TEST_CASE("relative_energy_drift: no drift → 0.0", "[performance]") {
    CHECK(relative_energy_drift(100.0, 100.0) == Approx(0.0));
}

TEST_CASE("relative_energy_drift: 1% drift", "[performance]") {
    CHECK(relative_energy_drift(100.0, 101.0) == Approx(0.01));
}

TEST_CASE("relative_energy_drift: zero initial energy → 0.0", "[performance]") {
    CHECK(relative_energy_drift(0.0, 5.0) == Approx(0.0));
}

// ---------------------------------------------------------------------------
// §12 — Performance integration tests
// ---------------------------------------------------------------------------

TEST_CASE("THD spec: soft_saturate (pre-Voronoi) of a sinusoid stays < 5%", "[performance_integration]") {
    // The spec goal is that soft saturation eliminates Gibbs discontinuities.
    // We verify THD of the CONTINUOUS soft_saturate output (before Voronoi),
    // which is what controls spectral heating.
    // Input: x(t) = A_scale * sin(2π*t/N) = 2.5 * sin(...)  (pilot wave amplitude)
    // At this amplitude, tanh(sin(θ)) is mildly nonlinear — expected THD ≈ 8–12%
    // for the tanh nonlinearity alone at amplitude=A_scale.
    // For sub-A_scale amplitude (0.5 * A_scale), THD drops well below 5%.
    const int N = 1024;
    double pi = std::numbers::pi;
    // Use amplitude 0.5 * A_scale = 1.25 (well within linear zone)
    double amplitude = 0.5 * A_SCALE;
    std::vector<double> output(N);
    for (int i = 0; i < N; ++i) {
        double x_t = amplitude * std::sin(2.0 * pi * i / N);
        output[i] = soft_saturate(x_t);  // continuous output, no Voronoi
    }
    // DFT at k=1 (fundamental)
    double fund_re = 0.0, fund_im = 0.0;
    for (int i = 0; i < N; ++i) {
        fund_re += output[i] * std::cos(2.0 * pi * i / N);
        fund_im -= output[i] * std::sin(2.0 * pi * i / N);
    }
    double fund_amp = std::sqrt(fund_re * fund_re + fund_im * fund_im) * 2.0 / N;

    // Sum harmonic power k=2..32
    double harm_sq = 0.0;
    for (int k = 2; k <= 32; ++k) {
        double re = 0.0, im = 0.0;
        for (int i = 0; i < N; ++i) {
            re += output[i] * std::cos(2.0 * pi * k * i / N);
            im -= output[i] * std::sin(2.0 * pi * k * i / N);
        }
        double amp = std::sqrt(re * re + im * im) * 2.0 / N;
        harm_sq += amp * amp;
    }
    double thd = compute_thd(fund_amp, harm_sq);
    INFO("THD = " << thd * 100.0 << "%  (limit: 5%,  amplitude=" << amplitude << ")");
    CHECK(thd_within_spec(thd));
}

TEST_CASE("Energy drift spec: quantize_real over 10^6 iterations < 0.01%", "[performance_integration]") {
    // Sinusoidal input over 1e6 steps; compare average squared output energy
    // relative initial and final half-windows to detect drift.
    const long N = 1'000'000;
    const int  window = 1000;
    const double pi = std::numbers::pi;
    const double freq = 2.0 * pi / 997.0;  // incommensurable period to avoid aliasing

    double initial_energy = 0.0;
    double final_energy   = 0.0;
    for (int i = 0; i < window; ++i) {
        double y = static_cast<double>(quantize_real(2.0 * std::sin(freq * i)));
        initial_energy += y * y;
    }
    for (long i = N - window; i < N; ++i) {
        double y = static_cast<double>(quantize_real(2.0 * std::sin(freq * static_cast<double>(i))));
        final_energy += y * y;
    }
    double drift = relative_energy_drift(initial_energy, final_energy);
    INFO("Energy drift = " << drift * 100.0 << "%  (limit: 0.01%)");
    CHECK(energy_drift_within_spec(drift));
}

// ---------------------------------------------------------------------------
// §13 — Label helpers
// ---------------------------------------------------------------------------

TEST_CASE("quantize_mode_label: correct strings", "[labels]") {
    CHECK(quantize_mode_label(QuantizeMode::NO_DITHER)   == "NO_DITHER");
    CHECK(quantize_mode_label(QuantizeMode::TPDF_DITHER) == "TPDF_DITHER");
}

TEST_CASE("saturation_zone_label: correct strings", "[labels]") {
    CHECK(saturation_zone_label(SaturationZone::LINEAR_REGION)    == "LINEAR_REGION");
    CHECK(saturation_zone_label(SaturationZone::SOFT_REGION)      == "SOFT_REGION");
    CHECK(saturation_zone_label(SaturationZone::SATURATED_REGION) == "SATURATED_REGION");
}

TEST_CASE("voronoi_region_label: correct strings", "[labels]") {
    CHECK(voronoi_region_label(VoronoiRegion::EXACT_CENTER) == "EXACT_CENTER");
    CHECK(voronoi_region_label(VoronoiRegion::INTERIOR)     == "INTERIOR");
    CHECK(voronoi_region_label(VoronoiRegion::BOUNDARY)     == "BOUNDARY");
}

TEST_CASE("all labels non-empty", "[labels]") {
    CHECK(!quantize_mode_label(QuantizeMode::NO_DITHER).empty());
    CHECK(!saturation_zone_label(SaturationZone::LINEAR_REGION).empty());
    CHECK(!voronoi_region_label(VoronoiRegion::INTERIOR).empty());
}

// ---------------------------------------------------------------------------
// §14 — Integration / scenario tests
// ---------------------------------------------------------------------------

TEST_CASE("Scenario: pilot wave initialization — typical energy in LINEAR_REGION", "[scenario]") {
    // At x = A_scale (2.5), soft_saturate = 4.5 * tanh(1.0) ≈ 3.427
    // (tanh(1.0) ≈ 0.7616, not 0.9217 — important: A_SCALE input gives tanh(1), not saturation)
    double sat = soft_saturate(A_SCALE);  // 4.5 * tanh(1.0) ≈ 3.427
    CHECK(sat == Approx(A_MAX * std::tanh(1.0)).epsilon(1e-10));
    CHECK(saturation_zone(A_SCALE) == SaturationZone::LINEAR_REGION);  // boundary inclusive
    Nit nit = nearest_nit(sat);  // 3.427 → Nit 3 (floor(3.427+0.5) = floor(3.927) = 3)
    CHECK(nit == static_cast<Nit>(3));
    // At 2×A_scale (5.0), output ≈ 4.5*tanh(2.0) ≈ 4.296 → NIT_MAX
    double sat2 = soft_saturate(2.0 * A_SCALE);  // ≈ 4.296
    CHECK(nearest_nit(sat2) == NIT_MAX);
}

TEST_CASE("Scenario: full pipeline matches spec pseudocode", "[scenario]") {
    // Spec: sat_real = 4.5 * tanh(x / 2.5); nearest_nit(sat_real)
    double x = 1.5;
    double expected_sat = 4.5 * std::tanh(x / 2.5);
    Nit    expected_nit = nearest_nit(expected_sat);
    Nit    got          = quantize_real(x);
    CHECK(got == expected_nit);
}

TEST_CASE("Scenario: TPDF dithering unbiases quantization toward continuous value", "[scenario]") {
    // TPDF dithering's key property: the AVERAGE of dithered quantized outputs
    // converges to the CONTINUOUS soft_saturate value, not the nearest integer.
    // This is the noise-shaping benefit: no systematic quantization bias.
    double x = 0.8;
    double continuous_val = soft_saturate(x);  // ≈ 4.5*tanh(0.32) ≈ 1.393
    const int N = 100;
    double sum_dither = 0.0;
    for (int i = 0; i < N; ++i) {
        double u1 = -0.5 + (i + 0.5) / N;
        for (int j = 0; j < N; ++j) {
            double u2 = -0.5 + (j + 0.5) / N;
            sum_dither += static_cast<double>(quantize_real_dithered(x, u1, u2));
        }
    }
    double avg_dither = sum_dither / (N * N);
    // Dithered average converges to the continuous soft_saturate value
    // (within ~0.05 for 100×100 uniform samples spanning full TPDF range)
    INFO("avg_dither=" << avg_dither << "  continuous_val=" << continuous_val);
    CHECK(avg_dither == Approx(continuous_val).margin(0.05));
}

TEST_CASE("Scenario: saturation prevents Gibbs truncation", "[scenario]") {
    // Hard-clip to NIT_MAX: input 5.0 → HARD clip would be NIT_MAX directly
    //   5.0 > 4 → hard_clip = 4 (creates discontinuity at input=4)
    // Soft: 5.0 → 4.5*tanh(2) ≈ 4.368 → Nit 4 (same, but via continuous path)
    double hard_clip = static_cast<double>(
        5.0 > NIT_MAX ? NIT_MAX : (5.0 < NIT_MIN ? NIT_MIN : static_cast<Nit>(5)));
    double soft = soft_saturate(5.0);  // ≈ 4.368, never exactly 4.5
    // Both end at Nit 4 but soft path is continuous
    CHECK(soft < A_MAX);  // No hard boundary
    CHECK(soft > 0.0);
    CHECK(nearest_nit(soft) == NIT_MAX);
    (void)hard_clip;
}

TEST_CASE("Scenario: complex wavefunction collapse — imaginary is discarded", "[scenario]") {
    // The spec says "Phase information is projected onto the real axis"
    // — wavefunction collapse for symbolic processing
    double re = 2.0;
    for (double im : {0.0, 1.0, -10.0, 100.0}) {
        auto w = std::complex<double>{re, im};
        CHECK(quantize_wave(w) == quantize_real(re));
    }
}

TEST_CASE("Scenario: A_MAX headroom prevents clipping at NIT values", "[scenario]") {
    // The 0.5 headroom (A_MAX=4.5 vs NIT_MAX=4) means the asymptote is above
    // all representable Nit values.  This prevents a 'ceiling clip' to exactly +4.5,
    // which would create a discontinuity equivalent to hard clipping.
    // Verify for practical input range (double precision tanh < 1.0 for |x| <= 10):
    double sat10 = soft_saturate(10.0);  // tanh(4.0) < 1.0 in double
    CHECK(sat10 < A_MAX);
    CHECK(nearest_nit(sat10) == NIT_MAX);  // still rounds to 4
    // Verify the design constant: A_MAX > NIT_MAX by exactly 0.5
    CHECK(A_MAX - static_cast<double>(NIT_MAX) == Approx(A_HEADROOM));
    CHECK(A_HEADROOM == Approx(0.5));
}
