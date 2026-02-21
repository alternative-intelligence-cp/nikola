/**
 * @file phase8_foundations_test.cpp
 * @brief Phase 8 — Foundation layer tests: Nit arithmetic and MetricTensorCache.
 *
 * Covers:
 *   - Nit scalar gates (sum_gate, product_gate)
 *   - Nit saturation boundaries
 *   - quantize_wave / nit_to_float round-trip
 *   - Nyte radix-9 encode / decode / get
 *   - add_nit_batch scalar and (if available) AVX-512 paths
 *   - mul_nit_batch scalar correctness
 *   - has_avx512() runtime detection (smoke)
 *   - MetricTensorCache: flat metric, update, apply_inverse, log_det
 *   - MetricTensorCache: Cholesky on known 9×9 identity and diagonal matrices
 *   - MetricTensorCache: dirty-flag (cache hit vs recompute)
 *   - MetricTensorCache: non-PD matrix rejection
 *   - cholesky_9x9 and substitution round-trip
 *   - Integration: g⁻¹ g v ≈ v for random positive-definite metric
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/foundation/nit.hpp>
#include <nikola/physics/metric_tensor.hpp>

#include <array>
#include <vector>
#include <complex>
#include <cmath>
#include <numeric>
#include <random>

using namespace nikola::foundation;
using namespace nikola::physics;
using Catch::Approx;

// ============================================================================
// Helpers
// ============================================================================

static std::array<double, METRIC_LOWER_SIZE> make_diagonal_metric(
    const std::array<double, METRIC_DIM>& diag)
{
    std::array<double, METRIC_LOWER_SIZE> g{};
    for (int i = 0; i < METRIC_DIM; ++i)
        g[metric_lower_idx(i, i)] = diag[i];
    return g;
}

/// Build a random symmetric positive-definite 9×9 metric via A^T A + εI.
static std::array<double, METRIC_LOWER_SIZE> make_random_pd_metric(
    unsigned seed = 42, double epsilon = 1.0)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.3, 0.3);

    // Flatten lower triangle of A (9×9)
    double A[METRIC_DIM][METRIC_DIM]{};
    for (int i = 0; i < METRIC_DIM; ++i)
        for (int j = 0; j <= i; ++j)
            A[i][j] = dist(rng);

    // g = A * A^T + epsilon * I  (always PD)
    std::array<double, METRIC_LOWER_SIZE> g{};
    for (int i = 0; i < METRIC_DIM; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = (i == j) ? epsilon : 0.0;
            for (int k = 0; k < METRIC_DIM; ++k)
                s += A[i][k] * A[j][k];
            g[metric_lower_idx(i, j)] = s;
        }
    }
    return g;
}

// ============================================================================
// Nit: scalar gates
// ============================================================================

TEST_CASE("Nit sum_gate: basic addition", "[nit][scalar]") {
    CHECK(sum_gate(0, 0) == 0);
    CHECK(sum_gate(2, 1) == 3);
    CHECK(sum_gate(-2, -1) == -3);
    CHECK(sum_gate(1, -1) == 0);
    CHECK(sum_gate(3, 2) == 4);    // 5 → saturates to 4
    CHECK(sum_gate(-3, -2) == -4); // -5 → saturates to -4
}

TEST_CASE("Nit sum_gate: saturation boundaries", "[nit][scalar]") {
    CHECK(sum_gate(4, 1) == NIT_MAX);   // 5 → 4
    CHECK(sum_gate(-4, -1) == NIT_MIN); // -5 → -4
    CHECK(sum_gate(4, 4) == NIT_MAX);
    CHECK(sum_gate(-4, -4) == NIT_MIN);
    CHECK(sum_gate(4, 0) == 4);
    CHECK(sum_gate(0, -4) == -4);
}

TEST_CASE("Nit sum_gate: identity and inverse", "[nit][scalar]") {
    for (Nit v = NIT_MIN; v <= NIT_MAX; ++v) {
        CHECK(sum_gate(v, 0) == v);        // additive identity
        CHECK(sum_gate(v, static_cast<Nit>(-v)) == 0); // additive inverse (within range)
    }
}

TEST_CASE("Nit product_gate: zero product", "[nit][scalar]") {
    for (Nit v = NIT_MIN; v <= NIT_MAX; ++v) {
        CHECK(product_gate(v, 0) == 0);
        CHECK(product_gate(0, v) == 0);
    }
}

TEST_CASE("Nit product_gate: identity element is ±1", "[nit][scalar]") {
    for (Nit v = NIT_MIN; v <= NIT_MAX; ++v) {
        // |product_gate(v, 1)| should equal max(|v|-step, 0) or v directly
        // The table is: product_gate(v, 1) = v for |v| <= 1
        if (v == 0) CHECK(product_gate(v, 1) == 0);
        if (v == 1) CHECK(product_gate(v, 1) == 1);
        if (v == -1) CHECK(product_gate(v, -1) == 1);
    }
}

TEST_CASE("Nit product_gate: commutativity", "[nit][scalar]") {
    for (Nit a = NIT_MIN; a <= NIT_MAX; ++a)
        for (Nit b = NIT_MIN; b <= NIT_MAX; ++b)
            CHECK(product_gate(a, b) == product_gate(b, a));
}

TEST_CASE("Nit product_gate: specific values from spec", "[nit][scalar]") {
    // Spec values from the 9×9 table in the engineering guide
    CHECK(product_gate(-4, -4) == 4);
    CHECK(product_gate( 4,  4) == 4);
    CHECK(product_gate(-4,  4) == -4);
    CHECK(product_gate( 4, -4) == -4);
    CHECK(product_gate( 3,  3) == 4);   // saturated wave mixing
    CHECK(product_gate(-3, -3) == 4);
}

// ============================================================================
// Nit: quantisation
// ============================================================================

TEST_CASE("quantize_real: round-trip approx", "[nit][quant]") {
    for (Nit n = NIT_MIN; n <= NIT_MAX; ++n) {
        const float f = nit_to_float(n);
        CHECK(quantize_real(f) == n);
    }
}

TEST_CASE("quantize_real: boundary clamp", "[nit][quant]") {
    CHECK(quantize_real( 2.0f) == NIT_MAX);
    CHECK(quantize_real(-2.0f) == NIT_MIN);
    CHECK(quantize_real( 1.0f) == NIT_MAX);
    CHECK(quantize_real(-1.0f) == NIT_MIN);
}

TEST_CASE("quantize_wave: zero amplitude maps to zero", "[nit][quant]") {
    CHECK(quantize_wave({0.0, 0.0}) == NIT_ZERO);
}

TEST_CASE("quantize_wave: unit real amplitude", "[nit][quant]") {
    const Nit q = quantize_wave({1.0, 0.0});
    CHECK(q == NIT_MAX);   // real(1)/mag(1) = 1.0 → quantize(1.0) = 4
}

TEST_CASE("quantize_wave: negative real amplitude", "[nit][quant]") {
    const Nit q = quantize_wave({-1.0, 0.0});
    CHECK(q == NIT_MIN);
}

TEST_CASE("quantize_wave: imaginary component ignored after normalisation",
          "[nit][quant]")
{
    // psi = (0.0, 1.0) → real after normalise = 0.0 → quantize = 0
    const Nit q = quantize_wave({0.0, 1.0});
    CHECK(q == 0);
}

// ============================================================================
// Nyte: radix-9 packing
// ============================================================================

TEST_CASE("Nyte: encode then decode round-trip", "[nyte]") {
    const std::array<Nit, 5> nits = {-4, -2, 0, 2, 4};
    const Nyte ny = Nyte::encode(nits);
    CHECK(ny.decode() == nits);
}

TEST_CASE("Nyte: all-zero", "[nyte]") {
    const std::array<Nit, 5> nits = {0, 0, 0, 0, 0};
    const Nyte ny = Nyte::encode(nits);
    CHECK(ny.packed == Nyte::POWERS[0] * 4 + Nyte::POWERS[1] * 4 +
                       Nyte::POWERS[2] * 4 + Nyte::POWERS[3] * 4 +
                       Nyte::POWERS[4] * 4);  // each digit = (0+4) = 4
    CHECK(ny.decode() == nits);
}

TEST_CASE("Nyte: all min", "[nyte]") {
    const std::array<Nit, 5> nits = {-4, -4, -4, -4, -4};
    const Nyte ny = Nyte::encode(nits);
    CHECK(ny.packed == 0);  // each digit = (−4+4) = 0
    CHECK(ny.decode() == nits);
}

TEST_CASE("Nyte: all max", "[nyte]") {
    const std::array<Nit, 5> nits = {4, 4, 4, 4, 4};
    const Nyte ny = Nyte::encode(nits);
    CHECK(ny.packed == 9*9*9*9*8 + 9*9*9*8 + 9*9*8 + 9*8 + 8); // 59048
    CHECK(ny.decode() == nits);
}

TEST_CASE("Nyte: individual get() matches decode()", "[nyte]") {
    const std::array<Nit, 5> nits = {-3, -1, 0, 2, 3};
    const Nyte ny = Nyte::encode(nits);
    for (int i = 0; i < 5; ++i)
        CHECK(ny.get(i) == nits[i]);
}

TEST_CASE("Nyte: packed fits in uint16_t", "[nyte]") {
    const std::array<Nit, 5> nits = {4, 4, 4, 4, 4};
    const Nyte ny = Nyte::encode(nits);
    CHECK(ny.packed <= 65535u);
}

// ============================================================================
// Nit: batch operations
// ============================================================================

TEST_CASE("add_nit_batch: correctness vs scalar", "[nit][batch]") {
    constexpr size_t N = 200;
    std::vector<Nit> a(N), b(N), result(N), expected(N);
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<Nit>((i % 9) - 4);
        b[i] = static_cast<Nit>(((i + 3) % 9) - 4);
        expected[i] = sum_gate(a[i], b[i]);
    }
    add_nit_batch(a.data(), b.data(), result.data(), N);
    CHECK(result == expected);
}

TEST_CASE("add_nit_batch: alignment not required (count not multiple of 64)",
          "[nit][batch]")
{
    constexpr size_t N = 7;  // less than 64
    std::vector<Nit> a = {-4, -3, 0, 1, 2, 3, 4};
    std::vector<Nit> b = { 4,  3, 0, -1, -2, -3, -4};
    std::vector<Nit> result(N);
    add_nit_batch(a.data(), b.data(), result.data(), N);
    for (size_t i = 0; i < N; ++i)
        CHECK(result[i] == sum_gate(a[i], b[i]));
}

TEST_CASE("mul_nit_batch: correctness vs scalar", "[nit][batch]") {
    constexpr size_t N = 130;
    std::vector<Nit> a(N), b(N), result(N), expected(N);
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<Nit>((i % 9) - 4);
        b[i] = static_cast<Nit>(((i + 5) % 9) - 4);
        expected[i] = product_gate(a[i], b[i]);
    }
    mul_nit_batch(a.data(), b.data(), result.data(), N);
    CHECK(result == expected);
}

TEST_CASE("has_avx512: returns bool without crashing", "[nit][avx512]") {
    bool r = has_avx512();
    (void)r;  // just verifying no crash
    SUCCEED("has_avx512() executed without error");
}

// ============================================================================
// metric_lower_idx helper
// ============================================================================

TEST_CASE("metric_lower_idx: diagonal positions", "[metric][idx]") {
    // Diagonal element (k,k) should be at index k(k+1)/2 + k
    for (int k = 0; k < METRIC_DIM; ++k)
        CHECK(metric_lower_idx(k, k) == k * (k + 1) / 2 + k);
}

TEST_CASE("metric_lower_idx: total count is 45", "[metric][idx]") {
    // The highest index + 1 = 45
    CHECK(metric_lower_idx(8, 8) + 1 == METRIC_LOWER_SIZE);
    CHECK(METRIC_LOWER_SIZE == 45);
}

// ============================================================================
// cholesky_9x9: identity and diagonal
// ============================================================================

TEST_CASE("cholesky_9x9: identity matrix", "[metric][cholesky]") {
    std::array<double, METRIC_LOWER_SIZE> g{};
    for (int i = 0; i < METRIC_DIM; ++i)
        g[metric_lower_idx(i, i)] = 1.0;

    std::array<double, METRIC_LOWER_SIZE> L{};
    REQUIRE(cholesky_9x9(g, L));

    // Cholesky of I is I
    for (int i = 0; i < METRIC_DIM; ++i)
        CHECK(L[metric_lower_idx(i, i)] == Approx(1.0));
    // Off-diagonal = 0
    for (int i = 1; i < METRIC_DIM; ++i)
        for (int j = 0; j < i; ++j)
            CHECK(L[metric_lower_idx(i, j)] == Approx(0.0).margin(1e-14));
}

TEST_CASE("cholesky_9x9: diagonal metric with known sqrt", "[metric][cholesky]") {
    const std::array<double, METRIC_DIM> diag = {4, 9, 16, 25, 36, 49, 64, 81, 100};
    const auto g = make_diagonal_metric(diag);

    std::array<double, METRIC_LOWER_SIZE> L{};
    REQUIRE(cholesky_9x9(g, L));

    // L[i][i] = sqrt(diag[i])
    for (int i = 0; i < METRIC_DIM; ++i)
        CHECK(L[metric_lower_idx(i, i)] == Approx(std::sqrt(diag[i])).epsilon(1e-12));
}

TEST_CASE("cholesky_9x9: reconstruction L * L^T = g", "[metric][cholesky]") {
    const auto g = make_random_pd_metric(7);
    std::array<double, METRIC_LOWER_SIZE> L{};
    REQUIRE(cholesky_9x9(g, L));

    // Reconstruct g from L: (L L^T)[i][j] = Σ_k L[i][k] L[j][k]
    for (int i = 0; i < METRIC_DIM; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = 0.0;
            for (int k = 0; k <= std::min(i, j); ++k)
                s += L[metric_lower_idx(i, k)] * L[metric_lower_idx(j, k)];
            CHECK(s == Approx(g[metric_lower_idx(i, j)]).epsilon(1e-10));
        }
    }
}

TEST_CASE("cholesky_9x9: non-positive-definite returns false", "[metric][cholesky]") {
    // All zeros is not PD (L[0][0] would require sqrt(0))
    std::array<double, METRIC_LOWER_SIZE> g{};
    std::array<double, METRIC_LOWER_SIZE> L{};
    CHECK_FALSE(cholesky_9x9(g, L));
}

// ============================================================================
// MetricTensorCache
// ============================================================================

TEST_CASE("MetricTensorCache: default constructed is invalid", "[metric][cache]") {
    MetricTensorCache cache;
    CHECK_FALSE(cache.is_valid());
}

TEST_CASE("MetricTensorCache::flat: identity metric", "[metric][cache]") {
    auto cache = MetricTensorCache::flat();
    REQUIRE(cache.is_valid());

    // g⁻¹ v = v for identity metric
    const std::array<double, METRIC_DIM> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const auto result = cache.apply_inverse(v);
    for (int i = 0; i < METRIC_DIM; ++i)
        CHECK(result[i] == Approx(v[i]).epsilon(1e-12));
}

TEST_CASE("MetricTensorCache: update_if_changed initialises cache", "[metric][cache]") {
    MetricTensorCache cache;
    const auto g = make_random_pd_metric(42);
    bool recomputed = cache.update_if_changed(g);
    CHECK(recomputed);
    CHECK(cache.is_valid());
}

TEST_CASE("MetricTensorCache: second update with same metric hits cache",
          "[metric][cache]")
{
    MetricTensorCache cache;
    const auto g = make_random_pd_metric(42);
    cache.update_if_changed(g);
    bool recomputed = cache.update_if_changed(g);  // same metric
    CHECK_FALSE(recomputed);  // cache hit
}

TEST_CASE("MetricTensorCache: update_if_changed recomputes on change",
          "[metric][cache]")
{
    MetricTensorCache cache;
    auto g1 = make_random_pd_metric(42);
    auto g2 = make_random_pd_metric(99);   // different metric
    cache.update_if_changed(g1);
    bool recomputed = cache.update_if_changed(g2);
    CHECK(recomputed);
}

TEST_CASE("MetricTensorCache: apply_inverse throws before update", "[metric][cache]") {
    MetricTensorCache cache;
    const std::array<double, METRIC_DIM> v{};
    CHECK_THROWS_AS(cache.apply_inverse(v), std::logic_error);
}

TEST_CASE("MetricTensorCache: g⁻¹ g v ≈ v (round-trip)", "[metric][cache]") {
    MetricTensorCache cache;
    const auto g = make_random_pd_metric(17);
    cache.force_update(g);

    const std::array<double, METRIC_DIM> v = {1.0, -0.5, 2.3, 0.1, -1.2,
                                               3.0,  0.7, -0.9, 1.1};
    // x = g⁻¹ v;  then g x should equal v
    const auto x = cache.apply_inverse(v);
    const auto gx = cache.apply(x);

    for (int i = 0; i < METRIC_DIM; ++i)
        CHECK(gx[i] == Approx(v[i]).epsilon(1e-9));
}

TEST_CASE("MetricTensorCache: non-PD metric throws on force_update", "[metric][cache]") {
    MetricTensorCache cache;
    std::array<double, METRIC_LOWER_SIZE> bad_g{};  // all zeros — not PD
    CHECK_THROWS_AS(cache.force_update(bad_g), std::invalid_argument);
    CHECK_FALSE(cache.is_valid());
}

TEST_CASE("MetricTensorCache: log_det identity metric = 0", "[metric][cache]") {
    auto cache = MetricTensorCache::flat();
    // det(I) = 1 → log(1) = 0
    CHECK(cache.log_det() == Approx(0.0).margin(1e-12));
}

TEST_CASE("MetricTensorCache: log_det diagonal metric = Σ log(diag)", "[metric][cache]") {
    const std::array<double, METRIC_DIM> diag = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    MetricTensorCache cache;
    cache.force_update(make_diagonal_metric(diag));

    double expected = 0.0;
    for (auto d : diag) expected += std::log(d);

    CHECK(cache.log_det() == Approx(expected).epsilon(1e-10));
}

TEST_CASE("MetricTensorCache: invalidate resets valid flag", "[metric][cache]") {
    auto cache = MetricTensorCache::flat();
    REQUIRE(cache.is_valid());
    cache.invalidate();
    CHECK_FALSE(cache.is_valid());
}

TEST_CASE("MetricTensorCache: invalidate then update_if_changed recomputes",
          "[metric][cache]")
{
    MetricTensorCache cache;
    const auto g = make_random_pd_metric(55);
    cache.force_update(g);
    cache.invalidate();
    bool recomputed = cache.update_if_changed(g);  // same g but cache was invalid
    CHECK(recomputed);
    CHECK(cache.is_valid());
}

// ============================================================================
// Integration: Nit pipeline — quantise wave → nit → float → check error
// ============================================================================

TEST_CASE("Nit pipeline: quantise → float quantisation error <= 0.125",
          "[nit][integration]")
{
    // Worst-case quantisation error: max distance from float to nearest 1/4 step
    // = 1/(2×4) = 0.125
    for (int q = -40; q <= 40; ++q) {
        const float x = static_cast<float>(q) / 40.f;  // in [-1, +1]
        const Nit n   = quantize_real(x);
        const float r = nit_to_float(n);
        CHECK(std::abs(r - x) <= 0.1251f);
    }
}

TEST_CASE("Nit + Nyte pipeline: encode complex wave series", "[nit][integration]") {
    // Build 5 complex wave amplitudes, quantise each to a Nit, pack into Nyte
    const std::array<std::complex<double>, 5> waves = {{
        {1.0,  0.0},
        {0.0,  1.0},
        {-1.0, 0.0},
        {0.7, 0.7},
        {0.0, 0.0}
    }};
    std::array<Nit, 5> nits{};
    for (int i = 0; i < 5; ++i)
        nits[i] = quantize_wave(waves[i]);

    const Nyte ny = Nyte::encode(nits);
    CHECK(ny.decode() == nits);  // lossless round-trip through Nyte
}
