/**
 * @file v033_covariant_transport_test.cpp
 * @brief Tests for GAP-M2: Covariant State Transport.
 *
 * Validates:
 *   - project_to_frame / embed_from_frame round-trip
 *   - transport_vector preserves g-norm
 *   - CovariantTransporter stateful API
 *   - Transport with flat (identity) metric is identity operation
 *   - Transport with non-trivial metric preserves inner products
 *   - Error handling for invalid states
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/physics/covariant_transport.hpp>
#include <nikola/physics/metric_tensor.hpp>

#include <cmath>
#include <array>
#include <random>

using namespace nikola::physics;
using Catch::Matchers::WithinAbs;

// ============================================================================
// Helpers
// ============================================================================

/// Build a flat (identity) metric in packed lower-triangle format.
static std::array<double, METRIC_LOWER_SIZE> flat_metric() {
    std::array<double, METRIC_LOWER_SIZE> g{};
    for (int i = 0; i < METRIC_DIM; ++i) {
        g[metric_lower_idx(i, i)] = 1.0;
    }
    return g;
}

/// Build a diagonal metric with specified diagonal values.
static std::array<double, METRIC_LOWER_SIZE>
diagonal_metric(const std::array<double, METRIC_DIM>& diag) {
    std::array<double, METRIC_LOWER_SIZE> g{};
    for (int i = 0; i < METRIC_DIM; ++i) {
        g[metric_lower_idx(i, i)] = diag[i];
    }
    return g;
}

/// Build a near-identity metric with small perturbations (still SPD).
static std::array<double, METRIC_LOWER_SIZE>
perturbed_metric(uint32_t seed, double epsilon = 0.01) {
    auto g = flat_metric();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-epsilon, epsilon);

    // Add symmetric perturbation, keep diagonal dominant for SPD
    for (int i = 0; i < METRIC_DIM; ++i) {
        for (int j = 0; j <= i; ++j) {
            if (i == j) {
                g[metric_lower_idx(i, j)] = 1.0 + std::abs(dist(rng));
            } else {
                double val = dist(rng) * 0.1;  // off-diagonal much smaller
                g[metric_lower_idx(i, j)] = val;
            }
        }
    }
    return g;
}

/// Build a test vector.
static std::array<double, METRIC_DIM> test_vector(uint32_t seed) {
    std::array<double, METRIC_DIM> h{};
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& v : h) v = dist(rng);
    return h;
}

// ============================================================================
// project_to_frame + embed_from_frame
// ============================================================================

TEST_CASE("§1 Flat metric: project + embed is identity", "[v033][transport]") {
    auto g = flat_metric();
    std::array<double, METRIC_LOWER_SIZE> L;
    REQUIRE(cholesky_9x9(g, L));

    auto h = test_vector(42);
    auto h_hat = project_to_frame(L, h);
    auto h_back = embed_from_frame(L, h_hat);

    for (int i = 0; i < METRIC_DIM; ++i) {
        REQUIRE_THAT(h_back[i], WithinAbs(h[i], 1e-12));
    }
}

TEST_CASE("§2 Diagonal metric: project + embed round-trip", "[v033][transport]") {
    std::array<double, METRIC_DIM> diag = {2.0, 3.0, 1.5, 4.0, 0.5, 1.0, 2.5, 3.5, 1.2};
    auto g = diagonal_metric(diag);
    std::array<double, METRIC_LOWER_SIZE> L;
    REQUIRE(cholesky_9x9(g, L));

    auto h = test_vector(123);
    auto h_hat = project_to_frame(L, h);
    auto h_back = embed_from_frame(L, h_hat);

    for (int i = 0; i < METRIC_DIM; ++i) {
        REQUIRE_THAT(h_back[i], WithinAbs(h[i], 1e-10));
    }
}

// ============================================================================
// transport_vector norm preservation
// ============================================================================

TEST_CASE("§3 Transport preserves g-norm: flat → flat", "[v033][transport]") {
    auto g = flat_metric();
    std::array<double, METRIC_LOWER_SIZE> L;
    REQUIRE(cholesky_9x9(g, L));

    auto h = test_vector(77);
    auto h_new = transport_vector(L, L, h);

    // With same L, transport should be identity
    for (int i = 0; i < METRIC_DIM; ++i) {
        REQUIRE_THAT(h_new[i], WithinAbs(h[i], 1e-12));
    }
}

TEST_CASE("§4 Transport preserves g-norm: diagonal → diagonal", "[v033][transport]") {
    std::array<double, METRIC_DIM> d1 = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    std::array<double, METRIC_DIM> d2 = {3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0};
    auto g1 = diagonal_metric(d1);
    auto g2 = diagonal_metric(d2);
    std::array<double, METRIC_LOWER_SIZE> L1, L2;
    REQUIRE(cholesky_9x9(g1, L1));
    REQUIRE(cholesky_9x9(g2, L2));

    auto h = test_vector(99);
    auto h_new = transport_vector(L1, L2, h);

    double norm_old = metric_norm_sq(g1, h);
    double norm_new = metric_norm_sq(g2, h_new);
    REQUIRE_THAT(norm_new, WithinAbs(norm_old, 1e-10));
}

TEST_CASE("§5 Transport preserves g-norm: perturbed → perturbed", "[v033][transport]") {
    auto g1 = perturbed_metric(42);
    auto g2 = perturbed_metric(43);
    std::array<double, METRIC_LOWER_SIZE> L1, L2;
    REQUIRE(cholesky_9x9(g1, L1));
    REQUIRE(cholesky_9x9(g2, L2));

    auto h = test_vector(55);
    auto h_new = transport_vector(L1, L2, h);

    double norm_old = metric_norm_sq(g1, h);
    double norm_new = metric_norm_sq(g2, h_new);
    REQUIRE_THAT(norm_new, WithinAbs(norm_old, 1e-10));
}

TEST_CASE("§6 Transport of zero vector is zero", "[v033][transport]") {
    auto g1 = perturbed_metric(10);
    auto g2 = perturbed_metric(20);
    std::array<double, METRIC_LOWER_SIZE> L1, L2;
    REQUIRE(cholesky_9x9(g1, L1));
    REQUIRE(cholesky_9x9(g2, L2));

    std::array<double, METRIC_DIM> zero{};
    auto result = transport_vector(L1, L2, zero);

    for (int i = 0; i < METRIC_DIM; ++i) {
        REQUIRE_THAT(result[i], WithinAbs(0.0, 1e-15));
    }
}

// ============================================================================
// metric_norm_sq
// ============================================================================

TEST_CASE("§7 Metric norm with flat metric = Euclidean norm", "[v033][transport]") {
    auto g = flat_metric();
    auto h = test_vector(33);

    double expected = 0.0;
    for (double v : h) expected += v * v;

    REQUIRE_THAT(metric_norm_sq(g, h), WithinAbs(expected, 1e-12));
}

TEST_CASE("§8 Metric norm with diagonal metric", "[v033][transport]") {
    std::array<double, METRIC_DIM> diag = {2.0, 3.0, 1.0, 4.0, 2.0, 1.0, 3.0, 2.0, 5.0};
    auto g = diagonal_metric(diag);
    std::array<double, METRIC_DIM> h = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Should be diag[0] * h[0]^2 = 2.0
    REQUIRE_THAT(metric_norm_sq(g, h), WithinAbs(2.0, 1e-12));
}

// ============================================================================
// CovariantTransporter
// ============================================================================

TEST_CASE("§9 Transporter: not ready before begin_transport", "[v033][transporter]") {
    CovariantTransporter ct;
    REQUIRE_FALSE(ct.is_ready());
}

TEST_CASE("§10 Transporter: ready after begin_transport", "[v033][transporter]") {
    auto g1 = perturbed_metric(1);
    auto g2 = perturbed_metric(2);
    MetricTensorCache c1(g1), c2(g2);

    CovariantTransporter ct;
    ct.begin_transport(c1, c2);
    REQUIRE(ct.is_ready());
}

TEST_CASE("§11 Transporter: transport without begin throws", "[v033][transporter]") {
    CovariantTransporter ct;
    auto h = test_vector(1);
    REQUIRE_THROWS_AS(ct.transport(h), std::logic_error);
}

TEST_CASE("§12 Transporter: invalid old cache throws", "[v033][transporter]") {
    MetricTensorCache c1;  // not initialized
    auto g2 = flat_metric();
    MetricTensorCache c2(g2);

    CovariantTransporter ct;
    REQUIRE_THROWS_AS(ct.begin_transport(c1, c2), std::logic_error);
}

TEST_CASE("§13 Transporter: invalid new cache throws", "[v033][transporter]") {
    auto g1 = flat_metric();
    MetricTensorCache c1(g1);
    MetricTensorCache c2;  // not initialized

    CovariantTransporter ct;
    REQUIRE_THROWS_AS(ct.begin_transport(c1, c2), std::logic_error);
}

TEST_CASE("§14 Transporter: transport preserves norm", "[v033][transporter]") {
    auto g1 = perturbed_metric(100);
    auto g2 = perturbed_metric(200);
    MetricTensorCache c1(g1), c2(g2);

    CovariantTransporter ct;
    ct.begin_transport(c1, c2);

    auto h = test_vector(42);
    auto h_new = ct.transport(h);

    REQUIRE(CovariantTransporter::verify_norm(g1, g2, h, h_new));
}

TEST_CASE("§15 Transporter: batch transport multiple vectors", "[v033][transporter]") {
    auto g1 = perturbed_metric(50);
    auto g2 = perturbed_metric(60);
    MetricTensorCache c1(g1), c2(g2);

    CovariantTransporter ct;
    ct.begin_transport(c1, c2);

    // Transport 10 different vectors
    for (uint32_t seed = 0; seed < 10; ++seed) {
        auto h = test_vector(seed);
        auto h_new = ct.transport(h);
        REQUIRE(CovariantTransporter::verify_norm(g1, g2, h, h_new));
    }
}

TEST_CASE("§16 Transporter: reset and re-begin", "[v033][transporter]") {
    auto g1 = flat_metric();
    auto g2 = perturbed_metric(70);
    auto g3 = perturbed_metric(80);
    MetricTensorCache c1(g1), c2(g2), c3(g3);

    CovariantTransporter ct;
    ct.begin_transport(c1, c2);
    REQUIRE(ct.is_ready());

    ct.reset();
    REQUIRE_FALSE(ct.is_ready());

    ct.begin_transport(c2, c3);
    REQUIRE(ct.is_ready());

    auto h = test_vector(42);
    auto h_new = ct.transport(h);
    REQUIRE(CovariantTransporter::verify_norm(g2, g3, h, h_new));
}

TEST_CASE("§17 verify_norm detects corrupt transport", "[v033][transporter]") {
    auto g1 = perturbed_metric(90);
    auto g2 = perturbed_metric(91);

    auto h = test_vector(42);
    // Deliberately corrupt: just scale the vector
    std::array<double, METRIC_DIM> h_bad = h;
    for (auto& v : h_bad) v *= 2.0;

    REQUIRE_FALSE(CovariantTransporter::verify_norm(g1, g2, h, h_bad));
}

// ============================================================================
// Transport composition
// ============================================================================

TEST_CASE("§18 Sequential transport: g1 → g2 → g3 preserves norm from g1 to g3",
          "[v033][transport]") {
    auto g1 = perturbed_metric(10);
    auto g2 = perturbed_metric(20);
    auto g3 = perturbed_metric(30);
    std::array<double, METRIC_LOWER_SIZE> L1, L2, L3;
    REQUIRE(cholesky_9x9(g1, L1));
    REQUIRE(cholesky_9x9(g2, L2));
    REQUIRE(cholesky_9x9(g3, L3));

    auto h = test_vector(42);
    auto h2 = transport_vector(L1, L2, h);
    auto h3 = transport_vector(L2, L3, h2);

    // Direct transport g1 → g3
    auto h_direct = transport_vector(L1, L3, h);

    // Both should give the same result
    for (int i = 0; i < METRIC_DIM; ++i) {
        REQUIRE_THAT(h3[i], WithinAbs(h_direct[i], 1e-10));
    }
}
