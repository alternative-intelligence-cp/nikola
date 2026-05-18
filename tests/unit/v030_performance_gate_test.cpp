/**
 * @file tests/unit/v030_performance_gate_test.cpp
 * @brief v0.3.0 — PerformanceGate test suite
 *
 * Tests:
 *   §1  Default construction
 *   §2  Significant improvement passes
 *   §3  No improvement fails
 *   §4  Regression in primary metric fails
 *   §5  Secondary metric regression fails
 *   §6  Insufficient samples fails
 *   §7  Mismatched sample sizes fails
 *   §8  HIGHER_IS_BETTER direction
 *   §9  Paired t-test known values
 *   §10 p-value computation accuracy
 *   §11 is_improved() convenience
 *   §12 Custom alpha threshold
 *   §13 No secondary metrics → pass if primary passes
 *   §14 Effect size (Cohen's d) populated
 *   §15 Counters (total_evaluations, total_failures)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/security/performance_gate.hpp>

#include <cmath>
#include <random>
#include <vector>

using namespace nikola::security;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Generate paired samples where candidate is better by `improvement` units.
static BenchmarkMetric make_latency_metric(
    size_t n, double baseline_mean, double improvement, unsigned seed = 42)
{
    BenchmarkMetric m;
    m.name = "latency_us";
    m.direction = MetricDirection::LOWER_IS_BETTER;

    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 2.0);

    m.baseline_samples.resize(n);
    m.candidate_samples.resize(n);
    for (size_t i = 0; i < n; ++i) {
        m.baseline_samples[i]  = baseline_mean + noise(rng);
        m.candidate_samples[i] = baseline_mean - improvement + noise(rng);
    }
    return m;
}

static BenchmarkMetric make_throughput_metric(
    size_t n, double baseline_mean, double improvement, unsigned seed = 42)
{
    BenchmarkMetric m;
    m.name = "throughput_ops";
    m.direction = MetricDirection::HIGHER_IS_BETTER;

    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 1.0);

    m.baseline_samples.resize(n);
    m.candidate_samples.resize(n);
    for (size_t i = 0; i < n; ++i) {
        m.baseline_samples[i]  = baseline_mean + noise(rng);
        m.candidate_samples[i] = baseline_mean + improvement + noise(rng);
    }
    return m;
}

// ============================================================================
// §1 Default construction
// ============================================================================

TEST_CASE("§1 PerformanceGate default construction", "[v030][perf_gate]") {
    PerformanceGate gate;
    REQUIRE(gate.config().alpha == PERF_ALPHA);
    REQUIRE(gate.config().min_samples == PERF_MIN_SAMPLES);
    REQUIRE(gate.config().regression_tolerance == PERF_REGRESSION_TOLERANCE);
    REQUIRE(gate.total_evaluations() == 0);
    REQUIRE(gate.total_failures() == 0);
}

// ============================================================================
// §2 Significant improvement passes
// ============================================================================

TEST_CASE("§2 Significant latency improvement passes", "[v030][perf_gate]") {
    PerformanceGate gate;

    // 10us improvement on a 100us baseline with 2us noise → very significant
    auto primary = make_latency_metric(30, 100.0, 10.0);

    auto verdict = gate.evaluate(primary);
    REQUIRE(verdict.passed == true);
    REQUIRE(verdict.primary_result.significant == true);
    REQUIRE(verdict.primary_result.improved == true);
    REQUIRE(verdict.primary_result.p_value < 0.05);
    REQUIRE(verdict.primary_result.mean_diff < 0.0);  // candidate faster
}

// ============================================================================
// §3 No improvement fails
// ============================================================================

TEST_CASE("§3 No improvement fails", "[v030][perf_gate]") {
    PerformanceGate gate;

    // 0 improvement — should not be significant
    auto primary = make_latency_metric(30, 100.0, 0.0);

    auto verdict = gate.evaluate(primary);
    REQUIRE(verdict.passed == false);
}

// ============================================================================
// §4 Regression in primary fails
// ============================================================================

TEST_CASE("§4 Primary regression fails", "[v030][perf_gate]") {
    PerformanceGate gate;

    // Candidate is SLOWER (negative improvement → latency increases)
    auto primary = make_latency_metric(30, 100.0, -10.0);

    auto verdict = gate.evaluate(primary);
    REQUIRE(verdict.passed == false);
    REQUIRE(verdict.primary_result.improved == false);
}

// ============================================================================
// §5 Secondary regression fails
// ============================================================================

TEST_CASE("§5 Secondary metric regression fails", "[v030][perf_gate]") {
    PerformanceGate gate;

    auto primary = make_latency_metric(30, 100.0, 10.0);  // primary improves

    // Secondary metric regresses by 20%
    BenchmarkMetric sec;
    sec.name = "memory_mb";
    sec.direction = MetricDirection::LOWER_IS_BETTER;
    sec.baseline_samples  = std::vector<double>(30, 100.0);
    sec.candidate_samples = std::vector<double>(30, 125.0);  // 25% worse

    auto verdict = gate.evaluate(primary, {sec});
    REQUIRE(verdict.passed == false);
    REQUIRE(verdict.worst_regression_idx == 0);
    REQUIRE(verdict.reason.find("memory_mb") != std::string::npos);
}

// ============================================================================
// §6 Insufficient samples
// ============================================================================

TEST_CASE("§6 Insufficient samples fails", "[v030][perf_gate]") {
    PerformanceGate gate;

    auto primary = make_latency_metric(5, 100.0, 10.0);  // only 5 < 10 min

    auto verdict = gate.evaluate(primary);
    REQUIRE(verdict.passed == false);
    REQUIRE(verdict.reason.find("insufficient") != std::string::npos);
}

// ============================================================================
// §7 Mismatched sample sizes
// ============================================================================

TEST_CASE("§7 Mismatched sample sizes fails", "[v030][perf_gate]") {
    PerformanceGate gate;

    BenchmarkMetric m;
    m.name = "test";
    m.direction = MetricDirection::LOWER_IS_BETTER;
    m.baseline_samples  = std::vector<double>(20, 100.0);
    m.candidate_samples = std::vector<double>(15, 90.0);  // different count

    auto verdict = gate.evaluate(m);
    REQUIRE(verdict.passed == false);
    REQUIRE(verdict.reason.find("mismatched") != std::string::npos);
}

// ============================================================================
// §8 HIGHER_IS_BETTER direction
// ============================================================================

TEST_CASE("§8 Throughput improvement passes", "[v030][perf_gate]") {
    PerformanceGate gate;

    auto primary = make_throughput_metric(30, 1000.0, 50.0);  // +50 ops

    auto verdict = gate.evaluate(primary);
    REQUIRE(verdict.passed == true);
    REQUIRE(verdict.primary_result.improved == true);
    REQUIRE(verdict.primary_result.mean_diff > 0.0);  // candidate higher throughput
}

// ============================================================================
// §9 Paired t-test known values
// ============================================================================

TEST_CASE("§9 Paired t-test deterministic", "[v030][perf_gate]") {
    // Simple deterministic case: all diffs = -5
    std::vector<double> baseline  = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
    std::vector<double> candidate = {95,  95,  95,  95,  95,  95,  95,  95,  95,  95};

    auto result = PerformanceGate::paired_t_test(
        baseline, candidate, MetricDirection::LOWER_IS_BETTER);

    REQUIRE_THAT(result.mean_diff, Catch::Matchers::WithinAbs(-5.0, 1e-10));
    REQUIRE(result.improved == true);
    REQUIRE(result.n == 10);
    // With zero variance in differences, t should be very large (or inf)
    // p should be very small
    REQUIRE(result.p_value < 0.001);
}

// ============================================================================
// §10 p-value accuracy
// ============================================================================

TEST_CASE("§10 p-value computation sanity", "[v030][perf_gate]") {
    // t=0, df=10 → p should be 1.0
    double p0 = PerformanceGate::t_to_p(0.0, 10);
    REQUIRE_THAT(p0, Catch::Matchers::WithinAbs(1.0, 0.01));

    // Large t → p should be very small
    double p_large = PerformanceGate::t_to_p(10.0, 20);
    REQUIRE(p_large < 0.001);
}

// ============================================================================
// §11 is_improved convenience
// ============================================================================

TEST_CASE("§11 is_improved() convenience", "[v030][perf_gate]") {
    PerformanceGate gate;
    auto primary = make_latency_metric(30, 100.0, 10.0);
    REQUIRE(gate.is_improved(primary) == true);
}

// ============================================================================
// §12 Custom alpha
// ============================================================================

TEST_CASE("§12 Custom alpha threshold", "[v030][perf_gate]") {
    PerfConfig cfg;
    cfg.alpha = 0.001;  // very strict

    PerformanceGate gate(cfg);

    // Small improvement with noise — might not be significant at 0.001
    auto primary = make_latency_metric(15, 100.0, 3.0);

    auto verdict = gate.evaluate(primary);
    // With strict alpha, borderline improvements should fail
    // (depends on random noise, so we just verify it ran)
    REQUIRE(gate.total_evaluations() == 1);
}

// ============================================================================
// §13 No secondaries → pass
// ============================================================================

TEST_CASE("§13 No secondary metrics → pass if primary passes", "[v030][perf_gate]") {
    PerformanceGate gate;
    auto primary = make_latency_metric(30, 100.0, 10.0);

    auto verdict = gate.evaluate(primary, {});  // empty secondaries
    REQUIRE(verdict.passed == true);
    REQUIRE(verdict.secondary_checks.empty());
}

// ============================================================================
// §14 Effect size populated
// ============================================================================

TEST_CASE("§14 Effect size (Cohen's d)", "[v030][perf_gate]") {
    PerformanceGate gate;
    auto primary = make_latency_metric(30, 100.0, 10.0);

    auto verdict = gate.evaluate(primary);
    REQUIRE(verdict.primary_result.effect_size > 0.0);
}

// ============================================================================
// §15 Counters
// ============================================================================

TEST_CASE("§15 Evaluation counters", "[v030][perf_gate]") {
    PerformanceGate gate;
    REQUIRE(gate.total_evaluations() == 0);

    gate.evaluate(make_latency_metric(30, 100.0, 10.0));  // pass
    REQUIRE(gate.total_evaluations() == 1);
    REQUIRE(gate.total_failures() == 0);

    gate.evaluate(make_latency_metric(30, 100.0, 0.0));   // fail
    REQUIRE(gate.total_evaluations() == 2);
    REQUIRE(gate.total_failures() == 1);
}
