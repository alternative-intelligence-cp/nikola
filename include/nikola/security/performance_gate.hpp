/**
 * @file security/performance_gate.hpp
 * @brief v0.3.0 — Performance Benchmarking Gate (SIE Gate 5 "The Efficiency Gate")
 *
 * Verifies that a self-improved candidate module demonstrates statistically
 * significant improvement in the target metric without regression in
 * secondary metrics.
 *
 * Algorithm:
 *   1. Collect paired samples: (baseline_i, candidate_i) for N benchmark runs.
 *   2. Paired t-test on the differences d_i = candidate_i - baseline_i.
 *   3. Reject if p >= 0.05 (no significant improvement).
 *   4. Check secondary metrics for regression (mean worsened beyond tolerance).
 *
 * Supports:
 *   - Latency metrics (lower is better)
 *   - Throughput metrics (higher is better)
 *   - ATP cost metrics (lower is better)
 *
 * Spec: §2.3 Gate 5 — "Performance Benchmarking (The Efficiency Gate)"
 *        docs/info/integration/sections/05_autonomous_systems/04_self_improvement.md
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace nikola::security {

// ============================================================================
// Constants
// ============================================================================

/// Default significance level.
inline constexpr double PERF_ALPHA = 0.05;

/// Default minimum sample size for valid t-test.
inline constexpr size_t PERF_MIN_SAMPLES = 10;

/// Default secondary metric regression tolerance (5%).
inline constexpr double PERF_REGRESSION_TOLERANCE = 0.05;

// ============================================================================
// MetricDirection — whether lower or higher is better
// ============================================================================

enum class MetricDirection : uint8_t {
    LOWER_IS_BETTER,   ///< Latency, ATP cost
    HIGHER_IS_BETTER,  ///< Throughput, accuracy
};

// ============================================================================
// BenchmarkMetric — a named measurement with direction
// ============================================================================

struct BenchmarkMetric {
    std::string      name;
    MetricDirection  direction{MetricDirection::LOWER_IS_BETTER};

    /// Paired samples: baseline values.
    std::vector<double> baseline_samples;

    /// Paired samples: candidate values.
    std::vector<double> candidate_samples;
};

// ============================================================================
// PerfTestResult — statistical test output
// ============================================================================

struct PerfTestResult {
    bool        significant{false};   ///< p < alpha (and improvement direction correct)
    double      t_statistic{0.0};     ///< t-value from paired t-test
    double      p_value{1.0};         ///< Two-tailed p-value
    double      mean_diff{0.0};       ///< Mean of differences (candidate - baseline)
    double      std_error{0.0};       ///< Standard error of the mean difference
    double      effect_size{0.0};     ///< Cohen's d
    size_t      n{0};                 ///< Number of paired observations
    bool        improved{false};      ///< True if direction of change is improvement
};

// ============================================================================
// PerfVerdict — full gate verdict
// ============================================================================

struct PerfVerdict {
    bool        passed{false};
    std::string reason;

    /// Primary metric test result.
    PerfTestResult primary_result;

    /// Secondary metric regression checks.
    struct SecondaryCheck {
        std::string name;
        double      baseline_mean{0.0};
        double      candidate_mean{0.0};
        double      change_pct{0.0};  ///< Positive = worse, negative = better
        bool        regressed{false};
    };
    std::vector<SecondaryCheck> secondary_checks;

    /// Index of worst-regressing secondary metric (-1 if none regressed).
    int  worst_regression_idx{-1};
};

// ============================================================================
// PerfConfig
// ============================================================================

struct PerfConfig {
    /// Significance level (reject H0 if p < alpha).
    double alpha = PERF_ALPHA;

    /// Minimum paired samples required.
    size_t min_samples = PERF_MIN_SAMPLES;

    /// Regression tolerance for secondary metrics (fraction, e.g. 0.05 = 5%).
    double regression_tolerance = PERF_REGRESSION_TOLERANCE;
};

// ============================================================================
// PerformanceGate
// ============================================================================

/**
 * @class PerformanceGate
 * @brief Statistical benchmarking gate for SIE candidate validation.
 *
 * Usage:
 *   PerformanceGate gate;
 *
 *   BenchmarkMetric primary;
 *   primary.name = "physics_loop_latency_us";
 *   primary.direction = MetricDirection::LOWER_IS_BETTER;
 *   primary.baseline_samples  = {120, 118, 125, ...};
 *   primary.candidate_samples = {110, 108, 112, ...};
 *
 *   auto verdict = gate.evaluate(primary, {secondary1, secondary2});
 *   if (!verdict.passed) { reject_candidate(); }
 *
 * Thread safety: evaluate() is const and thread-safe.
 */
class PerformanceGate {
public:
    PerformanceGate();
    explicit PerformanceGate(PerfConfig config);

    /**
     * Evaluate a candidate against the primary and secondary metrics.
     *
     * @param primary     Primary metric with paired samples.
     * @param secondaries Secondary metrics to check for regression.
     * @return PerfVerdict with pass/fail, t-test results, regression checks.
     */
    [[nodiscard]] PerfVerdict evaluate(
        const BenchmarkMetric& primary,
        const std::vector<BenchmarkMetric>& secondaries = {}) const;

    /**
     * Quick boolean check.
     */
    [[nodiscard]] bool is_improved(
        const BenchmarkMetric& primary,
        const std::vector<BenchmarkMetric>& secondaries = {}) const;

    /// Total evaluations.
    [[nodiscard]] uint64_t total_evaluations() const noexcept {
        return total_evaluations_;
    }

    /// Total failures.
    [[nodiscard]] uint64_t total_failures() const noexcept {
        return total_failures_;
    }

    /// Access config.
    [[nodiscard]] const PerfConfig& config() const noexcept { return cfg_; }

    // ── Static utility ───────────────────────────────────────────────────────

    /**
     * Paired t-test on differences.
     * H0: mean(differences) = 0.
     * Returns t-statistic and approximate two-tailed p-value.
     */
    [[nodiscard]] static PerfTestResult paired_t_test(
        const std::vector<double>& baseline,
        const std::vector<double>& candidate,
        MetricDirection direction);

    /**
     * Approximate two-tailed p-value from t-statistic and degrees of freedom.
     * Uses the regularized incomplete beta function approximation.
     */
    [[nodiscard]] static double t_to_p(double t, size_t df);

private:
    PerfConfig       cfg_;
    mutable uint64_t total_evaluations_{0};
    mutable uint64_t total_failures_{0};

    /// Regularized incomplete beta function I_x(a, b) — for p-value computation.
    [[nodiscard]] static double betainc(double x, double a, double b);

    /// Continued fraction expansion for incomplete beta (Lentz's method).
    [[nodiscard]] static double betacf(double x, double a, double b);

    /// Log of the beta function: ln(B(a,b)).
    [[nodiscard]] static double lnbeta(double a, double b);
};

}  // namespace nikola::security
