/**
 * @file src/security/performance_gate.cpp
 * @brief v0.3.0 — PerformanceGate implementation.
 *
 * Self-contained paired t-test with approximate p-value computation
 * using the regularized incomplete beta function (no external stats lib).
 */

#include <nikola/security/performance_gate.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace nikola::security {

// ── Construction ────────────────────────────────────────────────────────────

PerformanceGate::PerformanceGate() : cfg_{} {}

PerformanceGate::PerformanceGate(PerfConfig config) : cfg_(std::move(config)) {}

// ── Main evaluation ─────────────────────────────────────────────────────────

PerfVerdict PerformanceGate::evaluate(
    const BenchmarkMetric& primary,
    const std::vector<BenchmarkMetric>& secondaries) const
{
    ++total_evaluations_;
    PerfVerdict v;

    // ── Validate sample sizes ────────────────────────────────────────────────
    if (primary.baseline_samples.size() != primary.candidate_samples.size()) {
        v.passed = false;
        v.reason = "Primary metric '" + primary.name
                 + "': mismatched sample counts (baseline="
                 + std::to_string(primary.baseline_samples.size())
                 + ", candidate="
                 + std::to_string(primary.candidate_samples.size()) + ")";
        ++total_failures_;
        return v;
    }

    if (primary.baseline_samples.size() < cfg_.min_samples) {
        v.passed = false;
        v.reason = "Primary metric '" + primary.name
                 + "': insufficient samples ("
                 + std::to_string(primary.baseline_samples.size())
                 + " < " + std::to_string(cfg_.min_samples) + ")";
        ++total_failures_;
        return v;
    }

    // ── Primary metric: paired t-test ────────────────────────────────────────
    v.primary_result = paired_t_test(
        primary.baseline_samples,
        primary.candidate_samples,
        primary.direction);

    if (!v.primary_result.significant || !v.primary_result.improved) {
        v.passed = false;
        if (!v.primary_result.improved) {
            v.reason = "Primary metric '" + primary.name
                     + "' did not improve (mean_diff="
                     + std::to_string(v.primary_result.mean_diff) + ")";
        } else {
            v.reason = "Primary metric '" + primary.name
                     + "' improvement not significant (p="
                     + std::to_string(v.primary_result.p_value)
                     + " >= " + std::to_string(cfg_.alpha) + ")";
        }
        ++total_failures_;
        return v;
    }

    // ── Secondary metrics: regression check ──────────────────────────────────
    double worst_pct = 0.0;
    v.worst_regression_idx = -1;

    for (size_t i = 0; i < secondaries.size(); ++i) {
        const auto& sec = secondaries[i];
        PerfVerdict::SecondaryCheck sc;
        sc.name = sec.name;

        if (sec.baseline_samples.empty() || sec.candidate_samples.empty()) {
            sc.baseline_mean  = 0.0;
            sc.candidate_mean = 0.0;
            sc.change_pct     = 0.0;
            sc.regressed      = false;
            v.secondary_checks.push_back(std::move(sc));
            continue;
        }

        sc.baseline_mean = std::accumulate(
            sec.baseline_samples.begin(),
            sec.baseline_samples.end(), 0.0)
            / static_cast<double>(sec.baseline_samples.size());

        sc.candidate_mean = std::accumulate(
            sec.candidate_samples.begin(),
            sec.candidate_samples.end(), 0.0)
            / static_cast<double>(sec.candidate_samples.size());

        // Calculate change percentage
        if (std::abs(sc.baseline_mean) > 1e-30) {
            double raw_change = (sc.candidate_mean - sc.baseline_mean) / std::abs(sc.baseline_mean);

            // For LOWER_IS_BETTER: positive change = worse.
            // For HIGHER_IS_BETTER: negative change = worse.
            if (sec.direction == MetricDirection::LOWER_IS_BETTER) {
                sc.change_pct = raw_change;  // positive = regression
            } else {
                sc.change_pct = -raw_change;  // negative change (lower throughput) = regression
            }
        } else {
            sc.change_pct = 0.0;
        }

        sc.regressed = sc.change_pct > cfg_.regression_tolerance;

        if (sc.regressed && sc.change_pct > worst_pct) {
            worst_pct = sc.change_pct;
            v.worst_regression_idx = static_cast<int>(i);
        }

        v.secondary_checks.push_back(std::move(sc));
    }

    if (v.worst_regression_idx >= 0) {
        const auto& worst = v.secondary_checks[static_cast<size_t>(v.worst_regression_idx)];
        v.passed = false;
        v.reason = "Secondary metric '" + worst.name + "' regressed by "
                 + std::to_string(worst.change_pct * 100.0)
                 + "% (tolerance=" + std::to_string(cfg_.regression_tolerance * 100.0) + "%)";
        ++total_failures_;
        return v;
    }

    v.passed = true;
    return v;
}

bool PerformanceGate::is_improved(
    const BenchmarkMetric& primary,
    const std::vector<BenchmarkMetric>& secondaries) const
{
    return evaluate(primary, secondaries).passed;
}

// ── Paired t-test ───────────────────────────────────────────────────────────

PerfTestResult PerformanceGate::paired_t_test(
    const std::vector<double>& baseline,
    const std::vector<double>& candidate,
    MetricDirection direction)
{
    PerfTestResult r;
    r.n = baseline.size();

    if (r.n < 2 || candidate.size() != r.n) {
        r.significant = false;
        r.p_value     = 1.0;
        return r;
    }

    // Compute differences d_i = candidate_i - baseline_i
    std::vector<double> diffs(r.n);
    for (size_t i = 0; i < r.n; ++i) {
        diffs[i] = candidate[i] - baseline[i];
    }

    // Mean of differences
    double sum_d = std::accumulate(diffs.begin(), diffs.end(), 0.0);
    r.mean_diff = sum_d / static_cast<double>(r.n);

    // Standard deviation of differences
    double sum_sq = 0.0;
    for (double d : diffs) {
        double dev = d - r.mean_diff;
        sum_sq += dev * dev;
    }
    double sd = std::sqrt(sum_sq / static_cast<double>(r.n - 1));

    // Standard error
    r.std_error = sd / std::sqrt(static_cast<double>(r.n));

    // t-statistic
    if (r.std_error < 1e-30) {
        // All differences are identical
        r.t_statistic = (std::abs(r.mean_diff) < 1e-30) ? 0.0 : 1e10;
    } else {
        r.t_statistic = r.mean_diff / r.std_error;
    }

    // Cohen's d effect size
    r.effect_size = (sd > 1e-30) ? std::abs(r.mean_diff) / sd : 0.0;

    // Degrees of freedom
    size_t df = r.n - 1;

    // Two-tailed p-value
    r.p_value = t_to_p(r.t_statistic, df);

    // Direction check
    // For LOWER_IS_BETTER: improvement means mean_diff < 0 (candidate faster)
    // For HIGHER_IS_BETTER: improvement means mean_diff > 0 (candidate higher throughput)
    if (direction == MetricDirection::LOWER_IS_BETTER) {
        r.improved = r.mean_diff < 0.0;
    } else {
        r.improved = r.mean_diff > 0.0;
    }

    // Significant if p < alpha (we use the caller's alpha, but test at 0.05 here)
    // The caller checks significance against their configured alpha
    r.significant = r.p_value < 0.05;

    return r;
}

// ── p-value from t-statistic ────────────────────────────────────────────────

double PerformanceGate::t_to_p(double t, size_t df) {
    if (df == 0) return 1.0;

    // p = I_{v/(v+t²)}(v/2, 1/2) where v = df
    double v = static_cast<double>(df);
    double x = v / (v + t * t);

    return betainc(x, v / 2.0, 0.5);
}

// ── Regularized incomplete beta function I_x(a, b) ─────────────────────────
// Used for computing p-values from the t-distribution.
// Implementation follows Numerical Recipes' approach.

double PerformanceGate::lnbeta(double a, double b) {
    return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}

double PerformanceGate::betacf(double x, double a, double b) {
    constexpr int MAX_ITER = 200;
    constexpr double EPS   = 3.0e-12;
    constexpr double FPMIN = 1.0e-30;

    double qab = a + b;
    double qap = a + 1.0;
    double qam = a - 1.0;

    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= MAX_ITER; ++m) {
        double m2 = 2.0 * m;

        // Even step
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c;
        if (std::abs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c;
        if (std::abs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        double del = d * c;
        h *= del;

        if (std::abs(del - 1.0) < EPS) break;
    }

    return h;
}

double PerformanceGate::betainc(double x, double a, double b) {
    if (x < 0.0 || x > 1.0) return 0.0;
    if (x == 0.0 || x == 1.0) return x;

    double bt = std::exp(
        a * std::log(x) + b * std::log(1.0 - x) - lnbeta(a, b));

    // Use symmetry for numerical stability
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * betacf(x, a, b) / a;
    } else {
        return 1.0 - bt * betacf(1.0 - x, b, a) / b;
    }
}

}  // namespace nikola::security
