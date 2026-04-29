/**
 * @file src/security/voight_kampff.cpp
 * @brief v0.3.0 — VoightKampff alignment gate implementation.
 */

#include <nikola/security/voight_kampff.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace nikola::security {

// ── Construction ────────────────────────────────────────────────────────────

VoightKampff::VoightKampff() : cfg_{} {}

VoightKampff::VoightKampff(VKConfig config) : cfg_(std::move(config)) {}

// ── Baseline management ─────────────────────────────────────────────────────

void VoightKampff::set_baseline(
    const std::vector<std::vector<double>>& baseline_responses)
{
    if (baseline_responses.empty()) {
        throw std::invalid_argument(
            "VoightKampff: baseline must contain at least one response vector");
    }

    // Verify all vectors have the same dimensionality
    size_t dim = baseline_responses[0].size();
    for (size_t i = 1; i < baseline_responses.size(); ++i) {
        if (baseline_responses[i].size() != dim) {
            throw std::invalid_argument(
                "VoightKampff: baseline response vectors must all have the "
                "same dimensionality (query " + std::to_string(i)
                + " has " + std::to_string(baseline_responses[i].size())
                + ", expected " + std::to_string(dim) + ")");
        }
    }

    baseline_ = baseline_responses;
}

// ── Verification ────────────────────────────────────────────────────────────

VKVerdict VoightKampff::verify(
    const std::vector<std::vector<double>>& candidate_responses) const
{
    ++total_verifications_;
    VKVerdict v;

    // ── Structural checks ────────────────────────────────────────────────────
    if (baseline_.empty()) {
        v.passed = false;
        v.reason = "No baseline has been set";
        ++total_failures_;
        return v;
    }

    if (candidate_responses.size() != baseline_.size()) {
        v.passed = false;
        v.reason = "Candidate has " + std::to_string(candidate_responses.size())
                 + " responses, expected " + std::to_string(baseline_.size());
        ++total_failures_;
        return v;
    }

    // ── Per-query cosine similarity ──────────────────────────────────────────
    v.per_query_similarities.resize(baseline_.size());
    double sum_sim = 0.0;
    v.weakest_similarity = 1.0;
    v.weakest_query_idx  = 0;

    for (size_t i = 0; i < baseline_.size(); ++i) {
        if (candidate_responses[i].size() != baseline_[i].size()) {
            v.passed = false;
            v.reason = "Dimension mismatch at query " + std::to_string(i)
                     + ": candidate=" + std::to_string(candidate_responses[i].size())
                     + " baseline=" + std::to_string(baseline_[i].size());
            ++total_failures_;
            return v;
        }

        double sim = cosine_similarity(baseline_[i], candidate_responses[i]);
        v.per_query_similarities[i] = sim;
        sum_sim += sim;

        if (sim < v.weakest_similarity) {
            v.weakest_similarity = sim;
            v.weakest_query_idx  = static_cast<int>(i);
        }
    }

    v.overall_similarity = sum_sim / static_cast<double>(baseline_.size());

    // ── Overall threshold ────────────────────────────────────────────────────
    if (v.overall_similarity < cfg_.min_similarity) {
        v.passed = false;
        v.reason = "Overall similarity " + std::to_string(v.overall_similarity)
                 + " < " + std::to_string(cfg_.min_similarity);
        if (v.weakest_query_idx >= 0) {
            std::string label = (static_cast<size_t>(v.weakest_query_idx) < cfg_.query_labels.size())
                ? cfg_.query_labels[static_cast<size_t>(v.weakest_query_idx)]
                : "query_" + std::to_string(v.weakest_query_idx);
            v.reason += "; weakest: " + label + " ("
                     + std::to_string(v.weakest_similarity) + ")";
        }
        ++total_failures_;
        return v;
    }

    // ── Per-query threshold (optional) ───────────────────────────────────────
    if (cfg_.min_per_query_similarity > 0.0
        && v.weakest_similarity < cfg_.min_per_query_similarity) {
        v.passed = false;
        std::string label = (static_cast<size_t>(v.weakest_query_idx) < cfg_.query_labels.size())
            ? cfg_.query_labels[static_cast<size_t>(v.weakest_query_idx)]
            : "query_" + std::to_string(v.weakest_query_idx);
        v.reason = "Per-query similarity " + std::to_string(v.weakest_similarity)
                 + " < " + std::to_string(cfg_.min_per_query_similarity)
                 + " at " + label;
        ++total_failures_;
        return v;
    }

    v.passed = true;
    return v;
}

bool VoightKampff::is_aligned(
    const std::vector<std::vector<double>>& candidate_responses) const
{
    return verify(candidate_responses).passed;
}

// ── Cosine similarity ───────────────────────────────────────────────────────

double VoightKampff::cosine_similarity(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    if (a.size() != b.size() || a.empty()) return 0.0;

    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-30) return 0.0;

    return dot / denom;
}

}  // namespace nikola::security
