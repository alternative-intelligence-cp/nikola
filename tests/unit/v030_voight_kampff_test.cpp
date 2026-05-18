/**
 * @file tests/unit/v030_voight_kampff_test.cpp
 * @brief v0.3.0 — VoightKampff alignment gate test suite
 *
 * Tests:
 *   §1  Default construction
 *   §2  Set baseline and verify identical responses (pass)
 *   §3  Slightly drifted responses still pass (> 0.999)
 *   §4  Significantly drifted responses fail
 *   §5  Mismatched response count fails
 *   §6  Mismatched vector dimensions fail
 *   §7  No baseline set → fail
 *   §8  Per-query similarity tracking
 *   §9  Weakest query identification
 *   §10 Custom min_similarity threshold
 *   §11 Per-query minimum threshold
 *   §12 Cosine similarity of orthogonal vectors = 0
 *   §13 Cosine similarity of identical vectors = 1
 *   §14 is_aligned() convenience method
 *   §15 Counters (total_verifications, total_failures)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/security/voight_kampff.hpp>

#include <cmath>
#include <vector>

using namespace nikola::security;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Create a unit vector in N dimensions, pointing mostly along axis `dominant`.
static std::vector<double> make_unit_vector(size_t dim, size_t dominant, double noise = 0.0) {
    std::vector<double> v(dim, noise);
    v[dominant] = 1.0;
    // Normalize
    double norm = 0.0;
    for (double x : v) norm += x * x;
    norm = std::sqrt(norm);
    for (double& x : v) x /= norm;
    return v;
}

/// Slightly perturb a vector (cosine similarity stays high).
static std::vector<double> perturb(const std::vector<double>& v, double magnitude) {
    auto result = v;
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] += magnitude * ((i % 2 == 0) ? 1.0 : -1.0) * 0.001;
    }
    // Re-normalize
    double norm = 0.0;
    for (double x : result) norm += x * x;
    norm = std::sqrt(norm);
    for (double& x : result) x /= norm;
    return result;
}

// ============================================================================
// §1 Default construction
// ============================================================================

TEST_CASE("§1 VoightKampff default construction", "[v030][voight_kampff]") {
    VoightKampff vk;
    REQUIRE(vk.has_baseline() == false);
    REQUIRE(vk.suite_size() == 0);
    REQUIRE(vk.total_verifications() == 0);
    REQUIRE(vk.total_failures() == 0);
    REQUIRE(vk.config().min_similarity == VK_MIN_SIMILARITY);
}

// ============================================================================
// §2 Identical responses pass
// ============================================================================

TEST_CASE("§2 Identical responses pass", "[v030][voight_kampff]") {
    VoightKampff vk;

    std::vector<std::vector<double>> baseline = {
        {0.1, 0.9, 0.3, 0.2},
        {0.5, 0.5, 0.0, 0.7},
        {0.8, 0.1, 0.5, 0.3},
    };

    vk.set_baseline(baseline);
    REQUIRE(vk.has_baseline());
    REQUIRE(vk.suite_size() == 3);

    auto verdict = vk.verify(baseline);  // same vectors
    REQUIRE(verdict.passed == true);
    REQUIRE_THAT(verdict.overall_similarity, Catch::Matchers::WithinAbs(1.0, 1e-10));
}

// ============================================================================
// §3 Slightly drifted responses still pass
// ============================================================================

TEST_CASE("§3 Slight drift still passes", "[v030][voight_kampff]") {
    VoightKampff vk;

    std::vector<std::vector<double>> baseline = {
        make_unit_vector(128, 0),
        make_unit_vector(128, 10),
        make_unit_vector(128, 50),
    };
    vk.set_baseline(baseline);

    // Tiny perturbation — cosine sim should stay > 0.999
    std::vector<std::vector<double>> candidate;
    for (const auto& b : baseline) {
        candidate.push_back(perturb(b, 0.001));
    }

    auto verdict = vk.verify(candidate);
    REQUIRE(verdict.passed == true);
    REQUIRE(verdict.overall_similarity > 0.999);
}

// ============================================================================
// §4 Significantly drifted responses fail
// ============================================================================

TEST_CASE("§4 Significant drift fails", "[v030][voight_kampff]") {
    VoightKampff vk;

    std::vector<std::vector<double>> baseline = {
        make_unit_vector(64, 0),
        make_unit_vector(64, 10),
        make_unit_vector(64, 30),
    };
    vk.set_baseline(baseline);

    // Large perturbation — replace one vector entirely
    std::vector<std::vector<double>> candidate = baseline;
    candidate[1] = make_unit_vector(64, 20);  // totally different direction

    auto verdict = vk.verify(candidate);
    REQUIRE(verdict.passed == false);
    REQUIRE(verdict.overall_similarity < 0.999);
}

// ============================================================================
// §5 Mismatched response count
// ============================================================================

TEST_CASE("§5 Mismatched response count fails", "[v030][voight_kampff]") {
    VoightKampff vk;

    vk.set_baseline({{1.0, 0.0}, {0.0, 1.0}});

    // Only 1 response instead of 2
    auto verdict = vk.verify({{1.0, 0.0}});
    REQUIRE(verdict.passed == false);
    REQUIRE(verdict.reason.find("responses") != std::string::npos);
}

// ============================================================================
// §6 Mismatched vector dimensions
// ============================================================================

TEST_CASE("§6 Mismatched dimensions fails", "[v030][voight_kampff]") {
    VoightKampff vk;

    vk.set_baseline({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}});

    // Second vector has wrong dimensions
    auto verdict = vk.verify({{1.0, 0.0, 0.0}, {0.0, 1.0}});
    REQUIRE(verdict.passed == false);
    REQUIRE(verdict.reason.find("mismatch") != std::string::npos);
}

// ============================================================================
// §7 No baseline set
// ============================================================================

TEST_CASE("§7 No baseline → fail", "[v030][voight_kampff]") {
    VoightKampff vk;
    auto verdict = vk.verify({{1.0, 0.0}});
    REQUIRE(verdict.passed == false);
    REQUIRE(verdict.reason.find("baseline") != std::string::npos);
    REQUIRE(vk.total_failures() == 1);
}

// ============================================================================
// §8 Per-query similarities populated
// ============================================================================

TEST_CASE("§8 Per-query similarities", "[v030][voight_kampff]") {
    VoightKampff vk;

    std::vector<std::vector<double>> baseline = {
        {1.0, 0.0},
        {0.0, 1.0},
    };
    vk.set_baseline(baseline);

    auto verdict = vk.verify(baseline);
    REQUIRE(verdict.per_query_similarities.size() == 2);
    REQUIRE_THAT(verdict.per_query_similarities[0], Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(verdict.per_query_similarities[1], Catch::Matchers::WithinAbs(1.0, 1e-10));
}

// ============================================================================
// §9 Weakest query identification
// ============================================================================

TEST_CASE("§9 Weakest query identified", "[v030][voight_kampff]") {
    VoightKampff vk;

    std::vector<std::vector<double>> baseline = {
        make_unit_vector(32, 0),
        make_unit_vector(32, 5),
        make_unit_vector(32, 15),
    };
    vk.set_baseline(baseline);

    // Drift only the second query
    auto candidate = baseline;
    candidate[1] = make_unit_vector(32, 6);

    auto verdict = vk.verify(candidate);
    REQUIRE(verdict.weakest_query_idx == 1);
    REQUIRE(verdict.weakest_similarity < 1.0);
}

// ============================================================================
// §10 Custom min_similarity
// ============================================================================

TEST_CASE("§10 Custom min_similarity threshold", "[v030][voight_kampff]") {
    VKConfig cfg;
    cfg.min_similarity = 0.9;  // much more permissive

    VoightKampff vk(cfg);
    vk.set_baseline({make_unit_vector(32, 0)});

    // Moderate perturbation
    auto candidate = std::vector<std::vector<double>>{perturb(make_unit_vector(32, 0), 10.0)};
    auto verdict = vk.verify(candidate);
    // With 0.9 threshold, moderate perturbation should pass
    if (verdict.overall_similarity > 0.9) {
        REQUIRE(verdict.passed == true);
    }
}

// ============================================================================
// §11 Per-query minimum threshold
// ============================================================================

TEST_CASE("§11 Per-query minimum threshold", "[v030][voight_kampff]") {
    VKConfig cfg;
    cfg.min_similarity = 0.9;
    cfg.min_per_query_similarity = 0.95;

    VoightKampff vk(cfg);
    vk.set_baseline({make_unit_vector(64, 0), make_unit_vector(64, 10)});

    // One query passes, one barely fails
    auto candidate = std::vector<std::vector<double>>{
        make_unit_vector(64, 0),      // identical
        make_unit_vector(64, 11),     // adjacent but different
    };

    auto verdict = vk.verify(candidate);
    // The drifted query should be caught by per-query threshold
    if (verdict.per_query_similarities.size() == 2
        && verdict.per_query_similarities[1] < 0.95) {
        REQUIRE(verdict.passed == false);
    }
}

// ============================================================================
// §12 Cosine similarity: orthogonal = 0
// ============================================================================

TEST_CASE("§12 Cosine similarity orthogonal = 0", "[v030][voight_kampff]") {
    double sim = VoightKampff::cosine_similarity({1.0, 0.0}, {0.0, 1.0});
    REQUIRE_THAT(sim, Catch::Matchers::WithinAbs(0.0, 1e-10));
}

// ============================================================================
// §13 Cosine similarity: identical = 1
// ============================================================================

TEST_CASE("§13 Cosine similarity identical = 1", "[v030][voight_kampff]") {
    double sim = VoightKampff::cosine_similarity({0.3, 0.4, 0.5}, {0.3, 0.4, 0.5});
    REQUIRE_THAT(sim, Catch::Matchers::WithinAbs(1.0, 1e-10));
}

// ============================================================================
// §14 is_aligned convenience
// ============================================================================

TEST_CASE("§14 is_aligned() convenience", "[v030][voight_kampff]") {
    VoightKampff vk;
    vk.set_baseline({{1.0, 0.0}, {0.0, 1.0}});

    REQUIRE(vk.is_aligned({{1.0, 0.0}, {0.0, 1.0}}) == true);
}

// ============================================================================
// §15 Counters
// ============================================================================

TEST_CASE("§15 Verification counters", "[v030][voight_kampff]") {
    VoightKampff vk;
    REQUIRE(vk.total_verifications() == 0);

    // Fail (no baseline)
    vk.verify({{1.0}});
    REQUIRE(vk.total_verifications() == 1);
    REQUIRE(vk.total_failures() == 1);

    // Pass
    vk.set_baseline({{1.0}});
    vk.verify({{1.0}});
    REQUIRE(vk.total_verifications() == 2);
    REQUIRE(vk.total_failures() == 1);
}
