// ============================================================================
// phase39_curvature_bias_test.cpp   Phase 39 — Riemannian Curvature Bias
// ============================================================================
//
// Tests:
//   §1  mean_curvature: seed_manifold default resonance=0.5 → R̄ = 0.5
//   §2  mean_curvature: all-frozen resonance=1.0 → R̄ = 0.0
//   §3  mean_curvature: all-plastic resonance=0.0 → R̄ = 1.0
//   §4  npt_curvature_weights: strictly ascending (w[i] < w[i+1])
//   §5  npt_curvature_weights: head 7 == 1.0 (max, fully normalized)
//   §6  npt_curvature_weights: head 0 ≈ f₀/f₇ (cross-validated)
//   §7  apply_curvature_bias: alpha=0.0 → output identical to input
//   §8  apply_curvature_bias: R_mean=0 → output identical to input
//   §9  apply_curvature_bias: head 7 receives larger offset than head 0
//   §10 apply_curvature_bias: formula correct on single-head spot-check
//   §11 curvature_alpha accessor returns construction value
//   §12 forward(): frozen torus (R̄≈0) → scores indistinguishable from alpha=0
//   §13 forward(): plastic torus (R̄≈1) → score[7] > score[0]
//   §14 forward(): scores still sum to 1.0 with curvature bias active
//   §15 forward(): default alpha=0.5 + default seeded torus → score[7] > score[0]
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/neuroplastic_transformer.hpp>
#include <nikola/physics/wave_function.hpp>

#include <cmath>
#include <numeric>

using namespace nikola::cognitive;
using namespace nikola::physics;
namespace physics = nikola::physics;
using Approx = Catch::Approx;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sum all 8 head scores.
static float sum8(const std::array<float, NPT_NUM_HEADS>& a) {
    float s = 0.f;
    for (float v : a) s += v;
    return s;
}

/// Set every node's resonance to `value` via mutable grid access.
static void set_resonance_uniform(WaveFunction& wf, float value) {
    const size_t N = wf.num_nodes();
    float* r = wf.grid().resonance();
    for (size_t i = 0; i < N; ++i) r[i] = value;
}

// ---------------------------------------------------------------------------
// §1  mean_curvature: seed_manifold resonance=0.5 → R̄ = 0.5
// ---------------------------------------------------------------------------
TEST_CASE("§1 mean_curvature: default seed resonance=0.5 → R̄=0.5", "[Phase39][curvature]") {
    WaveFunction wf;
    wf.seed_manifold(3, 3, 1, 1.f, 42);
    // seed_manifold sets each node's resonance to 0.5 → R_i = 1-0.5 = 0.5 → R̄ = 0.5
    REQUIRE(wf.mean_curvature() == Approx(0.5).epsilon(1e-5));
}

// ---------------------------------------------------------------------------
// §2  mean_curvature: all-frozen (resonance=1.0) → R̄ = 0.0
// ---------------------------------------------------------------------------
TEST_CASE("§2 mean_curvature: frozen field (r=1.0) → R̄=0.0", "[Phase39][curvature]") {
    WaveFunction wf;
    wf.seed_manifold(3, 3, 1, 1.f, 42);
    set_resonance_uniform(wf, 1.0f);
    REQUIRE(wf.mean_curvature() == Approx(0.0).margin(1e-6));
}

// ---------------------------------------------------------------------------
// §3  mean_curvature: all-plastic (resonance=0.0) → R̄ = 1.0
// ---------------------------------------------------------------------------
TEST_CASE("§3 mean_curvature: plastic field (r=0.0) → R̄=1.0", "[Phase39][curvature]") {
    WaveFunction wf;
    wf.seed_manifold(3, 3, 1, 1.f, 42);
    set_resonance_uniform(wf, 0.0f);
    REQUIRE(wf.mean_curvature() == Approx(1.0).epsilon(1e-5));
}

// ---------------------------------------------------------------------------
// §4  npt_curvature_weights: strictly ascending
// ---------------------------------------------------------------------------
TEST_CASE("§4 npt_curvature_weights: strictly ascending", "[Phase39][weights]") {
    const auto w = npt_curvature_weights();
    for (size_t i = 1; i < NPT_NUM_HEADS; ++i) {
        INFO("w[" << i-1 << "]=" << w[i-1] << " w[" << i << "]=" << w[i]);
        REQUIRE(w[i] > w[i-1]);
    }
}

// ---------------------------------------------------------------------------
// §5  npt_curvature_weights: head 7 == 1.0 (f₇/f₇)
// ---------------------------------------------------------------------------
TEST_CASE("§5 npt_curvature_weights: head 7 == 1.0", "[Phase39][weights]") {
    const auto w = npt_curvature_weights();
    REQUIRE(w[7] == Approx(1.0f).epsilon(1e-6f));
}

// ---------------------------------------------------------------------------
// §6  npt_curvature_weights: head 0 ≈ φ⁻⁷
// ---------------------------------------------------------------------------
TEST_CASE("§6 npt_curvature_weights: head 0 ≈ f₀/f₇ = φ⁻⁷", "[Phase39][weights]") {
    static constexpr double PHI = 1.6180339887498948482;
    // f₀/f₇ = 1/φ⁷
    double phi7 = 1.0;
    for (int i = 0; i < 7; ++i) phi7 *= PHI;
    const float expected = static_cast<float>(1.0 / phi7);

    const auto w = npt_curvature_weights();
    INFO("w[0]=" << w[0] << "  expected=" << expected);
    REQUIRE(w[0] == Approx(expected).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §7  apply_curvature_bias: alpha=0.0 → output identical to input
// ---------------------------------------------------------------------------
TEST_CASE("§7 apply_curvature_bias: alpha=0 → no change", "[Phase39][bias]") {
    std::array<float, NPT_NUM_HEADS> raw{1.f, -0.3f, 0.5f, -1.f, 0.2f, 0.7f, 0.f, 0.9f};
    const auto w = npt_curvature_weights();
    const auto biased = apply_curvature_bias(raw, 0.8f, w, 0.0f);  // alpha=0

    for (size_t i = 0; i < NPT_NUM_HEADS; ++i)
        REQUIRE(biased[i] == Approx(raw[i]).margin(1e-6f));
}

// ---------------------------------------------------------------------------
// §8  apply_curvature_bias: R_mean=0 → output identical to input
// ---------------------------------------------------------------------------
TEST_CASE("§8 apply_curvature_bias: R_mean=0 → no change", "[Phase39][bias]") {
    std::array<float, NPT_NUM_HEADS> raw{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    const auto w = npt_curvature_weights();
    const auto biased = apply_curvature_bias(raw, 0.0f, w, 0.5f);  // R_mean=0

    for (size_t i = 0; i < NPT_NUM_HEADS; ++i)
        REQUIRE(biased[i] == Approx(raw[i]).margin(1e-6f));
}

// ---------------------------------------------------------------------------
// §9  apply_curvature_bias: head 7 receives larger offset than head 0
// ---------------------------------------------------------------------------
TEST_CASE("§9 apply_curvature_bias: offset[7] > offset[0]", "[Phase39][bias]") {
    // Equal raw scores → only difference is the weight
    std::array<float, NPT_NUM_HEADS> raw{};
    raw.fill(0.f);
    const auto w = npt_curvature_weights();
    const auto biased = apply_curvature_bias(raw, 0.5f, w, 1.0f);  // R=0.5, alpha=1

    INFO("biased[0]=" << biased[0] << "  biased[7]=" << biased[7]);
    REQUIRE(biased[7] > biased[0]);
}

// ---------------------------------------------------------------------------
// §10 apply_curvature_bias: formula spot-check on head 3
// ---------------------------------------------------------------------------
TEST_CASE("§10 apply_curvature_bias: formula check — biased_i = raw_i + α·R̄·w_i",
          "[Phase39][bias]") {
    std::array<float, NPT_NUM_HEADS> raw{};
    raw.fill(0.f);
    raw[3] = 0.25f;  // only head 3 has non-zero raw

    const auto w     = npt_curvature_weights();
    const float R    = 0.6f;
    const float alpha = 0.5f;
    const auto biased = apply_curvature_bias(raw, R, w, alpha);

    // Head 3 expected: 0.25 + 0.5 * 0.6 * w[3]
    const float expected3 = 0.25f + alpha * R * w[3];
    REQUIRE(biased[3] == Approx(expected3).epsilon(1e-5f));

    // Head 0 expected: 0.0 + 0.5 * 0.6 * w[0]
    const float expected0 = alpha * R * w[0];
    REQUIRE(biased[0] == Approx(expected0).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §11 curvature_alpha accessor returns construction value
// ---------------------------------------------------------------------------
TEST_CASE("§11 NPT: curvature_alpha accessor", "[Phase39][npt]") {
    NeuroplasticTransformer npt_default(3);
    REQUIRE(npt_default.curvature_alpha() == Approx(0.5f).epsilon(1e-6f));

    NeuroplasticTransformer npt_zero(3, 1.0f, 0.0f);
    REQUIRE(npt_zero.curvature_alpha() == Approx(0.0f).margin(1e-6f));

    NeuroplasticTransformer npt_full(3, 1.0f, 1.0f);
    REQUIRE(npt_full.curvature_alpha() == Approx(1.0f).epsilon(1e-6f));
}

// ---------------------------------------------------------------------------
// §12 forward(): frozen torus (R̄=0) → same scores as alpha=0
// ---------------------------------------------------------------------------
TEST_CASE("§12 NPT forward(): frozen torus (R̄=0) matches alpha=0 result",
          "[Phase39][forward]") {
    // With R̄=0 the curvature bias term is 0 regardless of alpha.
    // Result should be identical to running with alpha=0.
    NeuroplasticTransformer npt_biased(3, 1.0f, 1.0f);      // alpha=1, but R=0
    NeuroplasticTransformer npt_off   (3, 1.0f, 0.0f);      // alpha=0

    WaveFunction torus_frozen;
    torus_frozen.seed_manifold(3, 3, 1, 1.f, 42);
    set_resonance_uniform(torus_frozen, 1.0f);   // R̄ = 0

    auto r_biased = npt_biased.forward(torus_frozen);
    auto r_off    = npt_off.forward(torus_frozen);

    for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
        INFO("head " << i << ": biased=" << r_biased.head_scores[i]
             << " off=" << r_off.head_scores[i]);
        REQUIRE(r_biased.head_scores[i] == Approx(r_off.head_scores[i]).margin(1e-5f));
    }
}

// ---------------------------------------------------------------------------
// §13 forward(): fully-plastic torus (R̄=1) → score[7] > score[0]
// ---------------------------------------------------------------------------
TEST_CASE("§13 NPT forward(): plastic torus → high-freq head favoured",
          "[Phase39][forward]") {
    // All Q are vacuum → raw correlations all = 0.
    // Bias = alpha * 1.0 * w[i] → w[7] >> w[0] → score[7] > score[0].
    NeuroplasticTransformer npt(3, 1.0f, 0.5f);

    WaveFunction torus_plastic;
    torus_plastic.seed_manifold(3, 3, 1, 1.f, 42);
    set_resonance_uniform(torus_plastic, 0.0f);   // R̄ = 1.0

    auto result = npt.forward(torus_plastic);

    INFO("score[0]=" << result.head_scores[0]
         << "  score[7]=" << result.head_scores[7]);
    REQUIRE(result.head_scores[7] > result.head_scores[0]);

    // Also confirm monotone ordering: scores should increase with head index
    // (since raw is uniform and weights are strictly ascending)
    for (size_t i = 1; i < NPT_NUM_HEADS; ++i)
        REQUIRE(result.head_scores[i] > result.head_scores[i-1]);
}

// ---------------------------------------------------------------------------
// §14 forward(): scores still sum to 1.0 with bias active
// ---------------------------------------------------------------------------
TEST_CASE("§14 NPT forward(): curvature-biased scores sum to 1.0",
          "[Phase39][forward]") {
    NeuroplasticTransformer npt(3, 1.0f, 0.5f);

    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);   // default resonance=0.5, R̄=0.5

    auto result = npt.forward(torus_wf);
    INFO("sum of scores = " << sum8(result.head_scores));
    REQUIRE(sum8(result.head_scores) == Approx(1.0f).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §15 forward(): default construction + default seed → score[7] > score[0]
// ---------------------------------------------------------------------------
TEST_CASE("§15 NPT forward(): default alpha=0.5 + standard torus → score[7] > score[0]",
          "[Phase39][forward]") {
    // Standard usage: default NPT, standard seeded torus.
    // Vacuum Q heads → raw=0 for all; default seed resonance=0.5 → R̄=0.5.
    // Bias = 0.5 * 0.5 * w[i] → w[7]=1.0 >> w[0]≈0.056 → score[7] > score[0].
    NeuroplasticTransformer npt;   // default grid_n=3, temperature=1, alpha=0.5

    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);

    auto result = npt.forward(torus_wf);

    INFO("score[0]=" << result.head_scores[0]
         << "  score[7]=" << result.head_scores[7]);
    REQUIRE(result.head_scores[7] > result.head_scores[0]);
}
