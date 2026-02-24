// ============================================================================
// phase38_wave_correlation_test.cpp   Phase 38 — Wave Correlation Kernel
// ============================================================================
//
// Tests:
//   §1  inner_product_re: identical fields → equals total_probability
//   §2  inner_product_re: size mismatch → 0
//   §3  inner_product_re: known 2-field value via direct injection
//   §4  wave_correlation: in-phase (Q = K, same seed) → ≈ +1.0
//   §5  wave_correlation: anti-phase (K psi negated) → ≈ −1.0
//   §6  wave_correlation: orthogonal fields → ≈ 0.0
//   §7  wave_correlation: symmetric (correlation(Q,K) == correlation(K,Q))
//   §8  wave_correlation: both vacuum → 0.0 (defined neutral)
//   §9  attention_softmax: output sums to 1.0
//   §10 attention_softmax: ordering preserved (higher in → higher out)
//   §11 attention_softmax: temperature sharpens the distribution
//   §12 attention_softmax: uniform input → each output ≈ 1/8
//   §13 forward(): scores sum to 1.0 (softmax property)
//   §14 forward(): vacuum Q + seeded torus → uniform 1/8 scores
//   §15 forward(): has_output == false (Phase 40 not yet)
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/neuroplastic_transformer.hpp>
#include <nikola/physics/wave_function.hpp>

#include <array>
#include <cmath>
#include <numeric>   // std::accumulate

using namespace nikola::cognitive;
using namespace nikola::physics;
namespace physics = nikola::physics;
using Approx = Catch::Approx;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a single-node WaveFunction (n=1 → 1^9 = 1 node), vacuum field.
static WaveFunction make_single_node_vacuum() {
    WaveFunction wf;
    wf.seed_manifold(1, 0, 1, 0.f, 42);   // amplitude=0 → vacuum
    return wf;
}

/// Sum of an 8-element array.
static float sum8(const std::array<float, NPT_NUM_HEADS>& a) {
    float s = 0.f;
    for (float v : a) s += v;
    return s;
}

// ---------------------------------------------------------------------------
// §1  inner_product_re: identical fields → equals total_probability
// ---------------------------------------------------------------------------
TEST_CASE("§1 inner_product_re: Re(⟨Q|Q⟩) == total_probability(Q)", "[Phase38][kernel]") {
    // When Q = K, Re(⟨Q|K⟩) = Σᵢ(Q_r² + Q_i²) = |Q|²
    WaveFunction Q;
    Q.seed_manifold(3, 3, 1, 1.f, 42);

    const double ip  = Q.inner_product_re(Q);
    const double prob = Q.total_probability();

    INFO("inner_product_re = " << ip << "  total_probability = " << prob);
    REQUIRE(ip == Approx(prob).epsilon(1e-6));
}

// ---------------------------------------------------------------------------
// §2  inner_product_re: size mismatch → 0.0
// ---------------------------------------------------------------------------
TEST_CASE("§2 inner_product_re: size mismatch returns 0", "[Phase38][kernel]") {
    WaveFunction Q, K;
    Q.seed_manifold(3, 3, 1, 1.f, 42);  // 3^9 = 19683 nodes
    K.seed_manifold(2, 3, 1, 1.f, 42);  // 2^9 =   512 nodes

    REQUIRE(Q.inner_product_re(K) == Approx(0.0).margin(1e-30));
    REQUIRE(K.inner_product_re(Q) == Approx(0.0).margin(1e-30));
}

// ---------------------------------------------------------------------------
// §3  inner_product_re: known value via direct injection on 1-node grid
// ---------------------------------------------------------------------------
TEST_CASE("§3 inner_product_re: known value from injected fields", "[Phase38][kernel]") {
    // Q = (0.3 + 0.0i),  K = (0.1 + 0.2i) at the single node.
    // Re(⟨Q|K⟩) = 0.3·0.1 + 0.0·0.2 = 0.03
    WaveFunction Q = make_single_node_vacuum();
    WaveFunction K = make_single_node_vacuum();

    Q.inject(0, {0.3f, 0.0f}, 1.f);
    K.inject(0, {0.1f, 0.2f}, 1.f);

    const double expected = 0.3 * 0.1 + 0.0 * 0.2;   // = 0.03
    REQUIRE(Q.inner_product_re(K) == Approx(expected).margin(1e-6));
}

// ---------------------------------------------------------------------------
// §4  wave_correlation: in-phase (Q and K identical) → ≈ +1.0
// ---------------------------------------------------------------------------
TEST_CASE("§4 wave_correlation: in-phase fields → +1.0", "[Phase38][correlation]") {
    // Two identically seeded wavefunctions are perfectly in-phase.
    // correlation = 2·Re(⟨Q|Q⟩) / (|Q|² + |Q|² + ε) = 2|Q|² / (2|Q|² + ε) → 1
    WaveFunction Q, K;
    Q.seed_manifold(3, 3, 1, 1.f, 7);
    K.seed_manifold(3, 3, 1, 1.f, 7);   // identical seed → identical psi

    const float corr = wave_correlation(Q, K);
    INFO("in-phase correlation = " << corr);
    REQUIRE(corr == Approx(1.0f).epsilon(1e-4f));
}

// ---------------------------------------------------------------------------
// §5  wave_correlation: anti-phase (K psi negated) → ≈ −1.0
// ---------------------------------------------------------------------------
TEST_CASE("§5 wave_correlation: anti-phase fields → -1.0", "[Phase38][correlation]") {
    // Seeding with amplitude = -1.0 negates the pilot wave → psi_K = -psi_Q.
    // Re(⟨Q|K⟩) = Σ(Q_r·(-Q_r) + Q_i·(-Q_i)) = -|Q|²
    // correlation = 2·(-|Q|²) / (|Q|² + |-Q|² + ε) = -2|Q|²/(2|Q|² + ε) → -1
    WaveFunction Q, K;
    Q.seed_manifold(3, 3, 1,  1.f, 13);
    K.seed_manifold(3, 3, 1, -1.f, 13);  // negated amplitude → psi_K = -psi_Q

    const float corr = wave_correlation(Q, K);
    INFO("anti-phase correlation = " << corr);
    REQUIRE(corr == Approx(-1.0f).epsilon(1e-4f));
}

// ---------------------------------------------------------------------------
// §6  wave_correlation: orthogonal fields → ≈ 0.0
// ---------------------------------------------------------------------------
TEST_CASE("§6 wave_correlation: orthogonal fields → 0.0", "[Phase38][correlation]") {
    // Construct on single-node grid:
    // Q = (0.3 + 0.0i),  K = (0.0 + 0.3i)   ← Re(⟨Q|K⟩) = 0.3·0 + 0·0.3 = 0
    WaveFunction Q = make_single_node_vacuum();
    WaveFunction K = make_single_node_vacuum();

    Q.inject(0, {0.3f, 0.0f}, 1.f);   // purely real
    K.inject(0, {0.0f, 0.3f}, 1.f);   // purely imaginary

    const float corr = wave_correlation(Q, K);
    INFO("orthogonal correlation = " << corr);
    REQUIRE(corr == Approx(0.0f).margin(1e-5f));
}

// ---------------------------------------------------------------------------
// §7  wave_correlation: symmetric
// ---------------------------------------------------------------------------
TEST_CASE("§7 wave_correlation: symmetric — corr(Q,K) == corr(K,Q)", "[Phase38][correlation]") {
    // Re(⟨Q|K⟩) = Re(⟨K|Q⟩), and the denominator is symmetric → must hold.
    WaveFunction Q, K;
    Q.seed_manifold(3, 2, 1, 1.f, 3);
    K.seed_manifold(3, 4, 2, 1.f, 9);

    const float qk = wave_correlation(Q, K);
    const float kq = wave_correlation(K, Q);

    INFO("corr(Q,K) = " << qk << "  corr(K,Q) = " << kq);
    REQUIRE(qk == Approx(kq).margin(1e-5f));
}

// ---------------------------------------------------------------------------
// §8  wave_correlation: both vacuum → 0.0
// ---------------------------------------------------------------------------
TEST_CASE("§8 wave_correlation: both vacuum → 0.0 (defined neutral)", "[Phase38][correlation]") {
    WaveFunction Q = make_single_node_vacuum();
    WaveFunction K = make_single_node_vacuum();

    REQUIRE(wave_correlation(Q, K) == Approx(0.0f).margin(1e-6f));
}

// ---------------------------------------------------------------------------
// §9  attention_softmax: outputs sum to 1.0
// ---------------------------------------------------------------------------
TEST_CASE("§9 attention_softmax: sum of outputs == 1.0", "[Phase38][softmax]") {
    std::array<float, NPT_NUM_HEADS> raw{1.f, -0.5f, 0.f, 0.3f, -1.f, 0.8f, 0.1f, 0.6f};
    const auto out = attention_softmax(raw, 1.0f);
    REQUIRE(sum8(out) == Approx(1.0f).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §10 attention_softmax: ordering preserved
// ---------------------------------------------------------------------------
TEST_CASE("§10 attention_softmax: higher raw score → higher output", "[Phase38][softmax]") {
    // If raw[2] > raw[5], then out[2] > out[5] (monotone transform).
    std::array<float, NPT_NUM_HEADS> raw{0.f, 0.f, 2.0f, 0.f, 0.f, -1.0f, 0.f, 0.f};
    const auto out = attention_softmax(raw, 1.0f);
    INFO("out[2]=" << out[2] << " out[5]=" << out[5]);
    REQUIRE(out[2] > out[5]);
    // Also check that the maximum input maps to the maximum output
    REQUIRE(out[2] == *std::max_element(out.begin(), out.end()));
}

// ---------------------------------------------------------------------------
// §11 attention_softmax: lower temperature → sharper (more peaked) distribution
// ---------------------------------------------------------------------------
TEST_CASE("§11 attention_softmax: low temperature sharpens distribution", "[Phase38][softmax]") {
    // A sharp distribution has higher max value (and lower entropy).
    std::array<float, NPT_NUM_HEADS> raw{1.f, 0.5f, 0.f, -0.5f, -1.f, 0.2f, 0.3f, 0.4f};

    const auto out_hi = attention_softmax(raw, 2.0f);   // diffuse (high τ)
    const auto out_lo = attention_softmax(raw, 0.1f);   // sharp   (low  τ)

    const float max_hi = *std::max_element(out_hi.begin(), out_hi.end());
    const float max_lo = *std::max_element(out_lo.begin(), out_lo.end());

    INFO("max at τ=2.0: " << max_hi << "  max at τ=0.1: " << max_lo);
    REQUIRE(max_lo > max_hi);   // low τ produces a more peaked maximum
}

// ---------------------------------------------------------------------------
// §12 attention_softmax: uniform input → each output ≈ 1/8
// ---------------------------------------------------------------------------
TEST_CASE("§12 attention_softmax: uniform input → 1/8 each", "[Phase38][softmax]") {
    std::array<float, NPT_NUM_HEADS> raw{};
    raw.fill(0.f);   // all equal
    const auto out = attention_softmax(raw, 1.0f);
    const float expected = 1.f / static_cast<float>(NPT_NUM_HEADS);
    for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
        INFO("out[" << i << "] = " << out[i]);
        REQUIRE(out[i] == Approx(expected).epsilon(1e-5f));
    }
}

// ---------------------------------------------------------------------------
// §13 forward(): scores sum to 1.0
// ---------------------------------------------------------------------------
TEST_CASE("§13 NPT forward(): scores are a valid probability distribution", "[Phase38][forward]") {
    NeuroplasticTransformer npt(3);
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);

    auto result = npt.forward(torus_wf);
    const float total = sum8(result.head_scores);

    INFO("sum of head_scores = " << total);
    REQUIRE(total == Approx(1.0f).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §14 forward(): vacuum Q + seeded torus, alpha=0 → uniform 1/8 scores
// ---------------------------------------------------------------------------
TEST_CASE("§14 NPT forward(): vacuum Q + alpha=0 torus → uniform softmax", "[Phase38][forward]") {
    // Disable curvature bias so this test remains a clean regression for
    // the Phase 38 correlation kernel: vacuum Q → all-zero correlations
    // → softmax([0,...,0]) = [1/8,...,1/8] exactly.
    NeuroplasticTransformer npt(3, 1.0f, 0.0f);   // alpha=0 disables bias
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 99);   // seeded (non-vacuum) torus

    auto result = npt.forward(torus_wf);
    const float expected = 1.f / static_cast<float>(NPT_NUM_HEADS);
    for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
        INFO("head_scores[" << i << "] = " << result.head_scores[i]);
        REQUIRE(result.head_scores[i] == Approx(expected).epsilon(1e-5f));
    }
}

// ---------------------------------------------------------------------------
// §15 forward(): has_output == true (Phase 40 heterodyne is live)
// ---------------------------------------------------------------------------
TEST_CASE("§15 NPT forward(): has_output is true from Phase 40", "[Phase38][forward]") {
    NeuroplasticTransformer npt(3);
    WaveFunction torus_wf;
    torus_wf.seed_manifold(3, 3, 1, 1.f, 42);

    auto result = npt.forward(torus_wf);
    REQUIRE(result.has_output == true);
    REQUIRE(result.output.is_finite());
    REQUIRE(result.output.num_nodes() == torus_wf.num_nodes());
}
