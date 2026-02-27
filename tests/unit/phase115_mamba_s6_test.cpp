/**
 * @file phase115_mamba_s6_test.cpp
 * @brief Phase 115 — Mamba S6 Selective Scan upgrade for CognitiveCore.
 *
 * Tests the input-dependent (selective) state-space model extension added to
 * SSMLayer in cognitive_core.hpp.  The S6 scan adds:
 *
 *   W_delta  (H×I) — input → per-element time-step Δ  (softplus-activated)
 *   W_Bsel   (H×I) — input → selective-B vector
 *
 * New API:
 *   ssm.randomise_selective(seed)
 *   ssm.selective_step(h, u)
 *   ssm.W_delta()   / ssm.W_Bsel()
 *
 * Tests covered:
 *   1.  W_delta and W_Bsel are zero-initialised at construction.
 *   2.  W_delta and W_Bsel have correct size (H×I).
 *   3.  randomise_selective() makes the projections non-zero.
 *   4.  selective_step: zero projections → step has no selective drive
 *       (only Ā·h term survives, = exp(Δ·A)·h_t with Δ=softplus(0)=log2).
 *   5.  selective_step: nonzero W_Bsel + nonzero input → state changes.
 *   6.  Selectivity core property: two identical initial states fed DIFFERENT
 *       inputs diverge after selective_step (but NOT after update_state with
 *       the same fixed B weights).
 *   7.  Selectivity consistency: same state + same input → same result.
 *   8.  selective_step: dimension mismatch throws std::invalid_argument.
 *   9.  selective_step: numerical stability — 50 iterations never NaN/Inf.
 *  10.  selective_step: composable with compute_output (output changes).
 *  11.  ZOH property: A_i=0 limit — B̄ = Δ·b_t (no divide-by-zero).
 *  12.  Ā clamping: positive A values still produce Ā ≤ 1 (no runaway).
 *  13.  selective_step: zero input → state decays toward zero
 *       (Ā·h with Ā < 1, B̄ ≈ 0).
 *  14.  CognitiveCore exposes ssm() which supports selective_step.
 *  15.  selective_step and update_state co-exist (both callable on same SSM).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/cognitive/cognitive_core.hpp>

#include <array>
#include <cmath>
#include <numeric>
#include <vector>

using namespace nikola::cognitive;
using nikola::foundation::TORUS_DIMS;   // 9

// Helper: make a 9D input array filled with a constant
static std::array<float, TORUS_DIMS> const_input(float v) {
    std::array<float, TORUS_DIMS> u{};
    u.fill(v);
    return u;
}

// Helper: L2 distance between two states
static float state_dist(const SSMLayer::State& a, const SSMLayer::State& b) {
    float d = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff = a[i] - b[i];
        d += diff * diff;
    }
    return std::sqrt(d);
}

// Helper: check no component is NaN or Inf
static bool state_finite(const SSMLayer::State& h) {
    for (float v : h)
        if (!std::isfinite(v)) return false;
    return true;
}

// ============================================================================
// Test 1: W_delta and W_Bsel zero-initialised at construction
// ============================================================================

TEST_CASE("S6: W_delta and W_Bsel are zero at construction",
          "[phase115][s6][gap3.2]") {
    SSMLayer ssm(16, 9, 5);

    for (float v : ssm.W_delta())
        REQUIRE(v == 0.f);
    for (float v : ssm.W_Bsel())
        REQUIRE(v == 0.f);
}

// ============================================================================
// Test 2: W_delta and W_Bsel have correct size (H×I)
// ============================================================================

TEST_CASE("S6: W_delta and W_Bsel have correct size H×I",
          "[phase115][s6][gap3.2]") {
    const int H = 32, I = 9, O = 7;
    SSMLayer ssm(H, I, O);

    REQUIRE(static_cast<int>(ssm.W_delta().size()) == H * I);
    REQUIRE(static_cast<int>(ssm.W_Bsel().size())  == H * I);
}

// ============================================================================
// Test 3: randomise_selective() populates W_delta and W_Bsel
// ============================================================================

TEST_CASE("S6: randomise_selective makes projections non-zero",
          "[phase115][s6][gap3.2]") {
    SSMLayer ssm(16, 9, 5);
    ssm.randomise_selective(101u);

    const auto& wd = ssm.W_delta();
    const auto& wb = ssm.W_Bsel();

    // At least one element must be non-zero after initialisation
    const float norm_wd = std::inner_product(wd.begin(), wd.end(), wd.begin(), 0.f);
    const float norm_wb = std::inner_product(wb.begin(), wb.end(), wb.begin(), 0.f);

    CHECK(norm_wd > 0.f);
    CHECK(norm_wb > 0.f);
}

// ============================================================================
// Test 4: zero projections → selective_step only contracts via Ā
// ============================================================================

TEST_CASE("S6: zero W_delta/W_Bsel → state decays by factor Ā",
          "[phase115][s6]") {
    // With W_delta = 0: raw_Δ = 0, Δ = softplus(0) = log(2) ≈ 0.693
    // With A_i = 0.5: Ā_i = exp(0.693 * 0.5) ≈ exp(0.347) ≈ 1.41 — but clamped to 1!
    // Actually softplus(0) = log(2), A=0.5 positive →
    //   Ā = clamp(exp(softplus(0)*0.5), 0, 1) = clamp(>1, 0, 1) = 1
    // → state gets multiplied by 1.0 (no contraction for positive A, clamped)
    //
    // Use A_i = -0.5: Ā = clamp(exp(-0.347), 0, 1) ≈ 0.707 < 1 (contraction)
    // W_Bsel = 0 → b_t_i = 0 → B̄_i = 0
    // So h_new_i = 0.707 * h_i

    SSMLayer ssm(8, 9, 5);
    ssm.set_uniform_A(-0.5f);   // A_i = -0.5 for all i

    // W_delta and W_Bsel remain zero (not called randomise_selective)

    SSMLayer::State h(8, 1.f);  // h_i = 1 for all i
    const auto u = const_input(1.f);

    ssm.selective_step(h, u);

    // Δ_i = softplus(0) = log(2) ≈ 0.693147
    const float delta    = std::log(2.f);
    const float a_bar    = std::clamp(std::exp(delta * -0.5f), 0.f, 1.f);
    // ≈ exp(-0.347) ≈ 0.707

    for (float v : h) {
        REQUIRE_THAT(v, Catch::Matchers::WithinAbs(a_bar, 1e-5f));
    }
}

// ============================================================================
// Test 5: nonzero W_Bsel + nonzero input → state changes from zero
// ============================================================================

TEST_CASE("S6: nonzero W_Bsel drives state from zero initial state",
          "[phase115][s6]") {
    SSMLayer ssm(8, 9, 5);
    ssm.set_uniform_A(-0.9f);

    // Set W_Bsel so row i, col 0 = 1.0, rest 0
    auto& wb = ssm.W_Bsel();
    for (int i = 0; i < 8; ++i)
        wb[static_cast<size_t>(i * 9)] = 1.0f;   // first column

    SSMLayer::State h = ssm.make_zero_state();
    const auto u = const_input(0.5f);   // all inputs = 0.5

    ssm.selective_step(h, u);

    // b_t_i = W_Bsel[i,0]*0.5 + ... = 0.5
    // B̄_i should be nonzero → h not all zeros
    bool any_nonzero = false;
    for (float v : h)
        if (std::abs(v) > 1e-7f) { any_nonzero = true; break; }

    CHECK(any_nonzero);
}

// ============================================================================
// Test 6: Selectivity core property — divergence under different inputs
// ============================================================================

TEST_CASE("S6: different inputs cause state divergence (selective property)",
          "[phase115][s6]") {
    // Two SSMs with identical weights and starting from same initial state.
    // Feed them DIFFERENT inputs.  After selective_step, states must differ.
    // (After update_state with same B, states differ too since B·u depends on u
    //  — this checks the selective path specifically.)

    SSMLayer ssm(16, 9, 5);
    ssm.randomise(42u);
    ssm.randomise_selective(42u);

    // Identical initial states
    SSMLayer::State hA = ssm.make_zero_state();
    SSMLayer::State hB = ssm.make_zero_state();

    hA.assign(hA.size(), 0.3f);   // start non-zero so Ā·h also differs
    hB = hA;                       // exact copy

    const auto uA = const_input(1.0f);
    const auto uB = const_input(-1.0f);   // opposite sign input

    ssm.selective_step(hA, uA);
    ssm.selective_step(hB, uB);

    // States must have diverged
    CHECK(state_dist(hA, hB) > 1e-4f);
}

// ============================================================================
// Test 7: Same state + same input → reproducible (deterministic)
// ============================================================================

TEST_CASE("S6: selective_step is deterministic (same input → same state)",
          "[phase115][s6]") {
    SSMLayer ssm(16, 9, 5);
    ssm.randomise(7u);
    ssm.randomise_selective(7u);

    SSMLayer::State h1(16, 0.5f);
    SSMLayer::State h2 = h1;

    const auto u = const_input(0.3f);

    ssm.selective_step(h1, u);
    ssm.selective_step(h2, u);

    for (size_t i = 0; i < h1.size(); ++i)
        REQUIRE(h1[i] == h2[i]);
}

// ============================================================================
// Test 8: Dimension mismatch throws
// ============================================================================

TEST_CASE("S6: selective_step throws on h dimension mismatch",
          "[phase115][s6]") {
    SSMLayer ssm(16, 9, 5);
    SSMLayer::State h_bad(8, 0.f);   // wrong size (8 ≠ 16)
    const auto u = const_input(0.f);

    REQUIRE_THROWS_AS(ssm.selective_step(h_bad, u), std::invalid_argument);
}

// ============================================================================
// Test 9: Numerical stability — 50 iterations, no NaN / Inf
// ============================================================================

TEST_CASE("S6: selective_step remains finite after 50 iterations",
          "[phase115][s6]") {
    const int H = 64, I = 9, O = 5;
    SSMLayer ssm(H, I, O);
    ssm.randomise(42u);
    ssm.randomise_selective(42u);

    SSMLayer::State h = ssm.make_zero_state();

    std::array<float, TORUS_DIMS> u{};
    for (int t = 0; t < 50; ++t) {
        // Vary input each step
        for (int k = 0; k < I; ++k)
            u[static_cast<size_t>(k)] = std::sin(static_cast<float>(t + k));

        ssm.selective_step(h, u);

        REQUIRE(state_finite(h));
    }
}

// ============================================================================
// Test 10: Composable with compute_output — output changes after selective step
// ============================================================================

TEST_CASE("S6: selective_step + compute_output produces non-zero output",
          "[phase115][s6]") {
    SSMLayer ssm(16, 9, 10);
    ssm.randomise(99u);
    ssm.randomise_selective(99u);

    SSMLayer::State h = ssm.make_zero_state();
    const auto u = const_input(1.0f);

    ssm.selective_step(h, u);

    std::vector<float> y;
    ssm.compute_output(h, y);

    REQUIRE(static_cast<int>(y.size()) == 10);

    float norm_y = 0.f;
    for (float v : y) norm_y += v * v;
    CHECK(norm_y > 0.f);
}

// ============================================================================
// Test 11: ZOH limit — A_i = 0 → B̄ = Δ · b_t (no divide-by-zero)
// ============================================================================

TEST_CASE("S6: ZOH with A=0 uses limit formula without division error",
          "[phase115][s6]") {
    SSMLayer ssm(4, 9, 3);
    // A stays zero (as initialised)

    // Set W_Bsel: row 0 col 0 = 2.0, everything else 0
    ssm.W_Bsel()[0] = 2.0f;

    SSMLayer::State h = ssm.make_zero_state();
    const auto u = const_input(1.0f);

    // Should not throw or produce NaN
    REQUIRE_NOTHROW(ssm.selective_step(h, u));
    REQUIRE(state_finite(h));

    // h[0] = Ā·0 + B̄ = 0 + Δ·b_t_0
    // b_t_0 = W_Bsel[0,0]*u[0] = 2.0 * 1.0 = 2.0
    // Δ = softplus(0) = log(2) ≈ 0.6931
    // B̄ = Δ * 2.0 ≈ 1.3863
    const float expected = std::log(2.f) * 2.0f;
    REQUIRE_THAT(h[0], Catch::Matchers::WithinAbs(expected, 1e-5f));
}

// ============================================================================
// Test 12: Positive A clamping — Ā ≤ 1 and state stays bounded
// ============================================================================

TEST_CASE("S6: positive A values are clamped so Ā ≤ 1 (no runaway)",
          "[phase115][s6]") {
    SSMLayer ssm(8, 9, 4);
    // Set A_i = +0.95 (positive, would mean Ā = exp(Δ*0.95) > 1 without clamp)
    for (float& v : ssm.A()) v = 0.95f;

    // Nonzero W_Bsel to ensure there is a drive
    for (float& v : ssm.W_Bsel()) v = 0.5f;

    SSMLayer::State h(8, 1.f);
    const auto u = const_input(1.f);

    // Run 20 steps — state must not diverge
    for (int t = 0; t < 20; ++t)
        ssm.selective_step(h, u);

    // If clamping works, all |h_i| are finite
    REQUIRE(state_finite(h));

    // Empirical bound: each Ā ≤ 1, so |h_i| is driven by B̄ accumulation only.
    // Check it doesn't explode into millions.
    for (float v : h)
        CHECK(std::abs(v) < 1e5f);
}

// ============================================================================
// Test 13: Zero input → state decays toward zero (no selective drive)
// ============================================================================

TEST_CASE("S6: zero input → state decays (Ā·h with B̄=0 since b_t=0)",
          "[phase115][s6]") {
    SSMLayer ssm(16, 9, 5);
    ssm.randomise_selective(5u);
    ssm.set_uniform_A(-0.5f);

    SSMLayer::State h(16, 1.f);   // initial state = 1 everywhere
    const auto u_zero = const_input(0.f);

    const float initial_norm = SSMLayer::state_norm(h);

    for (int t = 0; t < 20; ++t)
        ssm.selective_step(h, u_zero);

    const float final_norm = SSMLayer::state_norm(h);

    // With zero input b_t = 0 → B̄ = 0, so state is just Ā^20 · h_0.
    // With A = -0.5: Ā = exp(softplus(0)*-0.5) = exp(-log2*0.5) ≈ 0.707 per step.
    // After 20 steps: ≈ 0.707^20 < 0.001 — state must have shrunk significantly.
    CHECK(final_norm < initial_norm * 0.5f);
}

// ============================================================================
// Test 14: CognitiveCore exposes ssm() that supports selective_step
// ============================================================================

TEST_CASE("S6: CognitiveCore::ssm() supports selective_step",
          "[phase115][s6][cogncore]") {
    CognitiveCore brain(32, 9, 10, 1u);
    brain.ssm().randomise(1u);
    brain.ssm().randomise_selective(1u);

    auto h = brain.ssm().make_zero_state();
    const auto u = const_input(0.5f);

    REQUIRE_NOTHROW(brain.ssm().selective_step(h, u));
    REQUIRE(state_finite(h));
}

// ============================================================================
// Test 15: selective_step and update_state co-exist on same SSMLayer
// ============================================================================

TEST_CASE("S6: selective_step and update_state are both callable on same SSM",
          "[phase115][s6]") {
    SSMLayer ssm(16, 9, 5);
    ssm.randomise(42u);
    ssm.randomise_selective(42u);

    SSMLayer::State h_classic  = ssm.make_zero_state();
    SSMLayer::State h_selective = ssm.make_zero_state();

    const auto u = const_input(0.7f);

    // Use classic path on one state, selective path on the other
    ssm.update_state(h_classic, u);
    ssm.selective_step(h_selective, u);

    // They should differ (different algorithms)
    CHECK(state_dist(h_classic, h_selective) > 0.f);

    // Both should be finite
    REQUIRE(state_finite(h_classic));
    REQUIRE(state_finite(h_selective));
}
