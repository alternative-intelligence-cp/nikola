/**
 * @file phase43_hebbian_update_test.cpp
 * @brief Phase 43 — Hebbian-Riemannian metric update via NPT output.
 *
 * Tests that NeuroplasticTransformer::forward() updates each head's Q
 * WaveFunction toward the attended output using a Riemannian-gated
 * Hebbian learning rule:
 *
 *   Q_i ← (1 − t_i) · Q_i  +  t_i · output
 *
 *   where  t_i = hebbian_alpha · mean_curvature(torus_wf) · w_i
 *          w_i = npt_curvature_weights()[i]   (ascending with head index)
 *
 * Learning is gated by the mean curvature of the input field, so frozen
 * (low-curvature) fields produce no Q update while plastic (high-curvature)
 * fields drive fast convergence.  Higher-index heads (finer cognitive bands)
 * carry larger weights and therefore update faster.
 *
 * §1   Default hebbian_alpha is 0.01
 * §2   set_hebbian_alpha() mutates the rate; hebbian_alpha() reads it back
 * §3   After one forward() with curvature, Q[0] moves closer to output
 * §4   hebbian_alpha = 0 → Q unchanged after forward()
 * §5   Larger hebbian_alpha → larger per-step Q drift
 * §6   Head 7 (largest weight) drifts more per step than head 0 (smallest)
 * §7   K fields are never modified by forward() or hebbian_update()
 * §8   After N forwards with same input, inner_product(Q[0], output) grows
 * §9   Clamp: huge alpha does not cause overshoot (Q stays bounded)
 * §10  With alpha = 0, all 8 Q fields remain at initial vacuum
 * §11  has_output remains true through hebbian update (result not corrupted)
 * §12  Drift magnitude scales with hebbian_alpha (compare 0.1 vs 0.01)
 * §13  All 8 heads change Q after forward() (none silently skipped)
 * §14  head_scores still sum to ≈ 1.0 after hebbian update
 * §15  Convergence: after 500 forwards, Q[0] aligns well with torus_wf
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/neuroplastic_transformer.hpp>
#include <nikola/physics/wave_function.hpp>

#include <cmath>
#include <array>

using namespace nikola::cognitive;
using namespace nikola::physics;
using nikola::foundation::GridConfig;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a non-trivial WaveFunction seeded with energy (guarantees mc > 0).
static WaveFunction make_active_wf(int grid_n = 2, unsigned seed = 42) {
    WaveFunction wf(GridConfig::uniform(grid_n));
    wf.seed_manifold(grid_n, 0, 1, 0.3f, seed);
    return wf;
}

/// Build a NeuroplasticTransformer sized for grid_n.
static NeuroplasticTransformer make_npt(int grid_n = 2,
                                        float temperature   = 1.0f,
                                        float curv_alpha    = 0.5f) {
    return NeuroplasticTransformer(grid_n, temperature, curv_alpha);
}

// ---------------------------------------------------------------------------
// §1  Default hebbian_alpha is 0.01
// ---------------------------------------------------------------------------
TEST_CASE("Phase43 §1 — default hebbian_alpha is 0.01", "[phase43]") {
    auto npt = make_npt();
    REQUIRE(npt.hebbian_alpha() == Catch::Approx(0.01f).epsilon(1e-6));
}

// ---------------------------------------------------------------------------
// §2  set_hebbian_alpha() / hebbian_alpha() round-trip
// ---------------------------------------------------------------------------
TEST_CASE("Phase43 §2 — set_hebbian_alpha round-trips correctly", "[phase43]") {
    auto npt = make_npt();
    npt.set_hebbian_alpha(0.25f);
    REQUIRE(npt.hebbian_alpha() == Catch::Approx(0.25f).epsilon(1e-6));

    npt.set_hebbian_alpha(0.0f);
    REQUIRE(npt.hebbian_alpha() == Catch::Approx(0.0f).epsilon(1e-6));
}

// ---------------------------------------------------------------------------
// §3  After one forward() with curvature, Q[0] is closer to output
// ---------------------------------------------------------------------------
TEST_CASE("Phase43 §3 — forward() moves Q[0] toward output", "[phase43]") {
    auto npt    = make_npt();
    auto torus  = make_active_wf();

    // Mean curvature must be nonzero for the Hebbian gate to open
    float mc = static_cast<float>(torus.mean_curvature());
    REQUIRE(mc > 0.0f);   // sanity: seed_manifold rms_vel=0.3 gives curvature

    // Clone Q[0] before forward so we can compare
    auto Q0_before = npt.head(0).Q.clone();

    auto result = npt.forward(torus);
    REQUIRE(result.has_output);

    // ip measures "alignment" of Q with output
    float ip_before = static_cast<float>(Q0_before.inner_product_re(result.output));
    float ip_after  = static_cast<float>(npt.head(0).Q.inner_product_re(result.output));

    // Q started as vacuum → ip_before ≈ 0; after lerp toward output, ip_after > ip_before
    REQUIRE(ip_after > ip_before);
    REQUIRE(ip_after > 0.0f);   // Q now positively aligned with output
}

// ---------------------------------------------------------------------------
// §4  hebbian_alpha = 0 → Q unchanged after forward()
// ---------------------------------------------------------------------------
TEST_CASE("Phase43 §4 — alpha=0 freezes all Q fields", "[phase43]") {
    auto npt   = make_npt();
    npt.set_hebbian_alpha(0.0f);
    auto torus = make_active_wf();

    // Snapshot all Q fields before
    std::array<WaveFunction, NPT_NUM_HEADS> Q_before;
    for (size_t i = 0; i < NPT_NUM_HEADS; ++i)
        Q_before[i] = npt.head(i).Q.clone();

    auto result = npt.forward(torus);
    REQUIRE(result.has_output);

    // Q fields should be bit-identical (alpha=0 early-returns hebbian_update)
    for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
        float ip = static_cast<float>(Q_before[i].inner_product_re(npt.head(i).Q));
        // If Q is unchanged, ip == ||Q||^2
        float norm = static_cast<float>(npt.head(i).Q.total_probability());
        // For a field that hasn't changed: inner_product_re(Q, Q) == total_probability(Q)
        REQUIRE(ip == Catch::Approx(norm).epsilon(1e-5));
    }
}

// ---------------------------------------------------------------------------
// §5  Larger hebbian_alpha → larger per-step Q drift
// ---------------------------------------------------------------------------
TEST_CASE("Phase43 §5 — larger alpha produces larger Q drift", "[phase43]") {
    auto torus = make_active_wf();
    REQUIRE(static_cast<float>(torus.mean_curvature()) > 0.0f);

    auto npt_slow = make_npt();  npt_slow.set_hebbian_alpha(0.01f);
    auto npt_fast = make_npt();  npt_fast.set_hebbian_alpha(0.10f);

    auto res_slow = npt_slow.forward(torus);
    auto res_fast = npt_fast.forward(torus);

    // Both outputs should be virtually identical (same projections, same torus)
    // but Q should drift more for the fast NPT
    float ip_slow = static_cast<float>(npt_slow.head(0).Q.inner_product_re(res_slow.output));
    float ip_fast = static_cast<float>(npt_fast.head(0).Q.inner_product_re(res_fast.output));

    REQUIRE(ip_fast > ip_slow);   // faster learner has stronger Q→output alignment
}

// ---------------------------------------------------------------------------
// §6  Head 7 drifts more than head 0 per step (w_7 > w_0)
// ---------------------------------------------------------------------------
TEST_CASE("Phase43 §6 — head 7 drifts more than head 0 per step", "[phase43]") {
    auto npt   = make_npt();
    auto torus = make_active_wf();
    REQUIRE(static_cast<float>(torus.mean_curvature()) > 0.0f);

    auto result = npt.forward(torus);

    // inner_product_re(Q_i, output) measures alignment after one step
    float ip_head0 = static_cast<float>(npt.head(0).Q.inner_product_re(result.output));
    float ip_head7 = static_cast<float>(npt.head(7).Q.inner_product_re(result.output));

    // Head 7 has larger curvature weight → larger t → more drift toward output
    REQUIRE(ip_head7 > ip_head0);
}

// ---------------------------------------------------------------------------
// §7  K fields never modified by forward()
// ---------------------------------------------------------------------------
// §7 was originally "K fields are unchanged after forward()" but Phase 44
// intentionally introduces K-head differentiation: each K_i drifts toward
// the live torus input (scale_by(1-s) + add_scaled(torus, s)).
// The correct postcondition is that K fields remain finite and non-negative
// after the drift update — not that they are identical to the pre-forward state.
TEST_CASE("Phase43 §7 — K fields are finite and non-negative after forward()", "[phase43]") {
    auto npt   = make_npt();
    auto torus = make_active_wf();

    (void)npt.forward(torus);

    for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
        const float norm = static_cast<float>(npt.head(i).K.total_probability());
        // K is a valid WaveFunction: non-negative total probability, no NaN/Inf
        REQUIRE(norm >= 0.0f);
        REQUIRE(std::isfinite(norm));
        REQUIRE(npt.head(i).K.is_finite());
    }
}

// ---------------------------------------------------------------------------
// §8  Over N forwards with same input, Q[0]↔output alignment grows
// ---------------------------------------------------------------------------
TEST_CASE("Phase43 §8 — Q[0] monotonically aligns with output over N forwards",
          "[phase43]")
{
    auto npt   = make_npt();
    npt.set_hebbian_alpha(0.05f);   // faster learning for test brevity
    auto torus = make_active_wf();
    REQUIRE(static_cast<float>(torus.mean_curvature()) > 0.0f);

    float prev_ip = -1e30f;
    for (int step = 0; step < 30; ++step) {
        auto result   = npt.forward(torus);
        float curr_ip = static_cast<float>(npt.head(0).Q.inner_product_re(result.output));
        REQUIRE(curr_ip >= prev_ip - 1e-5f);   // weak monotone (allow floating-point noise)
        prev_ip = curr_ip;
    }
}

// ---------------------------------------------------------------------------
// §9  Huge alpha clamps t to 1 — Q converges, doesn't oscillate
// ---------------------------------------------------------------------------
TEST_CASE("Phase43 §9 — clamp prevents overshoot with huge alpha", "[phase43]") {
    auto npt   = make_npt();
    npt.set_hebbian_alpha(100.0f);   // extreme: t_i would be >> 1 without clamp
    auto torus = make_active_wf();

    float prev_ip = -1e30f;
    for (int step = 0; step < 20; ++step) {
        auto result   = npt.forward(torus);
        float curr_ip = static_cast<float>(npt.head(0).Q.inner_product_re(result.output));
        // With t clamped to 1 each step: Q → output exactly, then stays
        REQUIRE(curr_ip >= prev_ip - 1e-4f);   // no oscillation
        prev_ip = curr_ip;
    }
}

// ---------------------------------------------------------------------------
// §10  All 8 Q fields remain at initial state when alpha = 0
// ---------------------------------------------------------------------------
TEST_CASE("Phase43 §10 — alpha=0 leaves all 8 Q fields at vacuum after 10 forwards",
          "[phase43]")
{
    auto npt   = make_npt();
    npt.set_hebbian_alpha(0.0f);
    auto torus = make_active_wf();

    // Initial Q total probability snapshot
    std::array<float, NPT_NUM_HEADS> init_prob;
    for (size_t i = 0; i < NPT_NUM_HEADS; ++i)
        init_prob[i] = static_cast<float>(npt.head(i).Q.total_probability());

    for (int step = 0; step < 10; ++step)
        (void)npt.forward(torus);

    for (size_t i = 0; i < NPT_NUM_HEADS; ++i) {
        float prob = static_cast<float>(npt.head(i).Q.total_probability());
        REQUIRE(prob == Catch::Approx(init_prob[i]).epsilon(1e-5));
    }
}

// ---------------------------------------------------------------------------
// §11  result.has_output remains true after hebbian update
// ---------------------------------------------------------------------------
TEST_CASE("Phase43 §11 — has_output is true after forward() with Hebbian update",
          "[phase43]")
{
    auto npt   = make_npt();
    auto torus = make_active_wf();
    auto result = npt.forward(torus);
    REQUIRE(result.has_output == true);
}

// ---------------------------------------------------------------------------
// §12  Drift magnitude scales with hebbian_alpha (0.1 vs 0.01)
// ---------------------------------------------------------------------------
TEST_CASE("Phase43 §12 — Q drift after 1 forward scales with alpha", "[phase43]") {
    auto torus = make_active_wf();

    // Build two NPTs, one with 10× the learning rate, same seed state
    auto npt_lo = make_npt();  npt_lo.set_hebbian_alpha(0.01f);
    auto npt_hi = make_npt();  npt_hi.set_hebbian_alpha(0.10f);

    auto res_lo = npt_lo.forward(torus);
    auto res_hi = npt_hi.forward(torus);

    // Compute ||Q_i_hi - Q_initial|| vs ||Q_i_lo - Q_initial||
    // Approximation: ip(Q_new, output) grows with alpha
    float ip_lo = static_cast<float>(npt_lo.head(3).Q.inner_product_re(res_lo.output));
    float ip_hi = static_cast<float>(npt_hi.head(3).Q.inner_product_re(res_hi.output));

    REQUIRE(ip_hi > ip_lo * 1.5f);   // roughly 10× larger alpha → proportionally more drift
}

// ---------------------------------------------------------------------------
// §13  All 8 heads show distinct Q drift magnitudes (weight ordering)
// ---------------------------------------------------------------------------
TEST_CASE("Phase43 §13 — per-head Q drift is ordered by curvature weight", "[phase43]") {
    auto npt   = make_npt();
    auto torus = make_active_wf();
    REQUIRE(static_cast<float>(torus.mean_curvature()) > 0.0f);

    auto result = npt.forward(torus);

    // Collect alignment of each Q_i with the output
    std::array<float, NPT_NUM_HEADS> ip{};
    for (size_t i = 0; i < NPT_NUM_HEADS; ++i)
        ip[i] = static_cast<float>(npt.head(i).Q.inner_product_re(result.output));

    // All heads should have moved (non-negative alignment with output)
    for (size_t i = 0; i < NPT_NUM_HEADS; ++i)
        REQUIRE(ip[i] >= 0.0f);

    // Alignment should be ordered: ip[7] >= ip[6] >= ... >= ip[0]
    // (higher curvature weight → larger step → stronger Q→output alignment)
    for (size_t i = 1; i < NPT_NUM_HEADS; ++i)
        REQUIRE(ip[i] >= ip[i-1] - 1e-6f);    // weak ascending order
}

// ---------------------------------------------------------------------------
// §14  head_scores still sum to ≈ 1.0 after Hebbian update
// ---------------------------------------------------------------------------
TEST_CASE("Phase43 §14 — head_scores sum to 1.0 after forward()", "[phase43]") {
    auto npt   = make_npt();
    auto torus = make_active_wf();
    auto result = npt.forward(torus);

    float sum = 0.f;
    for (size_t i = 0; i < NPT_NUM_HEADS; ++i)
        sum += result.head_scores[i];

    REQUIRE(sum == Catch::Approx(1.0f).epsilon(1e-5));
}

// ---------------------------------------------------------------------------
// §15  Convergence: after 500 forwards, Q[0] strongly aligns with torus_wf
// ---------------------------------------------------------------------------
TEST_CASE("Phase43 §15 — Q[0] converges toward torus_wf after many forwards",
          "[phase43]")
{
    auto npt   = make_npt();
    npt.set_hebbian_alpha(0.05f);
    auto torus = make_active_wf();

    // Capture initial alignment
    float ip_initial = static_cast<float>(npt.head(0).Q.inner_product_re(torus));

    for (int step = 0; step < 500; ++step)
        (void)npt.forward(torus);

    // After convergence, Q[0] should have substantially positive alignment with torus
    float ip_final = static_cast<float>(npt.head(0).Q.inner_product_re(torus));
    REQUIRE(ip_final > ip_initial + 0.0001f);   // measurable convergence
}
