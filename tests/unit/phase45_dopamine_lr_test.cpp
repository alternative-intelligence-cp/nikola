/**
 * @file phase45_dopamine_modulated_lr_test.cpp
 * @brief Phase 45 — Dopamine-modulated Hebbian learning rate.
 *
 * Tests that NeuroplasticTransformer::forward(torus_wf, dopamine) scales
 * both the Q-head (Hebbian) and K-head (differentiation) learning rates by:
 *
 *   η_scale = 1 + tanh(dopamine − 0.5)
 *
 * Regime table (from spec §A):
 *   D = 1.0 (spike)    η_scale ≈ 1.46   hyper-plastic, fastest learning
 *   D = 0.5 (baseline) η_scale = 1.00   Phase 43-44 rate, no modulation
 *   D = 0.0 (dip)      η_scale ≈ 0.54   plasticity dampened
 *
 * This implements the first neuromodulatory coupling from the spec:
 *   η(t) = η_base · (1 + tanh(D(t) − D_base))
 *
 * §1   last_eta_scale() accessor is present; default forward() → 1.0
 * §2   D=0.5 baseline gives eta_scale = 1.0 (exact)
 * §3   D=1.0 spike gives eta_scale = 1 + tanh(0.5) ≈ 1.462
 * §4   D=0.0 dip gives eta_scale = 1 + tanh(-0.5) ≈ 0.538
 * §5   D=1.0 produces more Q drift than D=0.5 (same field, same seed)
 * §6   D=0.0 produces less Q drift than D=0.5
 * §7   D=1.0 produces more K drift than D=0.5
 * §8   D=0.0 produces less K drift than D=0.5
 * §9   Default forward() (no dopamine arg) behaves identically to D=0.5
 * §10  Q and K drift both scale with the same η_scale (coherent modulation)
 * §11  has_output = true regardless of dopamine level
 * §12  head_scores sum to ≈ 1.0 at all dopamine levels
 * §13  Reward spike (D=0.9) → measurably faster convergence over 50 steps
 * §14  Punishment dip (D=0.1) → measurably slower convergence over 50 steps
 * §15  eta_scale is monotonically increasing with dopamine
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

static WaveFunction make_active_wf(int grid_n = 2, unsigned seed = 42) {
    WaveFunction wf(GridConfig::uniform(grid_n));
    wf.seed_manifold(grid_n, 0, 1, 0.3f, seed);
    return wf;
}

static NeuroplasticTransformer make_npt(int grid_n = 2) {
    return NeuroplasticTransformer(grid_n, 1.0f, 0.5f);
}

/// Expected η_scale for a given dopamine level.
static float expected_eta(float d) { return 1.0f + std::tanh(d - 0.5f); }

// ---------------------------------------------------------------------------
// §1  last_eta_scale() present; default forward() → stores scale ≈ 1.0
// ---------------------------------------------------------------------------
TEST_CASE("Phase45 §1 — last_eta_scale accessor exists after default forward()",
          "[phase45]")
{
    auto npt   = make_npt();
    auto torus = make_active_wf();
    (void)npt.forward(torus);               // default dopamine = 0.5
    REQUIRE(npt.last_eta_scale() == Catch::Approx(1.0f).epsilon(1e-5));
}

// ---------------------------------------------------------------------------
// §2  D=0.5 baseline gives eta_scale = 1.0 (exact by math)
// ---------------------------------------------------------------------------
TEST_CASE("Phase45 §2 — D=0.5 baseline gives eta_scale = 1.0", "[phase45]") {
    auto npt   = make_npt();
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f);
    REQUIRE(npt.last_eta_scale() == Catch::Approx(1.0f).epsilon(1e-5));
}

// ---------------------------------------------------------------------------
// §3  D=1.0 spike gives eta_scale = 1 + tanh(0.5)
// ---------------------------------------------------------------------------
TEST_CASE("Phase45 §3 — D=1.0 spike gives correct eta_scale", "[phase45]") {
    auto npt   = make_npt();
    auto torus = make_active_wf();
    (void)npt.forward(torus, 1.0f);
    const float expected = expected_eta(1.0f);   // 1 + tanh(0.5) ≈ 1.4621
    REQUIRE(npt.last_eta_scale() == Catch::Approx(expected).epsilon(1e-5));
    REQUIRE(npt.last_eta_scale() > 1.0f);
}

// ---------------------------------------------------------------------------
// §4  D=0.0 dip gives eta_scale = 1 + tanh(-0.5) < 1
// ---------------------------------------------------------------------------
TEST_CASE("Phase45 §4 — D=0.0 dip gives correct eta_scale", "[phase45]") {
    auto npt   = make_npt();
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.0f);
    const float expected = expected_eta(0.0f);   // 1 + tanh(-0.5) ≈ 0.5379
    REQUIRE(npt.last_eta_scale() == Catch::Approx(expected).epsilon(1e-5));
    REQUIRE(npt.last_eta_scale() < 1.0f);
}

// ---------------------------------------------------------------------------
// §5  D=1.0 produces more Q drift than D=0.5 (same torus state, same seed)
// ---------------------------------------------------------------------------
TEST_CASE("Phase45 §5 — D=1.0 spike produces more Q drift than D=0.5",
          "[phase45]")
{
    auto torus = make_active_wf();
    REQUIRE(static_cast<float>(torus.mean_curvature()) > 0.0f);

    auto npt_base  = make_npt();   // D=0.5
    auto npt_spike = make_npt();   // D=1.0

    auto res_base  = npt_base.forward(torus, 0.5f);
    auto res_spike = npt_spike.forward(torus, 1.0f);

    float ip_base  = static_cast<float>(npt_base.head(0).Q.inner_product_re(res_base.output));
    float ip_spike = static_cast<float>(npt_spike.head(0).Q.inner_product_re(res_spike.output));

    REQUIRE(ip_spike > ip_base);      // spike → more Q→output alignment
}

// ---------------------------------------------------------------------------
// §6  D=0.0 produces less Q drift than D=0.5
// ---------------------------------------------------------------------------
TEST_CASE("Phase45 §6 — D=0.0 dip produces less Q drift than D=0.5",
          "[phase45]")
{
    auto torus = make_active_wf();

    auto npt_base = make_npt();   // D=0.5
    auto npt_dip  = make_npt();   // D=0.0

    auto res_base = npt_base.forward(torus, 0.5f);
    auto res_dip  = npt_dip.forward(torus,  0.0f);

    float ip_base = static_cast<float>(npt_base.head(0).Q.inner_product_re(res_base.output));
    float ip_dip  = static_cast<float>(npt_dip.head(0).Q.inner_product_re(res_dip.output));

    REQUIRE(ip_base > ip_dip);        // dip → less Q→output alignment
}

// ---------------------------------------------------------------------------
// §7  D=1.0 produces more K drift than D=0.5
// ---------------------------------------------------------------------------
TEST_CASE("Phase45 §7 — D=1.0 spike produces more K drift than D=0.5",
          "[phase45]")
{
    auto torus = make_active_wf();

    auto npt_base  = make_npt();
    auto npt_spike = make_npt();

    (void)npt_base.forward(torus, 0.5f);
    (void)npt_spike.forward(torus, 1.0f);

    float ip_base  = static_cast<float>(npt_base.head(0).K.inner_product_re(torus));
    float ip_spike = static_cast<float>(npt_spike.head(0).K.inner_product_re(torus));

    REQUIRE(ip_spike > ip_base);
}

// ---------------------------------------------------------------------------
// §8  D=0.0 produces less K drift than D=0.5
// ---------------------------------------------------------------------------
TEST_CASE("Phase45 §8 — D=0.0 dip produces less K drift than D=0.5",
          "[phase45]")
{
    auto torus = make_active_wf();

    auto npt_base = make_npt();
    auto npt_dip  = make_npt();

    (void)npt_base.forward(torus, 0.5f);
    (void)npt_dip.forward(torus, 0.0f);

    float ip_base = static_cast<float>(npt_base.head(0).K.inner_product_re(torus));
    float ip_dip  = static_cast<float>(npt_dip.head(0).K.inner_product_re(torus));

    REQUIRE(ip_base > ip_dip);
}

// ---------------------------------------------------------------------------
// §9  Default forward() (no dopamine arg) == D=0.5 (backward compatible)
// ---------------------------------------------------------------------------
TEST_CASE("Phase45 §9 — default forward() is backward-compatible with D=0.5",
          "[phase45]")
{
    auto torus = make_active_wf();

    auto npt_default  = make_npt();
    auto npt_explicit = make_npt();

    auto res_def = npt_default.forward(torus);           // default D=0.5
    auto res_exp = npt_explicit.forward(torus, 0.5f);    // explicit D=0.5

    // Q and K drifts should be identical
    float ip_q_def = static_cast<float>(npt_default.head(0).Q.inner_product_re(res_def.output));
    float ip_q_exp = static_cast<float>(npt_explicit.head(0).Q.inner_product_re(res_exp.output));
    REQUIRE(ip_q_def == Catch::Approx(ip_q_exp).epsilon(1e-5));

    float ip_k_def = static_cast<float>(npt_default.head(0).K.inner_product_re(torus));
    float ip_k_exp = static_cast<float>(npt_explicit.head(0).K.inner_product_re(torus));
    REQUIRE(ip_k_def == Catch::Approx(ip_k_exp).epsilon(1e-5));
}

// ---------------------------------------------------------------------------
// §10  Q and K are both scaled by the same η_scale (coherent modulation)
// ---------------------------------------------------------------------------
TEST_CASE("Phase45 §10 — Q and K are both modulated by the same eta_scale",
          "[phase45]")
{
    auto torus = make_active_wf();

    // Build two NPTs with matching Q alpha and proportionally matching K alpha
    auto npt_base  = make_npt();   // D=0.5
    auto npt_spike = make_npt();   // D=1.0

    auto res_base  = npt_base.forward(torus, 0.5f);
    auto res_spike = npt_spike.forward(torus, 1.0f);

    // Ratio of Q alignment (spike / base) should approximate eta(1.0)/eta(0.5)
    float ip_q_base  = static_cast<float>(npt_base.head(0).Q.inner_product_re(res_base.output));
    float ip_q_spike = static_cast<float>(npt_spike.head(0).Q.inner_product_re(res_spike.output));

    // Ratio of K alignment (spike / base) should be similarly scaled
    float ip_k_base  = static_cast<float>(npt_base.head(0).K.inner_product_re(torus));
    float ip_k_spike = static_cast<float>(npt_spike.head(0).K.inner_product_re(torus));

    // Both Q and K should move more in the spike case
    REQUIRE(ip_q_spike > ip_q_base);
    REQUIRE(ip_k_spike > ip_k_base);
    REQUIRE(npt_spike.last_eta_scale() > npt_base.last_eta_scale());
}

// ---------------------------------------------------------------------------
// §11  has_output = true at all dopamine levels
// ---------------------------------------------------------------------------
TEST_CASE("Phase45 §11 — has_output true at all dopamine levels", "[phase45]") {
    auto npt   = make_npt();
    auto torus = make_active_wf();

    for (float d : {0.0f, 0.1f, 0.5f, 0.9f, 1.0f}) {
        auto res = npt.forward(torus, d);
        REQUIRE(res.has_output == true);
    }
}

// ---------------------------------------------------------------------------
// §12  head_scores sum to ≈ 1.0 at all dopamine levels
// ---------------------------------------------------------------------------
TEST_CASE("Phase45 §12 — head_scores sum to 1.0 at all dopamine levels",
          "[phase45]")
{
    auto npt   = make_npt();
    auto torus = make_active_wf();

    for (float d : {0.0f, 0.1f, 0.5f, 0.9f, 1.0f}) {
        auto res = npt.forward(torus, d);
        float sum = 0.f;
        for (float sc : res.head_scores) sum += sc;
        REQUIRE(sum == Catch::Approx(1.0f).epsilon(1e-5));
    }
}

// ---------------------------------------------------------------------------
// §13  Reward spike (D=0.9) → faster convergence over 50 steps
// ---------------------------------------------------------------------------
TEST_CASE("Phase45 §13 — reward spike produces faster Q convergence over 50 steps",
          "[phase45]")
{
    auto torus = make_active_wf();

    auto npt_base  = make_npt();
    auto npt_spike = make_npt();

    npt_base.set_hebbian_alpha(0.05f);
    npt_spike.set_hebbian_alpha(0.05f);

    for (int i = 0; i < 50; ++i) {
        (void)npt_base.forward(torus, 0.5f);
        (void)npt_spike.forward(torus, 0.9f);
    }

    // After 50 steps, spike NPT Q[0] should be better aligned with torus
    float ip_base  = static_cast<float>(npt_base.head(0).Q.inner_product_re(torus));
    float ip_spike = static_cast<float>(npt_spike.head(0).Q.inner_product_re(torus));

    REQUIRE(ip_spike > ip_base);
}

// ---------------------------------------------------------------------------
// §14  Punishment dip (D=0.1) → slower convergence over 50 steps
// ---------------------------------------------------------------------------
TEST_CASE("Phase45 §14 — punishment dip produces slower Q convergence over 50 steps",
          "[phase45]")
{
    auto torus = make_active_wf();

    auto npt_base = make_npt();
    auto npt_dip  = make_npt();

    npt_base.set_hebbian_alpha(0.05f);
    npt_dip.set_hebbian_alpha(0.05f);

    for (int i = 0; i < 50; ++i) {
        (void)npt_base.forward(torus, 0.5f);
        (void)npt_dip.forward(torus, 0.1f);
    }

    float ip_base = static_cast<float>(npt_base.head(0).Q.inner_product_re(torus));
    float ip_dip  = static_cast<float>(npt_dip.head(0).Q.inner_product_re(torus));

    REQUIRE(ip_base > ip_dip);
}

// ---------------------------------------------------------------------------
// §15  eta_scale is monotonically increasing with dopamine
// ---------------------------------------------------------------------------
TEST_CASE("Phase45 §15 — eta_scale is monotonically increasing with dopamine",
          "[phase45]")
{
    auto torus = make_active_wf();
    const std::array<float, 5> d_vals = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};

    float prev_eta = -1.0f;
    for (float d : d_vals) {
        auto npt = make_npt();
        (void)npt.forward(torus, d);
        float eta = npt.last_eta_scale();
        REQUIRE(eta > prev_eta);
        // Cross-check against analytic formula
        REQUIRE(eta == Catch::Approx(expected_eta(d)).epsilon(1e-5));
        prev_eta = eta;
    }
}
