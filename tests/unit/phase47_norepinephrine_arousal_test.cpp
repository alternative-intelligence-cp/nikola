/**
 * @file phase47_norepinephrine_arousal_test.cpp
 * @brief Phase 47 — Norepinephrine arousal modulation of attention temperature.
 *
 * Tests that NeuroplasticTransformer::forward(torus_wf, dopamine, serotonin,
 * norepinephrine) modulates the effective attention softmax temperature via:
 *
 *   τ_eff = τ / (1 + N)
 *
 * Mirrors spec §5.1 refractive index formula:
 *   s_eff(t) = s_local / (1 + N_t)
 *
 * Physical interpretation:
 *   High N (hypervigilance / stress):
 *     τ_eff → small → sharp softmax → winner-take-all attention →
 *     one dominant head saturates (tunnel vision)
 *   Low N (calm / deep focus):
 *     τ_eff → τ   → broad, flat softmax → all heads contribute equally →
 *     diffuse multi-scale integration (creative association)
 *
 * Regime table (from spec §5.1):
 *   N = 1.0  τ_eff = τ/2     sharp attention (panic mode)
 *   N = 0.5  τ_eff = τ/1.5   moderate focus (baseline)
 *   N = 0.0  τ_eff = τ       maximum breadth (deep calm)
 *
 * §1   last_tau_eff() accessor present; 4-arg call stores τ_eff for N=0.5
 * §2   N=0.0 → τ_eff = τ (full temperature, unchanged)
 * §3   N=1.0 → τ_eff = τ/2 (half temperature, sharpened)
 * §4   N=0.5 baseline → τ_eff = τ/1.5
 * §5   τ_eff is monotonically decreasing with N across [0, 0.25, 0.5, 0.75, 1.0]
 * §6   High N → head_scores more concentrated (max score higher than at low N)
 * §7   Low N → head_scores more uniform (max score lower than at high N)
 * §8   has_output = true at all norepinephrine levels
 * §9   head_scores sum to ≈ 1.0 at all norepinephrine levels
 * §10  Backward compat: 2-arg forward(wf, dop) still works, NE default = 0.5
 * §11  Backward compat: 3-arg forward(wf, dop, ser) still works, NE default = 0.5
 * §12  N is clamped to [0, 1]: values > 1.0 are treated as 1.0
 * §13  τ_eff formula τ/(1+N) matches expected math across 5 NE levels
 * §14  NE modulates τ_eff independently of dopamine (η_scale) and serotonin (λ_s)
 * §15  At N=1.0, dominant head score is measurably higher than at N=0.0 on same field
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/neuroplastic_transformer.hpp>
#include <nikola/physics/wave_function.hpp>

#include <cmath>
#include <array>
#include <numeric>
#include <algorithm>

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

static NeuroplasticTransformer make_npt(int grid_n = 2, float temperature = 1.0f) {
    return NeuroplasticTransformer(grid_n, temperature, 0.5f);
}

/// Expected τ_eff for a given norepinephrine level and base temperature.
static float expected_tau_eff(float ne, float tau_base = 1.0f) {
    const float ne_clamped = std::clamp(ne, 0.0f, 1.0f);
    return tau_base / (1.0f + ne_clamped);
}

/// Sum of all head scores (should be ≈1.0).
static float sum_scores(const AttentionResult& r) {
    return std::accumulate(r.head_scores.begin(), r.head_scores.end(), 0.0f);
}

/// Max head score.
static float max_score(const AttentionResult& r) {
    return *std::max_element(r.head_scores.begin(), r.head_scores.end());
}

// ---------------------------------------------------------------------------
// §1  last_tau_eff() present; 4-arg call stores τ_eff for N=0.5
// ---------------------------------------------------------------------------
TEST_CASE("Phase47 §1 — last_tau_eff() accessor present; stores correct value at N=0.5",
          "[phase47]")
{
    auto npt   = make_npt(2, 1.0f);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 0.5f);
    const float expected = expected_tau_eff(0.5f, npt.temperature());
    REQUIRE(npt.last_tau_eff() == Catch::Approx(expected).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §2  N=0.0 (deep calm) → τ_eff = τ  (unchanged from base temperature)
// ---------------------------------------------------------------------------
TEST_CASE("Phase47 §2 — N=0.0 deep calm gives τ_eff = τ (no reduction)",
          "[phase47]")
{
    auto npt   = make_npt(2, 1.0f);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 0.0f);
    REQUIRE(npt.last_tau_eff() == Catch::Approx(npt.temperature()).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §3  N=1.0 (panic) → τ_eff = τ/2
// ---------------------------------------------------------------------------
TEST_CASE("Phase47 §3 — N=1.0 panic gives τ_eff = τ/2",
          "[phase47]")
{
    const float tau = 1.5f;
    auto npt   = make_npt(2, tau);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 1.0f);
    REQUIRE(npt.last_tau_eff() == Catch::Approx(tau / 2.0f).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §4  N=0.5 baseline → τ_eff = τ/1.5
// ---------------------------------------------------------------------------
TEST_CASE("Phase47 §4 — N=0.5 baseline gives τ_eff = τ/1.5",
          "[phase47]")
{
    const float tau = 0.8f;
    auto npt   = make_npt(2, tau);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 0.5f);
    REQUIRE(npt.last_tau_eff() == Catch::Approx(tau / 1.5f).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §5  τ_eff monotonically decreasing with N
// ---------------------------------------------------------------------------
TEST_CASE("Phase47 §5 — τ_eff monotonically decreasing as N increases",
          "[phase47]")
{
    const float tau = 1.0f;
    auto npt = make_npt(2, tau);
    const std::array<float, 5> ne_levels = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    float prev = std::numeric_limits<float>::max();
    for (float ne : ne_levels) {
        auto torus = make_active_wf();
        (void)npt.forward(torus, 0.5f, 0.5f, ne);
        const float tau_e = npt.last_tau_eff();
        REQUIRE(tau_e < prev);
        REQUIRE(tau_e == Catch::Approx(expected_tau_eff(ne, tau)).epsilon(1e-5f));
        prev = tau_e;
    }
}

// ---------------------------------------------------------------------------
// §6  High N → head_scores more concentrated (higher max score)
// ---------------------------------------------------------------------------
TEST_CASE("Phase47 §6 — N=1.0 produces higher max head score than N=0.0",
          "[phase47]")
{
    // Use a large base temperature so the NE effect on softmax sharpness is clear.
    // At high τ the distribution is flat; NE halving τ makes it meaningfully sharper.
    const float tau = 2.0f;
    auto npt_hi_ne = make_npt(2, tau);
    auto npt_lo_ne = make_npt(2, tau);

    // Warm up both with identical passes so Q/K are the same state
    for (int i = 0; i < 3; ++i) {
        auto t1 = make_active_wf(2, 50 + i);
        auto t2 = make_active_wf(2, 50 + i);
        (void)npt_hi_ne.forward(t1, 0.5f, 0.5f, 0.5f);
        (void)npt_lo_ne.forward(t2, 0.5f, 0.5f, 0.5f);
    }

    auto torus_hi = make_active_wf(2, 999);
    auto torus_lo = make_active_wf(2, 999);
    auto result_hi = npt_hi_ne.forward(torus_hi, 0.5f, 0.5f, 1.0f);   // N=1.0 sharp
    auto result_lo = npt_lo_ne.forward(torus_lo, 0.5f, 0.5f, 0.0f);   // N=0.0 broad

    // Sharper attention → dominant head gets a higher fraction of the weight
    REQUIRE(max_score(result_hi) > max_score(result_lo));
}

// ---------------------------------------------------------------------------
// §7  Low N → head_scores more uniform (lower max score than high N)
// ---------------------------------------------------------------------------
TEST_CASE("Phase47 §7 — N=0.0 produces lower max head score than N=1.0",
          "[phase47]")
{
    const float tau = 2.0f;
    auto npt_hi = make_npt(2, tau);
    auto npt_lo = make_npt(2, tau);

    auto t1 = make_active_wf(2, 77);
    auto t2 = make_active_wf(2, 77);
    auto r_hi = npt_hi.forward(t1, 0.5f, 0.5f, 1.0f);
    auto r_lo = npt_lo.forward(t2, 0.5f, 0.5f, 0.0f);

    REQUIRE(max_score(r_lo) < max_score(r_hi));
}

// ---------------------------------------------------------------------------
// §8  has_output = true at all norepinephrine levels
// ---------------------------------------------------------------------------
TEST_CASE("Phase47 §8 — has_output is true regardless of norepinephrine",
          "[phase47]")
{
    auto npt = make_npt();
    for (float ne : std::array<float, 4>{0.0f, 0.3f, 0.7f, 1.0f}) {
        auto torus = make_active_wf();
        auto result = npt.forward(torus, 0.5f, 0.5f, ne);
        REQUIRE(result.has_output);
    }
}

// ---------------------------------------------------------------------------
// §9  head_scores sum to ≈ 1.0 at all norepinephrine levels
// ---------------------------------------------------------------------------
TEST_CASE("Phase47 §9 — head_scores sum to 1.0 at all norepinephrine levels",
          "[phase47]")
{
    auto npt = make_npt();
    for (float ne : std::array<float, 5>{0.0f, 0.25f, 0.5f, 0.75f, 1.0f}) {
        auto torus = make_active_wf();
        auto result = npt.forward(torus, 0.5f, 0.5f, ne);
        REQUIRE(sum_scores(result) == Catch::Approx(1.0f).epsilon(1e-4f));
    }
}

// ---------------------------------------------------------------------------
// §10  Backward compat: 2-arg forward(wf, dop) — NE defaults to 0.5
// ---------------------------------------------------------------------------
TEST_CASE("Phase47 §10 — 2-arg forward() backward compat; NE=0.5 default",
          "[phase47]")
{
    auto npt   = make_npt(2, 1.0f);
    auto torus = make_active_wf();
    auto result = npt.forward(torus, 0.5f);   // only dopamine provided
    REQUIRE(result.has_output);
    REQUIRE(npt.last_tau_eff()
            == Catch::Approx(expected_tau_eff(0.5f, npt.temperature())).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §11  Backward compat: 3-arg forward(wf, dop, ser) — NE defaults to 0.5
// ---------------------------------------------------------------------------
TEST_CASE("Phase47 §11 — 3-arg forward() backward compat; NE=0.5 default",
          "[phase47]")
{
    auto npt   = make_npt(2, 1.0f);
    auto torus = make_active_wf();
    auto result = npt.forward(torus, 0.7f, 0.3f);   // dopamine + serotonin, no NE
    REQUIRE(result.has_output);
    REQUIRE(npt.last_tau_eff()
            == Catch::Approx(expected_tau_eff(0.5f, npt.temperature())).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §12  N is clamped: values > 1.0 treated as 1.0; τ_eff = τ/2
// ---------------------------------------------------------------------------
TEST_CASE("Phase47 §12 — norepinephrine is clamped to [0,1]; N=2.0 → τ_eff = τ/2",
          "[phase47]")
{
    const float tau = 1.0f;
    auto npt   = make_npt(2, tau);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 2.0f);    // out-of-range NE
    // Clamped to 1.0: τ_eff = τ/2
    REQUIRE(npt.last_tau_eff() == Catch::Approx(tau / 2.0f).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §13  τ_eff formula τ/(1+N) matches expected math across 5 NE levels
// ---------------------------------------------------------------------------
TEST_CASE("Phase47 §13 — τ_eff = τ/(1+N) matches formula across all NE levels",
          "[phase47]")
{
    const float tau = 1.2f;
    auto npt = make_npt(2, tau);
    const std::array<float, 5> ne_levels = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    for (float ne : ne_levels) {
        auto torus = make_active_wf();
        (void)npt.forward(torus, 0.5f, 0.5f, ne);
        REQUIRE(npt.last_tau_eff()
                == Catch::Approx(expected_tau_eff(ne, tau)).epsilon(1e-5f));
    }
}

// ---------------------------------------------------------------------------
// §14  NE modulates τ_eff independent of dopamine η_scale and serotonin λ_s
// ---------------------------------------------------------------------------
TEST_CASE("Phase47 §14 — NE / dopamine / serotonin modulate independent axes",
          "[phase47]")
{
    auto npt = make_npt(2, 1.0f);
    npt.set_serotonin_lambda_base(0.01f);

    // Case A: High D, low S, mid N
    auto t1 = make_active_wf();
    (void)npt.forward(t1, 1.0f, 0.0f, 0.5f);
    const float eta_hi_d   = npt.last_eta_scale();
    const float lambda_lo_s = npt.last_lambda_s();
    const float tau_mid_n   = npt.last_tau_eff();

    // Case B: Low D, high S, high N
    auto t2 = make_active_wf();
    (void)npt.forward(t2, 0.0f, 1.0f, 1.0f);
    const float eta_lo_d    = npt.last_eta_scale();
    const float lambda_hi_s = npt.last_lambda_s();
    const float tau_hi_n    = npt.last_tau_eff();

    // Dopamine: high D → higher η_scale
    REQUIRE(eta_hi_d > eta_lo_d);
    // Serotonin: high S → higher λ_s
    REQUIRE(lambda_hi_s > lambda_lo_s);
    // NE: high N → lower τ_eff
    REQUIRE(tau_hi_n < tau_mid_n);

    // Cross-check exact formulas
    REQUIRE(eta_hi_d   == Catch::Approx(1.0f + std::tanh(1.0f - 0.5f)).epsilon(1e-5f));
    REQUIRE(tau_hi_n   == Catch::Approx(expected_tau_eff(1.0f, 1.0f)).epsilon(1e-5f));
    REQUIRE(tau_mid_n  == Catch::Approx(expected_tau_eff(0.5f, 1.0f)).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §15  At N=1.0 dominant head score is measurably higher than at N=0.0
//      (same field, large base τ to amplify the effect)
// ---------------------------------------------------------------------------
TEST_CASE("Phase47 §15 — dominant head score at N=1.0 > dominant at N=0.0 "
          "(sharpening effect verified)",
          "[phase47]")
{
    // Use τ=4.0: very flat softmax at N=0; halved to τ=2.0 at N=1 → measurably sharper.
    const float tau = 4.0f;
    auto npt_hi = make_npt(2, tau);
    auto npt_lo = make_npt(2, tau);

    // Identical warm-up
    for (int i = 0; i < 5; ++i) {
        auto t1 = make_active_wf(2, 10 + i);
        auto t2 = make_active_wf(2, 10 + i);
        (void)npt_hi.forward(t1, 0.5f, 0.5f, 0.5f);
        (void)npt_lo.forward(t2, 0.5f, 0.5f, 0.5f);
    }

    auto field_hi = make_active_wf(2, 555);
    auto field_lo = make_active_wf(2, 555);
    auto r_hi = npt_hi.forward(field_hi, 0.5f, 0.5f, 1.0f);
    auto r_lo = npt_lo.forward(field_lo, 0.5f, 0.5f, 0.0f);

    // τ_eff at N=1 is half of τ_eff at N=0 → sharper distribution
    REQUIRE(npt_hi.last_tau_eff() < npt_lo.last_tau_eff());
    // Dominant head gets more weight under higher NE
    REQUIRE(max_score(r_hi) > max_score(r_lo));
}
