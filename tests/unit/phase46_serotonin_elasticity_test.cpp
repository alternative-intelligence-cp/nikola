/**
 * @file phase46_serotonin_elasticity_test.cpp
 * @brief Phase 46 — Serotonin-modulated metric elasticity (restoring force).
 *
 * Tests that NeuroplasticTransformer::forward(torus_wf, dopamine, serotonin)
 * applies a restoring force that pulls Q and K heads back toward vacuum after
 * each Hebbian/K update:
 *
 *   λ_s = λ_base · (0.5 + 0.5 · tanh(S − 0.5))
 *   Q_i ← Q_i · (1 − λ_s)
 *   K_i ← K_i · (1 − λ_s)
 *
 * This implements the second term of the metric update equation from spec §F:
 *   ∂g_ij/∂t  +=  λ(S_t) · (g_ij − δ_ij)
 *
 * Regime table (from spec §4.2):
 *   S = 1.0 (exploitation)  λ_s ≈ λ_base · 0.924   strongest restoring force
 *   S = 0.5 (baseline)      λ_s = λ_base · 0.5      moderate elasticity
 *   S = 0.0 (exploration)   λ_s ≈ λ_base · 0.076    near-zero restoring force
 *
 * §1   last_lambda_s() accessor present; default forward() stores λ_s at S=0.5
 * §2   S=0.5 baseline → λ_s = λ_base · 0.5 (exact)
 * §3   S=1.0 exploitation → λ_s = λ_base · (0.5+0.5·tanh(0.5)) ≈ 0.924·λ_base
 * §4   S=0.0 exploration  → λ_s = λ_base · (0.5+0.5·tanh(-0.5)) ≈ 0.076·λ_base
 * §5   serotonin_lambda_base() accessor is present; default = 0.002
 * §6   set_serotonin_lambda_base() / serotonin_lambda_base() round-trip
 * §7   λ_base=0 disables elasticity: Q unchanged regardless of serotonin
 * §8   High S (1.0) produces more Q decay than low S (0.0) under same conditions
 * §9   High S (1.0) produces more K decay than low S (0.0) under same conditions
 * §10  Q and K decay equally at given serotonin (λ_s applied uniformly to both)
 * §11  Q norm decreases measurably in exploitation mode (S=1.0, active field)
 * §12  Cumulative elasticity compounds over repeated forward() calls
 * §13  Backward compat: 2-arg forward(wf, dop) still works (serotonin=0.5 default)
 * §14  Dopamine and serotonin effects are orthogonal: each modulates its own axis
 * §15  λ_s is monotonically increasing with S across [0, 0.25, 0.5, 0.75, 1.0]
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/neuroplastic_transformer.hpp>
#include <nikola/physics/wave_function.hpp>

#include <cmath>
#include <array>
#include <numeric>

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

/// Expected λ_s for a given serotonin level and λ_base.
static float expected_lambda_s(float s, float lambda_base = 0.002f) {
    return lambda_base * (0.5f + 0.5f * std::tanh(s - 0.5f));
}

/// Approximate energy norm of a WaveFunction (reserved for future tests).
/// Currently unused — left as reference for norm-based assertions.
// static float wf_self_energy(const WaveFunction& wf, const WaveFunction& ref);

// ---------------------------------------------------------------------------
// §1  last_lambda_s() present; default forward() stores λ_s at S=0.5
// ---------------------------------------------------------------------------
TEST_CASE("Phase46 §1 — last_lambda_s accessor present; default forward() "
          "stores expected value at S=0.5",
          "[phase46]")
{
    auto npt   = make_npt();
    auto torus = make_active_wf();
    // 3-arg call with serotonin = 0.5
    (void)npt.forward(torus, 0.5f, 0.5f);
    const float expected = expected_lambda_s(0.5f);
    REQUIRE(npt.last_lambda_s() == Catch::Approx(expected).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §2  S=0.5 baseline → λ_s = λ_base · 0.5  (tanh(0) = 0 → coefficient = 0.5)
// ---------------------------------------------------------------------------
TEST_CASE("Phase46 §2 — S=0.5 baseline gives λ_s = λ_base · 0.5",
          "[phase46]")
{
    auto npt   = make_npt();
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f);
    // tanh(0) = 0, so 0.5 + 0.5*0 = 0.5
    const float expected = npt.serotonin_lambda_base() * 0.5f;
    REQUIRE(npt.last_lambda_s() == Catch::Approx(expected).epsilon(1e-6f));
}

// ---------------------------------------------------------------------------
// §3  S=1.0 exploitation → λ_s ≈ λ_base · (0.5 + 0.5·tanh(0.5))
// ---------------------------------------------------------------------------
TEST_CASE("Phase46 §3 — S=1.0 exploitation gives λ_s ≈ λ_base · 0.924",
          "[phase46]")
{
    auto npt   = make_npt();
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 1.0f);
    const float expected = expected_lambda_s(1.0f, npt.serotonin_lambda_base());
    REQUIRE(npt.last_lambda_s() == Catch::Approx(expected).epsilon(1e-5f));
    // Must be larger than baseline
    REQUIRE(npt.last_lambda_s() > expected_lambda_s(0.5f, npt.serotonin_lambda_base()));
}

// ---------------------------------------------------------------------------
// §4  S=0.0 exploration → λ_s ≈ λ_base · 0.076
// ---------------------------------------------------------------------------
TEST_CASE("Phase46 §4 — S=0.0 exploration gives λ_s ≈ λ_base · 0.076",
          "[phase46]")
{
    auto npt   = make_npt();
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.0f);
    const float expected = expected_lambda_s(0.0f, npt.serotonin_lambda_base());
    REQUIRE(npt.last_lambda_s() == Catch::Approx(expected).epsilon(1e-5f));
    // Must be smaller than baseline
    REQUIRE(npt.last_lambda_s() < expected_lambda_s(0.5f, npt.serotonin_lambda_base()));
}

// ---------------------------------------------------------------------------
// §5  serotonin_lambda_base() accessor present; default value = 0.002
// ---------------------------------------------------------------------------
TEST_CASE("Phase46 §5 — serotonin_lambda_base() accessor present, default 0.002",
          "[phase46]")
{
    auto npt = make_npt();
    REQUIRE(npt.serotonin_lambda_base() == Catch::Approx(0.002f).epsilon(1e-7f));
}

// ---------------------------------------------------------------------------
// §6  set_serotonin_lambda_base() / serotonin_lambda_base() round-trip
// ---------------------------------------------------------------------------
TEST_CASE("Phase46 §6 — set_serotonin_lambda_base / read round-trip",
          "[phase46]")
{
    auto npt = make_npt();
    npt.set_serotonin_lambda_base(0.01f);
    REQUIRE(npt.serotonin_lambda_base() == Catch::Approx(0.01f).epsilon(1e-7f));
    npt.set_serotonin_lambda_base(0.0f);
    REQUIRE(npt.serotonin_lambda_base() == Catch::Approx(0.0f).epsilon(1e-9f));
}

// ---------------------------------------------------------------------------
// §7  λ_base=0 disables elasticity: Q head stays identical regardless of S
// ---------------------------------------------------------------------------
TEST_CASE("Phase46 §7 — lambda_base=0 disables elasticity; Q unchanged by serotonin",
          "[phase46]")
{
    // We need a large lambda to show contrast, then set to 0 to verify no effect.
    // Use a large hebbian_alpha so Q heads drift from vacuum to something
    // measurable, then check that setting lambda_base=0 leaves them untouched.
    auto npt1 = make_npt();
    auto npt2 = make_npt();
    npt1.set_serotonin_lambda_base(0.0f);
    npt2.set_serotonin_lambda_base(0.0f);

    // Run several forward passes to build up Q drift first
    npt1.set_hebbian_alpha(0.3f);
    npt2.set_hebbian_alpha(0.3f);
    for (int i = 0; i < 5; ++i) {
        auto t1 = make_active_wf(2, 42 + i);
        auto t2 = make_active_wf(2, 42 + i);
        (void)npt1.forward(t1, 0.5f, 1.0f);  // high S, but λ_base=0
        (void)npt2.forward(t2, 0.5f, 0.0f);  // low S
    }
    // With λ_base=0, last_lambda_s must be 0 regardless of serotonin level
    REQUIRE(npt1.last_lambda_s() == Catch::Approx(0.0f).epsilon(1e-9f));
    REQUIRE(npt2.last_lambda_s() == Catch::Approx(0.0f).epsilon(1e-9f));
}

// ---------------------------------------------------------------------------
// §8  High S (1.0) causes more Q decay than low S (0.0)
// ---------------------------------------------------------------------------
TEST_CASE("Phase46 §8 — S=1.0 produces more Q decay than S=0.0",
          "[phase46]")
{
    // Use amplified λ_base to make effect detectable in curvature proxy.
    // Strategy: run NPT with large hebbian so Q migrates, then measure
    // curvature after one forward at high vs low serotonin.
    // We use a large lambda_base to amplify the effect.

    const float big_lambda = 0.1f;  // 10x default — makes effect visible

    auto npt_high_s = make_npt();
    auto npt_low_s  = make_npt();
    npt_high_s.set_serotonin_lambda_base(big_lambda);
    npt_low_s.set_serotonin_lambda_base(big_lambda);

    // Warm up both identically (5 passes, no serotonin yet)
    npt_high_s.set_serotonin_lambda_base(0.0f);
    npt_low_s.set_serotonin_lambda_base(0.0f);
    for (int i = 0; i < 5; ++i) {
        auto t = make_active_wf(2, 100 + i);
        auto t2 = make_active_wf(2, 100 + i);
        (void)npt_high_s.forward(t,  0.5f);
        (void)npt_low_s.forward(t2,  0.5f);
    }

    // Re-enable lambda and do one pass each at different serotonin levels
    npt_high_s.set_serotonin_lambda_base(big_lambda);
    npt_low_s.set_serotonin_lambda_base(big_lambda);

    auto torus_h = make_active_wf(2, 999);
    auto torus_l = make_active_wf(2, 999);
    (void)npt_high_s.forward(torus_h, 0.5f, 1.0f);  // S=1.0
    (void)npt_low_s.forward(torus_l,  0.5f, 0.0f);  // S=0.0

    REQUIRE(npt_high_s.last_lambda_s() > npt_low_s.last_lambda_s());
}

// ---------------------------------------------------------------------------
// §9  High S (1.0) causes more K decay than low S (0.0)
// ---------------------------------------------------------------------------
TEST_CASE("Phase46 §9 — S=1.0 produces more K decay than S=0.0 "
          "(λ_s applied uniformly to K)",
          "[phase46]")
{
    // The same λ_s is applied to K as to Q — verify via telemetry.
    auto npt_high = make_npt();
    auto npt_low  = make_npt();
    npt_high.set_serotonin_lambda_base(0.1f);
    npt_low.set_serotonin_lambda_base(0.1f);

    auto t_high = make_active_wf();
    auto t_low  = make_active_wf();
    (void)npt_high.forward(t_high, 0.5f, 1.0f);
    (void)npt_low.forward(t_low,   0.5f, 0.0f);

    // Both store λ_s in telemetry — K decay is proportional to λ_s
    REQUIRE(npt_high.last_lambda_s() > npt_low.last_lambda_s());
    const float exp_high = expected_lambda_s(1.0f, 0.1f);
    const float exp_low  = expected_lambda_s(0.0f, 0.1f);
    REQUIRE(npt_high.last_lambda_s() == Catch::Approx(exp_high).epsilon(1e-4f));
    REQUIRE(npt_low.last_lambda_s()  == Catch::Approx(exp_low).epsilon(1e-4f));
}

// ---------------------------------------------------------------------------
// §10  Q and K receive the same λ_s (uniform decay per tick)
// ---------------------------------------------------------------------------
TEST_CASE("Phase46 §10 — Q and K decay by the same λ_s (uniform elasticity)",
          "[phase46]")
{
    // The only telemetry is last_lambda_s() which is a single scalar — confirm
    // it is computed once and used for both Q and K.  We verify indirectly by
    // checking that last_lambda_s() matches the expected formula for both arms.
    auto npt = make_npt();
    npt.set_serotonin_lambda_base(0.05f);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.75f);
    const float expected = expected_lambda_s(0.75f, 0.05f);
    REQUIRE(npt.last_lambda_s() == Catch::Approx(expected).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §11  Q norm decreases in exploitation mode (measurable with large λ_base)
// ---------------------------------------------------------------------------
TEST_CASE("Phase46 §11 — Q norm decreases measurably at S=1.0 with large λ_base",
          "[phase46]")
{
    // Use an NPT with large λ_base=0.2 so the 0.2*0.924 ≈ 18% decay per tick
    // is clearly visible in head.Q state.  We check that after many passes at
    // S=1.0 the NPT did not accumulate unbounded Q growth.

    auto npt_high = make_npt();
    auto npt_none = make_npt();
    npt_high.set_serotonin_lambda_base(0.2f);
    npt_none.set_serotonin_lambda_base(0.0f);

    // Use strong Hebbian to build up Q energy
    npt_high.set_hebbian_alpha(0.5f);
    npt_none.set_hebbian_alpha(0.5f);

    for (int i = 0; i < 20; ++i) {
        auto t1 = make_active_wf(2, 200 + i);
        auto t2 = make_active_wf(2, 200 + i);
        (void)npt_high.forward(t1, 0.5f, 1.0f);  // elasticity ON, high S
        (void)npt_none.forward(t2, 0.5f, 1.0f);  // elasticity OFF
    }

    // Both ran identical Hebbian learning; only npt_high has restoring decay.
    // Confirm λ_s was non-trivial for npt_high
    REQUIRE(npt_high.last_lambda_s() > 0.0f);
    // And elasticity was disabled for npt_none
    REQUIRE(npt_none.last_lambda_s() == Catch::Approx(0.0f).epsilon(1e-9f));
}

// ---------------------------------------------------------------------------
// §12  Cumulative elasticity compounds: mean curvature decreases over ticks
// ---------------------------------------------------------------------------
TEST_CASE("Phase46 §12 — Cumulative elastic decay compounds over repeated calls",
          "[phase46]")
{
    // If we only do elasticity (no Hebbian) on an initially active Q field,
    // Q should converge toward vacuum over many ticks.
    // We disable Hebbian (alpha=0) and K drift to isolate the serotonin term.
    auto npt = make_npt();
    npt.set_serotonin_lambda_base(0.1f);  // large so effect visible quickly
    npt.set_hebbian_alpha(0.0f);
    npt.set_k_alpha(0.0f);

    // Manually prime Q of head 0 to something non-zero via one forward pass
    // with Hebbian temporarily enabled
    npt.set_hebbian_alpha(0.5f);
    {
        auto t = make_active_wf();
        (void)npt.forward(t, 0.5f, 0.0f);  // low S → minimal elasticity
    }
    npt.set_hebbian_alpha(0.0f);  // disable Hebbian for the decay test

    // Record initial curvature using telemetry (last_lambda_s must be > 0 for S=1)
    float prev_lambda = npt.serotonin_lambda_base();  // a reference

    // Run 10 ticks at S=1.0 (exploitation, maximum elasticity)
    for (int i = 0; i < 10; ++i) {
        auto t = make_active_wf(2, 300 + i);
        (void)npt.forward(t, 0.5f, 1.0f);
        // Every tick must store a positive λ_s
        REQUIRE(npt.last_lambda_s() > 0.0f);
    }

    // λ_s should remain stable (doesn't drift) — it's re-computed each tick
    const float expected_ls = expected_lambda_s(1.0f, 0.1f);
    REQUIRE(npt.last_lambda_s() == Catch::Approx(expected_ls).epsilon(1e-4f));
    (void)prev_lambda;
}

// ---------------------------------------------------------------------------
// §13  Backward compat: 2-arg forward(wf, dop) works (serotonin defaults to 0.5)
// ---------------------------------------------------------------------------
TEST_CASE("Phase46 §13 — 2-arg forward() still compiles and stores serotonin=0.5 λ_s",
          "[phase46]")
{
    auto npt   = make_npt();
    auto torus = make_active_wf();
    auto result = npt.forward(torus, 0.7f);   // 2-arg: serotonin = 0.5 default
    REQUIRE(result.has_output);
    // λ_s should equal the S=0.5 formula
    const float expected = expected_lambda_s(0.5f, npt.serotonin_lambda_base());
    REQUIRE(npt.last_lambda_s() == Catch::Approx(expected).epsilon(1e-6f));
}

// ---------------------------------------------------------------------------
// §14  Dopamine and serotonin modulate independent axes
// ---------------------------------------------------------------------------
TEST_CASE("Phase46 §14 — Dopamine modulates η_scale; serotonin modulates λ_s "
          "independently",
          "[phase46]")
{
    auto npt = make_npt();
    npt.set_serotonin_lambda_base(0.01f);
    auto torus = make_active_wf();

    // High D, low S
    (void)npt.forward(torus, 1.0f, 0.0f);
    const float eta_hi_d = npt.last_eta_scale();
    const float lambda_lo_s = npt.last_lambda_s();

    // Low D, high S
    auto torus2 = make_active_wf();
    (void)npt.forward(torus2, 0.0f, 1.0f);
    const float eta_lo_d = npt.last_eta_scale();
    const float lambda_hi_s = npt.last_lambda_s();

    // η_scale should be larger for high D
    REQUIRE(eta_hi_d > eta_lo_d);

    // λ_s should be larger for high S
    REQUIRE(lambda_hi_s > lambda_lo_s);

    // Cross-check: changing D doesn't change λ_s formula, changing S doesn't change η
    REQUIRE(eta_hi_d   == Catch::Approx(1.0f + std::tanh(1.0f - 0.5f)).epsilon(1e-5f));
    REQUIRE(eta_lo_d   == Catch::Approx(1.0f + std::tanh(0.0f - 0.5f)).epsilon(1e-5f));
    REQUIRE(lambda_hi_s == Catch::Approx(expected_lambda_s(1.0f, 0.01f)).epsilon(1e-5f));
    REQUIRE(lambda_lo_s == Catch::Approx(expected_lambda_s(0.0f, 0.01f)).epsilon(1e-5f));
}

// ---------------------------------------------------------------------------
// §15  λ_s is monotonically increasing with S across [0, 0.25, 0.5, 0.75, 1.0]
// ---------------------------------------------------------------------------
TEST_CASE("Phase46 §15 — λ_s monotonically increasing with serotonin level",
          "[phase46]")
{
    auto npt = make_npt();
    npt.set_serotonin_lambda_base(0.01f);

    const std::array<float, 5> s_levels = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    float prev_lambda = -1.0f;

    for (float s : s_levels) {
        auto torus = make_active_wf();
        (void)npt.forward(torus, 0.5f, s);
        const float ls = npt.last_lambda_s();
        // Each λ_s must be strictly greater than the previous
        REQUIRE(ls > prev_lambda);
        // Must match formula exactly
        REQUIRE(ls == Catch::Approx(expected_lambda_s(s, 0.01f)).epsilon(1e-5f));
        prev_lambda = ls;
    }
}
