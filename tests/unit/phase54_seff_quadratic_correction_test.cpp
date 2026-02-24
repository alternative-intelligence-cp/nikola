/**
 * @file phase54_seff_quadratic_correction_test.cpp
 * @brief Phase 54 — Spec §"Non-Linear Interaction Terms": s_eff(N) = s_local/(1+N²)
 *
 * Spec: docs/info/integration/sections/05_autonomous_systems/
 *       01_computational_neurochemistry.md  §"Non-Linear Interaction Terms" §3
 *
 * **Correction from Phase 47:**
 *   Phase 47 implemented the linear approximation τ_eff = τ/(1+N).
 *   The spec explicitly defines a *quadratic* refractive index formula:
 *     s_eff(N) = s_local / (1 + N²)
 *   Phase 54 corrects the NPT forward() to use τ_eff = τ / (1 + N²).
 *
 * Behavioural difference:
 *   - At N=0 and N=1 the formulas are *identical* (0²=0, 1²=1).
 *   - For 0 < N < 1, the quadratic gives LESS suppression: 1+N² < 1+N.
 *   - This means the baseline (N=0.5) is now τ/1.25 instead of τ/1.5.
 *   - The quadratic curve is "flat" near 0 (low-NE states robustly preserved),
 *     then accelerates steeply near N=1 ("Flash of Insight" — abrupt onset).
 *
 * Tests (20 cases):
 *   §1  – §4   Formula boundary conditions: N=0, N=0.5, N=1, clamped N>1
 *   §5  – §7   Quadratic vs linear comparison: quad > linear for 0 < N < 1
 *   §8         Regime table check: N=0.25, 0.5, 0.75, 1.0
 *   §9         Monotone decreasing
 *   §10        Derivative near N=0 is small (flat quadratic onset)
 *   §11        Derivative near N=1 is large (steep quadratic onset)
 *   §12        "Flash of Insight" N=0.8 gives τ/(1+0.64) = τ/1.64
 *   §13        N=0.9 → τ/(1+0.81) = τ/1.81
 *   §14        head_scores sum to 1.0 at all NE levels
 *   §15        has_output true at all NE levels
 *   §16        τ_eff stored in last_tau_eff() for N=0.5 → τ/1.25
 *   §17        τ_eff stored in last_tau_eff() for N=0.75 → τ/1.5625
 *   §18        Phase 47 backward compat: N=0 → τ (same endpoint)
 *   §19        Phase 47 backward compat: N=1 → τ/2 (same endpoint)
 *   §20        Multi-N sweep: last_tau_eff() = τ/(1+N²) for 8 levels
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <array>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <nikola/cognitive/neuroplastic_transformer.hpp>
#include <nikola/physics/wave_function.hpp>

using namespace nikola::cognitive;
using namespace nikola::physics;
using nikola::foundation::GridConfig;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Spec-correct τ_eff formula: τ / (1 + N²).
static float tau_n2(float ne, float tau_base) {
    const float n = std::clamp(ne, 0.0f, 1.0f);
    return tau_base / (1.0f + n * n);
}

/// Old Phase-47 linear formula (reference for comparison): τ / (1 + N).
static float tau_linear(float ne, float tau_base) {
    const float n = std::clamp(ne, 0.0f, 1.0f);
    return tau_base / (1.0f + n);
}

static NeuroplasticTransformer make_npt(int grid_n = 2, float temperature = 1.0f) {
    return NeuroplasticTransformer(grid_n, temperature, /*curvature_alpha=*/0.5f);
}

static WaveFunction make_active_wf(int grid_n = 2, uint32_t seed = 42) {
    WaveFunction wf(GridConfig::uniform(grid_n));
    wf.seed_manifold(grid_n, 0, 1, 0.3f, seed);
    return wf;
}

// ── §1–§4  Formula boundary conditions ───────────────────────────────────────

TEST_CASE("Phase54 §1 — N=0.0 : τ_eff = τ (quadratic agrees with linear at zero)", "[phase54]") {
    const float tau = 1.2f;
    auto npt   = make_npt(2, tau);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 0.0f);
    REQUIRE_THAT(npt.last_tau_eff(), WithinAbs(tau, 1e-5f));
}

TEST_CASE("Phase54 §2 — N=0.5 : τ_eff = τ/1.25  (quadratic: 1+N²=1.25, NOT linear 1.5)", "[phase54]") {
    const float tau = 1.0f;
    auto npt   = make_npt(2, tau);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 0.5f);
    REQUIRE_THAT(npt.last_tau_eff(), WithinAbs(tau / 1.25f, 1e-5f));
    // Explicit regression: must NOT equal the old linear value τ/1.5.
    REQUIRE(std::abs(npt.last_tau_eff() - tau / 1.5f) > 0.05f);
}

TEST_CASE("Phase54 §3 — N=1.0 : τ_eff = τ/2  (quadratic agrees with linear at unity)", "[phase54]") {
    const float tau = 0.9f;
    auto npt   = make_npt(2, tau);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 1.0f);
    REQUIRE_THAT(npt.last_tau_eff(), WithinAbs(tau / 2.0f, 1e-5f));
}

TEST_CASE("Phase54 §4 — N=2.0 clamped to 1.0 : τ_eff = τ/2", "[phase54]") {
    const float tau = 1.5f;
    auto npt   = make_npt(2, tau);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 2.0f);   // out-of-range
    REQUIRE_THAT(npt.last_tau_eff(), WithinAbs(tau / 2.0f, 1e-5f));
}

// ── §5–§7  Quadratic > linear for all interior 0 < N < 1 ─────────────────────

TEST_CASE("Phase54 §5 — quadratic τ_eff > linear for N=0.25 (less base suppression)", "[phase54]") {
    const float tau = 1.0f;
    auto npt   = make_npt(2, tau);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 0.25f);
    const float quad   = npt.last_tau_eff();
    const float linear = tau_linear(0.25f, tau);
    // quad = τ/(1+0.0625) = τ/1.0625 ≈ 0.941; linear = τ/1.25 = 0.8
    REQUIRE(quad > linear);
}

TEST_CASE("Phase54 §6 — quadratic τ_eff > linear for N=0.5 (weakened baseline damping)", "[phase54]") {
    const float tau = 1.0f;
    auto npt   = make_npt(2, tau);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 0.5f);
    const float quad   = npt.last_tau_eff();
    const float linear = tau_linear(0.5f, tau);
    // quad = τ/1.25 = 0.8; linear = τ/1.5 ≈ 0.667
    REQUIRE(quad > linear);
    REQUIRE_THAT(quad - linear, WithinAbs(0.8f - (1.0f/1.5f), 0.01f));
}

TEST_CASE("Phase54 §7 — quadratic τ_eff > linear for N=0.75", "[phase54]") {
    const float tau = 1.0f;
    auto npt   = make_npt(2, tau);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 0.75f);
    const float quad   = npt.last_tau_eff();
    const float linear = tau_linear(0.75f, tau);
    REQUIRE(quad > linear);
}

// ── §8  Regime table values ───────────────────────────────────────────────────

TEST_CASE("Phase54 §8 — regime table: N=0.25,0.5,0.75,1.0 match τ/(1+N²)", "[phase54]") {
    const float tau = 2.0f;
    const std::array<float, 4> ne_vals   = {0.25f, 0.5f, 0.75f, 1.0f};
    const std::array<float, 4> expected  = {
        tau / (1.0f + 0.0625f),   // N=0.25 → 1.0625
        tau / (1.0f + 0.25f),     // N=0.5  → 1.25
        tau / (1.0f + 0.5625f),   // N=0.75 → 1.5625
        tau / (1.0f + 1.0f),      // N=1.0  → 2.0
    };
    auto npt = make_npt(2, tau);
    for (int i = 0; i < 4; ++i) {
        auto torus = make_active_wf(2, 100 + i);
        (void)npt.forward(torus, 0.5f, 0.5f, ne_vals[i]);
        REQUIRE_THAT(npt.last_tau_eff(), WithinAbs(expected[i], 1e-4f));
    }
}

// ── §9  Monotone decreasing ───────────────────────────────────────────────────

TEST_CASE("Phase54 §9 — τ_eff monotonically decreasing with N under quadratic formula", "[phase54]") {
    const float tau = 1.0f;
    auto npt = make_npt(2, tau);
    const std::array<float, 6> ne_levels = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};
    float prev = std::numeric_limits<float>::max();
    for (float ne : ne_levels) {
        auto torus = make_active_wf(2, static_cast<uint32_t>(ne * 1000));
        (void)npt.forward(torus, 0.5f, 0.5f, ne);
        const float curr = npt.last_tau_eff();
        REQUIRE(curr < prev);
        REQUIRE_THAT(curr, WithinAbs(tau_n2(ne, tau), 1e-5f));
        prev = curr;
    }
}

// ── §10–§11  Curvature: flat onset, steep near N=1 ───────────────────────────

TEST_CASE("Phase54 §10 — quadratic suppression near N=0 is small (flat onset)", "[phase54]") {
    // dB/dN = τ·2N / (1+N²)² at N→0 → 0, so small N has minimal suppression.
    const float tau = 1.0f;
    const float reduction_at_0_1 = tau - tau_n2(0.1f, tau);   // τ - τ/(1+0.01) ≈ 0.0099
    const float reduction_at_0_5 = tau - tau_n2(0.5f, tau);   // τ - τ/1.25 = 0.2
    // Small N (0.1) gives much less reduction than mid N (0.5)
    REQUIRE(reduction_at_0_1 < reduction_at_0_5 * 0.15f);
}

TEST_CASE("Phase54 §11 — quadratic suppression near N=1 is steep (Flash of Insight onset)", "[phase54]") {
    // Near N=1 the quadratic changes faster per unit N than near N=0.
    const float tau = 1.0f;
    const float step = 0.1f;
    const float delta_near_0 = tau_n2(0.0f, tau) - tau_n2(step, tau);        // small
    const float delta_near_1 = tau_n2(1.0f - step, tau) - tau_n2(1.0f, tau); // larger
    REQUIRE(delta_near_1 > delta_near_0);
}

// ── §12–§13  Flash-of-Insight regime ─────────────────────────────────────────

TEST_CASE("Phase54 §12 — N=0.8 (high arousal) : τ_eff = τ/(1+0.64) = τ/1.64", "[phase54]") {
    const float tau = 1.0f;
    auto npt   = make_npt(2, tau);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 0.8f);
    REQUIRE_THAT(npt.last_tau_eff(), WithinAbs(tau / 1.64f, 1e-4f));
}

TEST_CASE("Phase54 §13 — N=0.9 (near-panic) : τ_eff = τ/(1+0.81) = τ/1.81", "[phase54]") {
    const float tau = 1.0f;
    auto npt   = make_npt(2, tau);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 0.9f);
    REQUIRE_THAT(npt.last_tau_eff(), WithinAbs(tau / 1.81f, 1e-4f));
}

// ── §14–§15  Structural health ────────────────────────────────────────────────

TEST_CASE("Phase54 §14 — head_scores sum to 1.0 at all NE levels under N² formula", "[phase54]") {
    auto npt = make_npt();
    for (float ne : std::array<float, 5>{0.0f, 0.25f, 0.5f, 0.75f, 1.0f}) {
        auto torus  = make_active_wf(2, static_cast<uint32_t>(ne * 1000));
        auto result = npt.forward(torus, 0.5f, 0.5f, ne);
        const float total = std::accumulate(result.head_scores.begin(),
                                             result.head_scores.end(), 0.0f);
        REQUIRE_THAT(total, WithinAbs(1.0f, 1e-4f));
    }
}

TEST_CASE("Phase54 §15 — has_output true at all NE levels under N² formula", "[phase54]") {
    auto npt = make_npt();
    for (float ne : std::array<float, 4>{0.0f, 0.3f, 0.7f, 1.0f}) {
        auto torus  = make_active_wf(2, static_cast<uint32_t>(ne * 500));
        auto result = npt.forward(torus, 0.5f, 0.5f, ne);
        REQUIRE(result.has_output);
    }
}

// ── §16–§17  last_tau_eff() telemetry ────────────────────────────────────────

TEST_CASE("Phase54 §16 — last_tau_eff() stores τ/(1+N²) for N=0.5 → τ/1.25", "[phase54]") {
    const float tau = 2.5f;
    auto npt   = make_npt(2, tau);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 0.5f);
    REQUIRE_THAT(npt.last_tau_eff(), WithinAbs(tau / 1.25f, 1e-5f));
}

TEST_CASE("Phase54 §17 — last_tau_eff() stores τ/(1+N²) for N=0.75 → τ/1.5625", "[phase54]") {
    const float tau = 1.6f;
    auto npt   = make_npt(2, tau);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 0.75f);
    REQUIRE_THAT(npt.last_tau_eff(), WithinAbs(tau / 1.5625f, 1e-5f));
}

// ── §18–§19  Phase 47 backward compatibility at endpoints ─────────────────────

TEST_CASE("Phase54 §18 — N=0 endpoint identical to Phase-47 linear (both give τ)", "[phase54]") {
    const float tau = 0.7f;
    auto npt   = make_npt(2, tau);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 0.0f);
    REQUIRE_THAT(npt.last_tau_eff(), WithinAbs(tau, 1e-5f));
    REQUIRE_THAT(npt.last_tau_eff(), WithinAbs(tau_linear(0.0f, tau), 1e-5f));
    REQUIRE_THAT(npt.last_tau_eff(), WithinAbs(tau_n2(0.0f, tau), 1e-5f));
}

TEST_CASE("Phase54 §19 — N=1 endpoint identical to Phase-47 linear (both give τ/2)", "[phase54]") {
    const float tau = 0.7f;
    auto npt   = make_npt(2, tau);
    auto torus = make_active_wf();
    (void)npt.forward(torus, 0.5f, 0.5f, 1.0f);
    REQUIRE_THAT(npt.last_tau_eff(), WithinAbs(tau / 2.0f, 1e-5f));
    REQUIRE_THAT(npt.last_tau_eff(), WithinAbs(tau_linear(1.0f, tau), 1e-5f));
    REQUIRE_THAT(npt.last_tau_eff(), WithinAbs(tau_n2(1.0f, tau), 1e-5f));
}

// ── §20  Full sweep ───────────────────────────────────────────────────────────

TEST_CASE("Phase54 §20 — 8-level NE sweep: last_tau_eff() == τ/(1+N²) across range", "[phase54]") {
    const float tau = 1.3f;
    auto npt = make_npt(2, tau);
    const std::array<float, 8> ne_levels = {
        0.0f, 0.1f, 0.2f, 0.4f, 0.5f, 0.6f, 0.8f, 1.0f
    };
    for (float ne : ne_levels) {
        auto torus = make_active_wf(2, static_cast<uint32_t>(ne * 100 + 200));
        (void)npt.forward(torus, 0.5f, 0.5f, ne);
        REQUIRE_THAT(npt.last_tau_eff(),
                     WithinAbs(tau_n2(ne, tau), 1e-4f));
    }
}
