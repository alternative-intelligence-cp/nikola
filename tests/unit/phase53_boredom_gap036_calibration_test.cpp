/**
 * @file phase53_boredom_gap036_calibration_test.cpp
 * @brief Phase 53 — GAP-036: Boredom Singularity k-Parameter Calibration
 *
 * Spec: docs/info/integration/sections/05_autonomous_systems/
 *       01_computational_neurochemistry.md  §"GAP-036 RESOLUTION"
 *
 * GAP-036 replaces the uncalibrated BOREDOM_K stub with a properly derived
 * time-domain logistic B(t) = 1 / (1 + exp(−k·(elapsed − T_half))) whose
 * boundary conditions produce a 10-minute exploration cycle:
 *   B(0s)   ≈ 0.10   (right after novelty)
 *   B(600s) ≈ 0.85   (trigger threshold crossed at ~546s, ≫ θ_explore=0.8)
 *
 * The logistic is added as an opt-in "time-domain mode" on BoredomRegulator
 * (constructor arg time_domain_mode=true). Default Phase-49 mode unchanged.
 *
 * Tests (20 cases):
 *   §1  – §5   Constants: k_sec, T_half_sec, k_tick, T_half_ticks, k·T_half
 *   §6  – §10  Mode management: default false, accessor, elapsed init, accum, novelty-reset
 *   §11 – §13  Boundary conditions: B(0)≈0.10, B(335)≈0.50, B(600)≈0.85
 *   §14 – §15  should_explore() false at t=0, true at t=600
 *   §16        Monotonicity over time
 *   §17 – §18  drain() resets elapsed_s_ and boredom
 *   §19        last_delta_b() at T_half == k·B·(1−B) = k/4
 *   §20        Phase-49 default mode unchanged (backward compat)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <nikola/autonomy/entropy_estimator.hpp>

using namespace nikola::autonomy;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ── Convenience helpers ───────────────────────────────────────────────────────

/// Create a BoredomRegulator in GAP-036 time-domain mode.
static BoredomRegulator make_tdm() {
    return BoredomRegulator{BOREDOM_ALPHA_ACC, BOREDOM_K, BOREDOM_DECAY_RATE, /*time_domain=*/true};
}

// ── §1–§5  Constants ─────────────────────────────────────────────────────────

TEST_CASE("Phase53 §1 — BOREDOM_K_SEC ≈ 0.00656", "[phase53]") {
    REQUIRE_THAT(BOREDOM_K_SEC, WithinAbs(0.00656f, 1e-5f));
}

TEST_CASE("Phase53 §2 — BOREDOM_T_HALF_SEC == 335.0", "[phase53]") {
    REQUIRE_THAT(BOREDOM_T_HALF_SEC, WithinAbs(335.0f, 0.5f));
}

TEST_CASE("Phase53 §3 — BOREDOM_K_TICK == k_sec / 1000", "[phase53]") {
    // k_tick = 0.00656 / 1000 = 6.56e-6
    const float expected = BOREDOM_K_SEC / 1000.0f;
    REQUIRE_THAT(BOREDOM_K_TICK, WithinRel(expected, 0.01f));
}

TEST_CASE("Phase53 §4 — BOREDOM_T_HALF_TICKS == T_half_sec * 1000", "[phase53]") {
    const float expected = BOREDOM_T_HALF_SEC * 1000.0f;
    REQUIRE_THAT(BOREDOM_T_HALF_TICKS, WithinRel(expected, 0.001f));
}

TEST_CASE("Phase53 §5 — k·T_half ≈ ln(9) ≈ 2.197 (spec derivation check)", "[phase53]") {
    // From spec §GAP-036 Step 1: B(0)=0.10 implies k·T_half = ln(9).
    const float product  = BOREDOM_K_SEC * BOREDOM_T_HALF_SEC;
    const float ln9      = std::log(9.0f);   // ≈ 2.1972
    REQUIRE_THAT(product, WithinAbs(ln9, 0.05f));
}

// ── §6–§10  Mode management ───────────────────────────────────────────────────

TEST_CASE("Phase53 §6 — Default BoredomRegulator is NOT in time-domain mode", "[phase53]") {
    BoredomRegulator br;
    REQUIRE_FALSE(br.is_time_domain_mode());
}

TEST_CASE("Phase53 §7 — time_domain_mode=true stored correctly", "[phase53]") {
    auto br = make_tdm();
    REQUIRE(br.is_time_domain_mode());
}

TEST_CASE("Phase53 §8 — elapsed_since_novelty_s() starts at 0", "[phase53]") {
    auto br = make_tdm();
    REQUIRE_THAT(br.elapsed_since_novelty_s(), WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("Phase53 §9 — elapsed_s grows with zero-entropy updates", "[phase53]") {
    auto br = make_tdm();
    br.update(0.0f, 100.0f);   // 100 s, H=0 → no novelty
    REQUIRE_THAT(br.elapsed_since_novelty_s(), WithinAbs(100.0f, 1e-4f));
    br.update(0.0f, 50.0f);
    REQUIRE_THAT(br.elapsed_since_novelty_s(), WithinAbs(150.0f, 1e-4f));
}

TEST_CASE("Phase53 §10 — Novelty (H ≥ ENTROPY_TARGET) resets elapsed counter", "[phase53]") {
    auto br = make_tdm();
    // Accumulate 300 s with low entropy
    br.update(0.0f, 300.0f);
    REQUIRE(br.elapsed_since_novelty_s() > 0.0f);   // has accumulated

    // One high-entropy tick (novelty)
    br.update(ENTROPY_TARGET, 1.0f);
    REQUIRE_THAT(br.elapsed_since_novelty_s(), WithinAbs(0.0f, 1e-6f));
    // After reset, boredom drops back toward initial value
    REQUIRE(br.level() < 0.15f);
}

// ── §11–§13  Boundary conditions ─────────────────────────────────────────────

TEST_CASE("Phase53 §11 — B(elapsed=0) ≈ 0.10 (spec boundary condition 1)", "[phase53]") {
    // Immediately after construction (no ticks yet), elapsed=0:
    //   B = 1 / (1 + exp(k·T_half)) = 1/10 = 0.10
    auto br = make_tdm();
    REQUIRE_THAT(br.level(), WithinAbs(0.10f, 0.005f));
}

TEST_CASE("Phase53 §12 — B(elapsed=T_half=335s) ≈ 0.50 (inflection point)", "[phase53]") {
    auto br = make_tdm();
    br.update(0.0f, BOREDOM_T_HALF_SEC);   // advance exactly to inflection
    REQUIRE_THAT(br.level(), WithinAbs(0.50f, 0.005f));
}

TEST_CASE("Phase53 §13 — B(elapsed=600s) ≈ 0.85 (spec boundary condition 2)", "[phase53]") {
    // Spec derivation: B(600s) ≈ 0.85 with k=0.00656, T_half=335.
    auto br = make_tdm();
    br.update(0.0f, 600.0f);
    REQUIRE_THAT(br.level(), WithinAbs(0.85f, 0.01f));
}

// ── §14–§15  should_explore() ─────────────────────────────────────────────────

TEST_CASE("Phase53 §14 — should_explore() false at elapsed=0 (B≈0.10 < θ=0.8)", "[phase53]") {
    auto br = make_tdm();
    REQUIRE_FALSE(br.should_explore());
}

TEST_CASE("Phase53 §15 — should_explore() true at elapsed=600s (B≈0.85 > θ=0.8)", "[phase53]") {
    auto br = make_tdm();
    br.update(0.0f, 600.0f);
    REQUIRE(br.should_explore());
}

// ── §16  Monotonicity ─────────────────────────────────────────────────────────

TEST_CASE("Phase53 §16 — B is monotonically increasing with elapsed time", "[phase53]") {
    auto br = make_tdm();
    float prev = br.level();
    for (int i = 1; i <= 12; ++i) {
        br.update(0.0f, 50.0f);   // 50s steps up to 600s
        const float curr = br.level();
        REQUIRE(curr > prev);
        prev = curr;
    }
}

// ── §17–§18  drain() ──────────────────────────────────────────────────────────

TEST_CASE("Phase53 §17 — drain() resets elapsed_s_ to 0 in time-domain mode", "[phase53]") {
    auto br = make_tdm();
    br.update(0.0f, 400.0f);           // → elapsed > 0, B > 0.5
    REQUIRE(br.elapsed_since_novelty_s() > 0.0f);
    br.drain(0.5f);
    REQUIRE_THAT(br.elapsed_since_novelty_s(), WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("Phase53 §18 — drain() restores boredom to ~0.10 in time-domain mode", "[phase53]") {
    auto br = make_tdm();
    br.update(0.0f, 600.0f);           // B ≈ 0.85
    REQUIRE(br.should_explore());
    br.drain(0.5f);
    // After drain, elapsed reset → B back to initial ~0.10.
    REQUIRE_THAT(br.level(), WithinAbs(0.10f, 0.005f));
    REQUIRE_FALSE(br.should_explore());
}

// ── §19  last_delta_b() ───────────────────────────────────────────────────────

TEST_CASE("Phase53 §19 — last_delta_b() at T_half == k·B·(1−B) = k/4", "[phase53]") {
    // At the inflection point B=0.5, instantaneous rate = k·0.5·0.5 = k/4.
    auto br = make_tdm();
    br.update(0.0f, BOREDOM_T_HALF_SEC);
    const float expected_rate = BOREDOM_K_SEC * 0.25f;   // k/4 ≈ 0.00164
    REQUIRE_THAT(br.last_delta_b(), WithinAbs(expected_rate, 1e-5f));
}

// ── §20  Phase-49 backward compatibility ──────────────────────────────────────

TEST_CASE("Phase53 §20 — Default (Phase-49) mode still uses entropy-driven formula", "[phase53]") {
    // In default mode (time_domain_mode=false), BoredomRegulator uses the
    // AUTO-04 formula: ΔB = alpha_acc·(1-tanh(k·H))·dt − decay·dt.
    // At H=0 (no entropy), ΔB ≈ alpha_acc − decay = 0.1 − 0.01 = 0.09 per second.
    BoredomRegulator br_default;   // Phase-49 defaults
    br_default.update(0.0f, 1.0f);
    const float expected_delta = BOREDOM_ALPHA_ACC - BOREDOM_DECAY_RATE;   // ≈ 0.09
    REQUIRE_THAT(br_default.level(),        WithinAbs(expected_delta, 0.002f));
    REQUIRE_THAT(br_default.last_delta_b(), WithinAbs(BOREDOM_ALPHA_ACC, 1e-5f));

    // Confirm NOT affected by GAP-036 constants (elapsed_s_ irrelevant).
    REQUIRE_FALSE(br_default.is_time_domain_mode());
    REQUIRE_THAT(br_default.elapsed_since_novelty_s(), WithinAbs(0.0f, 1e-6f));
}
