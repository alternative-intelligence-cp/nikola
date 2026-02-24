/**
 * @file phase51_failure_mode_guards_test.cpp
 * @brief Phase 51 — spec §9 Failure Mode Guards: Anhedonia Trap (§9.1) + Mania Loop (§9.2)
 *
 * §9.1 Anhedonia Trap:
 *   The Physics Oracle monitors D(t).  If D < 0.1 for > 1000 consecutive cycles it
 *   triggers an "Emergency Stimulus" — a synthetic Reward::POSITIVE signal — to
 *   jumpstart the plasticity engine.
 *
 * §9.2 Mania Loop:
 *   If MANIA_GUARD_RING_SIZE (3) CuriosityGoals fire within MANIA_DETECT_WINDOW (10)
 *   ticks, Mania is detected. Mitigation: serotonin is artificially boosted ("sedative"),
 *   and goal emission is suppressed for mania_suppression_secs seconds.
 *
 * Test strategy:
 *   - §9.1 tests use small anhedonia_window (20) to avoid 1000-tick loops.
 *   - §9.2 "mania" tests use psi = {1.f}/{0.f} (H=0 ⇒ peak boredom accumulation) + dt=1.0.
 *   - §9.2 "no mania" tests use dt=0.5 to space goals > MANIA_DETECT_WINDOW apart.
 */

#define NIKOLA_AUTONOMY_ENGINE_IMPL
#include <nikola/autonomy/autonomy_engine.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace nikola::autonomy;
using Catch::Matchers::WithinAbs;

// ── helpers ──────────────────────────────────────────────────────────────────

// psi with a single node of probability 1 → H = 0 → maximum boredom accumulation
static const std::vector<float> kSinglePsiR = {1.0f};
static const std::vector<float> kSinglePsiI = {0.0f};
// psi with two equal nodes → H = 1 bit → slower boredom accumulation
static const std::vector<float> kTwoPsiR = {0.707f, 0.707f};
static const std::vector<float> kTwoPsiI = {0.0f,   0.0f  };

/// Advance engine with no wavefunction (boredom block skipped) + specified reward.
static void tick_neg(AutonomyEngine& eng, int n = 1, float dt = 0.01f) {
    for (int i = 0; i < n; ++i)
        eng.tick(dt, {}, {}, Reward::NEGATIVE, 0.0f);
}

static void tick_pos(AutonomyEngine& eng, int n = 1, float dt = 0.01f) {
    for (int i = 0; i < n; ++i)
        eng.tick(dt, {}, {}, Reward::POSITIVE, 0.0f);
}

static void tick_neutral(AutonomyEngine& eng, int n = 1, float dt = 0.01f) {
    for (int i = 0; i < n; ++i)
        eng.tick(dt, {}, {}, Reward::NEUTRAL, 0.0f);
}

/// Tick with H≈0 psi (drives fast boredom) using given dt.
static void tick_bored(AutonomyEngine& eng, int n = 1, float dt = 1.0f) {
    for (int i = 0; i < n; ++i)
        eng.tick(dt, kSinglePsiR, kSinglePsiI, Reward::NEUTRAL, 0.0f);
}

/// Tick with H=1 psi (slower boredom) using given dt.
static void tick_relaxed(AutonomyEngine& eng, int n = 1, float dt = 0.5f) {
    for (int i = 0; i < n; ++i)
        eng.tick(dt, kTwoPsiR, kTwoPsiI, Reward::NEUTRAL, 0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  §9.1  Anhedonia Trap
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Anhedonia_IsAnhedonic_FalseOnFreshEngine", "[phase51]") {
    AutonomyEngine eng;
    // Default dopamine = 0.5 ≫ threshold 0.1
    REQUIRE_FALSE(eng.is_anhedonic());
    REQUIRE(eng.emergency_stimulus_count() == 0u);
}

TEST_CASE("Anhedonia_IsAnhedonic_TrueAfterNegativeReward", "[phase51]") {
    AutonomyEngine eng;
    // Drive D to 0 with a NEGATIVE tick (empty psi, tiny dt → very small decay recovery)
    eng.tick(0.01f, {}, {}, Reward::NEGATIVE, 0.0f);
    // D = 0 (from NEGATIVE update) + ~0.0025 (decay recovery) ≈ 0.0025 < 0.1
    REQUIRE(eng.is_anhedonic());
    REQUIRE(eng.dopamine() < ANHEDONIA_D_THRESHOLD);
}

TEST_CASE("Anhedonia_IsAnhedonic_FalseAfterPositiveReward", "[phase51]") {
    AutonomyEngine eng;
    // First push dopamine low
    tick_neg(eng, 3);
    REQUIRE(eng.is_anhedonic());
    // Then recover with a POSITIVE reward
    tick_pos(eng, 1);
    REQUIRE_FALSE(eng.is_anhedonic());
    REQUIRE(eng.dopamine() > ANHEDONIA_D_THRESHOLD);
}

TEST_CASE("Anhedonia_EmergencyStimulus_FiresAfterWindow", "[phase51]") {
    // Use small window (20) so test completes quickly
    AutonomyConfig cfg;
    cfg.anhedonia_window = 20u;
    AutonomyEngine eng{cfg};

    // Tick 20 times with NEGATIVE + tiny dt so D stays < threshold after each tick
    tick_neg(eng, 20);

    // Emergency Stimulus should have fired exactly once
    REQUIRE(eng.emergency_stimulus_count() == 1u);
}

TEST_CASE("Anhedonia_EmergencyStimulus_BoostsDopamine", "[phase51]") {
    AutonomyConfig cfg;
    cfg.anhedonia_window = 20u;
    AutonomyEngine eng{cfg};

    tick_neg(eng, 20);
    REQUIRE(eng.emergency_stimulus_count() == 1u);

    // After ES, dopamine should be well above the anhedonia threshold
    REQUIRE(eng.dopamine() > ANHEDONIA_D_THRESHOLD);
}

TEST_CASE("Anhedonia_EmergencyStimulus_ResetsAnhedonicState", "[phase51]") {
    AutonomyConfig cfg;
    cfg.anhedonia_window = 20u;
    AutonomyEngine eng{cfg};

    tick_neg(eng, 20);
    REQUIRE(eng.emergency_stimulus_count() == 1u);
    // Immediately after the tick where ES fired, D is high → not anhedonic
    REQUIRE_FALSE(eng.is_anhedonic());
}

TEST_CASE("Anhedonia_CycleReset_OnIntervening_PositiveReward", "[phase51]") {
    // Window = 15. Tick 10 NEGs (no ES yet), then 1 POS (resets cycle),
    // then 14 NEGs (not enough for another ES), then 1 final NEG → ES fires.
    AutonomyConfig cfg;
    cfg.anhedonia_window = 15u;
    AutonomyEngine eng{cfg};

    tick_neg(eng, 10);
    REQUIRE(eng.emergency_stimulus_count() == 0u);  // no ES yet

    tick_pos(eng, 1);                                // D rises → cycle resets

    tick_neg(eng, 14);
    REQUIRE(eng.emergency_stimulus_count() == 0u);  // still 0: need 15 CONSEC, only 14 done

    tick_neg(eng, 1);                               // 15th consecutive NEGATIVE
    REQUIRE(eng.emergency_stimulus_count() == 1u);  // ES fires
}

TEST_CASE("Anhedonia_MultipleStimuli_Counted", "[phase51]") {
    AutonomyConfig cfg;
    cfg.anhedonia_window = 20u;
    AutonomyEngine eng{cfg};

    // Three windows of 20 consecutive NEGATIVE ticks → 3 ES events
    tick_neg(eng, 20);
    REQUIRE(eng.emergency_stimulus_count() == 1u);

    tick_neg(eng, 20);
    REQUIRE(eng.emergency_stimulus_count() == 2u);

    tick_neg(eng, 20);
    REQUIRE(eng.emergency_stimulus_count() == 3u);
}

// ─────────────────────────────────────────────────────────────────────────────
//  §9.2  Mania Loop
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Mania_NoTrigger_OnSlowGoalRapidfire", "[phase51]") {
    // With dt=0.5 and H≈1 psi, boredom accumulates at ~0.034/tick.
    // Goals fire roughly every 24 ticks → ring span >> MANIA_DETECT_WINDOW.
    AutonomyEngine eng;
    tick_relaxed(eng, 100, 0.5f);
    // Mania should NOT have been triggered
    REQUIRE(eng.mania_suppress_count() == 0u);
    REQUIRE_FALSE(eng.is_mania_suppressed());
}

TEST_CASE("Mania_Detected_On3GoalsIn10Ticks", "[phase51]") {
    // With dt=1.0 and H=0 psi, goals fire at ticks ~9, ~13, ~16 (span=7 < 10).
    AutonomyEngine eng;

    int fire_count = 0;
    eng.on_curiosity_goal = [&](CuriosityGoal) { ++fire_count; };

    tick_bored(eng, 20, 1.0f);

    // Mania should have been detected (ring of 3 goals within window)
    REQUIRE(eng.mania_suppress_count() >= 1u);
    REQUIRE(fire_count >= 3);  // at least 3 goals fired before mania suppressed
}

TEST_CASE("Mania_Suppression_BlocksGoalEmission", "[phase51]") {
    // Trigger mania, then tick 29 more seconds — all within suppression window.
    // Use a long suppression (100 s) so there's no chance of expiry during the test.
    AutonomyConfig cfg;
    cfg.mania_suppression_secs = 100.0f;
    AutonomyEngine eng{cfg};

    int fire_count_before = 0;
    eng.on_curiosity_goal = [&](CuriosityGoal) { ++fire_count_before; };

    // Phase 1: let mania trigger (≥16 ticks usually enough)
    tick_bored(eng, 20, 1.0f);
    REQUIRE(eng.mania_suppress_count() >= 1u);
    REQUIRE(eng.is_mania_suppressed());

    int fire_after = 0;
    eng.on_curiosity_goal = [&](CuriosityGoal) { ++fire_after; };

    // Phase 2: tick 29 more seconds; still well within the 100 s suppression window
    tick_bored(eng, 29, 1.0f);

    REQUIRE(fire_after == 0);
    REQUIRE(eng.is_mania_suppressed());
}

TEST_CASE("Mania_Suppression_BoostsSerotonin", "[phase51]") {
    AutonomyEngine eng;
    float initial_5ht = eng.serotonin();  // 0.5 by default

    tick_bored(eng, 20, 1.0f);  // trigger mania

    REQUIRE(eng.mania_suppress_count() >= 1u);
    // Serotonin boosted by MANIA_SEROTONIN_BOOST = 0.4  (clamped to 1.0)
    REQUIRE(eng.serotonin() > initial_5ht + 0.3f);
}

TEST_CASE("Mania_IsManiaSupressed_ObserverCorrect", "[phase51]") {
    AutonomyConfig cfg;
    cfg.mania_suppression_secs = 5.0f;  // short window for test speed
    AutonomyEngine eng{cfg};

    REQUIRE_FALSE(eng.is_mania_suppressed());

    tick_bored(eng, 20, 1.0f);  // trigger mania
    REQUIRE(eng.mania_suppress_count() >= 1u);
    REQUIRE(eng.is_mania_suppressed());

    // Tick 5 s (dt=1.0) → suppression timer drops to 0
    tick_bored(eng, 5, 1.0f);
    REQUIRE_FALSE(eng.is_mania_suppressed());
}

TEST_CASE("Mania_SuppressCount_Telemetry", "[phase51]") {
    // Two distinct mania events should each increment the counter.
    AutonomyConfig cfg;
    cfg.mania_suppression_secs = 5.0f;
    AutonomyEngine eng{cfg};

    // First mania event
    tick_bored(eng, 20, 1.0f);
    REQUIRE(eng.mania_suppress_count() >= 1u);
    uint32_t after_first = eng.mania_suppress_count();

    // Allow suppression to expire, then trigger a second mania
    tick_bored(eng, 40, 1.0f);   // 5 s expiry + enough to build boredom + 3 more rapid goals
    REQUIRE(eng.mania_suppress_count() > after_first);
}

TEST_CASE("Mania_Suppression_ExpiresAndGoalsResume", "[phase51]") {
    // Short suppression → goals must resume once timer hits zero.
    AutonomyConfig cfg;
    cfg.mania_suppression_secs = 5.0f;
    AutonomyEngine eng{cfg};

    // Trigger mania
    tick_bored(eng, 20, 1.0f);
    REQUIRE(eng.mania_suppress_count() >= 1u);
    uint32_t goals_at_mania = eng.curiosity_goal_count();

    // Tick through and past suppression window (boredom stays high)
    tick_bored(eng, 20, 1.0f);

    // After expiry + boredom rebuild, at least one new goal should have fired
    REQUIRE(eng.curiosity_goal_count() > goals_at_mania);
}
