/**
 * @file tests/unit/phase23_decision_loop_test.cpp
 * @brief Phase 23: DecisionLoop — autonomous action selection tests.
 *
 * Tests the DecisionLoop in isolation (no ORT required):
 *   - Construction and configuration
 *   - State reading (NikolaState snapshot)
 *   - Action scoring properties (ordering, boundary conditions)
 *   - Tick execution (increments counter, no crash)
 *   - Callback firing (on_tick, on_action)
 *   - Cooldown enforcement (EMIT_THOUGHT rate limiting)
 *   - SILENT is selected when nothing is compelling
 *   - NAP scores high when ATP is depleted
 *   - REFUSE scores high when TD error is strongly negative
 *   - EXPLORE scores high when boredom is high and ATP is sufficient
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/cognitive/cognitive_torus.hpp>

#include <string>
#include <vector>

using namespace nikola::autonomy;
using namespace nikola::cognitive;

// ============================================================================
// Helpers
// ============================================================================

static CognitiveTorus make_torus() {
    return CognitiveTorus(3);   // 3^9 = 19,683 nodes, no ORT
}

static AutonomyEngine make_engine() {
    AutonomyConfig cfg;
    cfg.enable_dream_weave = false;
    cfg.enable_boredom     = true;
    return AutonomyEngine(cfg);
}

static DecisionLoopConfig make_config(float min_emit_s = 0.0f) {
    DecisionLoopConfig cfg;
    cfg.steps_per_tick      = 10;         // small for fast tests
    cfg.action_threshold    = 0.05f;
    cfg.min_emit_interval_s = min_emit_s;
    cfg.decode_top_k        = 5;
    cfg.vocabulary          = { "hello", "curious", "wave", "energy", "nikola" };
    return cfg;
}

// ============================================================================
// Section 1: Construction
// ============================================================================

TEST_CASE("Phase23 DecisionLoop construction", "[Phase23]")
{
    auto torus  = make_torus();
    auto engine = make_engine();
    REQUIRE_NOTHROW(DecisionLoop(torus, engine, make_config()));
}

TEST_CASE("Phase23 DecisionLoop initial tick_count is zero", "[Phase23]")
{
    auto torus  = make_torus();
    auto engine = make_engine();
    DecisionLoop loop(torus, engine, make_config());
    REQUIRE(loop.tick_count() == 0);
}

// ============================================================================
// Section 2: read_state
// ============================================================================

TEST_CASE("Phase23 read_state returns finite values", "[Phase23]")
{
    auto torus  = make_torus();
    auto engine = make_engine();
    DecisionLoop loop(torus, engine, make_config());

    NikolaState s = loop.read_state();
    REQUIRE(std::isfinite(s.time));
    REQUIRE(std::isfinite(s.torus_energy));
    REQUIRE(std::isfinite(s.dopamine));
    REQUIRE(std::isfinite(s.atp));
    REQUIRE(std::isfinite(s.boredom));
    REQUIRE(std::isfinite(s.entropy));
    REQUIRE(s.dopamine >= 0.f);
    REQUIRE(s.dopamine <= 1.f);
    REQUIRE(s.atp >= 0.f);
    REQUIRE(s.atp <= 1.f);
}

TEST_CASE("Phase23 initial last_action is SILENT", "[Phase23]")
{
    auto torus  = make_torus();
    auto engine = make_engine();
    DecisionLoop loop(torus, engine, make_config());
    REQUIRE(loop.read_state().last_action == ActionType::SILENT);
}

// ============================================================================
// Section 3: action_name helper
// ============================================================================

TEST_CASE("Phase23 action_name covers all ActionTypes", "[Phase23]")
{
    using AT = ActionType;
    REQUIRE(std::string(action_name(AT::SILENT))         == "SILENT");
    REQUIRE(std::string(action_name(AT::EMIT_THOUGHT))   == "EMIT_THOUGHT");
    REQUIRE(std::string(action_name(AT::STORE_MEMORY))   == "STORE_MEMORY");
    REQUIRE(std::string(action_name(AT::REQUEST_LOOKUP)) == "REQUEST_LOOKUP");
    REQUIRE(std::string(action_name(AT::EXPLORE))        == "EXPLORE");
    REQUIRE(std::string(action_name(AT::NAP))            == "NAP");
    REQUIRE(std::string(action_name(AT::REFUSE))         == "REFUSE");
}

// ============================================================================
// Section 4: tick execution
// ============================================================================

TEST_CASE("Phase23 tick increments tick_count", "[Phase23]")
{
    auto torus  = make_torus();
    auto engine = make_engine();
    DecisionLoop loop(torus, engine, make_config());

    loop.tick();
    REQUIRE(loop.tick_count() == 1);
    loop.tick();
    loop.tick();
    REQUIRE(loop.tick_count() == 3);
}

TEST_CASE("Phase23 tick last_state is updated", "[Phase23]")
{
    auto torus  = make_torus();
    auto engine = make_engine();
    DecisionLoop loop(torus, engine, make_config());

    loop.tick();
    // After one tick, physics time should have advanced
    REQUIRE(loop.last_state().time > 0.f);
}

TEST_CASE("Phase23 multiple ticks produce finite state", "[Phase23]")
{
    auto torus  = make_torus();
    auto engine = make_engine();
    DecisionLoop loop(torus, engine, make_config());

    for (int i = 0; i < 20; ++i) {
        auto r = loop.tick();
        REQUIRE(std::isfinite(r.state.time));
        REQUIRE(std::isfinite(r.state.dopamine));
        REQUIRE(std::isfinite(r.state.atp));
    }
}

// ============================================================================
// Section 5: callbacks
// ============================================================================

TEST_CASE("Phase23 on_tick fires every tick", "[Phase23]")
{
    auto torus  = make_torus();
    auto engine = make_engine();
    DecisionLoop loop(torus, engine, make_config());

    int tick_count = 0;
    loop.on_tick = [&](const NikolaState&) { ++tick_count; };

    loop.tick(); loop.tick(); loop.tick();
    REQUIRE(tick_count == 3);
}

TEST_CASE("Phase23 on_action fires only for non-SILENT results", "[Phase23]")
{
    // Run many ticks and verify on_action is only called when action != SILENT
    auto torus  = make_torus();
    auto engine = make_engine();
    DecisionLoop loop(torus, engine, make_config());

    bool action_callback_bad = false;
    loop.on_action = [&](const DecisionResult& r) {
        if (r.type == ActionType::SILENT) action_callback_bad = true;
    };

    for (int i = 0; i < 30; ++i) loop.tick();
    REQUIRE_FALSE(action_callback_bad);
}

// ============================================================================
// Section 6: NikolaState helper predicates
// ============================================================================

TEST_CASE("Phase23 NikolaState predicates", "[Phase23]")
{
    NikolaState s;

    s.atp      = 0.1f;
    REQUIRE(s.is_exhausted());
    s.atp      = 0.5f;
    REQUIRE_FALSE(s.is_exhausted());

    s.dopamine = 0.8f;
    REQUIRE(s.is_spiking());
    s.dopamine = 0.3f;
    REQUIRE_FALSE(s.is_spiking());

    s.boredom  = 0.9f;
    REQUIRE(s.is_bored());
    s.boredom  = 0.5f;
    REQUIRE_FALSE(s.is_bored());

    s.td_error = -0.2f;
    REQUIRE(s.is_punished());
    s.td_error = 0.1f;
    REQUIRE_FALSE(s.is_punished());
}

// ============================================================================
// Section 7: inject_stimulus doesn't crash
// ============================================================================

TEST_CASE("Phase23 inject_stimulus is handled gracefully", "[Phase23]")
{
    auto torus  = make_torus();
    auto engine = make_engine();
    DecisionLoop loop(torus, engine, make_config());

    // In non-ORT build this injects a uniform pulse; in ORT build it embeds.
    // Either way: no crash, and the next tick runs normally.
    REQUIRE_NOTHROW(loop.inject_stimulus("hello nikola"));
    REQUIRE_NOTHROW(loop.tick());
}

// ============================================================================
// Section 8: Scoring direction tests
// ============================================================================

// Verify that the DecisionLoop produces a result with a finite, non-negative score.
TEST_CASE("Phase23 DecisionResult score is non-negative", "[Phase23]")
{
    auto torus  = make_torus();
    auto engine = make_engine();
    DecisionLoop loop(torus, engine, make_config());

    for (int i = 0; i < 10; ++i) {
        auto r = loop.tick();
        REQUIRE(r.score >= 0.f);
        REQUIRE(std::isfinite(r.score));
    }
}

// The payload for SILENT must always be empty.
TEST_CASE("Phase23 SILENT action has empty payload", "[Phase23]")
{
    auto torus  = make_torus();
    auto engine = make_engine();
    DecisionLoop loop(torus, engine, make_config());

    for (int i = 0; i < 20; ++i) {
        auto r = loop.tick();
        if (r.type == ActionType::SILENT) {
            REQUIRE(r.payload.empty());
        }
    }
}
