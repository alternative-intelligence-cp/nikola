// ============================================================================
// phase42_npt_reason_test.cpp   Phase 42 — NPT wired into DecisionLoop
// ============================================================================
//
// Tests:
//   §1   ActionType::REASON has integer value 9
//   §2   action_name(ActionType::REASON) == "REASON"
//   §3   DecisionLoop constructs without crash (NPT member initialised)
//   §4   Before REASON fires: last_npt_result().has_output == false
//   §5   REASON fires within the first N ticks on a freshly seeded torus
//   §6   After REASON fires: last_npt_result().has_output == true
//   §7   After REASON fires: output node count matches torus node count
//   §8   After REASON fires: result.payload contains "reasoning"
//   §9   After REASON fires: last_state().last_action == REASON
//   §10  After REASON fires: torus total_probability > 0 (field still alive)
//   §11  After REASON fires: torus total_probability is finite
//   §12  After REASON fires: last_npt_result().output.is_finite()
//   §13  After REASON fires: payload contains "entropy=" substring
//   §14  After REASON fires: payload contains "top_head=" substring
//   §15  Cooldown: the tick immediately after REASON does not fire REASON again
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/cognitive/cognitive_torus.hpp>

#include <string>
#include <vector>

using namespace nikola::autonomy;
using namespace nikola::cognitive;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static CognitiveTorus make_torus(int grid_n = 2) {
    return CognitiveTorus(grid_n);
}

static AutonomyEngine make_engine() {
    AutonomyConfig cfg;
    return AutonomyEngine(cfg);
}

/// Build a DecisionLoop with no ORT (non-ORT path) and a minimal vocab.
static DecisionLoop make_loop(CognitiveTorus& torus, AutonomyEngine& engine) {
    DecisionLoopConfig cfg;
    cfg.steps_per_tick   = 5;   // fast ticks for testing
    cfg.action_threshold = 0.0f;
    cfg.vocabulary       = {"hello", "nikola"};
    return DecisionLoop(torus, engine, cfg);
}

/// Run ticks until ActionType::REASON fires or until max_ticks is reached.
/// Returns the first DecisionResult with type == REASON, or the last result.
static DecisionResult run_until_reason(DecisionLoop& loop, int max_ticks = 30) {
    DecisionResult last;
    for (int i = 0; i < max_ticks; ++i) {
        last = loop.tick();
        if (last.type == ActionType::REASON) break;
    }
    return last;
}

// ---------------------------------------------------------------------------
// §1   ActionType::REASON has integer value 9
// ---------------------------------------------------------------------------
TEST_CASE("§1 ActionType::REASON == 9", "[Phase42][enum]") {
    REQUIRE(static_cast<int>(ActionType::REASON) == 9);
}

// ---------------------------------------------------------------------------
// §2   action_name returns "REASON"
// ---------------------------------------------------------------------------
TEST_CASE("§2 action_name(REASON) == \"REASON\"", "[Phase42][enum]") {
    REQUIRE(std::string(action_name(ActionType::REASON)) == "REASON");
}

// ---------------------------------------------------------------------------
// §3   DecisionLoop constructs without crash
// ---------------------------------------------------------------------------
TEST_CASE("§3 DecisionLoop with NPT constructs cleanly", "[Phase42][construct]") {
    auto torus  = make_torus();
    auto engine = make_engine();
    REQUIRE_NOTHROW(make_loop(torus, engine));
}

// ---------------------------------------------------------------------------
// §4   Before REASON fires: last_npt_result().has_output == false
// ---------------------------------------------------------------------------
TEST_CASE("§4 Before REASON fires: last_npt_result().has_output == false",
          "[Phase42][initial]") {
    auto torus  = make_torus();
    auto engine = make_engine();
    auto loop   = make_loop(torus, engine);

    // last_npt_result() starts with has_output = false (Phase 42 invariant).
    REQUIRE(loop.last_npt_result().has_output == false);
}

// ---------------------------------------------------------------------------
// §5   REASON fires within N ticks on a freshly seeded torus
// ---------------------------------------------------------------------------
TEST_CASE("§5 REASON fires within 30 ticks on freshly seeded torus",
          "[Phase42][firing]") {
    auto torus  = make_torus();
    auto engine = make_engine();
    auto loop   = make_loop(torus, engine);

    bool reason_fired = false;
    for (int i = 0; i < 30 && !reason_fired; ++i) {
        if (loop.tick().type == ActionType::REASON) reason_fired = true;
    }

    REQUIRE(reason_fired);
}

// Group §6-§15: run until REASON fires, then check all post-REASON invariants.
// Using a combined fixture so the loop context is shared after §5 fired once.

struct ReasonFired {
    CognitiveTorus  torus;
    AutonomyEngine  engine;
    DecisionLoop    loop;
    DecisionResult  result;
    size_t          torus_nodes_before;

    ReasonFired()
        : torus(make_torus())
        , engine(make_engine())
        , loop(make_loop(torus, engine))
    {
        torus_nodes_before = torus.num_nodes();
        result = run_until_reason(loop, 30);
    }

    bool fired() const { return result.type == ActionType::REASON; }
};

// ---------------------------------------------------------------------------
// §6   After REASON: last_npt_result().has_output == true
// ---------------------------------------------------------------------------
TEST_CASE("§6 After REASON: last_npt_result().has_output == true",
          "[Phase42][postfire]") {
    ReasonFired f;
    REQUIRE(f.fired());
    REQUIRE(f.loop.last_npt_result().has_output == true);
}

// ---------------------------------------------------------------------------
// §7   After REASON: output node count matches torus node count
// ---------------------------------------------------------------------------
TEST_CASE("§7 After REASON: output node count matches torus", "[Phase42][postfire]") {
    ReasonFired f;
    REQUIRE(f.fired());
    REQUIRE(f.loop.last_npt_result().output.num_nodes() == f.torus_nodes_before);
}

// ---------------------------------------------------------------------------
// §8   After REASON: payload contains "reasoning"
// ---------------------------------------------------------------------------
TEST_CASE("§8 After REASON: payload contains \"reasoning\"", "[Phase42][postfire]") {
    ReasonFired f;
    REQUIRE(f.fired());
    REQUIRE(f.result.payload.find("reasoning") != std::string::npos);
}

// ---------------------------------------------------------------------------
// §9   After REASON: last_state().last_action == REASON
// ---------------------------------------------------------------------------
TEST_CASE("§9 After REASON: last_state().last_action == REASON", "[Phase42][postfire]") {
    ReasonFired f;
    REQUIRE(f.fired());
    REQUIRE(f.loop.last_state().last_action == ActionType::REASON);
}

// ---------------------------------------------------------------------------
// §10  After REASON: torus still has finite, positive total probability
// ---------------------------------------------------------------------------
TEST_CASE("§10 After REASON: torus total_probability > 0", "[Phase42][postfire]") {
    ReasonFired f;
    REQUIRE(f.fired());
    REQUIRE(f.torus.total_probability() > 0.0);
}

// ---------------------------------------------------------------------------
// §11  After REASON: torus total_probability is finite
// ---------------------------------------------------------------------------
TEST_CASE("§11 After REASON: torus energy is finite", "[Phase42][postfire]") {
    ReasonFired f;
    REQUIRE(f.fired());
    REQUIRE(std::isfinite(static_cast<float>(f.torus.total_probability())));
}

// ---------------------------------------------------------------------------
// §12  After REASON: last_npt_result().output.is_finite()
// ---------------------------------------------------------------------------
TEST_CASE("§12 After REASON: NPT output WaveFunction is_finite()", "[Phase42][postfire]") {
    ReasonFired f;
    REQUIRE(f.fired());
    REQUIRE(f.loop.last_npt_result().output.is_finite());
}

// ---------------------------------------------------------------------------
// §13  After REASON: payload contains "entropy="
// ---------------------------------------------------------------------------
TEST_CASE("§13 After REASON: payload contains \"entropy=\"", "[Phase42][postfire]") {
    ReasonFired f;
    REQUIRE(f.fired());
    REQUIRE(f.result.payload.find("entropy=") != std::string::npos);
}

// ---------------------------------------------------------------------------
// §14  After REASON: payload contains "top_head="
// ---------------------------------------------------------------------------
TEST_CASE("§14 After REASON: payload contains \"top_head=\"", "[Phase42][postfire]") {
    ReasonFired f;
    REQUIRE(f.fired());
    REQUIRE(f.result.payload.find("top_head=") != std::string::npos);
}

// ---------------------------------------------------------------------------
// §15  Cooldown: tick immediately after REASON does not fire REASON again
// ---------------------------------------------------------------------------
TEST_CASE("§15 Cooldown: tick right after REASON does not fire REASON",
          "[Phase42][cooldown]") {
    ReasonFired f;
    REQUIRE(f.fired());

    // After REASON just fired, last_reason_time_ == now.
    // seconds_since(now) ≈ 0 < 3.0 → score_reason returns 0.
    // The very next tick should NOT select REASON.
    const DecisionResult next = f.loop.tick();
    REQUIRE(next.type != ActionType::REASON);
}
