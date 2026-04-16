/**
 * @file tests/unit/goal_system_test.cpp
 * @brief Phase 33 — GoalSystem unit tests (Catch2 v3).
 *
 * Tests:
 *   - GoalDAG: construction, add/remove, dependency edges, cycle detection,
 *     sub-goals, serialization round-trip
 *   - GoalSystem: creation, completion, abandonment, progress, priority
 *     scoring, active goal selection, dopamine integration, autonomous
 *     motivation, persistence
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <nikola/autonomy/goal_system.hpp>

#include <string>
#include <vector>

using namespace nikola::autonomy;

// ============================================================================
// GoalDAG — basic construction
// ============================================================================

TEST_CASE("GoalDAG — empty DAG", "[goal][dag][unit]") {
    GoalDAG dag;
    CHECK(dag.empty());
    CHECK(dag.size() == 0);
    CHECK(dag.all_goals().empty());
}

TEST_CASE("GoalDAG — add a single goal", "[goal][dag][unit]") {
    GoalDAG dag;
    Goal g;
    g.description = "Learn C++";
    g.tier = GoalTier::SHORT;
    g.urgency = 0.8f;
    g.importance = 0.9f;

    auto id = dag.add_goal(g);
    CHECK(id != 0);
    CHECK(dag.size() == 1);
    CHECK_FALSE(dag.empty());

    const auto* stored = dag.get(id);
    REQUIRE(stored != nullptr);
    CHECK(stored->description == "Learn C++");
    CHECK(stored->tier == GoalTier::SHORT);
    CHECK(stored->created_at > 0.0);
}

TEST_CASE("GoalDAG — add multiple goals", "[goal][dag][unit]") {
    GoalDAG dag;

    Goal g1; g1.description = "Goal A";
    Goal g2; g2.description = "Goal B";
    Goal g3; g3.description = "Goal C";

    auto id1 = dag.add_goal(g1);
    auto id2 = dag.add_goal(g2);
    auto id3 = dag.add_goal(g3);

    CHECK(dag.size() == 3);
    CHECK(id1 != id2);
    CHECK(id2 != id3);
    CHECK(dag.get(id1)->description == "Goal A");
    CHECK(dag.get(id2)->description == "Goal B");
    CHECK(dag.get(id3)->description == "Goal C");
}

TEST_CASE("GoalDAG — add goal with parent", "[goal][dag][unit]") {
    GoalDAG dag;

    Goal parent; parent.description = "Parent";
    auto pid = dag.add_goal(parent);

    Goal child; child.description = "Child"; child.parent_id = pid;
    auto cid = dag.add_goal(child);

    CHECK(cid != 0);
    CHECK(dag.get(cid)->parent_id == pid);

    auto subs = dag.sub_goals(pid);
    REQUIRE(subs.size() == 1);
    CHECK(subs[0] == cid);
}

TEST_CASE("GoalDAG — add goal with nonexistent parent fails", "[goal][dag][unit]") {
    GoalDAG dag;
    Goal g; g.description = "Orphan"; g.parent_id = 999;
    CHECK(dag.add_goal(g) == 0);
    CHECK(dag.empty());
}

TEST_CASE("GoalDAG — remove goal", "[goal][dag][unit]") {
    GoalDAG dag;
    Goal g; g.description = "Temp";
    auto id = dag.add_goal(g);
    CHECK(dag.size() == 1);

    CHECK(dag.remove_goal(id));
    CHECK(dag.empty());
    CHECK(dag.get(id) == nullptr);
}

TEST_CASE("GoalDAG — remove nonexistent goal fails", "[goal][dag][unit]") {
    GoalDAG dag;
    CHECK_FALSE(dag.remove_goal(999));
}

TEST_CASE("GoalDAG — remove goal re-parents children", "[goal][dag][unit]") {
    GoalDAG dag;

    Goal parent; parent.description = "Parent";
    auto pid = dag.add_goal(parent);

    Goal child; child.description = "Child"; child.parent_id = pid;
    auto cid = dag.add_goal(child);

    dag.remove_goal(pid);
    CHECK(dag.get(cid)->parent_id == 0);  // Re-parented to top-level
}

// ============================================================================
// GoalDAG — dependency edges
// ============================================================================

TEST_CASE("GoalDAG — add dependency", "[goal][dag][unit]") {
    GoalDAG dag;
    Goal g1; g1.description = "A";
    Goal g2; g2.description = "B";
    auto a = dag.add_goal(g1);
    auto b = dag.add_goal(g2);

    CHECK(dag.add_dependency(a, b));  // A depends on B

    auto deps = dag.dependencies_of(a);
    REQUIRE(deps.size() == 1);
    CHECK(deps[0] == b);

    auto rdeps = dag.dependents_of(b);
    REQUIRE(rdeps.size() == 1);
    CHECK(rdeps[0] == a);
}

TEST_CASE("GoalDAG — is_blocked with dependency", "[goal][dag][unit]") {
    GoalDAG dag;
    Goal g1; g1.description = "A";
    Goal g2; g2.description = "B";
    auto a = dag.add_goal(g1);
    auto b = dag.add_goal(g2);

    dag.add_dependency(a, b);

    CHECK(dag.is_blocked(a));  // B not completed yet
    CHECK_FALSE(dag.is_blocked(b));  // B has no deps

    // Complete B
    dag.get(b)->status = GoalStatus::COMPLETED;
    CHECK_FALSE(dag.is_blocked(a));  // Now unblocked
}

TEST_CASE("GoalDAG — is_blocked with sub-goals", "[goal][dag][unit]") {
    GoalDAG dag;
    Goal parent; parent.description = "Parent";
    auto pid = dag.add_goal(parent);

    Goal child; child.description = "Child"; child.parent_id = pid;
    auto cid = dag.add_goal(child);

    CHECK(dag.is_blocked(pid));  // Child not completed

    dag.get(cid)->status = GoalStatus::COMPLETED;
    CHECK_FALSE(dag.is_blocked(pid));  // Now unblocked
}

TEST_CASE("GoalDAG — dependency on nonexistent goal fails", "[goal][dag][unit]") {
    GoalDAG dag;
    Goal g; g.description = "A";
    auto a = dag.add_goal(g);

    CHECK_FALSE(dag.add_dependency(a, 999));
    CHECK_FALSE(dag.add_dependency(999, a));
}

TEST_CASE("GoalDAG — duplicate dependency is idempotent", "[goal][dag][unit]") {
    GoalDAG dag;
    Goal g1; g1.description = "A";
    Goal g2; g2.description = "B";
    auto a = dag.add_goal(g1);
    auto b = dag.add_goal(g2);

    CHECK(dag.add_dependency(a, b));
    CHECK(dag.add_dependency(a, b));  // Idempotent
    CHECK(dag.dependencies_of(a).size() == 1);
}

TEST_CASE("GoalDAG — remove dependency", "[goal][dag][unit]") {
    GoalDAG dag;
    Goal g1; g1.description = "A";
    Goal g2; g2.description = "B";
    auto a = dag.add_goal(g1);
    auto b = dag.add_goal(g2);

    dag.add_dependency(a, b);
    CHECK(dag.remove_dependency(a, b));
    CHECK(dag.dependencies_of(a).empty());
    CHECK(dag.dependents_of(b).empty());
}

// ============================================================================
// GoalDAG — cycle detection
// ============================================================================

TEST_CASE("GoalDAG — self-dependency rejected", "[goal][dag][unit]") {
    GoalDAG dag;
    Goal g; g.description = "A";
    auto a = dag.add_goal(g);

    CHECK_FALSE(dag.add_dependency(a, a));
}

TEST_CASE("GoalDAG — direct cycle rejected", "[goal][dag][unit]") {
    GoalDAG dag;
    Goal g1; g1.description = "A";
    Goal g2; g2.description = "B";
    auto a = dag.add_goal(g1);
    auto b = dag.add_goal(g2);

    CHECK(dag.add_dependency(a, b));   // A → B
    CHECK_FALSE(dag.add_dependency(b, a));  // B → A would create cycle
}

TEST_CASE("GoalDAG — transitive cycle rejected", "[goal][dag][unit]") {
    GoalDAG dag;
    Goal g1; g1.description = "A";
    Goal g2; g2.description = "B";
    Goal g3; g3.description = "C";
    auto a = dag.add_goal(g1);
    auto b = dag.add_goal(g2);
    auto c = dag.add_goal(g3);

    CHECK(dag.add_dependency(a, b));   // A → B
    CHECK(dag.add_dependency(b, c));   // B → C
    CHECK_FALSE(dag.add_dependency(c, a));  // C → A would create cycle
}

TEST_CASE("GoalDAG — diamond dependency is fine", "[goal][dag][unit]") {
    GoalDAG dag;
    Goal g1; g1.description = "A";
    Goal g2; g2.description = "B";
    Goal g3; g3.description = "C";
    Goal g4; g4.description = "D";
    auto a = dag.add_goal(g1);
    auto b = dag.add_goal(g2);
    auto c = dag.add_goal(g3);
    auto d = dag.add_goal(g4);

    CHECK(dag.add_dependency(a, b));  // A → B
    CHECK(dag.add_dependency(a, c));  // A → C
    CHECK(dag.add_dependency(b, d));  // B → D
    CHECK(dag.add_dependency(c, d));  // C → D (diamond, not cycle)
}

TEST_CASE("GoalDAG — would_create_cycle is non-destructive", "[goal][dag][unit]") {
    GoalDAG dag;
    Goal g1; g1.description = "A";
    Goal g2; g2.description = "B";
    auto a = dag.add_goal(g1);
    auto b = dag.add_goal(g2);

    dag.add_dependency(a, b);

    CHECK(dag.would_create_cycle(b, a));    // Would create cycle
    CHECK_FALSE(dag.would_create_cycle(a, b));  // Already exists, no new cycle

    // DAG unchanged
    CHECK(dag.dependencies_of(a).size() == 1);
    CHECK(dag.dependents_of(a).empty());
}

// ============================================================================
// GoalDAG — serialization
// ============================================================================

TEST_CASE("GoalDAG — serialize/deserialize round-trip", "[goal][dag][unit]") {
    GoalDAG dag;

    Goal g1; g1.description = "Learn Rust"; g1.tier = GoalTier::LONG;
    g1.urgency = 0.3f; g1.importance = 0.9f;
    auto id1 = dag.add_goal(g1);

    Goal g2; g2.description = "Read docs"; g2.tier = GoalTier::SHORT;
    g2.parent_id = id1; g2.urgency = 0.8f; g2.importance = 0.7f;
    auto id2 = dag.add_goal(g2);

    Goal g3; g3.description = "Build project"; g3.tier = GoalTier::MID;
    g3.parent_id = id1; g3.urgency = 0.5f; g3.importance = 0.8f;
    auto id3 = dag.add_goal(g3);

    dag.add_dependency(id3, id2);  // Build depends on Read

    auto buf = dag.serialize();
    CHECK_FALSE(buf.empty());

    GoalDAG dag2;
    CHECK(dag2.deserialize(buf));
    CHECK(dag2.size() == 3);

    const auto* r1 = dag2.get(id1);
    REQUIRE(r1 != nullptr);
    CHECK(r1->description == "Learn Rust");
    CHECK(r1->tier == GoalTier::LONG);

    const auto* r2 = dag2.get(id2);
    REQUIRE(r2 != nullptr);
    CHECK(r2->parent_id == id1);

    auto deps = dag2.dependencies_of(id3);
    REQUIRE(deps.size() == 1);
    CHECK(deps[0] == id2);
}

TEST_CASE("GoalDAG — deserialize empty buffer fails", "[goal][dag][unit]") {
    GoalDAG dag;
    CHECK_FALSE(dag.deserialize(nullptr, 0));
    std::vector<uint8_t> garbage = {1, 2, 3, 4};
    CHECK_FALSE(dag.deserialize(garbage));
}

TEST_CASE("GoalDAG — clear resets everything", "[goal][dag][unit]") {
    GoalDAG dag;
    Goal g; g.description = "A";
    dag.add_goal(g);
    dag.clear();
    CHECK(dag.empty());
}

// ============================================================================
// Goal struct
// ============================================================================

TEST_CASE("Goal — is_actionable and is_terminal", "[goal][unit]") {
    Goal g;
    g.status = GoalStatus::ACTIVE;
    CHECK(g.is_actionable());
    CHECK_FALSE(g.is_terminal());

    g.status = GoalStatus::BLOCKED;
    CHECK_FALSE(g.is_actionable());
    CHECK_FALSE(g.is_terminal());

    g.status = GoalStatus::COMPLETED;
    CHECK_FALSE(g.is_actionable());
    CHECK(g.is_terminal());

    g.status = GoalStatus::ABANDONED;
    CHECK_FALSE(g.is_actionable());
    CHECK(g.is_terminal());

    g.status = GoalStatus::PAUSED;
    CHECK_FALSE(g.is_actionable());
    CHECK_FALSE(g.is_terminal());
}

TEST_CASE("Goal — recompute_priority", "[goal][unit]") {
    Goal g;
    g.urgency = 0.8f;
    g.importance = 0.5f;
    g.feasibility = 1.0f;
    g.recompute_priority();
    CHECK(g.priority == Catch::Approx(0.4f));
}

TEST_CASE("GoalTier — string labels", "[goal][unit]") {
    CHECK(std::string(goal_tier_str(GoalTier::SHORT)) == "SHORT");
    CHECK(std::string(goal_tier_str(GoalTier::MID)) == "MID");
    CHECK(std::string(goal_tier_str(GoalTier::LONG)) == "LONG");
}

TEST_CASE("GoalStatus — string labels", "[goal][unit]") {
    CHECK(std::string(goal_status_str(GoalStatus::ACTIVE)) == "ACTIVE");
    CHECK(std::string(goal_status_str(GoalStatus::BLOCKED)) == "BLOCKED");
    CHECK(std::string(goal_status_str(GoalStatus::COMPLETED)) == "COMPLETED");
    CHECK(std::string(goal_status_str(GoalStatus::ABANDONED)) == "ABANDONED");
    CHECK(std::string(goal_status_str(GoalStatus::PAUSED)) == "PAUSED");
}

// ============================================================================
// GoalSystem — creation and lifecycle
// ============================================================================

TEST_CASE("GoalSystem — create goal", "[goal][system][unit]") {
    GoalSystem sys;
    auto id = sys.create_goal("Learn C++", GoalTier::SHORT, 0.8f, 0.9f);
    CHECK(id != 0);
    CHECK(sys.goal_count() == 1);

    const auto* g = sys.get_goal(id);
    REQUIRE(g != nullptr);
    CHECK(g->description == "Learn C++");
    CHECK(g->tier == GoalTier::SHORT);
}

TEST_CASE("GoalSystem — create sub-goal", "[goal][system][unit]") {
    GoalSystem sys;
    auto parent = sys.create_goal("Master C++", GoalTier::LONG);
    auto child = sys.create_goal("Read Stroustrup", GoalTier::SHORT, 0.7f, 0.8f, parent);
    CHECK(child != 0);
    CHECK(sys.get_goal(child)->parent_id == parent);
}

TEST_CASE("GoalSystem — complete goal triggers reward", "[goal][system][unit]") {
    GoalSystem sys;
    float total_reward = 0.0f;
    std::string last_desc;
    sys.set_reward_fn([&](float d, const std::string& desc) {
        total_reward += d;
        last_desc = desc;
    });

    auto id = sys.create_goal("Test goal", GoalTier::SHORT);
    CHECK(sys.complete_goal(id));
    CHECK(total_reward == Catch::Approx(0.3f));  // SHORT reward
    CHECK(last_desc.find("goal_completed") != std::string::npos);
    CHECK(sys.completed_count() == 1);
}

TEST_CASE("GoalSystem — complete goal with different tiers", "[goal][system][unit]") {
    float reward = 0.0f;
    auto test_tier = [&](GoalTier tier, float expected) {
        GoalSystem sys;
        reward = 0.0f;
        sys.set_reward_fn([&](float d, const std::string&) { reward += d; });
        auto id = sys.create_goal("G", tier);
        sys.complete_goal(id);
        CHECK(reward == Catch::Approx(expected));
    };

    test_tier(GoalTier::SHORT, 0.3f);
    test_tier(GoalTier::MID, 0.6f);
    test_tier(GoalTier::LONG, 1.0f);
}

TEST_CASE("GoalSystem — cannot complete goal with active sub-goals",
          "[goal][system][unit]") {
    GoalSystem sys;
    auto parent = sys.create_goal("Parent", GoalTier::MID);
    sys.create_goal("Child", GoalTier::SHORT, 0.5f, 0.5f, parent);

    CHECK_FALSE(sys.complete_goal(parent));  // Child still active
}

TEST_CASE("GoalSystem — complete goal after sub-goals done", "[goal][system][unit]") {
    GoalSystem sys;
    auto parent = sys.create_goal("Parent", GoalTier::MID);
    auto child = sys.create_goal("Child", GoalTier::SHORT, 0.5f, 0.5f, parent);

    CHECK(sys.complete_goal(child));
    CHECK(sys.complete_goal(parent));
    CHECK(sys.completed_count() == 2);
}

TEST_CASE("GoalSystem — abandon goal triggers penalty", "[goal][system][unit]") {
    GoalSystem sys;
    float total_reward = 0.0f;
    sys.set_reward_fn([&](float d, const std::string&) { total_reward += d; });

    auto id = sys.create_goal("Doomed", GoalTier::SHORT);
    CHECK(sys.abandon_goal(id));
    CHECK(total_reward == Catch::Approx(-0.15f));
    CHECK(sys.abandoned_count() == 1);
}

TEST_CASE("GoalSystem — cannot complete already completed goal",
          "[goal][system][unit]") {
    GoalSystem sys;
    auto id = sys.create_goal("G", GoalTier::SHORT);
    CHECK(sys.complete_goal(id));
    CHECK_FALSE(sys.complete_goal(id));
}

TEST_CASE("GoalSystem — cannot abandon completed goal", "[goal][system][unit]") {
    GoalSystem sys;
    auto id = sys.create_goal("G", GoalTier::SHORT);
    sys.complete_goal(id);
    CHECK_FALSE(sys.abandon_goal(id));
}

TEST_CASE("GoalSystem — pause and resume", "[goal][system][unit]") {
    GoalSystem sys;
    auto id = sys.create_goal("G", GoalTier::SHORT);

    CHECK(sys.pause_goal(id));
    CHECK(sys.get_goal(id)->status == GoalStatus::PAUSED);

    CHECK(sys.resume_goal(id));
    CHECK(sys.get_goal(id)->status == GoalStatus::ACTIVE);
}

TEST_CASE("GoalSystem — cannot pause non-active goal", "[goal][system][unit]") {
    GoalSystem sys;
    auto id = sys.create_goal("G", GoalTier::SHORT);
    sys.complete_goal(id);
    CHECK_FALSE(sys.pause_goal(id));
}

// ============================================================================
// GoalSystem — progress and proportional dopamine
// ============================================================================

TEST_CASE("GoalSystem — update_progress triggers proportional reward",
          "[goal][system][unit]") {
    GoalSystem sys;
    float total_reward = 0.0f;
    sys.set_reward_fn([&](float d, const std::string&) { total_reward += d; });

    auto id = sys.create_goal("Progressive", GoalTier::MID);
    CHECK(sys.update_progress(id, 0.5f));
    CHECK(total_reward > 0.0f);
    CHECK(sys.get_goal(id)->progress == Catch::Approx(0.5f));
}

TEST_CASE("GoalSystem — progress backward doesn't reward", "[goal][system][unit]") {
    GoalSystem sys;
    float total_reward = 0.0f;
    sys.set_reward_fn([&](float d, const std::string&) { total_reward += d; });

    auto id = sys.create_goal("G", GoalTier::SHORT);
    sys.update_progress(id, 0.5f);
    float after_50 = total_reward;

    sys.update_progress(id, 0.3f);  // Going backward
    CHECK(total_reward == Catch::Approx(after_50));  // No additional reward
}

TEST_CASE("GoalSystem — progress clamped to [0, 1]", "[goal][system][unit]") {
    GoalSystem sys;
    auto id = sys.create_goal("G", GoalTier::SHORT);
    sys.update_progress(id, 1.5f);
    CHECK(sys.get_goal(id)->progress == Catch::Approx(1.0f));

    sys.update_progress(id, -0.5f);
    CHECK(sys.get_goal(id)->progress == Catch::Approx(0.0f));
}

// ============================================================================
// GoalSystem — active goal selection
// ============================================================================

TEST_CASE("GoalSystem — active_goal returns highest priority", "[goal][system][unit]") {
    GoalSystem sys;
    sys.create_goal("Low", GoalTier::SHORT, 0.2f, 0.3f);
    auto high_id = sys.create_goal("High", GoalTier::SHORT, 0.9f, 0.9f);

    const auto* active = sys.active_goal();
    REQUIRE(active != nullptr);
    CHECK(active->id == high_id);
}

TEST_CASE("GoalSystem — active_goal skips blocked goals", "[goal][system][unit]") {
    GoalSystem sys;
    auto high = sys.create_goal("High priority", GoalTier::MID, 0.9f, 0.9f);
    auto dep = sys.create_goal("Dependency", GoalTier::SHORT, 0.3f, 0.5f);
    auto low = sys.create_goal("Low but ready", GoalTier::SHORT, 0.4f, 0.5f);
    (void)low;

    sys.add_dependency(high, dep);

    const auto* active = sys.active_goal();
    REQUIRE(active != nullptr);
    // High is blocked, so active should be dep or low (whichever has higher priority)
    CHECK(active->id != high);
}

TEST_CASE("GoalSystem — active_goal returns nullptr when empty",
          "[goal][system][unit]") {
    GoalSystem sys;
    CHECK(sys.active_goal() == nullptr);
}

TEST_CASE("GoalSystem — active_goal returns nullptr when all completed",
          "[goal][system][unit]") {
    GoalSystem sys;
    auto id = sys.create_goal("Done", GoalTier::SHORT);
    sys.complete_goal(id);
    CHECK(sys.active_goal() == nullptr);
}

// ============================================================================
// GoalSystem — dependency + cycle detection
// ============================================================================

TEST_CASE("GoalSystem — add dependency", "[goal][system][unit]") {
    GoalSystem sys;
    auto a = sys.create_goal("A", GoalTier::SHORT);
    auto b = sys.create_goal("B", GoalTier::SHORT);

    CHECK(sys.add_dependency(a, b));
}

TEST_CASE("GoalSystem — cycle detection through GoalSystem", "[goal][system][unit]") {
    GoalSystem sys;
    auto a = sys.create_goal("A", GoalTier::SHORT);
    auto b = sys.create_goal("B", GoalTier::SHORT);

    CHECK(sys.add_dependency(a, b));
    CHECK_FALSE(sys.add_dependency(b, a));  // Would create cycle
}

// ============================================================================
// GoalSystem — blocked status management
// ============================================================================

TEST_CASE("GoalSystem — dependency completion unblocks dependent",
          "[goal][system][unit]") {
    GoalSystem sys;
    auto a = sys.create_goal("Dependent", GoalTier::MID, 0.9f, 0.9f);
    auto b = sys.create_goal("Prerequisite", GoalTier::SHORT, 0.5f, 0.5f);

    sys.add_dependency(a, b);

    // A should be blocked
    CHECK(sys.get_goal(a)->status == GoalStatus::BLOCKED);

    // Complete B
    sys.complete_goal(b);

    // A should now be active
    CHECK(sys.get_goal(a)->status == GoalStatus::ACTIVE);
}

// ============================================================================
// GoalSystem — autonomous motivation
// ============================================================================

TEST_CASE("GoalSystem — high boredom generates training goal",
          "[goal][system][unit]") {
    GoalSystem sys;
    CHECK(sys.goal_count() == 0);

    sys.check_motivation(0.9f, 0.5f, 1000);

    CHECK(sys.goal_count() == 1);
    const auto* g = sys.active_goal();
    REQUIRE(g != nullptr);
    CHECK(g->description.find("training") != std::string::npos);
}

TEST_CASE("GoalSystem — low boredom does not generate goal",
          "[goal][system][unit]") {
    GoalSystem sys;
    sys.check_motivation(0.3f, 0.5f, 1000);
    CHECK(sys.goal_count() == 0);
}

TEST_CASE("GoalSystem — sustained low dopamine generates exploration goal",
          "[goal][system][unit]") {
    GoalSystem sys;

    // Simulate 50+ ticks of low dopamine, with ticks spaced far enough apart
    // for rate limiting (> 500 apart from tick 0)
    for (int i = 0; i < 55; ++i) {
        sys.check_motivation(0.3f, 0.1f, static_cast<uint64_t>(1000 + i));
    }

    // Should have generated an exploration goal
    bool found_exploration = false;
    for (const auto* g : sys.all_goals()) {
        if (g->description.find("Exploration") != std::string::npos) {
            found_exploration = true;
        }
    }
    CHECK(found_exploration);
}

TEST_CASE("GoalSystem — training goals rate-limited", "[goal][system][unit]") {
    GoalSystem sys;
    sys.check_motivation(0.9f, 0.5f, 1000);
    CHECK(sys.goal_count() == 1);

    // Same tick range — should not create another
    sys.check_motivation(0.9f, 0.5f, 1001);
    CHECK(sys.goal_count() == 1);

    // Far enough apart — should create another
    sys.check_motivation(0.9f, 0.5f, 2000);
    CHECK(sys.goal_count() == 2);
}

// ============================================================================
// GoalSystem — persistence
// ============================================================================

TEST_CASE("GoalSystem — serialize/deserialize round-trip", "[goal][system][unit]") {
    GoalSystem sys;
    auto a = sys.create_goal("Alpha", GoalTier::LONG, 0.5f, 0.9f);
    auto b = sys.create_goal("Beta", GoalTier::SHORT, 0.8f, 0.7f, a);
    sys.add_dependency(a, b);
    sys.complete_goal(b);

    auto buf = sys.serialize();
    CHECK_FALSE(buf.empty());

    GoalSystem sys2;
    CHECK(sys2.deserialize(buf));
    CHECK(sys2.goal_count() == 2);
    CHECK(sys2.completed_count() == 1);

    const auto* ra = sys2.get_goal(a);
    REQUIRE(ra != nullptr);
    CHECK(ra->description == "Alpha");
    CHECK(ra->tier == GoalTier::LONG);

    const auto* rb = sys2.get_goal(b);
    REQUIRE(rb != nullptr);
    CHECK(rb->status == GoalStatus::COMPLETED);
}

TEST_CASE("GoalSystem — deserialize empty buffer fails", "[goal][system][unit]") {
    GoalSystem sys;
    CHECK_FALSE(sys.deserialize(nullptr, 0));
}

// ============================================================================
// GoalSystem — no reward_fn doesn't crash
// ============================================================================

TEST_CASE("GoalSystem — operations work without reward_fn", "[goal][system][unit]") {
    GoalSystem sys;
    auto id = sys.create_goal("No callback", GoalTier::SHORT);
    CHECK(sys.update_progress(id, 0.5f));
    CHECK(sys.complete_goal(id));
    // Should not crash
}

// ============================================================================
// GoalSystem — goals_with_status
// ============================================================================

TEST_CASE("GoalSystem — goals_with_status filtering", "[goal][system][unit]") {
    GoalSystem sys;
    auto a = sys.create_goal("A", GoalTier::SHORT);
    sys.create_goal("B", GoalTier::SHORT);
    sys.complete_goal(a);

    auto completed = sys.goals_with_status(GoalStatus::COMPLETED);
    CHECK(completed.size() == 1);

    auto active = sys.goals_with_status(GoalStatus::ACTIVE);
    CHECK(active.size() == 1);
}
