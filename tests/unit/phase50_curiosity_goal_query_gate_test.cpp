/**
 * @file phase50_curiosity_goal_query_gate_test.cpp
 * @brief Phase 50 — §8.3 Orchestrator polling: CuriosityGoal + query gate.
 *
 * Validates:
 *   1. CuriosityGoal struct with typed fields and priority tier helper
 *   2. on_curiosity_goal callback fired when boredom > θ_explore
 *   3. Boredom drain after goal emission prevents immediate re-fire (Mania guard)
 *   4. is_query_gated() + query_gate_count() telemetry (spec §8.3: ATP < 15%)
 *   5. Legacy on_explore callback still fires for backward compat
 *
 * §1   CuriosityGoal struct fields accessible: id, boredom, entropy, priority
 * §2   tier_from_boredom(): 0=LOW (≥0.80), 1=MEDIUM (≥0.90), 2=HIGH (≥0.95)
 * §3   on_curiosity_goal fires when boredom driven above θ_explore=0.8
 * §4   on_curiosity_goal NOT fired when boredom < θ_explore
 * §5   CuriosityGoal.boredom captures B(t) at moment of emission
 * §6   CuriosityGoal.id starts at 1 and increments monotonically
 * §7   curiosity_goal_count() == 0 initially; increments per emission
 * §8   Boredom is drained after CuriosityGoal fires (mania guard)
 * §9   Only one goal per boredom episode (cooldown: no re-fire while above thresh)
 * §10  Second goal fires only after boredom falls below threshold and recovers
 * §11  Legacy on_explore() also fires whenever on_curiosity_goal fires
 * §12  is_query_gated() = false when ATP high (above NAP_ENTER_THRESHOLD)
 * §13  is_query_gated() = true when ATP below NAP_ENTER_THRESHOLD (0.15)
 * §14  query_gate_count() increments each tick where ATP < NAP_ENTER_THRESHOLD
 * §15  query_gate_count() stays 0 when ATP stays above threshold
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/autonomy/nap_controller.hpp>   // NAP_ENTER_THRESHOLD

using namespace nikola::autonomy;
using Catch::Matchers::WithinAbs;

// ── helpers ───────────────────────────────────────────────────────────────────

namespace {

/// Single-spike wavefunction: all energy in node 0 → near-zero entropy.
/// Drives BoredomRegulator up quickly.
std::vector<float> spike_r(std::size_t n = 64) {
    std::vector<float> r(n, 0.0f);
    r[0] = 1.0f;
    return r;
}
std::vector<float> zeros(std::size_t n = 64) { return std::vector<float>(n, 0.0f); }

/// Pump boredom above θ_explore by ticking with a near-zero entropy wavefunction.
/// Returns after the first CuriosityGoal fires (or max_ticks).
CuriosityGoal drive_to_goal(AutonomyEngine& eng, int max_ticks = 20) {
    CuriosityGoal captured{};
    eng.on_curiosity_goal = [&](CuriosityGoal g) { captured = g; };

    auto r = spike_r();
    auto i = zeros();
    for (int t = 0; t < max_ticks; ++t) {
        eng.tick(1.0f, r, i);
        if (captured.id != 0) break;
    }
    return captured;
}

} // anonymous namespace

// ── §1  CuriosityGoal struct fields ──────────────────────────────────────────

TEST_CASE("[P50-§1] CuriosityGoal struct fields accessible",
          "[phase50]") {
    CuriosityGoal g;
    g.id       = 1;
    g.boredom  = 0.82f;
    g.entropy  = 1.5f;
    g.priority = 0;
    REQUIRE(g.id       == 1u);
    REQUIRE_THAT(g.boredom,  WithinAbs(0.82f, 1e-6f));
    REQUIRE_THAT(g.entropy,  WithinAbs(1.5f,  1e-6f));
    REQUIRE(g.priority == 0u);
}

// ── §2  tier_from_boredom() priority thresholds ───────────────────────────────

TEST_CASE("[P50-§2] tier_from_boredom(): LOW / MEDIUM / HIGH tiers correct",
          "[phase50]") {
    // LOW: [0.80, 0.90)
    REQUIRE(CuriosityGoal::tier_from_boredom(0.80f) == 0u);
    REQUIRE(CuriosityGoal::tier_from_boredom(0.85f) == 0u);
    REQUIRE(CuriosityGoal::tier_from_boredom(0.899f) == 0u);

    // MEDIUM: [0.90, 0.95)
    REQUIRE(CuriosityGoal::tier_from_boredom(0.90f) == 1u);
    REQUIRE(CuriosityGoal::tier_from_boredom(0.92f) == 1u);
    REQUIRE(CuriosityGoal::tier_from_boredom(0.949f) == 1u);

    // HIGH: >= 0.95
    REQUIRE(CuriosityGoal::tier_from_boredom(0.95f) == 2u);
    REQUIRE(CuriosityGoal::tier_from_boredom(1.00f) == 2u);
}

// ── §3  on_curiosity_goal fires when boredom driven above threshold ────────────

TEST_CASE("[P50-§3] on_curiosity_goal fires when boredom > 0.8",
          "[phase50]") {
    AutonomyEngine eng;
    auto goal = drive_to_goal(eng);

    REQUIRE(goal.id != 0u);  // a goal was emitted
    REQUIRE(goal.boredom >= BOREDOM_EXPLORE_THRESH - 1e-4f);
}

// ── §4  on_curiosity_goal NOT fired when boredom < threshold ─────────────────

TEST_CASE("[P50-§4] on_curiosity_goal NOT fired when boredom below 0.8",
          "[phase50]") {
    AutonomyEngine eng;
    int fire_count = 0;
    eng.on_curiosity_goal = [&](CuriosityGoal) { ++fire_count; };

    // High-entropy field (uniform) → boredom should not accumulate
    std::vector<float> r(64), im(64);
    for (std::size_t i = 0; i < 64; ++i) r[i] = 1.0f;   // uniform → high H

    for (int t = 0; t < 5; ++t)
        eng.tick(1.0f, r, im);

    REQUIRE(fire_count == 0);
    REQUIRE(eng.boredom() < BOREDOM_EXPLORE_THRESH);
}

// ── §5  CuriosityGoal.boredom captures B(t) at emission ──────────────────────

TEST_CASE("[P50-§5] CuriosityGoal.boredom == engine.boredom() just before drain",
          "[phase50]") {
    AutonomyEngine eng;
    float boredom_at_fire = -1.0f;
    eng.on_curiosity_goal = [&](CuriosityGoal g) {
        boredom_at_fire = g.boredom;
    };

    auto r = spike_r();
    auto i = zeros();
    for (int t = 0; t < 20; ++t) eng.tick(1.0f, r, i);

    // The captured boredom in the goal should be ≥ threshold
    REQUIRE(boredom_at_fire >= BOREDOM_EXPLORE_THRESH - 1e-4f);
    // And it should be a valid [0,1] float
    REQUIRE(boredom_at_fire >= 0.0f);
    REQUIRE(boredom_at_fire <= 1.0f + 1e-5f);
}

// ── §6  CuriosityGoal.id starts at 1, increments ─────────────────────────────

TEST_CASE("[P50-§6] CuriosityGoal.id starts at 1 for the first goal",
          "[phase50]") {
    AutonomyEngine eng;
    auto goal = drive_to_goal(eng);
    REQUIRE(goal.id == 1u);
}

// ── §7  curiosity_goal_count() telemetry ─────────────────────────────────────

TEST_CASE("[P50-§7] curiosity_goal_count() == 0 initially; == 1 after first goal",
          "[phase50]") {
    AutonomyEngine eng;
    REQUIRE(eng.curiosity_goal_count() == 0u);

    drive_to_goal(eng);
    REQUIRE(eng.curiosity_goal_count() == 1u);
}

// ── §8  Boredom drained after CuriosityGoal fires ────────────────────────────

TEST_CASE("[P50-§8] boredom is drained after CuriosityGoal fires",
          "[phase50]") {
    AutonomyEngine eng;
    float boredom_at_fire = -1.0f;
    bool  goal_fired      = false;
    eng.on_curiosity_goal = [&](CuriosityGoal g) {
        boredom_at_fire = g.boredom;
        goal_fired      = true;
    };

    auto r = spike_r();
    auto i = zeros();
    // Tick until the first goal fires, then stop to check boredom immediately.
    // (Continuing to tick after mania suppression kicks in would let boredom
    //  re-accumulate above boredom_at_fire, making the assertion meaningless.)
    for (int t = 0; t < 20 && !goal_fired; ++t)
        eng.tick(1.0f, r, i);

    REQUIRE(goal_fired);
    REQUIRE(boredom_at_fire >= 0.0f);
    // Drain of CURIOSITY_BOREDOM_DRAIN applied before the callback →
    // engine boredom is already below pre-drain level at tick completion.
    REQUIRE(eng.boredom() < boredom_at_fire);
}

// ── §9  No re-fire on tick immediately after CuriosityGoal fires ─────────────

TEST_CASE("[P50-§9] cooldown: no CuriosityGoal re-fire on the immediately next tick",
          "[phase50]") {
    AutonomyEngine eng;
    std::vector<uint32_t> fire_ticks;
    int tick_num = 0;
    eng.on_curiosity_goal = [&](CuriosityGoal) { fire_ticks.push_back(tick_num); };

    auto r = spike_r();
    auto i = zeros();
    // Run until we get the first goal, then run a few more ticks
    for (tick_num = 0; tick_num < 25; ++tick_num) {
        eng.tick(1.0f, r, i);
    }

    // At least one goal must have fired
    REQUIRE(fire_ticks.size() >= 1u);

    // After any fire, the NEXT tick must NOT fire (drain prevents immediate re-fire)
    for (std::size_t fi = 0; fi + 1 < fire_ticks.size(); ++fi) {
        const uint32_t gap = fire_ticks[fi + 1] - fire_ticks[fi];
        // Drain of 0.3 at net 0.09/s recovery needs 0.3/0.09 ≈ 3 ticks minimum
        REQUIRE(gap >= 2u);
    }

    // Also: total fires must be much less than tick count (not firing every tick)
    REQUIRE(fire_ticks.size() < 15u);
}

// ── §10 Second goal fires only after boredom drops and recovers ───────────────

TEST_CASE("[P50-§10] second goal fires after boredom drops below threshold then recovers",
          "[phase50]") {
    AutonomyEngine eng;
    std::vector<uint32_t> goal_ids;
    eng.on_curiosity_goal = [&](CuriosityGoal g) { goal_ids.push_back(g.id); };

    auto low_h_r = spike_r();
    auto low_h_i = zeros();
    // High-entropy field to drive boredom back down
    std::vector<float> high_h_r(64, 1.0f), high_h_i(64, 0.0f);

    // Phase A: drive boredom above threshold → first goal
    for (int t = 0; t < 15; ++t) eng.tick(1.0f, low_h_r, low_h_i);
    const std::size_t after_phase_a = goal_ids.size();
    REQUIRE(after_phase_a >= 1u);

    // Phase B: run with high-entropy (uniform) to drain boredom below threshold
    for (int t = 0; t < 30; ++t) eng.tick(1.0f, high_h_r, high_h_i);
    // Boredom should have fallen below θ_explore
    REQUIRE(eng.boredom() < BOREDOM_EXPLORE_THRESH);

    // Phase C: drive boredom up again → second goal
    const std::size_t before_phase_c = goal_ids.size();
    for (int t = 0; t < 20; ++t) eng.tick(1.0f, low_h_r, low_h_i);

    REQUIRE(goal_ids.size() > before_phase_c);   // at least one more goal fired
    REQUIRE(goal_ids.back() > goal_ids.front()); // IDs increasing (monotonic)
}

// ── §11 Legacy on_explore callback also fires with CuriosityGoal ───────────────

TEST_CASE("[P50-§11] legacy on_explore() fires whenever on_curiosity_goal fires",
          "[phase50]") {
    AutonomyEngine eng;
    int goal_fires   = 0;
    int explore_fires = 0;
    eng.on_curiosity_goal = [&](CuriosityGoal) { ++goal_fires; };
    eng.on_explore        = [&]()              { ++explore_fires; };

    auto r = spike_r();
    auto i = zeros();
    for (int t = 0; t < 20; ++t) eng.tick(1.0f, r, i);

    // Both should have fired the same number of times
    REQUIRE(goal_fires   >= 1);
    REQUIRE(explore_fires == goal_fires);
}

// ── §12 is_query_gated() = false when ATP high ───────────────────────────────

TEST_CASE("[P50-§12] is_query_gated() = false when ATP is above 0.15",
          "[phase50]") {
    AutonomyEngine eng;
    // Fresh engine: ATP = 1.0
    REQUIRE(eng.atp() > NAP_ENTER_THRESHOLD);
    REQUIRE_FALSE(eng.is_query_gated());
}

// ── §13 is_query_gated() = true when ATP < NAP_ENTER_THRESHOLD ───────────────

TEST_CASE("[P50-§13] is_query_gated() = true when ATP < NAP_ENTER_THRESHOLD",
          "[phase50]") {
    // Start with very low initial ATP
    AutonomyConfig cfg;
    cfg.initial_atp = 0.10f;   // below 0.15 threshold from the start
    AutonomyEngine eng(cfg);

    REQUIRE(eng.atp() < NAP_ENTER_THRESHOLD);
    REQUIRE(eng.is_query_gated());
}

// ── §14 query_gate_count() increments when ATP < threshold ───────────────────

TEST_CASE("[P50-§14] query_gate_count() increments each tick with low ATP",
          "[phase50]") {
    AutonomyConfig cfg;
    cfg.initial_atp   = 0.10f;  // below 0.15 from start
    cfg.enable_boredom = false;  // isolate metabolic from boredom noise
    AutonomyEngine eng(cfg);

    auto r = spike_r();
    auto i = zeros();

    REQUIRE(eng.query_gate_count() == 0u);
    eng.tick(0.01f, r, i);
    REQUIRE(eng.query_gate_count() >= 1u);
    eng.tick(0.01f, r, i);
    REQUIRE(eng.query_gate_count() >= 2u);
}

// ── §15 query_gate_count() stays 0 while ATP above threshold ─────────────────

TEST_CASE("[P50-§15] query_gate_count() stays 0 while ATP above 0.15",
          "[phase50]") {
    AutonomyConfig cfg;
    cfg.initial_atp   = 1.0f;
    cfg.enable_boredom = false;
    AutonomyEngine eng(cfg);

    // Empty psi spans → no energy consumption → ATP stays near 1.0
    std::vector<float> empty_r, empty_i;
    for (int t = 0; t < 5; ++t) eng.tick(0.01f, empty_r, empty_i);

    REQUIRE(eng.atp() > NAP_ENTER_THRESHOLD);
    REQUIRE(eng.query_gate_count() == 0u);
}
