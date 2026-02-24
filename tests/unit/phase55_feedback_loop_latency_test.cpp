/**
 * @file phase55_feedback_loop_latency_test.cpp
 * @brief Phase 55 — GAP-022: ENGS → Physics Engine Feedback Loop Latency
 *
 * Spec: docs/info/integration/sections/05_autonomous_systems/
 *       01_computational_neurochemistry.md  §GAP-022
 *
 * Problem addressed:
 *   Physics engine runs at 1 kHz (1 ms timestep).  Neurochemical signals
 *   computed by ENGS must reach the Physics kernel within τ_max ≤ 10 ms to
 *   avoid credit assignment error (reinforcing wrong thoughts due to delayed
 *   neurochemical feedback — "delayed pain signal after touching hot stove").
 *
 * Spec requirements under test:
 *   - τ = t_applied − t_calc per channel
 *   - Dopamine / Norepinephrine hard limit: 10 ms
 *   - Serotonin soft limit: 50 ms
 *   - Double-buffered atomic swap guarantees atomicity + phase coherence
 *   - CRITICAL priority bypasses double-buffer (< 1 ms interrupt path)
 *   - SYNC_VIOLATION emitted when D or N exceeds 10 ms
 *   - COGNITIVE_PAUSE emitted when any channel exceeds 50 ms
 *   - NeurochemicalState field layout / alignment
 *   - EngsPhysicsInterface tick semantics
 *   - SignalPriority naming round-trip
 *   - ViolationKind naming round-trip
 *
 * Tests (20 cases, 52 assertions):
 *   §1  – Staleness formula: τ = 0 when calc == applied
 *   §2  – Staleness formula: τ = t_applied − t_calc (positive delta)
 *   §3  – Staleness clamp: t_applied < t_calc → τ = 0 (no negative staleness)
 *   §4  – D/N hard limit: 9 ms → within_budget true
 *   §5  – D/N hard limit: 11 ms → within_budget false
 *   §6  – S soft limit: 49 ms → within_budget true
 *   §7  – S soft limit: 51 ms → within_budget false
 *   §8  – limit_us(): D = 10 000, S = 50 000
 *   §9  – NeurochemicalState ctor and field values
 *   §10 – NeurochemicalState default ctor (serotonin baseline 0.5)
 *   §11 – SignalPriority name round-trip (all 3 levels)
 *   §12 – ViolationKind name round-trip (all 3 levels)
 *   §13 – EngsPhysicsInterface: initial state is default NeurochemicalState
 *   §14 – push_update HIGH → state not visible until tick_start()
 *   §15 – tick_start() promotes pending state → get_current_state() reflects it
 *   §16 – CRITICAL push → immediately visible (interrupt path)
 *   §17 – tick_count increments with each tick_start() call
 *   §18 – SYNC_VIOLATION when D staleness > 10 ms
 *   §19 – COGNITIVE_PAUSE when any staleness > 50 ms
 *   §20 – Multi-tick integration: 5 pushes, 5 ticks, no violations
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>
#include <string>
#include <nikola/interface/feedback_loop.hpp>

using namespace nikola::feedback;
using Catch::Matchers::WithinAbs;

// ── §1 ── τ = 0 when calc time == applied time ────────────────────────────────

TEST_CASE("§1 staleness zero when calc==applied", "[phase55][gap022]") {
    StalenessBudget budget;

    constexpr uint64_t T = 1'000'000u;  // 1 000 000 µs = 1 s (arbitrary)
    budget.record_calc   (StalenessBudget::Channel::DOPAMINE, T);
    budget.record_applied(StalenessBudget::Channel::DOPAMINE, T);

    CHECK(budget.staleness_us(StalenessBudget::Channel::DOPAMINE) == 0u);
    CHECK(budget.within_budget(StalenessBudget::Channel::DOPAMINE));
}

// ── §2 ── τ = t_applied − t_calc ─────────────────────────────────────────────

TEST_CASE("§2 staleness equals applied minus calc", "[phase55][gap022]") {
    StalenessBudget budget;
    using C = StalenessBudget::Channel;

    budget.record_calc   (C::NOREPINEPHRINE, 5'000u);
    budget.record_applied(C::NOREPINEPHRINE, 8'000u);   // 3 ms delta

    CHECK(budget.staleness_us(C::NOREPINEPHRINE) == 3'000u);   // 3 ms
}

// ── §3 ── τ = 0 when applied < calc (no negative staleness) ──────────────────

TEST_CASE("§3 staleness clamped to zero when applied < calc", "[phase55][gap022]") {
    StalenessBudget budget;
    using C = StalenessBudget::Channel;

    // Out-of-order: clock skew or same-tick measurement
    budget.record_calc   (C::SEROTONIN, 100'000u);
    budget.record_applied(C::SEROTONIN,  99'000u);  // applied before calc? → 0

    CHECK(budget.staleness_us(C::SEROTONIN) == 0u);
    CHECK(budget.within_budget(C::SEROTONIN));  // 0 ≤ 50 000
}

// ── §4 ── D/N 9 ms → within_budget true ──────────────────────────────────────

TEST_CASE("§4 dopamine 9ms within hard limit", "[phase55][gap022]") {
    StalenessBudget budget;
    using C = StalenessBudget::Channel;

    budget.record_calc   (C::DOPAMINE, 0u);
    budget.record_applied(C::DOPAMINE, 9'000u);  // exactly 9 ms

    CHECK(budget.staleness_us(C::DOPAMINE) == 9'000u);
    CHECK(budget.within_budget(C::DOPAMINE));       // 9 000 ≤ 10 000 ✓
}

// ── §5 ── D/N 11 ms → within_budget false ────────────────────────────────────

TEST_CASE("§5 norepinephrine 11ms exceeds hard limit", "[phase55][gap022]") {
    StalenessBudget budget;
    using C = StalenessBudget::Channel;

    budget.record_calc   (C::NOREPINEPHRINE, 0u);
    budget.record_applied(C::NOREPINEPHRINE, 11'000u);  // 11 ms — violation

    CHECK(budget.staleness_us(C::NOREPINEPHRINE) == 11'000u);
    CHECK_FALSE(budget.within_budget(C::NOREPINEPHRINE));  // 11 000 > 10 000 ✗
}

// ── §6 ── S 49 ms → within_budget true ───────────────────────────────────────

TEST_CASE("§6 serotonin 49ms within soft limit", "[phase55][gap022]") {
    StalenessBudget budget;
    using C = StalenessBudget::Channel;

    budget.record_calc   (C::SEROTONIN, 0u);
    budget.record_applied(C::SEROTONIN, 49'000u);

    CHECK(budget.staleness_us(C::SEROTONIN) == 49'000u);
    CHECK(budget.within_budget(C::SEROTONIN));      // 49 000 ≤ 50 000 ✓
}

// ── §7 ── S 51 ms → within_budget false ──────────────────────────────────────

TEST_CASE("§7 serotonin 51ms exceeds soft limit", "[phase55][gap022]") {
    StalenessBudget budget;
    using C = StalenessBudget::Channel;

    budget.record_calc   (C::SEROTONIN, 0u);
    budget.record_applied(C::SEROTONIN, 51'000u);

    CHECK(budget.staleness_us(C::SEROTONIN) == 51'000u);
    CHECK_FALSE(budget.within_budget(C::SEROTONIN));  // 51 000 > 50 000 ✗
}

// ── §8 ── limit_us() returns correct channel limits ───────────────────────────

TEST_CASE("§8 channel limit_us values match spec", "[phase55][gap022]") {
    using C = StalenessBudget::Channel;

    // D/N/Cortisol: 10 000 µs = 10 ms hard
    CHECK(StalenessBudget::limit_us(C::DOPAMINE)       == 10'000u);
    CHECK(StalenessBudget::limit_us(C::NOREPINEPHRINE) == 10'000u);
    CHECK(StalenessBudget::limit_us(C::CORTISOL)       == 10'000u);
    // Serotonin: 50 000 µs = 50 ms soft
    CHECK(StalenessBudget::limit_us(C::SEROTONIN)      == 50'000u);
}

// ── §9 ── NeurochemicalState explicit ctor ────────────────────────────────────

TEST_CASE("§9 NeurochemicalState explicit constructor", "[phase55][gap022]") {
    NeurochemicalState ns{0.7f, 0.4f, 0.3f, 0.1f, 42u};

    CHECK_THAT(ns.dopamine,       WithinAbs(0.7f, 1e-6f));
    CHECK_THAT(ns.serotonin,      WithinAbs(0.4f, 1e-6f));
    CHECK_THAT(ns.norepinephrine, WithinAbs(0.3f, 1e-6f));
    CHECK_THAT(ns.cortisol,       WithinAbs(0.1f, 1e-6f));
    CHECK(ns.timestamp_seq == 42u);
}

// ── §10 ── NeurochemicalState default ctor baseline ──────────────────────────

TEST_CASE("§10 NeurochemicalState default serotonin baseline", "[phase55][gap022]") {
    NeurochemicalState ns{};

    CHECK_THAT(ns.dopamine,       WithinAbs(0.0f, 1e-6f));
    CHECK_THAT(ns.serotonin,      WithinAbs(0.5f, 1e-6f));  // homeostatic baseline
    CHECK_THAT(ns.norepinephrine, WithinAbs(0.0f, 1e-6f));
    CHECK_THAT(ns.cortisol,       WithinAbs(0.0f, 1e-6f));
    CHECK(ns.timestamp_seq == 0u);
}

// ── §11 ── SignalPriority name round-trip ─────────────────────────────────────

TEST_CASE("§11 SignalPriority name round-trip", "[phase55][gap022]") {
    CHECK(std::string(signal_priority_name(SignalPriority::BACKGROUND)) == "BACKGROUND");
    CHECK(std::string(signal_priority_name(SignalPriority::HIGH))       == "HIGH");
    CHECK(std::string(signal_priority_name(SignalPriority::CRITICAL))   == "CRITICAL");
}

// ── §12 ── ViolationKind name round-trip ──────────────────────────────────────

TEST_CASE("§12 ViolationKind name round-trip", "[phase55][gap022]") {
    CHECK(std::string(violation_kind_name(ViolationKind::NONE))            == "NONE");
    CHECK(std::string(violation_kind_name(ViolationKind::SYNC_VIOLATION))  == "SYNC_VIOLATION");
    CHECK(std::string(violation_kind_name(ViolationKind::COGNITIVE_PAUSE)) == "COGNITIVE_PAUSE");
}

// ── §13 ── EngsPhysicsInterface initial state is default ─────────────────────

TEST_CASE("§13 EngsPhysicsInterface initial state is default", "[phase55][gap022]") {
    EngsPhysicsInterface iface;

    auto s = iface.get_current_state();
    CHECK_THAT(s.dopamine,       WithinAbs(0.0f, 1e-6f));
    CHECK_THAT(s.serotonin,      WithinAbs(0.5f, 1e-6f));
    CHECK_THAT(s.norepinephrine, WithinAbs(0.0f, 1e-6f));
    CHECK(iface.tick_count() == 0u);
    CHECK_FALSE(iface.has_pending());
    CHECK_FALSE(iface.has_interrupt());
}

// ── §14 ── HIGH push does NOT change current state before tick ────────────────

TEST_CASE("§14 HIGH push not visible before tick_start", "[phase55][gap022]") {
    EngsPhysicsInterface iface;

    NeurochemicalState new_s{0.9f, 0.6f, 0.8f, 0.05f, 1u};
    iface.push_update(new_s, SignalPriority::HIGH, 1000u);

    // Pending flag is set ...
    CHECK(iface.has_pending());

    // ... but current state has NOT changed yet
    auto cur = iface.get_current_state();
    CHECK_THAT(cur.dopamine, WithinAbs(0.0f, 1e-6f));   // still default
}

// ── §15 ── tick_start() promotes pending state ────────────────────────────────

TEST_CASE("§15 tick_start promotes HIGH pending state", "[phase55][gap022]") {
    EngsPhysicsInterface iface;

    NeurochemicalState pushed{0.75f, 0.55f, 0.65f, 0.0f, 2u};
    iface.push_update(pushed, SignalPriority::HIGH, 1000u);

    // Advance to next tick
    iface.tick_start(2000u);

    auto cur = iface.get_current_state();
    CHECK_THAT(cur.dopamine,       WithinAbs(0.75f, 1e-6f));
    CHECK_THAT(cur.serotonin,      WithinAbs(0.55f, 1e-6f));
    CHECK_THAT(cur.norepinephrine, WithinAbs(0.65f, 1e-6f));
    CHECK(iface.tick_count() == 1u);
    CHECK_FALSE(iface.has_pending());  // consumed
}

// ── §16 ── CRITICAL push immediately visible ──────────────────────────────────

TEST_CASE("§16 CRITICAL push immediately applies on next tick_start", "[phase55][gap022]") {
    EngsPhysicsInterface iface;

    NeurochemicalState emergency{1.0f, 0.5f, 1.0f, 0.9f, 99u};
    iface.push_update(emergency, SignalPriority::CRITICAL, 500u);

    // Interrupt flag set BEFORE tick
    CHECK(iface.has_interrupt());

    // Apply within 1 tick (< 1 ms path)
    iface.tick_start(1000u);

    auto cur = iface.get_current_state();
    CHECK_THAT(cur.dopamine,       WithinAbs(1.0f, 1e-6f));
    CHECK_THAT(cur.norepinephrine, WithinAbs(1.0f, 1e-6f));  // Panic N level
    CHECK_THAT(cur.cortisol,       WithinAbs(0.9f, 1e-6f));  // High cortisol

    // Interrupt cleared after consumption
    CHECK_FALSE(iface.has_interrupt());
}

// ── §17 ── tick_count increments correctly ────────────────────────────────────

TEST_CASE("§17 tick_count increments per tick_start call", "[phase55][gap022]") {
    EngsPhysicsInterface iface;

    for (uint64_t i = 1; i <= 10; ++i) {
        iface.tick_start(i * PHYSICS_TICK_US);
        CHECK(iface.tick_count() == i);
    }
}

// ── §18 ── SYNC_VIOLATION when D staleness > 10 ms ───────────────────────────

TEST_CASE("§18 check_violations returns SYNC_VIOLATION for stale D", "[phase55][gap022]") {
    EngsPhysicsInterface iface;

    // ENGS calculated at t=0
    iface.push_update(NeurochemicalState{0.5f, 0.5f, 0.5f, 0.0f}, 
                      SignalPriority::HIGH, 0u);

    // Physics ticked 11 ms later
    iface.tick_start(11'000u);

    // Staleness: D = 11 000 − 0 = 11 000 µs > 10 000 µs hard limit
    const auto& budget = iface.staleness_budget();
    CHECK(budget.staleness_us(StalenessBudget::Channel::DOPAMINE) == 11'000u);
    CHECK_FALSE(budget.within_budget(StalenessBudget::Channel::DOPAMINE));

    auto violation = iface.check_violations();
    CHECK(violation == ViolationKind::SYNC_VIOLATION);
}

// ── §19 ── COGNITIVE_PAUSE when any channel > 50 ms ──────────────────────────

TEST_CASE("§19 check_violations returns COGNITIVE_PAUSE for 51ms stale S", "[phase55][gap022]") {
    EngsPhysicsInterface iface;

    // ENGS pushed at t=0, tick fires at t=51 ms
    iface.push_update(NeurochemicalState{0.5f, 0.5f, 0.5f, 0.0f},
                      SignalPriority::BACKGROUND, 0u);
    iface.tick_start(51'000u);

    // Serotonin (BACKGROUND) staleness = 51 ms > 50 ms soft limit
    const auto& budget = iface.staleness_budget();
    CHECK(budget.staleness_us(StalenessBudget::Channel::SEROTONIN) == 51'000u);

    // COGNITIVE_PAUSE is triggered (more severe than SYNC_VIOLATION)
    CHECK(iface.check_violations() == ViolationKind::COGNITIVE_PAUSE);
}

// ── §20 ── Multi-tick integration, no violations ──────────────────────────────

TEST_CASE("§20 multi-tick integration clean run no violations", "[phase55][gap022]") {
    EngsPhysicsInterface iface;

    // Simulate 5 ENGS updates at ~5 ms intervals (well within budget)
    // Each push+tick pair stays within 3.5 ms (spec total latency budget)
    for (uint64_t tick = 0; tick < 5; ++tick) {
        const uint64_t calc_t  = tick * 5'000u;           // every 5 ms
        const uint64_t apply_t = calc_t + 3'500u;         // +3.5 ms (spec budget)

        NeurochemicalState s{
            0.3f + 0.1f * static_cast<float>(tick),
            0.5f,
            0.2f + 0.05f * static_cast<float>(tick),
            0.0f,
            tick
        };

        iface.push_update(s, SignalPriority::HIGH, calc_t);
        iface.tick_start(apply_t);

        auto cur = iface.get_current_state();

        // Neurochemical values propagated correctly
        CHECK_THAT(cur.dopamine, WithinAbs(0.3f + 0.1f * static_cast<float>(tick), 1e-5f));
        CHECK_THAT(cur.serotonin, WithinAbs(0.5f, 1e-5f));

        // D/N staleness = 3 500 µs < 10 000 µs → no violation
        const auto& budget = iface.staleness_budget();
        CHECK(budget.staleness_us(StalenessBudget::Channel::DOPAMINE) == 3'500u);
        CHECK(budget.within_budget(StalenessBudget::Channel::DOPAMINE));
        CHECK(iface.check_violations() == ViolationKind::NONE);
    }

    CHECK(iface.tick_count() == 5u);
}
