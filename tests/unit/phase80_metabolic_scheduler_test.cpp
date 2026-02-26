// =============================================================================
// phase80_metabolic_scheduler_test.cpp
// Phase 80 — GAP-013: Transactional Metabolic Scheduling with RAII Locks
//
// Tests every constant, enum value, and pure function declared in
// nikola/persistence/metabolic_scheduler.hpp against the spec values
// documented in §GAP-013 of 04_nap_system.md.
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "nikola/persistence/metabolic_scheduler.hpp"

using namespace nikola::persistence;
using Catch::Approx;

// ---------------------------------------------------------------------------
// §1 — Spec constants
// ---------------------------------------------------------------------------

TEST_CASE("ATP_MAX is 10000", "[constants]") {
    CHECK(ATP_MAX == Approx(10000.0f));
}

TEST_CASE("RECHARGE_RATE_PER_S is 50", "[constants]") {
    CHECK(RECHARGE_RATE_PER_S == Approx(50.0f));
}

TEST_CASE("SOFT_THRESHOLD_ATP is 1500", "[constants]") {
    CHECK(SOFT_THRESHOLD_ATP == Approx(1500.0f));
}

TEST_CASE("HARD_THRESHOLD_ATP is 500", "[constants]") {
    CHECK(HARD_THRESHOLD_ATP == Approx(500.0f));
}

TEST_CASE("SOFT_THRESHOLD_PCT is 15", "[constants]") {
    CHECK(SOFT_THRESHOLD_PCT == 15);
}

TEST_CASE("HARD_THRESHOLD_PCT is 5", "[constants]") {
    CHECK(HARD_THRESHOLD_PCT == 5);
}

TEST_CASE("CRITICAL_RESERVE_ATP equals HARD_THRESHOLD_ATP", "[constants]") {
    CHECK(CRITICAL_RESERVE_ATP == Approx(HARD_THRESHOLD_ATP));
}

TEST_CASE("GRACE_PERIOD_S derived as E_hard/Eburn = 500/100 = 5.0", "[constants]") {
    CHECK(GRACE_PERIOD_S == Approx(5.0f));
    // Verify derivation
    CHECK(BURN_RATE_PER_UNIT == Approx(100.0f));
    CHECK(HARD_THRESHOLD_ATP / BURN_RATE_PER_UNIT == Approx(GRACE_PERIOD_S));
}

TEST_CASE("WAL_CHUNK_COST_ATP is 100", "[constants]") {
    CHECK(WAL_CHUNK_COST_ATP == Approx(100.0f));
}

TEST_CASE("OVERDRAFT_PENALTY_FACTOR is 0.10", "[constants]") {
    CHECK(OVERDRAFT_PENALTY_FACTOR == Approx(0.10f));
}

TEST_CASE("COMA_RECHARGE_HOURS is 1", "[constants]") {
    CHECK(COMA_RECHARGE_HOURS == 1);
}

TEST_CASE("COMA_RECHARGE_S is 3600", "[constants]") {
    CHECK(COMA_RECHARGE_S == Approx(3600.0f));
}

TEST_CASE("PANIC_LOCK_TIMEOUT_S is 5", "[constants]") {
    CHECK(PANIC_LOCK_TIMEOUT_S == Approx(5.0f));
}

TEST_CASE("SOFT_THRESHOLD_ATP equals 15% of ATP_MAX", "[constants]") {
    CHECK(SOFT_THRESHOLD_ATP == Approx(ATP_MAX * 0.15f));
}

TEST_CASE("HARD_THRESHOLD_ATP equals 5% of ATP_MAX", "[constants]") {
    CHECK(HARD_THRESHOLD_ATP == Approx(ATP_MAX * 0.05f));
}

TEST_CASE("FULL_RECHARGE_FROM_HARD_S is 190 seconds", "[constants]") {
    // (10000 - 500) / 50 = 190 s
    CHECK(FULL_RECHARGE_FROM_HARD_S == Approx(190.0f));
}

TEST_CASE("Task cost constants match spec examples", "[constants]") {
    CHECK(TASK_COST_PDF_CHUNK_ATP  == Approx(50.0f));
    CHECK(TASK_COST_EMBEDDINGS_ATP == Approx(500.0f));
    CHECK(TASK_COST_LMDB_STORE_ATP == Approx(20.0f));
}

// ---------------------------------------------------------------------------
// §2 — MetabolicZone enum ordinal values
// ---------------------------------------------------------------------------

TEST_CASE("MetabolicZone ordinals: NORMAL=0, WARNING=1, CRITICAL=2", "[enums]") {
    CHECK(static_cast<uint8_t>(MetabolicZone::NORMAL)   == 0);
    CHECK(static_cast<uint8_t>(MetabolicZone::WARNING)  == 1);
    CHECK(static_cast<uint8_t>(MetabolicZone::CRITICAL) == 2);
}

TEST_CASE("NapState ordinals: ACTIVE_WAKING=0..COMA=4", "[enums]") {
    CHECK(static_cast<uint8_t>(NapState::ACTIVE_WAKING)       == 0);
    CHECK(static_cast<uint8_t>(NapState::METABOLIC_WARNING)   == 1);
    CHECK(static_cast<uint8_t>(NapState::CRITICAL_EXHAUSTION) == 2);
    CHECK(static_cast<uint8_t>(NapState::NAP)                 == 3);
    CHECK(static_cast<uint8_t>(NapState::COMA)                == 4);
}

TEST_CASE("TaskInitiationResult ordinals", "[enums]") {
    CHECK(static_cast<uint8_t>(TaskInitiationResult::GRANTED)            == 0);
    CHECK(static_cast<uint8_t>(TaskInitiationResult::REJECTED_SOFT)      == 1);
    CHECK(static_cast<uint8_t>(TaskInitiationResult::REJECTED_HARD)      == 2);
    CHECK(static_cast<uint8_t>(TaskInitiationResult::REJECTED_PREFLIGHT) == 3);
}

TEST_CASE("LockReleaseOutcome ordinals", "[enums]") {
    CHECK(static_cast<uint8_t>(LockReleaseOutcome::CLEAN)             == 0);
    CHECK(static_cast<uint8_t>(LockReleaseOutcome::OVERDRAFT_PENALTY) == 1);
    CHECK(static_cast<uint8_t>(LockReleaseOutcome::PANIC_ABORT)       == 2);
}

TEST_CASE("WalYieldStatus ordinals", "[enums]") {
    CHECK(static_cast<uint8_t>(WalYieldStatus::CONTINUE)        == 0);
    CHECK(static_cast<uint8_t>(WalYieldStatus::YIELD_REQUESTED) == 1);
    CHECK(static_cast<uint8_t>(WalYieldStatus::PANIC_ABORT)     == 2);
}

TEST_CASE("EmergencyAction ordinals", "[enums]") {
    CHECK(static_cast<uint8_t>(EmergencyAction::NONE)       == 0);
    CHECK(static_cast<uint8_t>(EmergencyAction::SET_PANIC)  == 1);
    CHECK(static_cast<uint8_t>(EmergencyAction::DIRTY_DUMP) == 2);
    CHECK(static_cast<uint8_t>(EmergencyAction::ENTER_COMA) == 3);
}

// ---------------------------------------------------------------------------
// §3 — classify_zone
// ---------------------------------------------------------------------------

TEST_CASE("classify_zone: full ATP is NORMAL", "[classify_zone]") {
    CHECK(classify_zone(10000.0f) == MetabolicZone::NORMAL);
}

TEST_CASE("classify_zone: ATP just above 1500 is NORMAL", "[classify_zone]") {
    CHECK(classify_zone(1501.0f) == MetabolicZone::NORMAL);
}

TEST_CASE("classify_zone: ATP exactly 1500 is WARNING (boundary inclusive)", "[classify_zone]") {
    CHECK(classify_zone(1500.0f) == MetabolicZone::WARNING);
}

TEST_CASE("classify_zone: ATP 1000 is WARNING", "[classify_zone]") {
    CHECK(classify_zone(1000.0f) == MetabolicZone::WARNING);
}

TEST_CASE("classify_zone: ATP just above 500 is WARNING", "[classify_zone]") {
    CHECK(classify_zone(501.0f) == MetabolicZone::WARNING);
}

TEST_CASE("classify_zone: ATP exactly 500 is CRITICAL (boundary inclusive)", "[classify_zone]") {
    CHECK(classify_zone(500.0f) == MetabolicZone::CRITICAL);
}

TEST_CASE("classify_zone: ATP 499 is CRITICAL", "[classify_zone]") {
    CHECK(classify_zone(499.0f) == MetabolicZone::CRITICAL);
}

TEST_CASE("classify_zone: ATP 0 is CRITICAL", "[classify_zone]") {
    CHECK(classify_zone(0.0f) == MetabolicZone::CRITICAL);
}

TEST_CASE("classify_zone: ATP 5000 mid-range is NORMAL", "[classify_zone]") {
    CHECK(classify_zone(5000.0f) == MetabolicZone::NORMAL);
}

// ---------------------------------------------------------------------------
// §4 — nap_state_for_zone / nap_state_for_atp
// ---------------------------------------------------------------------------

TEST_CASE("nap_state_for_zone: NORMAL → ACTIVE_WAKING", "[nap_state]") {
    CHECK(nap_state_for_zone(MetabolicZone::NORMAL) == NapState::ACTIVE_WAKING);
}

TEST_CASE("nap_state_for_zone: WARNING → METABOLIC_WARNING", "[nap_state]") {
    CHECK(nap_state_for_zone(MetabolicZone::WARNING) == NapState::METABOLIC_WARNING);
}

TEST_CASE("nap_state_for_zone: CRITICAL → CRITICAL_EXHAUSTION", "[nap_state]") {
    CHECK(nap_state_for_zone(MetabolicZone::CRITICAL) == NapState::CRITICAL_EXHAUSTION);
}

TEST_CASE("nap_state_for_atp: full ATP gives ACTIVE_WAKING", "[nap_state]") {
    CHECK(nap_state_for_atp(10000.0f) == NapState::ACTIVE_WAKING);
}

TEST_CASE("nap_state_for_atp: 1000 ATP gives METABOLIC_WARNING", "[nap_state]") {
    CHECK(nap_state_for_atp(1000.0f) == NapState::METABOLIC_WARNING);
}

TEST_CASE("nap_state_for_atp: 200 ATP gives CRITICAL_EXHAUSTION", "[nap_state]") {
    CHECK(nap_state_for_atp(200.0f) == NapState::CRITICAL_EXHAUSTION);
}

// ---------------------------------------------------------------------------
// §5 — Task initiation gating predicates
// ---------------------------------------------------------------------------

TEST_CASE("is_unrestricted_initiation: true only in Zone I", "[initiation]") {
    CHECK(is_unrestricted_initiation(10000.0f) == true);
    CHECK(is_unrestricted_initiation(1501.0f)  == true);
    CHECK(is_unrestricted_initiation(1500.0f)  == false); // boundary
    CHECK(is_unrestricted_initiation(1000.0f)  == false);
    CHECK(is_unrestricted_initiation(400.0f)   == false);
}

TEST_CASE("is_high_cost_task_blocked: false only in Zone I", "[initiation]") {
    CHECK(is_high_cost_task_blocked(10000.0f) == false);
    CHECK(is_high_cost_task_blocked(1501.0f)  == false);
    CHECK(is_high_cost_task_blocked(1500.0f)  == true);
    CHECK(is_high_cost_task_blocked(700.0f)   == true);
    CHECK(is_high_cost_task_blocked(100.0f)   == true);
}

TEST_CASE("is_all_initiation_blocked: true only in Zone III", "[initiation]") {
    CHECK(is_all_initiation_blocked(10000.0f) == false);
    CHECK(is_all_initiation_blocked(1000.0f)  == false);
    CHECK(is_all_initiation_blocked(501.0f)   == false);
    CHECK(is_all_initiation_blocked(500.0f)   == true);  // boundary
    CHECK(is_all_initiation_blocked(0.0f)     == true);
}

// ---------------------------------------------------------------------------
// §6 — preflight_feasible
// ---------------------------------------------------------------------------

TEST_CASE("preflight_feasible: atp=8000 cost=100 passes easily", "[preflight]") {
    CHECK(preflight_feasible(8000.0f, 100.0f) == true);
}

TEST_CASE("preflight_feasible: atp=700 cost=100 reserve=500 → 700-100=600 > 500", "[preflight]") {
    CHECK(preflight_feasible(700.0f, 100.0f, 500.0f) == true);
}

TEST_CASE("preflight_feasible: atp=700 cost=200 reserve=500 → 700-200=500 NOT > 500", "[preflight]") {
    // must be strictly greater than reserve
    CHECK(preflight_feasible(700.0f, 200.0f, 500.0f) == false);
}

TEST_CASE("preflight_feasible: atp=700 cost=201 reserve=500 → fails", "[preflight]") {
    CHECK(preflight_feasible(700.0f, 201.0f, 500.0f) == false);
}

TEST_CASE("preflight_feasible: atp=600 cost=50 default reserve=500 → 550 > 500", "[preflight]") {
    CHECK(preflight_feasible(600.0f, 50.0f) == true);
}

TEST_CASE("preflight_feasible: atp=550 cost=60 default reserve=500 → 490 NOT > 500", "[preflight]") {
    CHECK(preflight_feasible(550.0f, 60.0f) == false);
}

TEST_CASE("preflight_feasible: zero cost always passes with healthy ATP", "[preflight]") {
    CHECK(preflight_feasible(2000.0f, 0.0f) == true);
}

// ---------------------------------------------------------------------------
// §7 — task_initiation_result
// ---------------------------------------------------------------------------

TEST_CASE("task_initiation_result: Zone I, affordable → GRANTED", "[task_result]") {
    CHECK(task_initiation_result(9000.0f, 100.0f) == TaskInitiationResult::GRANTED);
}

TEST_CASE("task_initiation_result: Zone I, cost triggers preflight fail → REJECTED_PREFLIGHT", "[task_result]") {
    // atp=1600, cost=1101 → preflight: 1600-1101=499 NOT > 500
    CHECK(task_initiation_result(1600.0f, 1101.0f) == TaskInitiationResult::REJECTED_PREFLIGHT);
}

TEST_CASE("task_initiation_result: Zone II REJECTED_SOFT", "[task_result]") {
    CHECK(task_initiation_result(1000.0f, 100.0f) == TaskInitiationResult::REJECTED_SOFT);
}

TEST_CASE("task_initiation_result: Zone III REJECTED_HARD", "[task_result]") {
    CHECK(task_initiation_result(300.0f, 10.0f) == TaskInitiationResult::REJECTED_HARD);
}

TEST_CASE("task_initiation_result: Zone III even with tiny cost → REJECTED_HARD", "[task_result]") {
    CHECK(task_initiation_result(0.0f, 0.0f) == TaskInitiationResult::REJECTED_HARD);
}

TEST_CASE("task_initiation_result: Zone I at exact soft threshold minus 1 → GRANTED if preflight ok", "[task_result]") {
    // atp=1502, cost=1, preflight: 1501 > 500 → yes
    CHECK(task_initiation_result(1502.0f, 1.0f) == TaskInitiationResult::GRANTED);
}

// ---------------------------------------------------------------------------
// §8 — should_yield (WAL checkpoint)
// ---------------------------------------------------------------------------

TEST_CASE("should_yield: Zone I → CONTINUE", "[wal_yield]") {
    CHECK(should_yield(10000.0f) == WalYieldStatus::CONTINUE);
    CHECK(should_yield(1501.0f)  == WalYieldStatus::CONTINUE);
}

TEST_CASE("should_yield: exactly at soft threshold → YIELD_REQUESTED", "[wal_yield]") {
    CHECK(should_yield(1500.0f) == WalYieldStatus::YIELD_REQUESTED);
}

TEST_CASE("should_yield: Zone II mid → YIELD_REQUESTED", "[wal_yield]") {
    CHECK(should_yield(1000.0f) == WalYieldStatus::YIELD_REQUESTED);
    CHECK(should_yield(501.0f)  == WalYieldStatus::YIELD_REQUESTED);
}

TEST_CASE("should_yield: exactly at hard threshold → PANIC_ABORT", "[wal_yield]") {
    CHECK(should_yield(500.0f) == WalYieldStatus::PANIC_ABORT);
}

TEST_CASE("should_yield: Zone III → PANIC_ABORT", "[wal_yield]") {
    CHECK(should_yield(499.0f) == WalYieldStatus::PANIC_ABORT);
    CHECK(should_yield(0.0f)   == WalYieldStatus::PANIC_ABORT);
}

// ---------------------------------------------------------------------------
// §9 — lock_release_outcome
// ---------------------------------------------------------------------------

TEST_CASE("lock_release_outcome: released above HARD → CLEAN", "[lock_release]") {
    CHECK(lock_release_outcome(10000.0f) == LockReleaseOutcome::CLEAN);
    CHECK(lock_release_outcome(1500.0f)  == LockReleaseOutcome::CLEAN);
    CHECK(lock_release_outcome(501.0f)   == LockReleaseOutcome::CLEAN);
}

TEST_CASE("lock_release_outcome: exactly at HARD → still CLEAN (not strictly below)", "[lock_release]") {
    // spec says "below" HARD_THRESHOLD triggers penalty — 500.0 == HARD is not below
    CHECK(lock_release_outcome(500.0f) == LockReleaseOutcome::CLEAN);
}

TEST_CASE("lock_release_outcome: below HARD → OVERDRAFT_PENALTY", "[lock_release]") {
    CHECK(lock_release_outcome(499.0f) == LockReleaseOutcome::OVERDRAFT_PENALTY);
    CHECK(lock_release_outcome(0.0f)   == LockReleaseOutcome::OVERDRAFT_PENALTY);
}

// ---------------------------------------------------------------------------
// §10 — overdraft_e_max
// ---------------------------------------------------------------------------

TEST_CASE("overdraft_e_max: 10000 * 0.9 = 9000", "[overdraft]") {
    CHECK(overdraft_e_max(10000.0f) == Approx(9000.0f));
}

TEST_CASE("overdraft_e_max: 9000 * 0.9 = 8100", "[overdraft]") {
    CHECK(overdraft_e_max(9000.0f) == Approx(8100.0f));
}

TEST_CASE("overdraft_e_max: 5000 → 4500", "[overdraft]") {
    CHECK(overdraft_e_max(5000.0f) == Approx(4500.0f));
}

TEST_CASE("overdraft_e_max: ten consecutive penalties from 10000", "[overdraft]") {
    float cap = ATP_MAX;
    for (int i = 0; i < 10; ++i) cap = overdraft_e_max(cap);
    // 10000 * 0.9^10 ≈ 3486.78
    CHECK(cap == Approx(10000.0f * 0.9f * 0.9f * 0.9f * 0.9f * 0.9f
                                * 0.9f * 0.9f * 0.9f * 0.9f * 0.9f).epsilon(0.01f));
}

TEST_CASE("overdraft_penalties_until: convergence below half ATP_MAX", "[overdraft]") {
    // capacity halves roughly every 7 penalties (0.9^7 ≈ 0.478)
    int n = overdraft_penalties_until(ATP_MAX / 2.0f);
    CHECK(n > 0);
    CHECK(n < 15);  // should converge quickly
}

TEST_CASE("overdraft_penalties_until: 0 penalties needed when already at floor", "[overdraft]") {
    // If cap is already below floor the function returns 0 before first iteration
    // cap=ATP_MAX > floor=ATP_MAX → first iteration fires → n≥1
    // test: floor = ATP_MAX+1 → immediately below → n=0
    CHECK(overdraft_penalties_until(ATP_MAX + 1.0f) == 0);
}

// ---------------------------------------------------------------------------
// §11 — recharge_delta and recharge arithmetic
// ---------------------------------------------------------------------------

TEST_CASE("recharge_delta: from 0 ATP, 1s → 50 ATP", "[recharge]") {
    CHECK(recharge_delta(0.0f, 1.0f) == Approx(50.0f));
}

TEST_CASE("recharge_delta: from 0 ATP, 10s → 500 ATP", "[recharge]") {
    CHECK(recharge_delta(0.0f, 10.0f) == Approx(500.0f));
}

TEST_CASE("recharge_delta: clamps at ATP_MAX", "[recharge]") {
    CHECK(recharge_delta(9990.0f, 5.0f) == Approx(ATP_MAX));  // 9990+250 > 10000
}

TEST_CASE("recharge_delta: exactly at ATP_MAX stays clamped", "[recharge]") {
    CHECK(recharge_delta(10000.0f, 100.0f) == Approx(ATP_MAX));
}

TEST_CASE("recharge_delta: 500 ATP, 100s → clamped to 10000", "[recharge]") {
    // 500 + 50*100 = 5500 < 10000, so no clamp
    CHECK(recharge_delta(500.0f, 100.0f) == Approx(5500.0f));
}

TEST_CASE("seconds_to_full_recharge: from HARD_THRESHOLD = 190s", "[recharge]") {
    CHECK(seconds_to_full_recharge(HARD_THRESHOLD_ATP) == Approx(190.0f));
}

TEST_CASE("seconds_to_full_recharge: from 0 = 200s", "[recharge]") {
    CHECK(seconds_to_full_recharge(0.0f) == Approx(200.0f));
}

TEST_CASE("seconds_to_full_recharge: already full → 0s", "[recharge]") {
    CHECK(seconds_to_full_recharge(ATP_MAX) == Approx(0.0f));
}

TEST_CASE("atp_after_recharge: 0 ATP, 60s → 3000 ATP", "[recharge]") {
    CHECK(atp_after_recharge(0.0f, 60.0f) == Approx(3000.0f));
}

TEST_CASE("atp_after_recharge: clamps at ATP_MAX", "[recharge]") {
    CHECK(atp_after_recharge(9900.0f, 10.0f) == Approx(ATP_MAX));
}

// ---------------------------------------------------------------------------
// §12 — atp_percentage
// ---------------------------------------------------------------------------

TEST_CASE("atp_percentage: full ATP = 100%", "[percentage]") {
    CHECK(atp_percentage(10000.0f) == Approx(100.0f));
}

TEST_CASE("atp_percentage: SOFT_THRESHOLD = 15%", "[percentage]") {
    CHECK(atp_percentage(SOFT_THRESHOLD_ATP) == Approx(15.0f));
}

TEST_CASE("atp_percentage: HARD_THRESHOLD = 5%", "[percentage]") {
    CHECK(atp_percentage(HARD_THRESHOLD_ATP) == Approx(5.0f));
}

TEST_CASE("atp_percentage: 0 ATP = 0%", "[percentage]") {
    CHECK(atp_percentage(0.0f) == Approx(0.0f));
}

TEST_CASE("atp_percentage: 5000 ATP = 50%", "[percentage]") {
    CHECK(atp_percentage(5000.0f) == Approx(50.0f));
}

TEST_CASE("atp_percentage: zero e_max returns 0 safely", "[percentage]") {
    CHECK(atp_percentage(1000.0f, 0.0f) == Approx(0.0f));
}

TEST_CASE("is_in_warning_band: 10% is in band (5,15]", "[percentage]") {
    CHECK(is_in_warning_band(10.0f) == true);
}

TEST_CASE("is_in_warning_band: exactly 15% is in band", "[percentage]") {
    CHECK(is_in_warning_band(15.0f) == true);
}

TEST_CASE("is_in_warning_band: 16% is NOT in band", "[percentage]") {
    CHECK(is_in_warning_band(16.0f) == false);
}

TEST_CASE("is_in_warning_band: exactly 5% is NOT in band (exclusive lower)", "[percentage]") {
    CHECK(is_in_warning_band(5.0f) == false);
}

TEST_CASE("is_in_warning_band: 1% is NOT in band", "[percentage]") {
    CHECK(is_in_warning_band(1.0f) == false);
}

// ---------------------------------------------------------------------------
// §13 — Timeout predicates
// ---------------------------------------------------------------------------

TEST_CASE("grace_period_elapsed: false if under 5s", "[timeouts]") {
    CHECK(grace_period_elapsed(0.0f)  == false);
    CHECK(grace_period_elapsed(4.9f)  == false);
    CHECK(grace_period_elapsed(4.99f) == false);
}

TEST_CASE("grace_period_elapsed: true at and beyond 5.0s", "[timeouts]") {
    CHECK(grace_period_elapsed(5.0f) == true);
    CHECK(grace_period_elapsed(6.0f) == true);
    CHECK(grace_period_elapsed(100.0f) == true);
}

TEST_CASE("panic_lock_timeout_elapsed: false at or below 5.0s", "[timeouts]") {
    CHECK(panic_lock_timeout_elapsed(0.0f)  == false);
    CHECK(panic_lock_timeout_elapsed(5.0f)  == false);  // "> 5s" spec → not at exactly 5
}

TEST_CASE("panic_lock_timeout_elapsed: true strictly greater than 5.0s", "[timeouts]") {
    CHECK(panic_lock_timeout_elapsed(5.01f) == true);
    CHECK(panic_lock_timeout_elapsed(10.0f) == true);
}

TEST_CASE("coma_complete: false before 3600s", "[timeouts]") {
    CHECK(coma_complete(0.0f)    == false);
    CHECK(coma_complete(3599.0f) == false);
}

TEST_CASE("coma_complete: true at and after 3600s", "[timeouts]") {
    CHECK(coma_complete(3600.0f) == true);
    CHECK(coma_complete(7200.0f) == true);
}

// ---------------------------------------------------------------------------
// §14 — emergency_action_needed
// ---------------------------------------------------------------------------

TEST_CASE("emergency_action_needed: healthy ATP → NONE regardless of locks", "[emergency]") {
    CHECK(emergency_action_needed(5000.0f, 5, 10.0f) == EmergencyAction::NONE);
    CHECK(emergency_action_needed(1000.0f, 3,  6.0f) == EmergencyAction::NONE);
}

TEST_CASE("emergency_action_needed: HARD but no locks → NONE", "[emergency]") {
    CHECK(emergency_action_needed(400.0f, 0, 10.0f) == EmergencyAction::NONE);
}

TEST_CASE("emergency_action_needed: HARD + locks, short age → SET_PANIC", "[emergency]") {
    // age = 3s ≤ 5s → only SET_PANIC
    CHECK(emergency_action_needed(300.0f, 2, 3.0f) == EmergencyAction::SET_PANIC);
}

TEST_CASE("emergency_action_needed: HARD + locks, age exactly 5s → SET_PANIC (not over)", "[emergency]") {
    CHECK(emergency_action_needed(300.0f, 1, 5.0f) == EmergencyAction::SET_PANIC);
}

TEST_CASE("emergency_action_needed: HARD + locks, age > 5s → ENTER_COMA", "[emergency]") {
    CHECK(emergency_action_needed(100.0f, 3, 5.01f)  == EmergencyAction::ENTER_COMA);
    CHECK(emergency_action_needed(0.0f,   1, 100.0f) == EmergencyAction::ENTER_COMA);
}

TEST_CASE("emergency_action_needed: exactly at HARD boundary is CRITICAL zone → triggers", "[emergency]") {
    // 500 → classify_zone returns CRITICAL, so atp <= HARD triggers
    CHECK(emergency_action_needed(500.0f, 1, 0.0f) == EmergencyAction::SET_PANIC);
}

TEST_CASE("emergency_action_needed: one above HARD → NONE (not critical)", "[emergency]") {
    CHECK(emergency_action_needed(501.0f, 5, 10.0f) == EmergencyAction::NONE);
}

// ---------------------------------------------------------------------------
// §15 — Label helpers
// ---------------------------------------------------------------------------

TEST_CASE("zone_label: all zones have non-empty labels", "[labels]") {
    CHECK(!zone_label(MetabolicZone::NORMAL).empty());
    CHECK(!zone_label(MetabolicZone::WARNING).empty());
    CHECK(!zone_label(MetabolicZone::CRITICAL).empty());
}

TEST_CASE("zone_label: correct text", "[labels]") {
    CHECK(zone_label(MetabolicZone::NORMAL)   == "ZONE_I_NORMAL");
    CHECK(zone_label(MetabolicZone::WARNING)  == "ZONE_II_WARNING");
    CHECK(zone_label(MetabolicZone::CRITICAL) == "ZONE_III_CRITICAL");
}

TEST_CASE("nap_state_label: all states have non-empty labels", "[labels]") {
    CHECK(!nap_state_label(NapState::ACTIVE_WAKING).empty());
    CHECK(!nap_state_label(NapState::METABOLIC_WARNING).empty());
    CHECK(!nap_state_label(NapState::CRITICAL_EXHAUSTION).empty());
    CHECK(!nap_state_label(NapState::NAP).empty());
    CHECK(!nap_state_label(NapState::COMA).empty());
}

TEST_CASE("nap_state_label: correct text", "[labels]") {
    CHECK(nap_state_label(NapState::ACTIVE_WAKING)       == "ACTIVE_WAKING");
    CHECK(nap_state_label(NapState::METABOLIC_WARNING)   == "METABOLIC_WARNING");
    CHECK(nap_state_label(NapState::CRITICAL_EXHAUSTION) == "CRITICAL_EXHAUSTION");
    CHECK(nap_state_label(NapState::NAP)                 == "NAP");
    CHECK(nap_state_label(NapState::COMA)                == "COMA");
}

TEST_CASE("task_initiation_label: correct text", "[labels]") {
    CHECK(task_initiation_label(TaskInitiationResult::GRANTED)            == "GRANTED");
    CHECK(task_initiation_label(TaskInitiationResult::REJECTED_SOFT)      == "REJECTED_SOFT");
    CHECK(task_initiation_label(TaskInitiationResult::REJECTED_HARD)      == "REJECTED_HARD");
    CHECK(task_initiation_label(TaskInitiationResult::REJECTED_PREFLIGHT) == "REJECTED_PREFLIGHT");
}

TEST_CASE("lock_release_label: correct text", "[labels]") {
    CHECK(lock_release_label(LockReleaseOutcome::CLEAN)             == "CLEAN");
    CHECK(lock_release_label(LockReleaseOutcome::OVERDRAFT_PENALTY) == "OVERDRAFT_PENALTY");
    CHECK(lock_release_label(LockReleaseOutcome::PANIC_ABORT)       == "PANIC_ABORT");
}

TEST_CASE("wal_yield_label: correct text", "[labels]") {
    CHECK(wal_yield_label(WalYieldStatus::CONTINUE)        == "CONTINUE");
    CHECK(wal_yield_label(WalYieldStatus::YIELD_REQUESTED) == "YIELD_REQUESTED");
    CHECK(wal_yield_label(WalYieldStatus::PANIC_ABORT)     == "PANIC_ABORT");
}

TEST_CASE("emergency_action_label: correct text", "[labels]") {
    CHECK(emergency_action_label(EmergencyAction::NONE)       == "NONE");
    CHECK(emergency_action_label(EmergencyAction::SET_PANIC)  == "SET_PANIC");
    CHECK(emergency_action_label(EmergencyAction::DIRTY_DUMP) == "DIRTY_DUMP");
    CHECK(emergency_action_label(EmergencyAction::ENTER_COMA) == "ENTER_COMA");
}

// ---------------------------------------------------------------------------
// §16 — Integration / scenario tests
// ---------------------------------------------------------------------------

TEST_CASE("Scenario: PDF ingestion pipeline respects zones", "[scenario]") {
    // Spec example: 3 steps
    //   Step 1: chunk text  — 50 ATP
    //   Step 2: embeddings  — 500 ATP
    //   Step 3: LMDB store  — 20 ATP
    // Starting at full ATP; test preflight for each step
    float atp = ATP_MAX;

    // Step 1: 50 ATP cost
    CHECK(task_initiation_result(atp, TASK_COST_PDF_CHUNK_ATP) == TaskInitiationResult::GRANTED);
    atp -= TASK_COST_PDF_CHUNK_ATP;
    CHECK(atp == Approx(9950.0f));

    // Step 2: 500 ATP cost — still Zone I
    CHECK(task_initiation_result(atp, TASK_COST_EMBEDDINGS_ATP) == TaskInitiationResult::GRANTED);
    atp -= TASK_COST_EMBEDDINGS_ATP;
    CHECK(atp == Approx(9450.0f));

    // Step 3: 20 ATP cost
    CHECK(task_initiation_result(atp, TASK_COST_LMDB_STORE_ATP) == TaskInitiationResult::GRANTED);
}

TEST_CASE("Scenario: Zone transition walkthrough", "[scenario]") {
    // Start full, drain into each zone
    float atp = ATP_MAX;
    CHECK(classify_zone(atp) == MetabolicZone::NORMAL);

    // Drain to just inside Zone II
    atp = 1400.0f;
    CHECK(classify_zone(atp) == MetabolicZone::WARNING);
    CHECK(nap_state_for_atp(atp) == NapState::METABOLIC_WARNING);
    CHECK(should_yield(atp) == WalYieldStatus::YIELD_REQUESTED);

    // Drain to Zone III
    atp = 300.0f;
    CHECK(classify_zone(atp) == MetabolicZone::CRITICAL);
    CHECK(nap_state_for_atp(atp) == NapState::CRITICAL_EXHAUSTION);
    CHECK(should_yield(atp) == WalYieldStatus::PANIC_ABORT);
    CHECK(is_all_initiation_blocked(atp) == true);
}

TEST_CASE("Scenario: overdraft cascade reduces capacity across 3 cycles", "[scenario]") {
    float cap = ATP_MAX;
    cap = overdraft_e_max(cap); // 9000
    cap = overdraft_e_max(cap); // 8100
    cap = overdraft_e_max(cap); // 7290
    CHECK(cap == Approx(7290.0f).epsilon(0.01f));
    CHECK(cap < ATP_MAX);
}

TEST_CASE("Scenario: recharge from hard threshold back to full takes 190s", "[scenario]") {
    float t = seconds_to_full_recharge(HARD_THRESHOLD_ATP);
    CHECK(t == Approx(190.0f));
    float atp_after = atp_after_recharge(HARD_THRESHOLD_ATP, t);
    CHECK(atp_after == Approx(ATP_MAX));
}

TEST_CASE("Scenario: emergency escalates from SET_PANIC to ENTER_COMA after 5s", "[scenario]") {
    float atp     = 200.0f;  // deep into Zone III
    int   locks   = 3;

    // At 0s — SET_PANIC
    CHECK(emergency_action_needed(atp, locks, 0.0f)  == EmergencyAction::SET_PANIC);
    // At 5s — still SET_PANIC (not strictly over)
    CHECK(emergency_action_needed(atp, locks, 5.0f)  == EmergencyAction::SET_PANIC);
    // At 5.01s — ENTER_COMA
    CHECK(emergency_action_needed(atp, locks, 5.01f) == EmergencyAction::ENTER_COMA);
}

TEST_CASE("Scenario: coma lasts exactly 3600s then system can re-enter ACTIVE_WAKING", "[scenario]") {
    CHECK(coma_complete(3599.0f) == false);
    CHECK(coma_complete(3600.0f) == true);
    // after coma, system is assumed fully recharged:
    float atp_on_wake = ATP_MAX;
    CHECK(classify_zone(atp_on_wake) == MetabolicZone::NORMAL);
    CHECK(nap_state_for_atp(atp_on_wake) == NapState::ACTIVE_WAKING);
}
