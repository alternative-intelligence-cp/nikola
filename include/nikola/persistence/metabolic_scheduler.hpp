#pragma once
// =============================================================================
// nikola/persistence/metabolic_scheduler.hpp
// Phase 80 — GAP-013: Transactional Metabolic Scheduling with RAII Locks
//
// SOURCE: Gemini Deep Research Round 2, Tasks 13-15 (December 14, 2025)
// SPEC:   docs/info/integration/sections/06_persistence/04_nap_system.md
//         §GAP-013 (lines 4302–4397)
//
// Models the three-tier ATP energy model that prevents thermodynamic race
// conditions when the system transitions from Active Waking → Metabolic
// Warning → Critical Exhaustion → Nap → (Coma on panic).
//
// All constants are explicit spec values; all logic is pure constexpr —
// no external runtime required.
// =============================================================================

#include <cstdint>
#include <string_view>
#include <cmath>

namespace nikola::persistence {

// ---------------------------------------------------------------------------
// § Enumerations
// ---------------------------------------------------------------------------

/// Metabolic zone classification.  Maps directly to table rows in §GAP-013.
///   Zone I   — Normal         (ATP >  1500, > 15%)
///   Zone II  — Soft Limit     (ATP  500–1500, 5–15%)
///   Zone III — Hard Limit     (ATP ≤  500, ≤ 5%)
enum class MetabolicZone : uint8_t {
    NORMAL   = 0,   ///< Zone I  — unrestricted task initiation
    WARNING  = 1,   ///< Zone II — high-cost tasks rejected; running continue
    CRITICAL = 2,   ///< Zone III — forced nap; grace period for active locks
};

/// Cognitive operational state driven by MetabolicZone transitions.
enum class NapState : uint8_t {
    ACTIVE_WAKING       = 0,  ///< Zone I  — full processing
    METABOLIC_WARNING   = 1,  ///< Zone II — conservative mode
    CRITICAL_EXHAUSTION = 2,  ///< Zone III — graceful drain
    NAP                 = 3,  ///< Recharging (RECHARGE_RATE_PER_S ATP/s)
    COMA                = 4,  ///< Emergency panic-mode rest (1 hr)
};

/// Result returned by the pre-flight / task-initiation gating logic.
enum class TaskInitiationResult : uint8_t {
    GRANTED           = 0,  ///< atp OK, preflight OK → lock may be acquired
    REJECTED_SOFT     = 1,  ///< Zone II — task is high-cost; reject
    REJECTED_HARD     = 2,  ///< Zone III — all initiation blocked
    REJECTED_PREFLIGHT= 3,  ///< atp − cost ≤ E_critical_reserve
};

/// Outcome classification when a RAII ScopedLock is released.
enum class LockReleaseOutcome : uint8_t {
    CLEAN            = 0,  ///< Released with ATP still above HARD_THRESHOLD
    OVERDRAFT_PENALTY= 1,  ///< ATP dipped below HARD_THRESHOLD during lock
    PANIC_ABORT      = 2,  ///< Emergency abort triggered before release
};

/// WAL micro-transaction yield decision.
enum class WalYieldStatus : uint8_t {
    CONTINUE       = 0,  ///< atp OK, keep processing the current WAL chunk
    YIELD_REQUESTED= 1,  ///< atp below SOFT_THRESHOLD — commit WAL, release
    PANIC_ABORT    = 2,  ///< atp at or below HARD_THRESHOLD — force break
};

/// Emergency actions escalated when PANIC_LOCK_TIMEOUT is exceeded.
enum class EmergencyAction : uint8_t {
    NONE        = 0,  ///< Conditions not met
    SET_PANIC   = 1,  ///< panic_mode = true; physics loops break
    DIRTY_DUMP  = 2,  ///< Write TorusGridSoA to crash.nik
    ENTER_COMA  = 3,  ///< Transition to NapState::COMA (1 hr recharge)
};

// ---------------------------------------------------------------------------
// § Spec Constants — §GAP-013 all explicit, no TBDs
// ---------------------------------------------------------------------------

/// Maximum ATP reservoir (100% capacity).
inline constexpr float ATP_MAX = 10'000.0f;

/// ATP replenishment rate during nap/recharge cycles (ATP per second).
inline constexpr float RECHARGE_RATE_PER_S = 50.0f;

/// Zone I/II boundary — ATP absolute value (= ATP_MAX × 0.15).
inline constexpr float SOFT_THRESHOLD_ATP = 1'500.0f;

/// Zone II/III boundary — ATP absolute value (= ATP_MAX × 0.05).
inline constexpr float HARD_THRESHOLD_ATP = 500.0f;

/// Zone I/II boundary as integer percentage.
inline constexpr int   SOFT_THRESHOLD_PCT = 15;

/// Zone II/III boundary as integer percentage.
inline constexpr int   HARD_THRESHOLD_PCT = 5;

/// Criticality reserve used in pre-flight feasibility: same as HARD_THRESHOLD.
/// Derived: E_critical_reserve ≡ HARD_THRESHOLD_ATP (500 ATP).
inline constexpr float CRITICAL_RESERVE_ATP = HARD_THRESHOLD_ATP;

/// Grace period for active locks when entering Zone III (seconds).
/// Spec derivation: T_grace = E_hard / Ė = 500 / 100 = 5.0 s.
inline constexpr float GRACE_PERIOD_S = 5.0f;

/// Implied ATP burn rate used in T_grace derivation (ATP per unit).
inline constexpr float BURN_RATE_PER_UNIT = 100.0f;

/// WAL micro-transaction chunk cost (ATP per chunk).
/// Long-running tasks checkpoint every ≈100 ATP of work.
inline constexpr float WAL_CHUNK_COST_ATP = 100.0f;

/// Overdraft penalty fraction: E_max_next = E_max × (1 − OVERDRAFT_PENALTY_FACTOR).
inline constexpr float OVERDRAFT_PENALTY_FACTOR = 0.10f;

/// Coma recharge duration in hours (1 hr, from "1hr recharge" spec text).
inline constexpr int   COMA_RECHARGE_HOURS = 1;

/// Coma recharge duration in seconds (= COMA_RECHARGE_HOURS × 3600).
inline constexpr float COMA_RECHARGE_S = 3'600.0f;

/// Lock age (seconds) before emergency abort is triggered at HARD_THRESHOLD.
/// Spec: "if active_locks > 0 for >5s" triggers panic.
inline constexpr float PANIC_LOCK_TIMEOUT_S = 5.0f;

/// Number of Power Iteration steps used in SpectralStabilizer (cross-ref GAP-032).
inline constexpr int   POWER_ITER_STEPS = 5;

/// Percentage of ATP_MAX that is reclaimed per full recharge nap cycle.
/// During a nap the system recharges until atp == ATP_MAX at RECHARGE_RATE_PER_S.
/// Time to full recharge from HARD_THRESHOLD: (ATP_MAX − HARD_THRESHOLD) / RECHARGE_RATE_PER_S = 190 s.
inline constexpr float FULL_RECHARGE_FROM_HARD_S =
    (ATP_MAX - HARD_THRESHOLD_ATP) / RECHARGE_RATE_PER_S;

/// Example task costs from spec (for documentation; used in tests).
inline constexpr float TASK_COST_PDF_CHUNK_ATP     = 50.0f;
inline constexpr float TASK_COST_EMBEDDINGS_ATP    = 500.0f;
inline constexpr float TASK_COST_LMDB_STORE_ATP    = 20.0f;

// ---------------------------------------------------------------------------
// § Zone classification
// ---------------------------------------------------------------------------

/// Return the MetabolicZone for a given ATP level.
/// Boundaries are inclusive/exclusive as per spec table:
///   NORMAL   : atp >  1500
///   WARNING  : atp in [500, 1500]
///   CRITICAL : atp <  500  (≤ HARD_THRESHOLD)
[[nodiscard]] constexpr MetabolicZone classify_zone(float atp) noexcept {
    if (atp > SOFT_THRESHOLD_ATP)  return MetabolicZone::NORMAL;
    if (atp > HARD_THRESHOLD_ATP)  return MetabolicZone::WARNING;
    return MetabolicZone::CRITICAL;
}

/// Map a MetabolicZone directly to its corresponding NapState.
[[nodiscard]] constexpr NapState nap_state_for_zone(MetabolicZone z) noexcept {
    switch (z) {
        case MetabolicZone::NORMAL:   return NapState::ACTIVE_WAKING;
        case MetabolicZone::WARNING:  return NapState::METABOLIC_WARNING;
        case MetabolicZone::CRITICAL: return NapState::CRITICAL_EXHAUSTION;
    }
    return NapState::CRITICAL_EXHAUSTION;
}

/// Convenience overload: NapState directly from ATP level.
[[nodiscard]] constexpr NapState nap_state_for_atp(float atp) noexcept {
    return nap_state_for_zone(classify_zone(atp));
}

// ---------------------------------------------------------------------------
// § Task initiation gating
// ---------------------------------------------------------------------------

/// Returns true only when full (Zone I / ACTIVE_WAKING) task initiation
/// is permitted — i.e., any task type may begin.
[[nodiscard]] constexpr bool is_unrestricted_initiation(float atp) noexcept {
    return classify_zone(atp) == MetabolicZone::NORMAL;
}

/// Returns true when new high-cost task initiation is blocked
/// (Zone II: WARNING or Zone III: CRITICAL).
[[nodiscard]] constexpr bool is_high_cost_task_blocked(float atp) noexcept {
    return classify_zone(atp) != MetabolicZone::NORMAL;
}

/// Returns true when ALL new task initiation is blocked (Zone III only).
[[nodiscard]] constexpr bool is_all_initiation_blocked(float atp) noexcept {
    return classify_zone(atp) == MetabolicZone::CRITICAL;
}

/// Pre-flight feasibility check: may a task costing `cost_atp` be initiated
/// while the system currently holds `current_atp`, keeping `reserve_atp`
/// as a safety floor?
/// Spec formula: E(t) − (N × C_est) > E_critical_reserve
[[nodiscard]] constexpr bool preflight_feasible(
    float current_atp,
    float cost_atp,
    float reserve_atp = CRITICAL_RESERVE_ATP) noexcept
{
    return (current_atp - cost_atp) > reserve_atp;
}

/// Full initiation gate: zone check + pre-flight.
/// Returns the exact TaskInitiationResult describing why access is granted/denied.
[[nodiscard]] constexpr TaskInitiationResult
task_initiation_result(float current_atp, float cost_atp) noexcept {
    MetabolicZone z = classify_zone(current_atp);
    if (z == MetabolicZone::CRITICAL) return TaskInitiationResult::REJECTED_HARD;
    if (z == MetabolicZone::WARNING)  return TaskInitiationResult::REJECTED_SOFT;
    if (!preflight_feasible(current_atp, cost_atp))
        return TaskInitiationResult::REJECTED_PREFLIGHT;
    return TaskInitiationResult::GRANTED;
}

// ---------------------------------------------------------------------------
// § WAL yield / should_yield
// ---------------------------------------------------------------------------

/// Determine whether a long-running WAL loop should pause after this chunk.
/// CONTINUE     : atp > SOFT_THRESHOLD (Zone I — keep going)
/// YIELD_REQUESTED : in Zone II (WARNING)  — commit WAL and release lock
/// PANIC_ABORT  : at or below HARD_THRESHOLD (Zone III)  — force stop
[[nodiscard]] constexpr WalYieldStatus should_yield(float atp) noexcept {
    if (atp <= HARD_THRESHOLD_ATP)    return WalYieldStatus::PANIC_ABORT;
    if (atp <= SOFT_THRESHOLD_ATP)    return WalYieldStatus::YIELD_REQUESTED;
    return WalYieldStatus::CONTINUE;
}

// ---------------------------------------------------------------------------
// § Lock release outcome
// ---------------------------------------------------------------------------

/// Classify the outcome of releasing a ScopedLock given the ATP at release time.
[[nodiscard]] constexpr LockReleaseOutcome
lock_release_outcome(float atp_at_release) noexcept {
    if (atp_at_release < HARD_THRESHOLD_ATP) return LockReleaseOutcome::OVERDRAFT_PENALTY;
    return LockReleaseOutcome::CLEAN;
}

// ---------------------------------------------------------------------------
// § Overdraft penalty
// ---------------------------------------------------------------------------

/// Compute the new E_max after an overdraft event.
/// Spec: E_max_next = E_max × (1 − 0.10)
/// System wakes with 10% less capacity to force conservative planning.
[[nodiscard]] constexpr float overdraft_e_max(float e_max) noexcept {
    return e_max * (1.0f - OVERDRAFT_PENALTY_FACTOR);
}

/// Number of consecutive overdraft penalties before capacity falls below
/// a given floor.  Useful for alerting / testing convergence.
/// n_penalties such that ATP_MAX × (1-0.1)^n < floor
[[nodiscard]] constexpr int overdraft_penalties_until(
    float floor_atp,
    float e_max = ATP_MAX) noexcept
{
    int n = 0;
    float cap = e_max;
    while (cap > floor_atp && n < 1000) {
        cap *= (1.0f - OVERDRAFT_PENALTY_FACTOR);
        ++n;
    }
    return n;
}

// ---------------------------------------------------------------------------
// § Recharge arithmetic
// ---------------------------------------------------------------------------

/// Compute ATP after one recharge tick of `dt` seconds.
/// Clamped at ATP_MAX — cannot exceed maximum capacity.
[[nodiscard]] constexpr float recharge_delta(float current_atp, float dt) noexcept {
    float next = current_atp + RECHARGE_RATE_PER_S * dt;
    return next < ATP_MAX ? next : ATP_MAX;
}

/// Number of recharge seconds required to reach full capacity from `start_atp`.
[[nodiscard]] constexpr float seconds_to_full_recharge(float start_atp) noexcept {
    if (start_atp >= ATP_MAX) return 0.0f;
    return (ATP_MAX - start_atp) / RECHARGE_RATE_PER_S;
}

/// ATP level after `elapsed_s` seconds of nap-recharge starting from `start_atp`.
[[nodiscard]] constexpr float atp_after_recharge(float start_atp, float elapsed_s) noexcept {
    float next = start_atp + RECHARGE_RATE_PER_S * elapsed_s;
    return next < ATP_MAX ? next : ATP_MAX;
}

// ---------------------------------------------------------------------------
// § ATP percentage
// ---------------------------------------------------------------------------

/// Convert absolute ATP to percentage of ATP_MAX (0.0–100.0).
[[nodiscard]] constexpr float atp_percentage(float atp, float e_max = ATP_MAX) noexcept {
    if (e_max <= 0.0f) return 0.0f;
    return (atp / e_max) * 100.0f;
}

/// True if the ATP percentage matches the soft threshold band [5%, 15%].
[[nodiscard]] constexpr bool is_in_warning_band(float atp_pct) noexcept {
    return atp_pct > static_cast<float>(HARD_THRESHOLD_PCT)
        && atp_pct <= static_cast<float>(SOFT_THRESHOLD_PCT);
}

// ---------------------------------------------------------------------------
// § Timeout predicates
// ---------------------------------------------------------------------------

/// Grace period: returns true once `elapsed_s` has exceeded GRACE_PERIOD_S.
/// Used by the lock-drain protocol before forced Nap entry.
[[nodiscard]] constexpr bool grace_period_elapsed(float elapsed_s) noexcept {
    return elapsed_s >= GRACE_PERIOD_S;
}

/// Panic lock timeout: returns true if active_locks remain after PANIC_LOCK_TIMEOUT_S.
/// Triggers the three-step emergency abort (panic, dump, coma).
[[nodiscard]] constexpr bool panic_lock_timeout_elapsed(float elapsed_s) noexcept {
    return elapsed_s > PANIC_LOCK_TIMEOUT_S;
}

/// Coma ended: true once the system has rested for the full COMA_RECHARGE_S.
[[nodiscard]] constexpr bool coma_complete(float elapsed_s) noexcept {
    return elapsed_s >= COMA_RECHARGE_S;
}

// ---------------------------------------------------------------------------
// § Emergency abort escalation
// ---------------------------------------------------------------------------

/// Determine which emergency action is needed.
/// Spec: "if E(t) ≤ E_hard AND active_locks > 0 for >5s":
///   Step 1 — SET_PANIC  (returned on first call, implying all three fire in order)
///   Step 2 — DIRTY_DUMP (after panic set)
///   Step 3 — ENTER_COMA (final state)
///
/// For the pure policy model we return the *highest* urgency action reached:
///   ENTER_COMA  if atp ≤ HARD and active_locks > 0 and lock_age_s > PANIC_LOCK_TIMEOUT_S
///   SET_PANIC   if atp ≤ HARD and active_locks > 0
///   NONE        otherwise
[[nodiscard]] constexpr EmergencyAction
emergency_action_needed(float atp, int active_locks, float lock_age_s) noexcept {
    if (atp > HARD_THRESHOLD_ATP || active_locks <= 0)
        return EmergencyAction::NONE;
    if (panic_lock_timeout_elapsed(lock_age_s))
        return EmergencyAction::ENTER_COMA;
    return EmergencyAction::SET_PANIC;
}

// ---------------------------------------------------------------------------
// § Label helpers
// ---------------------------------------------------------------------------

[[nodiscard]] constexpr std::string_view zone_label(MetabolicZone z) noexcept {
    switch (z) {
        case MetabolicZone::NORMAL:   return "ZONE_I_NORMAL";
        case MetabolicZone::WARNING:  return "ZONE_II_WARNING";
        case MetabolicZone::CRITICAL: return "ZONE_III_CRITICAL";
    }
    return "UNKNOWN_ZONE";
}

[[nodiscard]] constexpr std::string_view nap_state_label(NapState s) noexcept {
    switch (s) {
        case NapState::ACTIVE_WAKING:       return "ACTIVE_WAKING";
        case NapState::METABOLIC_WARNING:   return "METABOLIC_WARNING";
        case NapState::CRITICAL_EXHAUSTION: return "CRITICAL_EXHAUSTION";
        case NapState::NAP:                 return "NAP";
        case NapState::COMA:                return "COMA";
    }
    return "UNKNOWN_NAP_STATE";
}

[[nodiscard]] constexpr std::string_view task_initiation_label(
    TaskInitiationResult r) noexcept
{
    switch (r) {
        case TaskInitiationResult::GRANTED:            return "GRANTED";
        case TaskInitiationResult::REJECTED_SOFT:      return "REJECTED_SOFT";
        case TaskInitiationResult::REJECTED_HARD:      return "REJECTED_HARD";
        case TaskInitiationResult::REJECTED_PREFLIGHT: return "REJECTED_PREFLIGHT";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr std::string_view lock_release_label(
    LockReleaseOutcome o) noexcept
{
    switch (o) {
        case LockReleaseOutcome::CLEAN:             return "CLEAN";
        case LockReleaseOutcome::OVERDRAFT_PENALTY: return "OVERDRAFT_PENALTY";
        case LockReleaseOutcome::PANIC_ABORT:       return "PANIC_ABORT";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr std::string_view wal_yield_label(WalYieldStatus s) noexcept {
    switch (s) {
        case WalYieldStatus::CONTINUE:        return "CONTINUE";
        case WalYieldStatus::YIELD_REQUESTED: return "YIELD_REQUESTED";
        case WalYieldStatus::PANIC_ABORT:     return "PANIC_ABORT";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr std::string_view emergency_action_label(
    EmergencyAction a) noexcept
{
    switch (a) {
        case EmergencyAction::NONE:       return "NONE";
        case EmergencyAction::SET_PANIC:  return "SET_PANIC";
        case EmergencyAction::DIRTY_DUMP: return "DIRTY_DUMP";
        case EmergencyAction::ENTER_COMA: return "ENTER_COMA";
    }
    return "UNKNOWN";
}

} // namespace nikola::persistence
