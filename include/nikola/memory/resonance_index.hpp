#pragma once
// ============================================================
// nikola/memory/resonance_index.hpp
// GAP-024  Ingestion Pipeline → Resonance Index Synchronization
// Namespace: nikola::memory
// C++23 — header-only, stateless
// ============================================================
//
// Encodes the policy rules for the asynchronous LSM-style
// synchronization protocol that decouples the Ingestion Pipeline
// (write path) from the Physics Engine (read path), eliminating
// the "Neurogenesis Seizure" pathology (Finding MEM-04).
//
// Three-phase protocol:
//   Phase 1 — Ingestion (Write):   O(1) CAS insert into MemTable
//   Phase 2 — Propagation (Merge): Background shadow merge (100ms poll)
//   Phase 3 — Atomic Swap:         Pointer exchange at tick boundary
//
// Visibility lag formula:
//   T_lag = T_batch + T_merge + T_swap  ≤  MAX_VISIBILITY_LAG_MS  (500 ms)
// ============================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace nikola::memory {

// ────────────────────────────────────────────────────────────────────────────
// §1  LSM synchronisation constants
//     (spec §GAP-024 "Synchronization Protocol" and "Performance")
// ────────────────────────────────────────────────────────────────────────────

/// Minor merge trigger: MemTable size threshold (nodes)
constexpr size_t   MEMTABLE_FLUSH_THRESHOLD    = 10'000u;

/// Minor merge trigger: maximum elapsed time without flush (milliseconds)
constexpr uint64_t FLUSH_INTERVAL_MS           = 1'000u;   // 1 second

/// Merger thread poll interval (milliseconds)
constexpr uint64_t MERGE_POLL_INTERVAL_MS      = 100u;

/// Maximum acceptable visibility lag from ingestion to active index (ms)
/// Spec rationale: mimics human short-term memory encoding latency
constexpr uint64_t MAX_VISIBILITY_LAG_MS       = 500u;

/// Minor merge (MemTable → Level-0 SSTable) must complete within this time
constexpr uint64_t MINOR_MERGE_MAX_MS          = 10u;

/// Major rebuild trigger: fragmentation index above this fraction triggers
/// full Hilbert-sorted consolidation (20% poor spatial locality)
constexpr float    FRAGMENTATION_REBUILD_THRESHOLD = 0.20f;

/// Major rebuild trigger: ATP reserve below this fraction triggers rebuild
/// during Nap State (15% reserve = low-power mode)
constexpr float    MAJOR_REBUILD_ATP_THRESHOLD = 0.15f;

// ────────────────────────────────────────────────────────────────────────────
// §2  Enumeration types
// ────────────────────────────────────────────────────────────────────────────

/// Which LSM merge level was triggered
enum class MergeLevel : uint8_t {
    NONE,   ///< No merge needed
    MINOR,  ///< MemTable → Level-0 SSTable (fast, lightweight)
    MAJOR   ///< Full Hilbert re-sort and consolidation (Nap-only)
};

/// Reason a major rebuild was initiated
enum class RebuildReason : uint8_t {
    NONE,           ///< No rebuild needed
    NAP_ATP,        ///< ATP < 15%: system in Nap State
    FRAGMENTATION,  ///< Fragmentation index > 20%
    BOTH            ///< Both conditions simultaneously active
};

/// State of the Active Index during a merge cycle
enum class IndexState : uint8_t {
    STABLE,       ///< Active index serving queries; no merge in progress
    SHADOW_BUILD, ///< Shadow index being built; Active still serving
    SWAPPING      ///< Nanosecond atomic swap in progress (tick boundary)
};

// ────────────────────────────────────────────────────────────────────────────
// §3  Minor merge trigger conditions
//     Trigger = (MemTable.size > 10,000) OR (elapsed_ms > 1,000)
// ────────────────────────────────────────────────────────────────────────────

/// Returns true when the MemTable has exceeded the node count threshold.
[[nodiscard]] inline bool is_flush_threshold_exceeded(size_t memtable_size) noexcept
{
    return memtable_size > MEMTABLE_FLUSH_THRESHOLD;
}

/// Returns true when the interval since the last merge exceeds 1 second.
[[nodiscard]] inline bool is_flush_interval_elapsed(uint64_t elapsed_ms) noexcept
{
    return elapsed_ms > FLUSH_INTERVAL_MS;
}

/// Returns true when either minor-merge trigger condition is satisfied.
[[nodiscard]] inline bool should_trigger_minor_merge(
    size_t   memtable_size,
    uint64_t elapsed_ms) noexcept
{
    return is_flush_threshold_exceeded(memtable_size) ||
           is_flush_interval_elapsed(elapsed_ms);
}

// ────────────────────────────────────────────────────────────────────────────
// §4  Major rebuild trigger conditions
//     Trigger = (ATP < 15%) OR (fragmentation > 20%)
// ────────────────────────────────────────────────────────────────────────────

/// Returns true when ATP reserves are low enough to enter Nap State.
///
/// @param atp_fraction  Current ATP as fraction [0, 1]
/// @throws std::invalid_argument if atp_fraction is outside [0, 1]
[[nodiscard]] inline bool is_major_rebuild_triggered_by_atp(float atp_fraction)
{
    if (atp_fraction < 0.0f || atp_fraction > 1.0f)
        throw std::invalid_argument(
            "is_major_rebuild_triggered_by_atp: atp_fraction must be in [0, 1]");
    return atp_fraction < MAJOR_REBUILD_ATP_THRESHOLD;
}

/// Returns true when spatial fragmentation warrants a Hilbert re-sort.
///
/// @param fragmentation_fraction  Fraction of nodes with poor spatial locality
/// @throws std::invalid_argument if fragmentation_fraction is outside [0, 1]
[[nodiscard]] inline bool is_major_rebuild_triggered_by_fragmentation(
    float fragmentation_fraction)
{
    if (fragmentation_fraction < 0.0f || fragmentation_fraction > 1.0f)
        throw std::invalid_argument(
            "is_major_rebuild_triggered_by_fragmentation: "
            "fragmentation_fraction must be in [0, 1]");
    return fragmentation_fraction > FRAGMENTATION_REBUILD_THRESHOLD;
}

/// Returns the most specific RebuildReason for the given system state.
///
/// @throws std::invalid_argument if either parameter is outside [0, 1]
[[nodiscard]] inline RebuildReason classify_rebuild_reason(
    float atp_fraction,
    float fragmentation_fraction)
{
    const bool atp_trig  = is_major_rebuild_triggered_by_atp(atp_fraction);
    const bool frag_trig = is_major_rebuild_triggered_by_fragmentation(
                               fragmentation_fraction);

    if (atp_trig && frag_trig) return RebuildReason::BOTH;
    if (atp_trig)              return RebuildReason::NAP_ATP;
    if (frag_trig)             return RebuildReason::FRAGMENTATION;
    return RebuildReason::NONE;
}

/// Returns true when any major rebuild condition is active.
[[nodiscard]] inline bool should_trigger_major_rebuild(
    float atp_fraction,
    float fragmentation_fraction)
{
    return classify_rebuild_reason(atp_fraction, fragmentation_fraction)
               != RebuildReason::NONE;
}

// ────────────────────────────────────────────────────────────────────────────
// §5  Merge level selection
//     Major takes priority over minor (BOTH active → MAJOR)
// ────────────────────────────────────────────────────────────────────────────

/// Determine which merge level (if any) is required.
///
/// @param memtable_size           Current MemTable node count
/// @param elapsed_ms              Milliseconds since last merge
/// @param atp_fraction            Current ATP [0, 1]
/// @param fragmentation_fraction  Current fragmentation index [0, 1]
[[nodiscard]] inline MergeLevel classify_merge_level(
    size_t   memtable_size,
    uint64_t elapsed_ms,
    float    atp_fraction,
    float    fragmentation_fraction)
{
    if (should_trigger_major_rebuild(atp_fraction, fragmentation_fraction))
        return MergeLevel::MAJOR;
    if (should_trigger_minor_merge(memtable_size, elapsed_ms))
        return MergeLevel::MINOR;
    return MergeLevel::NONE;
}

// ────────────────────────────────────────────────────────────────────────────
// §6  Visibility lag  (T_lag = T_batch + T_merge + T_swap)
// ────────────────────────────────────────────────────────────────────────────

/// Compute the expected visibility lag for a given batch / merge / swap cost.
///
/// @param T_batch_ms  Time for batch collection / MemTable drain (ms)
/// @param T_merge_ms  Time for shadow index construction and optimisation (ms)
/// @param T_swap_ms   Nanoseconds → converted ms; time for atomic pointer swap
/// @return            T_lag in milliseconds
/// @throws std::invalid_argument if any component is negative
[[nodiscard]] inline double visibility_lag_ms(
    double T_batch_ms,
    double T_merge_ms,
    double T_swap_ms)
{
    if (T_batch_ms < 0.0 || T_merge_ms < 0.0 || T_swap_ms < 0.0)
        throw std::invalid_argument(
            "visibility_lag_ms: all time components must be non-negative");
    return T_batch_ms + T_merge_ms + T_swap_ms;
}

/// Returns true when the computed visibility lag satisfies the spec budget.
[[nodiscard]] inline bool is_visibility_lag_acceptable(double lag_ms) noexcept
{
    return lag_ms <= static_cast<double>(MAX_VISIBILITY_LAG_MS);
}

/// Returns true when a minor merge completed within its allowed 10 ms budget.
[[nodiscard]] inline bool is_minor_merge_within_budget(double merge_ms) noexcept
{
    return merge_ms < static_cast<double>(MINOR_MERGE_MAX_MS);
}

// ────────────────────────────────────────────────────────────────────────────
// §7  Snapshot isolation predicates
// ────────────────────────────────────────────────────────────────────────────

/// Returns true when a snapshot has active readers that prevent deletion.
///
/// Spec: std::shared_ptr ref-counting or hazard pointers guarantee that
/// the snapshot is not freed whilst any query holds a reference.
/// A ref_count of 1 means only the active_snapshot atomic holds it
/// (no live queries); ≥ 2 means at least one reader is active.
[[nodiscard]] inline bool has_active_readers(long ref_count) noexcept
{
    return ref_count >= 2;
}

/// Returns true when a snapshot is safe to delete (no live readers).
[[nodiscard]] inline bool is_snapshot_safe_to_discard(long ref_count) noexcept
{
    return ref_count <= 1;
}

// ────────────────────────────────────────────────────────────────────────────
// §8  Hilbert key ordering (spatial locality invariant)
// ────────────────────────────────────────────────────────────────────────────

/// Compare two Hilbert keys: returns true if a precedes b in linearised order.
///
/// The Active Index must be maintained in ascending Hilbert-key order after
/// each merge so that spatially adjacent nodes remain cache-adjacent.
[[nodiscard]] inline bool hilbert_precedes(uint64_t a, uint64_t b) noexcept
{
    return a < b;
}

/// Returns true when an array of Hilbert keys is sorted (spatial locality ok).
[[nodiscard]] inline bool is_hilbert_sorted(
    const uint64_t* keys,
    size_t          n) noexcept
{
    if (n == 0u) return true;
    for (size_t i = 1; i < n; ++i)
        if (keys[i] < keys[i - 1]) return false;
    return true;
}

/// Returns true when a single key lies within a contiguous Hilbert range
/// [range_lo, range_hi] (inclusive) — used for partition routing.
[[nodiscard]] inline bool key_in_range(
    uint64_t key,
    uint64_t range_lo,
    uint64_t range_hi) noexcept
{
    return key >= range_lo && key <= range_hi;
}

// ────────────────────────────────────────────────────────────────────────────
// §9  Merge poll scheduling
// ────────────────────────────────────────────────────────────────────────────

/// Returns true when the merger thread's next wake-up is due.
///
/// The merger polls every MERGE_POLL_INTERVAL_MS (100 ms).
/// @param elapsed_since_last_poll_ms  Time since the thread last checked
[[nodiscard]] inline bool is_merge_poll_due(
    uint64_t elapsed_since_last_poll_ms) noexcept
{
    return elapsed_since_last_poll_ms >= MERGE_POLL_INTERVAL_MS;
}

/// Number of full merge-poll cycles in a given elapsed time window.
[[nodiscard]] inline uint64_t poll_cycles_in_window(uint64_t window_ms) noexcept
{
    if (MERGE_POLL_INTERVAL_MS == 0u) return 0u;
    return window_ms / MERGE_POLL_INTERVAL_MS;
}

// ────────────────────────────────────────────────────────────────────────────
// §10  Atomic-swap safety predicates
// ────────────────────────────────────────────────────────────────────────────

/// Returns true when the system is at a safe tick boundary for the atomic swap.
///
/// Spec: swap occurs in a "microsecond window between ticks".
/// This predicate abstracts that check: swap is only permitted when
/// the physics engine has just finished a tick (between_ticks == true).
[[nodiscard]] inline bool is_safe_swap_point(bool between_ticks) noexcept
{
    return between_ticks;
}

/// Returns true when the shadow index has been fully built and is ready
/// to replace the active index.
[[nodiscard]] inline bool shadow_ready_for_swap(IndexState state) noexcept
{
    return state == IndexState::SHADOW_BUILD;   // valid to exchange now
}

// ────────────────────────────────────────────────────────────────────────────
// §11  Diagnostic name functions
// ────────────────────────────────────────────────────────────────────────────

[[nodiscard]] inline std::string_view merge_level_name(MergeLevel l) noexcept
{
    switch (l) {
        case MergeLevel::NONE:  return "NONE";
        case MergeLevel::MINOR: return "MINOR";
        case MergeLevel::MAJOR: return "MAJOR";
        default:                return "UNKNOWN";
    }
}

[[nodiscard]] inline std::string_view rebuild_reason_name(RebuildReason r) noexcept
{
    switch (r) {
        case RebuildReason::NONE:          return "NONE";
        case RebuildReason::NAP_ATP:       return "NAP_ATP";
        case RebuildReason::FRAGMENTATION: return "FRAGMENTATION";
        case RebuildReason::BOTH:          return "BOTH";
        default:                           return "UNKNOWN";
    }
}

[[nodiscard]] inline std::string_view index_state_name(IndexState s) noexcept
{
    switch (s) {
        case IndexState::STABLE:       return "STABLE";
        case IndexState::SHADOW_BUILD: return "SHADOW_BUILD";
        case IndexState::SWAPPING:     return "SWAPPING";
        default:                       return "UNKNOWN";
    }
}

} // namespace nikola::memory
