// SPDX-License-Identifier: MIT
// GAP-024: Ingestion Pipeline → Resonance Index Synchronisation Policy
// Phase 92 — nikola::memory
//
// Constants and predicates that govern the LSM-style asynchronous merge
// loop that keeps the ResonanceIndex consistent with the Ingestion Pipeline
// without stalling the Physics Engine (the "Neurogenesis Seizure" fix).
//
// Architecture:
//   MemTable (lock-free skip-list, write path)
//       → background Merger Thread
//           → Shadow Index (copy-on-write)
//               → atomic pointer swap at tick boundary
//                   → Active Index (read path, Physics Engine)
//
// Source: 04_memory_data_systems.md §"GAP-024 Ingestion Pipeline Sync"

#pragma once

#include <cstdint>
#include <string_view>

namespace nikola::memory {

// ─── Merger trigger thresholds ────────────────────────────────────────────────

/// Trigger a minor (incremental) merge when the MemTable accumulates this
/// many new nodes, whichever comes first vs. the time threshold.
inline constexpr uint32_t MEMTABLE_MERGE_TRIGGER_NODES   = 10'000;

/// Trigger a minor merge if this many milliseconds have elapsed since the
/// last merge, regardless of MemTable size.
inline constexpr uint32_t MEMTABLE_MERGE_TRIGGER_MS       = 1'000;   ///< 1 second

/// Background merger thread sleep interval between trigger checks (ms).
inline constexpr uint32_t MERGER_POLL_INTERVAL_MS         = 100;

// ─── Full-rebuild triggers ────────────────────────────────────────────────────

/// ATP reserve fraction below which the system is considered in "Nap" state;
/// a full index rebuild (re-sort by Hilbert curve) is safe to run.
inline constexpr double   REBUILD_ATP_NAP_THRESHOLD        = 0.15;

/// Spatial fragmentation index above which a full rebuild restores locality.
/// Fragmentation = fraction of nodes whose Hilbert order is out-of-sequence.
inline constexpr double   REBUILD_FRAGMENTATION_THRESHOLD  = 0.20;   ///< 20 %

// ─── Consistency / latency specification ─────────────────────────────────────

/// Maximum permitted visibility lag: time between node ingestion completing
/// and that node being active in the Physics Engine's Active Index.
/// T_lag = T_batch + T_merge + T_swap  ≤  500 ms
inline constexpr uint32_t VISIBILITY_LAG_MAX_MS            = 500;

// ─── Snapshot isolation ───────────────────────────────────────────────────────

/// Strategy used for snapshot isolation during queries: std::shared_ptr
/// reference-count based hazard (readers hold a shared_ptr to active snapshot).
inline constexpr std::string_view SNAPSHOT_ISOLATION_STRATEGY = "shared_ptr_refcount";

// ─── Merge types ──────────────────────────────────────────────────────────────

enum class MergeType : uint8_t {
    INCREMENTAL  = 0,  ///< Minor: MemTable → Level-0 SSTable (fast, low impact)
    FULL_REBUILD = 1   ///< Major: Consolidate all SSTables, re-sort by Hilbert curve
};

/// Reason that triggered a particular merge cycle.
enum class MergeTrigger : uint8_t {
    NODE_THRESHOLD    = 0,  ///< MemTable > MEMTABLE_MERGE_TRIGGER_NODES
    TIME_ELAPSED      = 1,  ///< Time since last merge > MEMTABLE_MERGE_TRIGGER_MS
    NAP_STATE         = 2,  ///< ATP < REBUILD_ATP_NAP_THRESHOLD → full rebuild safe
    FRAGMENTATION     = 3   ///< Fragmentation index > REBUILD_FRAGMENTATION_THRESHOLD
};

// ─── Policy predicates ───────────────────────────────────────────────────────

/// True when an incremental merge should be triggered.
[[nodiscard]] constexpr bool should_merge_incremental(
    uint32_t memtable_nodes, uint32_t ms_since_last_merge) noexcept
{
    return memtable_nodes >= MEMTABLE_MERGE_TRIGGER_NODES
        || ms_since_last_merge >= MEMTABLE_MERGE_TRIGGER_MS;
}

/// True when a full rebuild is warranted (nap or fragmentation).
[[nodiscard]] constexpr bool should_rebuild_full(
    double atp_reserve, double fragmentation_index) noexcept
{
    return atp_reserve < REBUILD_ATP_NAP_THRESHOLD
        || fragmentation_index > REBUILD_FRAGMENTATION_THRESHOLD;
}

/// True when the visibility lag is within spec.
[[nodiscard]] constexpr bool visibility_lag_acceptable(uint32_t lag_ms) noexcept {
    return lag_ms <= VISIBILITY_LAG_MAX_MS;
}

/// Determine the merge type implied by the given trigger.
[[nodiscard]] constexpr MergeType merge_type_for(MergeTrigger t) noexcept {
    switch (t) {
        case MergeTrigger::NAP_STATE:
        case MergeTrigger::FRAGMENTATION:
            return MergeType::FULL_REBUILD;
        default:
            return MergeType::INCREMENTAL;
    }
}

// ─── Label helpers ───────────────────────────────────────────────────────────

[[nodiscard]] constexpr std::string_view merge_type_label(MergeType m) noexcept {
    switch (m) {
        case MergeType::INCREMENTAL:  return "incremental";
        case MergeType::FULL_REBUILD: return "full_rebuild";
        default:                      return "unknown";
    }
}

[[nodiscard]] constexpr std::string_view merge_trigger_label(MergeTrigger t) noexcept {
    switch (t) {
        case MergeTrigger::NODE_THRESHOLD: return "node_threshold";
        case MergeTrigger::TIME_ELAPSED:   return "time_elapsed";
        case MergeTrigger::NAP_STATE:      return "nap_state";
        case MergeTrigger::FRAGMENTATION:  return "fragmentation";
        default:                           return "unknown";
    }
}

} // namespace nikola::memory
