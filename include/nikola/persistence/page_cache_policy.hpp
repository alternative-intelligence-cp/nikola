// ============================================================
// include/nikola/persistence/page_cache_policy.hpp
// Phase 72 — GAP-027a  LMDB Memory-Mapped I/O Page Cache Management
// ============================================================
// Models the context-aware madvise() policy dispatch for the
// LMDB-backed 9D-TWI persistence layer.
//
// Three madvise modes:
//   SEQUENTIAL  — Hilbert scans & GGUF export
//   RANDOM      — Neurogenesis / sparse active-wake updates
//   WILLNEED    — Predictive prefetch of future trajectories
//
// Two storage profiles:
//   SSD_NVME    — Aggressive prefetch, MDB_NORDAHEAD, async commits
//   HDD         — Maximise sequentiality, full-copy compaction on Nap
//
// Page eviction protection:
//   mlock() pinning of active wavefront + MADV_HUGEPAGE for hot arrays
// ============================================================
#pragma once

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace nikola::persistence {

// ────────────────────────────────────────────────────────────────────────────
// §1  Performance characteristics
// ────────────────────────────────────────────────────────────────────────────

/// Maximum I/O stall reduction factor achieved by switching to SEQUENTIAL
/// mode during Mamba-9D / GGUF-export linear scans (spec: "up to 100×").
constexpr uint32_t SEQUENTIAL_SCAN_MAX_IO_IMPROVEMENT_FACTOR = 100u;

// ────────────────────────────────────────────────────────────────────────────
// §2  Page-alignment constants
// ────────────────────────────────────────────────────────────────────────────

/// Standard x86-64 / ARM64 base page size (4 KiB).
constexpr uint64_t PAGE_SIZE_BYTES = 4'096u;

/// Huge-page size for wavefunction arrays (2 MiB transparent huge pages).
constexpr uint64_t HUGE_PAGE_SIZE_BYTES = 2u * 1024u * 1024u;

/// Mask to align an offset down to the nearest page boundary.
constexpr uint64_t PAGE_ALIGN_MASK = ~(PAGE_SIZE_BYTES - 1u);

// ────────────────────────────────────────────────────────────────────────────
// §3  Enumerations
// ────────────────────────────────────────────────────────────────────────────

/// Operational state of the cognitive engine — determines madvise policy.
enum class SystemState : uint8_t {
    DREAM_WEAVE,    ///< Mamba-9D Hilbert scan (sleep consolidation)
    GGUF_EXPORT,    ///< Sequential DB read for GGUF snapshot
    ACTIVE_WAKE,    ///< Neurogenesis / sparse random updates (learning)
    NAP_COMPACT,    ///< Background LSM compaction
    IDLE,           ///< No active workload
};

/// The madvise hint communicated to the OS kernel.
enum class MadviseHint : uint8_t {
    SEQUENTIAL,     ///< MADV_SEQUENTIAL — aggressive prefetch, drop after use
    RANDOM,         ///< MADV_RANDOM     — disable read-ahead
    WILLNEED,       ///< MADV_WILLNEED   — async prefault specific pages
    HUGEPAGE,       ///< MADV_HUGEPAGE   — promote to transparent huge pages
    NORMAL,         ///< MADV_NORMAL     — restore default OS behaviour
};

/// Storage medium profile affecting prefetch aggressiveness.
enum class StorageProfile : uint8_t {
    SSD_NVME,   ///< Low random-access latency; aggressive prefetch enabled
    HDD,        ///< High seek penalty; maximise sequentiality at all costs
};

/// LMDB commit durability mode.
enum class CommitPolicy : uint8_t {
    SYNC,           ///< MDB_SYNC  — fsync after every commit (safest)
    NOSYNC,         ///< MDB_NOSYNC — async commit (safe on SSD with internal buffer)
    NOMETASYNC,     ///< MDB_NOMETASYNC — async metadata only
};

// ────────────────────────────────────────────────────────────────────────────
// §4  Core policy dispatch
// ────────────────────────────────────────────────────────────────────────────

/// Return the primary madvise hint to apply to the mapped DB region for the
/// given system state.  Drives the hot path of optimize_page_cache().
[[nodiscard]] constexpr MadviseHint primary_madvise_hint(SystemState state) noexcept {
    switch (state) {
        case SystemState::DREAM_WEAVE: return MadviseHint::SEQUENTIAL;
        case SystemState::GGUF_EXPORT: return MadviseHint::SEQUENTIAL;
        case SystemState::ACTIVE_WAKE: return MadviseHint::RANDOM;
        case SystemState::NAP_COMPACT: return MadviseHint::SEQUENTIAL;
        case SystemState::IDLE:        return MadviseHint::NORMAL;
    }
    return MadviseHint::NORMAL; // unreachable
}

/// Return true when the given state triggers sequential scan mode.
/// Sequential scan mode also enables MADV_HUGEPAGE on the full region.
[[nodiscard]] constexpr bool is_sequential_scan_mode(SystemState state) noexcept {
    return primary_madvise_hint(state) == MadviseHint::SEQUENTIAL;
}

/// Return true when the given state triggers random access mode (disables read-ahead).
[[nodiscard]] constexpr bool is_random_access_mode(SystemState state) noexcept {
    return primary_madvise_hint(state) == MadviseHint::RANDOM;
}

/// Return true when the state benefits from MADV_HUGEPAGE promotion.
/// Applies to all sequential-scan states to minimise TLB misses.
[[nodiscard]] constexpr bool should_apply_hugepage(SystemState state) noexcept {
    return is_sequential_scan_mode(state);
}

/// Return true when the state uses predictive MADV_WILLNEED prefetch.
/// DREAM_WEAVE runs the Mamba-9D attention model which yields predicted regions.
[[nodiscard]] constexpr bool supports_predictive_prefetch(SystemState state) noexcept {
    return state == SystemState::DREAM_WEAVE;
}

// ────────────────────────────────────────────────────────────────────────────
// §5  Storage profile policy
// ────────────────────────────────────────────────────────────────────────────

/// SSD/NVMe allows manual prefetch management (MDB_NORDAHEAD flag).
[[nodiscard]] constexpr bool use_nordahead(StorageProfile profile) noexcept {
    return profile == StorageProfile::SSD_NVME;
}

/// Recommended commit policy for a given storage profile.
[[nodiscard]] constexpr CommitPolicy recommended_commit_policy(StorageProfile profile) noexcept {
    switch (profile) {
        case StorageProfile::SSD_NVME: return CommitPolicy::NOSYNC;  // reliable internal buffer
        case StorageProfile::HDD:      return CommitPolicy::SYNC;    // mechanical reliability required
    }
    return CommitPolicy::SYNC; // unreachable
}

/// HDD profile forces SEQUENTIAL globally to exploit drive controller's prefetch.
[[nodiscard]] constexpr bool force_sequential_on_hdd(StorageProfile profile) noexcept {
    return profile == StorageProfile::HDD;
}

/// HDD requires full-copy compaction during Nap to restore perfect sequentiality.
[[nodiscard]] constexpr bool requires_full_copy_compact(StorageProfile profile, SystemState state) noexcept {
    return profile == StorageProfile::HDD && state == SystemState::NAP_COMPACT;
}

/// Effective hint considering overrides from storage profile.
/// HDD forces SEQUENTIAL regardless of state; otherwise use primary_madvise_hint.
[[nodiscard]] constexpr MadviseHint effective_madvise_hint(
        SystemState state, StorageProfile profile) noexcept {
    if (force_sequential_on_hdd(profile)) return MadviseHint::SEQUENTIAL;
    return primary_madvise_hint(state);
}

// ────────────────────────────────────────────────────────────────────────────
// §6  Page alignment utilities
// ────────────────────────────────────────────────────────────────────────────

/// Align an offset down to the nearest page boundary.
[[nodiscard]] constexpr uint64_t align_to_page(uint64_t offset) noexcept {
    return offset & PAGE_ALIGN_MASK;
}

/// True when an offset is already page-aligned.
[[nodiscard]] constexpr bool is_page_aligned(uint64_t offset) noexcept {
    return (offset & (PAGE_SIZE_BYTES - 1u)) == 0u;
}

/// Compute the byte offset into the mapped region for a given Hilbert index.
/// Each node occupies node_size_bytes; result is NOT yet page-aligned.
[[nodiscard]] constexpr uint64_t hilbert_index_to_offset(
        uint64_t hilbert_index, uint64_t node_size_bytes) noexcept {
    return hilbert_index * node_size_bytes;
}

/// Compute the page-aligned mmap offset for a Hilbert index.
[[nodiscard]] constexpr uint64_t hilbert_index_to_page_offset(
        uint64_t hilbert_index, uint64_t node_size_bytes) noexcept {
    return align_to_page(hilbert_index_to_offset(hilbert_index, node_size_bytes));
}

/// Number of 4 KiB pages that span a contiguous range of Hilbert nodes.
/// Accounts for the page-alignment rounding at both ends.
[[nodiscard]] constexpr uint64_t pages_spanning_range(
        uint64_t first_hilbert_index,
        uint64_t node_count,
        uint64_t node_size_bytes) noexcept {
    if (node_count == 0u || node_size_bytes == 0u) return 0u;
    const uint64_t start_page = align_to_page(first_hilbert_index * node_size_bytes);
    const uint64_t end_byte   = (first_hilbert_index + node_count) * node_size_bytes;
    const uint64_t end_page   = align_to_page(end_byte + PAGE_SIZE_BYTES - 1u);
    return (end_page - start_page) / PAGE_SIZE_BYTES;
}

// ────────────────────────────────────────────────────────────────────────────
// §7  Memory-lock / huge-page eligibility
// ────────────────────────────────────────────────────────────────────────────

/// True when the region size qualifies for huge-page promotion.
/// Minimum region: one full huge page (2 MiB).
[[nodiscard]] constexpr bool is_huge_page_eligible(uint64_t region_size_bytes) noexcept {
    return region_size_bytes >= HUGE_PAGE_SIZE_BYTES;
}

/// True when a region is worth pinning with mlock() (non-trivial size).
/// Threshold: at least one standard page.
[[nodiscard]] constexpr bool is_mlock_eligible(uint64_t region_size_bytes) noexcept {
    return region_size_bytes >= PAGE_SIZE_BYTES;
}

/// In ACTIVE_WAKE, the hot wavefront should be mlock()-pinned when permitted.
[[nodiscard]] constexpr bool should_pin_wavefront(SystemState state) noexcept {
    return state == SystemState::ACTIVE_WAKE;
}

// ────────────────────────────────────────────────────────────────────────────
// §8  Access pattern classification
// ────────────────────────────────────────────────────────────────────────────

/// Return true when the access pattern is strictly sequential (Hilbert order).
[[nodiscard]] constexpr bool is_hilbert_sequential_access(SystemState state) noexcept {
    return state == SystemState::DREAM_WEAVE ||
           state == SystemState::GGUF_EXPORT ||
           state == SystemState::NAP_COMPACT;
}

/// Return true when the access pattern is sparse / non-sequential.
[[nodiscard]] constexpr bool is_sparse_random_access(SystemState state) noexcept {
    return state == SystemState::ACTIVE_WAKE;
}

/// Scan pollution risk: a large sequential scan can evict hot physics pages.
/// Only true when GGUF_EXPORT runs concurrently with potential physics access.
[[nodiscard]] constexpr bool has_scan_pollution_risk(SystemState state) noexcept {
    return state == SystemState::GGUF_EXPORT;
}

// ────────────────────────────────────────────────────────────────────────────
// §9  LMDB flag recommendations
// ────────────────────────────────────────────────────────────────────────────

/// MDB_NORDAHEAD disables OS read-ahead, letting us manage prefetch manually.
/// Recommended on SSD/NVMe only; HDD benefits from kernel read-ahead.
[[nodiscard]] constexpr bool recommend_mdb_nordahead(StorageProfile profile) noexcept {
    return profile == StorageProfile::SSD_NVME;
}

/// MDB_NOSYNC enables async commits — safe on SSD with internal write buffer.
[[nodiscard]] constexpr bool recommend_mdb_nosync(StorageProfile profile) noexcept {
    return profile == StorageProfile::SSD_NVME;
}

// ────────────────────────────────────────────────────────────────────────────
// §10  Prefetch trajectory helpers
// ────────────────────────────────────────────────────────────────────────────

/// WILLNEED hint is applied per predicted Hilbert index.
/// True when the madvise WILLNEED call is warranted for this index.
[[nodiscard]] constexpr bool should_prefetch_hilbert_index(
        uint64_t predicted_hilbert_index,
        uint64_t db_node_count) noexcept {
    return predicted_hilbert_index < db_node_count;
}

/// Number of 4 KiB WILLNEED page hints required for a trajectory of n distinct Hilbert indices.
/// Conservatively: one hint per index (some may alias to the same page — handled by kernel).
[[nodiscard]] constexpr uint64_t prefetch_hint_count(uint64_t trajectory_size) noexcept {
    return trajectory_size;
}

// ────────────────────────────────────────────────────────────────────────────
// §11  Diagnostic names
// ────────────────────────────────────────────────────────────────────────────

[[nodiscard]] constexpr std::string_view system_state_name(SystemState s) noexcept {
    switch (s) {
        case SystemState::DREAM_WEAVE: return "DREAM_WEAVE";
        case SystemState::GGUF_EXPORT: return "GGUF_EXPORT";
        case SystemState::ACTIVE_WAKE: return "ACTIVE_WAKE";
        case SystemState::NAP_COMPACT: return "NAP_COMPACT";
        case SystemState::IDLE:        return "IDLE";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view madvise_hint_name(MadviseHint h) noexcept {
    switch (h) {
        case MadviseHint::SEQUENTIAL: return "SEQUENTIAL";
        case MadviseHint::RANDOM:     return "RANDOM";
        case MadviseHint::WILLNEED:   return "WILLNEED";
        case MadviseHint::HUGEPAGE:   return "HUGEPAGE";
        case MadviseHint::NORMAL:     return "NORMAL";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view storage_profile_name(StorageProfile p) noexcept {
    switch (p) {
        case StorageProfile::SSD_NVME: return "SSD_NVME";
        case StorageProfile::HDD:      return "HDD";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view commit_policy_name(CommitPolicy c) noexcept {
    switch (c) {
        case CommitPolicy::SYNC:        return "SYNC";
        case CommitPolicy::NOSYNC:      return "NOSYNC";
        case CommitPolicy::NOMETASYNC:  return "NOMETASYNC";
    }
    return "unknown";
}

} // namespace nikola::persistence
