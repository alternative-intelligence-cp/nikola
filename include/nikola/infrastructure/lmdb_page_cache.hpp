#pragma once
// nikola/infrastructure/lmdb_page_cache.hpp
//
// GAP-027: LMDB Memory-Mapped I/O Page Cache Management
// Source: Gemini Deep Research Round 2, Batch 37-40
// Spec: docs/info/integration/sections/04_infrastructure/06_database_persistence.md §GAP-027
//
// Context-aware madvise policy selection for LMDB mmap regions. Different
// Nikola subsystems access the TorusGridSoA with incompatible access patterns
// (sequential Hilbert scan, sparse random neurogenesis, predictive prefetch).
// Passively relying on OS LRU eviction is suboptimal; linear scans evict hot
// physics nodes causing stall-inducing page faults. This header encodes the
// policy rules, storage-medium profiles, and page-eviction priorities that
// drive an active cognitive memory management subsystem.
//
// Header-only, no external dependencies (no lmdb.h required).

#include <cstdint>
#include <cstddef>

namespace nikola::infrastructure {

// ─── System State ────────────────────────────────────────────────────────────

/// Active operational mode driving page-cache policy selection.
/// Different states produce radically different LMDB access patterns.
enum class PageCacheSystemState : uint8_t {
    ACTIVE_WAKE   = 0,  ///< Online learning; sparse neurogenesis; random updates
    DREAM_WEAVE   = 1,  ///< Offline replay; sequential Hilbert scan of full grid
    GGUF_EXPORT   = 2,  ///< Full sequential scan for checkpoint / GGUF snapshot
    NAP_COMPACT   = 3,  ///< Nap-cycle compaction; sequential read-write pass
    IDLE          = 4,  ///< No active access; no policy change required
};

// ─── Storage Medium ───────────────────────────────────────────────────────────

/// Physical storage medium hosting the LMDB database file.
/// Dictates aggressiveness of prefetching and compaction strategy.
enum class StorageMedium : uint8_t {
    SSD_NVME  = 0,  ///< Low random-access latency; aggressive prefetch safe
    HDD       = 1,  ///< High seek penalty; maximize sequentiality, avoid random
};

// ─── madvise Policy ───────────────────────────────────────────────────────────

/// madvise(2) hint to apply to mapped LMDB region(s).
/// Mirrors the POSIX MADV_* constants at the spec level.
enum class MadvisePolicy : uint8_t {
    SEQUENTIAL  = 0,  ///< MADV_SEQUENTIAL: aggressive prefetch + quick discard
    RANDOM      = 1,  ///< MADV_RANDOM: disable read-ahead, save I/O bandwidth
    WILLNEED    = 2,  ///< MADV_WILLNEED: async fault specified range into RAM
    HUGEPAGE    = 3,  ///< MADV_HUGEPAGE: promote region to Huge Pages (2 MB TLB)
    NORMAL      = 4,  ///< MADV_NORMAL: restore default OS behaviour
};

// ─── LMDB Open-Flag Profiles ──────────────────────────────────────────────────

/// LMDB open-flag bitmask values relevant to page-cache management.
/// These are the bit positions from lmdb.h; encoded here so the policy layer
/// can be reasoned about without linking lmdb.
enum class LmdbFlag : uint32_t {
    NORDAHEAD = 0x00800000u,  ///< Disable OS read-ahead (we manage via WILLNEED)
    NOSYNC    = 0x00010000u,  ///< Async commits; skip fsync per write (WAL only)
    WRITEMAP  = 0x00080000u,  ///< Direct mmap writes (required for madvise control)
    MAPASYNC  = 0x00100000u,  ///< Async flush in WRITEMAP mode
};

// ─── Commit Policy ────────────────────────────────────────────────────────────

/// Commit durability policy driving fsync behaviour.
enum class CommitPolicy : uint8_t {
    SYNCHRONOUS   = 0,  ///< fsync on every commit — fully durable, higher latency
    ASYNCHRONOUS  = 1,  ///< Omit fsync (MDB_NOSYNC) — relies on SSD write buffer
    WAL_ASYNC     = 2,  ///< Async for WAL checkpoints; sync for epoch boundaries
};

// ─── Eviction Priority ────────────────────────────────────────────────────────

/// Memory-eviction priority level for a LMDB page region.
enum class EvictionPriority : uint8_t {
    PINNED     = 0,  ///< mlock()ed; kernel never evicts (highest protection)
    HOT        = 1,  ///< MADV_HUGEPAGE; reduced TLB pressure, rarely evicted
    WARM       = 2,  ///< Default; evicted under normal memory pressure
    COLD       = 3,  ///< Metadata / infrequently accessed; evict first
};

// ─── Performance Characteristics ──────────────────────────────────────────────

/// Maximum I/O stall reduction factor achievable by policy switching during
/// sequential scans (vs. default OS LRU with scan pollution).
inline constexpr int IO_STALL_REDUCTION_MAX_X     = 100;

/// NODE_SIZE_BYTES used when mapping a Hilbert index to a page offset.
/// Matches TorusGridSoA node payload size (see GAP-021).
inline constexpr std::size_t NODE_SIZE_BYTES       = 232UL;

/// Linux default page size (4 KiB); page-align all MADV_WILLNEED calls to this.
inline constexpr std::size_t PAGE_SIZE_BYTES        = 4096UL;

// ─── Policy Selection ─────────────────────────────────────────────────────────

/// Primary madvise policy for the LMDB mmap region given the current system
/// state. Applied to the full mapped region each time state transitions.
///
/// DREAM_WEAVE / GGUF_EXPORT / NAP_COMPACT → SEQUENTIAL
///   Strictly sequential Hilbert traversal; prefetch aggressively and discard
///   pages after use to prevent scan pollution of physics hot-set.
///
/// ACTIVE_WAKE → RANDOM
///   Sparse neurogenesis at high-energy coordinates; disabling read-ahead
///   avoids fetching neighbours that will not be visited.
///
/// IDLE → NORMAL
///   Restore default OS policy; no hint overhead.
[[nodiscard]] constexpr MadvisePolicy primary_page_policy(
    PageCacheSystemState state) noexcept
{
    switch (state) {
        case PageCacheSystemState::DREAM_WEAVE:  return MadvisePolicy::SEQUENTIAL;
        case PageCacheSystemState::GGUF_EXPORT:  return MadvisePolicy::SEQUENTIAL;
        case PageCacheSystemState::NAP_COMPACT:  return MadvisePolicy::SEQUENTIAL;
        case PageCacheSystemState::ACTIVE_WAKE:  return MadvisePolicy::RANDOM;
        case PageCacheSystemState::IDLE:         return MadvisePolicy::NORMAL;
    }
    return MadvisePolicy::NORMAL;
}

/// Secondary / companion policy applied alongside the primary, or NORMAL if
/// no companion is needed.
///
/// Sequential states → HUGEPAGE companion: promotes pages into 2 MB TLB
/// entries so long linear scans incur fewer TLB misses.
/// ACTIVE_WAKE / IDLE → NORMAL (no companion).
[[nodiscard]] constexpr MadvisePolicy companion_page_policy(
    PageCacheSystemState state) noexcept
{
    switch (state) {
        case PageCacheSystemState::DREAM_WEAVE: return MadvisePolicy::HUGEPAGE;
        case PageCacheSystemState::GGUF_EXPORT: return MadvisePolicy::HUGEPAGE;
        case PageCacheSystemState::NAP_COMPACT: return MadvisePolicy::HUGEPAGE;
        default:                                return MadvisePolicy::NORMAL;
    }
}

/// True when the current state produces strictly sequential Hilbert-order
/// access — the condition under which SEQUENTIAL + HUGEPAGE prevent scan
/// pollution.
[[nodiscard]] constexpr bool is_sequential_access_state(
    PageCacheSystemState state) noexcept
{
    return state == PageCacheSystemState::DREAM_WEAVE ||
           state == PageCacheSystemState::GGUF_EXPORT ||
           state == PageCacheSystemState::NAP_COMPACT;
}

/// True when the current state produces sparse random access that should
/// suppress read-ahead entirely.
[[nodiscard]] constexpr bool is_random_access_state(
    PageCacheSystemState state) noexcept
{
    return state == PageCacheSystemState::ACTIVE_WAKE;
}

/// True when the predictive WILLNEED prefetch path is enabled — only during
/// ACTIVE_WAKE when Mamba-9D attention surfaces a predicted coordinate set.
[[nodiscard]] constexpr bool willneed_prefetch_enabled(
    PageCacheSystemState state) noexcept
{
    return state == PageCacheSystemState::ACTIVE_WAKE;
}

// ─── Storage Medium Profiles ──────────────────────────────────────────────────

/// True when the medium supports aggressive multi-threaded WILLNEED prefetch
/// without incurring seek penalty.
[[nodiscard]] constexpr bool supports_aggressive_prefetch(
    StorageMedium medium) noexcept
{
    return medium == StorageMedium::SSD_NVME;
}

/// LMDB NORDAHEAD flag should be set on SSD/NVMe so that the kernel's own
/// read-ahead is suppressed in favour of our explicit WILLNEED calls.
[[nodiscard]] constexpr bool should_set_nordahead(
    StorageMedium medium) noexcept
{
    return medium == StorageMedium::SSD_NVME;
}

/// Async commit (MDB_NOSYNC) is acceptable on SSD where the internal write
/// buffer provides adequate durability for non-critical WAL checkpoints.
[[nodiscard]] constexpr CommitPolicy wal_commit_policy(
    StorageMedium medium) noexcept
{
    return (medium == StorageMedium::SSD_NVME)
               ? CommitPolicy::ASYNCHRONOUS
               : CommitPolicy::SYNCHRONOUS;
}

/// HDD profile requires a full-copy compact during the Nap cycle to restore
/// perfect file sequentiality (Hilbert scans then map to physical rotations
/// without seek jitter). SSD can tolerate fragmentation.
[[nodiscard]] constexpr bool requires_full_copy_compact(
    StorageMedium medium) noexcept
{
    return medium == StorageMedium::HDD;
}

/// On HDD, MADV_RANDOM must be disabled globally; force SEQUENTIAL to engage
/// the drive controller's read-ahead cache.
[[nodiscard]] constexpr bool force_sequential_on_hdd(
    StorageMedium medium) noexcept
{
    return medium == StorageMedium::HDD;
}

/// Effective primary policy accounting for HDD sequential override.
/// On HDD, all access states coerce to SEQUENTIAL to avoid head seeks.
[[nodiscard]] constexpr MadvisePolicy effective_policy(
    PageCacheSystemState state, StorageMedium medium) noexcept
{
    if (force_sequential_on_hdd(medium)) {
        return MadvisePolicy::SEQUENTIAL;
    }
    return primary_page_policy(state);
}

// ─── Page Eviction and Pinning ────────────────────────────────────────────────

/// Hot wavefunction amplitude arrays should be promoted to Huge Pages to
/// minimise TLB misses during the 1 kHz physics tick. Only worthwhile when
/// the region is large enough for 2 MB pages to matter.
///
/// @param region_bytes  Size of the region to evaluate.
/// @returns true when the region is ≥ 2 MB (one Huge Page).
[[nodiscard]] constexpr bool should_use_hugepage(
    std::size_t region_bytes) noexcept
{
    constexpr std::size_t HUGE_PAGE_BYTES = 2UL * 1024UL * 1024UL;
    return region_bytes >= HUGE_PAGE_BYTES;
}

/// The Active Wavefront — the set of torus-grid nodes currently being updated
/// by the physics engine — should be pinned via mlock() to prevent swapping.
/// A violation of the 1ms tick budget occurs if even one page-fault stalls
/// the symplectic integrator.
///
/// @param region_bytes  Size of the active wavefront region.
/// @param memlock_available  True when the process has CAP_IPC_LOCK or
///                           sufficient RLIMIT_MEMLOCK headroom.
/// @returns true when pinning is both recommended and feasible.
[[nodiscard]] constexpr bool should_pin_region(
    std::size_t region_bytes,
    bool        memlock_available) noexcept
{
    return memlock_available && region_bytes > 0;
}

/// Eviction priority tier for a named region category.
enum class RegionCategory : uint8_t {
    ACTIVE_WAVEFRONT   = 0,  ///< Current wavefunction amplitudes — never swap
    HOT_WAVEFUNCTION   = 1,  ///< Wavefunction arrays (phase, amplitude SoA)
    WARM_TOPOLOGY      = 2,  ///< Adjacency / Morton-key index tables
    COLD_METADATA      = 3,  ///< Header pages, stats, free-list cursor
};

[[nodiscard]] constexpr EvictionPriority eviction_priority(
    RegionCategory category) noexcept
{
    switch (category) {
        case RegionCategory::ACTIVE_WAVEFRONT: return EvictionPriority::PINNED;
        case RegionCategory::HOT_WAVEFUNCTION: return EvictionPriority::HOT;
        case RegionCategory::WARM_TOPOLOGY:    return EvictionPriority::WARM;
        case RegionCategory::COLD_METADATA:    return EvictionPriority::COLD;
    }
    return EvictionPriority::COLD;
}

// ─── Hilbert Prefetch Geometry ────────────────────────────────────────────────

/// Byte offset of a Hilbert-indexed node within the LMDB mmap region.
[[nodiscard]] constexpr std::size_t hilbert_node_offset(
    uint64_t hilbert_index) noexcept
{
    return hilbert_index * NODE_SIZE_BYTES;
}

/// Page-aligned offset for a given byte offset (align down to PAGE_SIZE_BYTES).
[[nodiscard]] constexpr std::size_t page_align_down(
    std::size_t byte_offset) noexcept
{
    return byte_offset & ~(PAGE_SIZE_BYTES - 1UL);
}

/// Number of full pages spanned by a node starting at a given offset.
/// Always at least 1; may be 2 if the node straddles a page boundary.
[[nodiscard]] constexpr std::size_t pages_for_node(
    std::size_t byte_offset) noexcept
{
    const std::size_t aligned = page_align_down(byte_offset);
    const std::size_t end     = byte_offset + NODE_SIZE_BYTES;
    return (end - aligned + PAGE_SIZE_BYTES - 1UL) / PAGE_SIZE_BYTES;
}

// ─── Label Functions ──────────────────────────────────────────────────────────

[[nodiscard]] constexpr const char* page_cache_state_name(
    PageCacheSystemState state) noexcept
{
    switch (state) {
        case PageCacheSystemState::ACTIVE_WAKE:  return "ACTIVE_WAKE";
        case PageCacheSystemState::DREAM_WEAVE:  return "DREAM_WEAVE";
        case PageCacheSystemState::GGUF_EXPORT:  return "GGUF_EXPORT";
        case PageCacheSystemState::NAP_COMPACT:  return "NAP_COMPACT";
        case PageCacheSystemState::IDLE:         return "IDLE";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr const char* storage_medium_name(
    StorageMedium medium) noexcept
{
    switch (medium) {
        case StorageMedium::SSD_NVME: return "SSD_NVME";
        case StorageMedium::HDD:      return "HDD";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr const char* madvise_policy_name(
    MadvisePolicy policy) noexcept
{
    switch (policy) {
        case MadvisePolicy::SEQUENTIAL: return "SEQUENTIAL";
        case MadvisePolicy::RANDOM:     return "RANDOM";
        case MadvisePolicy::WILLNEED:   return "WILLNEED";
        case MadvisePolicy::HUGEPAGE:   return "HUGEPAGE";
        case MadvisePolicy::NORMAL:     return "NORMAL";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr const char* commit_policy_name(
    CommitPolicy policy) noexcept
{
    switch (policy) {
        case CommitPolicy::SYNCHRONOUS:  return "SYNCHRONOUS";
        case CommitPolicy::ASYNCHRONOUS: return "ASYNCHRONOUS";
        case CommitPolicy::WAL_ASYNC:    return "WAL_ASYNC";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr const char* eviction_priority_name(
    EvictionPriority priority) noexcept
{
    switch (priority) {
        case EvictionPriority::PINNED: return "PINNED";
        case EvictionPriority::HOT:    return "HOT";
        case EvictionPriority::WARM:   return "WARM";
        case EvictionPriority::COLD:   return "COLD";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr const char* region_category_name(
    RegionCategory category) noexcept
{
    switch (category) {
        case RegionCategory::ACTIVE_WAVEFRONT: return "ACTIVE_WAVEFRONT";
        case RegionCategory::HOT_WAVEFUNCTION: return "HOT_WAVEFUNCTION";
        case RegionCategory::WARM_TOPOLOGY:    return "WARM_TOPOLOGY";
        case RegionCategory::COLD_METADATA:    return "COLD_METADATA";
    }
    return "UNKNOWN";
}

} // namespace nikola::infrastructure
