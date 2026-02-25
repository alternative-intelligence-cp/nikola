// ============================================================
// tests/unit/phase72_page_cache_policy_test.cpp
// Phase 72 — GAP-027a  LMDB Memory-Mapped I/O Page Cache Management
// ============================================================
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstdint>
#include <array>

#include "nikola/persistence/page_cache_policy.hpp"

using namespace nikola::persistence;

// ────────────────────────────────────────────────────────────────────────────
// §1  Performance characteristic constants
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("SEQUENTIAL_SCAN_MAX_IO_IMPROVEMENT_FACTOR is 100", "[constants]") {
    CHECK(SEQUENTIAL_SCAN_MAX_IO_IMPROVEMENT_FACTOR == 100u);
}

// ────────────────────────────────────────────────────────────────────────────
// §2  Page-alignment constants
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("PAGE_SIZE_BYTES is 4096 (4 KiB)", "[constants]") {
    CHECK(PAGE_SIZE_BYTES == 4'096u);
}

TEST_CASE("HUGE_PAGE_SIZE_BYTES is 2 MiB", "[constants]") {
    CHECK(HUGE_PAGE_SIZE_BYTES == 2u * 1024u * 1024u);
    CHECK(HUGE_PAGE_SIZE_BYTES == 2'097'152u);
}

TEST_CASE("HUGE_PAGE_SIZE_BYTES is exactly 512 standard pages", "[constants]") {
    CHECK(HUGE_PAGE_SIZE_BYTES / PAGE_SIZE_BYTES == 512u);
}

TEST_CASE("PAGE_ALIGN_MASK masks off lower 12 bits", "[constants]") {
    // 4096 = 0x1000; mask = ~0xFFF (64-bit) = ...FFFFF000
    CHECK((PAGE_ALIGN_MASK & 0xFFFu) == 0u);
    CHECK((PAGE_ALIGN_MASK & ~static_cast<uint64_t>(0xFFF)) == PAGE_ALIGN_MASK);
}

// ────────────────────────────────────────────────────────────────────────────
// §3  primary_madvise_hint: sequential scan states
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("primary_madvise_hint: DREAM_WEAVE → SEQUENTIAL", "[policy]") {
    CHECK(primary_madvise_hint(SystemState::DREAM_WEAVE) == MadviseHint::SEQUENTIAL);
}

TEST_CASE("primary_madvise_hint: GGUF_EXPORT → SEQUENTIAL", "[policy]") {
    CHECK(primary_madvise_hint(SystemState::GGUF_EXPORT) == MadviseHint::SEQUENTIAL);
}

TEST_CASE("primary_madvise_hint: NAP_COMPACT → SEQUENTIAL", "[policy]") {
    CHECK(primary_madvise_hint(SystemState::NAP_COMPACT) == MadviseHint::SEQUENTIAL);
}

// ────────────────────────────────────────────────────────────────────────────
// §4  primary_madvise_hint: random / idle states
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("primary_madvise_hint: ACTIVE_WAKE → RANDOM (disables read-ahead)", "[policy]") {
    CHECK(primary_madvise_hint(SystemState::ACTIVE_WAKE) == MadviseHint::RANDOM);
}

TEST_CASE("primary_madvise_hint: IDLE → NORMAL (restore OS default)", "[policy]") {
    CHECK(primary_madvise_hint(SystemState::IDLE) == MadviseHint::NORMAL);
}

// ────────────────────────────────────────────────────────────────────────────
// §5  is_sequential_scan_mode / is_random_access_mode
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("is_sequential_scan_mode: true for DREAM_WEAVE, GGUF_EXPORT, NAP_COMPACT", "[policy]") {
    CHECK(is_sequential_scan_mode(SystemState::DREAM_WEAVE) == true);
    CHECK(is_sequential_scan_mode(SystemState::GGUF_EXPORT) == true);
    CHECK(is_sequential_scan_mode(SystemState::NAP_COMPACT) == true);
}

TEST_CASE("is_sequential_scan_mode: false for ACTIVE_WAKE and IDLE", "[policy]") {
    CHECK(is_sequential_scan_mode(SystemState::ACTIVE_WAKE) == false);
    CHECK(is_sequential_scan_mode(SystemState::IDLE)        == false);
}

TEST_CASE("is_random_access_mode: only ACTIVE_WAKE is random", "[policy]") {
    CHECK(is_random_access_mode(SystemState::ACTIVE_WAKE)  == true);
    CHECK(is_random_access_mode(SystemState::DREAM_WEAVE)  == false);
    CHECK(is_random_access_mode(SystemState::GGUF_EXPORT)  == false);
    CHECK(is_random_access_mode(SystemState::NAP_COMPACT)  == false);
    CHECK(is_random_access_mode(SystemState::IDLE)         == false);
}

TEST_CASE("sequential and random modes are mutually exclusive for all states", "[policy]") {
    for (auto s : {SystemState::DREAM_WEAVE, SystemState::GGUF_EXPORT,
                   SystemState::ACTIVE_WAKE, SystemState::NAP_COMPACT,
                   SystemState::IDLE}) {
        const bool seq  = is_sequential_scan_mode(s);
        const bool rand = is_random_access_mode(s);
        const bool both = seq && rand;
        CHECK_FALSE(both); // cannot both be true
    }
}

// ────────────────────────────────────────────────────────────────────────────
// §6  should_apply_hugepage
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("should_apply_hugepage: only sequential-scan states get huge pages", "[policy]") {
    CHECK(should_apply_hugepage(SystemState::DREAM_WEAVE) == true);
    CHECK(should_apply_hugepage(SystemState::GGUF_EXPORT) == true);
    CHECK(should_apply_hugepage(SystemState::NAP_COMPACT) == true);
    CHECK(should_apply_hugepage(SystemState::ACTIVE_WAKE) == false);
    CHECK(should_apply_hugepage(SystemState::IDLE)        == false);
}

// ────────────────────────────────────────────────────────────────────────────
// §7  supports_predictive_prefetch
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("supports_predictive_prefetch: only DREAM_WEAVE runs Mamba-9D attention", "[policy]") {
    CHECK(supports_predictive_prefetch(SystemState::DREAM_WEAVE) == true);
    CHECK(supports_predictive_prefetch(SystemState::GGUF_EXPORT) == false);
    CHECK(supports_predictive_prefetch(SystemState::ACTIVE_WAKE) == false);
    CHECK(supports_predictive_prefetch(SystemState::NAP_COMPACT) == false);
    CHECK(supports_predictive_prefetch(SystemState::IDLE)        == false);
}

// ────────────────────────────────────────────────────────────────────────────
// §8  Storage profile: SSD/NVMe flags
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("use_nordahead: SSD uses MDB_NORDAHEAD; HDD does not", "[storage_profile]") {
    CHECK(use_nordahead(StorageProfile::SSD_NVME) == true);
    CHECK(use_nordahead(StorageProfile::HDD)      == false);
}

TEST_CASE("recommended_commit_policy: SSD → NOSYNC; HDD → SYNC", "[storage_profile]") {
    CHECK(recommended_commit_policy(StorageProfile::SSD_NVME) == CommitPolicy::NOSYNC);
    CHECK(recommended_commit_policy(StorageProfile::HDD)      == CommitPolicy::SYNC);
}

TEST_CASE("force_sequential_on_hdd: HDD forces SEQUENTIAL globally", "[storage_profile]") {
    CHECK(force_sequential_on_hdd(StorageProfile::HDD)      == true);
    CHECK(force_sequential_on_hdd(StorageProfile::SSD_NVME) == false);
}

TEST_CASE("requires_full_copy_compact: only HDD during NAP_COMPACT", "[storage_profile]") {
    CHECK(requires_full_copy_compact(StorageProfile::HDD,      SystemState::NAP_COMPACT) == true);
    CHECK(requires_full_copy_compact(StorageProfile::SSD_NVME, SystemState::NAP_COMPACT) == false);
    CHECK(requires_full_copy_compact(StorageProfile::HDD,      SystemState::IDLE)        == false);
    CHECK(requires_full_copy_compact(StorageProfile::HDD,      SystemState::DREAM_WEAVE) == false);
}

// ────────────────────────────────────────────────────────────────────────────
// §9  effective_madvise_hint: HDD overrides all states to SEQUENTIAL
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("effective_madvise_hint on HDD: all states return SEQUENTIAL", "[storage_profile]") {
    for (auto s : {SystemState::DREAM_WEAVE, SystemState::GGUF_EXPORT,
                   SystemState::ACTIVE_WAKE, SystemState::NAP_COMPACT,
                   SystemState::IDLE}) {
        CHECK(effective_madvise_hint(s, StorageProfile::HDD) == MadviseHint::SEQUENTIAL);
    }
}

TEST_CASE("effective_madvise_hint on SSD: matches primary_madvise_hint", "[storage_profile]") {
    for (auto s : {SystemState::DREAM_WEAVE, SystemState::GGUF_EXPORT,
                   SystemState::ACTIVE_WAKE, SystemState::NAP_COMPACT,
                   SystemState::IDLE}) {
        CHECK(effective_madvise_hint(s, StorageProfile::SSD_NVME) == primary_madvise_hint(s));
    }
}

// ────────────────────────────────────────────────────────────────────────────
// §10  Page alignment utilities
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("align_to_page: already-aligned offset unchanged", "[page_alignment]") {
    CHECK(align_to_page(0u)      == 0u);
    CHECK(align_to_page(4'096u)  == 4'096u);
    CHECK(align_to_page(8'192u)  == 8'192u);
}

TEST_CASE("align_to_page: rounds down to page boundary", "[page_alignment]") {
    CHECK(align_to_page(1u)      == 0u);
    CHECK(align_to_page(4'095u)  == 0u);
    CHECK(align_to_page(4'097u)  == 4'096u);
    CHECK(align_to_page(8'193u)  == 8'192u);
}

TEST_CASE("is_page_aligned: multiples of PAGE_SIZE are aligned", "[page_alignment]") {
    CHECK(is_page_aligned(0u)      == true);
    CHECK(is_page_aligned(4'096u)  == true);
    CHECK(is_page_aligned(8'192u)  == true);
    CHECK(is_page_aligned(1u)      == false);
    CHECK(is_page_aligned(4'095u)  == false);
    CHECK(is_page_aligned(4'097u)  == false);
}

TEST_CASE("hilbert_index_to_offset: linear product", "[page_alignment]") {
    CHECK(hilbert_index_to_offset(0u,   128u) == 0u);
    CHECK(hilbert_index_to_offset(1u,   128u) == 128u);
    CHECK(hilbert_index_to_offset(32u,  128u) == 4'096u);      // exactly one page
    CHECK(hilbert_index_to_offset(100u, 64u)  == 6'400u);
}

TEST_CASE("hilbert_index_to_page_offset: result is page-aligned", "[page_alignment]") {
    // 128-byte node: index 1 → offset 128 → page_offset 0
    CHECK(is_page_aligned(hilbert_index_to_page_offset(1u,   128u)));
    // 512-byte node: index 9 → offset 4608 → page_offset 4096
    CHECK(is_page_aligned(hilbert_index_to_page_offset(9u,   512u)));
    CHECK(hilbert_index_to_page_offset(9u, 512u) == 4'096u);
}

TEST_CASE("pages_spanning_range: zero nodes → zero pages", "[page_alignment]") {
    CHECK(pages_spanning_range(0u, 0u, 128u) == 0u);
    CHECK(pages_spanning_range(100u, 0u, 64u) == 0u);
}

TEST_CASE("pages_spanning_range: range within one page", "[page_alignment]") {
    // 10 nodes × 128 bytes = 1280 bytes, start aligned → 1 page
    CHECK(pages_spanning_range(0u, 10u, 128u) == 1u);
}

TEST_CASE("pages_spanning_range: exactly 32 nodes × 128 bytes = 1 page", "[page_alignment]") {
    CHECK(pages_spanning_range(0u, 32u, 128u) == 1u);
}

TEST_CASE("pages_spanning_range: 33 nodes × 128 bytes spans 2 pages", "[page_alignment]") {
    CHECK(pages_spanning_range(0u, 33u, 128u) == 2u);
}

// ────────────────────────────────────────────────────────────────────────────
// §11  Huge-page and mlock eligibility
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("is_huge_page_eligible: below 2 MiB returns false", "[eviction]") {
    CHECK(is_huge_page_eligible(0u)                        == false);
    CHECK(is_huge_page_eligible(HUGE_PAGE_SIZE_BYTES - 1u) == false);
}

TEST_CASE("is_huge_page_eligible: at and above 2 MiB returns true", "[eviction]") {
    CHECK(is_huge_page_eligible(HUGE_PAGE_SIZE_BYTES)      == true);
    CHECK(is_huge_page_eligible(HUGE_PAGE_SIZE_BYTES + 1u) == true);
    CHECK(is_huge_page_eligible(64u * 1024u * 1024u)       == true);
}

TEST_CASE("is_mlock_eligible: below one page is not eligible", "[eviction]") {
    CHECK(is_mlock_eligible(0u)                    == false);
    CHECK(is_mlock_eligible(PAGE_SIZE_BYTES - 1u)  == false);
}

TEST_CASE("is_mlock_eligible: at and above one page is eligible", "[eviction]") {
    CHECK(is_mlock_eligible(PAGE_SIZE_BYTES)       == true);
    CHECK(is_mlock_eligible(PAGE_SIZE_BYTES + 1u)  == true);
}

TEST_CASE("should_pin_wavefront: only ACTIVE_WAKE should mlock", "[eviction]") {
    CHECK(should_pin_wavefront(SystemState::ACTIVE_WAKE)  == true);
    CHECK(should_pin_wavefront(SystemState::DREAM_WEAVE)  == false);
    CHECK(should_pin_wavefront(SystemState::GGUF_EXPORT)  == false);
    CHECK(should_pin_wavefront(SystemState::NAP_COMPACT)  == false);
    CHECK(should_pin_wavefront(SystemState::IDLE)         == false);
}

// ────────────────────────────────────────────────────────────────────────────
// §12  Access pattern classification
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("is_hilbert_sequential_access: DREAM_WEAVE, GGUF_EXPORT, NAP_COMPACT", "[access_pattern]") {
    CHECK(is_hilbert_sequential_access(SystemState::DREAM_WEAVE) == true);
    CHECK(is_hilbert_sequential_access(SystemState::GGUF_EXPORT) == true);
    CHECK(is_hilbert_sequential_access(SystemState::NAP_COMPACT) == true);
    CHECK(is_hilbert_sequential_access(SystemState::ACTIVE_WAKE) == false);
    CHECK(is_hilbert_sequential_access(SystemState::IDLE)        == false);
}

TEST_CASE("is_sparse_random_access: only ACTIVE_WAKE", "[access_pattern]") {
    CHECK(is_sparse_random_access(SystemState::ACTIVE_WAKE)  == true);
    CHECK(is_sparse_random_access(SystemState::DREAM_WEAVE)  == false);
    CHECK(is_sparse_random_access(SystemState::GGUF_EXPORT)  == false);
    CHECK(is_sparse_random_access(SystemState::NAP_COMPACT)  == false);
    CHECK(is_sparse_random_access(SystemState::IDLE)         == false);
}

TEST_CASE("has_scan_pollution_risk: only GGUF_EXPORT risks evicting physics hot pages", "[access_pattern]") {
    CHECK(has_scan_pollution_risk(SystemState::GGUF_EXPORT)  == true);
    CHECK(has_scan_pollution_risk(SystemState::DREAM_WEAVE)  == false);
    CHECK(has_scan_pollution_risk(SystemState::ACTIVE_WAKE)  == false);
    CHECK(has_scan_pollution_risk(SystemState::NAP_COMPACT)  == false);
    CHECK(has_scan_pollution_risk(SystemState::IDLE)         == false);
}

// ────────────────────────────────────────────────────────────────────────────
// §13  LMDB flag recommendations
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("recommend_mdb_nordahead: SSD only", "[lmdb_flags]") {
    CHECK(recommend_mdb_nordahead(StorageProfile::SSD_NVME) == true);
    CHECK(recommend_mdb_nordahead(StorageProfile::HDD)      == false);
}

TEST_CASE("recommend_mdb_nosync: SSD only", "[lmdb_flags]") {
    CHECK(recommend_mdb_nosync(StorageProfile::SSD_NVME) == true);
    CHECK(recommend_mdb_nosync(StorageProfile::HDD)      == false);
}

TEST_CASE("recommend_mdb_nordahead and recommend_mdb_nosync agree on profile", "[lmdb_flags]") {
    // Both flags are SSD-only; they must agree
    CHECK(recommend_mdb_nordahead(StorageProfile::SSD_NVME) == recommend_mdb_nosync(StorageProfile::SSD_NVME));
    CHECK(recommend_mdb_nordahead(StorageProfile::HDD)      == recommend_mdb_nosync(StorageProfile::HDD));
}

// ────────────────────────────────────────────────────────────────────────────
// §14  Prefetch trajectory helpers
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("should_prefetch_hilbert_index: valid index within DB", "[prefetch]") {
    CHECK(should_prefetch_hilbert_index(0u,    1'000u) == true);
    CHECK(should_prefetch_hilbert_index(999u,  1'000u) == true);
    CHECK(should_prefetch_hilbert_index(1'000u, 1'000u) == false);  // equal = out of bounds
    CHECK(should_prefetch_hilbert_index(2'000u, 1'000u) == false);
}

TEST_CASE("prefetch_hint_count: one hint per trajectory point", "[prefetch]") {
    CHECK(prefetch_hint_count(0u)    == 0u);
    CHECK(prefetch_hint_count(1u)    == 1u);
    CHECK(prefetch_hint_count(100u)  == 100u);
    CHECK(prefetch_hint_count(1'000u) == 1'000u);
}

// ────────────────────────────────────────────────────────────────────────────
// §15  Diagnostic names
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("system_state_name: all 5 states", "[diagnostics]") {
    CHECK(system_state_name(SystemState::DREAM_WEAVE) == "DREAM_WEAVE");
    CHECK(system_state_name(SystemState::GGUF_EXPORT) == "GGUF_EXPORT");
    CHECK(system_state_name(SystemState::ACTIVE_WAKE) == "ACTIVE_WAKE");
    CHECK(system_state_name(SystemState::NAP_COMPACT) == "NAP_COMPACT");
    CHECK(system_state_name(SystemState::IDLE)        == "IDLE");
}

TEST_CASE("madvise_hint_name: all 5 hints", "[diagnostics]") {
    CHECK(madvise_hint_name(MadviseHint::SEQUENTIAL) == "SEQUENTIAL");
    CHECK(madvise_hint_name(MadviseHint::RANDOM)     == "RANDOM");
    CHECK(madvise_hint_name(MadviseHint::WILLNEED)   == "WILLNEED");
    CHECK(madvise_hint_name(MadviseHint::HUGEPAGE)   == "HUGEPAGE");
    CHECK(madvise_hint_name(MadviseHint::NORMAL)     == "NORMAL");
}

TEST_CASE("storage_profile_name: SSD_NVME and HDD", "[diagnostics]") {
    CHECK(storage_profile_name(StorageProfile::SSD_NVME) == "SSD_NVME");
    CHECK(storage_profile_name(StorageProfile::HDD)      == "HDD");
}

TEST_CASE("commit_policy_name: all 3 policies", "[diagnostics]") {
    CHECK(commit_policy_name(CommitPolicy::SYNC)       == "SYNC");
    CHECK(commit_policy_name(CommitPolicy::NOSYNC)     == "NOSYNC");
    CHECK(commit_policy_name(CommitPolicy::NOMETASYNC) == "NOMETASYNC");
}

TEST_CASE("All diagnostic names are non-empty", "[diagnostics]") {
    CHECK_FALSE(system_state_name(SystemState::IDLE).empty());
    CHECK_FALSE(madvise_hint_name(MadviseHint::WILLNEED).empty());
    CHECK_FALSE(storage_profile_name(StorageProfile::HDD).empty());
    CHECK_FALSE(commit_policy_name(CommitPolicy::NOMETASYNC).empty());
}

// ────────────────────────────────────────────────────────────────────────────
// §16  Invariants
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("Invariant: HUGE_PAGE_SIZE_BYTES > PAGE_SIZE_BYTES", "[invariants]") {
    CHECK(HUGE_PAGE_SIZE_BYTES > PAGE_SIZE_BYTES);
}

TEST_CASE("Invariant: PAGE_SIZE_BYTES is power of 2", "[invariants]") {
    CHECK((PAGE_SIZE_BYTES & (PAGE_SIZE_BYTES - 1u)) == 0u);
}

TEST_CASE("Invariant: HUGE_PAGE_SIZE_BYTES is power of 2", "[invariants]") {
    CHECK((HUGE_PAGE_SIZE_BYTES & (HUGE_PAGE_SIZE_BYTES - 1u)) == 0u);
}

TEST_CASE("Invariant: sequential scan ↔ is_hilbert_sequential_access agree", "[invariants]") {
    for (auto s : {SystemState::DREAM_WEAVE, SystemState::GGUF_EXPORT,
                   SystemState::ACTIVE_WAKE, SystemState::NAP_COMPACT,
                   SystemState::IDLE}) {
        CHECK(is_sequential_scan_mode(s) == is_hilbert_sequential_access(s));
    }
}

TEST_CASE("Invariant: should_apply_hugepage ↔ is_sequential_scan_mode", "[invariants]") {
    for (auto s : {SystemState::DREAM_WEAVE, SystemState::GGUF_EXPORT,
                   SystemState::ACTIVE_WAKE, SystemState::NAP_COMPACT,
                   SystemState::IDLE}) {
        CHECK(should_apply_hugepage(s) == is_sequential_scan_mode(s));
    }
}

TEST_CASE("Invariant: HDD SYNC commit matches conservative profile expectations", "[invariants]") {
    CHECK(recommended_commit_policy(StorageProfile::HDD) == CommitPolicy::SYNC);
    CHECK(recommend_mdb_nosync(StorageProfile::HDD) == false);
}

TEST_CASE("Invariant: align_to_page is idempotent", "[invariants]") {
    for (uint64_t v : {0u, 1u, 4'095u, 4'096u, 4'097u, 8'192u}) {
        CHECK(align_to_page(align_to_page(v)) == align_to_page(v));
    }
}

TEST_CASE("Invariant: hilbert_index_to_page_offset(idx, sz) <= hilbert_index_to_offset(idx, sz)", "[invariants]") {
    CHECK(hilbert_index_to_page_offset(1u, 200u) <= hilbert_index_to_offset(1u, 200u));
    CHECK(hilbert_index_to_page_offset(10u, 64u) <= hilbert_index_to_offset(10u, 64u));
}

// ────────────────────────────────────────────────────────────────────────────
// §17  Integration scenarios
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("Integration: Mamba-9D DREAM_WEAVE scan configuration", "[integration]") {
    const SystemState state                       = SystemState::DREAM_WEAVE;
    [[maybe_unused]] const StorageProfile prof    = StorageProfile::SSD_NVME;

    // Correct hint dispatched
    CHECK(primary_madvise_hint(state) == MadviseHint::SEQUENTIAL);
    // Hugepages enabled to minimise TLB misses during sequential scan
    CHECK(should_apply_hugepage(state)    == true);
    // Predictive WILLNEED prefetch available (Mamba-9D attention predictions)
    CHECK(supports_predictive_prefetch(state) == true);
    // On SSD: NORDAHEAD lets us manage prefetch
    CHECK(use_nordahead(prof) == true);
    // Effective hint unchanged by SSD profile
    CHECK(effective_madvise_hint(state, prof) == MadviseHint::SEQUENTIAL);
    // Scan is Hilbert-sequential
    CHECK(is_hilbert_sequential_access(state) == true);
}

TEST_CASE("Integration: GGUF export — scan pollution mitigation", "[integration]") {
    const SystemState state = SystemState::GGUF_EXPORT;

    // SEQUENTIAL ensures pages are dropped after scanning → avoids evicting hot physics pages
    CHECK(primary_madvise_hint(state)     == MadviseHint::SEQUENTIAL);
    CHECK(has_scan_pollution_risk(state)  == true);       // risk exists, SEQUENTIAL mitigates it
    CHECK(is_sequential_scan_mode(state)  == true);       // mitigation applied
    // No wavefront pinning needed (physics not actively running during export)
    CHECK(should_pin_wavefront(state)     == false);
}

TEST_CASE("Integration: neurogenesis ACTIVE_WAKE sparse update", "[integration]") {
    const SystemState state   = SystemState::ACTIVE_WAKE;

    // Disable read-ahead to save I/O bandwidth on sparse pattern
    CHECK(primary_madvise_hint(state)      == MadviseHint::RANDOM);
    CHECK(is_random_access_mode(state)     == true);
    CHECK(is_sequential_scan_mode(state)   == false);
    // No huge page promotion for sparse access
    CHECK(should_apply_hugepage(state)     == false);
    // Hot wavefront should be mlock()-pinned
    CHECK(should_pin_wavefront(state)      == true);
    // No predictive prefetch during wake (unpredictable neurogenesis locations)
    CHECK(supports_predictive_prefetch(state) == false);
}

TEST_CASE("Integration: HDD legacy archive — maximum sequentiality", "[integration]") {
    const StorageProfile prof = StorageProfile::HDD;

    // HDD forces SEQUENTIAL for all states to exploit drive controller
    for (auto s : {SystemState::DREAM_WEAVE, SystemState::GGUF_EXPORT,
                   SystemState::ACTIVE_WAKE, SystemState::NAP_COMPACT,
                   SystemState::IDLE}) {
        CHECK(effective_madvise_hint(s, prof) == MadviseHint::SEQUENTIAL);
    }
    // Full-copy compaction required during Nap to restore physical ordering
    CHECK(requires_full_copy_compact(prof, SystemState::NAP_COMPACT) == true);
    // Conservative SYNC commits to protect against mechanical failures
    CHECK(recommended_commit_policy(prof) == CommitPolicy::SYNC);
    // No manual nordahead management on HDD
    CHECK(use_nordahead(prof) == false);
}

TEST_CASE("Integration: trajectory prefetch for predicted semantic region", "[integration]") {
    // DREAM_WEAVE supports predictive prefetch
    CHECK(supports_predictive_prefetch(SystemState::DREAM_WEAVE) == true);

    // 5 predicted Hilbert indices in a DB of 10,000 nodes
    constexpr uint64_t DB_SIZE = 10'000u;
    constexpr uint64_t trajectory[] = {100u, 200u, 300u, 400u, 500u};
    for (uint64_t idx : trajectory) {
        CHECK(should_prefetch_hilbert_index(idx, DB_SIZE) == true);
    }
    // Out-of-range index not prefetched
    CHECK(should_prefetch_hilbert_index(10'000u, DB_SIZE) == false);

    // 5 prefetch hints issued
    CHECK(prefetch_hint_count(5u) == 5u);

    // Page-aligned offsets computed from node geometry (128-byte nodes)
    constexpr uint64_t node_sz = 128u;
    for (uint64_t idx : trajectory) {
        const uint64_t page_off = hilbert_index_to_page_offset(idx, node_sz);
        CHECK(is_page_aligned(page_off));
        CHECK(page_off <= hilbert_index_to_offset(idx, node_sz));
    }
}
