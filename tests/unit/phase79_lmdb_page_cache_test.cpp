// phase79_lmdb_page_cache_test.cpp
//
// GAP-027: LMDB Memory-Mapped I/O Page Cache Management
// Spec: docs/info/integration/sections/04_infrastructure/06_database_persistence.md §GAP-027
//
// Tests: enums, primary/companion policy selection, medium profiles,
// HDD sequential override, Hugepage/mlock predicates, eviction priority,
// Hilbert page geometry, label functions, integration scenarios.

#include <catch2/catch_test_macros.hpp>
#include <nikola/infrastructure/lmdb_page_cache.hpp>

using namespace nikola::infrastructure;

// ─── §1 PageCacheSystemState Enum Values ─────────────────────────────────────

TEST_CASE("PageCacheSystemState ACTIVE_WAKE is 0", "[state][enum]") {
    REQUIRE(static_cast<uint8_t>(PageCacheSystemState::ACTIVE_WAKE) == 0);
}

TEST_CASE("PageCacheSystemState DREAM_WEAVE is 1", "[state][enum]") {
    REQUIRE(static_cast<uint8_t>(PageCacheSystemState::DREAM_WEAVE) == 1);
}

TEST_CASE("PageCacheSystemState GGUF_EXPORT is 2", "[state][enum]") {
    REQUIRE(static_cast<uint8_t>(PageCacheSystemState::GGUF_EXPORT) == 2);
}

TEST_CASE("PageCacheSystemState NAP_COMPACT is 3", "[state][enum]") {
    REQUIRE(static_cast<uint8_t>(PageCacheSystemState::NAP_COMPACT) == 3);
}

TEST_CASE("PageCacheSystemState IDLE is 4", "[state][enum]") {
    REQUIRE(static_cast<uint8_t>(PageCacheSystemState::IDLE) == 4);
}

TEST_CASE("Five distinct PageCacheSystemState values", "[state][enum]") {
    REQUIRE(PageCacheSystemState::ACTIVE_WAKE != PageCacheSystemState::DREAM_WEAVE);
    REQUIRE(PageCacheSystemState::DREAM_WEAVE != PageCacheSystemState::GGUF_EXPORT);
    REQUIRE(PageCacheSystemState::GGUF_EXPORT != PageCacheSystemState::NAP_COMPACT);
    REQUIRE(PageCacheSystemState::NAP_COMPACT != PageCacheSystemState::IDLE);
}

// ─── §2 StorageMedium Enum Values ─────────────────────────────────────────────

TEST_CASE("StorageMedium SSD_NVME is 0", "[medium][enum]") {
    REQUIRE(static_cast<uint8_t>(StorageMedium::SSD_NVME) == 0);
}

TEST_CASE("StorageMedium HDD is 1", "[medium][enum]") {
    REQUIRE(static_cast<uint8_t>(StorageMedium::HDD) == 1);
}

TEST_CASE("SSD_NVME and HDD are distinct", "[medium][enum]") {
    REQUIRE(StorageMedium::SSD_NVME != StorageMedium::HDD);
}

// ─── §3 MadvisePolicy Enum Values ─────────────────────────────────────────────

TEST_CASE("MadvisePolicy SEQUENTIAL is 0", "[policy][enum]") {
    REQUIRE(static_cast<uint8_t>(MadvisePolicy::SEQUENTIAL) == 0);
}

TEST_CASE("MadvisePolicy RANDOM is 1", "[policy][enum]") {
    REQUIRE(static_cast<uint8_t>(MadvisePolicy::RANDOM) == 1);
}

TEST_CASE("MadvisePolicy WILLNEED is 2", "[policy][enum]") {
    REQUIRE(static_cast<uint8_t>(MadvisePolicy::WILLNEED) == 2);
}

TEST_CASE("MadvisePolicy HUGEPAGE is 3", "[policy][enum]") {
    REQUIRE(static_cast<uint8_t>(MadvisePolicy::HUGEPAGE) == 3);
}

TEST_CASE("MadvisePolicy NORMAL is 4", "[policy][enum]") {
    REQUIRE(static_cast<uint8_t>(MadvisePolicy::NORMAL) == 4);
}

TEST_CASE("Five distinct MadvisePolicy values", "[policy][enum]") {
    REQUIRE(MadvisePolicy::SEQUENTIAL != MadvisePolicy::RANDOM);
    REQUIRE(MadvisePolicy::RANDOM     != MadvisePolicy::WILLNEED);
    REQUIRE(MadvisePolicy::WILLNEED   != MadvisePolicy::HUGEPAGE);
    REQUIRE(MadvisePolicy::HUGEPAGE   != MadvisePolicy::NORMAL);
}

// ─── §4 LmdbFlag Constants ────────────────────────────────────────────────────

TEST_CASE("LmdbFlag NORDAHEAD value", "[lmdb_flag]") {
    REQUIRE(static_cast<uint32_t>(LmdbFlag::NORDAHEAD) == 0x00800000u);
}

TEST_CASE("LmdbFlag NOSYNC value", "[lmdb_flag]") {
    REQUIRE(static_cast<uint32_t>(LmdbFlag::NOSYNC) == 0x00010000u);
}

TEST_CASE("LmdbFlag WRITEMAP value", "[lmdb_flag]") {
    REQUIRE(static_cast<uint32_t>(LmdbFlag::WRITEMAP) == 0x00080000u);
}

TEST_CASE("LmdbFlag MAPASYNC value", "[lmdb_flag]") {
    REQUIRE(static_cast<uint32_t>(LmdbFlag::MAPASYNC) == 0x00100000u);
}

TEST_CASE("NORDAHEAD and NOSYNC are distinct bits", "[lmdb_flag]") {
    auto n = static_cast<uint32_t>(LmdbFlag::NORDAHEAD);
    auto s = static_cast<uint32_t>(LmdbFlag::NOSYNC);
    REQUIRE((n & s) == 0u);  // no overlapping bits
}

// ─── §5 CommitPolicy Enum Values ──────────────────────────────────────────────

TEST_CASE("CommitPolicy SYNCHRONOUS is 0", "[commit][enum]") {
    REQUIRE(static_cast<uint8_t>(CommitPolicy::SYNCHRONOUS) == 0);
}

TEST_CASE("CommitPolicy ASYNCHRONOUS is 1", "[commit][enum]") {
    REQUIRE(static_cast<uint8_t>(CommitPolicy::ASYNCHRONOUS) == 1);
}

TEST_CASE("CommitPolicy WAL_ASYNC is 2", "[commit][enum]") {
    REQUIRE(static_cast<uint8_t>(CommitPolicy::WAL_ASYNC) == 2);
}

// ─── §6 EvictionPriority Enum Values ──────────────────────────────────────────

TEST_CASE("EvictionPriority PINNED is 0 (highest protection)", "[eviction][enum]") {
    REQUIRE(static_cast<uint8_t>(EvictionPriority::PINNED) == 0);
}

TEST_CASE("EvictionPriority HOT is 1", "[eviction][enum]") {
    REQUIRE(static_cast<uint8_t>(EvictionPriority::HOT) == 1);
}

TEST_CASE("EvictionPriority WARM is 2", "[eviction][enum]") {
    REQUIRE(static_cast<uint8_t>(EvictionPriority::WARM) == 2);
}

TEST_CASE("EvictionPriority COLD is 3 (lowest protection)", "[eviction][enum]") {
    REQUIRE(static_cast<uint8_t>(EvictionPriority::COLD) == 3);
}

TEST_CASE("EvictionPriority ordering: PINNED < HOT < WARM < COLD", "[eviction][enum]") {
    REQUIRE(EvictionPriority::PINNED < EvictionPriority::HOT);
    REQUIRE(EvictionPriority::HOT    < EvictionPriority::WARM);
    REQUIRE(EvictionPriority::WARM   < EvictionPriority::COLD);
}

// ─── §7 Performance Constants ─────────────────────────────────────────────────

TEST_CASE("IO_STALL_REDUCTION_MAX_X is 100 (100x improvement claim)", "[constants]") {
    REQUIRE(IO_STALL_REDUCTION_MAX_X == 100);
}

TEST_CASE("NODE_SIZE_BYTES is 232 (TorusGridSoA node payload, GAP-021)", "[constants]") {
    REQUIRE(NODE_SIZE_BYTES == 232UL);
}

TEST_CASE("PAGE_SIZE_BYTES is 4096 (Linux default page)", "[constants]") {
    REQUIRE(PAGE_SIZE_BYTES == 4096UL);
}

// ─── §8 primary_page_policy() ─────────────────────────────────────────────────

TEST_CASE("DREAM_WEAVE: primary policy is SEQUENTIAL", "[primary_policy]") {
    REQUIRE(primary_page_policy(PageCacheSystemState::DREAM_WEAVE) == MadvisePolicy::SEQUENTIAL);
}

TEST_CASE("GGUF_EXPORT: primary policy is SEQUENTIAL", "[primary_policy]") {
    REQUIRE(primary_page_policy(PageCacheSystemState::GGUF_EXPORT) == MadvisePolicy::SEQUENTIAL);
}

TEST_CASE("NAP_COMPACT: primary policy is SEQUENTIAL", "[primary_policy]") {
    REQUIRE(primary_page_policy(PageCacheSystemState::NAP_COMPACT) == MadvisePolicy::SEQUENTIAL);
}

TEST_CASE("ACTIVE_WAKE: primary policy is RANDOM (disable read-ahead)", "[primary_policy]") {
    REQUIRE(primary_page_policy(PageCacheSystemState::ACTIVE_WAKE) == MadvisePolicy::RANDOM);
}

TEST_CASE("IDLE: primary policy is NORMAL (restore default)", "[primary_policy]") {
    REQUIRE(primary_page_policy(PageCacheSystemState::IDLE) == MadvisePolicy::NORMAL);
}

// ─── §9 companion_page_policy() ───────────────────────────────────────────────

TEST_CASE("DREAM_WEAVE: companion policy is HUGEPAGE", "[companion_policy]") {
    REQUIRE(companion_page_policy(PageCacheSystemState::DREAM_WEAVE) == MadvisePolicy::HUGEPAGE);
}

TEST_CASE("GGUF_EXPORT: companion policy is HUGEPAGE", "[companion_policy]") {
    REQUIRE(companion_page_policy(PageCacheSystemState::GGUF_EXPORT) == MadvisePolicy::HUGEPAGE);
}

TEST_CASE("NAP_COMPACT: companion policy is HUGEPAGE", "[companion_policy]") {
    REQUIRE(companion_page_policy(PageCacheSystemState::NAP_COMPACT) == MadvisePolicy::HUGEPAGE);
}

TEST_CASE("ACTIVE_WAKE: companion policy is NORMAL (no companion)", "[companion_policy]") {
    REQUIRE(companion_page_policy(PageCacheSystemState::ACTIVE_WAKE) == MadvisePolicy::NORMAL);
}

TEST_CASE("IDLE: companion policy is NORMAL", "[companion_policy]") {
    REQUIRE(companion_page_policy(PageCacheSystemState::IDLE) == MadvisePolicy::NORMAL);
}

// ─── §10 is_sequential_access_state() ────────────────────────────────────────

TEST_CASE("DREAM_WEAVE is sequential access state", "[sequential_state]") {
    REQUIRE(is_sequential_access_state(PageCacheSystemState::DREAM_WEAVE));
}

TEST_CASE("GGUF_EXPORT is sequential access state", "[sequential_state]") {
    REQUIRE(is_sequential_access_state(PageCacheSystemState::GGUF_EXPORT));
}

TEST_CASE("NAP_COMPACT is sequential access state", "[sequential_state]") {
    REQUIRE(is_sequential_access_state(PageCacheSystemState::NAP_COMPACT));
}

TEST_CASE("ACTIVE_WAKE is NOT a sequential access state", "[sequential_state]") {
    REQUIRE_FALSE(is_sequential_access_state(PageCacheSystemState::ACTIVE_WAKE));
}

TEST_CASE("IDLE is NOT a sequential access state", "[sequential_state]") {
    REQUIRE_FALSE(is_sequential_access_state(PageCacheSystemState::IDLE));
}

// ─── §11 is_random_access_state() ────────────────────────────────────────────

TEST_CASE("ACTIVE_WAKE is random access state (neurogenesis)", "[random_state]") {
    REQUIRE(is_random_access_state(PageCacheSystemState::ACTIVE_WAKE));
}

TEST_CASE("DREAM_WEAVE is NOT random access state", "[random_state]") {
    REQUIRE_FALSE(is_random_access_state(PageCacheSystemState::DREAM_WEAVE));
}

TEST_CASE("GGUF_EXPORT is NOT random access state", "[random_state]") {
    REQUIRE_FALSE(is_random_access_state(PageCacheSystemState::GGUF_EXPORT));
}

TEST_CASE("NAP_COMPACT is NOT random access state", "[random_state]") {
    REQUIRE_FALSE(is_random_access_state(PageCacheSystemState::NAP_COMPACT));
}

TEST_CASE("IDLE is NOT random access state", "[random_state]") {
    REQUIRE_FALSE(is_random_access_state(PageCacheSystemState::IDLE));
}

// ─── §12 willneed_prefetch_enabled() ─────────────────────────────────────────

TEST_CASE("ACTIVE_WAKE enables WILLNEED prefetch (predicted trajectories)", "[willneed]") {
    REQUIRE(willneed_prefetch_enabled(PageCacheSystemState::ACTIVE_WAKE));
}

TEST_CASE("DREAM_WEAVE does not use WILLNEED prefetch (already SEQUENTIAL)", "[willneed]") {
    REQUIRE_FALSE(willneed_prefetch_enabled(PageCacheSystemState::DREAM_WEAVE));
}

TEST_CASE("GGUF_EXPORT does not use WILLNEED prefetch", "[willneed]") {
    REQUIRE_FALSE(willneed_prefetch_enabled(PageCacheSystemState::GGUF_EXPORT));
}

TEST_CASE("IDLE does not use WILLNEED prefetch", "[willneed]") {
    REQUIRE_FALSE(willneed_prefetch_enabled(PageCacheSystemState::IDLE));
}

// ─── §13 SSD/NVMe Profile ─────────────────────────────────────────────────────

TEST_CASE("SSD_NVME supports aggressive prefetch", "[ssd_profile]") {
    REQUIRE(supports_aggressive_prefetch(StorageMedium::SSD_NVME));
}

TEST_CASE("HDD does NOT support aggressive prefetch", "[ssd_profile]") {
    REQUIRE_FALSE(supports_aggressive_prefetch(StorageMedium::HDD));
}

TEST_CASE("SSD_NVME: should set NORDAHEAD (manual WILLNEED management)", "[ssd_profile]") {
    REQUIRE(should_set_nordahead(StorageMedium::SSD_NVME));
}

TEST_CASE("HDD: should NOT set NORDAHEAD (let drive controller read-ahead)", "[ssd_profile]") {
    REQUIRE_FALSE(should_set_nordahead(StorageMedium::HDD));
}

TEST_CASE("SSD_NVME WAL commit policy is ASYNCHRONOUS", "[ssd_profile]") {
    REQUIRE(wal_commit_policy(StorageMedium::SSD_NVME) == CommitPolicy::ASYNCHRONOUS);
}

TEST_CASE("HDD WAL commit policy is SYNCHRONOUS", "[ssd_profile]") {
    REQUIRE(wal_commit_policy(StorageMedium::HDD) == CommitPolicy::SYNCHRONOUS);
}

// ─── §14 HDD Profile ──────────────────────────────────────────────────────────

TEST_CASE("HDD requires full-copy compact during Nap cycle", "[hdd_profile]") {
    REQUIRE(requires_full_copy_compact(StorageMedium::HDD));
}

TEST_CASE("SSD_NVME does NOT require full-copy compact", "[hdd_profile]") {
    REQUIRE_FALSE(requires_full_copy_compact(StorageMedium::SSD_NVME));
}

TEST_CASE("HDD forces SEQUENTIAL globally", "[hdd_profile]") {
    REQUIRE(force_sequential_on_hdd(StorageMedium::HDD));
}

TEST_CASE("SSD_NVME does NOT force SEQUENTIAL globally", "[hdd_profile]") {
    REQUIRE_FALSE(force_sequential_on_hdd(StorageMedium::SSD_NVME));
}

// ─── §15 effective_policy() — HDD Override ────────────────────────────────────

TEST_CASE("HDD effective policy for ACTIVE_WAKE is SEQUENTIAL (override)", "[effective_policy]") {
    REQUIRE(effective_policy(PageCacheSystemState::ACTIVE_WAKE, StorageMedium::HDD)
            == MadvisePolicy::SEQUENTIAL);
}

TEST_CASE("HDD effective policy for DREAM_WEAVE is SEQUENTIAL", "[effective_policy]") {
    REQUIRE(effective_policy(PageCacheSystemState::DREAM_WEAVE, StorageMedium::HDD)
            == MadvisePolicy::SEQUENTIAL);
}

TEST_CASE("HDD effective policy for IDLE is SEQUENTIAL (force override)", "[effective_policy]") {
    REQUIRE(effective_policy(PageCacheSystemState::IDLE, StorageMedium::HDD)
            == MadvisePolicy::SEQUENTIAL);
}

TEST_CASE("SSD effective policy for ACTIVE_WAKE is RANDOM (no override)", "[effective_policy]") {
    REQUIRE(effective_policy(PageCacheSystemState::ACTIVE_WAKE, StorageMedium::SSD_NVME)
            == MadvisePolicy::RANDOM);
}

TEST_CASE("SSD effective policy for DREAM_WEAVE is SEQUENTIAL", "[effective_policy]") {
    REQUIRE(effective_policy(PageCacheSystemState::DREAM_WEAVE, StorageMedium::SSD_NVME)
            == MadvisePolicy::SEQUENTIAL);
}

TEST_CASE("SSD effective policy for IDLE is NORMAL", "[effective_policy]") {
    REQUIRE(effective_policy(PageCacheSystemState::IDLE, StorageMedium::SSD_NVME)
            == MadvisePolicy::NORMAL);
}

TEST_CASE("SSD effective policy for GGUF_EXPORT is SEQUENTIAL", "[effective_policy]") {
    REQUIRE(effective_policy(PageCacheSystemState::GGUF_EXPORT, StorageMedium::SSD_NVME)
            == MadvisePolicy::SEQUENTIAL);
}

// ─── §16 should_use_hugepage() ────────────────────────────────────────────────

TEST_CASE("Region of exactly 2 MB qualifies for Huge Pages", "[hugepage]") {
    REQUIRE(should_use_hugepage(2UL * 1024UL * 1024UL));
}

TEST_CASE("Region larger than 2 MB qualifies for Huge Pages", "[hugepage]") {
    REQUIRE(should_use_hugepage(4UL * 1024UL * 1024UL));
}

TEST_CASE("Region of 2 MB - 1 byte does NOT qualify for Huge Pages", "[hugepage]") {
    REQUIRE_FALSE(should_use_hugepage(2UL * 1024UL * 1024UL - 1UL));
}

TEST_CASE("Region of 1 MB does NOT qualify for Huge Pages", "[hugepage]") {
    REQUIRE_FALSE(should_use_hugepage(1UL * 1024UL * 1024UL));
}

TEST_CASE("Region of 0 bytes does NOT qualify for Huge Pages", "[hugepage]") {
    REQUIRE_FALSE(should_use_hugepage(0UL));
}

// ─── §17 should_pin_region() ──────────────────────────────────────────────────

TEST_CASE("Non-empty region with memlock available should be pinned", "[pin_region]") {
    REQUIRE(should_pin_region(4096UL, true));
}

TEST_CASE("Large region with memlock available should be pinned", "[pin_region]") {
    REQUIRE(should_pin_region(64UL * 1024UL * 1024UL, true));
}

TEST_CASE("Non-empty region without memlock should NOT be pinned", "[pin_region]") {
    REQUIRE_FALSE(should_pin_region(4096UL, false));
}

TEST_CASE("Zero-byte region should NOT be pinned even with memlock", "[pin_region]") {
    REQUIRE_FALSE(should_pin_region(0UL, true));
}

TEST_CASE("Zero-byte region without memlock should NOT be pinned", "[pin_region]") {
    REQUIRE_FALSE(should_pin_region(0UL, false));
}

// ─── §18 eviction_priority() ──────────────────────────────────────────────────

TEST_CASE("ACTIVE_WAVEFRONT eviction priority is PINNED", "[eviction_priority]") {
    REQUIRE(eviction_priority(RegionCategory::ACTIVE_WAVEFRONT) == EvictionPriority::PINNED);
}

TEST_CASE("HOT_WAVEFUNCTION eviction priority is HOT", "[eviction_priority]") {
    REQUIRE(eviction_priority(RegionCategory::HOT_WAVEFUNCTION) == EvictionPriority::HOT);
}

TEST_CASE("WARM_TOPOLOGY eviction priority is WARM", "[eviction_priority]") {
    REQUIRE(eviction_priority(RegionCategory::WARM_TOPOLOGY) == EvictionPriority::WARM);
}

TEST_CASE("COLD_METADATA eviction priority is COLD", "[eviction_priority]") {
    REQUIRE(eviction_priority(RegionCategory::COLD_METADATA) == EvictionPriority::COLD);
}

TEST_CASE("ACTIVE_WAVEFRONT has higher protection than COLD_METADATA", "[eviction_priority]") {
    REQUIRE(eviction_priority(RegionCategory::ACTIVE_WAVEFRONT)
            < eviction_priority(RegionCategory::COLD_METADATA));
}

TEST_CASE("Region priority ordering: wavefront < wavefunction < topology < metadata", "[eviction_priority]") {
    REQUIRE(eviction_priority(RegionCategory::ACTIVE_WAVEFRONT)
            < eviction_priority(RegionCategory::HOT_WAVEFUNCTION));
    REQUIRE(eviction_priority(RegionCategory::HOT_WAVEFUNCTION)
            < eviction_priority(RegionCategory::WARM_TOPOLOGY));
    REQUIRE(eviction_priority(RegionCategory::WARM_TOPOLOGY)
            < eviction_priority(RegionCategory::COLD_METADATA));
}

// ─── §19 hilbert_node_offset() ────────────────────────────────────────────────

TEST_CASE("Hilbert index 0 maps to byte offset 0", "[hilbert_offset]") {
    REQUIRE(hilbert_node_offset(0) == 0UL);
}

TEST_CASE("Hilbert index 1 maps to byte offset 232 (NODE_SIZE_BYTES)", "[hilbert_offset]") {
    REQUIRE(hilbert_node_offset(1) == 232UL);
}

TEST_CASE("Hilbert index 10 maps to byte offset 2320", "[hilbert_offset]") {
    REQUIRE(hilbert_node_offset(10) == 2320UL);
}

TEST_CASE("Hilbert index 18 maps to byte offset 4176 (crosses first page boundary)", "[hilbert_offset]") {
    REQUIRE(hilbert_node_offset(18) == 4176UL);
}

TEST_CASE("hilbert_node_offset is linear in index", "[hilbert_offset]") {
    REQUIRE(hilbert_node_offset(5) == hilbert_node_offset(4) + NODE_SIZE_BYTES);
    REQUIRE(hilbert_node_offset(100) == 100UL * NODE_SIZE_BYTES);
}

// ─── §20 page_align_down() ────────────────────────────────────────────────────

TEST_CASE("page_align_down(0) is 0", "[page_align]") {
    REQUIRE(page_align_down(0UL) == 0UL);
}

TEST_CASE("page_align_down(4095) is 0 (still in first page)", "[page_align]") {
    REQUIRE(page_align_down(4095UL) == 0UL);
}

TEST_CASE("page_align_down(4096) is 4096 (start of second page)", "[page_align]") {
    REQUIRE(page_align_down(4096UL) == 4096UL);
}

TEST_CASE("page_align_down(8191) is 4096", "[page_align]") {
    REQUIRE(page_align_down(8191UL) == 4096UL);
}

TEST_CASE("page_align_down(8192) is 8192", "[page_align]") {
    REQUIRE(page_align_down(8192UL) == 8192UL);
}

TEST_CASE("page_align_down strips sub-page bits", "[page_align]") {
    REQUIRE(page_align_down(4097UL) == 4096UL);
    REQUIRE(page_align_down(4100UL) == 4096UL);
}

// ─── §21 pages_for_node() ─────────────────────────────────────────────────────

TEST_CASE("Node at offset 0: 232 bytes fits within first page (1 page)", "[pages_for_node]") {
    // aligned=0, end=232, (232+4095)/4096 = 1
    REQUIRE(pages_for_node(0UL) == 1UL);
}

TEST_CASE("Node at offset 4096: starts at page boundary, fits within one page (1 page)", "[pages_for_node]") {
    // aligned=4096, end=4328, (4328-4096+4095)/4096 = 4327/4096 = 1
    REQUIRE(pages_for_node(4096UL) == 1UL);
}

TEST_CASE("Node at offset 4000: straddles first/second page boundary (2 pages)", "[pages_for_node]") {
    // aligned=0, end=4232, (4232+4095)/4096 = 8327/4096 = 2
    REQUIRE(pages_for_node(4000UL) == 2UL);
}

TEST_CASE("Node at offset 232: still in first page (1 page)", "[pages_for_node]") {
    // aligned=0, end=464, (464+4095)/4096 = 4559/4096 = 1
    REQUIRE(pages_for_node(232UL) == 1UL);
}

TEST_CASE("Node at offset 3900: end=4132 straddles page 0 and page 4096 (2 pages)", "[pages_for_node]") {
    // aligned=0, end=4132, (4132+4095)/4096 = 8227/4096 = 2
    REQUIRE(pages_for_node(3900UL) == 2UL);
}

TEST_CASE("Node at offset 8000: straddles 2nd/3rd page boundary (2 pages)", "[pages_for_node]") {
    // aligned=4096*1=4096, wait: 8000 / 4096 = page 1 (4096..8191)
    // aligned=page_align_down(8000)=4096, end=8232
    // (8232-4096+4095)/4096 = 8231/4096 = 2
    REQUIRE(pages_for_node(8000UL) == 2UL);
}

// ─── §22 Label Functions ──────────────────────────────────────────────────────

TEST_CASE("page_cache_state_name ACTIVE_WAKE", "[labels]") {
    REQUIRE(std::string(page_cache_state_name(PageCacheSystemState::ACTIVE_WAKE)) == "ACTIVE_WAKE");
}

TEST_CASE("page_cache_state_name DREAM_WEAVE", "[labels]") {
    REQUIRE(std::string(page_cache_state_name(PageCacheSystemState::DREAM_WEAVE)) == "DREAM_WEAVE");
}

TEST_CASE("page_cache_state_name GGUF_EXPORT", "[labels]") {
    REQUIRE(std::string(page_cache_state_name(PageCacheSystemState::GGUF_EXPORT)) == "GGUF_EXPORT");
}

TEST_CASE("page_cache_state_name NAP_COMPACT", "[labels]") {
    REQUIRE(std::string(page_cache_state_name(PageCacheSystemState::NAP_COMPACT)) == "NAP_COMPACT");
}

TEST_CASE("page_cache_state_name IDLE", "[labels]") {
    REQUIRE(std::string(page_cache_state_name(PageCacheSystemState::IDLE)) == "IDLE");
}

TEST_CASE("storage_medium_name SSD_NVME", "[labels]") {
    REQUIRE(std::string(storage_medium_name(StorageMedium::SSD_NVME)) == "SSD_NVME");
}

TEST_CASE("storage_medium_name HDD", "[labels]") {
    REQUIRE(std::string(storage_medium_name(StorageMedium::HDD)) == "HDD");
}

TEST_CASE("madvise_policy_name SEQUENTIAL", "[labels]") {
    REQUIRE(std::string(madvise_policy_name(MadvisePolicy::SEQUENTIAL)) == "SEQUENTIAL");
}

TEST_CASE("madvise_policy_name RANDOM", "[labels]") {
    REQUIRE(std::string(madvise_policy_name(MadvisePolicy::RANDOM)) == "RANDOM");
}

TEST_CASE("madvise_policy_name WILLNEED", "[labels]") {
    REQUIRE(std::string(madvise_policy_name(MadvisePolicy::WILLNEED)) == "WILLNEED");
}

TEST_CASE("madvise_policy_name HUGEPAGE", "[labels]") {
    REQUIRE(std::string(madvise_policy_name(MadvisePolicy::HUGEPAGE)) == "HUGEPAGE");
}

TEST_CASE("madvise_policy_name NORMAL", "[labels]") {
    REQUIRE(std::string(madvise_policy_name(MadvisePolicy::NORMAL)) == "NORMAL");
}

TEST_CASE("commit_policy_name SYNCHRONOUS", "[labels]") {
    REQUIRE(std::string(commit_policy_name(CommitPolicy::SYNCHRONOUS)) == "SYNCHRONOUS");
}

TEST_CASE("commit_policy_name ASYNCHRONOUS", "[labels]") {
    REQUIRE(std::string(commit_policy_name(CommitPolicy::ASYNCHRONOUS)) == "ASYNCHRONOUS");
}

TEST_CASE("eviction_priority_name PINNED", "[labels]") {
    REQUIRE(std::string(eviction_priority_name(EvictionPriority::PINNED)) == "PINNED");
}

TEST_CASE("eviction_priority_name COLD", "[labels]") {
    REQUIRE(std::string(eviction_priority_name(EvictionPriority::COLD)) == "COLD");
}

TEST_CASE("region_category_name ACTIVE_WAVEFRONT", "[labels]") {
    REQUIRE(std::string(region_category_name(RegionCategory::ACTIVE_WAVEFRONT)) == "ACTIVE_WAVEFRONT");
}

TEST_CASE("region_category_name COLD_METADATA", "[labels]") {
    REQUIRE(std::string(region_category_name(RegionCategory::COLD_METADATA)) == "COLD_METADATA");
}

// ─── §23 Integration Scenarios ────────────────────────────────────────────────

TEST_CASE("SSD scenario: ACTIVE_WAKE — RANDOM primary, no companion, WILLNEED enabled", "[integration]") {
    auto state  = PageCacheSystemState::ACTIVE_WAKE;
    auto medium = StorageMedium::SSD_NVME;
    REQUIRE(effective_policy(state, medium)  == MadvisePolicy::RANDOM);
    REQUIRE(companion_page_policy(state)     == MadvisePolicy::NORMAL);
    REQUIRE(willneed_prefetch_enabled(state) == true);
    REQUIRE(is_random_access_state(state)    == true);
    REQUIRE(is_sequential_access_state(state) == false);
}

TEST_CASE("SSD scenario: DREAM_WEAVE — SEQUENTIAL + HUGEPAGE companion, no WILLNEED", "[integration]") {
    auto state  = PageCacheSystemState::DREAM_WEAVE;
    auto medium = StorageMedium::SSD_NVME;
    REQUIRE(effective_policy(state, medium)   == MadvisePolicy::SEQUENTIAL);
    REQUIRE(companion_page_policy(state)      == MadvisePolicy::HUGEPAGE);
    REQUIRE(willneed_prefetch_enabled(state)  == false);
    REQUIRE(is_sequential_access_state(state) == true);
}

TEST_CASE("SSD scenario: GGUF_EXPORT — SEQUENTIAL + HUGEPAGE, prevents scan pollution", "[integration]") {
    auto state  = PageCacheSystemState::GGUF_EXPORT;
    auto medium = StorageMedium::SSD_NVME;
    REQUIRE(effective_policy(state, medium)   == MadvisePolicy::SEQUENTIAL);
    REQUIRE(companion_page_policy(state)      == MadvisePolicy::HUGEPAGE);
    REQUIRE(is_sequential_access_state(state) == true);
    REQUIRE(supports_aggressive_prefetch(medium) == true);
    REQUIRE(should_set_nordahead(medium)         == true);
}

TEST_CASE("HDD scenario: all states map to SEQUENTIAL effective policy", "[integration]") {
    auto hdd = StorageMedium::HDD;
    for (auto state : {
        PageCacheSystemState::ACTIVE_WAKE,
        PageCacheSystemState::DREAM_WEAVE,
        PageCacheSystemState::GGUF_EXPORT,
        PageCacheSystemState::NAP_COMPACT,
        PageCacheSystemState::IDLE
    }) {
        REQUIRE(effective_policy(state, hdd) == MadvisePolicy::SEQUENTIAL);
    }
}

TEST_CASE("HDD scenario: requires compaction, no NORDAHEAD, sync commits", "[integration]") {
    auto hdd = StorageMedium::HDD;
    REQUIRE(requires_full_copy_compact(hdd) == true);
    REQUIRE(should_set_nordahead(hdd)       == false);
    REQUIRE(wal_commit_policy(hdd)          == CommitPolicy::SYNCHRONOUS);
    REQUIRE(supports_aggressive_prefetch(hdd) == false);
}

TEST_CASE("Eviction policy: active wavefront must be pinned regardless of region size", "[integration]") {
    REQUIRE(eviction_priority(RegionCategory::ACTIVE_WAVEFRONT) == EvictionPriority::PINNED);
    REQUIRE(should_pin_region(4096UL, true)  == true);
    REQUIRE(should_pin_region(4096UL, false) == false);
}

TEST_CASE("Hugepage promotion threshold: exactly 2 MB boundary", "[integration]") {
    constexpr std::size_t TWO_MB = 2UL * 1024UL * 1024UL;
    REQUIRE(should_use_hugepage(TWO_MB)     == true);
    REQUIRE(should_use_hugepage(TWO_MB - 1) == false);
}

TEST_CASE("Hilbert geometry: NODE_SIZE_BYTES × index = page-aligned offset for index 18", "[integration]") {
    // Index 18: offset 4176, in page starting at 4096 (within one page after alignment)
    const std::size_t off = hilbert_node_offset(18);
    REQUIRE(off == 4176UL);
    REQUIRE(page_align_down(off) == 4096UL);
    REQUIRE(pages_for_node(off) == 1UL);  // 4176..4408 is within page 4096..8191
}

TEST_CASE("Hilbert geometry: straddling node has 2 pages", "[integration]") {
    // Index 17: offset 3944, end 4176 → crosses page 0→4096
    const std::size_t off = hilbert_node_offset(17);
    REQUIRE(off == 3944UL);
    REQUIRE(page_align_down(off) == 0UL);
    REQUIRE(pages_for_node(off) == 2UL);
}

TEST_CASE("Sequential states: all three get SEQUENTIAL primary on SSD", "[integration]") {
    int seq_count = 0;
    for (auto state : {
        PageCacheSystemState::DREAM_WEAVE,
        PageCacheSystemState::GGUF_EXPORT,
        PageCacheSystemState::NAP_COMPACT
    }) {
        if (primary_page_policy(state) == MadvisePolicy::SEQUENTIAL) ++seq_count;
    }
    REQUIRE(seq_count == 3);
}

TEST_CASE("Only ACTIVE_WAKE gets RANDOM primary policy on SSD", "[integration]") {
    int random_count = 0;
    for (auto state : {
        PageCacheSystemState::ACTIVE_WAKE,
        PageCacheSystemState::DREAM_WEAVE,
        PageCacheSystemState::GGUF_EXPORT,
        PageCacheSystemState::NAP_COMPACT,
        PageCacheSystemState::IDLE
    }) {
        if (primary_page_policy(state) == MadvisePolicy::RANDOM) ++random_count;
    }
    REQUIRE(random_count == 1);
}

TEST_CASE("LMDB flag bits are non-overlapping across NORDAHEAD and NOSYNC", "[integration]") {
    uint32_t combined = static_cast<uint32_t>(LmdbFlag::NORDAHEAD) |
                        static_cast<uint32_t>(LmdbFlag::NOSYNC);
    REQUIRE(combined == (0x00800000u | 0x00010000u));
    REQUIRE((combined & static_cast<uint32_t>(LmdbFlag::NORDAHEAD)) != 0u);
    REQUIRE((combined & static_cast<uint32_t>(LmdbFlag::NOSYNC))    != 0u);
}
