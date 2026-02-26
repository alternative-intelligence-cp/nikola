/**
 * phase99_lsm_neurogenesis_test.cpp
 *
 * Phase 99 — GAP-028: LSM-DMC Neurogenesis Pruning Compactor
 *
 * Tests the LsmMemTable and compact_to_lmdb() machinery that forms the
 * L0 (memory) → L1 (LMDB SSTable) compaction tier of the Differential
 * Manifold Checkpointing subsystem.
 *
 * Sections:
 *   1. Compile-time constants
 *   2. LsmMemTable — basic operations
 *   3. LsmMemTable — sorted_events() ordering
 *   4. CompactionStats — prune_ratio()
 *   5. compact_to_lmdb — MERGE_ONLY: all events survive, duplicates merged
 *   6. compact_to_lmdb — PRUNE_LOW: low-resonance entries evicted
 *   7. compact_to_lmdb — FULL: same pruning as PRUNE_LOW (structure check)
 *   8. compact_to_lmdb — readback via lsm_read_record()
 *   9. compact_to_lmdb — empty MemTable returns zero stats
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "nikola/persistence/lsm_neurogenesis.hpp"

#include <filesystem>
#include <cmath>

using namespace nikola::persistence;

// ─── helper: temporary LMDB directory (cleaned up after each test) ──────────
static std::filesystem::path tmp_db(const std::string& suffix) {
    auto p = std::filesystem::temp_directory_path() /
             ("nikola_p99_" + suffix);
    std::filesystem::remove_all(p);
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 1 — Compile-time constants
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase99 — compile-time constants", "[phase99][constants]")
{
    STATIC_CHECK(MEMTABLE_MAX_ENTRIES  == 4096);
    STATIC_CHECK(LSM_LMDB_MAPSIZE     == 64ULL * 1024 * 1024);
    // PRUNE_RESONANCE_THRESHOLD is float (runtime constant)
    CHECK(PRUNE_RESONANCE_THRESHOLD == Catch::Approx(0.01f));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 2 — LsmMemTable: basic operations
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase99 — LsmMemTable: basic operations", "[phase99][memtable]")
{
    SECTION("Default-constructed is empty") {
        LsmMemTable m;
        CHECK(m.size()  == 0u);
        CHECK(m.empty() == true);
        CHECK(m.full()  == false);
    }

    SECTION("append() increments size") {
        LsmMemTable m;
        m.append({1, 10, 0.5f, 0.3f});
        CHECK(m.size() == 1u);
        CHECK(m.empty() == false);
        m.append({2, 20, 0.2f, 0.1f});
        CHECK(m.size() == 2u);
    }

    SECTION("clear() resets to empty") {
        LsmMemTable m;
        m.append({1, 1, 1.0f, 1.0f});
        m.append({2, 2, 1.0f, 1.0f});
        m.clear();
        CHECK(m.size()  == 0u);
        CHECK(m.empty() == true);
    }

    SECTION("full() when MEMTABLE_MAX_ENTRIES appended") {
        LsmMemTable m;
        for (std::size_t i = 0; i < MEMTABLE_MAX_ENTRIES; ++i)
            m.append({static_cast<uint64_t>(i), i, 1.0f, 1.0f});
        CHECK(m.full() == true);
        CHECK(m.size() == MEMTABLE_MAX_ENTRIES);
    }

    SECTION("append() on full table throws std::overflow_error") {
        LsmMemTable m;
        for (std::size_t i = 0; i < MEMTABLE_MAX_ENTRIES; ++i)
            m.append({static_cast<uint64_t>(i), i, 1.0f, 1.0f});
        CHECK_THROWS_AS(m.append({9999, 9999, 0.0f, 0.0f}),
                        std::overflow_error);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 3 — LsmMemTable: sorted_events() ordering
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase99 — LsmMemTable: sorted_events()", "[phase99][sort]")
{
    SECTION("Returns events sorted by hilbert_key ascending") {
        LsmMemTable m;
        m.append({300, 1, 0.1f, 0.1f});
        m.append({100, 2, 0.2f, 0.2f});
        m.append({200, 3, 0.3f, 0.3f});
        auto sv = m.sorted_events();
        REQUIRE(sv.size() == 3u);
        CHECK(sv[0].hilbert_key == 100u);
        CHECK(sv[1].hilbert_key == 200u);
        CHECK(sv[2].hilbert_key == 300u);
    }

    SECTION("Duplicate hilbert_key entries are sorted by tick ascending") {
        LsmMemTable m;
        m.append({42, 30, 0.1f, 0.1f});
        m.append({42, 10, 0.2f, 0.2f});
        m.append({42, 20, 0.3f, 0.3f});
        auto sv = m.sorted_events();
        REQUIRE(sv.size() == 3u);
        CHECK(sv[0].tick == 10u);
        CHECK(sv[1].tick == 20u);
        CHECK(sv[2].tick == 30u);
    }

    SECTION("sorted_events() does not mutate the MemTable") {
        LsmMemTable m;
        m.append({5, 1, 1.0f, 1.0f});
        m.append({1, 2, 1.0f, 1.0f});
        const std::size_t sz_before = m.size();
        auto sv = m.sorted_events();
        CHECK(m.size() == sz_before);
        // Original order preserved in internal buffer (first append is still index 0)
        auto sv2 = m.sorted_events();
        CHECK(sv2[0].hilbert_key == sv[0].hilbert_key);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 4 — CompactionStats: prune_ratio()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase99 — CompactionStats: prune_ratio()", "[phase99][stats]")
{
    SECTION("All events written → ratio == 0") {
        CompactionStats s{10, 0, 0, 10};
        CHECK(s.prune_ratio() == Catch::Approx(0.0));
    }

    SECTION("All events pruned → ratio == 1") {
        CompactionStats s{10, 0, 0, 0};
        CHECK(s.prune_ratio() == Catch::Approx(1.0));
    }

    SECTION("Half pruned → ratio == 0.5") {
        CompactionStats s{10, 0, 0, 5};
        CHECK(s.prune_ratio() == Catch::Approx(0.5));
    }

    SECTION("Zero input → ratio == 0 (no division by zero)") {
        CompactionStats s{0, 0, 0, 0};
        CHECK(s.prune_ratio() == Catch::Approx(0.0));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 5 — compact_to_lmdb: MERGE_ONLY
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase99 — compact_to_lmdb: MERGE_ONLY strategy", "[phase99][compact]")
{
    SECTION("Three distinct keys → all survive, events_written == 3") {
        auto db = tmp_db("merge_only_distinct");
        LsmMemTable m;
        m.append({10, 1, 0.5f, 0.001f});  // low resonance but MERGE_ONLY
        m.append({20, 2, 0.3f, 0.005f});
        m.append({30, 3, 0.1f, 0.002f});

        auto stats = compact_to_lmdb(m, db, CompactionStrategy::MERGE_ONLY);
        CHECK(stats.events_in      == 3u);
        CHECK(stats.events_merged  == 0u);
        CHECK(stats.events_pruned  == 0u);
        CHECK(stats.events_written == 3u);
        std::filesystem::remove_all(db);
    }

    SECTION("Two events same hilbert_key → merged into 1, resonance summed") {
        auto db = tmp_db("merge_only_dup");
        LsmMemTable m;
        m.append({42, 10, 1.0f, 0.3f});
        m.append({42, 20, 2.0f, 0.5f});  // same key, newer tick
        m.append({99, 30, 0.1f, 0.1f});

        auto stats = compact_to_lmdb(m, db, CompactionStrategy::MERGE_ONLY);
        CHECK(stats.events_in      == 3u);
        CHECK(stats.events_merged  == 1u);  // one extra event collapsed
        CHECK(stats.events_pruned  == 0u);
        CHECK(stats.events_written == 2u);  // 2 unique keys
        std::filesystem::remove_all(db);
    }

    SECTION("Four events, two groups of two → 2 written, 2 merged") {
        auto db = tmp_db("merge_only_two_groups");
        LsmMemTable m;
        m.append({1, 5,  1.0f, 0.2f});
        m.append({1, 10, 1.0f, 0.3f});
        m.append({7, 3,  0.5f, 0.1f});
        m.append({7, 8,  0.5f, 0.4f});

        auto stats = compact_to_lmdb(m, db, CompactionStrategy::MERGE_ONLY);
        CHECK(stats.events_in      == 4u);
        CHECK(stats.events_merged  == 2u);
        CHECK(stats.events_written == 2u);
        std::filesystem::remove_all(db);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 6 — compact_to_lmdb: PRUNE_LOW
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase99 — compact_to_lmdb: PRUNE_LOW strategy", "[phase99][prune]")
{
    SECTION("Low-resonance entry pruned") {
        auto db = tmp_db("prune_low_basic");
        LsmMemTable m;
        m.append({10, 1, 0.5f, 0.005f});  // resonance < threshold → pruned
        m.append({20, 2, 0.3f, 0.100f});  // resonance >= threshold → kept

        auto stats = compact_to_lmdb(m, db, CompactionStrategy::PRUNE_LOW);
        CHECK(stats.events_in      == 2u);
        CHECK(stats.events_pruned  == 1u);
        CHECK(stats.events_written == 1u);
        std::filesystem::remove_all(db);
    }

    SECTION("All resonance above threshold → nothing pruned") {
        auto db = tmp_db("prune_low_none");
        LsmMemTable m;
        m.append({1, 1, 0.5f, 0.5f});
        m.append({2, 2, 0.5f, 0.5f});
        m.append({3, 3, 0.5f, 0.5f});

        auto stats = compact_to_lmdb(m, db, CompactionStrategy::PRUNE_LOW);
        CHECK(stats.events_pruned  == 0u);
        CHECK(stats.events_written == 3u);
        std::filesystem::remove_all(db);
    }

    SECTION("Merged resonance just at threshold → NOT pruned") {
        // merge_count=2: resonance = 0.005 + 0.006 = 0.011 >= 0.01
        auto db = tmp_db("prune_low_threshold");
        LsmMemTable m;
        m.append({55, 1, 0.1f, 0.005f});
        m.append({55, 2, 0.1f, 0.006f});

        auto stats = compact_to_lmdb(m, db, CompactionStrategy::PRUNE_LOW);
        CHECK(stats.events_merged  == 1u);
        CHECK(stats.events_pruned  == 0u);   // 0.011 >= 0.01
        CHECK(stats.events_written == 1u);
        std::filesystem::remove_all(db);
    }

    SECTION("Merged resonance below threshold → pruned") {
        // resonance = 0.003 + 0.004 = 0.007 < 0.01
        auto db = tmp_db("prune_low_below");
        LsmMemTable m;
        m.append({77, 1, 0.1f, 0.003f});
        m.append({77, 2, 0.1f, 0.004f});

        auto stats = compact_to_lmdb(m, db, CompactionStrategy::PRUNE_LOW);
        CHECK(stats.events_merged  == 1u);
        CHECK(stats.events_pruned  == 1u);
        CHECK(stats.events_written == 0u);
        std::filesystem::remove_all(db);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 7 — compact_to_lmdb: FULL strategy
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase99 — compact_to_lmdb: FULL strategy", "[phase99][full]")
{
    SECTION("FULL behaves like PRUNE_LOW for pruning decisions") {
        auto db = tmp_db("full_prune");
        LsmMemTable m;
        m.append({10, 1, 0.5f, 0.002f});  // low: pruned
        m.append({20, 2, 0.5f, 0.200f});  // high: kept

        auto stats = compact_to_lmdb(m, db, CompactionStrategy::FULL);
        CHECK(stats.events_pruned  == 1u);
        CHECK(stats.events_written == 1u);
        std::filesystem::remove_all(db);
    }

    SECTION("FULL prune_ratio is correct") {
        auto db = tmp_db("full_ratio");
        LsmMemTable m;
        for (int i = 0; i < 4; ++i)
            m.append({static_cast<uint64_t>(i), static_cast<uint64_t>(i), 0.5f, 0.5f});
        for (int i = 4; i < 8; ++i)
            m.append({static_cast<uint64_t>(i), static_cast<uint64_t>(i), 0.5f, 0.001f});

        auto stats = compact_to_lmdb(m, db, CompactionStrategy::FULL);
        CHECK(stats.events_in      == 8u);
        CHECK(stats.events_written == 4u);
        CHECK(stats.prune_ratio()  == Catch::Approx(0.5));
        std::filesystem::remove_all(db);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 8 — lsm_read_record(): readback from LMDB
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase99 — lsm_read_record(): readback", "[phase99][readback]")
{
    SECTION("Written record reads back with correct values") {
        auto db = tmp_db("readback_basic");
        LsmMemTable m;
        m.append({42, 100, 3.5f, 0.8f});

        [[maybe_unused]] auto st1 = compact_to_lmdb(m, db, CompactionStrategy::MERGE_ONLY);

        NeurogenesisRecord rec{};
        bool found = lsm_read_record(db, 42ULL, rec);
        CHECK(found == true);
        CHECK(rec.latest_tick     == 100u);
        CHECK(rec.total_energy    == Catch::Approx(3.5f));
        CHECK(rec.total_resonance == Catch::Approx(0.8f));
        CHECK(rec.merge_count     == 1u);
        std::filesystem::remove_all(db);
    }

    SECTION("Merged record: latest_tick is max, values summed, merge_count correct") {
        auto db = tmp_db("readback_merged");
        LsmMemTable m;
        m.append({7, 50,  1.0f, 0.3f});
        m.append({7, 80,  2.0f, 0.4f});  // tick=80 is newer
        m.append({7, 30,  0.5f, 0.1f});

        [[maybe_unused]] auto st2 = compact_to_lmdb(m, db, CompactionStrategy::MERGE_ONLY);

        NeurogenesisRecord rec{};
        bool found = lsm_read_record(db, 7ULL, rec);
        CHECK(found == true);
        CHECK(rec.latest_tick     == 80u);
        CHECK(rec.total_energy    == Catch::Approx(3.5f));   // 1+2+0.5
        CHECK(rec.total_resonance == Catch::Approx(0.8f));   // 0.3+0.4+0.1
        CHECK(rec.merge_count     == 3u);
        std::filesystem::remove_all(db);
    }

    SECTION("Pruned key returns false from lsm_read_record()") {
        auto db = tmp_db("readback_pruned");
        LsmMemTable m;
        m.append({99, 1, 0.1f, 0.001f});  // resonance < threshold → pruned

        [[maybe_unused]] auto st3 = compact_to_lmdb(m, db, CompactionStrategy::PRUNE_LOW);

        NeurogenesisRecord rec{};
        // Key was pruned — however the LMDB env may not have been created if
        // survivors was empty.  lsm_read_record should return false gracefully.
        // (LMDB will fail to open a non-existent env; catch accordingly)
        bool found = false;
        try {
            found = lsm_read_record(db, 99ULL, rec);
        } catch (const std::runtime_error&) {
            found = false;  // Acceptable: empty DB was not created
        }
        CHECK(found == false);
        std::filesystem::remove_all(db);
    }

    SECTION("Non-existent hilbert_key returns false") {
        auto db = tmp_db("readback_missing");
        LsmMemTable m;
        m.append({10, 1, 0.5f, 0.5f});
        [[maybe_unused]] auto st4 = compact_to_lmdb(m, db, CompactionStrategy::MERGE_ONLY);

        NeurogenesisRecord rec{};
        CHECK(lsm_read_record(db, 999ULL, rec) == false);
        std::filesystem::remove_all(db);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 9 — Empty MemTable corner case
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase99 — compact_to_lmdb: empty MemTable", "[phase99][edge]")
{
    SECTION("Empty MemTable compaction returns all-zero stats") {
        auto db = tmp_db("empty_memtable");
        LsmMemTable m;

        auto stats = compact_to_lmdb(m, db);
        CHECK(stats.events_in      == 0u);
        CHECK(stats.events_merged  == 0u);
        CHECK(stats.events_pruned  == 0u);
        CHECK(stats.events_written == 0u);
        CHECK(stats.prune_ratio()  == Catch::Approx(0.0));
        // No LMDB env was created — db_path may not exist
        std::filesystem::remove_all(db);
    }
}
