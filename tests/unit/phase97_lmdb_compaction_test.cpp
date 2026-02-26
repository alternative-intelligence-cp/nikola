/**
 * @file tests/unit/phase97_lmdb_compaction_test.cpp
 * @brief Phase 97: LMDB Compaction Validator — GAP-040 (Catch2 v3).
 *
 * Validates the full LMDB compaction lifecycle specified in §"Spinning Disk
 * (HDD) Profile" of 04_infrastructure/06_database_persistence.md:
 *
 *   "During Nap compaction, perform Full Copy Compact. Read fragmented DB
 *    and write fresh, perfectly sequential copy."
 *
 * Test sections:
 *
 *   Section 1 — Policy layer: compaction_needed(), requires_full_copy_compact()
 *   Section 2 — LmdbCompactionStats structure
 *   Section 3 — Full lifecycle: write → delete → compact → verify integrity
 *   Section 4 — File size reduction: compacted DB ≤ original DB
 *   Section 5 — Record count matches (N − deleted) in compacted DB
 *   Section 6 — Deleted records absent from compacted DB
 *   Section 7 — LMDB environment open/close RAII safety
 *   Section 8 — compact_lmdb() raises on invalid path
 *
 * All tests use /tmp/nikola_p97_XX/ paths (created and destroyed per-test).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/infrastructure/lmdb_compaction.hpp>
#include <nikola/infrastructure/lmdb_page_cache.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lmdb.h>

using namespace nikola::infrastructure;
namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
//  Test helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a temporary DB path for the given tag, clean any previous remnants.
static std::string tmp_path(const char* tag) {
    std::string p = "/tmp/nikola_p97_" + std::string(tag);
    fs::remove_all(p);
    return p;
}

/// RAII wrapper that destroys a temp directory on scope exit.
struct TmpGuard {
    std::string path;
    explicit TmpGuard(std::string p) : path(std::move(p)) {}
    ~TmpGuard() { fs::remove_all(path); }
};

//
// Populate an LMDB environment.  Returns the number of records actually written.
//
static std::size_t populate_lmdb(
    const std::string& env_path,
    std::size_t        record_count,
    std::size_t        value_size,   // bytes per value
    std::size_t        mapsize = 64UL * 1024UL * 1024UL)   // 64 MiB
{
    // Fill value buffer with a deterministic pattern
    std::string val_buf(value_size, '\0');
    for (std::size_t i = 0; i < value_size; ++i) {
        val_buf[i] = static_cast<char>(i & 0xFF);
    }

    fs::create_directories(env_path);

    MDB_env* env = nullptr;
    REQUIRE(::mdb_env_create(&env) == 0);
    REQUIRE(::mdb_env_set_mapsize(env, mapsize) == 0);
    REQUIRE(::mdb_env_open(env, env_path.c_str(), 0, 0664) == 0);

    MDB_txn* txn = nullptr;
    REQUIRE(::mdb_txn_begin(env, nullptr, 0, &txn) == 0);

    MDB_dbi dbi = 0;
    REQUIRE(::mdb_dbi_open(txn, nullptr, MDB_CREATE, &dbi) == 0);

    std::size_t written = 0;
    for (std::size_t i = 0; i < record_count; ++i) {
        const std::string key = "key_" + std::to_string(i);

        MDB_val k{key.size(),   const_cast<char*>(key.data())};
        MDB_val v{val_buf.size(), const_cast<char*>(val_buf.data())};

        if (::mdb_put(txn, dbi, &k, &v, 0) == 0) {
            ++written;
        }
    }

    ::mdb_txn_commit(txn);
    ::mdb_dbi_close(env, dbi);
    ::mdb_env_close(env);
    return written;
}

//
// Delete a subset of records from an LMDB environment.
// Deletes records with keys "key_0", "key_step", "key_2*step", …
//
static std::size_t delete_records(
    const std::string& env_path,
    std::size_t        total_records,
    std::size_t        step,          // delete every step-th record starting at 0
    std::size_t        mapsize = 64UL * 1024UL * 1024UL)
{
    MDB_env* env = nullptr;
    REQUIRE(::mdb_env_create(&env) == 0);
    REQUIRE(::mdb_env_set_mapsize(env, mapsize) == 0);
    REQUIRE(::mdb_env_open(env, env_path.c_str(), 0, 0664) == 0);

    MDB_txn* txn = nullptr;
    REQUIRE(::mdb_txn_begin(env, nullptr, 0, &txn) == 0);

    MDB_dbi dbi = 0;
    REQUIRE(::mdb_dbi_open(txn, nullptr, 0, &dbi) == 0);

    std::size_t deleted = 0;
    for (std::size_t i = 0; i < total_records; i += step) {
        const std::string key = "key_" + std::to_string(i);
        MDB_val k{key.size(), const_cast<char*>(key.data())};
        if (::mdb_del(txn, dbi, &k, nullptr) == 0) {
            ++deleted;
        }
    }

    ::mdb_txn_commit(txn);
    ::mdb_dbi_close(env, dbi);
    ::mdb_env_close(env);
    return deleted;
}

//
// Count records in an LMDB environment.
//
static std::size_t count_records(
    const std::string& env_path,
    std::size_t        mapsize = 64UL * 1024UL * 1024UL)
{
    MDB_env* env = nullptr;
    if (::mdb_env_create(&env) != 0) return 0;
    ::mdb_env_set_mapsize(env, mapsize);
    if (::mdb_env_open(env, env_path.c_str(), MDB_RDONLY, 0664) != 0) {
        ::mdb_env_close(env);
        return 0;
    }

    MDB_txn* txn = nullptr;
    if (::mdb_txn_begin(env, nullptr, MDB_RDONLY, &txn) != 0) {
        ::mdb_env_close(env);
        return 0;
    }

    MDB_dbi dbi = 0;
    std::size_t count = 0;
    if (::mdb_dbi_open(txn, nullptr, 0, &dbi) == 0) {
        MDB_stat s{};
        if (::mdb_stat(txn, dbi, &s) == 0) {
            count = s.ms_entries;
        }
        ::mdb_dbi_close(env, dbi);
    }

    ::mdb_txn_abort(txn);
    ::mdb_env_close(env);
    return count;
}

//
// Check whether a specific key exists in the DB.
//
static bool key_exists(
    const std::string& env_path,
    const std::string& key,
    std::size_t        mapsize = 64UL * 1024UL * 1024UL)
{
    MDB_env* env = nullptr;
    if (::mdb_env_create(&env) != 0) return false;
    ::mdb_env_set_mapsize(env, mapsize);
    if (::mdb_env_open(env, env_path.c_str(), MDB_RDONLY, 0664) != 0) {
        ::mdb_env_close(env);
        return false;
    }

    MDB_txn* txn = nullptr;
    bool found = false;
    if (::mdb_txn_begin(env, nullptr, MDB_RDONLY, &txn) == 0) {
        MDB_dbi dbi = 0;
        if (::mdb_dbi_open(txn, nullptr, 0, &dbi) == 0) {
            MDB_val k{key.size(), const_cast<char*>(key.data())};
            MDB_val v{};
            found = (::mdb_get(txn, dbi, &k, &v) == 0);
            ::mdb_dbi_close(env, dbi);
        }
        ::mdb_txn_abort(txn);
    }

    ::mdb_env_close(env);
    return found;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 1 — Policy layer
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase97 — policy: requires_full_copy_compact() is true only for HDD",
          "[phase97][policy]")
{
    CHECK( requires_full_copy_compact(StorageMedium::HDD));
    CHECK(!requires_full_copy_compact(StorageMedium::SSD_NVME));
}

TEST_CASE("Phase97 — policy: compaction_needed() requires Nap+HDD",
          "[phase97][policy]")
{
    // Only true during NAP_COMPACT on HDD
    CHECK( compaction_needed(PageCacheSystemState::NAP_COMPACT, StorageMedium::HDD));

    // SSD — no mandatory compaction
    CHECK(!compaction_needed(PageCacheSystemState::NAP_COMPACT, StorageMedium::SSD_NVME));

    // Other states — no compaction even on HDD
    CHECK(!compaction_needed(PageCacheSystemState::ACTIVE_WAKE,  StorageMedium::HDD));
    CHECK(!compaction_needed(PageCacheSystemState::DREAM_WEAVE,  StorageMedium::HDD));
    CHECK(!compaction_needed(PageCacheSystemState::GGUF_EXPORT,  StorageMedium::HDD));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 2 — LmdbCompactionStats structure
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase97 — LmdbCompactionStats: reduced() reflects size relationship",
          "[phase97][stats]")
{
    LmdbCompactionStats s;

    SECTION("smaller after compact") {
        s.bytes_before = 10000;
        s.bytes_after  = 6000;
        s.ratio = 0.6;
        CHECK(s.reduced());
    }

    SECTION("same size — not reduced") {
        s.bytes_before = 10000;
        s.bytes_after  = 10000;
        s.ratio = 1.0;
        CHECK_FALSE(s.reduced());
    }

    SECTION("larger after (degenerate) — not reduced") {
        s.bytes_before = 5000;
        s.bytes_after  = 6000;
        s.ratio = 1.2;
        CHECK_FALSE(s.reduced());
    }

    SECTION("default-constructed stats") {
        LmdbCompactionStats dflt;
        CHECK(dflt.bytes_before == 0);
        CHECK(dflt.bytes_after  == 0);
        CHECK(dflt.records      == 0);
        CHECK(dflt.ratio == Catch::Approx(1.0));
        CHECK_FALSE(dflt.reduced());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 3 — Full lifecycle: write → delete → compact → verify
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase97 — lifecycle: write 400 records, delete 200, compact, verify integrity",
          "[phase97][lifecycle]")
{
    constexpr std::size_t TOTAL   = 400;
    constexpr std::size_t STEP    = 2;     // delete every 2nd key (index 0,2,4,…)
    constexpr std::size_t DELETED = TOTAL / STEP;  // = 200
    constexpr std::size_t ALIVE   = TOTAL - DELETED; // = 200
    constexpr std::size_t VALSIZE = 512;   // 512 bytes per value → ~200 KB total data

    const std::string src = tmp_path("lifecycle_src");
    const std::string dst = tmp_path("lifecycle_dst");
    TmpGuard src_guard{src};
    TmpGuard dst_guard{dst};

    // ── Write ────────────────────────────────────────────────────────────────
    std::size_t written = populate_lmdb(src, TOTAL, VALSIZE);
    REQUIRE(written == TOTAL);

    // Verify pre-compact record count in source
    CHECK(count_records(src) == TOTAL);

    // ── Delete every other record ─────────────────────────────────────────────
    std::size_t deleted = delete_records(src, TOTAL, STEP);
    REQUIRE(deleted == DELETED);
    CHECK(count_records(src) == ALIVE);

    // ── Compact ───────────────────────────────────────────────────────────────
    LmdbCompactionStats stats = compact_lmdb(src, dst);

    INFO("bytes_before=" << stats.bytes_before
         << "  bytes_after=" << stats.bytes_after
         << "  ratio=" << stats.ratio
         << "  records=" << stats.records);

    // compact_lmdb must report the correct live record count
    CHECK(stats.records == ALIVE);

    // File size must not grow during compaction
    CHECK(stats.bytes_after <= stats.bytes_before);

    // ── Verify data integrity in compacted DB ────────────────────────────────
    CHECK(count_records(dst) == ALIVE);

    // Spot-check: keys that were kept (odd indices: key_1, key_3, … key_399)
    for (std::size_t i = 1; i < TOTAL; i += STEP) {
        const std::string key = "key_" + std::to_string(i);
        CHECK(key_exists(dst, key));
    }

    // Spot-check: keys that were deleted (even indices: key_0, key_2, … key_398)
    for (std::size_t i = 0; i < TOTAL; i += STEP) {
        const std::string key = "key_" + std::to_string(i);
        CHECK_FALSE(key_exists(dst, key));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 4 — File size reduction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase97 — file size: compacted DB is strictly smaller than heavily fragmented original",
          "[phase97][size]")
{
    // Write 1000 records × 1024 bytes = ~1 MiB data, then delete 90%
    constexpr std::size_t TOTAL       = 1000;
    constexpr std::size_t KEEP_STEP   = 10;   // keep key_0, key_10, key_20, …
    constexpr std::size_t VALSIZE     = 1024; // 1 KiB per value

    // Strategy: write all, then delete all EXCEPT multiples of 10

    const std::string src = tmp_path("size_src");
    const std::string dst = tmp_path("size_dst");
    TmpGuard src_guard{src};
    TmpGuard dst_guard{dst};

    // Write
    REQUIRE(populate_lmdb(src, TOTAL, VALSIZE) == TOTAL);

    // Delete every record whose index is NOT a multiple of KEEP_STEP
    {
        MDB_env* env = nullptr;
        REQUIRE(::mdb_env_create(&env) == 0);
        REQUIRE(::mdb_env_set_mapsize(env, 64UL * 1024UL * 1024UL) == 0);
        REQUIRE(::mdb_env_open(env, src.c_str(), 0, 0664) == 0);

        MDB_txn* txn = nullptr;
        REQUIRE(::mdb_txn_begin(env, nullptr, 0, &txn) == 0);

        MDB_dbi dbi = 0;
        REQUIRE(::mdb_dbi_open(txn, nullptr, 0, &dbi) == 0);

        for (std::size_t i = 0; i < TOTAL; ++i) {
            if (i % KEEP_STEP != 0) {
                const std::string key = "key_" + std::to_string(i);
                MDB_val k{key.size(), const_cast<char*>(key.data())};
                ::mdb_del(txn, dbi, &k, nullptr);
            }
        }
        ::mdb_txn_commit(txn);
        ::mdb_dbi_close(env, dbi);
        ::mdb_env_close(env);
    }

    const std::size_t alive = TOTAL / KEEP_STEP; // = 100

    INFO("alive records after deletion: " << alive);
    REQUIRE(count_records(src) == alive);

    LmdbCompactionStats stats = compact_lmdb(src, dst);

    INFO("bytes_before=" << stats.bytes_before
         << "  bytes_after=" << stats.bytes_after
         << "  ratio=" << stats.ratio
         << "  records=" << stats.records);

    // Compacted file must be smaller — 90% of data was deleted
    CHECK(stats.reduced());
    CHECK(stats.ratio < 0.85);              // at least 15% reduction
    CHECK(stats.records == alive);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 5 — Record count correctness
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase97 — record count: compacted DB has exactly N-deleted records",
          "[phase97][count]")
{
    constexpr std::size_t TOTAL   = 300;
    constexpr std::size_t STEP    = 3;
    constexpr std::size_t DELETED = TOTAL / STEP;
    constexpr std::size_t ALIVE   = TOTAL - DELETED;

    const std::string src = tmp_path("count_src");
    const std::string dst = tmp_path("count_dst");
    TmpGuard src_guard{src};
    TmpGuard dst_guard{dst};

    REQUIRE(populate_lmdb(src, TOTAL, 256) == TOTAL);
    REQUIRE(delete_records(src, TOTAL, STEP) == DELETED);

    auto stats = compact_lmdb(src, dst);
    CHECK(stats.records == ALIVE);
    CHECK(count_records(dst) == ALIVE);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 6 — Deleted records are absent
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase97 — integrity: deleted records do not appear in compacted DB",
          "[phase97][integrity]")
{
    constexpr std::size_t TOTAL  = 100;
    constexpr std::size_t VALSIZE = 200;

    const std::string src = tmp_path("del_src");
    const std::string dst = tmp_path("del_dst");
    TmpGuard src_guard{src};
    TmpGuard dst_guard{dst};

    REQUIRE(populate_lmdb(src, TOTAL, VALSIZE) == TOTAL);

    // Delete keys 50–74 (a contiguous range)
    {
        MDB_env* env = nullptr;
        REQUIRE(::mdb_env_create(&env) == 0);
        REQUIRE(::mdb_env_set_mapsize(env, 16UL * 1024UL * 1024UL) == 0);
        REQUIRE(::mdb_env_open(env, src.c_str(), 0, 0664) == 0);
        MDB_txn* txn = nullptr;
        REQUIRE(::mdb_txn_begin(env, nullptr, 0, &txn) == 0);
        MDB_dbi dbi = 0;
        REQUIRE(::mdb_dbi_open(txn, nullptr, 0, &dbi) == 0);
        for (std::size_t i = 50; i < 75; ++i) {
            const std::string k = "key_" + std::to_string(i);
            MDB_val kv{k.size(), const_cast<char*>(k.data())};
            ::mdb_del(txn, dbi, &kv, nullptr);
        }
        ::mdb_txn_commit(txn);
        ::mdb_dbi_close(env, dbi);
        ::mdb_env_close(env);
    }

    [[maybe_unused]] auto _ = compact_lmdb(src, dst);

    // Keys 50–74 must be absent
    for (std::size_t i = 50; i < 75; ++i) {
        CHECK_FALSE(key_exists(dst, "key_" + std::to_string(i)));
    }
    // Keys 0–49 and 75–99 must be present
    CHECK(key_exists(dst, "key_0"));
    CHECK(key_exists(dst, "key_49"));
    CHECK(key_exists(dst, "key_75"));
    CHECK(key_exists(dst, "key_99"));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 7 — Empty-database compact
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase97 — empty DB: compact of empty database succeeds with 0 records",
          "[phase97][edge]")
{
    const std::string src = tmp_path("empty_src");
    const std::string dst = tmp_path("empty_dst");
    TmpGuard src_guard{src};
    TmpGuard dst_guard{dst};

    // Create an empty environment (no records)
    {
        fs::create_directories(src);
        MDB_env* env = nullptr;
        REQUIRE(::mdb_env_create(&env) == 0);
        REQUIRE(::mdb_env_set_mapsize(env, 1UL * 1024UL * 1024UL) == 0);
        REQUIRE(::mdb_env_open(env, src.c_str(), 0, 0664) == 0);
        MDB_txn* txn = nullptr;
        REQUIRE(::mdb_txn_begin(env, nullptr, 0, &txn) == 0);
        MDB_dbi dbi = 0;
        REQUIRE(::mdb_dbi_open(txn, nullptr, MDB_CREATE, &dbi) == 0);
        // No records written
        ::mdb_txn_commit(txn);
        ::mdb_dbi_close(env, dbi);
        ::mdb_env_close(env);
    }

    REQUIRE_NOTHROW([&]() {
        auto stats = compact_lmdb(src, dst, 1UL * 1024UL * 1024UL);
        CHECK(stats.records == 0);
        CHECK(stats.bytes_after > 0);  // file exists but is minimal LMDB header
    }());
}

// ─────────────────────────────────────────────────────────────────────────────
//  Section 8 — compact_lmdb() throws on bad source path
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase97 — error handling: compact_lmdb() throws on invalid source path",
          "[phase97][error]")
{
    const std::string bad_src = "/tmp/nikola_p97_does_not_exist_xyz";
    const std::string dst     = tmp_path("bad_dst");
    TmpGuard dst_guard{dst};

    // Ensure the nonexistent path is absent
    fs::remove_all(bad_src);

    CHECK_THROWS_AS(compact_lmdb(bad_src, dst), std::runtime_error);
}
