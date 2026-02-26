/**
 * @file include/nikola/infrastructure/lmdb_compaction.hpp
 * @brief GAP-040: LMDB Full-Copy Compaction Validator
 *
 * Spec: docs/info/integration/sections/04_infrastructure/06_database_persistence.md
 *       §"Spinning Disk (HDD) Profile" — Full Copy Compact strategy
 *
 * LMDB never reclaims pages in-place; deleted records leave "holes" in the
 * B-tree that accumulate over time.  The only way to reclaim disk space is a
 * full-copy compact via `mdb_env_copy2(..., MDB_CP_COMPACT)`, which writes
 * only live pages to a new file.  This header provides:
 *
 *   1. `LmdbCompactionStats`  — file-size measurements before / after compact.
 *   2. `compact_lmdb()`       — RAII-safe wrapper around mdb_env_copy2.
 *   3. `compaction_needed()`  — integrates with the policy layer in
 *                               lmdb_page_cache.hpp to decide when to compact.
 *
 * The Nikola Nap-cycle scheduler calls `compaction_needed()` during the low-
 * activity nap window to decide whether to trigger a compact operation on the
 * DMC persistence store.  The compacted file is then atomically renamed over
 * the original to minimise downtime.
 *
 * Usage (called during Nap cycle):
 * @code
 *   namespace ni = nikola::infrastructure;
 *   if (ni::compaction_needed(state, medium)) {
 *       auto stats = ni::compact_lmdb(src_dir, tmp_compact_dir);
 *       if (stats) {
 *           // atomic rename: tmp_compact_dir → src_dir
 *       }
 *   }
 * @endcode
 */

#pragma once

#include <nikola/infrastructure/lmdb_page_cache.hpp>

#include <cstdint>
#include <cstdio>          // std::rename
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

#include <lmdb.h>          // mdb_env_copy2, MDB_CP_COMPACT, mdb_env_*
#include <sys/stat.h>      // ::stat, struct stat

namespace nikola::infrastructure {

// ─────────────────────────────────────────────────────────────────────────────
//  Data structures
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Measurements captured during an LMDB compaction operation.
 *
 * Compare `bytes_before` and `bytes_after` to quantify reclaimed space.
 * A well-fragmented database should see `bytes_after < bytes_before`.
 */
struct LmdbCompactionStats {
    std::uintmax_t bytes_before = 0;   ///< data.mdb size before compact
    std::uintmax_t bytes_after  = 0;   ///< data.mdb size after compact
    std::uintmax_t records      = 0;   ///< live records verified in compacted DB
    double         ratio        = 1.0; ///< bytes_after / bytes_before (< 1 = good)

    /// True when compaction actually reduced the file size.
    [[nodiscard]] bool reduced() const noexcept {
        return bytes_after < bytes_before;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Policy integration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Return true when the current state + storage medium warrants a
 *        full-copy compaction (i.e., it is a Nap cycle AND the medium is HDD
 *        or the database has grown beyond a reasonable fragmentation threshold).
 *
 * On SSD/NVMe the cost of a full-copy compact is low but rarely necessary;
 * on HDD it is *required* during the Nap cycle to maintain sequential access
 * for Hilbert-curve scans (spec: "Full Copy Compact on HDD/Nap").
 */
[[nodiscard]] constexpr bool compaction_needed(
    PageCacheSystemState state, StorageMedium medium) noexcept
{
    return (state == PageCacheSystemState::NAP_COMPACT) &&
           requires_full_copy_compact(medium);
}

// ─────────────────────────────────────────────────────────────────────────────
//  compact_lmdb()
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Perform a full-copy compact of an LMDB environment.
 *
 * Opens `src_env_path` read-only, calls `mdb_env_copy2(MDB_CP_COMPACT)` to
 * write a compacted copy to `dst_env_path`, then re-opens the destination to
 * count live records and measure file sizes.
 *
 * The caller is responsible for atomically replacing `src_env_path` with
 * `dst_env_path` (e.g., std::filesystem::rename) if the compaction succeeds.
 *
 * @param src_env_path  Directory containing the source LMDB `data.mdb` file.
 * @param dst_env_path  Directory that will receive the compacted `data.mdb`.
 *                      Created if it does not exist.
 * @param mapsize_bytes LMDB map size for the source environment (must be ≥
 *                      the current DB size; default 16 MiB for unit tests).
 *
 * @returns LmdbCompactionStats on success.
 * @throws  std::runtime_error on any MDB error.
 */
[[nodiscard]] inline LmdbCompactionStats compact_lmdb(
    const std::string& src_env_path,
    const std::string& dst_env_path,
    std::size_t        mapsize_bytes = 16UL * 1024UL * 1024UL)
{
    namespace fs = std::filesystem;

    // ── Step 1: measure source file size ─────────────────────────────────────
    const std::string src_data = src_env_path + "/data.mdb";
    LmdbCompactionStats stats;

    {
        struct ::stat st{};
        if (::stat(src_data.c_str(), &st) == 0) {
            stats.bytes_before = static_cast<std::uintmax_t>(st.st_size);
        }
    }

    // ── Step 2: open source environment (read-write to allow copy) ───────────
    MDB_env* src_env = nullptr;
    if (int rc = ::mdb_env_create(&src_env); rc != 0) {
        throw std::runtime_error("compact_lmdb: mdb_env_create failed: " +
                                 std::string(::mdb_strerror(rc)));
    }

    // Require RAII cleanup even if later steps throw
    struct EnvGuard {
        MDB_env* e;
        explicit EnvGuard(MDB_env* env) : e(env) {}
        ~EnvGuard() { if (e) ::mdb_env_close(e); }
    } src_guard{src_env};

    if (int rc = ::mdb_env_set_mapsize(src_env, mapsize_bytes); rc != 0) {
        throw std::runtime_error("compact_lmdb: mdb_env_set_mapsize failed: " +
                                 std::string(::mdb_strerror(rc)));
    }
    if (int rc = ::mdb_env_open(src_env, src_env_path.c_str(), 0, 0664); rc != 0) {
        throw std::runtime_error("compact_lmdb: mdb_env_open(src) failed: " +
                                 std::string(::mdb_strerror(rc)));
    }

    // ── Step 3: create destination directory and compact ─────────────────────
    fs::create_directories(dst_env_path);

    if (int rc = ::mdb_env_copy2(src_env, dst_env_path.c_str(), MDB_CP_COMPACT);
        rc != 0) {
        throw std::runtime_error("compact_lmdb: mdb_env_copy2 failed: " +
                                 std::string(::mdb_strerror(rc)));
    }

    // ── Step 4: measure destination file size ────────────────────────────────
    const std::string dst_data = dst_env_path + "/data.mdb";
    {
        struct ::stat st{};
        if (::stat(dst_data.c_str(), &st) == 0) {
            stats.bytes_after = static_cast<std::uintmax_t>(st.st_size);
        }
    }
    stats.ratio = (stats.bytes_before > 0)
                      ? static_cast<double>(stats.bytes_after) /
                        static_cast<double>(stats.bytes_before)
                      : 1.0;

    // ── Step 5: open compacted DB and count live records ─────────────────────
    MDB_env* dst_env = nullptr;
    if (::mdb_env_create(&dst_env) == 0) {
        EnvGuard dst_guard{dst_env};
        ::mdb_env_set_mapsize(dst_env, mapsize_bytes);

        if (::mdb_env_open(dst_env, dst_env_path.c_str(), MDB_RDONLY, 0664) == 0) {
            MDB_txn* txn = nullptr;
            if (::mdb_txn_begin(dst_env, nullptr, MDB_RDONLY, &txn) == 0) {
                MDB_dbi dbi = 0;
                if (::mdb_dbi_open(txn, nullptr, 0, &dbi) == 0) {
                    MDB_stat dbi_stat{};
                    if (::mdb_stat(txn, dbi, &dbi_stat) == 0) {
                        stats.records = dbi_stat.ms_entries;
                    }
                    ::mdb_dbi_close(dst_env, dbi);
                }
                ::mdb_txn_abort(txn);
            }
        }
    }

    return stats;
}

} // namespace nikola::infrastructure
