/**
 * @file lsm_neurogenesis.hpp
 * @brief GAP-028: LSM-DMC neurogenesis pruning compactor.
 *
 * Implements the Log-Structured Merge (LSM) layer for the Differential
 * Manifold Checkpointing (DMC) subsystem.  Neurogenesis events — state-delta
 * records that capture every new node created by the wave-interference engine —
 * are first accumulated in an in-memory L0 MemTable, then compacted (merged +
 * pruned) into an LMDB-backed L1 SSTable during the metabolic "Nap" cycle.
 *
 * Key properties (from §GAP-028):
 *  - Append-only MemTable accumulates up to MEMTABLE_MAX_ENTRIES before flush.
 *  - Compaction sorts by Hilbert key (spatial locality), merges duplicate keys
 *    (keep latest tick, accumulate resonance), and evicts entries whose
 *    cumulative resonance falls below the PRUNE_RESONANCE_THRESHOLD.
 *  - LMDB is the L1 SSTable store: key = 8-byte Hilbert index, value = packed
 *    NeurogenesisRecord.
 *
 * Spec:  §"Backup Architecture: The Log-Structured Manifold" of
 *         04_memory_data_systems.md (GAP-028)
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include <lmdb.h>

namespace nikola::persistence {

// ─────────────────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum entries in one MemTable page before it must be flushed.
inline constexpr std::size_t MEMTABLE_MAX_ENTRIES = 4096;

/// LMDB map-size for a single L1 SSTable (64 MiB — plenty for 4K merged recs)
inline constexpr std::size_t LSM_LMDB_MAPSIZE = 64ULL * 1024 * 1024;

/// Resonance values strictly below this are pruned during PRUNE_LOW / FULL
/// compaction.  Mirrors the metabolic eviction threshold from §GAP-034.
inline constexpr float PRUNE_RESONANCE_THRESHOLD = 0.01f;

// ─────────────────────────────────────────────────────────────────────────────
//  Data types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @struct NeurogenesisEvent
 * @brief Single append to the MemTable — records a state-delta for one node.
 *
 * Fields:
 *  - hilbert_key  : Hilbert curve index that gives spatial sort order.
 *  - tick         : Physics clock tick at event time (monotone counter).
 *  - delta_energy : |ΔΨ|² — wavefunction energy of this delta.
 *  - resonance    : Semantic resonance score contributed by this event.
 */
struct NeurogenesisEvent {
    uint64_t hilbert_key{0};
    uint64_t tick{0};
    float    delta_energy{0.0f};
    float    resonance{0.0f};
};

/**
 * @struct NeurogenesisRecord
 * @brief Merged / compacted record stored as the LMDB value (24 bytes).
 *
 * After merging all NeurogenesisEvents sharing the same hilbert_key:
 *  - latest_tick       : tick of the newest contributing event.
 *  - total_energy      : sum of all delta_energy values.
 *  - total_resonance   : sum of all resonance values.
 *  - merge_count       : number of events merged into this record.
 */
#pragma pack(push, 1)
struct NeurogenesisRecord {
    uint64_t latest_tick{0};
    float    total_energy{0.0f};
    float    total_resonance{0.0f};
    uint32_t merge_count{0};
};
#pragma pack(pop)
static_assert(sizeof(NeurogenesisRecord) == 20,
              "NeurogenesisRecord must be exactly 20 bytes");

// ─────────────────────────────────────────────────────────────────────────────
//  Compaction strategy
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @enum CompactionStrategy
 * @brief Controls how aggressively the compactor prunes during flush.
 */
enum class CompactionStrategy : uint8_t {
    MERGE_ONLY = 0, ///< Merge duplicate hilbert_key entries; never prune.
    PRUNE_LOW  = 1, ///< Merge + evict entries below resonance_threshold.
    FULL       = 2, ///< Merge + prune + re-sort LMDB (full defragmentation).
};

// ─────────────────────────────────────────────────────────────────────────────
//  L0 MemTable
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @class LsmMemTable
 * @brief Append-only in-memory L0 buffer for neurogenesis events.
 *
 * Accumulates NeurogenesisEvent objects until flushed (compacted) to LMDB.
 * Not thread-safe — the caller must synchronise if multiple writers exist.
 */
class LsmMemTable {
public:
    /// Append a neurogenesis event.  Throws std::overflow_error if full().
    void append(NeurogenesisEvent ev) {
        if (full()) {
            throw std::overflow_error(
                "LsmMemTable::append — table is full; flush before appending");
        }
        entries_.push_back(ev);
    }

    /// Number of events currently buffered.
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    /// True when size() >= MEMTABLE_MAX_ENTRIES.
    [[nodiscard]] bool full() const noexcept {
        return entries_.size() >= MEMTABLE_MAX_ENTRIES;
    }

    /// True when no events are buffered.
    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }

    /// Discard all buffered events (reset after flush).
    void clear() noexcept { entries_.clear(); }

    /**
     * @brief Return a copy of the buffered events sorted by hilbert_key.
     *        Entries with equal hilbert_key are secondarily sorted by tick
     *        (ascending) so the last entry in each group is the newest.
     */
    [[nodiscard]] std::vector<NeurogenesisEvent> sorted_events() const {
        auto copy = entries_;
        std::sort(copy.begin(), copy.end(),
                  [](const NeurogenesisEvent& a, const NeurogenesisEvent& b) {
                      if (a.hilbert_key != b.hilbert_key)
                          return a.hilbert_key < b.hilbert_key;
                      return a.tick < b.tick;
                  });
        return copy;
    }

private:
    std::vector<NeurogenesisEvent> entries_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Compaction stats
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @struct CompactionStats
 * @brief Counters returned by compact_to_lmdb().
 */
struct CompactionStats {
    std::size_t events_in{0};     ///< Total events read from the MemTable.
    std::size_t events_merged{0}; ///< Duplicate hilbert_keys collapsed.
    std::size_t events_pruned{0}; ///< Entries evicted for low resonance.
    std::size_t events_written{0};///< Surviving entries written to LMDB.

    /// Fraction of input events that were either merged away or pruned.
    [[nodiscard]] double prune_ratio() const noexcept {
        if (events_in == 0) return 0.0;
        return static_cast<double>(events_in - events_written) /
               static_cast<double>(events_in);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  LMDB helper RAII guard
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

struct LmdbEnvGuard {
    MDB_env* env{nullptr};
    explicit LmdbEnvGuard(MDB_env* e) : env(e) {}
    ~LmdbEnvGuard() { if (env) { mdb_env_close(env); env = nullptr; } }
    LmdbEnvGuard(const LmdbEnvGuard&) = delete;
    LmdbEnvGuard& operator=(const LmdbEnvGuard&) = delete;
};

struct LmdbTxnGuard {
    MDB_txn* txn{nullptr};
    explicit LmdbTxnGuard(MDB_txn* t) : txn(t) {}
    ~LmdbTxnGuard() { if (txn) { mdb_txn_abort(txn); txn = nullptr; } }
    void commit() {
        if (txn) {
            int rc = mdb_txn_commit(txn);
            txn = nullptr;
            if (rc != MDB_SUCCESS)
                throw std::runtime_error(std::string("mdb_txn_commit: ") + mdb_strerror(rc));
        }
    }
    LmdbTxnGuard(const LmdbTxnGuard&) = delete;
    LmdbTxnGuard& operator=(const LmdbTxnGuard&) = delete;
};

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
//  Primary API: compact_to_lmdb()
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Compact a MemTable into an LMDB L1 SSTable.
 *
 * Steps:
 *  1. Extract sorted_events() from @p mem.
 *  2. Merge groups of identical hilbert_key: keep latest tick, sum energy and
 *     resonance, increment merge_count.
 *  3. (If strategy != MERGE_ONLY) Prune merged records whose total_resonance is
 *     strictly below @p resonance_threshold.
 *  4. Open (or create) the LMDB database at @p db_path and write surviving
 *     NeurogenesisRecord values.
 *  5. Return CompactionStats.  Does NOT call mem.clear() — caller decides.
 *
 * @param mem                  Source MemTable (read-only during compaction).
 * @param db_path              Filesystem path for the LMDB directory/file.
 * @param strategy             Compaction strategy.
 * @param resonance_threshold  Prune threshold (default: PRUNE_RESONANCE_THRESHOLD).
 * @return CompactionStats
 * @throws std::runtime_error on LMDB errors.
 */
[[nodiscard]] inline CompactionStats compact_to_lmdb(
    const LsmMemTable& mem,
    const std::filesystem::path& db_path,
    CompactionStrategy strategy           = CompactionStrategy::PRUNE_LOW,
    float              resonance_threshold = PRUNE_RESONANCE_THRESHOLD)
{
    CompactionStats stats;
    auto sorted = mem.sorted_events();
    stats.events_in = sorted.size();

    if (sorted.empty()) return stats;

    // ── Step 1: Merge duplicate hilbert_key groups ────────────────────────
    std::vector<std::pair<uint64_t, NeurogenesisRecord>> merged;
    merged.reserve(sorted.size());

    for (std::size_t i = 0; i < sorted.size(); ) {
        const uint64_t key = sorted[i].hilbert_key;
        NeurogenesisRecord rec{};
        rec.latest_tick     = sorted[i].tick;
        rec.total_energy    = 0.0f;
        rec.total_resonance = 0.0f;
        rec.merge_count     = 0;

        std::size_t j = i;
        while (j < sorted.size() && sorted[j].hilbert_key == key) {
            if (sorted[j].tick > rec.latest_tick)
                rec.latest_tick = sorted[j].tick;
            rec.total_energy    += sorted[j].delta_energy;
            rec.total_resonance += sorted[j].resonance;
            ++rec.merge_count;
            ++j;
        }
        stats.events_merged += (j - i - 1);  // extra events collapsed
        merged.emplace_back(key, rec);
        i = j;
    }

    // ── Step 2: Prune low-resonance entries ───────────────────────────────
    std::vector<std::pair<uint64_t, NeurogenesisRecord>> survivors;
    survivors.reserve(merged.size());

    for (auto& [key, rec] : merged) {
        bool prune = (strategy != CompactionStrategy::MERGE_ONLY) &&
                     (rec.total_resonance < resonance_threshold);
        if (prune) {
            ++stats.events_pruned;
        } else {
            survivors.emplace_back(key, rec);
        }
    }

    stats.events_written = survivors.size();
    if (survivors.empty()) return stats;

    // ── Step 3: Open (or create) LMDB and write survivors ─────────────────
    std::filesystem::create_directories(db_path);

    MDB_env* raw_env = nullptr;
    if (int rc = mdb_env_create(&raw_env); rc != MDB_SUCCESS)
        throw std::runtime_error(std::string("mdb_env_create: ") + mdb_strerror(rc));
    detail::LmdbEnvGuard env_guard(raw_env);

    if (int rc = mdb_env_set_mapsize(raw_env, LSM_LMDB_MAPSIZE); rc != MDB_SUCCESS)
        throw std::runtime_error(std::string("mdb_env_set_mapsize: ") + mdb_strerror(rc));

    if (int rc = mdb_env_open(raw_env, db_path.c_str(), 0, 0644); rc != MDB_SUCCESS)
        throw std::runtime_error(std::string("mdb_env_open: ") + mdb_strerror(rc));

    MDB_txn* raw_txn = nullptr;
    if (int rc = mdb_txn_begin(raw_env, nullptr, 0, &raw_txn); rc != MDB_SUCCESS)
        throw std::runtime_error(std::string("mdb_txn_begin: ") + mdb_strerror(rc));
    detail::LmdbTxnGuard txn_guard(raw_txn);

    MDB_dbi dbi{};
    if (int rc = mdb_dbi_open(raw_txn, nullptr, MDB_CREATE | MDB_INTEGERKEY, &dbi); rc != MDB_SUCCESS)
        throw std::runtime_error(std::string("mdb_dbi_open: ") + mdb_strerror(rc));

    for (auto& [key, rec] : survivors) {
        MDB_val k{ sizeof(key),  const_cast<void*>(static_cast<const void*>(&key)) };
        MDB_val v{ sizeof(rec),  const_cast<void*>(static_cast<const void*>(&rec))  };
        if (int rc = mdb_put(raw_txn, dbi, &k, &v, 0); rc != MDB_SUCCESS)
            throw std::runtime_error(std::string("mdb_put: ") + mdb_strerror(rc));
    }

    txn_guard.commit();
    return stats;
}

/**
 * @brief Read back a NeurogenesisRecord from an existing L1 SSTable.
 *
 * @param db_path      Filesystem path to the LMDB directory created by compact_to_lmdb().
 * @param hilbert_key  Key to look up.
 * @param out          Output record (populated on success).
 * @return true if found, false if the key does not exist.
 * @throws std::runtime_error on LMDB errors.
 */
[[nodiscard]] inline bool lsm_read_record(
    const std::filesystem::path& db_path,
    uint64_t                     hilbert_key,
    NeurogenesisRecord&          out)
{
    MDB_env* raw_env = nullptr;
    if (int rc = mdb_env_create(&raw_env); rc != MDB_SUCCESS)
        throw std::runtime_error(std::string("mdb_env_create: ") + mdb_strerror(rc));
    detail::LmdbEnvGuard env_guard(raw_env);

    if (int rc = mdb_env_set_mapsize(raw_env, LSM_LMDB_MAPSIZE); rc != MDB_SUCCESS)
        throw std::runtime_error(std::string("mdb_env_set_mapsize: ") + mdb_strerror(rc));

    if (int rc = mdb_env_open(raw_env, db_path.c_str(), MDB_RDONLY, 0644); rc != MDB_SUCCESS)
        throw std::runtime_error(std::string("mdb_env_open (read): ") + mdb_strerror(rc));

    MDB_txn* raw_txn = nullptr;
    if (int rc = mdb_txn_begin(raw_env, nullptr, MDB_RDONLY, &raw_txn); rc != MDB_SUCCESS)
        throw std::runtime_error(std::string("mdb_txn_begin (read): ") + mdb_strerror(rc));
    detail::LmdbTxnGuard txn_guard(raw_txn);

    MDB_dbi dbi{};
    if (int rc = mdb_dbi_open(raw_txn, nullptr, MDB_INTEGERKEY, &dbi); rc != MDB_SUCCESS) {
        if (rc == MDB_NOTFOUND) return false;
        throw std::runtime_error(std::string("mdb_dbi_open (read): ") + mdb_strerror(rc));
    }

    MDB_val k{ sizeof(hilbert_key), const_cast<void*>(static_cast<const void*>(&hilbert_key)) };
    MDB_val v{};
    int rc = mdb_get(raw_txn, dbi, &k, &v);
    if (rc == MDB_NOTFOUND) return false;
    if (rc != MDB_SUCCESS)
        throw std::runtime_error(std::string("mdb_get: ") + mdb_strerror(rc));

    std::memcpy(&out, v.mv_data, sizeof(NeurogenesisRecord));
    return true;
}

} // namespace nikola::persistence
