/**
 * @file cognitive/lmdb_memory_store.hpp
 * @brief Phase 136 — LMDB-backed persistence for SemanticMemory.
 *
 * Provides two free functions:
 *
 *   save_lmdb(SemanticMemory const&, path)  — write all records to LMDB
 *   load_lmdb(SemanticMemory&, path)        — load all records from LMDB
 *
 * And a lower-level RAII class LmdbMemoryStore for incremental operations.
 *
 * Why LMDB over the binary flat file (SemanticMemory::save / load):
 *   - ACID writes: no torn records on power failure / process kill
 *   - Incremental upserts: one mdb_put per record — no full-file rewrite
 *   - Ordered B-tree keys: Hilbert indices stored big-endian → spatial locality
 *     during cursor iteration matches the neurogenesis access pattern
 *   - Memory-mapped: zero-copy reads — the OS page cache is the DB cache
 *   - Concurrent readers: multiple DecisionLoop instances can read simultaneously
 *
 * Key / value layout:
 *   Key:   8 bytes, big-endian uint64 (Hilbert MemoryKey for ordering).
 *   Value: variable-length packed binary:
 *     [0..3]   uint32_t  n_nodes       (count of psi_real / psi_imag elements)
 *     [4..7]   float     strength
 *     [8..11]  float     age_seconds
 *     [12..15] uint32_t  access_count
 *     [16 .. 16 + n_nodes*4 - 1]               float[] psi_real
 *     [16 + n_nodes*4 .. 16 + 2*n_nodes*4 - 1] float[] psi_imag
 *   Total: 16 + 8 * n_nodes bytes per record.
 *
 * Map size:  Default 256 MiB.  For the 3^9 = 19,683-node grid this supports
 *   ~1,600 fully-loaded records with room for compaction overhead.  Resize
 *   by setting LmdbMemoryStore::MAP_SIZE_BYTES before the first open call.
 *
 * Requires: lmdb.h / liblmdb.  Link with: target_link_libraries(... lmdb)
 *
 * Thread safety: LmdbMemoryStore instances are NOT thread-safe.  Use one
 * instance per thread or protect externally.
 */
#pragma once

#include <nikola/cognitive/semantic_memory.hpp>  // MemoryKey, MemoryRecord, SemanticMemory

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <lmdb.h>

namespace nikola::cognitive {

// ============================================================================
// Constants
// ============================================================================

/// Default LMDB map size (256 MiB).  Increase for large memory stores.
inline constexpr std::size_t LMDB_MEMORY_MAP_SIZE = 256ULL * 1024 * 1024;

/// Magic number in value header to detect corrupted / version-mismatched DBs.
inline constexpr uint32_t LMDB_MEMORY_MAGIC = 0x4E4D454Du;  // "NMEM"

// ============================================================================
// Internal serialisation helpers
// ============================================================================

namespace detail {

/// Convert a uint64 to big-endian bytes for lexicographic key ordering.
inline void encode_key_be(uint64_t v, uint8_t out[8]) noexcept
{
    for (int i = 7; i >= 0; --i) {
        out[i] = static_cast<uint8_t>(v & 0xffu);
        v >>= 8;
    }
}

/// Decode 8 big-endian bytes back to uint64.
inline uint64_t decode_key_be(const uint8_t in[8]) noexcept
{
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v = (v << 8) | in[i];
    return v;
}

/// Serialise one MemoryRecord into a byte buffer.
inline std::vector<uint8_t> pack_record(const MemoryRecord& rec)
{
    const uint32_t n = static_cast<uint32_t>(rec.psi_real.size());
    // Layout: magic(4) + n_nodes(4) + strength(4) + age_seconds(4) +
    //         access_count(4)  = 20 bytes header
    //         + psi_real(n*4) + psi_imag(n*4)
    const std::size_t size = 20u + 2u * n * sizeof(float);
    std::vector<uint8_t> buf(size);
    uint8_t* p = buf.data();

    std::memcpy(p,      &LMDB_MEMORY_MAGIC,   4); p += 4;
    std::memcpy(p,      &n,                   4); p += 4;
    std::memcpy(p,      &rec.strength,         4); p += 4;
    std::memcpy(p,      &rec.age_seconds,      4); p += 4;
    std::memcpy(p,      &rec.access_count,     4); p += 4;
    if (n > 0) {
        std::memcpy(p,  rec.psi_real.data(), n * sizeof(float)); p += n * sizeof(float);
        std::memcpy(p,  rec.psi_imag.data(), n * sizeof(float));
    }
    return buf;
}

/// Deserialise a MemoryRecord from raw LMDB value bytes.
/// Returns false if the buffer is too small or the magic is wrong.
inline bool unpack_record(const void* data, std::size_t size, MemoryKey key, MemoryRecord& out)
{
    if (size < 20u) return false;
    const uint8_t* p = static_cast<const uint8_t*>(data);

    uint32_t magic = 0;
    std::memcpy(&magic, p, 4); p += 4;
    if (magic != LMDB_MEMORY_MAGIC) return false;

    uint32_t n = 0;
    std::memcpy(&n, p, 4); p += 4;

    const std::size_t expected = 20u + 2u * n * sizeof(float);
    if (size < expected || n > (1u << 20)) return false;  // sanity: max 1M nodes

    out.key = key;
    std::memcpy(&out.strength,     p, 4); p += 4;
    std::memcpy(&out.age_seconds,  p, 4); p += 4;
    std::memcpy(&out.access_count, p, 4); p += 4;
    out.psi_real.resize(n);
    out.psi_imag.resize(n);
    if (n > 0) {
        std::memcpy(out.psi_real.data(), p, n * sizeof(float)); p += n * sizeof(float);
        std::memcpy(out.psi_imag.data(), p, n * sizeof(float));
    }
    return true;
}

}  // namespace detail

// ============================================================================
// LmdbMemoryStore
// ============================================================================

/**
 * @brief RAII LMDB environment wrapper for SemanticMemory persistence.
 *
 * Opens (or creates) an LMDB database at the given directory path.
 * The environment is closed in the destructor.
 *
 * Typical usage:
 * @code
 *   LmdbMemoryStore store("/home/nikola/.nikola_memory.lmdb");
 *   store.save_all(memory);          // persist all in-RAM records
 *   // ...later...
 *   store.load_all(memory);          // restore from LMDB
 * @endcode
 */
class LmdbMemoryStore {
public:
    /// Configurable map size before first open.
    std::size_t map_size = LMDB_MEMORY_MAP_SIZE;

    /**
     * @brief Open or create the LMDB environment at @p path.
     *
     * Creates the directory if it does not yet exist.
     *
     * @param path  Filesystem path to the LMDB directory.
     * @throws std::runtime_error on LMDB error.
     */
    explicit LmdbMemoryStore(const std::string& path)
        : path_(path)
    {
        std::filesystem::create_directories(path);
        open_env();
    }

    ~LmdbMemoryStore() { close_env(); }

    // Non-copyable
    LmdbMemoryStore(const LmdbMemoryStore&)            = delete;
    LmdbMemoryStore& operator=(const LmdbMemoryStore&) = delete;
    LmdbMemoryStore(LmdbMemoryStore&&)                 = default;
    LmdbMemoryStore& operator=(LmdbMemoryStore&&)      = default;

    // ------------------------------------------------------------------
    // Bulk operations
    // ------------------------------------------------------------------

    /**
     * @brief Write (upsert) all records from @p mem into LMDB.
     *
     * Existing records with the same key are overwritten.
     * All writes are committed atomically in one transaction.
     *
     * @throws std::runtime_error on I/O or LMDB error.
     */
    void save_all(const SemanticMemory& mem)
    {
        MDB_txn* txn = nullptr;
        check(mdb_txn_begin(env_, nullptr, 0, &txn), "mdb_txn_begin");
        MDB_dbi dbi = 0;
        if (int rc = mdb_dbi_open(txn, nullptr, MDB_CREATE, &dbi); rc != MDB_SUCCESS) {
            mdb_txn_abort(txn);
            throw std::runtime_error(std::string("mdb_dbi_open: ") + mdb_strerror(rc));
        }

        for (const auto& [key, rec] : mem.records()) {
            uint8_t kbuf[8];
            detail::encode_key_be(key, kbuf);
            MDB_val mkey{ 8, kbuf };

            auto vbuf = detail::pack_record(rec);
            MDB_val mval{ vbuf.size(), vbuf.data() };

            if (int rc = mdb_put(txn, dbi, &mkey, &mval, 0); rc != MDB_SUCCESS) {
                mdb_txn_abort(txn);
                throw std::runtime_error(std::string("mdb_put: ") + mdb_strerror(rc));
            }
        }

        check(mdb_txn_commit(txn), "mdb_txn_commit");
    }

    /**
     * @brief Load all records from LMDB into @p mem (merge, not replace).
     *
     * Existing in-RAM records are retained; LMDB records with the same key
     * overwrite them (LMDB is the source of truth for cross-session state).
     *
     * @returns Number of records successfully loaded.
     * @throws  std::runtime_error on LMDB error.
     */
    [[nodiscard]] std::size_t load_all(SemanticMemory& mem)
    {
        MDB_txn* txn = nullptr;
        check(mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn), "mdb_txn_begin(RO)");

        MDB_dbi dbi = 0;
        if (int rc = mdb_dbi_open(txn, nullptr, 0, &dbi); rc != MDB_SUCCESS) {
            mdb_txn_abort(txn);
            if (rc == MDB_NOTFOUND) return 0;  // empty / new database — not an error
            throw std::runtime_error(std::string("mdb_dbi_open(RO): ") + mdb_strerror(rc));
        }

        MDB_cursor* cursor = nullptr;
        if (int rc = mdb_cursor_open(txn, dbi, &cursor); rc != MDB_SUCCESS) {
            mdb_txn_abort(txn);
            throw std::runtime_error(std::string("mdb_cursor_open: ") + mdb_strerror(rc));
        }

        std::size_t loaded = 0;
        MDB_val mkey{}, mval{};
        while (mdb_cursor_get(cursor, &mkey, &mval, MDB_NEXT) == MDB_SUCCESS) {
            if (mkey.mv_size != 8) continue;
            const MemoryKey key = detail::decode_key_be(
                static_cast<const uint8_t*>(mkey.mv_data));
            MemoryRecord rec;
            if (detail::unpack_record(mval.mv_data, mval.mv_size, key, rec)) {
                mem.insert_record(std::move(rec));
                ++loaded;
            }
        }

        mdb_cursor_close(cursor);
        mdb_txn_abort(txn);  // read-only: abort is fine
        return loaded;
    }

    /**
     * @brief Upsert a single MemoryRecord (for incremental writes).
     *
     * Cheaper than save_all() when only one record has changed.
     */
    void upsert(const MemoryRecord& rec)
    {
        MDB_txn* txn = nullptr;
        check(mdb_txn_begin(env_, nullptr, 0, &txn), "mdb_txn_begin");
        MDB_dbi dbi = 0;
        if (int rc = mdb_dbi_open(txn, nullptr, MDB_CREATE, &dbi); rc != MDB_SUCCESS) {
            mdb_txn_abort(txn);
            throw std::runtime_error(std::string("mdb_dbi_open: ") + mdb_strerror(rc));
        }
        uint8_t kbuf[8];
        detail::encode_key_be(rec.key, kbuf);
        MDB_val mkey{ 8, kbuf };
        auto vbuf = detail::pack_record(rec);
        MDB_val mval{ vbuf.size(), vbuf.data() };
        if (int rc = mdb_put(txn, dbi, &mkey, &mval, 0); rc != MDB_SUCCESS) {
            mdb_txn_abort(txn);
            throw std::runtime_error(std::string("mdb_put: ") + mdb_strerror(rc));
        }
        check(mdb_txn_commit(txn), "mdb_txn_commit");
    }

    /**
     * @brief Delete a record by key.
     */
    void erase(MemoryKey key)
    {
        MDB_txn* txn = nullptr;
        check(mdb_txn_begin(env_, nullptr, 0, &txn), "mdb_txn_begin");
        MDB_dbi dbi = 0;
        if (int rc = mdb_dbi_open(txn, nullptr, 0, &dbi); rc != MDB_SUCCESS) {
            mdb_txn_abort(txn);
            if (rc == MDB_NOTFOUND) return;
            throw std::runtime_error(std::string("mdb_dbi_open: ") + mdb_strerror(rc));
        }
        uint8_t kbuf[8];
        detail::encode_key_be(key, kbuf);
        MDB_val mkey{ 8, kbuf };
        if (int rc = mdb_del(txn, dbi, &mkey, nullptr); rc != MDB_SUCCESS && rc != MDB_NOTFOUND) {
            mdb_txn_abort(txn);
            throw std::runtime_error(std::string("mdb_del: ") + mdb_strerror(rc));
        }
        check(mdb_txn_commit(txn), "mdb_txn_commit");
    }

    /**
     * @brief Return the number of live records currently in the database.
     */
    [[nodiscard]] std::size_t db_size()
    {
        MDB_txn* txn = nullptr;
        check(mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn), "mdb_txn_begin(stat)");
        MDB_dbi dbi = 0;
        if (int rc = mdb_dbi_open(txn, nullptr, 0, &dbi); rc != MDB_SUCCESS) {
            mdb_txn_abort(txn);
            if (rc == MDB_NOTFOUND) return 0;
            throw std::runtime_error(std::string("mdb_dbi_open(stat): ") + mdb_strerror(rc));
        }
        MDB_stat stat{};
        check(mdb_stat(txn, dbi, &stat), "mdb_stat");
        mdb_txn_abort(txn);
        return static_cast<std::size_t>(stat.ms_entries);
    }

private:
    std::string path_;
    MDB_env*    env_{nullptr};

    void open_env()
    {
        check(mdb_env_create(&env_), "mdb_env_create");
        check(mdb_env_set_mapsize(env_, map_size), "mdb_env_set_mapsize");
        // MDB_NOSUBDIR: file path points to data file, not directory
        // MDB_NOSYNC: faster writes; synced on env_close (acceptable for memory cache)
        const unsigned flags = 0;  // safe defaults
        if (int rc = mdb_env_open(env_, path_.c_str(), flags, 0664); rc != MDB_SUCCESS) {
            mdb_env_close(env_);
            env_ = nullptr;
            throw std::runtime_error(std::string("mdb_env_open('") + path_ + "'): " + mdb_strerror(rc));
        }
    }

    void close_env() noexcept
    {
        if (env_) {
            mdb_env_close(env_);
            env_ = nullptr;
        }
    }

    static void check(int rc, const char* ctx)
    {
        if (rc != MDB_SUCCESS)
            throw std::runtime_error(std::string(ctx) + ": " + mdb_strerror(rc));
    }
};

// ============================================================================
// Convenience free functions
// ============================================================================

/**
 * @brief Save all SemanticMemory records to an LMDB database at @p path.
 *
 * @param mem   Source memory store.
 * @param path  Filesystem directory for the LMDB database (created if absent).
 * @throws std::runtime_error on error.
 */
inline void save_lmdb(const SemanticMemory& mem, const std::string& path)
{
    LmdbMemoryStore store(path);
    store.save_all(mem);
}

/**
 * @brief Load all records from LMDB at @p path into @p mem.
 *
 * @param mem   Target memory store; existing records are merged (not replaced).
 * @param path  Filesystem directory for the LMDB database.
 * @returns     Number of records loaded (0 if database is absent or empty).
 * @throws      std::runtime_error on LMDB error (not on absent database).
 */
[[nodiscard]] inline std::size_t load_lmdb(SemanticMemory& mem, const std::string& path)
{
    if (!std::filesystem::exists(path)) return 0;
    LmdbMemoryStore store(path);
    return store.load_all(mem);
}

}  // namespace nikola::cognitive
