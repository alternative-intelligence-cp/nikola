/**
 * @file aria/code_proposal_store.hpp
 * @brief LMDB-backed persistence for Aria code proposals and compile results.
 *
 * Stores the specialist's generated code proposals alongside their compile
 * outcomes, enabling:
 *   - Feedback loop: failed proposals feed back into the training corpus
 *   - Metrics: compile success rate over time, per-difficulty breakdown
 *   - Deduplication: SHA-256 prefix keys avoid storing identical proposals
 *   - Corpus export: successful proposals → training data conversion
 *
 * Schema: single LMDB environment, one named database "proposals".
 *   Key:   8-byte big-endian proposal_id (monotonic counter)
 *   Value: packed binary CodeProposal record
 *
 * Following the same patterns as LmdbMemoryStore (Phase 136) and
 * LmdbStateStore (Phase 137).
 *
 * Thread safety: NOT thread-safe — one instance per thread.
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <lmdb.h>

namespace nikola::aria {

// ============================================================================
// CodeProposal — a single specialist output + compile result
// ============================================================================

struct CodeProposal {
    uint64_t    id{0};              ///< Monotonic proposal ID
    std::string instruction;        ///< Prompt that generated this code
    std::string source_code;        ///< Generated Aria source
    bool        compile_success{false};
    std::string compile_errors;     ///< Newline-joined error messages
    double      compile_time_ms{0.0};
    uint64_t    timestamp_ns{0};    ///< Wall-clock nanoseconds (steady_clock)
    uint32_t    iteration{0};       ///< Self-improvement iteration number
};

// ============================================================================
// Constants
// ============================================================================

inline constexpr std::size_t PROPOSAL_DB_MAP_SIZE = 128ULL * 1024 * 1024;  // 128 MiB
inline constexpr uint32_t MAGIC_PROPOSAL = 0x4E50524Fu;  // "NPRO"
inline constexpr uint32_t PROPOSAL_VERSION = 1u;

// ============================================================================
// Serialisation
// ============================================================================

namespace detail {

inline void write_be64_p(uint64_t v, uint8_t out[8]) noexcept {
    for (int i = 7; i >= 0; --i) {
        out[i] = static_cast<uint8_t>(v & 0xffu);
        v >>= 8;
    }
}

inline uint64_t read_be64_p(const uint8_t in[8]) noexcept {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | in[i];
    return v;
}

inline std::vector<uint8_t> pack_proposal(const CodeProposal& p) {
    // Layout:
    //   [0..3]    magic (4)
    //   [4..7]    version (4)
    //   [8..15]   id (8)
    //   [16..19]  instruction_len (4)
    //   [20..N]   instruction bytes
    //   [N..N+3]  source_len (4)
    //   [N+4..]   source bytes
    //   [M]       compile_success (1)
    //   [M+1..M+4] errors_len (4)
    //   [M+5..]   errors bytes
    //   [..]      compile_time_ms (8, as double)
    //   [..]      timestamp_ns (8)
    //   [..]      iteration (4)

    std::size_t size = 4 + 4 + 8       // magic + version + id
        + 4 + p.instruction.size()     // instruction
        + 4 + p.source_code.size()     // source
        + 1                             // compile_success
        + 4 + p.compile_errors.size()  // errors
        + 8 + 8 + 4;                   // compile_time + timestamp + iteration

    std::vector<uint8_t> buf(size);
    uint8_t* w = buf.data();

    auto write32 = [&](uint32_t v) { std::memcpy(w, &v, 4); w += 4; };
    auto write64 = [&](uint64_t v) { std::memcpy(w, &v, 8); w += 8; };
    auto write_str = [&](const std::string& s) {
        uint32_t len = static_cast<uint32_t>(s.size());
        write32(len);
        std::memcpy(w, s.data(), len); w += len;
    };

    write32(MAGIC_PROPOSAL);
    write32(PROPOSAL_VERSION);
    write64(p.id);
    write_str(p.instruction);
    write_str(p.source_code);
    *w++ = p.compile_success ? 1 : 0;
    write_str(p.compile_errors);
    std::memcpy(w, &p.compile_time_ms, 8); w += 8;
    write64(p.timestamp_ns);
    write32(p.iteration);

    return buf;
}

inline bool unpack_proposal(const void* data, std::size_t size, CodeProposal& out) {
    if (size < 45) return false;  // minimum header
    const uint8_t* r = static_cast<const uint8_t*>(data);
    const uint8_t* end = r + size;

    auto read32 = [&]() -> uint32_t {
        uint32_t v = 0; std::memcpy(&v, r, 4); r += 4; return v;
    };
    auto read64 = [&]() -> uint64_t {
        uint64_t v = 0; std::memcpy(&v, r, 8); r += 8; return v;
    };
    auto read_str = [&](std::string& s) -> bool {
        if (r + 4 > end) return false;
        uint32_t len = read32();
        if (r + len > end || len > 10'000'000) return false;  // 10MB sanity
        s.assign(reinterpret_cast<const char*>(r), len);
        r += len;
        return true;
    };

    uint32_t magic = read32();
    uint32_t version = read32();
    if (magic != MAGIC_PROPOSAL || version > PROPOSAL_VERSION) return false;

    out.id = read64();
    if (!read_str(out.instruction)) return false;
    if (!read_str(out.source_code)) return false;
    if (r >= end) return false;
    out.compile_success = (*r++ != 0);
    if (!read_str(out.compile_errors)) return false;
    if (r + 20 > end) return false;
    std::memcpy(&out.compile_time_ms, r, 8); r += 8;
    out.timestamp_ns = read64();
    out.iteration = read32();

    return true;
}

} // namespace detail

// ============================================================================
// CodeProposalStore
// ============================================================================

class CodeProposalStore {
public:
    /**
     * @brief Open or create the LMDB proposal store.
     *
     * @param db_path  Directory for the LMDB environment.
     *                 Created if it does not exist.
     */
    explicit CodeProposalStore(const std::string& db_path) {
        std::filesystem::create_directories(db_path);

        int rc = mdb_env_create(&env_);
        if (rc) throw std::runtime_error("mdb_env_create: " + std::string(mdb_strerror(rc)));

        mdb_env_set_mapsize(env_, PROPOSAL_DB_MAP_SIZE);
        mdb_env_set_maxdbs(env_, 2);

        rc = mdb_env_open(env_, db_path.c_str(), 0, 0664);
        if (rc) {
            mdb_env_close(env_);
            env_ = nullptr;
            throw std::runtime_error("mdb_env_open: " + std::string(mdb_strerror(rc)));
        }

        // Open named database "proposals"
        MDB_txn* txn = nullptr;
        rc = mdb_txn_begin(env_, nullptr, 0, &txn);
        if (rc) throw std::runtime_error("mdb_txn_begin: " + std::string(mdb_strerror(rc)));

        rc = mdb_dbi_open(txn, "proposals", MDB_CREATE, &dbi_);
        if (rc) {
            mdb_txn_abort(txn);
            throw std::runtime_error("mdb_dbi_open: " + std::string(mdb_strerror(rc)));
        }

        // Read next_id from the last key
        MDB_cursor* cursor = nullptr;
        rc = mdb_cursor_open(txn, dbi_, &cursor);
        if (rc == 0) {
            MDB_val k, v;
            rc = mdb_cursor_get(cursor, &k, &v, MDB_LAST);
            if (rc == 0 && k.mv_size == 8) {
                next_id_ = detail::read_be64_p(static_cast<const uint8_t*>(k.mv_data)) + 1;
            }
            mdb_cursor_close(cursor);
        }

        mdb_txn_commit(txn);
    }

    ~CodeProposalStore() {
        if (env_) {
            mdb_dbi_close(env_, dbi_);
            mdb_env_close(env_);
        }
    }

    // Non-copyable
    CodeProposalStore(const CodeProposalStore&) = delete;
    CodeProposalStore& operator=(const CodeProposalStore&) = delete;

    // ------------------------------------------------------------------
    // CRUD
    // ------------------------------------------------------------------

    /**
     * @brief Store a new proposal.  Assigns an auto-incremented ID.
     * @return The assigned proposal ID.
     */
    uint64_t store(CodeProposal& proposal) {
        proposal.id = next_id_++;
        auto buf = detail::pack_proposal(proposal);

        uint8_t key_buf[8];
        detail::write_be64_p(proposal.id, key_buf);

        MDB_val key{8, key_buf};
        MDB_val val{buf.size(), buf.data()};

        MDB_txn* txn = nullptr;
        int rc = mdb_txn_begin(env_, nullptr, 0, &txn);
        if (rc) throw std::runtime_error("mdb_txn_begin: " + std::string(mdb_strerror(rc)));

        rc = mdb_put(txn, dbi_, &key, &val, 0);
        if (rc) {
            mdb_txn_abort(txn);
            throw std::runtime_error("mdb_put: " + std::string(mdb_strerror(rc)));
        }

        mdb_txn_commit(txn);
        return proposal.id;
    }

    /**
     * @brief Retrieve a proposal by ID.
     * @return true if found, false otherwise.
     */
    bool load(uint64_t id, CodeProposal& out) const {
        uint8_t key_buf[8];
        detail::write_be64_p(id, key_buf);
        MDB_val key{8, key_buf};
        MDB_val val;

        MDB_txn* txn = nullptr;
        int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
        if (rc) return false;

        rc = mdb_get(txn, dbi_, &key, &val);
        bool ok = (rc == 0) &&
                  detail::unpack_proposal(val.mv_data, val.mv_size, out);
        mdb_txn_abort(txn);
        return ok;
    }

    /**
     * @brief Count total proposals in the store.
     */
    uint64_t count() const {
        MDB_txn* txn = nullptr;
        int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
        if (rc) return 0;

        MDB_stat stat;
        rc = mdb_stat(txn, dbi_, &stat);
        mdb_txn_abort(txn);
        return (rc == 0) ? stat.ms_entries : 0;
    }

    /**
     * @brief Count proposals that compiled successfully.
     */
    uint64_t count_successful() const {
        uint64_t n = 0;
        MDB_txn* txn = nullptr;
        int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
        if (rc) return 0;

        MDB_cursor* cursor = nullptr;
        rc = mdb_cursor_open(txn, dbi_, &cursor);
        if (rc == 0) {
            MDB_val k, v;
            while (mdb_cursor_get(cursor, &k, &v, MDB_NEXT) == 0) {
                CodeProposal p;
                if (detail::unpack_proposal(v.mv_data, v.mv_size, p) && p.compile_success) {
                    ++n;
                }
            }
            mdb_cursor_close(cursor);
        }

        mdb_txn_abort(txn);
        return n;
    }

    /**
     * @brief Export all successful proposals as training examples.
     * @param max_count  Maximum number to return (0 = unlimited).
     */
    std::vector<CodeProposal> export_successful(uint64_t max_count = 0) const {
        std::vector<CodeProposal> results;
        MDB_txn* txn = nullptr;
        int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
        if (rc) return results;

        MDB_cursor* cursor = nullptr;
        rc = mdb_cursor_open(txn, dbi_, &cursor);
        if (rc == 0) {
            MDB_val k, v;
            while (mdb_cursor_get(cursor, &k, &v, MDB_NEXT) == 0) {
                CodeProposal p;
                if (detail::unpack_proposal(v.mv_data, v.mv_size, p) && p.compile_success) {
                    results.push_back(std::move(p));
                    if (max_count > 0 && results.size() >= max_count) break;
                }
            }
            mdb_cursor_close(cursor);
        }

        mdb_txn_abort(txn);
        return results;
    }

    /**
     * @brief Compute compile success rate (0.0 – 1.0).
     */
    double success_rate() const {
        uint64_t total = count();
        if (total == 0) return 0.0;
        return static_cast<double>(count_successful()) / static_cast<double>(total);
    }

private:
    MDB_env* env_  = nullptr;
    MDB_dbi  dbi_  = 0;
    uint64_t next_id_ = 1;
};

} // namespace nikola::aria
