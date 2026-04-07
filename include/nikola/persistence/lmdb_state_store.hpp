/**
 * @file persistence/lmdb_state_store.hpp
 * @brief Phase 137 — LMDB-backed full state persistence for Nikola.
 *
 * Provides cross-session persistence for:
 *   - NikolaState snapshots (metabolic/cognitive state per tick)
 *   - Ψ wavefunction checkpoints (full grid restore)
 *   - AutobiographicalMemory (events, skills, values)
 *
 * Uses a single LMDB environment with five named databases:
 *   "state"      — NikolaState keyed by tick
 *   "checkpoint"  — Ψ field keyed by tick
 *   "events"      — LifeEvents keyed by tick
 *   "skills"      — SkillLevel keyed by name
 *   "values"      — ValueEntry keyed by name
 *
 * Separate from the Phase 136 SemanticMemory LMDB store to avoid breaking
 * existing tests and to allow independent backup/migration.
 *
 * Schema: docs/architecture/memory_schema.md
 * Requires: lmdb.h / liblmdb.  Link with: target_link_libraries(... lmdb)
 *
 * Thread safety: NOT thread-safe — use one instance per thread or protect
 * externally (same contract as LmdbMemoryStore).
 */
#pragma once

#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/interior/autobiography.hpp>
#include <nikola/physics/wave_function.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <lmdb.h>

namespace nikola::persistence {

// ============================================================================
// Constants
// ============================================================================

inline constexpr std::size_t STATE_DB_MAP_SIZE  = 512ULL * 1024 * 1024;  // 512 MiB
inline constexpr int         STATE_DB_MAX_DBS   = 6;

inline constexpr uint32_t MAGIC_STATE      = 0x4E4B5354u;  // "NKST"
inline constexpr uint32_t MAGIC_CHECKPOINT = 0x4E434B50u;  // "NCKP"
inline constexpr uint32_t MAGIC_EVENT      = 0x4E455654u;  // "NEVT"
inline constexpr uint32_t STATE_VERSION    = 1u;

// ============================================================================
// Serialisation helpers
// ============================================================================

namespace detail {

inline void write_be64(uint64_t v, uint8_t out[8]) noexcept
{
    for (int i = 7; i >= 0; --i) {
        out[i] = static_cast<uint8_t>(v & 0xffu);
        v >>= 8;
    }
}

inline uint64_t read_be64(const uint8_t in[8]) noexcept
{
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v = (v << 8) | in[i];
    return v;
}

// --- NikolaState ---

inline std::vector<uint8_t> pack_state(const autonomy::NikolaState& s, uint64_t /*tick*/)
{
    // Calculate token sizes
    std::size_t tokens_size = 4;  // n_tokens (uint32)
    for (const auto& tok : s.tokens)
        tokens_size += 2 + tok.size();  // uint16 len + chars

    const std::size_t size = 37 + tokens_size;
    std::vector<uint8_t> buf(size);
    uint8_t* p = buf.data();

    std::memcpy(p, &MAGIC_STATE, 4);           p += 4;
    std::memcpy(p, &STATE_VERSION, 4);         p += 4;
    std::memcpy(p, &s.time, 4);               p += 4;
    std::memcpy(p, &s.torus_energy, 4);       p += 4;
    std::memcpy(p, &s.dopamine, 4);           p += 4;
    std::memcpy(p, &s.td_error, 4);           p += 4;
    std::memcpy(p, &s.atp, 4);               p += 4;
    std::memcpy(p, &s.boredom, 4);            p += 4;
    std::memcpy(p, &s.entropy, 4);            p += 4;
    *p = static_cast<uint8_t>(s.last_action);  p += 1;

    const uint32_t n_tokens = static_cast<uint32_t>(s.tokens.size());
    std::memcpy(p, &n_tokens, 4);              p += 4;
    for (const auto& tok : s.tokens) {
        const uint16_t len = static_cast<uint16_t>(tok.size());
        std::memcpy(p, &len, 2);               p += 2;
        std::memcpy(p, tok.data(), len);       p += len;
    }

    return buf;
}

inline bool unpack_state(const void* data, std::size_t size,
                         autonomy::NikolaState& out)
{
    if (size < 41) return false;  // minimum: 37 header + 4 (n_tokens=0)
    const uint8_t* p = static_cast<const uint8_t*>(data);

    uint32_t magic = 0, version = 0;
    std::memcpy(&magic, p, 4);   p += 4;
    std::memcpy(&version, p, 4); p += 4;
    if (magic != MAGIC_STATE || version > STATE_VERSION) return false;

    std::memcpy(&out.time, p, 4);          p += 4;
    std::memcpy(&out.torus_energy, p, 4);  p += 4;
    std::memcpy(&out.dopamine, p, 4);      p += 4;
    std::memcpy(&out.td_error, p, 4);      p += 4;
    std::memcpy(&out.atp, p, 4);           p += 4;
    std::memcpy(&out.boredom, p, 4);       p += 4;
    std::memcpy(&out.entropy, p, 4);       p += 4;
    out.last_action = static_cast<autonomy::ActionType>(*p); p += 1;

    uint32_t n_tokens = 0;
    std::memcpy(&n_tokens, p, 4);          p += 4;
    if (n_tokens > 1000) return false;  // sanity

    out.tokens.clear();
    out.tokens.reserve(n_tokens);
    const uint8_t* end = static_cast<const uint8_t*>(data) + size;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        if (p + 2 > end) return false;
        uint16_t len = 0;
        std::memcpy(&len, p, 2); p += 2;
        if (p + len > end) return false;
        out.tokens.emplace_back(reinterpret_cast<const char*>(p), len);
        p += len;
    }
    return true;
}

// --- Ψ checkpoint ---

inline std::vector<uint8_t> pack_checkpoint(const physics::WaveFunction& wf,
                                             uint64_t /*tick*/)
{
    const uint32_t n = static_cast<uint32_t>(wf.num_nodes());
    const uint32_t grid_n = static_cast<uint32_t>(wf.grid().grid_n());
    const float t = wf.time();
    const double H = wf.total_probability() + wf.total_kinetic_energy();
    const uint32_t reserved = 0;

    const std::size_t size = 28u + 4u * n * sizeof(float);
    std::vector<uint8_t> buf(size);
    uint8_t* p = buf.data();

    std::memcpy(p, &MAGIC_CHECKPOINT, 4);  p += 4;
    std::memcpy(p, &n, 4);                p += 4;
    std::memcpy(p, &t, 4);                p += 4;
    std::memcpy(p, &H, 8);                p += 8;
    std::memcpy(p, &grid_n, 4);           p += 4;
    std::memcpy(p, &reserved, 4);         p += 4;

    if (n > 0) {
        std::memcpy(p, wf.grid().psi_real(), n * sizeof(float)); p += n * sizeof(float);
        std::memcpy(p, wf.grid().psi_imag(), n * sizeof(float)); p += n * sizeof(float);
        std::memcpy(p, wf.grid().vel_real(), n * sizeof(float)); p += n * sizeof(float);
        std::memcpy(p, wf.grid().vel_imag(), n * sizeof(float));
    }
    return buf;
}

struct CheckpointHeader {
    uint32_t n_nodes    = 0;
    float    time_secs  = 0.f;
    double   hamiltonian= 0.0;
    uint32_t grid_n     = 3;
};

inline bool unpack_checkpoint(const void* data, std::size_t size,
                              CheckpointHeader& hdr,
                              physics::WaveFunction& wf)
{
    if (size < 28) return false;
    const uint8_t* p = static_cast<const uint8_t*>(data);

    uint32_t magic = 0;
    std::memcpy(&magic, p, 4); p += 4;
    if (magic != MAGIC_CHECKPOINT) return false;

    std::memcpy(&hdr.n_nodes, p, 4);      p += 4;
    std::memcpy(&hdr.time_secs, p, 4);    p += 4;
    std::memcpy(&hdr.hamiltonian, p, 8);   p += 8;
    std::memcpy(&hdr.grid_n, p, 4);       p += 4;
    p += 4;  // reserved

    const std::size_t expected = 28u + 4u * hdr.n_nodes * sizeof(float);
    if (size < expected || hdr.n_nodes > (1u << 20)) return false;

    // Allocate grid and populate
    wf.seed_manifold(static_cast<int>(hdr.grid_n), 0, 1, 0.f, 0);
    if (wf.num_nodes() != hdr.n_nodes) return false;

    if (hdr.n_nodes > 0) {
        std::memcpy(wf.grid().psi_real(), p, hdr.n_nodes * sizeof(float));
        p += hdr.n_nodes * sizeof(float);
        std::memcpy(wf.grid().psi_imag(), p, hdr.n_nodes * sizeof(float));
        p += hdr.n_nodes * sizeof(float);
        std::memcpy(wf.grid().vel_real(), p, hdr.n_nodes * sizeof(float));
        p += hdr.n_nodes * sizeof(float);
        std::memcpy(wf.grid().vel_imag(), p, hdr.n_nodes * sizeof(float));
    }
    return true;
}

// --- Autobiographical events ---

/// Pack a NikolaState inline (without magic/version header) for event records.
inline void pack_state_inline(const autonomy::NikolaState& s, std::vector<uint8_t>& buf)
{
    const std::size_t start = buf.size();
    // 29 bytes fixed + tokens
    buf.resize(start + 33);
    uint8_t* p = buf.data() + start;

    std::memcpy(p, &s.time, 4);               p += 4;
    std::memcpy(p, &s.torus_energy, 4);       p += 4;
    std::memcpy(p, &s.dopamine, 4);           p += 4;
    std::memcpy(p, &s.td_error, 4);           p += 4;
    std::memcpy(p, &s.atp, 4);               p += 4;
    std::memcpy(p, &s.boredom, 4);            p += 4;
    std::memcpy(p, &s.entropy, 4);            p += 4;
    *p = static_cast<uint8_t>(s.last_action);  p += 1;

    const uint32_t n_tokens = static_cast<uint32_t>(s.tokens.size());
    std::memcpy(p, &n_tokens, 4);

    for (const auto& tok : s.tokens) {
        const uint16_t len = static_cast<uint16_t>(tok.size());
        buf.resize(buf.size() + 2 + len);
        p = buf.data() + buf.size() - 2 - len;
        std::memcpy(p, &len, 2);
        std::memcpy(p + 2, tok.data(), len);
    }
}

inline bool unpack_state_inline(const uint8_t*& p, const uint8_t* end,
                                autonomy::NikolaState& out)
{
    if (p + 33 > end) return false;

    std::memcpy(&out.time, p, 4);          p += 4;
    std::memcpy(&out.torus_energy, p, 4);  p += 4;
    std::memcpy(&out.dopamine, p, 4);      p += 4;
    std::memcpy(&out.td_error, p, 4);      p += 4;
    std::memcpy(&out.atp, p, 4);           p += 4;
    std::memcpy(&out.boredom, p, 4);       p += 4;
    std::memcpy(&out.entropy, p, 4);       p += 4;
    out.last_action = static_cast<autonomy::ActionType>(*p); p += 1;

    uint32_t n_tokens = 0;
    std::memcpy(&n_tokens, p, 4); p += 4;
    if (n_tokens > 1000) return false;

    out.tokens.clear();
    for (uint32_t i = 0; i < n_tokens; ++i) {
        if (p + 2 > end) return false;
        uint16_t len = 0;
        std::memcpy(&len, p, 2); p += 2;
        if (p + len > end) return false;
        out.tokens.emplace_back(reinterpret_cast<const char*>(p), len);
        p += len;
    }
    return true;
}

inline std::vector<uint8_t> pack_event(const interior::LifeEvent& evt)
{
    std::vector<uint8_t> buf;
    buf.reserve(256);

    // Header: magic + significance + affect + n_tags + desc_len
    buf.resize(24);
    uint8_t* p = buf.data();

    std::memcpy(p, &MAGIC_EVENT, 4);                            p += 4;
    std::memcpy(p, &evt.significance, 8);                       p += 8;
    const int32_t affect = static_cast<int32_t>(evt.dominant_affect);
    std::memcpy(p, &affect, 4);                                 p += 4;
    const uint32_t n_tags = static_cast<uint32_t>(evt.tags.size());
    std::memcpy(p, &n_tags, 4);                                 p += 4;
    const uint32_t desc_len = static_cast<uint32_t>(evt.description.size());
    std::memcpy(p, &desc_len, 4);

    // Description
    buf.insert(buf.end(), evt.description.begin(), evt.description.end());

    // Tags
    for (const auto& tag : evt.tags) {
        const uint16_t tlen = static_cast<uint16_t>(tag.size());
        buf.resize(buf.size() + 2 + tlen);
        p = buf.data() + buf.size() - 2 - tlen;
        std::memcpy(p, &tlen, 2);
        std::memcpy(p + 2, tag.data(), tlen);
    }

    // Inline NikolaState
    pack_state_inline(evt.state, buf);

    return buf;
}

inline bool unpack_event(const void* data, std::size_t size,
                         uint64_t tick, interior::LifeEvent& out)
{
    if (size < 24) return false;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    const uint8_t* end = p + size;

    uint32_t magic = 0;
    std::memcpy(&magic, p, 4); p += 4;
    if (magic != MAGIC_EVENT) return false;

    out.tick = tick;
    std::memcpy(&out.significance, p, 8);   p += 8;
    int32_t affect = 0;
    std::memcpy(&affect, p, 4);             p += 4;
    out.dominant_affect = static_cast<interior::Affect>(affect);
    uint32_t n_tags = 0, desc_len = 0;
    std::memcpy(&n_tags, p, 4);             p += 4;
    std::memcpy(&desc_len, p, 4);           p += 4;

    if (n_tags > 1000 || desc_len > 100000) return false;
    if (p + desc_len > end) return false;
    out.description.assign(reinterpret_cast<const char*>(p), desc_len);
    p += desc_len;

    out.tags.clear();
    for (uint32_t i = 0; i < n_tags; ++i) {
        if (p + 2 > end) return false;
        uint16_t tlen = 0;
        std::memcpy(&tlen, p, 2); p += 2;
        if (p + tlen > end) return false;
        out.tags.emplace_back(reinterpret_cast<const char*>(p), tlen);
        p += tlen;
    }

    return unpack_state_inline(p, end, out.state);
}

// --- Skills / Values ---

inline std::vector<uint8_t> pack_skill(const interior::SkillLevel& s)
{
    std::vector<uint8_t> buf(32);
    uint8_t* p = buf.data();
    std::memcpy(p, &s.proficiency, 8);     p += 8;
    std::memcpy(p, &s.last_tick, 8);       p += 8;
    std::memcpy(p, &s.practice_count, 8);  p += 8;
    std::memcpy(p, &s.success_count, 8);
    return buf;
}

inline bool unpack_skill(const void* data, std::size_t size,
                         const std::string& name, interior::SkillLevel& out)
{
    if (size < 32) return false;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    out.skill_name = name;
    std::memcpy(&out.proficiency, p, 8);     p += 8;
    std::memcpy(&out.last_tick, p, 8);       p += 8;
    std::memcpy(&out.practice_count, p, 8);  p += 8;
    std::memcpy(&out.success_count, p, 8);
    return true;
}

inline std::vector<uint8_t> pack_value(const interior::ValueEntry& v)
{
    std::vector<uint8_t> buf(16);
    uint8_t* p = buf.data();
    std::memcpy(p, &v.importance, 8);     p += 8;
    std::memcpy(p, &v.update_count, 8);
    return buf;
}

inline bool unpack_value(const void* data, std::size_t size,
                         const std::string& name, interior::ValueEntry& out)
{
    if (size < 16) return false;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    out.value_name = name;
    std::memcpy(&out.importance, p, 8);    p += 8;
    std::memcpy(&out.update_count, p, 8);
    return true;
}

}  // namespace detail

// ============================================================================
// LmdbStateStore
// ============================================================================

/**
 * @brief RAII LMDB environment for full Nikola state persistence.
 *
 * Opens (or creates) an LMDB environment at the given directory with five
 * named databases: state, checkpoint, events, skills, values.
 *
 * Typical usage:
 * @code
 *   LmdbStateStore store("/home/nikola/.nikola_state.lmdb");
 *
 *   // Save state + checkpoint
 *   store.save_state(state, tick);
 *   store.save_checkpoint(wf, tick);
 *
 *   // Save autobiography
 *   store.save_autobiography(auto_memory);
 *
 *   // Restore on next session
 *   store.load_latest_state(state, tick);
 *   store.load_latest_checkpoint(wf);
 *   store.load_autobiography(auto_memory);
 * @endcode
 */
class LmdbStateStore {
public:
    std::size_t map_size = STATE_DB_MAP_SIZE;

    explicit LmdbStateStore(const std::string& path)
        : path_(path)
    {
        std::filesystem::create_directories(path);
        open_env();
    }

    ~LmdbStateStore() { close_env(); }

    LmdbStateStore(const LmdbStateStore&)            = delete;
    LmdbStateStore& operator=(const LmdbStateStore&) = delete;
    LmdbStateStore(LmdbStateStore&&)                 = default;
    LmdbStateStore& operator=(LmdbStateStore&&)      = default;

    // ------------------------------------------------------------------
    // NikolaState persistence
    // ------------------------------------------------------------------

    /**
     * @brief Save a NikolaState snapshot at the given tick.
     */
    void save_state(const autonomy::NikolaState& state, uint64_t tick)
    {
        auto buf = detail::pack_state(state, tick);
        put_record("state", tick, buf.data(), buf.size());
    }

    /**
     * @brief Load the most recent NikolaState (highest tick key).
     * @returns true if a state was loaded, false if database is empty.
     */
    [[nodiscard]] bool load_latest_state(autonomy::NikolaState& state,
                                          uint64_t& tick_out)
    {
        MDB_val mkey{}, mval{};
        if (!get_last_record("state", mkey, mval, tick_out))
            return false;
        return detail::unpack_state(mval.mv_data, mval.mv_size, state);
    }

    /**
     * @brief Count of saved state records.
     */
    [[nodiscard]] std::size_t state_count()
    {
        return db_count("state");
    }

    // ------------------------------------------------------------------
    // Ψ wavefunction checkpoints
    // ------------------------------------------------------------------

    /**
     * @brief Save a wavefunction checkpoint.
     */
    void save_checkpoint(const physics::WaveFunction& wf, uint64_t tick)
    {
        auto buf = detail::pack_checkpoint(wf, tick);
        put_record("checkpoint", tick, buf.data(), buf.size());
    }

    /**
     * @brief Load the most recent Ψ checkpoint.
     * @param[out] hdr  Checkpoint header with stored Hamiltonian for verification.
     * @returns true if a checkpoint was loaded.
     */
    [[nodiscard]] bool load_latest_checkpoint(physics::WaveFunction& wf,
                                               detail::CheckpointHeader& hdr)
    {
        uint64_t tick = 0;
        MDB_val mkey{}, mval{};
        if (!get_last_record("checkpoint", mkey, mval, tick))
            return false;
        return detail::unpack_checkpoint(mval.mv_data, mval.mv_size, hdr, wf);
    }

    /**
     * @brief Count of saved checkpoints.
     */
    [[nodiscard]] std::size_t checkpoint_count()
    {
        return db_count("checkpoint");
    }

    // ------------------------------------------------------------------
    // Autobiographical memory persistence
    // ------------------------------------------------------------------

    /**
     * @brief Save all events, skills, and values from an AutobiographicalMemory.
     *
     * Overwrites existing records. All writes in one ACID transaction.
     */
    void save_autobiography(const interior::AutobiographicalMemory& mem)
    {
        MDB_txn* txn = nullptr;
        check(mdb_txn_begin(env_, nullptr, 0, &txn), "txn_begin(save_auto)");

        // Events
        {
            MDB_dbi dbi = open_named_dbi(txn, "events");
            for (const auto& evt : mem.events()) {
                uint8_t kbuf[8];
                detail::write_be64(evt.tick, kbuf);
                MDB_val mkey{8, kbuf};
                auto vbuf = detail::pack_event(evt);
                MDB_val mval{vbuf.size(), vbuf.data()};
                if (int rc = mdb_put(txn, dbi, &mkey, &mval, 0); rc != MDB_SUCCESS) {
                    mdb_txn_abort(txn);
                    throw std::runtime_error(std::string("mdb_put(events): ") + mdb_strerror(rc));
                }
            }
        }

        // Skills
        {
            MDB_dbi dbi = open_named_dbi(txn, "skills");
            // Clear old skills first (in case some were removed)
            mdb_drop(txn, dbi, 0);  // 0 = empty, don't delete
            for (const auto& skill : mem.get_skills()) {
                MDB_val mkey{skill.skill_name.size(),
                             const_cast<char*>(skill.skill_name.data())};
                auto vbuf = detail::pack_skill(skill);
                MDB_val mval{vbuf.size(), vbuf.data()};
                if (int rc = mdb_put(txn, dbi, &mkey, &mval, 0); rc != MDB_SUCCESS) {
                    mdb_txn_abort(txn);
                    throw std::runtime_error(std::string("mdb_put(skills): ") + mdb_strerror(rc));
                }
            }
        }

        // Values
        {
            MDB_dbi dbi = open_named_dbi(txn, "values");
            mdb_drop(txn, dbi, 0);
            for (const auto& val : mem.value_entries()) {
                MDB_val mkey{val.value_name.size(),
                             const_cast<char*>(val.value_name.data())};
                auto vbuf = detail::pack_value(val);
                MDB_val mval{vbuf.size(), vbuf.data()};
                if (int rc = mdb_put(txn, dbi, &mkey, &mval, 0); rc != MDB_SUCCESS) {
                    mdb_txn_abort(txn);
                    throw std::runtime_error(std::string("mdb_put(values): ") + mdb_strerror(rc));
                }
            }
        }

        check(mdb_txn_commit(txn), "txn_commit(save_auto)");
    }

    /**
     * @brief Load all events, skills, and values into an AutobiographicalMemory.
     * @returns Total number of records loaded (events + skills + values).
     */
    [[nodiscard]] std::size_t load_autobiography(interior::AutobiographicalMemory& mem)
    {
        std::size_t loaded = 0;

        MDB_txn* txn = nullptr;
        check(mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn), "txn_begin(load_auto)");

        // Events
        {
            MDB_dbi dbi = 0;
            if (mdb_dbi_open(txn, "events", 0, &dbi) == MDB_SUCCESS) {
                MDB_cursor* cur = nullptr;
                if (mdb_cursor_open(txn, dbi, &cur) == MDB_SUCCESS) {
                    MDB_val mkey{}, mval{};
                    while (mdb_cursor_get(cur, &mkey, &mval, MDB_NEXT) == MDB_SUCCESS) {
                        if (mkey.mv_size != 8) continue;
                        uint64_t tick = detail::read_be64(
                            static_cast<const uint8_t*>(mkey.mv_data));
                        interior::LifeEvent evt;
                        if (detail::unpack_event(mval.mv_data, mval.mv_size, tick, evt)) {
                            mem.record_event(evt.description, evt.state,
                                             evt.dominant_affect, evt.significance,
                                             evt.tags);
                            ++loaded;
                        }
                    }
                    mdb_cursor_close(cur);
                }
            }
        }

        // Skills
        {
            MDB_dbi dbi = 0;
            if (mdb_dbi_open(txn, "skills", 0, &dbi) == MDB_SUCCESS) {
                MDB_cursor* cur = nullptr;
                if (mdb_cursor_open(txn, dbi, &cur) == MDB_SUCCESS) {
                    MDB_val mkey{}, mval{};
                    while (mdb_cursor_get(cur, &mkey, &mval, MDB_NEXT) == MDB_SUCCESS) {
                        std::string name(static_cast<const char*>(mkey.mv_data),
                                         mkey.mv_size);
                        interior::SkillLevel skill;
                        if (detail::unpack_skill(mval.mv_data, mval.mv_size, name, skill)) {
                            // Restore: update_skill can't set arbitrary proficiency,
                            // so we simulate practice history
                            for (uint64_t i = 0; i < skill.success_count; ++i)
                                mem.update_skill(name, true, skill.last_tick);
                            for (uint64_t i = 0; i < (skill.practice_count - skill.success_count); ++i)
                                mem.update_skill(name, false, skill.last_tick);
                            ++loaded;
                        }
                    }
                    mdb_cursor_close(cur);
                }
            }
        }

        // Values
        {
            MDB_dbi dbi = 0;
            if (mdb_dbi_open(txn, "values", 0, &dbi) == MDB_SUCCESS) {
                MDB_cursor* cur = nullptr;
                if (mdb_cursor_open(txn, dbi, &cur) == MDB_SUCCESS) {
                    MDB_val mkey{}, mval{};
                    while (mdb_cursor_get(cur, &mkey, &mval, MDB_NEXT) == MDB_SUCCESS) {
                        std::string name(static_cast<const char*>(mkey.mv_data),
                                         mkey.mv_size);
                        interior::ValueEntry val;
                        if (detail::unpack_value(mval.mv_data, mval.mv_size, name, val)) {
                            // Restore via repeated update_value calls
                            // Each call adds delta * LEARN_RATE; we need to get to val.importance
                            // from default 0.5. delta = (importance - 0.5) / LEARN_RATE
                            double delta = (val.importance - 0.5) / interior::AUTOBIOGRAPHY_VALUE_LEARN_RATE;
                            mem.update_value(name, delta);
                            ++loaded;
                        }
                    }
                    mdb_cursor_close(cur);
                }
            }
        }

        mdb_txn_abort(txn);  // read-only
        return loaded;
    }

    /**
     * @brief Save a single autobiography event immediately.
     *
     * Called in real-time as events are recorded (via the on_event callback).
     */
    void save_event(const interior::LifeEvent& evt)
    {
        auto buf = detail::pack_event(evt);
        put_record("events", evt.tick, buf.data(), buf.size());
    }

    // ------------------------------------------------------------------
    // State dump (for --state-dump CLI)
    // ------------------------------------------------------------------

    /**
     * @brief Return a human-readable summary of the latest saved state.
     */
    [[nodiscard]] std::string dump_latest() const
    {
        std::string result;
        MDB_txn* txn = nullptr;
        if (mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn) != MDB_SUCCESS)
            return "(failed to open transaction)\n";

        // State
        {
            MDB_dbi dbi = 0;
            if (mdb_dbi_open(txn, "state", 0, &dbi) == MDB_SUCCESS) {
                MDB_cursor* cur = nullptr;
                if (mdb_cursor_open(txn, dbi, &cur) == MDB_SUCCESS) {
                    MDB_val mkey{}, mval{};
                    if (mdb_cursor_get(cur, &mkey, &mval, MDB_LAST) == MDB_SUCCESS) {
                        uint64_t tick = detail::read_be64(
                            static_cast<const uint8_t*>(mkey.mv_data));
                        autonomy::NikolaState s;
                        if (detail::unpack_state(mval.mv_data, mval.mv_size, s)) {
                            result += "=== Latest NikolaState (tick " + std::to_string(tick) + ") ===\n";
                            result += "  time:         " + std::to_string(s.time) + "\n";
                            result += "  torus_energy: " + std::to_string(s.torus_energy) + "\n";
                            result += "  dopamine:     " + std::to_string(s.dopamine) + "\n";
                            result += "  td_error:     " + std::to_string(s.td_error) + "\n";
                            result += "  atp:          " + std::to_string(s.atp) + "\n";
                            result += "  boredom:      " + std::to_string(s.boredom) + "\n";
                            result += "  entropy:      " + std::to_string(s.entropy) + "\n";
                            result += "  last_action:  " + std::string(autonomy::action_name(s.last_action)) + "\n";
                            result += "  tokens:       " + std::to_string(s.tokens.size()) + "\n";
                        }
                    }
                    mdb_cursor_close(cur);
                }
            }
        }

        // Checkpoint count
        {
            MDB_dbi dbi = 0;
            if (mdb_dbi_open(txn, "checkpoint", 0, &dbi) == MDB_SUCCESS) {
                MDB_stat stat{};
                mdb_stat(txn, dbi, &stat);
                result += "\n=== Checkpoints: " + std::to_string(stat.ms_entries) + " ===\n";
            }
        }

        // Events count
        {
            MDB_dbi dbi = 0;
            if (mdb_dbi_open(txn, "events", 0, &dbi) == MDB_SUCCESS) {
                MDB_stat stat{};
                mdb_stat(txn, dbi, &stat);
                result += "=== Autobiography: " + std::to_string(stat.ms_entries) + " events ===\n";
            }
        }

        mdb_txn_abort(txn);
        return result;
    }

private:
    std::string path_;
    MDB_env*    env_{nullptr};

    void open_env()
    {
        check(mdb_env_create(&env_), "env_create");
        check(mdb_env_set_mapsize(env_, map_size), "env_set_mapsize");
        check(mdb_env_set_maxdbs(env_, STATE_DB_MAX_DBS), "env_set_maxdbs");
        if (int rc = mdb_env_open(env_, path_.c_str(), 0, 0664); rc != MDB_SUCCESS) {
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

    MDB_dbi open_named_dbi(MDB_txn* txn, const char* name)
    {
        MDB_dbi dbi = 0;
        if (int rc = mdb_dbi_open(txn, name, MDB_CREATE, &dbi); rc != MDB_SUCCESS) {
            mdb_txn_abort(txn);
            throw std::runtime_error(std::string("mdb_dbi_open('") + name + "'): " + mdb_strerror(rc));
        }
        return dbi;
    }

    /// Write a record to a named database, keyed by tick (big-endian uint64).
    void put_record(const char* db_name, uint64_t tick,
                    const void* data, std::size_t size)
    {
        MDB_txn* txn = nullptr;
        check(mdb_txn_begin(env_, nullptr, 0, &txn), "txn_begin(put)");
        MDB_dbi dbi = open_named_dbi(txn, db_name);

        uint8_t kbuf[8];
        detail::write_be64(tick, kbuf);
        MDB_val mkey{8, kbuf};
        MDB_val mval{size, const_cast<void*>(data)};

        if (int rc = mdb_put(txn, dbi, &mkey, &mval, 0); rc != MDB_SUCCESS) {
            mdb_txn_abort(txn);
            throw std::runtime_error(std::string("mdb_put(") + db_name + "): " + mdb_strerror(rc));
        }
        check(mdb_txn_commit(txn), "txn_commit(put)");
    }

    /// Get the last (highest key) record from a named database.
    /// Returns false if database is empty or doesn't exist.
    /// NOTE: the returned MDB_val pointers are only valid within the
    /// read transaction — caller must copy before txn ends.
    /// This opens a read txn that is aborted internally, so the caller
    /// must unpack the data from the returned copy within this call.
    bool get_last_record(const char* db_name, MDB_val& mkey_out,
                         MDB_val& mval_out, uint64_t& tick_out)
    {
        MDB_txn* txn = nullptr;
        check(mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn), "txn_begin(get_last)");

        MDB_dbi dbi = 0;
        if (mdb_dbi_open(txn, db_name, 0, &dbi) != MDB_SUCCESS) {
            mdb_txn_abort(txn);
            return false;
        }

        MDB_cursor* cur = nullptr;
        if (mdb_cursor_open(txn, dbi, &cur) != MDB_SUCCESS) {
            mdb_txn_abort(txn);
            return false;
        }

        MDB_val mk{}, mv{};
        if (mdb_cursor_get(cur, &mk, &mv, MDB_LAST) != MDB_SUCCESS) {
            mdb_cursor_close(cur);
            mdb_txn_abort(txn);
            return false;
        }

        // Copy the data out before closing the transaction
        last_key_buf_.assign(static_cast<uint8_t*>(mk.mv_data),
                             static_cast<uint8_t*>(mk.mv_data) + mk.mv_size);
        last_val_buf_.assign(static_cast<uint8_t*>(mv.mv_data),
                             static_cast<uint8_t*>(mv.mv_data) + mv.mv_size);

        mdb_cursor_close(cur);
        mdb_txn_abort(txn);

        if (last_key_buf_.size() == 8)
            tick_out = detail::read_be64(last_key_buf_.data());

        mkey_out = {last_key_buf_.size(), last_key_buf_.data()};
        mval_out = {last_val_buf_.size(), last_val_buf_.data()};
        return true;
    }

    /// Count records in a named database.
    std::size_t db_count(const char* db_name)
    {
        MDB_txn* txn = nullptr;
        check(mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn), "txn_begin(count)");
        MDB_dbi dbi = 0;
        if (mdb_dbi_open(txn, db_name, 0, &dbi) != MDB_SUCCESS) {
            mdb_txn_abort(txn);
            return 0;
        }
        MDB_stat stat{};
        mdb_stat(txn, dbi, &stat);
        mdb_txn_abort(txn);
        return static_cast<std::size_t>(stat.ms_entries);
    }

    static void check(int rc, const char* ctx)
    {
        if (rc != MDB_SUCCESS)
            throw std::runtime_error(std::string(ctx) + ": " + mdb_strerror(rc));
    }

    // Buffers for get_last_record (avoids dangling LMDB pointers)
    std::vector<uint8_t> last_key_buf_;
    std::vector<uint8_t> last_val_buf_;
};

// ============================================================================
// Convenience free functions
// ============================================================================

/**
 * @brief Save NikolaState + optional Ψ checkpoint to LMDB.
 */
inline void save_state_lmdb(const autonomy::NikolaState& state,
                             const physics::WaveFunction& wf,
                             uint64_t tick,
                             const std::string& path,
                             bool save_checkpoint = false)
{
    LmdbStateStore store(path);
    store.save_state(state, tick);
    if (save_checkpoint)
        store.save_checkpoint(wf, tick);
}

/**
 * @brief Load latest NikolaState from LMDB.
 * @returns true if state was loaded.
 */
[[nodiscard]] inline bool load_state_lmdb(autonomy::NikolaState& state,
                                           uint64_t& tick,
                                           const std::string& path)
{
    if (!std::filesystem::exists(path)) return false;
    LmdbStateStore store(path);
    return store.load_latest_state(state, tick);
}

}  // namespace nikola::persistence
