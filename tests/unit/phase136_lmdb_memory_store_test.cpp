// =============================================================================
// NIKOLA — Phase 136 — LMDB-backed SemanticMemory persistence test suite
// =============================================================================
// Tests the LmdbMemoryStore class and save_lmdb / load_lmdb free functions
// introduced in include/nikola/cognitive/lmdb_memory_store.hpp.
//
//   §1  save_lmdb / load_lmdb round-trip: all record fields preserved exactly
//   §2  Multiple records survive round-trip
//   §3  load_lmdb returns 0 when path does not exist (graceful first-run)
//   §4  save_lmdb creates the LMDB directory if absent
//   §5  save_lmdb overwrites existing records (no stale data after re-save)
//   §6  psi_real / psi_imag vectors match byte-for-byte after round-trip
//   §7  LmdbMemoryStore::db_size() reports correct record count
//   §8  LmdbMemoryStore::upsert() — single-record incremental write
//   §9  LmdbMemoryStore::erase() — record removed from DB
//   §10 Cross-session: second SemanticMemory reads data written by first
//   §11 Empty memory save/load produces 0-record DB
//   §12 Binary backward-compat: legacy .bin file still loads via save()/load()
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <nikola/cognitive/semantic_memory.hpp>
#include <nikola/cognitive/lmdb_memory_store.hpp>
#include <nikola/physics/wave_function.hpp>

namespace fs = std::filesystem;
using namespace nikola::cognitive;
using namespace nikola::physics;
using Catch::Approx;

// ============================================================================
// Helpers
// ============================================================================

/// Build a temporary LMDB path that is cleaned up after each test.
struct TmpLmdb {
    fs::path dir;
    TmpLmdb() {
        dir = fs::temp_directory_path() / ("nikola_phase136_" + std::to_string(
            std::hash<std::thread::id>{}(std::this_thread::get_id()) ^
            static_cast<size_t>(std::chrono::steady_clock::now().time_since_epoch().count())
        ));
        fs::remove_all(dir);
    }
    ~TmpLmdb() { fs::remove_all(dir); }
    std::string str() const { return dir.string(); }
};

/// Seed a WaveFunction on a small 2^9 = 512-node grid.
static WaveFunction make_wf(float amplitude = 0.3f, uint32_t seed = 42)
{
    WaveFunction wf;
    wf.seed_manifold(2, 3, 1, amplitude, seed);
    wf.grid().precompute_adjacency();
    return wf;
}

/// Store a wavefunction into SemanticMemory and return the assigned key.
static MemoryKey store_wf(SemanticMemory& mem, WaveFunction& wf)
{
    return mem.store(wf);
}

/// Build a synthetic MemoryRecord with an explicit key for multi-record tests.
/// Avoids the risk of wave-function peak collisions producing duplicate Hilbert keys.
static MemoryRecord make_rec(MemoryKey key, float strength = 0.5f,
                             float age = 0.0f, uint32_t access = 0u)
{
    MemoryRecord r;
    r.key          = key;
    r.strength     = strength;
    r.age_seconds  = age;
    r.access_count = access;
    r.psi_real     = {0.1f, 0.2f, 0.3f};
    r.psi_imag     = {0.4f, 0.5f, 0.6f};
    return r;
}

// ============================================================================
// §1  Round-trip: all fields preserved
// ============================================================================

TEST_CASE("Phase136: save_lmdb/load_lmdb round-trip preserves all fields",
          "[phase136][lmdb][persistence]")
{
    TmpLmdb tmp;
    SemanticMemory src;
    auto wf = make_wf(0.5f, 7);
    const MemoryKey key = store_wf(src, wf);

    // Manually set metadata via insert_record
    MemoryRecord meta = *src.get(key);
    meta.strength    = 0.75f;
    meta.age_seconds = 42.5f;
    meta.access_count = 3u;
    src.insert_record(std::move(meta));

    save_lmdb(src, tmp.str());

    SemanticMemory dst;
    const size_t n = load_lmdb(dst, tmp.str());
    REQUIRE(n == 1u);

    const MemoryRecord* rec = dst.get(key);
    REQUIRE(rec != nullptr);
    REQUIRE(rec->key          == key);
    REQUIRE(rec->strength     == Approx(0.75f));
    REQUIRE(rec->age_seconds  == Approx(42.5f));
    REQUIRE(rec->access_count == 3u);
}

// ============================================================================
// §2  Multiple records survive round-trip
// ============================================================================

TEST_CASE("Phase136: multiple records round-trip correctly",
          "[phase136][lmdb][persistence]")
{
    TmpLmdb tmp;
    SemanticMemory src;
    std::vector<MemoryKey> keys;

    for (int i = 0; i < 5; ++i) {
        const MemoryKey k = static_cast<MemoryKey>(1000u + i);
        src.insert_record(make_rec(k, 0.1f * (i + 1)));
        keys.push_back(k);
    }

    save_lmdb(src, tmp.str());

    SemanticMemory dst;
    const size_t n = load_lmdb(dst, tmp.str());
    REQUIRE(n == keys.size());

    for (MemoryKey k : keys) {
        REQUIRE(dst.contains(k));
    }
}

// ============================================================================
// §3  Graceful first-run: absent path → 0 records
// ============================================================================

TEST_CASE("Phase136: load_lmdb returns 0 for nonexistent path",
          "[phase136][lmdb][persistence]")
{
    SemanticMemory mem;
    const size_t n = load_lmdb(mem, "/tmp/this_path_cannot_exist_phase136_test");
    REQUIRE(n == 0u);
    REQUIRE(mem.size() == 0u);
}

// ============================================================================
// §4  save_lmdb creates the directory automatically
// ============================================================================

TEST_CASE("Phase136: save_lmdb creates LMDB directory if absent",
          "[phase136][lmdb][persistence]")
{
    TmpLmdb tmp;
    REQUIRE_FALSE(fs::exists(tmp.dir));

    SemanticMemory src;
    auto wf = make_wf();
    store_wf(src, wf);

    REQUIRE_NOTHROW(save_lmdb(src, tmp.str()));
    REQUIRE(fs::exists(tmp.dir));
    REQUIRE(fs::exists(tmp.dir / "data.mdb"));
}

// ============================================================================
// §5  Stale data: re-save replaces old records
// ============================================================================

TEST_CASE("Phase136: save_lmdb overwrites stale records on re-save",
          "[phase136][lmdb][persistence]")
{
    TmpLmdb tmp;
    SemanticMemory src;
    auto wf = make_wf(0.4f, 11);
    const MemoryKey key = store_wf(src, wf);

    // First save
    save_lmdb(src, tmp.str());

    // Mutate strength via insert_record and re-save
    MemoryRecord upd = *src.get(key);
    upd.strength = 0.99f;
    src.insert_record(std::move(upd));
    save_lmdb(src, tmp.str());

    SemanticMemory dst;
    load_lmdb(dst, tmp.str());
    REQUIRE(dst.get(key) != nullptr);
    REQUIRE(dst.get(key)->strength == Approx(0.99f));
}

// ============================================================================
// §6  psi vectors match byte-for-byte
// ============================================================================

TEST_CASE("Phase136: psi_real and psi_imag vectors match after round-trip",
          "[phase136][lmdb][persistence]")
{
    TmpLmdb tmp;
    SemanticMemory src;
    auto wf = make_wf(0.6f, 99);
    const MemoryKey key = store_wf(src, wf);

    const std::vector<float> orig_real = src.get(key)->psi_real;
    const std::vector<float> orig_imag = src.get(key)->psi_imag;

    save_lmdb(src, tmp.str());

    SemanticMemory dst;
    load_lmdb(dst, tmp.str());

    const MemoryRecord* rec = dst.get(key);
    REQUIRE(rec != nullptr);
    REQUIRE(rec->psi_real.size() == orig_real.size());
    REQUIRE(rec->psi_imag.size() == orig_imag.size());
    for (size_t i = 0; i < orig_real.size(); ++i) {
        REQUIRE(rec->psi_real[i] == orig_real[i]);
        REQUIRE(rec->psi_imag[i] == orig_imag[i]);
    }
}

// ============================================================================
// §7  LmdbMemoryStore::db_size()
// ============================================================================

TEST_CASE("Phase136: LmdbMemoryStore::db_size() matches record count",
          "[phase136][lmdb][persistence]")
{
    TmpLmdb tmp;
    SemanticMemory src;
    for (int i = 0; i < 3; ++i) {
        src.insert_record(make_rec(static_cast<MemoryKey>(2000u + i), 0.2f * (i + 1)));
    }

    {
        LmdbMemoryStore store(tmp.str());
        store.save_all(src);
        REQUIRE(store.db_size() == 3u);
    }
}

// ============================================================================
// §8  LmdbMemoryStore::upsert() — incremental single-record write
// ============================================================================

TEST_CASE("Phase136: LmdbMemoryStore::upsert() adds and updates a record",
          "[phase136][lmdb][persistence]")
{
    TmpLmdb tmp;
    SemanticMemory src;
    auto wf = make_wf(0.3f, 55);
    const MemoryKey key = store_wf(src, wf);
    {
        LmdbMemoryStore store(tmp.str());
        store.save_all(src);
        REQUIRE(store.db_size() == 1u);

        // Update via upsert (make a copy since src.get returns const*)
        MemoryRecord updated = *src.get(key);
        updated.strength = 0.55f;
        store.upsert(updated);
        REQUIRE(store.db_size() == 1u);  // still one record, just updated
    }

    SemanticMemory dst;
    load_lmdb(dst, tmp.str());
    REQUIRE(dst.get(key)->strength == Approx(0.55f));
}

// ============================================================================
// §9  LmdbMemoryStore::erase()
// ============================================================================

TEST_CASE("Phase136: LmdbMemoryStore::erase() removes a single record",
          "[phase136][lmdb][persistence]")
{
    TmpLmdb tmp;
    SemanticMemory src;
    std::vector<MemoryKey> keys;
    for (int i = 0; i < 3; ++i) {
        const MemoryKey k = static_cast<MemoryKey>(3000u + i);
        src.insert_record(make_rec(k, 0.3f));
        keys.push_back(k);
    }

    {
        LmdbMemoryStore store(tmp.str());
        store.save_all(src);
        REQUIRE(store.db_size() == 3u);
        store.erase(keys[1]);
        REQUIRE(store.db_size() == 2u);
    }

    SemanticMemory dst;
    load_lmdb(dst, tmp.str());
    REQUIRE(dst.size() == 2u);
    REQUIRE_FALSE(dst.contains(keys[1]));
    REQUIRE(dst.contains(keys[0]));
    REQUIRE(dst.contains(keys[2]));
}

// ============================================================================
// §10 Cross-session: second SemanticMemory reads what first wrote
// ============================================================================

TEST_CASE("Phase136: cross-session persistence — second instance reads first",
          "[phase136][lmdb][persistence]")
{
    TmpLmdb tmp;

    // Session 1: write
    MemoryKey key;
    float saved_strength;
    {
        SemanticMemory session1;
        auto wf = make_wf(0.7f, 13);
        key = store_wf(session1, wf);
        MemoryRecord upd = *session1.get(key);
        upd.strength = 0.88f;
        saved_strength = upd.strength;
        session1.insert_record(std::move(upd));
        save_lmdb(session1, tmp.str());
    }

    // Session 2: read
    {
        SemanticMemory session2;
        const size_t n = load_lmdb(session2, tmp.str());
        REQUIRE(n >= 1u);
        const MemoryRecord* rec = session2.get(key);
        REQUIRE(rec != nullptr);
        REQUIRE(rec->strength == Approx(saved_strength));
    }
}

// ============================================================================
// §11 Empty memory: save/load produces 0-record DB
// ============================================================================

TEST_CASE("Phase136: empty SemanticMemory save then load gives 0 records",
          "[phase136][lmdb][persistence]")
{
    TmpLmdb tmp;
    SemanticMemory empty;
    save_lmdb(empty, tmp.str());

    SemanticMemory dst;
    const size_t n = load_lmdb(dst, tmp.str());
    REQUIRE(n == 0u);
    REQUIRE(dst.size() == 0u);
}

// ============================================================================
// §12 Binary backward-compat: legacy .bin file still works via save()/load()
// ============================================================================

TEST_CASE("Phase136: legacy binary save/load still works alongside LMDB",
          "[phase136][lmdb][persistence]")
{
    const std::string bin_path = (fs::temp_directory_path() / "phase136_compat_test.bin").string();
    {
        SemanticMemory src;
        auto wf = make_wf(0.4f, 77);
        store_wf(src, wf);
        src.save(bin_path);
    }

    SemanticMemory dst;
    const size_t n = dst.load(bin_path);
    REQUIRE(n == 1u);

    std::remove(bin_path.c_str());
}
