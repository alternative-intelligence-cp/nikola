// ============================================================================
// phase109_memory_persistence_test.cpp
// Phase 109 — Cross-Session Memory Continuity
// ============================================================================
//
// Tests:
//   §1  SemanticMemory::save() / load() roundtrip — all fields preserved
//   §2  save() / load() preserves multiple records
//   §3  load() returns 0 when file does not exist (graceful first-run)
//   §4  save() overwrites existing file (no stale data)
//   §5  loaded records survive consolidate() when strength is above threshold
//   §6  DecisionLoop with memory_path loads records at construction
//   §7  DecisionLoop::memory() count matches snapshot record count
//   §8  psi field vectors match exactly after save/load
//   §9  Cross-session: second DecisionLoop reads snapshot written by first
//   §10 Empty-memory save / load produces 0-record snapshot
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cstdio>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "nikola/cognitive/semantic_memory.hpp"
#include "nikola/cognitive/cognitive_torus.hpp"
#include "nikola/autonomy/autonomy_engine.hpp"
#include "nikola/autonomy/decision_loop.hpp"

namespace fs = std::filesystem;
using namespace nikola::cognitive;
using namespace nikola::autonomy;
using Approx = Catch::Approx;

static const std::string MEM_PATH = "/tmp/nikola_phase109_mem_test.bin";
static void cleanup() { std::remove(MEM_PATH.c_str()); }

static CognitiveTorus make_torus(const std::string& seed = "hello nikola")
{
    CognitiveTorus t(3);
    t.inject_text(seed, 0.0);
    t.run(5, t.safe_dt());
    return t;
}

static DecisionLoopConfig make_cfg(const std::string& mem = MEM_PATH)
{
    DecisionLoopConfig cfg;
    cfg.steps_per_tick       = 5;
    cfg.action_threshold     = 0.0f;
    cfg.min_store_interval_s = 0.0f;
    cfg.min_emit_interval_s  = 0.0f;
    cfg.vocabulary           = {"hello", "nikola", "memory", "wave"};
    cfg.memory_path          = mem;
    return cfg;
}

// §1 ─────────────────────────────────────────────────────────────────────────

TEST_CASE("SemanticMemory: save/load roundtrip — single record",
          "[phase109][memory][persistence]")
{
    cleanup();
    CognitiveTorus t = make_torus("hello nikola");
    SemanticMemory mem_a;
    const MemoryKey key = mem_a.store(t.wave_function());
    mem_a.superpose(key, 1.0f, t.wave_function());  // bump access_count
    REQUIRE(mem_a.size() == 1);

    const MemoryRecord* orig = mem_a.get(key);
    REQUIRE(orig != nullptr);
    const float orig_strength    = orig->strength;
    const float orig_age         = orig->age_seconds;
    const uint32_t orig_access   = orig->access_count;
    const std::vector<float> orig_real = orig->psi_real;
    const std::vector<float> orig_imag = orig->psi_imag;

    mem_a.save(MEM_PATH);
    REQUIRE(fs::exists(MEM_PATH));

    SemanticMemory mem_b;
    const size_t loaded = mem_b.load(MEM_PATH);
    REQUIRE(loaded == 1);
    const MemoryRecord* rec = mem_b.get(key);
    REQUIRE(rec != nullptr);

    CHECK(rec->key          == key);
    CHECK(rec->strength     == Approx(orig_strength).epsilon(1e-6f));
    CHECK(rec->age_seconds  == Approx(orig_age).epsilon(1e-6f));
    CHECK(rec->access_count == orig_access);
    REQUIRE(rec->psi_real.size() == orig_real.size());
    for (size_t i = 0; i < orig_real.size(); ++i)
        CHECK(rec->psi_real[i] == Approx(orig_real[i]).epsilon(1e-6f));
    for (size_t i = 0; i < orig_imag.size(); ++i)
        CHECK(rec->psi_imag[i] == Approx(orig_imag[i]).epsilon(1e-6f));
    cleanup();
}

// §2 ─────────────────────────────────────────────────────────────────────────

TEST_CASE("SemanticMemory: save/load roundtrip — multiple records",
          "[phase109][memory][persistence]")
{
    cleanup();
    SemanticMemory mem_a;
    for (const auto& s : {"alpha", "beta", "gamma", "delta"}) {
        CognitiveTorus t = make_torus(s); mem_a.store(t.wave_function());
    }
    const size_t unique = mem_a.size();
    REQUIRE(unique >= 1);
    mem_a.save(MEM_PATH);

    SemanticMemory mem_b;
    REQUIRE(mem_b.load(MEM_PATH) == unique);
    for (const auto& key : mem_a.all_keys()) {
        const MemoryRecord* b = mem_b.get(key);
        REQUIRE(b != nullptr);
        CHECK(b->psi_real.size() == mem_a.get(key)->psi_real.size());
    }
    cleanup();
}

// §3 ─────────────────────────────────────────────────────────────────────────

TEST_CASE("SemanticMemory: load nonexistent file returns 0",
          "[phase109][memory][persistence]")
{
    SemanticMemory mem;
    CHECK(mem.load("/tmp/does_not_exist_phase109_xyzzy.bin") == 0);
    CHECK(mem.empty());
}

// §4 ─────────────────────────────────────────────────────────────────────────

TEST_CASE("SemanticMemory: save() overwrites — no stale records",
          "[phase109][memory][persistence]")
{
    cleanup();
    // First save: potentially many records.
    {
        SemanticMemory mem;
        for (const auto& s : {"alpha","beta","gamma"}) {
            CognitiveTorus t = make_torus(s); mem.store(t.wave_function());
        }
        mem.save(MEM_PATH);
    }
    // Second save: single record.
    SemanticMemory single;
    { CognitiveTorus t = make_torus("only-one"); single.store(t.wave_function()); }
    single.save(MEM_PATH);

    SemanticMemory check;
    check.load(MEM_PATH);
    CHECK(check.size() == single.size());
    cleanup();
}

// §5 ─────────────────────────────────────────────────────────────────────────

TEST_CASE("SemanticMemory: loaded records survive consolidate()",
          "[phase109][memory][persistence]")
{
    cleanup();
    CognitiveTorus t = make_torus("hello");
    SemanticMemory mem_a;
    const MemoryKey key = mem_a.store(t.wave_function());
    mem_a.save(MEM_PATH);

    SemanticMemory mem_b;
    mem_b.load(MEM_PATH);
    REQUIRE(mem_b.get(key) != nullptr);
    CHECK(mem_b.get(key)->strength > SemanticMemory::MIN_STRENGTH);
    mem_b.consolidate();
    CHECK(mem_b.contains(key));
    cleanup();
}

// §6 ─────────────────────────────────────────────────────────────────────────

TEST_CASE("DecisionLoop: memory_path auto-loads snapshot at construction",
          "[phase109][decision_loop][persistence]")
{
    cleanup();
    {
        CognitiveTorus t = make_torus("nikola resonance");
        SemanticMemory pre;
        pre.store(t.wave_function());
        REQUIRE(pre.size() >= 1);
        pre.save(MEM_PATH);
    }
    CognitiveTorus torus = make_torus("nikola");
    AutonomyConfig ac; ac.enable_dream_weave = false;
    AutonomyEngine engine(ac);
    DecisionLoop loop(torus, engine, make_cfg());
    CHECK(loop.memory().size() >= 1);
    cleanup();
}

// §7 ─────────────────────────────────────────────────────────────────────────

TEST_CASE("DecisionLoop: memory() count matches persisted snapshot count",
          "[phase109][decision_loop][persistence]")
{
    cleanup();
    size_t stored_n = 0;
    {
        SemanticMemory pre;
        for (const auto& s : {"alpha","beta","gamma"}) {
            CognitiveTorus t = make_torus(s); pre.store(t.wave_function());
        }
        stored_n = pre.size();
        pre.save(MEM_PATH);
    }
    CognitiveTorus torus = make_torus("alpha");
    AutonomyConfig ac; ac.enable_dream_weave = false;
    AutonomyEngine engine(ac);
    DecisionLoop loop(torus, engine, make_cfg());
    CHECK(loop.memory().size() == stored_n);
    cleanup();
}

// §8 ─────────────────────────────────────────────────────────────────────────

TEST_CASE("SemanticMemory: psi field vectors match byte-for-byte after save/load",
          "[phase109][memory][persistence]")
{
    cleanup();
    CognitiveTorus t = make_torus("wave field resonance");
    SemanticMemory mem_a;
    const MemoryKey key = mem_a.store(t.wave_function());
    const auto snap_real = mem_a.get(key)->psi_real;
    const auto snap_imag = mem_a.get(key)->psi_imag;
    REQUIRE(!snap_real.empty());
    mem_a.save(MEM_PATH);

    SemanticMemory mem_b;
    mem_b.load(MEM_PATH);
    const MemoryRecord* rec = mem_b.get(key);
    REQUIRE(rec != nullptr);
    REQUIRE(rec->psi_real.size() == snap_real.size());

    bool real_ok = true, imag_ok = true;
    for (size_t i = 0; i < snap_real.size(); ++i) {
        if (rec->psi_real[i] != snap_real[i]) { real_ok = false; break; }
        if (rec->psi_imag[i] != snap_imag[i]) { imag_ok = false; break; }
    }
    CHECK(real_ok);
    CHECK(imag_ok);
    cleanup();
}

// §9 ─────────────────────────────────────────────────────────────────────────

TEST_CASE("DecisionLoop: cross-session — second session restores memory",
          "[phase109][decision_loop][persistence]")
{
    cleanup();
    size_t session_a_size = 0;
    {
        SemanticMemory snap;
        for (const auto& s : {"consciousness","wave","resonance"}) {
            CognitiveTorus t = make_torus(s); snap.store(t.wave_function());
        }
        session_a_size = snap.size();
        snap.save(MEM_PATH);
    }
    {
        CognitiveTorus torus_b = make_torus("quantum field");
        AutonomyConfig ac; ac.enable_dream_weave = false;
        AutonomyEngine engine_b(ac);
        DecisionLoop loop_b(torus_b, engine_b, make_cfg());
        CHECK(loop_b.memory().size() == session_a_size);
    }
    cleanup();
}

// §10 ────────────────────────────────────────────────────────────────────────

TEST_CASE("SemanticMemory: empty save/load roundtrip",
          "[phase109][memory][persistence]")
{
    cleanup();
    SemanticMemory empty_a;
    REQUIRE(empty_a.empty());
    empty_a.save(MEM_PATH);
    REQUIRE(fs::exists(MEM_PATH));

    SemanticMemory empty_b;
    CHECK(empty_b.load(MEM_PATH) == 0);
    CHECK(empty_b.empty());
    cleanup();
}
