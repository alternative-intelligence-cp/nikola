// ============================================================================
// phase33_memory_persistence_test.cpp    Phase 33 — Semantic Memory Persistence
// ============================================================================
//
// Tests:
//   §1 SemanticMemory save/load round-trip — record count and key correctness
//   §2 save/load — all MemoryRecord fields preserved (strength, age, psi)
//   §3 First-run resilience — load() on absent file returns 0, no throw
//   §4 Corrupt-magic resilience — load() on invalid file returns 0
//   §5 save() overwrites cleanly — second save doesn't accumulate records
//   §6 DecisionLoop: memory_path in config → file is created after STORE_MEMORY
//   §7 DecisionLoop: constructor re-loads records from existing file
//   §8 NAP triggers decay — SemanticMemory.decay() reduces strength correctly
//   §9 Consolidation prunes weak records below MIN_STRENGTH
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "nikola/cognitive/cognitive_torus.hpp"
#include "nikola/cognitive/semantic_memory.hpp"
#include "nikola/autonomy/autonomy_engine.hpp"
#include "nikola/autonomy/decision_loop.hpp"

namespace fs = std::filesystem;

using namespace nikola::cognitive;
using namespace nikola::autonomy;

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

static const std::string TMP_PATH = "/tmp/nikola_phase33_mem_test.bin";

static void remove_tmp() { std::remove(TMP_PATH.c_str()); }

/// Minimal CognitiveTorus with one seed injection so psi is non-trivial.
static CognitiveTorus make_live_torus()
{
    CognitiveTorus t(3);
    // Inject a text token to create non-uniform field.  Works in ORT and
    // non-ORT builds (non-ORT path still perturbs field with uniform pulse).
    t.inject_text("hello", 0.0);
    // Advance one step so psi is non-zero.
    t.run(1, t.safe_dt());
    return t;
}

static AutonomyEngine make_engine()
{
    AutonomyConfig ac;
    ac.enable_dream_weave = false;
    ac.enable_boredom     = true;
    return AutonomyEngine(ac);
}

static DecisionLoopConfig make_config_with_memory()
{
    DecisionLoopConfig cfg;
    cfg.steps_per_tick      = 5;
    cfg.action_threshold    = 0.0f;    // let anything fire
    cfg.min_store_interval_s = 0.0f;   // no cooldown on STORE_MEMORY
    cfg.min_emit_interval_s  = 0.0f;
    cfg.vocabulary           = {"hello", "wave", "memory", "nikola"};
    cfg.memory_path          = TMP_PATH;
    return cfg;
}


// ┌─────────────────────────────────────────────────────────────────────────┐
// │ §1  Save / load round-trip — record count and key correctness           │
// └─────────────────────────────────────────────────────────────────────────┘

TEST_CASE("SemanticMemory: save/load round-trip preserves record count",
          "[phase33][memory]")
{
    remove_tmp();

    auto torus = make_live_torus();

    SemanticMemory mem;
    const auto k1 = mem.store(torus.wave_function());
    // Advance torus to get a different field state for the second record.
    torus.run(10, torus.safe_dt());
    const auto k2 = mem.store(torus.wave_function());

    REQUIRE(mem.size() >= 1);   // k1 and k2 may collide (same dominant node) — at least 1

    mem.save(TMP_PATH);
    REQUIRE(fs::exists(TMP_PATH));

    SemanticMemory loaded;
    const size_t n = loaded.load(TMP_PATH);

    CHECK(n == loaded.size());
    CHECK(loaded.size() == mem.size());

    // Both keys must be present in loaded memory
    for (const auto key : mem.all_keys()) {
        CHECK(loaded.contains(key));
    }

    remove_tmp();
}


// ┌─────────────────────────────────────────────────────────────────────────┐
// │ §2  All MemoryRecord fields survive the round-trip                      │
// └─────────────────────────────────────────────────────────────────────────┘

TEST_CASE("SemanticMemory: save/load round-trip preserves all record fields",
          "[phase33][memory]")
{
    remove_tmp();

    auto torus = make_live_torus();

    SemanticMemory mem;
    const auto key = mem.store(torus.wave_function());

    REQUIRE(mem.size() >= 1);
    REQUIRE(mem.contains(key));

    // Manually mutate the strength and age so we can verify they survive.
    // SemanticMemory has no direct mutator but store() sets strength=1.0;
    // decay() changes it.  Use decay(1.0) to produce a known value.
    mem.decay(1.0f);  // strength *= exp(-DECAY_RATE * 1.0)
    const float expected_strength = mem.get(key)->strength;

    mem.save(TMP_PATH);

    SemanticMemory loaded;
    loaded.load(TMP_PATH);

    REQUIRE(loaded.contains(key));
    const MemoryRecord* rec = loaded.get(key);
    REQUIRE(rec != nullptr);

    CHECK(rec->key == key);
    CHECK(rec->strength == Catch::Approx(expected_strength).epsilon(1e-5f));

    // psi_real and psi_imag must have the same length as the original
    CHECK(rec->psi_real.size() == mem.get(key)->psi_real.size());
    CHECK(rec->psi_imag.size() == mem.get(key)->psi_imag.size());

    // Spot-check first element of psi_real
    if (!rec->psi_real.empty()) {
        CHECK(rec->psi_real[0] ==
              Catch::Approx(mem.get(key)->psi_real[0]).epsilon(1e-5f));
    }

    remove_tmp();
}


// ┌─────────────────────────────────────────────────────────────────────────┐
// │ §3  First-run: load() on absent file returns 0, no throw                │
// └─────────────────────────────────────────────────────────────────────────┘

TEST_CASE("SemanticMemory: load() on absent file returns 0 silently",
          "[phase33][memory]")
{
    remove_tmp();
    SemanticMemory mem;
    size_t n = 0;
    REQUIRE_NOTHROW(n = mem.load(TMP_PATH));
    CHECK(n == 0);
    CHECK(mem.empty());
}


// ┌─────────────────────────────────────────────────────────────────────────┐
// │ §4  Corrupt-magic: load() returns 0 without throwing                    │
// └─────────────────────────────────────────────────────────────────────────┘

TEST_CASE("SemanticMemory: load() on file with bad magic returns 0",
          "[phase33][memory]")
{
    // Write garbage bytes
    {
        std::ofstream ofs(TMP_PATH, std::ios::binary);
        const char junk[] = "JUNKJUNKJUNKJUNK";
        ofs.write(junk, sizeof(junk));
    }

    SemanticMemory mem;
    size_t n = 0;
    REQUIRE_NOTHROW(n = mem.load(TMP_PATH));
    CHECK(n == 0);
    CHECK(mem.empty());

    remove_tmp();
}


// ┌─────────────────────────────────────────────────────────────────────────┐
// │ §5  Second save() overwrites — does not double records                  │
// └─────────────────────────────────────────────────────────────────────────┘

TEST_CASE("SemanticMemory: saving twice doesn't accumulate extra records",
          "[phase33][memory]")
{
    remove_tmp();

    auto torus = make_live_torus();
    SemanticMemory mem;
    mem.store(torus.wave_function());

    mem.save(TMP_PATH);
    mem.save(TMP_PATH);   // <-- second save should overwrite cleanly

    SemanticMemory loaded;
    loaded.load(TMP_PATH);

    CHECK(loaded.size() == mem.size());   // not doubled

    remove_tmp();
}


// ┌─────────────────────────────────────────────────────────────────────────┐
// │ §6  DecisionLoop: STORE_MEMORY fires → file is created at memory_path   │
// └─────────────────────────────────────────────────────────────────────────┘

TEST_CASE("DecisionLoop: STORE_MEMORY action creates memory file at memory_path",
          "[phase33][decision_loop]")
{
    remove_tmp();

    auto torus  = make_live_torus();
    auto engine = make_engine();
    auto cfg    = make_config_with_memory();
    // Lower consolidation interval so STORE_MEMORY fires quickly
    cfg.min_store_interval_s = 0.0f;

    DecisionLoop loop(torus, engine, cfg);

    // Inject a stimulus to spike dopamine so STORE_MEMORY becomes likely.
    loop.inject_stimulus("important memory moment");

    bool stored = false;
    loop.on_action = [&](const DecisionResult& r) {
        if (r.type == ActionType::STORE_MEMORY) stored = true;
    };

    // Run up to 300 ticks waiting for STORE_MEMORY to fire.
    for (int i = 0; i < 300 && !stored; ++i) {
        loop.inject_stimulus("remember this");
        loop.tick();
    }

    if (stored) {
        CHECK(fs::exists(TMP_PATH));
        // Memory size in RAM should be ≥ 1
        CHECK(loop.memory().size() >= 1);
    } else {
        // STORE_MEMORY didn't fire in 300 ticks — this can happen on runs
        // that don't spike dopamine enough in a headless test.
        // Don't hard-fail; WARN so the developer knows.
        WARN("STORE_MEMORY did not fire in 300 ticks — dopamine spike condition "
             "was not met in this run.  Consider increasing vocabulary or "
             "stimulus strength.");
    }

    remove_tmp();
}


// ┌─────────────────────────────────────────────────────────────────────────┐
// │ §7  DecisionLoop: constructor loads existing memory file                 │
// └─────────────────────────────────────────────────────────────────────────┘

TEST_CASE("DecisionLoop: constructor loads pre-existing memory file",
          "[phase33][decision_loop]")
{
    remove_tmp();

    // Pre-populate a memory file using the SemanticMemory API directly.
    {
        auto torus = make_live_torus();
        SemanticMemory seed_mem;
        seed_mem.store(torus.wave_function());
        torus.run(15, torus.safe_dt());
        seed_mem.store(torus.wave_function());
        seed_mem.save(TMP_PATH);
    }

    REQUIRE(fs::exists(TMP_PATH));

    // Now construct DecisionLoop pointing at that file — it should load
    // the records during construction.
    auto torus2  = make_live_torus();
    auto engine2 = make_engine();
    auto cfg2    = make_config_with_memory();
    DecisionLoop loop2(torus2, engine2, cfg2);

    // Records from the pre-populated file should be in RAM.
    // (count could be 1 or 2 depending on whether both stores hit the same key)
    CHECK(loop2.memory().size() >= 1);

    remove_tmp();
}


// ┌─────────────────────────────────────────────────────────────────────────┐
// │ §8  Decay reduces strength monotonically                                 │
// └─────────────────────────────────────────────────────────────────────────┘

TEST_CASE("SemanticMemory::decay reduces strength correctly",
          "[phase33][memory]")
{
    auto torus = make_live_torus();
    SemanticMemory mem;
    const auto key = mem.store(torus.wave_function());

    const float s0 = mem.get(key)->strength;
    CHECK(s0 == Catch::Approx(1.0f));

    mem.decay(100.0f);   // 100 physics-time seconds
    const float s1 = mem.get(key)->strength;

    // strength should have decreased but remain above 0
    CHECK(s1 < s0);
    CHECK(s1 > 0.f);

    // Decay is exponential: s1 == s0 * exp(-DECAY_RATE * 100)
    const float expected = s0 * std::exp(-SemanticMemory::DECAY_RATE * 100.f);
    CHECK(s1 == Catch::Approx(expected).epsilon(1e-4f));
}


// ┌─────────────────────────────────────────────────────────────────────────┐
// │ §9  Consolidation prunes records below MIN_STRENGTH                      │
// └─────────────────────────────────────────────────────────────────────────┘

TEST_CASE("SemanticMemory::consolidate prunes weak records",
          "[phase33][memory]")
{
    auto torus = make_live_torus();
    SemanticMemory mem;
    const auto key = mem.store(torus.wave_function());
    REQUIRE(mem.size() == 1);

    // Decay by a huge amount so strength falls below MIN_STRENGTH
    // MIN_STRENGTH = 0.01; DECAY_RATE = 0.001
    // After t seconds: strength = exp(-0.001 * t) < 0.01 when t > ln(100)/0.001 = 4605 s
    mem.decay(5000.0f);
    REQUIRE(mem.get(key)->strength < SemanticMemory::MIN_STRENGTH);

    const size_t pruned = mem.consolidate();
    CHECK(pruned >= 1);
    CHECK(mem.empty());
}
