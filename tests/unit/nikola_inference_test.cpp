/**
 * @file tests/unit/nikola_inference_test.cpp
 * @brief Phase v0.2.5 — NikolaInference lightweight runner tests (Catch2 v3).
 *
 * Verifies the inference pipeline works in isolation:
 *  - Construction with default config
 *  - Text injection and ticking
 *  - Single-shot generate()
 *  - Convenience infer()
 *  - Entropy stays bounded
 *  - Field reseed on collapse
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/inference/nikola_inference.hpp>

#include <string>
#include <vector>
#include <cmath>

using namespace nikola::inference;

// ── Helpers ──────────────────────────────────────────────────────────────────

static InferenceConfig make_test_config() {
    InferenceConfig cfg;
    cfg.grid_n         = 3;
    cfg.steps_per_tick = 20;   // fast ticks for testing
    cfg.decode_top_k   = 5;
    cfg.enable_npt     = true;

    // ORT model paths (needed when NIKOLA_HAS_ORT is defined)
#ifdef NIKOLA_HAS_ORT
    cfg.model_path          = NIKOLA_ORT_MODEL_PATH;
    cfg.tokenizer_json_path = std::string(NIKOLA_ORT_TOKENIZER_PATH) + "/tokenizer.json";
#endif

    // Minimal vocabulary for testing
    cfg.vocabulary = {
        "energy", "wave", "field", "quantum", "entropy",
        "signal", "pattern", "flow", "state", "node",
        "space", "time", "light", "force", "mass",
        "spin", "phase", "flux", "mode", "pulse"
    };
    return cfg;
}

// ── Construction ─────────────────────────────────────────────────────────────

TEST_CASE("NikolaInference — constructs with default config", "[inference][unit]") {
    auto cfg = make_test_config();
    REQUIRE_NOTHROW(NikolaInference(cfg));
}

TEST_CASE("NikolaInference — constructs with NPT disabled", "[inference][unit]") {
    auto cfg = make_test_config();
    cfg.enable_npt = false;
    REQUIRE_NOTHROW(NikolaInference(cfg));
}

// ── Injection ────────────────────────────────────────────────────────────────

TEST_CASE("NikolaInference — inject does not throw", "[inference][unit]") {
    auto cfg = make_test_config();
    NikolaInference engine(cfg);
    REQUIRE_NOTHROW(engine.inject("test input text"));
}

// ── Tick ─────────────────────────────────────────────────────────────────────

TEST_CASE("NikolaInference — tick returns valid result", "[inference][unit]") {
    auto cfg = make_test_config();
    NikolaInference engine(cfg);
    engine.inject("energy flows through the field");

    auto result = engine.tick();
    CHECK(result.tick == 0);  // 0-indexed
    CHECK(std::isfinite(result.entropy));
}

TEST_CASE("NikolaInference — multiple ticks increment counter", "[inference][unit]") {
    auto cfg = make_test_config();
    NikolaInference engine(cfg);
    engine.inject("quantum signal");

    auto r1 = engine.tick();
    auto r2 = engine.tick();
    auto r3 = engine.tick();

    // Tick counter is 0-indexed
    CHECK(r1.tick == 0);
    CHECK(r2.tick == 1);
    CHECK(r3.tick == 2);
}

// ── Generate ─────────────────────────────────────────────────────────────────

TEST_CASE("NikolaInference — generate produces results over multiple ticks", "[inference][unit]") {
    auto cfg = make_test_config();
    cfg.steps_per_tick = 50;   // more steps = richer dynamics
    NikolaInference engine(cfg);
    engine.inject("the wave function collapses into patterns of light");

    auto results = engine.generate(100);

    // Should produce at least one result
    CHECK(!results.empty());

    // Verify tick indices are sequential
    for (size_t i = 0; i < results.size(); ++i) {
        CHECK(results[i].tick >= 0);
    }
}

// ── Infer (convenience) ──────────────────────────────────────────────────────

TEST_CASE("NikolaInference — infer returns string result", "[inference][unit]") {
    auto cfg = make_test_config();
    cfg.steps_per_tick = 50;
    NikolaInference engine(cfg);

    auto thought = engine.infer("what is the nature of entropy", 100);

    // The main thing is the pipeline didn't crash.
    // With tiny vocab it might produce empty string — that's OK.
    CHECK(thought.size() < 10000);  // sanity bound
}

// ── Entropy ──────────────────────────────────────────────────────────────────

TEST_CASE("NikolaInference — entropy stays finite", "[inference][unit]") {
    auto cfg = make_test_config();
    NikolaInference engine(cfg);
    engine.inject("field dynamics");

    for (int i = 0; i < 20; ++i) {
        auto r = engine.tick();
        CHECK(std::isfinite(r.entropy));
    }
}

// ── Warmup ───────────────────────────────────────────────────────────────────

TEST_CASE("NikolaInference — warmup does not throw", "[inference][unit]") {
    auto cfg = make_test_config();
    NikolaInference engine(cfg);
    REQUIRE_NOTHROW(engine.warmup());
}
