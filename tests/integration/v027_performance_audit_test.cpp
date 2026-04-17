/**
 * @file tests/integration/v027_performance_audit_test.cpp
 * @brief v0.2.7 — Performance audit benchmarks for the 0.2.x series.
 *
 * Measures:
 *   §P-1  GoalDAG at scale (1000+ goals)
 *   §P-2  Ingestion throughput (chunking + filtering)
 *   §P-3  Inference tick throughput
 *   §P-4  Personality engine throughput
 *   §P-5  LMDB state persistence throughput
 *   §P-6  Security pipeline latency
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/autonomy/goal_system.hpp>
#include <nikola/autonomy/auto_ingestor.hpp>
#include <nikola/autonomy/ingestion_filter.hpp>
#include <nikola/interior/preference_engine.hpp>
#include <nikola/interior/personality_drift.hpp>
#include <nikola/interior/autobiography.hpp>
#include <nikola/persistence/lmdb_state_store.hpp>
#include <nikola/security/csvp.hpp>
#include <nikola/security/code_blacklist.hpp>
#include <nikola/security/kvm_sandbox.hpp>
#include <nikola/security/ebpf_monitor.hpp>
#include <nikola/security/anomaly_detector.hpp>
#include <nikola/security/security_pipeline.hpp>
#include <nikola/inference/nikola_inference.hpp>
#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/cognitive/cognitive_torus.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace nikola::autonomy;
using namespace nikola::cognitive;
using namespace nikola::interior;
using namespace nikola::security;
using namespace nikola::persistence;
using Catch::Approx;
namespace fs = std::filesystem;

namespace {

struct TempDir {
    std::string path;
    TempDir(const std::string& prefix = "nikola_perf_")
        : path("/tmp/" + prefix + std::to_string(getpid()))
    { fs::create_directories(path); }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
};

using Clock = std::chrono::steady_clock;

template <typename Fn>
double measure_ms(Fn&& fn) {
    auto start = Clock::now();
    fn();
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // anon namespace

// ============================================================================
// §P-1 GoalDAG at Scale
// ============================================================================

TEST_CASE("§P-1 GoalDAG handles 1000+ goals without degradation",
          "[v027][benchmark][goal]") {
    GoalSystem goals;

    // Create 1000 goals
    double create_ms = measure_ms([&]() {
        for (int i = 0; i < 1000; ++i) {
            goals.create_goal("Goal #" + std::to_string(i),
                              static_cast<GoalTier>(i % 3),
                              0.5f + (i % 10) * 0.05f,
                              0.5f + (i % 5) * 0.1f);
        }
    });

    REQUIRE(goals.goal_count() == 1000);
    INFO("1000 goal creation: " << create_ms << " ms");
    CHECK(create_ms < 5000.0);  // Should be well under 5 seconds

    // Complete half
    double complete_ms = measure_ms([&]() {
        auto all = goals.all_goals();
        int completed = 0;
        for (const auto* g : all) {
            if (completed >= 500) break;
            goals.complete_goal(g->id);
            completed++;
        }
    });

    INFO("500 goal completions: " << complete_ms << " ms");
    CHECK(goals.completed_count() == 500);

    // Serialize
    double serialize_ms = 0;
    std::vector<uint8_t> data;
    serialize_ms = measure_ms([&]() {
        data = goals.serialize();
    });

    INFO("Serialize 1000 goals: " << serialize_ms << " ms (" << data.size() << " bytes)");
    CHECK_FALSE(data.empty());
    CHECK(serialize_ms < 2000.0);

    // Deserialize
    double deserialize_ms = measure_ms([&]() {
        GoalSystem restored;
        restored.deserialize(data);
    });

    INFO("Deserialize 1000 goals: " << deserialize_ms << " ms");
    CHECK(deserialize_ms < 2000.0);
}

// ============================================================================
// §P-2 Ingestion Throughput
// ============================================================================

TEST_CASE("§P-2 Ingestion chunking throughput",
          "[v027][benchmark][ingestion]") {
    // Generate a 1MB text document
    std::string megabyte;
    megabyte.reserve(1024 * 1024);
    for (int i = 0; i < 10000; ++i) {
        megabyte += "This is paragraph number " + std::to_string(i) +
                    " containing information about various scientific topics "
                    "including physics, mathematics, and computer science.\n\n";
    }

    double chunk_ms = 0;
    std::vector<std::string> chunks;
    chunk_ms = measure_ms([&]() {
        chunks = AutoIngestor::chunk_paragraphs(megabyte);
    });

    INFO("Chunk " << megabyte.size() / 1024 << " KB into "
         << chunks.size() << " chunks: " << chunk_ms << " ms");
    INFO("Throughput: " << (megabyte.size() / 1024.0 / 1024.0) / (chunk_ms / 1000.0)
         << " MB/s");
    CHECK(chunks.size() > 100);
    CHECK(chunk_ms < 10000.0);  // Should finish within 10 seconds
}

TEST_CASE("§P-3 IngestionFilter SimHash throughput",
          "[v027][benchmark][ingestion]") {
    IngestionFilter filter;

    double hash_ms = measure_ms([&]() {
        for (int i = 0; i < 10000; ++i) {
            std::string text = "Unique content number " + std::to_string(i) +
                               " with some padding text for testing";
            filter.check(text);
            filter.record_ingested(text);
        }
    });

    auto stats = filter.stats();
    INFO("10000 SimHash checks: " << hash_ms << " ms");
    INFO("Throughput: " << 10000.0 / (hash_ms / 1000.0) << " checks/s");
    CHECK(stats.total_checked == 10000);
    CHECK(hash_ms < 10000.0);
}

// ============================================================================
// §P-4 Inference Tick Throughput
// ============================================================================

TEST_CASE("§P-4 NikolaInference tick throughput",
          "[v027][benchmark][inference]") {
    nikola::inference::InferenceConfig cfg;
    cfg.grid_n = 3;
    cfg.steps_per_tick = 10;
    cfg.decode_top_k = 5;
    cfg.enable_gpu = false;
    cfg.enable_npt = false;
    cfg.model_path = "";
    cfg.vocabulary = {
        "hello", "world", "physics", "wave", "energy",
        "quantum", "field", "torus", "think", "learn"
    };

    try {
        nikola::inference::NikolaInference engine(cfg);
        engine.inject("benchmark stimulus");

        double tick_ms = measure_ms([&]() {
            for (int i = 0; i < 100; ++i) {
                engine.tick();
            }
        });

        INFO("100 inference ticks: " << tick_ms << " ms");
        INFO("Per-tick: " << tick_ms / 100.0 << " ms");
        INFO("Throughput: " << 100000.0 / tick_ms << " ticks/s");
        CHECK(tick_ms < 30000.0);  // 100 ticks in under 30 seconds
    } catch (const std::exception& e) {
        WARN("NikolaInference skipped (model unavailable): " << e.what());
    }
}

// ============================================================================
// §P-5 Personality Engine Throughput
// ============================================================================

TEST_CASE("§P-5 PersonalityDrift + PreferenceEngine throughput",
          "[v027][benchmark][personality]") {
    PersonalityDrift drift;
    PreferenceEngine prefs;

    double drift_ms = measure_ms([&]() {
        for (int i = 0; i < 10000; ++i) {
            ExperienceOutcome outcome;
            outcome.success = (i % 3 == 0) ? 0.8f : -0.3f;
            outcome.action_type = i % 12;
            outcome.risk_taken = 0.5f;
            outcome.complexity = 0.4f;
            drift.apply_outcome(outcome);
        }
    });

    double pref_ms = measure_ms([&]() {
        for (int i = 0; i < 10000; ++i) {
            prefs.learn(static_cast<PreferenceDomain>(i % 5),
                        "topic_" + std::to_string(i % 50),
                        (i % 2 == 0) ? +0.1 : -0.05, i);
        }
    });

    INFO("10000 personality drift events: " << drift_ms << " ms");
    INFO("10000 preference learns: " << pref_ms << " ms");
    CHECK(drift_ms < 5000.0);
    CHECK(pref_ms < 5000.0);
}

// ============================================================================
// §P-6 LMDB State Persistence Throughput
// ============================================================================

TEST_CASE("§P-6 LMDB save/load throughput",
          "[v027][benchmark][persistence]") {
    TempDir tmp;
    LmdbStateStore store(tmp.path + "/bench.mdb");

    NikolaState state;
    state.atp = 0.5f;
    state.boredom = 0.3f;
    state.dopamine = 0.6f;
    state.entropy = 1.0f;

    double save_ms = measure_ms([&]() {
        for (uint64_t t = 0; t < 100; ++t) {
            state.atp = 0.01f * t;
            store.save_state(state, t);
        }
    });

    INFO("100 state saves: " << save_ms << " ms");
    INFO("Per-save: " << save_ms / 100.0 << " ms");
    CHECK(save_ms < 10000.0);

    double load_ms = measure_ms([&]() {
        NikolaState loaded;
        uint64_t tick;
        for (int i = 0; i < 100; ++i) {
            store.load_latest_state(loaded, tick);
        }
    });

    INFO("100 state loads: " << load_ms << " ms");
    INFO("Per-load: " << load_ms / 100.0 << " ms");
    CHECK(load_ms < 5000.0);
}

// ============================================================================
// §P-7 Security Pipeline Latency
// ============================================================================

TEST_CASE("§P-7 SecurityPipeline per-module latency",
          "[v027][benchmark][security]") {
    CodeSafetyVerifier   csvp;
    CodePatternBlacklist blacklist;
    KvmSandbox           sandbox;
    EbpfMonitor          ebpf;
    AnomalyDetector      anomaly;

    PipelineConfig cfg;
    SecurityPipeline pipeline(cfg, csvp, blacklist, sandbox, ebpf, anomaly);

    pipeline.set_deploy_callback([](const std::string&, const std::string&) -> bool {
        return true;
    });

    std::string safe_code = R"(
        #include <cstdint>
        extern "C" int32_t nikola_module_factory() { return 42; }
    )";

    double total_ms = measure_ms([&]() {
        for (int i = 0; i < 10; ++i) {
            pipeline.evaluate("bench_module_" + std::to_string(i), safe_code);
        }
    });

    INFO("10 pipeline runs: " << total_ms << " ms");
    INFO("Per-module: " << total_ms / 10.0 << " ms");
    CHECK(total_ms < 30000.0);
}

// ============================================================================
// §P-8 Full DecisionLoop Throughput
// ============================================================================

TEST_CASE("§P-8 DecisionLoop 1000-tick throughput",
          "[v027][benchmark][e2e]") {
    CognitiveTorus torus(3);
    AutonomyConfig ae_cfg;
    ae_cfg.enable_dream_weave = false;
    ae_cfg.enable_boredom = true;
    AutonomyEngine engine(ae_cfg);

    DecisionLoopConfig dl_cfg;
    dl_cfg.steps_per_tick = 10;
    dl_cfg.action_threshold = 0.05f;
    dl_cfg.min_emit_interval_s = 0.0f;
    dl_cfg.decode_top_k = 5;
    dl_cfg.vocabulary = {"hello", "world", "physics", "wave", "energy",
                         "quantum", "field", "torus", "think", "learn"};

    DecisionLoop loop(torus, engine, dl_cfg);
    loop.inject_stimulus("benchmark test");

    double ms = measure_ms([&]() {
        for (int i = 0; i < 1000; ++i) {
            auto r = loop.tick();
            (void)r;
        }
    });

    INFO("1000 DecisionLoop ticks: " << ms << " ms");
    INFO("Per-tick: " << ms / 1000.0 << " ms");
    INFO("Throughput: " << 1000000.0 / ms << " ticks/s");
    CHECK(loop.tick_count() == 1000);
    CHECK(ms < 120000.0);  // 1000 ticks in under 2 minutes
}
