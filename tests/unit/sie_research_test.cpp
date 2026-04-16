/**
 * @file tests/unit/sie_research_test.cpp
 * @brief Phase 32 — SIE Research Integration unit tests (Catch2 v3).
 *
 * Tests the research phase wired into SelfImprovementEngine:
 *   - formulate_research_query() produces targeted queries from NikolaState
 *   - set_research_fn() wires a LookupFn into the SIE
 *   - run_cycle() calls the research function and populates SIECycleResult
 *   - Without research_fn, the SIE behaves identically to before
 *
 * These tests use a mock research function (no network required) and
 * exercise the SIE pipeline up to the specialist step (which fails
 * without a live API, giving us SPECIALIST_FAILED — but the research
 * phase completes before that).
 *
 * Live network tests are guarded by the [network] tag.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <nikola/autonomy/self_improvement_engine.hpp>
#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/autonomy/shadow_spine.hpp>
#include <nikola/autonomy/evolutionary_orchestrator.hpp>
#include <nikola/autonomy/metabolic_controller.hpp>
#include <nikola/autonomy/research_router.hpp>
#include <nikola/autonomy/tavily_oracle.hpp>
#include <nikola/autonomy/firecrawl_oracle.hpp>
#include <nikola/security/code_blacklist.hpp>
#include <nikola/security/hybrid_verifier.hpp>

#include <string>
#include <vector>

using namespace nikola::autonomy;
using namespace nikola::security;
using Catch::Matchers::ContainsSubstring;

// ============================================================================
// Minimal SIE fixture (no specialist, no store — just enough to run_cycle)
// ============================================================================

namespace {

static constexpr float k_atp_large = 100'000.0f;
static constexpr float k_nap_threshold = 5.0f;

struct SIEResearchFixture {
    CodePatternBlacklist    blacklist;
    MetabolicController     controller{k_atp_large, k_nap_threshold};
    EvolutionaryOrchestrator eo{controller, blacklist};
    HybridVerifier          hv;
    ShadowSpine             spine{eo, hv};
    SIEConfig               cfg;
    std::unique_ptr<SelfImprovementEngine> sie;

    // Track what the mock research function received
    std::vector<std::string> research_queries;
    std::string mock_research_content = "Mock research findings: "
        "exploration diversity can be improved by adding epsilon-greedy "
        "noise with a decay schedule. Recommended epsilon range: 0.1-0.3.";

    SIEResearchFixture() {
        cfg.specialist_server_path = "";  // No specialist
        cfg.ariac_path = "";
        cfg.gpp_path = "/usr/bin/g++";
        cfg.proposal_store_path = "";
        cfg.work_dir = "/tmp/nikola_sie_research_test_" +
                       std::to_string(getpid());

        sie = std::make_unique<SelfImprovementEngine>(spine, cfg);
    }

    /// Install mock research function that records queries and returns content.
    void install_mock_research() {
        sie->set_research_fn([this](const std::string& query) -> std::string {
            research_queries.push_back(query);
            return mock_research_content;
        });
    }

    /// Install research function that returns empty (simulating failed lookup).
    void install_empty_research() {
        sie->set_research_fn([this](const std::string& query) -> std::string {
            research_queries.push_back(query);
            return "";
        });
    }
};

} // anon namespace

// ─────────────────────────────────────────────────────────────────────────────
//  formulate_research_query() — static, no fixture needed
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("SIE — research query targets exploration for high boredom",
          "[sie_research][unit]") {
    SIEResearchFixture fix;
    fix.install_mock_research();

    NikolaState state;
    state.boredom = 0.9f;
    state.entropy = 1.0f;
    state.dopamine = 0.5f;
    state.atp = 0.8f;

    auto result = fix.sie->run_cycle(state);
    REQUIRE(fix.research_queries.size() == 1);
    CHECK_THAT(fix.research_queries[0], ContainsSubstring("exploration"));
    CHECK_THAT(fix.research_queries[0], ContainsSubstring("novelty"));
}

TEST_CASE("SIE — research query targets coherence for high entropy",
          "[sie_research][unit]") {
    SIEResearchFixture fix;
    fix.install_mock_research();

    NikolaState state;
    state.boredom = 0.3f;
    state.entropy = 2.5f;
    state.dopamine = 0.5f;
    state.atp = 0.8f;

    auto result = fix.sie->run_cycle(state);
    REQUIRE(fix.research_queries.size() == 1);
    CHECK_THAT(fix.research_queries[0], ContainsSubstring("coherence"));
    CHECK_THAT(fix.research_queries[0], ContainsSubstring("entropy"));
}

TEST_CASE("SIE — research query targets reward for low dopamine",
          "[sie_research][unit]") {
    SIEResearchFixture fix;
    fix.install_mock_research();

    NikolaState state;
    state.boredom = 0.3f;
    state.entropy = 1.0f;
    state.dopamine = 0.1f;
    state.atp = 0.8f;

    auto result = fix.sie->run_cycle(state);
    REQUIRE(fix.research_queries.size() == 1);
    CHECK_THAT(fix.research_queries[0], ContainsSubstring("reward"));
    CHECK_THAT(fix.research_queries[0], ContainsSubstring("dopamine"));
}

TEST_CASE("SIE — research query targets general for balanced state",
          "[sie_research][unit]") {
    SIEResearchFixture fix;
    fix.install_mock_research();

    NikolaState state;
    state.boredom = 0.5f;
    state.entropy = 1.5f;
    state.dopamine = 0.5f;
    state.atp = 0.8f;

    auto result = fix.sie->run_cycle(state);
    REQUIRE(fix.research_queries.size() == 1);
    CHECK_THAT(fix.research_queries[0], ContainsSubstring("cognitive"));
    CHECK_THAT(fix.research_queries[0], ContainsSubstring("parameter"));
}

// ─────────────────────────────────────────────────────────────────────────────
//  SIE with mock research function — run_cycle integration
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("SIE — run_cycle calls research function before specialist",
          "[sie_research][unit]") {
    SIEResearchFixture fix;
    fix.install_mock_research();

    NikolaState state;
    state.boredom = 0.9f;
    state.entropy = 1.0f;
    state.dopamine = 0.5f;
    state.atp = 0.8f;

    auto result = fix.sie->run_cycle(state);

    // Cycle will fail at specialist (no specialist configured),
    // but research phase should have completed before that
    CHECK(result.outcome == SIEOutcome::SPECIALIST_FAILED);

    // Research function was called
    REQUIRE(fix.research_queries.size() == 1);
    CHECK_THAT(fix.research_queries[0], ContainsSubstring("exploration"));

    // Research metadata populated in result
    CHECK_FALSE(result.research_query.empty());
    CHECK_THAT(result.research_query, ContainsSubstring("exploration"));
    CHECK_FALSE(result.research_content.empty());
    CHECK_THAT(result.research_content, ContainsSubstring("epsilon-greedy"));
    CHECK(result.research_source == "research_router");

    // Instruction should contain the research context
    CHECK_THAT(result.instruction, ContainsSubstring("BEGIN RESEARCH"));
    CHECK_THAT(result.instruction, ContainsSubstring("epsilon-greedy"));
    CHECK_THAT(result.instruction, ContainsSubstring("END RESEARCH"));
}

TEST_CASE("SIE — run_cycle without research function preserves old behaviour",
          "[sie_research][unit]") {
    SIEResearchFixture fix;
    // Do NOT install research function

    NikolaState state;
    state.boredom = 0.9f;
    state.entropy = 1.0f;
    state.dopamine = 0.5f;
    state.atp = 0.8f;

    auto result = fix.sie->run_cycle(state);

    CHECK(result.outcome == SIEOutcome::SPECIALIST_FAILED);

    // No research should have been attempted
    CHECK(result.research_query.empty());
    CHECK(result.research_content.empty());
    CHECK(result.research_source.empty());

    // Instruction should NOT have research block
    CHECK_THAT(result.instruction, !ContainsSubstring("BEGIN RESEARCH"));
    CHECK_THAT(result.instruction, ContainsSubstring("exploration diversity"));
}

TEST_CASE("SIE — run_cycle with empty research result skips injection",
          "[sie_research][unit]") {
    SIEResearchFixture fix;
    fix.install_empty_research();

    NikolaState state;
    state.boredom = 0.9f;
    state.entropy = 1.0f;
    state.dopamine = 0.5f;
    state.atp = 0.8f;

    auto result = fix.sie->run_cycle(state);

    CHECK(result.outcome == SIEOutcome::SPECIALIST_FAILED);

    // Research was called but returned empty
    REQUIRE(fix.research_queries.size() == 1);
    CHECK_FALSE(result.research_query.empty());
    CHECK(result.research_content.empty());
    CHECK(result.research_source.empty());

    // No research block in instruction (empty content → no injection)
    CHECK_THAT(result.instruction, !ContainsSubstring("BEGIN RESEARCH"));
}

TEST_CASE("SIE — research query varies by state weakness",
          "[sie_research][unit]") {
    SIEResearchFixture fix;
    fix.install_mock_research();

    // High entropy state
    {
        NikolaState state;
        state.boredom = 0.3f;
        state.entropy = 2.5f;
        state.dopamine = 0.5f;
        state.atp = 0.8f;

        auto result = fix.sie->run_cycle(state);
        REQUIRE(fix.research_queries.size() == 1);
        CHECK_THAT(fix.research_queries[0], ContainsSubstring("coherence"));
    }

    fix.research_queries.clear();

    // Low dopamine state
    {
        NikolaState state;
        state.boredom = 0.3f;
        state.entropy = 1.0f;
        state.dopamine = 0.1f;
        state.atp = 0.8f;

        auto result = fix.sie->run_cycle(state);
        REQUIRE(fix.research_queries.size() == 1);
        CHECK_THAT(fix.research_queries[0], ContainsSubstring("reward"));
    }
}

TEST_CASE("SIE — SIECycleResult research fields default empty",
          "[sie_research][unit]") {
    SIECycleResult r;
    CHECK(r.research_query.empty());
    CHECK(r.research_content.empty());
    CHECK(r.research_source.empty());
}

TEST_CASE("SIE — cycles_attempted increments with research enabled",
          "[sie_research][unit]") {
    SIEResearchFixture fix;
    fix.install_mock_research();

    NikolaState state;
    state.boredom = 0.5f;
    state.entropy = 1.0f;
    state.dopamine = 0.5f;
    state.atp = 0.8f;

    (void)fix.sie->run_cycle(state);
    (void)fix.sie->run_cycle(state);

    CHECK(fix.sie->cycles_attempted() == 2);
    CHECK(fix.research_queries.size() == 2);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Live network test: SIE + ResearchRouter
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("SIE — live research with ResearchRouter as_lookup_fn",
          "[sie_research][network]") {
    // Load API keys
    auto tavily_key = load_tavily_api_key(
        "/home/randy/Workspace/CREDS/creds/tavily.creds");
    auto firecrawl_key = load_firecrawl_api_key(
        "/home/randy/Workspace/CREDS/creds/firecrawl.creds");
    if (tavily_key.empty() || firecrawl_key.empty()) {
        SKIP("API keys not found");
    }

    // Build real research pipeline
    TavilyOracle tavily(tavily_key);
    FirecrawlOracle firecrawl(firecrawl_key);
    ResearchRouter router(tavily, firecrawl);

    // Build SIE with real research
    CodePatternBlacklist blacklist;
    MetabolicController controller{100'000.0f, 5.0f};
    EvolutionaryOrchestrator eo{controller, blacklist};
    HybridVerifier hv;
    ShadowSpine spine{eo, hv};

    SIEConfig cfg;
    cfg.specialist_server_path = "";
    cfg.work_dir = "/tmp/nikola_sie_live_research_" +
                   std::to_string(getpid());

    SelfImprovementEngine sie(spine, cfg);
    sie.set_research_fn(router.as_lookup_fn());

    // Run a cycle with high boredom
    NikolaState state;
    state.boredom = 0.9f;
    state.entropy = 1.0f;
    state.dopamine = 0.5f;
    state.atp = 0.8f;

    auto result = sie.run_cycle(state);

    // Will fail at specialist, but research should have worked
    CHECK(result.outcome == SIEOutcome::SPECIALIST_FAILED);

    INFO("Research query: " << result.research_query);
    INFO("Research content length: " << result.research_content.size());
    INFO("Research source: " << result.research_source);

    // Research should have populated
    CHECK_FALSE(result.research_query.empty());
    CHECK_FALSE(result.research_content.empty());
    CHECK(result.research_source == "research_router");

    // Instruction should contain the research context
    CHECK_THAT(result.instruction, ContainsSubstring("BEGIN RESEARCH"));

    // Router should have routed at least one query
    CHECK(router.route_count() >= 1);
}
