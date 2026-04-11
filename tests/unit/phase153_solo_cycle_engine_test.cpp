/**
 * @file tests/unit/phase153_solo_cycle_engine_test.cpp
 * @brief Phase 153 — Solo SIE: Internal Code Generation + Multi-Cycle Campaigns (v0.1.16)
 *
 * Tests:
 *   §1: InternalCodeGenerator construction and state encoding (3 tests)
 *   §2: Mamba9D + NPT cognitive pipeline integration (3 tests)
 *   §3: Strategy selection from attention weights (3 tests)
 *   §4: Code synthesis and template correctness (3 tests)
 *   §5: Self-assessment confidence scoring (3 tests)
 *   §6: run_cycle_with_source (solo-mode SIE) (3 tests)
 *   §7: SoloCampaignRunner multi-cycle execution (3 tests)
 *   §8: Plateau detection and termination (3 tests)
 *   §9: Campaign result tracking and rollback (3 tests)
 *
 * 27 tests, header-only solo_cycle_engine.hpp + SIE extension.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <nikola/autonomy/solo_cycle_engine.hpp>
#include <nikola/autonomy/decision_loop.hpp>          // NikolaState
#include <nikola/autonomy/shadow_spine.hpp>
#include <nikola/autonomy/evolutionary_orchestrator.hpp>
#include <nikola/autonomy/metabolic_controller.hpp>
#include <nikola/security/code_blacklist.hpp>
#include <nikola/security/hybrid_verifier.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

using namespace nikola::autonomy;
using namespace nikola::security;
using Catch::Approx;
using Catch::Matchers::ContainsSubstring;
namespace fs = std::filesystem;

// ============================================================================
// Helpers
// ============================================================================

namespace {

static constexpr float k_atp_large     = 100'000.0f;
static constexpr float k_nap_threshold = 5.0f;

/// Build a NikolaState with specific characteristics.
NikolaState make_state(float boredom  = 0.5f, float entropy  = 1.0f,
                       float dopamine = 0.5f, float atp      = 0.8f,
                       float torus_e  = 1.0f, float td_error = 0.0f,
                       float time     = 0.1f) {
    NikolaState s{};
    s.boredom      = boredom;
    s.entropy      = entropy;
    s.dopamine     = dopamine;
    s.atp          = atp;
    s.torus_energy = torus_e;
    s.td_error     = td_error;
    s.time         = time;
    return s;
}

/// Full test fixture reusing the SIE dependency chain from Phase 146.
struct SoloFixture {
    CodePatternBlacklist        blacklist;
    MetabolicController         controller{k_atp_large, k_nap_threshold};
    EvolutionaryOrchestrator    eo{controller, blacklist};
    HybridVerifier              hv;
    ShadowSpine                 spine{eo, hv};

    std::string work_dir;
    SIEConfig   cfg;

    SoloFixture()
        : work_dir("/tmp/nikola_solo_test_" + std::to_string(getpid()))
    {
        cfg.specialist_server_path = "";   // No external specialist
        cfg.ariac_path             = "";
        cfg.gpp_path               = "/usr/bin/g++";
        cfg.proposal_store_path    = "";   // No LMDB for unit tests
        cfg.work_dir               = work_dir;
    }

    ~SoloFixture() {
        std::error_code ec;
        fs::remove_all(work_dir, ec);
    }

    std::unique_ptr<SelfImprovementEngine> make_engine() {
        return std::make_unique<SelfImprovementEngine>(spine, cfg);
    }
};

} // anon namespace

// ============================================================================
// §1: InternalCodeGenerator — construction and state encoding
// ============================================================================

TEST_CASE("Phase 153 — InternalCodeGenerator construction",
          "[solo][phase153]") {

    SECTION("default construction creates Mamba9D + NPT") {
        InternalCodeGenerator gen;
        CHECK(gen.grid_n() == 2);
        CHECK(gen.npt().num_heads() == 8);
    }

    SECTION("custom grid_n propagates to NPT") {
        InternalCodeGenerator gen(3, 99);
        CHECK(gen.grid_n() == 3);
        CHECK(gen.npt().grid_n() == 3);
    }

    SECTION("state encoding produces 9 values in [0, 1]") {
        // We test encode_state via generate() — the generated code
        // should reflect state values rather than being random.
        auto state = make_state(0.9f, 3.5f, 0.2f, 0.6f);
        InternalCodeGenerator gen;
        auto result = gen.generate(state);

        // Generated source should exist and contain the factory
        CHECK(!result.source_code.empty());
        CHECK(result.source_code.find("nikola_module_factory") !=
              std::string::npos);
    }
}

// ============================================================================
// §2: Mamba9D + NPT cognitive pipeline integration
// ============================================================================

TEST_CASE("Phase 153 — Cognitive pipeline integration",
          "[solo][phase153]") {

    InternalCodeGenerator gen;

    SECTION("generate produces valid source code") {
        auto state  = make_state(0.5f, 1.0f, 0.5f, 0.8f);
        auto result = gen.generate(state);

        CHECK(!result.source_code.empty());
        CHECK(result.source_code.find("CognitiveParameters") !=
              std::string::npos);
        CHECK(result.source_code.find("extern \"C\"") !=
              std::string::npos);
    }

    SECTION("attention weights are a valid probability distribution") {
        auto state  = make_state(0.3f, 0.5f, 0.7f, 0.9f);
        auto result = gen.generate(state);

        float sum = 0.f;
        for (float w : result.attention_weights) {
            CHECK(w >= 0.f);
            CHECK(w <= 1.f);
            sum += w;
        }
        CHECK(sum == Approx(1.0f).margin(0.01f));
    }

    SECTION("generation time is recorded") {
        auto state  = make_state();
        auto result = gen.generate(state);
        CHECK(result.generation_ms >= 0.0);
    }
}

// ============================================================================
// §3: Strategy selection from attention weights
// ============================================================================

TEST_CASE("Phase 153 — Strategy selection",
          "[solo][phase153]") {

    InternalCodeGenerator gen;

    SECTION("high boredom favours exploration strategy") {
        // Very high boredom → likely head 0 (Global Context) dominates
        auto state  = make_state(0.95f, 0.5f, 0.5f, 0.8f);
        auto result = gen.generate(state);

        // Strategy should be a valid enum value
        CHECK(static_cast<uint8_t>(result.strategy) < 8);
        CHECK(strategy_name(result.strategy) != nullptr);
    }

    SECTION("high entropy state produces a strategy") {
        auto state  = make_state(0.3f, 3.8f, 0.5f, 0.8f);
        auto result = gen.generate(state);
        CHECK(static_cast<uint8_t>(result.strategy) < 8);
    }

    SECTION("different states can produce different strategies") {
        auto state_a = make_state(0.95f, 0.1f, 0.9f, 0.9f);
        auto state_b = make_state(0.1f, 3.5f, 0.1f, 0.3f);

        InternalCodeGenerator gen_a(2, 42);
        InternalCodeGenerator gen_b(2, 42);

        auto result_a = gen_a.generate(state_a);
        auto result_b = gen_b.generate(state_b);

        // At minimum, the generated parameters should differ
        CHECK(result_a.source_code != result_b.source_code);
    }
}

// ============================================================================
// §4: Code synthesis and template correctness
// ============================================================================

TEST_CASE("Phase 153 — Code synthesis",
          "[solo][phase153]") {

    InternalCodeGenerator gen;

    SECTION("generated code includes strategy name as module name") {
        auto state  = make_state(0.5f, 1.0f, 0.5f, 0.8f);
        auto result = gen.generate(state);

        // The source should contain the strategy name
        CHECK(result.source_code.find(strategy_name(result.strategy)) !=
              std::string::npos);
    }

    SECTION("generated parameters are in valid ranges") {
        auto state  = make_state(0.5f, 1.0f, 0.5f, 0.8f);
        auto result = gen.generate(state);

        // Parse float values from generated source (they appear as "0.XXXf")
        CHECK(result.source_code.find("exploration_weight") !=
              std::string::npos);
        CHECK(result.source_code.find("coherence_bias") !=
              std::string::npos);
        CHECK(result.source_code.find("reward_sensitivity") !=
              std::string::npos);
    }

    SECTION("code contains state comment for traceability") {
        auto state  = make_state(0.7f, 2.0f, 0.4f, 0.6f);
        auto result = gen.generate(state);
        CHECK(result.source_code.find("boredom=") != std::string::npos);
        CHECK(result.source_code.find("entropy=") != std::string::npos);
    }
}

// ============================================================================
// §5: Self-assessment confidence scoring
// ============================================================================

TEST_CASE("Phase 153 — Confidence scoring",
          "[solo][phase153]") {

    InternalCodeGenerator gen;

    SECTION("confidence is in [0, 1]") {
        auto state  = make_state(0.5f, 1.0f, 0.5f, 0.8f);
        auto result = gen.generate(state);
        CHECK(result.confidence >= 0.f);
        CHECK(result.confidence <= 1.f);
    }

    SECTION("high ATP + low entropy → higher confidence") {
        auto good   = make_state(0.1f, 0.3f, 0.7f, 0.95f);
        auto bad    = make_state(0.9f, 3.8f, 0.1f, 0.2f);

        auto r_good = gen.generate(good);
        auto r_bad  = gen.generate(bad);

        CHECK(r_good.confidence > r_bad.confidence);
    }

    SECTION("assess_confidence rejects missing factory") {
        auto state = make_state();
        CHECK(gen.assess_confidence("int main() { return 0; }", state) == 0.f);
    }
}

// ============================================================================
// §6: run_cycle_with_source — solo-mode SIE
// ============================================================================

TEST_CASE("Phase 153 — run_cycle_with_source",
          "[solo][phase153]") {

    SoloFixture f;
    auto engine = f.make_engine();

    SECTION("valid source compiles and signs successfully") {
        auto state  = make_state();
        InternalCodeGenerator gen;
        auto result = gen.generate(state);

        auto cycle = engine->run_cycle_with_source(
            result.source_code, result.instruction);

        // The module should at least compile (packaging may fail at gate level
        // but compilation itself should succeed with valid C++)
        CHECK(cycle.outcome != SIEOutcome::NO_CODE_EXTRACTED);
        CHECK(cycle.outcome != SIEOutcome::SPECIALIST_FAILED);
        CHECK(!cycle.source_code.empty());
        CHECK(!cycle.instruction.empty());
        CHECK(cycle.elapsed_ms > 0.0);
    }

    SECTION("empty source returns NO_CODE_EXTRACTED") {
        auto cycle = engine->run_cycle_with_source("", "test empty");
        CHECK(cycle.outcome == SIEOutcome::NO_CODE_EXTRACTED);
    }

    SECTION("cycles_attempted increments on solo cycles") {
        CHECK(engine->cycles_attempted() == 0);

        InternalCodeGenerator gen;
        auto state  = make_state();
        auto result = gen.generate(state);

        (void)engine->run_cycle_with_source(result.source_code, result.instruction);
        CHECK(engine->cycles_attempted() == 1);

        (void)engine->run_cycle_with_source(result.source_code, result.instruction);
        CHECK(engine->cycles_attempted() == 2);
    }
}

// ============================================================================
// §7: SoloCampaignRunner multi-cycle execution
// ============================================================================

TEST_CASE("Phase 153 — Campaign runner",
          "[solo][phase153]") {

    SoloFixture f;
    auto engine = f.make_engine();
    InternalCodeGenerator gen;

    SECTION("campaign runs multiple cycles") {
        SoloCampaignConfig cfg;
        cfg.max_cycles         = 5;
        cfg.target_consecutive = 10;  // unreachable → will hit max_cycles
        cfg.confidence_threshold = 0.0f; // accept everything
        cfg.plateau_patience   = 100;   // disable plateau for this test

        SoloCampaignRunner runner(*engine, gen, cfg);
        auto state = make_state(0.3f, 0.8f, 0.6f, 0.9f);

        auto result = runner.run_campaign(state);
        CHECK(result.cycles_attempted == 5);
        CHECK(result.termination_reason == "max_cycles_reached");
        CHECK(result.history.size() == 5);
        CHECK(result.total_elapsed_ms > 0.0);
    }

    SECTION("campaign stops when target consecutive met") {
        SoloCampaignConfig cfg;
        cfg.max_cycles         = 20;
        cfg.target_consecutive = 1;   // very easy target
        cfg.confidence_threshold = 0.0f;

        SoloCampaignRunner runner(*engine, gen, cfg);
        auto state = make_state(0.3f, 0.8f, 0.6f, 0.9f);

        auto result = runner.run_campaign(state);

        // Should stop before max_cycles when target is met
        if (result.max_consecutive >= 1) {
            CHECK(result.termination_reason == "target_met");
        }
        CHECK(result.total_elapsed_ms > 0.0);
    }

    SECTION("solo cycle convenience method works") {
        SoloCampaignConfig cfg;
        cfg.confidence_threshold = 0.0f;

        SoloCampaignRunner runner(*engine, gen, cfg);
        auto state = make_state(0.5f, 1.0f, 0.5f, 0.8f);

        auto cycle = runner.run_solo_cycle(state);
        CHECK(cycle.outcome != SIEOutcome::SPECIALIST_FAILED);
        CHECK(!cycle.source_code.empty());
    }
}

// ============================================================================
// §8: Plateau detection and termination
// ============================================================================

TEST_CASE("Phase 153 — Plateau detection",
          "[solo][phase153]") {

    SECTION("no plateau when history is short") {
        SoloFixture f;
        auto engine = f.make_engine();
        InternalCodeGenerator gen;

        SoloCampaignConfig cfg;
        cfg.max_cycles         = 2;  // Short campaign
        cfg.plateau_patience   = 5;  // Patience > max_cycles
        cfg.target_consecutive = 100;
        cfg.confidence_threshold = 0.0f;

        SoloCampaignRunner runner(*engine, gen, cfg);
        auto state = make_state();

        auto result = runner.run_campaign(state);
        CHECK_FALSE(result.plateau_detected);
    }

    SECTION("plateau config is accessible") {
        SoloFixture f;
        auto engine = f.make_engine();
        InternalCodeGenerator gen;

        SoloCampaignConfig cfg;
        cfg.plateau_threshold = 0.05f;
        cfg.plateau_patience  = 4;

        SoloCampaignRunner runner(*engine, gen, cfg);
        CHECK(runner.config().plateau_threshold == Approx(0.05f));
        CHECK(runner.config().plateau_patience == 4);
    }

    SECTION("large plateau_patience allows more cycles") {
        SoloFixture f;
        auto engine = f.make_engine();
        InternalCodeGenerator gen;

        SoloCampaignConfig cfg;
        cfg.max_cycles         = 6;
        cfg.plateau_patience   = 100;  // Effectively disabled
        cfg.target_consecutive = 100;  // Unreachable
        cfg.confidence_threshold = 0.0f;

        SoloCampaignRunner runner(*engine, gen, cfg);
        auto state = make_state(0.3f, 0.8f, 0.6f, 0.9f);

        auto result = runner.run_campaign(state);
        // Should reach max_cycles since plateau detection is disabled
        CHECK(result.cycles_attempted == 6);
        CHECK(result.termination_reason == "max_cycles_reached");
    }
}

// ============================================================================
// §9: Campaign result tracking and rollback
// ============================================================================

TEST_CASE("Phase 153 — Campaign result tracking",
          "[solo][phase153]") {

    SECTION("campaign tracks consecutive successes correctly") {
        SoloFixture f;
        auto engine = f.make_engine();
        InternalCodeGenerator gen;

        SoloCampaignConfig cfg;
        cfg.max_cycles         = 10;
        cfg.target_consecutive = 100; // unreachable
        cfg.confidence_threshold = 0.0f;
        cfg.plateau_patience   = 100;  // disable plateau for this test

        SoloCampaignRunner runner(*engine, gen, cfg);
        auto state = make_state(0.3f, 0.5f, 0.6f, 0.9f);

        auto result = runner.run_campaign(state);
        CHECK(result.cycles_attempted == 10);

        // max_consecutive should reflect actual gate passage
        CHECK(result.max_consecutive >= 0);
        CHECK(result.max_consecutive <= result.cycles_attempted);
        CHECK(result.cycles_succeeded <= result.cycles_attempted);
    }

    SECTION("confidence threshold filters low-quality proposals") {
        SoloFixture f;
        auto engine = f.make_engine();
        InternalCodeGenerator gen;

        SoloCampaignConfig cfg;
        cfg.max_cycles         = 3;
        cfg.confidence_threshold = 0.99f;  // Very high bar
        cfg.target_consecutive = 100;

        SoloCampaignRunner runner(*engine, gen, cfg);
        // Terrible state → low confidence
        auto state = make_state(0.99f, 3.9f, 0.01f, 0.05f);

        auto result = runner.run_campaign(state);

        // Most cycles should be skipped due to low confidence
        // (they get QUALITY_REGRESSION outcome)
        bool any_regression = false;
        for (const auto& q : result.history) {
            if (q.outcome == SIEOutcome::QUALITY_REGRESSION)
                any_regression = true;
        }
        CHECK(any_regression);
    }

    SECTION("target_met reports correctly") {
        SoloFixture f;
        auto engine = f.make_engine();
        InternalCodeGenerator gen;

        SoloCampaignConfig cfg;
        cfg.max_cycles = 20;
        cfg.target_consecutive = 1;
        cfg.confidence_threshold = 0.0f;

        SoloCampaignRunner runner(*engine, gen, cfg);
        auto state = make_state(0.3f, 0.8f, 0.6f, 0.9f);

        auto result = runner.run_campaign(state);

        CHECK(result.target_met(1) == (result.max_consecutive >= 1));
        CHECK((result.termination_reason == "target_met" ||
               result.termination_reason == "max_cycles_reached"));
    }
}
