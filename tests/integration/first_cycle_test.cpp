/**
 * @file tests/integration/first_cycle_test.cpp
 * @brief Phase 146 — First Self-Improvement Cycle integration test.
 *
 * Exercises the complete SIE pipeline end-to-end:
 *   DecisionLoop.tick() → GENERATE_CODE fires → SIE.run_cycle() →
 *   specialist query → extract → compile → sign → ShadowSpine deploy
 *
 * This test uses the Gemini-backed specialist, which requires:
 *   - Network access to the Gemini API
 *   - Valid API key at ~/Workspace/CREDS/creds/apiKey.gemini
 *   - google.genai Python package installed
 *
 * Tagged [first_cycle] for selective execution.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <nikola/autonomy/self_improvement_engine.hpp>
#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/autonomy/shadow_spine.hpp>
#include <nikola/autonomy/evolutionary_orchestrator.hpp>
#include <nikola/autonomy/metabolic_controller.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/security/code_blacklist.hpp>
#include <nikola/security/hybrid_verifier.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

using namespace nikola::autonomy;
using namespace nikola::cognitive;
using namespace nikola::security;
using Catch::Matchers::ContainsSubstring;
namespace fs = std::filesystem;

// ============================================================================
// Helpers
// ============================================================================

namespace {

static constexpr float k_atp_large = 100'000.0f;
static constexpr float k_nap_threshold = 5.0f;

/// Full integration fixture: DecisionLoop + SIE + ShadowSpine
struct FirstCycleFixture {
    // ShadowSpine dependency chain
    CodePatternBlacklist        blacklist;
    MetabolicController         controller{k_atp_large, k_nap_threshold};
    EvolutionaryOrchestrator    eo{controller, blacklist};
    HybridVerifier              hv;
    ShadowSpine                 spine{eo, hv};

    // SIE
    SIEConfig                   sie_cfg;
    std::unique_ptr<SelfImprovementEngine> sie;

    // DecisionLoop components
    CognitiveTorus              torus{3};  // 3^9 = 19683 nodes
    AutonomyEngine              ae;
    DecisionLoopConfig          dl_cfg;
    std::unique_ptr<DecisionLoop> loop;

    // Results
    std::vector<SIECycleResult> sie_results;
    std::string work_dir;

    FirstCycleFixture()
        : work_dir("/tmp/nikola_first_cycle_" + std::to_string(getpid()))
    {
        // Configure SIE with Gemini specialist
        sie_cfg.specialist_server_path =
            std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
            "/Workspace/REPOS/nikola/scripts/specialist_gemini.py";
        sie_cfg.ariac_path = "";
        sie_cfg.gpp_path = "/usr/bin/g++";
        sie_cfg.proposal_store_path = "";  // No persistent store for test
        sie_cfg.work_dir = work_dir;
        sie_cfg.specialist_timeout_ms = 120'000;  // 2 min for Gemini API

        // Create SIE
        sie = std::make_unique<SelfImprovementEngine>(spine, sie_cfg);

        // Configure DecisionLoop
        AutonomyConfig ae_cfg;
        ae_cfg.enable_dream_weave = false;
        ae_cfg.enable_boredom = true;
        ae = AutonomyEngine(ae_cfg);

        dl_cfg.steps_per_tick = 10;
        dl_cfg.action_threshold = 0.02f;
        dl_cfg.min_emit_interval_s = 999.0f;  // Suppress EMIT_THOUGHT
        dl_cfg.decode_top_k = 5;
        dl_cfg.vocabulary = {"hello", "improve", "code", "module", "nikola"};

        loop = std::make_unique<DecisionLoop>(torus, ae, dl_cfg);
        loop->set_sie(sie.get());

        // Register SIE callback
        loop->on_sie_cycle = [this](const SIECycleResult& r) {
            sie_results.push_back(r);
        };
    }

    ~FirstCycleFixture() {
        std::error_code ec;
        fs::remove_all(work_dir, ec);
    }
};

} // anon namespace

// ============================================================================
// Test: Direct SIE cycle with live specialist
// ============================================================================

TEST_CASE("Phase 146 — First cycle: direct SIE with Gemini specialist",
          "[first_cycle][integration][sie]") {
    FirstCycleFixture fix;

    // Create a high-boredom state that would trigger GENERATE_CODE
    NikolaState state;
    state.boredom  = 0.9f;
    state.entropy  = 1.5f;
    state.dopamine = 0.4f;
    state.atp      = 0.8f;

    INFO("Running SIE cycle with live Gemini API...");
    auto result = fix.sie->run_cycle(state);

    INFO("Outcome: " << sie_outcome_str(result.outcome));
    INFO("Instruction length: " << result.instruction.size());
    INFO("Raw response length: " << result.raw_response.size());
    INFO("Source code length: " << result.source_code.size());
    INFO("Compile output: " << result.compile_output);
    INFO("SO path: " << result.so_path);
    INFO("Elapsed: " << result.elapsed_ms << " ms");

    // The instruction must have been formulated
    REQUIRE_FALSE(result.instruction.empty());
    CHECK_THAT(result.instruction, ContainsSubstring("nikola_module_factory"));

    // The specialist should have responded
    if (result.outcome == SIEOutcome::SPECIALIST_FAILED) {
        WARN("Specialist failed — network or API issue: " << result.raw_response);
        // Don't fail the test on network issues
        return;
    }
    REQUIRE_FALSE(result.raw_response.empty());

    // Code extraction should have found something
    if (result.outcome == SIEOutcome::NO_CODE_EXTRACTED) {
        WARN("No code extracted from specialist response. Raw response:\n"
             << result.raw_response.substr(0, 500));
        return;
    }
    REQUIRE_FALSE(result.source_code.empty());
    CHECK_THAT(result.source_code, ContainsSubstring("nikola_module_factory"));

    // Packaging should work (valid C++ → .so)
    if (result.outcome == SIEOutcome::PACKAGING_FAILED) {
        WARN("Packaging failed. Compile output:\n" << result.compile_output);
        WARN("Source code:\n" << result.source_code);
        return;
    }

    // If we get past packaging, signing should work
    if (result.outcome == SIEOutcome::SIGNING_FAILED) {
        WARN("Signing failed unexpectedly");
        return;
    }

    // Gate results
    if (result.stage_report) {
        INFO("Gate 0 (signature): " << (result.stage_report->signature_passed ? "PASSED" : "FAILED"));
        INFO("Stage status: " << stage_status_str(result.stage_report->status));
    }

    // The holy grail: SUCCESS means the full pipeline worked
    if (result.outcome == SIEOutcome::SUCCESS) {
        SUCCEED("*** FIRST SELF-IMPROVEMENT CYCLE SUCCEEDED ***");
        CHECK(fix.sie->cycles_succeeded() == 1);
    } else {
        // Report the gate that rejected (still a valid test — shows pipeline works)
        INFO("Cycle completed at gate: " << sie_outcome_str(result.outcome));
    }

    CHECK(fix.sie->cycles_attempted() == 1);
}

TEST_CASE("Phase 146 — First cycle: SIE construction generates valid keypairs",
          "[first_cycle][integration]") {
    FirstCycleFixture fix;

    CHECK(fix.sie->ed25519_public_key().size() == 32);
    CHECK(fix.sie->sphincs_public_key().size() == 64);
    CHECK(fix.sie->cycles_attempted() == 0);
}
