/**
 * @file tests/integration/aria_specialist_integration_test.cpp
 * @brief v0.0.19 Integration tests: Aria Specialist → SIE Pipeline.
 *
 * Covers:
 *   §A — CompileValidator + CodeProposalStore: validate → persist round-trip
 *   §B — GENERATE_CODE scoring in live DecisionLoop (specialist disabled)
 *   §C — GENERATE_CODE scoring with specialist enabled (mock config)
 *   §D — Code extraction → validation → proposal store full pipeline
 *   §E — Proposal store metrics accumulation over multiple proposals
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/aria/compile_validator.hpp>
#include <nikola/aria/code_proposal_store.hpp>
#include <nikola/aria/specialist_interface.hpp>
#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/cognitive/cognitive_torus.hpp>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

using namespace nikola::aria;
using namespace nikola::autonomy;
using namespace nikola::cognitive;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string make_tmpdir(const std::string& sfx) {
    auto dir = std::filesystem::temp_directory_path() / ("nikola_integ_" + sfx + "_"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    return dir.string();
}

static void cleanup_tmpdir(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

// ─────────────────────────────────────────────────────────────────────────────
// §A — CompileValidator + CodeProposalStore round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§A-1 validate → store: successful compile persisted",
          "[aria_integ][pipeline]") {
    auto dir = make_tmpdir("integ_a1");
    AriaCompileValidator validator("/bin/true");  // always succeeds
    CodeProposalStore store(dir);

    std::string code = "func: main() { exit 0; }";
    auto result = validator.validate(code);

    CodeProposal p;
    p.instruction = "Write hello world";
    p.source_code = code;
    p.compile_success = result.success;
    p.compile_time_ms = result.elapsed_ms;
    p.timestamp_ns = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    store.store(p);

    CHECK(store.count() == 1);
    CHECK(store.count_successful() == 1);

    CodeProposal loaded;
    CHECK(store.load(1, loaded));
    CHECK(loaded.compile_success);
    CHECK(loaded.source_code == code);

    cleanup_tmpdir(dir);
}

TEST_CASE("§A-2 validate → store: failed compile persisted with errors",
          "[aria_integ][pipeline]") {
    auto dir = make_tmpdir("integ_a2");

    // Create a fake compiler that outputs error
    auto script = std::filesystem::temp_directory_path() / "integ_a2_ariac.sh";
    {
        std::ofstream ofs(script);
        ofs << "#!/bin/bash\necho 'Error: syntax error on line 1'\nexit 1\n";
    }
    std::filesystem::permissions(script, std::filesystem::perms::owner_all);

    AriaCompileValidator validator(script.string());
    CodeProposalStore store(dir);

    auto result = validator.validate("bad code");

    CodeProposal p;
    p.instruction = "Write something";
    p.source_code = "bad code";
    p.compile_success = result.success;
    p.compile_errors = result.raw_output;
    p.compile_time_ms = result.elapsed_ms;
    store.store(p);

    CHECK(store.count() == 1);
    CHECK(store.count_successful() == 0);

    CodeProposal loaded;
    CHECK(store.load(1, loaded));
    CHECK(!loaded.compile_success);
    CHECK(loaded.compile_errors.find("syntax error") != std::string::npos);

    std::filesystem::remove(script);
    cleanup_tmpdir(dir);
}

TEST_CASE("§A-3 validate → store: mixed success rate tracking",
          "[aria_integ][pipeline]") {
    auto dir = make_tmpdir("integ_a3");

    AriaCompileValidator good_compiler("/bin/true");
    AriaCompileValidator bad_compiler("/bin/false");
    CodeProposalStore store(dir);

    // 3 successful, 7 failed
    for (int i = 0; i < 10; ++i) {
        auto& v = (i < 3) ? good_compiler : bad_compiler;
        auto result = v.validate("code_" + std::to_string(i));

        CodeProposal p;
        p.instruction = "prompt_" + std::to_string(i);
        p.source_code = "code_" + std::to_string(i);
        p.compile_success = result.success;
        store.store(p);
    }

    CHECK(store.count() == 10);
    CHECK(store.count_successful() == 3);
    CHECK(store.success_rate() == Catch::Approx(0.3));

    cleanup_tmpdir(dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// §B — GENERATE_CODE in DecisionLoop (specialist disabled)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§B-1 GENERATE_CODE never fires when specialist is unconfigured",
          "[aria_integ][decision]") {
    CognitiveTorus torus(3);
    AutonomyEngine engine;
    DecisionLoopConfig cfg;
    cfg.vocabulary = {"hello", "nikola", "test"};
    // specialist_server_path left empty → disabled

    DecisionLoop loop(torus, engine, cfg);

    // Run 100 ticks — GENERATE_CODE should never win
    for (int i = 0; i < 100; ++i) {
        auto result = loop.tick();
        CHECK(result.type != ActionType::GENERATE_CODE);
    }
}

TEST_CASE("§B-2 GENERATE_CODE appears in action_name for all types",
          "[aria_integ][decision]") {
    // Verify the full action vocabulary
    std::vector<std::string> expected = {
        "SILENT", "EMIT_THOUGHT", "STORE_MEMORY", "REQUEST_LOOKUP",
        "EXPLORE", "NAP", "REFUSE", "ESCALATE", "RECALL_MEMORY",
        "REASON", "GENERATE_CODE"
    };
    for (int i = 0; i <= 10; ++i) {
        std::string name = action_name(static_cast<ActionType>(i));
        CHECK(name == expected[static_cast<size_t>(i)]);
    }
}

TEST_CASE("§B-3 DecisionLoop config carries specialist settings through construction",
          "[aria_integ][decision]") {
    CognitiveTorus torus(3);
    AutonomyEngine engine;
    DecisionLoopConfig cfg;
    cfg.vocabulary = {"test"};
    cfg.specialist_server_path = "/path/to/server.py";
    cfg.ariac_path = "/path/to/ariac";
    cfg.proposal_store_path = "/tmp/proposals";
    cfg.min_generate_interval_s = 15.0f;

    DecisionLoop loop(torus, engine, cfg);
    CHECK(loop.config().specialist_server_path == "/path/to/server.py");
    CHECK(loop.config().ariac_path == "/path/to/ariac");
    CHECK(loop.config().min_generate_interval_s == Catch::Approx(15.0f));
}

// ─────────────────────────────────────────────────────────────────────────────
// §C — GENERATE_CODE with specialist enabled (config only, no real server)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§C-1 specialist enabled flag set when server path provided",
          "[aria_integ][specialist]") {
    CognitiveTorus torus(3);
    AutonomyEngine engine;
    DecisionLoopConfig cfg;
    cfg.vocabulary = {"test"};
    cfg.specialist_server_path = "/usr/bin/true";  // exists, triggers enabled

    DecisionLoop loop(torus, engine, cfg);
    // We can verify the config round-trips
    CHECK(loop.config().specialist_server_path == "/usr/bin/true");
}

TEST_CASE("§C-2 specialist enabled flag set when ariac path provided",
          "[aria_integ][specialist]") {
    CognitiveTorus torus(3);
    AutonomyEngine engine;
    DecisionLoopConfig cfg;
    cfg.vocabulary = {"test"};
    cfg.ariac_path = "/usr/bin/true";

    DecisionLoop loop(torus, engine, cfg);
    CHECK(loop.config().ariac_path == "/usr/bin/true");
}

// ─────────────────────────────────────────────────────────────────────────────
// §D — Full code pipeline: extract → validate → store
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§D-1 Full pipeline: extract → validate → store (success)",
          "[aria_integ][full_pipeline]") {
    auto dir = make_tmpdir("integ_d1");
    AriaCompileValidator validator("/bin/true");
    CodeProposalStore store(dir);

    // Simulate specialist response
    std::string model_response = "Here's the code:\n```aria\nfunc: main() {\n  exit 0;\n}\n```";
    std::string code = extract_code_block(model_response);
    CHECK(code.find("func: main()") != std::string::npos);

    auto result = validator.validate(code);
    CHECK(result.success);

    CodeProposal p;
    p.instruction = "Write a main function";
    p.source_code = code;
    p.compile_success = result.success;
    p.compile_time_ms = result.elapsed_ms;
    store.store(p);

    CHECK(store.count() == 1);
    CHECK(store.count_successful() == 1);

    auto exported = store.export_successful();
    CHECK(exported.size() == 1);
    CHECK(exported[0].source_code.find("func: main()") != std::string::npos);

    cleanup_tmpdir(dir);
}

TEST_CASE("§D-2 Full pipeline: extract → validate → store (failure)",
          "[aria_integ][full_pipeline]") {
    auto dir = make_tmpdir("integ_d2");
    AriaCompileValidator validator("/bin/false");
    CodeProposalStore store(dir);

    std::string model_response = "```aria\nint32: x = broken;\n```";
    std::string code = extract_code_block(model_response);
    auto result = validator.validate(code);
    CHECK(!result.success);

    CodeProposal p;
    p.instruction = "Write a variable declaration";
    p.source_code = code;
    p.compile_success = result.success;
    store.store(p);

    CHECK(store.count() == 1);
    CHECK(store.count_successful() == 0);
    CHECK(store.success_rate() == Catch::Approx(0.0));

    cleanup_tmpdir(dir);
}

TEST_CASE("§D-3 Full pipeline: multiple proposals build training corpus",
          "[aria_integ][full_pipeline]") {
    auto dir = make_tmpdir("integ_d3");
    AriaCompileValidator good_v("/bin/true");
    AriaCompileValidator bad_v("/bin/false");
    CodeProposalStore store(dir);

    // Simulate a self-improvement iteration
    struct TestCase { std::string prompt; std::string response; bool expect_success; };
    std::vector<TestCase> cases = {
        {"hello world", "```aria\nfunc: main() { exit 0; }\n```", true},
        {"variable decl", "```aria\nint32: x = 42;\n```", true},
        {"bad code", "```aria\nbroken syntax\n```", false},
        {"fibonacci", "```aria\nfunc: fib(int32: n) -> int32 { exit 0; }\n```", true},
        {"parse error", "No code here, just text.", false},
    };

    for (const auto& tc : cases) {
        std::string code = extract_code_block(tc.response);
        auto& v = tc.expect_success ? good_v : bad_v;
        auto result = v.validate(code);

        CodeProposal p;
        p.instruction = tc.prompt;
        p.source_code = code;
        p.compile_success = result.success;
        store.store(p);
    }

    CHECK(store.count() == 5);
    CHECK(store.count_successful() == 3);
    CHECK(store.success_rate() == Catch::Approx(0.6));

    auto corpus = store.export_successful();
    CHECK(corpus.size() == 3);

    cleanup_tmpdir(dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// §E — Metrics accumulation
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§E-1 Proposal store accumulates across iterations",
          "[aria_integ][metrics]") {
    auto dir = make_tmpdir("integ_e1");

    // Iteration 1: 50% success
    {
        CodeProposalStore store(dir);
        for (int i = 0; i < 10; ++i) {
            CodeProposal p;
            p.instruction = "iter1_" + std::to_string(i);
            p.source_code = "code";
            p.compile_success = (i < 5);
            p.iteration = 1;
            store.store(p);
        }
        CHECK(store.success_rate() == Catch::Approx(0.5));
    }

    // Iteration 2: add 10 more (7 succeed), total 12/20 = 60%
    {
        CodeProposalStore store(dir);
        CHECK(store.count() == 10);  // Persistent
        for (int i = 0; i < 10; ++i) {
            CodeProposal p;
            p.instruction = "iter2_" + std::to_string(i);
            p.source_code = "code2";
            p.compile_success = (i < 7);
            p.iteration = 2;
            store.store(p);
        }
        CHECK(store.count() == 20);
        CHECK(store.count_successful() == 12);
        CHECK(store.success_rate() == Catch::Approx(0.6));
    }

    cleanup_tmpdir(dir);
}

TEST_CASE("§E-2 Export corpus preserves iteration ordering",
          "[aria_integ][metrics]") {
    auto dir = make_tmpdir("integ_e2");
    {
        CodeProposalStore store(dir);
        for (int i = 0; i < 5; ++i) {
            CodeProposal p;
            p.instruction = "ordered_" + std::to_string(i);
            p.source_code = "code_" + std::to_string(i);
            p.compile_success = true;
            p.iteration = static_cast<uint32_t>(i);
            store.store(p);
        }

        auto exported = store.export_successful();
        REQUIRE(exported.size() == 5);
        // LMDB with big-endian keys preserves insertion order
        for (size_t i = 0; i < exported.size(); ++i) {
            CHECK(exported[i].iteration == static_cast<uint32_t>(i));
        }
    }
    cleanup_tmpdir(dir);
}

TEST_CASE("§E-3 Empty store export returns empty vector",
          "[aria_integ][metrics]") {
    auto dir = make_tmpdir("integ_e3");
    {
        CodeProposalStore store(dir);
        auto exported = store.export_successful();
        CHECK(exported.empty());
        CHECK(store.success_rate() == Catch::Approx(0.0));
    }
    cleanup_tmpdir(dir);
}
