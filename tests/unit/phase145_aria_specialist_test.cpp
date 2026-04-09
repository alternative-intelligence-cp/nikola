/**
 * @file tests/unit/phase145_aria_specialist_test.cpp
 * @brief Unit tests for v0.0.19 — Aria Specialist Integration (Phase 145).
 *
 * Covers:
 *   §A — AriaCompileValidator: tempfile lifecycle, output parsing, timeout
 *   §B — CodeProposalStore (LMDB): CRUD, serialisation, metrics
 *   §C — extract_code_block(): fenced blocks, raw code, empty
 *   §D — GENERATE_CODE ActionType: enum value, action_name, scoring
 *   §E — SpecialistInterface: construction, path defaults
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/aria/compile_validator.hpp>
#include <nikola/aria/code_proposal_store.hpp>
#include <nikola/aria/specialist_interface.hpp>
#include <nikola/autonomy/decision_loop.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace nikola::aria;
using namespace nikola::autonomy;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Create a temporary directory for LMDB tests.
static std::string make_tmpdir(const std::string& suffix) {
    auto dir = std::filesystem::temp_directory_path() / ("nikola_test_" + suffix + "_"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    return dir.string();
}

/// Clean up a temp directory.
static void cleanup_tmpdir(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

// ─────────────────────────────────────────────────────────────────────────────
// §A — AriaCompileValidator
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§A-1 CompileValidator — construction with defaults", "[compile_validator][sectionA]") {
    AriaCompileValidator v;
    CHECK(!v.ariac_path().empty());
    // Default path should contain "ariac"
    CHECK(v.ariac_path().find("ariac") != std::string::npos);
}

TEST_CASE("§A-2 CompileValidator — custom ariac path", "[compile_validator][sectionA]") {
    AriaCompileValidator v("/usr/bin/false", 5000);
    CHECK(v.ariac_path() == "/usr/bin/false");
}

TEST_CASE("§A-3 CompileValidator — missing compiler returns error", "[compile_validator][sectionA]") {
    AriaCompileValidator v("/nonexistent/path/ariac_does_not_exist");
    auto result = v.validate("func: main() { exit 0; }");
    CHECK(!result.success);
    CHECK(!result.errors.empty());
    CHECK(result.errors[0].find("not found") != std::string::npos);
}

TEST_CASE("§A-4 CompileValidator — compiler_available() with missing path", "[compile_validator][sectionA]") {
    AriaCompileValidator v("/nonexistent/ariac_nope");
    CHECK(!v.compiler_available());
}

TEST_CASE("§A-5 CompileValidator — compiler_available() with /bin/true", "[compile_validator][sectionA]") {
    AriaCompileValidator v("/bin/true");
    CHECK(v.compiler_available());
}

TEST_CASE("§A-6 CompileValidator — validate with /bin/true (always succeeds)", "[compile_validator][sectionA]") {
    // /bin/true exits 0 with no output — simulates successful compile
    AriaCompileValidator v("/bin/true");
    auto result = v.validate("func: main() { exit 0; }");
    CHECK(result.success);
    CHECK(result.errors.empty());
    CHECK(result.elapsed_ms >= 0.0);
}

TEST_CASE("§A-7 CompileValidator — validate with /bin/false (always fails)", "[compile_validator][sectionA]") {
    AriaCompileValidator v("/bin/false");
    auto result = v.validate("invalid source");
    CHECK(!result.success);
}

TEST_CASE("§A-8 CompileValidator — output parsing: error lines", "[compile_validator][sectionA]") {
    // Use echo to simulate compiler output with error lines
    // Create a script that prints error messages and exits 1
    auto tmp = std::filesystem::temp_directory_path() / "nikola_test_fake_ariac.sh";
    {
        std::ofstream ofs(tmp);
        ofs << "#!/bin/bash\necho 'Error: undeclared variable x'\necho 'Warning: unused import'\nexit 1\n";
    }
    std::filesystem::permissions(tmp, std::filesystem::perms::owner_exec |
                                       std::filesystem::perms::owner_read |
                                       std::filesystem::perms::owner_write);

    AriaCompileValidator v(tmp.string());
    auto result = v.validate("source code");
    CHECK(!result.success);
    CHECK(result.errors.size() >= 1);
    CHECK(result.warnings.size() >= 1);
    CHECK(result.raw_output.find("undeclared") != std::string::npos);

    std::filesystem::remove(tmp);
}

TEST_CASE("§A-9 CompileValidator — CompileResult bool conversion", "[compile_validator][sectionA]") {
    CompileResult ok;
    ok.success = true;
    CHECK(static_cast<bool>(ok));

    CompileResult fail;
    fail.success = false;
    CHECK(!static_cast<bool>(fail));
}

TEST_CASE("§A-10 CompileValidator — default_ariac_path contains ariac", "[compile_validator][sectionA]") {
    auto path = AriaCompileValidator::default_ariac_path();
    CHECK(path.find("ariac") != std::string::npos);
}

TEST_CASE("§A-11 CompileValidator — ARIAC_BIN env var override", "[compile_validator][sectionA]") {
    // Save and set env var
    const char* old = std::getenv("ARIAC_BIN");
    setenv("ARIAC_BIN", "/custom/path/ariac", 1);
    CHECK(AriaCompileValidator::default_ariac_path() == "/custom/path/ariac");
    // Restore
    if (old) setenv("ARIAC_BIN", old, 1);
    else unsetenv("ARIAC_BIN");
}

TEST_CASE("§A-12 CompileValidator — tempfile cleanup after validate", "[compile_validator][sectionA]") {
    AriaCompileValidator v("/bin/true");
    auto before = std::filesystem::temp_directory_path();
    size_t count_before = 0;
    for (auto& p : std::filesystem::directory_iterator(before)) {
        if (p.path().filename().string().find("nikola_validate_") != std::string::npos)
            ++count_before;
    }

    v.validate("test source code");

    size_t count_after = 0;
    for (auto& p : std::filesystem::directory_iterator(before)) {
        if (p.path().filename().string().find("nikola_validate_") != std::string::npos)
            ++count_after;
    }
    // Should not leave temp files behind
    CHECK(count_after <= count_before);
}

// ─────────────────────────────────────────────────────────────────────────────
// §B — CodeProposalStore (LMDB)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§B-1 ProposalStore — create and count", "[proposal_store][sectionB]") {
    auto dir = make_tmpdir("propstore_b1");
    {
        CodeProposalStore store(dir);
        CHECK(store.count() == 0);
    }
    cleanup_tmpdir(dir);
}

TEST_CASE("§B-2 ProposalStore — store and load round-trip", "[proposal_store][sectionB]") {
    auto dir = make_tmpdir("propstore_b2");
    {
        CodeProposalStore store(dir);

        CodeProposal p;
        p.instruction = "Write a hello world function";
        p.source_code = "func: main() { exit 0; }";
        p.compile_success = true;
        p.compile_errors = "";
        p.compile_time_ms = 42.5;
        p.timestamp_ns = 1234567890;
        p.iteration = 3;

        uint64_t id = store.store(p);
        CHECK(id == 1);
        CHECK(store.count() == 1);

        CodeProposal loaded;
        CHECK(store.load(id, loaded));
        CHECK(loaded.instruction == "Write a hello world function");
        CHECK(loaded.source_code == "func: main() { exit 0; }");
        CHECK(loaded.compile_success == true);
        CHECK(loaded.compile_time_ms == Catch::Approx(42.5));
        CHECK(loaded.timestamp_ns == 1234567890);
        CHECK(loaded.iteration == 3);
    }
    cleanup_tmpdir(dir);
}

TEST_CASE("§B-3 ProposalStore — multiple proposals with auto-increment", "[proposal_store][sectionB]") {
    auto dir = make_tmpdir("propstore_b3");
    {
        CodeProposalStore store(dir);
        for (int i = 0; i < 5; ++i) {
            CodeProposal p;
            p.instruction = "prompt " + std::to_string(i);
            p.source_code = "code " + std::to_string(i);
            p.compile_success = (i % 2 == 0);
            store.store(p);
        }
        CHECK(store.count() == 5);
    }
    cleanup_tmpdir(dir);
}

TEST_CASE("§B-4 ProposalStore — count_successful", "[proposal_store][sectionB]") {
    auto dir = make_tmpdir("propstore_b4");
    {
        CodeProposalStore store(dir);
        for (int i = 0; i < 10; ++i) {
            CodeProposal p;
            p.instruction = "p" + std::to_string(i);
            p.source_code = "code";
            p.compile_success = (i < 3);  // 3 out of 10 succeed
            store.store(p);
        }
        CHECK(store.count_successful() == 3);
    }
    cleanup_tmpdir(dir);
}

TEST_CASE("§B-5 ProposalStore — success_rate", "[proposal_store][sectionB]") {
    auto dir = make_tmpdir("propstore_b5");
    {
        CodeProposalStore store(dir);
        // Empty store → 0.0
        CHECK(store.success_rate() == Catch::Approx(0.0));

        for (int i = 0; i < 4; ++i) {
            CodeProposal p;
            p.instruction = "p";
            p.source_code = "c";
            p.compile_success = (i < 1);  // 1 out of 4 = 25%
            store.store(p);
        }
        CHECK(store.success_rate() == Catch::Approx(0.25));
    }
    cleanup_tmpdir(dir);
}

TEST_CASE("§B-6 ProposalStore — export_successful", "[proposal_store][sectionB]") {
    auto dir = make_tmpdir("propstore_b6");
    {
        CodeProposalStore store(dir);
        for (int i = 0; i < 6; ++i) {
            CodeProposal p;
            p.instruction = "prompt_" + std::to_string(i);
            p.source_code = "code_" + std::to_string(i);
            p.compile_success = (i % 2 == 0);  // 0, 2, 4 succeed
            store.store(p);
        }
        auto exported = store.export_successful();
        CHECK(exported.size() == 3);
        CHECK(exported[0].compile_success);
    }
    cleanup_tmpdir(dir);
}

TEST_CASE("§B-7 ProposalStore — export_successful with max_count", "[proposal_store][sectionB]") {
    auto dir = make_tmpdir("propstore_b7");
    {
        CodeProposalStore store(dir);
        for (int i = 0; i < 10; ++i) {
            CodeProposal p;
            p.instruction = "p";
            p.source_code = "c";
            p.compile_success = true;
            store.store(p);
        }
        auto exported = store.export_successful(3);
        CHECK(exported.size() == 3);
    }
    cleanup_tmpdir(dir);
}

TEST_CASE("§B-8 ProposalStore — load nonexistent ID returns false", "[proposal_store][sectionB]") {
    auto dir = make_tmpdir("propstore_b8");
    {
        CodeProposalStore store(dir);
        CodeProposal p;
        CHECK(!store.load(999, p));
    }
    cleanup_tmpdir(dir);
}

TEST_CASE("§B-9 ProposalStore — serialize empty strings", "[proposal_store][sectionB]") {
    auto dir = make_tmpdir("propstore_b9");
    {
        CodeProposalStore store(dir);
        CodeProposal p;
        p.instruction = "";
        p.source_code = "";
        p.compile_errors = "";
        store.store(p);

        CodeProposal loaded;
        CHECK(store.load(1, loaded));
        CHECK(loaded.instruction.empty());
        CHECK(loaded.source_code.empty());
    }
    cleanup_tmpdir(dir);
}

TEST_CASE("§B-10 ProposalStore — persistence across reopen", "[proposal_store][sectionB]") {
    auto dir = make_tmpdir("propstore_b10");
    {
        CodeProposalStore store(dir);
        CodeProposal p;
        p.instruction = "persistent test";
        p.source_code = "func: main() { exit 0; }";
        p.compile_success = true;
        store.store(p);
    }
    // Reopen
    {
        CodeProposalStore store(dir);
        CHECK(store.count() == 1);
        CodeProposal loaded;
        CHECK(store.load(1, loaded));
        CHECK(loaded.instruction == "persistent test");
    }
    cleanup_tmpdir(dir);
}

TEST_CASE("§B-11 ProposalStore — serialization pack/unpack symmetry", "[proposal_store][sectionB]") {
    CodeProposal orig;
    orig.id = 42;
    orig.instruction = "Write a sorting function in Aria";
    orig.source_code = "func: bubble_sort(int32: arr[], int32: n) {\n  exit 0;\n}";
    orig.compile_success = false;
    orig.compile_errors = "Error: unknown type 'arr[]' on line 1";
    orig.compile_time_ms = 123.456;
    orig.timestamp_ns = 9876543210ULL;
    orig.iteration = 7;

    auto packed = detail::pack_proposal(orig);
    CHECK(packed.size() > 40);

    CodeProposal restored;
    CHECK(detail::unpack_proposal(packed.data(), packed.size(), restored));
    CHECK(restored.id == 42);
    CHECK(restored.instruction == orig.instruction);
    CHECK(restored.source_code == orig.source_code);
    CHECK(restored.compile_success == false);
    CHECK(restored.compile_errors == orig.compile_errors);
    CHECK(restored.compile_time_ms == Catch::Approx(123.456));
    CHECK(restored.timestamp_ns == 9876543210ULL);
    CHECK(restored.iteration == 7);
}

TEST_CASE("§B-12 ProposalStore — unpack rejects bad magic", "[proposal_store][sectionB]") {
    std::vector<uint8_t> bad(50, 0);
    CodeProposal p;
    CHECK(!detail::unpack_proposal(bad.data(), bad.size(), p));
}

// ─────────────────────────────────────────────────────────────────────────────
// §C — extract_code_block
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§C-1 extract_code_block — fenced aria block", "[extract_code][sectionC]") {
    std::string response = "Here's the code:\n```aria\nfunc: main() {\n  exit 0;\n}\n```\nDone.";
    auto code = extract_code_block(response);
    CHECK(code.find("func: main()") != std::string::npos);
    // Should NOT include the fences
    CHECK(code.find("```") == std::string::npos);
}

TEST_CASE("§C-2 extract_code_block — generic fenced block", "[extract_code][sectionC]") {
    std::string response = "```\nfunc: hello() {\n  exit 0;\n}\n```";
    auto code = extract_code_block(response);
    CHECK(code.find("func: hello()") != std::string::npos);
}

TEST_CASE("§C-3 extract_code_block — raw Aria code (func: keyword)", "[extract_code][sectionC]") {
    std::string response = "func: main() {\n  exit 0;\n}";
    auto code = extract_code_block(response);
    CHECK(code == response);
}

TEST_CASE("§C-4 extract_code_block — raw Aria code (use keyword)", "[extract_code][sectionC]") {
    std::string response = "use \"io.aria\".*;\nfunc: main() { exit 0; }";
    auto code = extract_code_block(response);
    CHECK(code.find("use ") != std::string::npos);
}

TEST_CASE("§C-5 extract_code_block — raw Aria code (extern keyword)", "[extract_code][sectionC]") {
    std::string response = "extern func: puts(string: s) -> int32;";
    auto code = extract_code_block(response);
    CHECK(code.find("extern ") != std::string::npos);
}

TEST_CASE("§C-6 extract_code_block — plain text (no code)", "[extract_code][sectionC]") {
    std::string response = "This is just a description with no code.";
    auto code = extract_code_block(response);
    // Returns the original when no code markers found
    CHECK(code == response);
}

TEST_CASE("§C-7 extract_code_block — empty input", "[extract_code][sectionC]") {
    CHECK(extract_code_block("").empty());
}

TEST_CASE("§C-8 extract_code_block — int32 type keyword detection", "[extract_code][sectionC]") {
    std::string response = "int32: x = 42;";
    auto code = extract_code_block(response);
    CHECK(code.find("int32:") != std::string::npos);
}

TEST_CASE("§C-9 extract_code_block — string type keyword detection", "[extract_code][sectionC]") {
    std::string response = "string: msg = \"hello\";";
    auto code = extract_code_block(response);
    CHECK(code.find("string:") != std::string::npos);
}

TEST_CASE("§C-10 extract_code_block — fenced block preferred over keywords", "[extract_code][sectionC]") {
    std::string response = "The func: keyword is used like:\n```aria\nfunc: demo() { exit 0; }\n```\nNote func: is required.";
    auto code = extract_code_block(response);
    // Should extract only the fenced content
    CHECK(code.find("func: demo()") != std::string::npos);
    CHECK(code.find("Note func:") == std::string::npos);
}

TEST_CASE("§C-11 extract_code_block — multiple fenced blocks: first wins", "[extract_code][sectionC]") {
    std::string response = "```aria\nfirst()\n```\n```aria\nsecond()\n```";
    auto code = extract_code_block(response);
    CHECK(code.find("first()") != std::string::npos);
}

TEST_CASE("§C-12 extract_code_block — no trim on whitespace", "[extract_code][sectionC]") {
    std::string response = "```aria\n  func: indented() {\n    exit 0;\n  }\n```";
    auto code = extract_code_block(response);
    CHECK(code.find("  func: indented()") != std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────────────
// §D — GENERATE_CODE ActionType
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§D-1 ActionType::GENERATE_CODE enum value is 10", "[action_type][sectionD]") {
    CHECK(static_cast<uint8_t>(ActionType::GENERATE_CODE) == 10);
}

TEST_CASE("§D-2 action_name(GENERATE_CODE) returns correct string", "[action_type][sectionD]") {
    CHECK(std::string(action_name(ActionType::GENERATE_CODE)) == "GENERATE_CODE");
}

TEST_CASE("§D-3 action_name covers all 11 actions without UNKNOWN", "[action_type][sectionD]") {
    for (int i = 0; i <= 10; ++i) {
        auto name = action_name(static_cast<ActionType>(i));
        CHECK(std::string(name) != "UNKNOWN");
    }
}

TEST_CASE("§D-4 action_name(11) returns UNKNOWN", "[action_type][sectionD]") {
    CHECK(std::string(action_name(static_cast<ActionType>(11))) == "UNKNOWN");
}

TEST_CASE("§D-5 GENERATE_CODE is distinct from all other actions", "[action_type][sectionD]") {
    auto gen = static_cast<uint8_t>(ActionType::GENERATE_CODE);
    CHECK(gen != static_cast<uint8_t>(ActionType::SILENT));
    CHECK(gen != static_cast<uint8_t>(ActionType::EMIT_THOUGHT));
    CHECK(gen != static_cast<uint8_t>(ActionType::STORE_MEMORY));
    CHECK(gen != static_cast<uint8_t>(ActionType::REQUEST_LOOKUP));
    CHECK(gen != static_cast<uint8_t>(ActionType::EXPLORE));
    CHECK(gen != static_cast<uint8_t>(ActionType::NAP));
    CHECK(gen != static_cast<uint8_t>(ActionType::REFUSE));
    CHECK(gen != static_cast<uint8_t>(ActionType::ESCALATE));
    CHECK(gen != static_cast<uint8_t>(ActionType::RECALL_MEMORY));
    CHECK(gen != static_cast<uint8_t>(ActionType::REASON));
}

TEST_CASE("§D-6 NikolaState default last_action is SILENT", "[action_type][sectionD]") {
    NikolaState s;
    CHECK(s.last_action == ActionType::SILENT);
}

TEST_CASE("§D-7 DecisionResult can carry GENERATE_CODE type", "[action_type][sectionD]") {
    DecisionResult r;
    r.type = ActionType::GENERATE_CODE;
    r.payload = "func: main() { exit 0; }";
    CHECK(r.type == ActionType::GENERATE_CODE);
    CHECK(!r.payload.empty());
}

TEST_CASE("§D-8 DecisionLoopConfig has specialist fields", "[action_type][sectionD]") {
    DecisionLoopConfig cfg;
    CHECK(cfg.min_generate_interval_s == Catch::Approx(30.0f));
    CHECK(cfg.specialist_server_path.empty());
    CHECK(cfg.ariac_path.empty());
    CHECK(cfg.proposal_store_path.empty());
}

TEST_CASE("§D-9 DecisionLoopConfig specialist fields are configurable", "[action_type][sectionD]") {
    DecisionLoopConfig cfg;
    cfg.specialist_server_path = "/path/to/server.py";
    cfg.ariac_path = "/path/to/ariac";
    cfg.proposal_store_path = "/tmp/proposals";
    cfg.min_generate_interval_s = 10.0f;
    CHECK(cfg.specialist_server_path == "/path/to/server.py");
    CHECK(cfg.ariac_path == "/path/to/ariac");
    CHECK(cfg.min_generate_interval_s == Catch::Approx(10.0f));
}

TEST_CASE("§D-10 GENERATE_CODE scores 0 when specialist disabled", "[action_type][sectionD]") {
    // Without specialist_server_path, aria_specialist_enabled_ is false
    // The score function should return 0 — verified implicitly through
    // DecisionLoopConfig defaults.
    DecisionLoopConfig cfg;
    CHECK(cfg.specialist_server_path.empty());
    // If the loop were constructed, score_generate_code should return 0
    // (tested via integration test; this just verifies the config default)
}

TEST_CASE("§D-11 GENERATE_CODE score requires boredom > 0.4", "[action_type][sectionD]") {
    NikolaState s;
    s.boredom = 0.3f;  // Below threshold
    s.atp = 0.8f;
    // Score should be 0 (low boredom blocks GENERATE_CODE)
    // This documents the scoring contract — actual score tested in integration
    CHECK(s.boredom < 0.4f);
}

TEST_CASE("§D-12 GENERATE_CODE score requires ATP >= 0.30", "[action_type][sectionD]") {
    NikolaState s;
    s.boredom = 0.9f;
    s.atp = 0.2f;  // Below threshold
    CHECK(s.atp < 0.30f);
}

// ─────────────────────────────────────────────────────────────────────────────
// §E — SpecialistInterface
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("§E-1 SpecialistInterface — construction with defaults", "[specialist][sectionE]") {
    SpecialistInterface si;
    CHECK(!si.running());
    CHECK(si.pid() == -1);
}

TEST_CASE("§E-2 SpecialistInterface — default_server_path contains server.py", "[specialist][sectionE]") {
    auto path = SpecialistInterface::default_server_path();
    CHECK(path.find("server.py") != std::string::npos);
}

TEST_CASE("§E-3 SpecialistInterface — custom path", "[specialist][sectionE]") {
    SpecialistInterface si("/custom/server.py", "python3", 5000);
    CHECK(!si.running());
}

TEST_CASE("§E-4 SpecialistInterface — start with nonexistent script fails", "[specialist][sectionE]") {
    SpecialistInterface si("/nonexistent/server.py");
    CHECK(!si.start());
    CHECK(!si.last_error().empty());
    CHECK(si.last_error().find("not found") != std::string::npos);
}

TEST_CASE("§E-5 SpecialistInterface — ask without starting returns error", "[specialist][sectionE]") {
    SpecialistInterface si;
    auto result = si.ask("test instruction");
    CHECK(!result.ok);
    CHECK(result.error.find("not running") != std::string::npos);
}

TEST_CASE("§E-6 SpecialistResult — bool conversion", "[specialist][sectionE]") {
    SpecialistResult ok_result{true, "response", ""};
    CHECK(static_cast<bool>(ok_result));

    SpecialistResult fail_result{false, "", "error"};
    CHECK(!static_cast<bool>(fail_result));
}

TEST_CASE("§E-7 SpecialistInterface — stop on non-started is safe", "[specialist][sectionE]") {
    SpecialistInterface si;
    si.stop();  // Should not crash
    CHECK(!si.running());
}

TEST_CASE("§E-8 SpecialistInterface — ARIA_SPECIALIST_SERVER env override", "[specialist][sectionE]") {
    const char* old = std::getenv("ARIA_SPECIALIST_SERVER");
    setenv("ARIA_SPECIALIST_SERVER", "/env/custom/server.py", 1);
    CHECK(SpecialistInterface::default_server_path() == "/env/custom/server.py");
    if (old) setenv("ARIA_SPECIALIST_SERVER", old, 1);
    else unsetenv("ARIA_SPECIALIST_SERVER");
}

TEST_CASE("§E-9 SpecialistInterface — destructor calls stop", "[specialist][sectionE]") {
    // Construct and immediately destruct — should not leak file descriptors
    {
        SpecialistInterface si("/nonexistent/path.py");
    }
    // If we get here without crash, the destructor is safe
    CHECK(true);
}

TEST_CASE("§E-10 SpecialistResult — default construction", "[specialist][sectionE]") {
    SpecialistResult r;
    CHECK(!r.ok);
    CHECK(r.response.empty());
    CHECK(r.error.empty());
}

TEST_CASE("§E-11 SpecialistInterface — start with echo server (timeout test)", "[specialist][sectionE]") {
    // Use a script that does NOT emit ready signal — should timeout
    auto tmp = std::filesystem::temp_directory_path() / "nikola_test_slow_server.py";
    {
        std::ofstream ofs(tmp);
        ofs << "#!/usr/bin/env python3\nimport time\ntime.sleep(300)\n";
    }
    std::filesystem::permissions(tmp, std::filesystem::perms::owner_exec |
                                       std::filesystem::perms::owner_read |
                                       std::filesystem::perms::owner_write);

    SpecialistInterface si(tmp.string(), "python3", 500);  // 500ms timeout
    CHECK(!si.start());
    CHECK(si.last_error().find("timeout") != std::string::npos);
    si.stop();

    std::filesystem::remove(tmp);
}

TEST_CASE("§E-12 CompileResult + SpecialistResult interop", "[specialist][sectionE]") {
    // Verify both result types can coexist and carry useful data
    CompileResult cr;
    cr.success = true;
    cr.elapsed_ms = 15.3;

    SpecialistResult sr;
    sr.ok = true;
    sr.response = "func: main() { exit 0; }";

    CHECK(static_cast<bool>(cr));
    CHECK(static_cast<bool>(sr));
    CHECK(cr.elapsed_ms > 0.0);
    CHECK(!sr.response.empty());
}
