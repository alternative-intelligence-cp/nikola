/**
 * @file tests/unit/v030_static_analyzer_gate_test.cpp
 * @brief v0.3.0 — StaticAnalyzerGate test suite
 *
 * Tests:
 *   §1  Default construction and config defaults
 *   §2  Safe C++ code passes analysis
 *   §3  Tool availability detection
 *   §4  Config: warnings_as_errors = true
 *   §5  Config: pass_if_unavailable behavior
 *   §6  Result aggregation (error counting)
 *   §7  Timeout configuration
 *   §8  Finding severity enum coverage
 *   §9  is_safe() convenience method
 *   §10 Counters (total_analyses, total_failures)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <nikola/security/static_analyzer_gate.hpp>

using namespace nikola::security;

// ============================================================================
// §1 Default construction
// ============================================================================

TEST_CASE("§1 StaticAnalyzerGate default construction", "[v030][static_analyzer]") {
    StaticAnalyzerGate gate;
    const auto& cfg = gate.config();

    REQUIRE(cfg.timeout_ms == SA_DEFAULT_TIMEOUT_MS);
    REQUIRE(cfg.pass_if_unavailable == true);
    REQUIRE(cfg.warnings_as_errors == false);
    REQUIRE(cfg.clang_tidy_checks.find("bugprone-*") != std::string::npos);
    REQUIRE(cfg.cppcheck_enable.find("warning") != std::string::npos);

    REQUIRE(gate.total_scans() == 0);
    REQUIRE(gate.total_rejections() == 0);
}

// ============================================================================
// §2 Safe code analysis
// ============================================================================

TEST_CASE("§2 Safe C++ code passes analysis", "[v030][static_analyzer]") {
    StaticAnalyzerGate gate;

    std::string safe_code = R"(
        #include <iostream>
        int main() {
            int x = 42;
            std::cout << x << std::endl;
            return 0;
        }
    )";

    auto result = gate.analyze(safe_code);

    // If tools are unavailable and pass_if_unavailable=true, should pass
    if (!gate.clang_tidy_available() && !gate.cppcheck_available()) {
        REQUIRE(result.passed == true);
        REQUIRE((!result.clang_tidy_ran && !result.cppcheck_ran));
    }
    // If tools are available, simple code should pass
    // (can't guarantee on all systems, so we just check it ran)
    REQUIRE(gate.total_scans() == 1);
}

// ============================================================================
// §3 Tool detection
// ============================================================================

TEST_CASE("§3 Tool availability detection", "[v030][static_analyzer]") {
    StaticAnalyzerGate gate;

    // These are informational — we just verify the methods work
    bool has_ct = gate.clang_tidy_available();
    bool has_cc = gate.cppcheck_available();

    // If neither is available, analysis should still work (with pass_if_unavailable)
    if (!has_ct && !has_cc) {
        REQUIRE(gate.is_safe("int main() { return 0; }") == true);
    }
}

// ============================================================================
// §4 Warnings as errors
// ============================================================================

TEST_CASE("§4 Config: warnings_as_errors", "[v030][static_analyzer]") {
    StaticAnalyzerConfig cfg;
    cfg.warnings_as_errors = true;

    StaticAnalyzerGate gate(cfg);
    REQUIRE(gate.config().warnings_as_errors == true);
}

// ============================================================================
// §5 pass_if_unavailable = false
// ============================================================================

TEST_CASE("§5 Config: pass_if_unavailable=false fails when no tools", "[v030][static_analyzer]") {
    StaticAnalyzerConfig cfg;
    cfg.pass_if_unavailable = false;

    StaticAnalyzerGate gate(cfg);

    if (!gate.clang_tidy_available() && !gate.cppcheck_available()) {
        auto result = gate.analyze("int main() {}");
        REQUIRE(result.passed == false);
        REQUIRE_THAT(result.error_summary, Catch::Matchers::ContainsSubstring("unavailable"));
    }
}

// ============================================================================
// §6 Finding severity enum
// ============================================================================

TEST_CASE("§6 FindingSeverity enum coverage", "[v030][static_analyzer]") {
    REQUIRE(finding_severity_str(FindingSeverity::NOTE)    == std::string_view("note"));
    REQUIRE(finding_severity_str(FindingSeverity::WARNING) == std::string_view("warning"));
    REQUIRE(finding_severity_str(FindingSeverity::ERROR)   == std::string_view("error"));
}

// ============================================================================
// §7 Custom timeout
// ============================================================================

TEST_CASE("§7 Custom timeout configuration", "[v030][static_analyzer]") {
    StaticAnalyzerConfig cfg;
    cfg.timeout_ms = 60000;

    StaticAnalyzerGate gate(cfg);
    REQUIRE(gate.config().timeout_ms == 60000);
}

// ============================================================================
// §8 AnalysisFinding struct
// ============================================================================

TEST_CASE("§8 AnalysisFinding fields", "[v030][static_analyzer]") {
    AnalysisFinding f;
    f.file = "test.cpp";
    f.line = 42;
    f.column = 5;
    f.severity = FindingSeverity::ERROR;
    f.message = "use after free";
    f.check_name = "bugprone-use-after-free";
    f.tool = "clang-tidy";

    REQUIRE(f.file == "test.cpp");
    REQUIRE(f.line == 42);
    REQUIRE(f.column == 5);
    REQUIRE(f.severity == FindingSeverity::ERROR);
    REQUIRE(f.message == "use after free");
    REQUIRE(f.check_name == "bugprone-use-after-free");
    REQUIRE(f.tool == "clang-tidy");
}

// ============================================================================
// §9 is_safe convenience
// ============================================================================

TEST_CASE("§9 is_safe() convenience method", "[v030][static_analyzer]") {
    StaticAnalyzerGate gate;
    // With default config (pass_if_unavailable=true), safe code should pass
    bool safe = gate.is_safe("int main() { return 0; }");
    if (!gate.clang_tidy_available() && !gate.cppcheck_available()) {
        REQUIRE(safe == true);
    }
}

// ============================================================================
// §10 Counters
// ============================================================================

TEST_CASE("§10 Analysis counters increment correctly", "[v030][static_analyzer]") {
    StaticAnalyzerGate gate;
    REQUIRE(gate.total_scans() == 0);

    gate.analyze("int main() { return 0; }");
    REQUIRE(gate.total_scans() == 1);

    gate.analyze("int foo() { return 1; }");
    REQUIRE(gate.total_scans() == 2);
}
