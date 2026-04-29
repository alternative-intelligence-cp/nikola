/**
 * @file security/static_analyzer_gate.hpp
 * @brief v0.3.0 — Static Analysis Gate (clang-tidy + cppcheck)
 *
 * Wraps clang-tidy and cppcheck as an optional SIE validation gate,
 * inserted between Gate 1 (CSVP) and Gate 2 (PhysicsOracle).
 *
 * If neither tool is available on the system, the gate logs a warning
 * and passes unconditionally (graceful degradation).  When tools ARE
 * available, any error-level finding rejects the candidate.
 *
 * Design rationale (from v0.3.0 gap analysis):
 *   CSVP's regex-based analysis cannot detect use-after-free, buffer
 *   overflow, type confusion, integer overflow, dangling pointers, or
 *   thread-safety violations.  clang-tidy's 700+ checks and cppcheck's
 *   static analysis cover these classes — essential for eventual Astrée
 *   readiness.
 *
 * Integration:
 *   - EO pipeline: run after Gate 1 (CSVP), before Gate 2 (physics)
 *   - SecurityPipeline: new STATIC_ANALYSIS stage after CSVP_VERIFY
 *   - SIEOutcome: STATIC_ANALYSIS_REJECTED
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nikola::security {

// ============================================================================
// Constants
// ============================================================================

/// Maximum source size the static analyzer will accept (1 MB).
inline constexpr size_t SA_MAX_SOURCE_BYTES = 1024 * 1024;

/// Default timeout for each tool invocation (30 seconds).
inline constexpr uint32_t SA_DEFAULT_TIMEOUT_MS = 30'000;

// ============================================================================
// Finding severity
// ============================================================================

enum class FindingSeverity : uint8_t {
    NOTE     = 0,    ///< Informational
    WARNING  = 1,    ///< Potential issue
    ERROR    = 2,    ///< Definite bug — rejects candidate
};

inline const char* finding_severity_str(FindingSeverity s) {
    switch (s) {
        case FindingSeverity::NOTE:    return "note";
        case FindingSeverity::WARNING: return "warning";
        case FindingSeverity::ERROR:   return "error";
    }
    return "unknown";
}

// ============================================================================
// AnalysisFinding — single diagnostic from a tool
// ============================================================================

struct AnalysisFinding {
    std::string      tool;        ///< "clang-tidy" or "cppcheck"
    std::string      check_name;  ///< e.g. "bugprone-use-after-move"
    std::string      message;     ///< Human-readable description
    std::string      file;        ///< Source file path
    size_t           line{0};     ///< Line number
    size_t           column{0};   ///< Column number
    FindingSeverity  severity{FindingSeverity::NOTE};
};

// ============================================================================
// StaticAnalysisResult
// ============================================================================

struct StaticAnalysisResult {
    bool                          passed{true};
    bool                          clang_tidy_available{false};
    bool                          cppcheck_available{false};
    bool                          clang_tidy_ran{false};
    bool                          cppcheck_ran{false};
    std::vector<AnalysisFinding>  findings;
    size_t                        error_count{0};
    size_t                        warning_count{0};
    std::string                   error_summary;     ///< First error message

    explicit operator bool() const noexcept { return passed; }

    /// Count findings by severity.
    [[nodiscard]] size_t count_severity(FindingSeverity sev) const noexcept {
        size_t n = 0;
        for (const auto& f : findings)
            if (f.severity == sev) ++n;
        return n;
    }
};

// ============================================================================
// StaticAnalyzerConfig
// ============================================================================

struct StaticAnalyzerConfig {
    /// Path to clang-tidy binary (empty = search PATH).
    std::string clang_tidy_path;

    /// Path to cppcheck binary (empty = search PATH).
    std::string cppcheck_path;

    /// Additional include paths for clang-tidy (e.g., Nikola headers).
    std::vector<std::string> include_paths;

    /// C++ standard to use (e.g., "c++17", "c++20").
    std::string cpp_standard = "c++17";

    /// clang-tidy checks to enable.  Empty = use tool defaults.
    std::string clang_tidy_checks =
        "bugprone-*,cert-*,clang-analyzer-*,concurrency-*,"
        "cppcoreguidelines-*,misc-*,performance-*,readability-*,"
        "-readability-magic-numbers,"
        "-cppcoreguidelines-avoid-magic-numbers,"
        "-readability-identifier-length";

    /// cppcheck enable flags.
    std::string cppcheck_enable = "warning,style,performance,portability";

    /// Per-tool timeout in milliseconds.
    uint32_t timeout_ms = SA_DEFAULT_TIMEOUT_MS;

    /// Maximum source size to analyze.
    size_t max_source_bytes = SA_MAX_SOURCE_BYTES;

    /// Treat warnings as errors (reject on any warning).
    bool warnings_as_errors = false;

    /// Skip analysis entirely if no tools found (true = pass, false = fail).
    bool pass_if_unavailable = true;
};

// ============================================================================
// StaticAnalyzerGate
// ============================================================================

/**
 * @class StaticAnalyzerGate
 * @brief Runs clang-tidy and cppcheck on candidate source code.
 *
 * Usage:
 *   StaticAnalyzerGate gate;
 *   auto result = gate.analyze(source_code);
 *   if (!result.passed) { reject(result.error_summary); }
 *
 * Tool detection happens once at construction.  If a tool is not found
 * in PATH and no explicit path is configured, it is skipped.
 */
class StaticAnalyzerGate {
public:
    StaticAnalyzerGate();
    explicit StaticAnalyzerGate(StaticAnalyzerConfig config);

    /**
     * Run all available static analysis tools on the given source code.
     *
     * @param source_code  C++ source to analyze.
     * @return StaticAnalysisResult with findings and pass/fail verdict.
     */
    [[nodiscard]] StaticAnalysisResult analyze(const std::string& source_code) const;

    /**
     * Quick boolean check — returns true iff all tools pass.
     */
    [[nodiscard]] bool is_safe(const std::string& source_code) const;

    /// True if clang-tidy was found on the system.
    [[nodiscard]] bool clang_tidy_available() const noexcept { return clang_tidy_found_; }

    /// True if cppcheck was found on the system.
    [[nodiscard]] bool cppcheck_available() const noexcept { return cppcheck_found_; }

    /// Total analyses run.
    [[nodiscard]] uint64_t total_scans() const noexcept { return total_scans_; }

    /// Total rejections.
    [[nodiscard]] uint64_t total_rejections() const noexcept { return total_rejections_; }

    /// Access config.
    [[nodiscard]] const StaticAnalyzerConfig& config() const noexcept { return cfg_; }

private:
    StaticAnalyzerConfig cfg_;
    bool                 clang_tidy_found_{false};
    bool                 cppcheck_found_{false};
    std::string          clang_tidy_bin_;
    std::string          cppcheck_bin_;
    mutable uint64_t     total_scans_{0};
    mutable uint64_t     total_rejections_{0};

    /// Detect tool availability by searching PATH.
    void detect_tools_();

    /// Find an executable in PATH, return full path or empty string.
    static std::string find_in_path_(const std::string& name);

    /// Run clang-tidy on a temporary source file.
    void run_clang_tidy_(const std::string& source_file,
                         StaticAnalysisResult& result) const;

    /// Run cppcheck on a temporary source file.
    void run_cppcheck_(const std::string& source_file,
                       StaticAnalysisResult& result) const;

    /// Execute a command with timeout, capture stdout+stderr.
    static std::string exec_with_timeout_(const std::string& cmd,
                                          uint32_t timeout_ms);

    /// Parse clang-tidy output into findings.
    static void parse_clang_tidy_output_(const std::string& output,
                                         StaticAnalysisResult& result);

    /// Parse cppcheck output into findings.
    static void parse_cppcheck_output_(const std::string& output,
                                       StaticAnalysisResult& result);
};

}  // namespace nikola::security
