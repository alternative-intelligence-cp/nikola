/**
 * @file security/csvp.hpp
 * @brief v0.1.19 — Code Safety Verification Protocol (CSVP)
 *
 * Multi-stage static analysis pipeline for self-generated code.
 * Extends Gate 1 (CodePatternBlacklist) with deeper analysis:
 *
 *   Stage 1: Pattern blacklist (delegates to CodePatternBlacklist)
 *   Stage 2: Structural analysis — AST-level checks for dangerous constructs
 *   Stage 3: Resource analysis  — detects unbounded loops, large allocations
 *   Stage 4: Physics invariant  — generated code must conserve energy invariants
 *
 * Each stage returns a typed verdict; all must pass for code to be approved.
 *
 * Integration with 4-gate pipeline:
 *   Gate 0 (ShadowSpine) → binary signature check
 *   Gate 1 (CSVP)        → THIS MODULE replaces raw blacklist
 *   Gate 2 (PhysicsOracle) → energy conservation at runtime
 *   Gate 3 (ModuleSwapper) → hot-swap with rollback
 *
 * Usage:
 *   CodeSafetyVerifier csvp;
 *   auto result = csvp.verify(source_code);
 *   if (!result.approved) { reject(result.stage, result.violations); }
 */
#pragma once

#include <cstdint>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <nikola/security/code_blacklist.hpp>

namespace nikola::security {

// ============================================================================
// Constants
// ============================================================================

inline constexpr size_t CSVP_MAX_SOURCE_BYTES       = 1024 * 1024;  // 1MB
inline constexpr size_t CSVP_MAX_FUNCTION_LINES     = 500;
inline constexpr size_t CSVP_MAX_NESTING_DEPTH      = 10;
inline constexpr size_t CSVP_MAX_ALLOCATION_BYTES   = 64 * 1024 * 1024;  // 64MB
inline constexpr size_t CSVP_MAX_LOOP_ITERATIONS    = 10'000'000;

// ============================================================================
// Verification stages
// ============================================================================

enum class CSVPStage : uint8_t {
    PATTERN_BLACKLIST = 1,   ///< Stage 1: regex pattern matching
    STRUCTURAL,              ///< Stage 2: structural / AST-level analysis
    RESOURCE,                ///< Stage 3: resource usage bounds checking
    PHYSICS_INVARIANT,       ///< Stage 4: physics invariant validation
};

inline const char* csvp_stage_str(CSVPStage s) {
    switch (s) {
        case CSVPStage::PATTERN_BLACKLIST: return "PATTERN_BLACKLIST";
        case CSVPStage::STRUCTURAL:        return "STRUCTURAL";
        case CSVPStage::RESOURCE:          return "RESOURCE";
        case CSVPStage::PHYSICS_INVARIANT: return "PHYSICS_INVARIANT";
    }
    return "UNKNOWN";
}

// ============================================================================
// Violations
// ============================================================================

struct CSVPViolation {
    CSVPStage   stage;
    std::string rule_name;
    std::string detail;
    size_t      line_number{0};
};

// ============================================================================
// Verification result
// ============================================================================

struct CSVPResult {
    bool                        approved{true};
    CSVPStage                   failed_stage{CSVPStage::PATTERN_BLACKLIST};
    std::vector<CSVPViolation>  violations;
    uint32_t                    stages_passed{0};

    explicit operator bool() const { return approved; }
};

// ============================================================================
// Physics invariant checker config
// ============================================================================

struct PhysicsInvariantConfig {
    bool require_energy_conservation = true;  ///< Must not create/destroy energy
    bool require_bounded_evolution   = true;  ///< State evolution must converge
    bool require_no_negative_mass    = true;  ///< No negative mass/energy values

    /// Patterns indicating energy conservation violation
    std::vector<std::string> violation_patterns = {
        "energy\\s*=\\s*-",                  // negative energy assignment
        "energy\\s*\\+=\\s*1e[0-9]{3,}",     // unreasonably large energy addition
        "mass\\s*=\\s*-",                    // negative mass assignment
    };
};

// ============================================================================
// CodeSafetyVerifier — main CSVP class
// ============================================================================

class CodeSafetyVerifier {
public:
    struct Config {
        size_t                max_source_bytes      = CSVP_MAX_SOURCE_BYTES;
        size_t                max_function_lines    = CSVP_MAX_FUNCTION_LINES;
        size_t                max_nesting_depth     = CSVP_MAX_NESTING_DEPTH;
        size_t                max_allocation_bytes  = CSVP_MAX_ALLOCATION_BYTES;
        size_t                max_loop_iterations   = CSVP_MAX_LOOP_ITERATIONS;
        bool                  run_physics_check     = true;
        PhysicsInvariantConfig physics;
    };

    CodeSafetyVerifier() : cfg_{} {}
    explicit CodeSafetyVerifier(Config cfg) : cfg_(std::move(cfg)) {}

    /**
     * Run all CSVP stages on source code.
     * Stages execute in order; first failure short-circuits.
     */
    CSVPResult verify(const std::string& source) const {
        CSVPResult result;

        // Pre-check: source size
        if (source.size() > cfg_.max_source_bytes) {
            result.approved     = false;
            result.failed_stage = CSVPStage::STRUCTURAL;
            result.violations.push_back({
                CSVPStage::STRUCTURAL, "source_too_large",
                "Source exceeds " + std::to_string(cfg_.max_source_bytes) + " bytes",
                0
            });
            return result;
        }

        // Stage 1: Pattern blacklist
        if (!run_stage1(source, result)) return result;
        result.stages_passed = 1;

        // Stage 2: Structural analysis
        if (!run_stage2(source, result)) return result;
        result.stages_passed = 2;

        // Stage 3: Resource analysis
        if (!run_stage3(source, result)) return result;
        result.stages_passed = 3;

        // Stage 4: Physics invariant
        if (cfg_.run_physics_check) {
            if (!run_stage4(source, result)) return result;
        }
        result.stages_passed = 4;

        return result;
    }

    /**
     * Quick boolean check — returns true iff all stages pass.
     */
    bool is_safe(const std::string& source) const {
        return verify(source).approved;
    }

    /**
     * Access the underlying blacklist for extension.
     */
    CodePatternBlacklist& blacklist() { return blacklist_; }
    const CodePatternBlacklist& blacklist() const { return blacklist_; }

    uint64_t total_scans()      const { return total_scans_; }
    uint64_t total_rejections() const { return total_rejections_; }

private:
    Config               cfg_;
    CodePatternBlacklist blacklist_;
    mutable uint64_t     total_scans_{0};
    mutable uint64_t     total_rejections_{0};

    // ── Stage 1: Pattern blacklist ──────────────────────────────────────────

    bool run_stage1(const std::string& source, CSVPResult& result) const {
        ++total_scans_;
        auto scan = blacklist_.check(source);
        if (!scan.safe) {
            result.approved     = false;
            result.failed_stage = CSVPStage::PATTERN_BLACKLIST;
            for (const auto& v : scan.violations) {
                result.violations.push_back({
                    CSVPStage::PATTERN_BLACKLIST,
                    v.pattern_name,
                    "Matched: " + v.matched_text,
                    v.line_number
                });
            }
            ++total_rejections_;
            return false;
        }
        return true;
    }

    // ── Stage 2: Structural analysis ────────────────────────────────────────

    bool run_stage2(const std::string& source, CSVPResult& result) const {
        auto lines = split_lines(source);
        bool ok = true;

        // Check function length
        size_t func_lines = 0;
        int brace_depth = 0;
        bool in_function = false;
        size_t func_start = 0;

        for (size_t i = 0; i < lines.size(); ++i) {
            const auto& line = lines[i];

            // Detect function start (simplified)
            if (!in_function && brace_depth == 0 &&
                std::regex_search(line, std::regex(R"(\w+\s+\w+\s*\([^)]*\)\s*\{)")))
            {
                in_function = true;
                func_start  = i + 1;
                func_lines  = 0;
            }

            for (char c : line) {
                if (c == '{') ++brace_depth;
                if (c == '}') --brace_depth;
            }

            if (in_function) ++func_lines;

            if (in_function && brace_depth == 0) {
                if (func_lines > cfg_.max_function_lines) {
                    result.violations.push_back({
                        CSVPStage::STRUCTURAL, "function_too_long",
                        "Function at line " + std::to_string(func_start) +
                        " is " + std::to_string(func_lines) + " lines (max " +
                        std::to_string(cfg_.max_function_lines) + ")",
                        func_start
                    });
                    ok = false;
                }
                in_function = false;
                func_lines  = 0;
            }
        }

        // Check nesting depth
        int max_depth = 0, depth = 0;
        for (size_t i = 0; i < lines.size(); ++i) {
            for (char c : lines[i]) {
                if (c == '{') { ++depth; if (depth > max_depth) max_depth = depth; }
                if (c == '}') --depth;
            }
            if (static_cast<size_t>(max_depth) > cfg_.max_nesting_depth) {
                result.violations.push_back({
                    CSVPStage::STRUCTURAL, "nesting_too_deep",
                    "Nesting depth " + std::to_string(max_depth) +
                    " exceeds max " + std::to_string(cfg_.max_nesting_depth),
                    i + 1
                });
                ok = false;
                break;
            }
        }

        // Check for goto (structured control flow required)
        std::regex goto_rx(R"(\bgoto\s+\w+)");
        for (size_t i = 0; i < lines.size(); ++i) {
            if (std::regex_search(lines[i], goto_rx)) {
                result.violations.push_back({
                    CSVPStage::STRUCTURAL, "goto_forbidden",
                    "goto statement detected", i + 1
                });
                ok = false;
            }
        }

        if (!ok) {
            result.approved     = false;
            result.failed_stage = CSVPStage::STRUCTURAL;
            ++total_rejections_;
        }
        return ok;
    }

    // ── Stage 3: Resource analysis ──────────────────────────────────────────

    bool run_stage3(const std::string& source, CSVPResult& result) const {
        auto lines = split_lines(source);
        bool ok = true;

        // Detect large static allocations (arrays) e.g. int arr[10000000]
        std::regex large_array_rx(R"(\w+\s+\w+\s*\[\s*(\d+)\s*\])");
        for (size_t i = 0; i < lines.size(); ++i) {
            std::smatch m;
            std::string line = lines[i];
            while (std::regex_search(line, m, large_array_rx)) {
                uint64_t sz = 0;
                try { sz = std::stoull(m[1].str()); } catch (...) {}
                if (sz * 8 > cfg_.max_allocation_bytes) {  // assume 8 bytes per element
                    result.violations.push_back({
                        CSVPStage::RESOURCE, "large_allocation",
                        "Array of " + m[1].str() + " elements (~" +
                        std::to_string(sz * 8 / (1024*1024)) + "MB)",
                        i + 1
                    });
                    ok = false;
                }
                line = m.suffix().str();
            }
        }

        // Detect unbounded while(true) / for(;;) without break
        std::regex infinite_loop_rx(R"(while\s*\(\s*(true|1)\s*\)|for\s*\(\s*;\s*;\s*\))");
        for (size_t i = 0; i < lines.size(); ++i) {
            if (std::regex_search(lines[i], infinite_loop_rx)) {
                // Check if there's a break/return within next 20 lines
                bool has_exit = false;
                size_t end = std::min(i + 20, lines.size());
                for (size_t j = i + 1; j < end; ++j) {
                    if (lines[j].find("break") != std::string::npos ||
                        lines[j].find("return") != std::string::npos) {
                        has_exit = true;
                        break;
                    }
                }
                if (!has_exit) {
                    result.violations.push_back({
                        CSVPStage::RESOURCE, "unbounded_loop",
                        "Potential infinite loop without visible break/return",
                        i + 1
                    });
                    ok = false;
                }
            }
        }

        // Detect large malloc/new allocations
        std::regex malloc_rx(R"((?:malloc|calloc|new\s+\w+\[)\s*\(?(\d+))");
        for (size_t i = 0; i < lines.size(); ++i) {
            std::smatch m;
            if (std::regex_search(lines[i], m, malloc_rx)) {
                uint64_t sz = 0;
                try { sz = std::stoull(m[1].str()); } catch (...) {}
                if (sz > cfg_.max_allocation_bytes) {
                    result.violations.push_back({
                        CSVPStage::RESOURCE, "large_heap_allocation",
                        "Heap allocation of " + std::to_string(sz) + " bytes",
                        i + 1
                    });
                    ok = false;
                }
            }
        }

        if (!ok) {
            result.approved     = false;
            result.failed_stage = CSVPStage::RESOURCE;
            ++total_rejections_;
        }
        return ok;
    }

    // ── Stage 4: Physics invariant validation ───────────────────────────────

    bool run_stage4(const std::string& source, CSVPResult& result) const {
        auto lines = split_lines(source);
        bool ok = true;

        for (const auto& pat_str : cfg_.physics.violation_patterns) {
            std::regex pat(pat_str);
            for (size_t i = 0; i < lines.size(); ++i) {
                if (std::regex_search(lines[i], pat)) {
                    result.violations.push_back({
                        CSVPStage::PHYSICS_INVARIANT, "physics_violation",
                        "Pattern '" + pat_str + "' matched",
                        i + 1
                    });
                    ok = false;
                }
            }
        }

        // Check for energy conservation: if code modifies energy,
        // there should be a corresponding conservation term
        if (cfg_.physics.require_energy_conservation) {
            bool modifies_energy = false;
            bool has_conservation = false;
            for (const auto& line : lines) {
                if (std::regex_search(line,
                    std::regex(R"(energy\s*[\+\-\*]?=)"))) {
                    modifies_energy = true;
                }
                if (line.find("total_energy") != std::string::npos ||
                    line.find("energy_conservation") != std::string::npos ||
                    line.find("conserve") != std::string::npos) {
                    has_conservation = true;
                }
            }
            if (modifies_energy && !has_conservation) {
                result.violations.push_back({
                    CSVPStage::PHYSICS_INVARIANT, "energy_conservation_missing",
                    "Code modifies energy without conservation tracking",
                    0
                });
                ok = false;
            }
        }

        if (!ok) {
            result.approved     = false;
            result.failed_stage = CSVPStage::PHYSICS_INVARIANT;
            ++total_rejections_;
        }
        return ok;
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    static std::vector<std::string> split_lines(const std::string& s) {
        std::vector<std::string> out;
        std::istringstream ss(s);
        std::string line;
        while (std::getline(ss, line)) out.push_back(line);
        if (out.empty()) out.push_back("");
        return out;
    }
};

} // namespace nikola::security
