/**
 * @file security/code_blacklist.hpp
 * @brief Gap 7.4 — CodePatternBlacklist
 *
 * Static analysis pass for self-generated code before compilation.
 * Rejects source that contains dangerous patterns; validates that
 * all #include directives reference only the allowed whitelist.
 *
 * Blacklisted patterns (from spec):
 *   system(), exec*(), fork(), popen(), asm()/`__asm__`,
 *   networking headers (sys/socket.h, netinet/), /proc/, /dev/
 *
 * Whitelisted includes (from spec):
 *   <math.h>, <cmath>, <vector>, <algorithm>, <iostream>
 *   (plus safe numeric/string headers added as extensions)
 *
 * Design notes:
 *   - Patterns compiled once at construction time
 *   - check() returns a detailed ScanResult with all violations
 *   - No external deps — stdlib <regex> only
 */
#pragma once

#include <algorithm>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace nikola::security {

// ============================================================================
// Gap 7.4 — CodePatternBlacklist
// ============================================================================

struct ScanViolation {
    std::string pattern_name;
    std::string matched_text;
    size_t      line_number{0};
};

struct ScanResult {
    bool                       safe{true};
    std::vector<ScanViolation> violations;

    explicit operator bool() const { return safe; }
};

/**
 * Scans C/C++ source code for dangerous patterns before VM execution.
 */
class CodePatternBlacklist {
public:
    CodePatternBlacklist() { compile_patterns(); }

    /**
     * Scan source code and return a detailed result.
     *
     * Steps:
     *   1. Scan each line for dangerous call patterns
     *   2. Extract all #include directives; reject any not on the whitelist
     *
     * @param source  Complete source code as a string
     * @return ScanResult with safe=true iff no violations found
     */
    ScanResult check(const std::string& source) const {
        ScanResult result;
        result.safe = true;

        // Split into lines for line-number reporting
        std::vector<std::string> lines;
        {
            std::istringstream ss(source);
            std::string line;
            while (std::getline(ss, line)) lines.push_back(line);
        }

        // Pass 1 — dangerous call / path patterns
        for (size_t ln = 0; ln < lines.size(); ++ln) {
            for (const auto& [name, pat] : dangerous_patterns_) {
                std::smatch m;
                if (std::regex_search(lines[ln], m, pat)) {
                    result.safe = false;
                    result.violations.push_back({name, m.str(), ln + 1});
                }
            }
        }

        // Pass 2 — include whitelist
        std::sregex_iterator it(source.begin(), source.end(), include_extract_);
        std::sregex_iterator end;
        for (; it != end; ++it) {
            const std::string stmt = it->str();
            // Approximate line number by counting newlines
            size_t ln = std::count(source.begin(),
                                   source.begin() + it->position(), '\n') + 1;
            if (!is_include_whitelisted(stmt)) {
                result.safe = false;
                result.violations.push_back({"disallowed_include", stmt, ln});
            }
        }

        return result;
    }

    /**
     * Quick boolean variant — returns true iff source is safe.
     */
    bool is_safe(const std::string& source) const {
        return check(source).safe;
    }

    // ── Pattern management ────────────────────────────────────────────────────

    /**
     * Add a custom dangerous pattern.
     * @param name   Human-readable label
     * @param regex_str  ECMAScript regex
     */
    void add_dangerous_pattern(const std::string& name,
                                const std::string& regex_str)
    {
        dangerous_patterns_.emplace_back(name, std::regex(regex_str));
    }

    /** Add a header name to the whitelist (e.g. "array" → allows <array>). */
    void add_allowed_include(const std::string& header_name) {
        extra_allowed_.push_back(header_name);
    }

    size_t dangerous_pattern_count() const { return dangerous_patterns_.size(); }
    size_t allowed_include_count()   const { return allowed_headers_.size() + extra_allowed_.size(); }

private:
    std::vector<std::pair<std::string, std::regex>> dangerous_patterns_;
    std::vector<std::string> allowed_headers_;     // canonical header names
    std::vector<std::string> extra_allowed_;        // user-added
    std::regex               include_extract_;      // extracts all #include lines

    void compile_patterns() {
        // ── Dangerous call patterns (spec Gap 7.4) ─────────────────────────
        dangerous_patterns_ = {
            {"system_call",     std::regex(R"(\bsystem\s*\()")},
            {"exec_family",     std::regex(R"(\bexec\w*\s*\()")},   // execve, execl, etc.
            {"fork_call",       std::regex(R"(\bfork\s*\()")},
            {"popen_call",      std::regex(R"(\bpopen\s*\()")},
            {"inline_asm_gnu",  std::regex(R"(\b__asm__\s*\()")},
            {"inline_asm",      std::regex(R"(\basm\s*\()")},
            {"socket_header",   std::regex(R"(#\s*include\s*[<"]sys/socket\.h[">])")},
            {"netinet_header",  std::regex(R"(#\s*include\s*[<"]netinet/)")},
            {"proc_path",       std::regex(R"(/proc/)")},
            {"dev_path",        std::regex(R"(/dev/(?!null))")},     // /dev/null allowed
            {"ptrace_call",     std::regex(R"(\bptrace\s*\()")},
            {"mmap_call",       std::regex(R"(\bmmap\s*\()")},
            {"dlopen_call",     std::regex(R"(\bdlopen\s*\()")},
        };

        // ── Allowed #include whitelist (spec Gap 7.4 + safe extensions) ────
        allowed_headers_ = {
            // spec-mandated
            "math.h", "cmath", "vector", "algorithm", "iostream",
            // safe numeric / string / containers
            "cstdint", "cstddef", "cstring", "cassert", "climits",
            "cfloat", "cstdlib",
            "array", "string", "numeric", "functional", "utility",
            "type_traits", "limits", "memory", "stdexcept",
            "tuple", "optional", "variant", "span",
            // I/O (read-only safe within sandbox)
            "sstream", "iomanip",
            // complex / linear algebra
            "complex", "valarray",
            // threading (inside sandbox is OK — contained)
            "atomic",
        };

        // Regex to extract all #include lines
        include_extract_ = std::regex(R"(#\s*include\s*[<"][^>"]+[>"])");
    }

    bool is_include_whitelisted(const std::string& include_stmt) const {
        // Extract header name from #include <name> or #include "name"
        std::regex name_rx(R"([<"]([^>"]+)[>"])");
        std::smatch m;
        if (!std::regex_search(include_stmt, m, name_rx)) return false;
        const std::string header = m[1].str();

        // Strip path prefix (e.g. bits/stdc++.h → stdc++.h)
        std::string base = header;
        const auto slash = base.rfind('/');
        if (slash != std::string::npos) base = base.substr(slash + 1);

        for (const auto& allowed : allowed_headers_)
            if (base == allowed || header == allowed) return true;
        for (const auto& allowed : extra_allowed_)
            if (base == allowed || header == allowed) return true;
        return false;
    }

    // Allow std::istringstream in the implementation
    struct std_access {
        static std::vector<std::string> split_lines(const std::string& s) {
            std::vector<std::string> out;
            std::string::size_type start = 0, pos;
            while ((pos = s.find('\n', start)) != std::string::npos) {
                out.push_back(s.substr(start, pos - start));
                start = pos + 1;
            }
            out.push_back(s.substr(start));
            return out;
        }
    };
};

} // namespace nikola::security
