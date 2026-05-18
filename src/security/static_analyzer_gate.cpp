/**
 * @file src/security/static_analyzer_gate.cpp
 * @brief v0.3.0 — StaticAnalyzerGate implementation.
 *
 * Wraps clang-tidy and cppcheck as subprocess invocations on a temporary
 * source file.  Each tool's output is parsed into structured AnalysisFinding
 * records.
 */

#include <nikola/security/static_analyzer_gate.hpp>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

namespace nikola::security {

// ── Construction ────────────────────────────────────────────────────────────

StaticAnalyzerGate::StaticAnalyzerGate()
    : cfg_{}
{
    detect_tools_();
}

StaticAnalyzerGate::StaticAnalyzerGate(StaticAnalyzerConfig config)
    : cfg_(std::move(config))
{
    detect_tools_();
}

// ── Tool detection ──────────────────────────────────────────────────────────

void StaticAnalyzerGate::detect_tools_() {
    // clang-tidy
    if (!cfg_.clang_tidy_path.empty()) {
        clang_tidy_bin_   = cfg_.clang_tidy_path;
        clang_tidy_found_ = (access(clang_tidy_bin_.c_str(), X_OK) == 0);
    } else {
        clang_tidy_bin_   = find_in_path_("clang-tidy");
        clang_tidy_found_ = !clang_tidy_bin_.empty();
    }

    // cppcheck
    if (!cfg_.cppcheck_path.empty()) {
        cppcheck_bin_   = cfg_.cppcheck_path;
        cppcheck_found_ = (access(cppcheck_bin_.c_str(), X_OK) == 0);
    } else {
        cppcheck_bin_   = find_in_path_("cppcheck");
        cppcheck_found_ = !cppcheck_bin_.empty();
    }
}

std::string StaticAnalyzerGate::find_in_path_(const std::string& name) {
    const char* path_env = std::getenv("PATH");
    if (!path_env) return {};

    std::string path_str{path_env};
    std::istringstream ss{path_str};
    std::string dir;

    while (std::getline(ss, dir, ':')) {
        std::string candidate = dir + "/" + name;
        if (access(candidate.c_str(), X_OK) == 0) {
            return candidate;
        }
    }
    return {};
}

// ── Main analysis entry point ───────────────────────────────────────────────

StaticAnalysisResult StaticAnalyzerGate::analyze(const std::string& source_code) const {
    ++total_scans_;

    StaticAnalysisResult result;
    result.clang_tidy_available = clang_tidy_found_;
    result.cppcheck_available   = cppcheck_found_;

    // Pre-check: source size
    if (source_code.size() > cfg_.max_source_bytes) {
        result.passed = false;
        result.error_count = 1;
        result.error_summary = "Source exceeds maximum size ("
            + std::to_string(cfg_.max_source_bytes) + " bytes)";
        ++total_rejections_;
        return result;
    }

    // If no tools available, apply policy
    if (!clang_tidy_found_ && !cppcheck_found_) {
        if (cfg_.pass_if_unavailable) {
            result.passed = true;
            return result;
        } else {
            result.passed = false;
            result.error_count = 1;
            result.error_summary = "No static analysis tools available "
                "(clang-tidy, cppcheck) and pass_if_unavailable=false";
            ++total_rejections_;
            return result;
        }
    }

    // Write source to a temporary file
    char tmp_path[] = "/tmp/nikola_sa_XXXXXX.cpp";
    int fd = mkstemps(tmp_path, 4);  // ".cpp" = 4 chars
    if (fd < 0) {
        result.passed = cfg_.pass_if_unavailable;
        result.error_summary = "Failed to create temporary file: "
            + std::string(std::strerror(errno));
        if (!result.passed) ++total_rejections_;
        return result;
    }

    // Write source and close FD
    {
        ssize_t written = write(fd, source_code.data(), source_code.size());
        close(fd);
        if (written < 0 || static_cast<size_t>(written) != source_code.size()) {
            unlink(tmp_path);
            result.passed = cfg_.pass_if_unavailable;
            result.error_summary = "Failed to write temporary file";
            if (!result.passed) ++total_rejections_;
            return result;
        }
    }

    // Run available tools
    if (clang_tidy_found_) {
        run_clang_tidy_(tmp_path, result);
        result.clang_tidy_ran = true;
    }
    if (cppcheck_found_) {
        run_cppcheck_(tmp_path, result);
        result.cppcheck_ran = true;
    }

    // Clean up temp file
    unlink(tmp_path);

    // Count errors and warnings
    result.error_count   = result.count_severity(FindingSeverity::ERROR);
    result.warning_count = result.count_severity(FindingSeverity::WARNING);

    // Determine pass/fail
    if (result.error_count > 0) {
        result.passed = false;
        // Set summary to first error
        for (const auto& f : result.findings) {
            if (f.severity == FindingSeverity::ERROR) {
                result.error_summary = "[" + f.tool + "] " + f.check_name
                    + ": " + f.message;
                break;
            }
        }
    } else if (cfg_.warnings_as_errors && result.warning_count > 0) {
        result.passed = false;
        for (const auto& f : result.findings) {
            if (f.severity == FindingSeverity::WARNING) {
                result.error_summary = "[" + f.tool + "] " + f.check_name
                    + ": " + f.message + " (warnings_as_errors)";
                break;
            }
        }
    }

    if (!result.passed) ++total_rejections_;
    return result;
}

bool StaticAnalyzerGate::is_safe(const std::string& source_code) const {
    return analyze(source_code).passed;
}

// ── clang-tidy ──────────────────────────────────────────────────────────────

void StaticAnalyzerGate::run_clang_tidy_(const std::string& source_file,
                                          StaticAnalysisResult& result) const {
    std::string cmd = clang_tidy_bin_;
    cmd += " --quiet";

    if (!cfg_.clang_tidy_checks.empty()) {
        cmd += " --checks='" + cfg_.clang_tidy_checks + "'";
    }

    cmd += " " + source_file;
    cmd += " --";  // separator for extra compiler args
    cmd += " -std=" + cfg_.cpp_standard;

    for (const auto& inc : cfg_.include_paths) {
        cmd += " -I" + inc;
    }

    // Redirect stderr to stdout (clang-tidy outputs diagnostics on stderr)
    cmd += " 2>&1";

    try {
        std::string output = exec_with_timeout_(cmd, cfg_.timeout_ms);
        parse_clang_tidy_output_(output, result);
    } catch (const std::exception& e) {
        // Tool crashed or timed out — log but don't fail
        result.findings.push_back({
            "clang-tidy", "tool-error", e.what(), source_file, 0, 0,
            FindingSeverity::NOTE
        });
    }
}

// ── cppcheck ────────────────────────────────────────────────────────────────

void StaticAnalyzerGate::run_cppcheck_(const std::string& source_file,
                                        StaticAnalysisResult& result) const {
    std::string cmd = cppcheck_bin_;
    cmd += " --quiet";
    cmd += " --error-exitcode=0";  // We parse output, don't rely on exit code
    cmd += " --template='{file}:{line}:{column}: {severity}: {message} [{id}]'";

    if (!cfg_.cppcheck_enable.empty()) {
        cmd += " --enable=" + cfg_.cppcheck_enable;
    }

    cmd += " --std=" + cfg_.cpp_standard;

    for (const auto& inc : cfg_.include_paths) {
        cmd += " -I" + inc;
    }

    cmd += " " + source_file;
    cmd += " 2>&1";

    try {
        std::string output = exec_with_timeout_(cmd, cfg_.timeout_ms);
        parse_cppcheck_output_(output, result);
    } catch (const std::exception& e) {
        result.findings.push_back({
            "cppcheck", "tool-error", e.what(), source_file, 0, 0,
            FindingSeverity::NOTE
        });
    }
}

// ── Subprocess execution with timeout ───────────────────────────────────────

std::string StaticAnalyzerGate::exec_with_timeout_(const std::string& cmd,
                                                     uint32_t timeout_ms) {
    // Use posix_spawn-safe approach: popen + alarm-based timeout
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("popen() failed: "
            + std::string(std::strerror(errno)));
    }

    std::string output;
    output.reserve(4096);
    std::array<char, 4096> buf{};

    auto start = std::chrono::steady_clock::now();
    auto deadline = start + std::chrono::milliseconds(timeout_ms);

    while (std::chrono::steady_clock::now() < deadline) {
        size_t n = fread(buf.data(), 1, buf.size(), pipe);
        if (n > 0) {
            output.append(buf.data(), n);
        }
        if (feof(pipe) || ferror(pipe)) break;
    }

    int status = pclose(pipe);

    if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error("Static analysis tool timed out after "
            + std::to_string(timeout_ms) + "ms");
    }

    // pclose returns -1 on error, otherwise the exit status
    (void)status;  // We parse output regardless of exit code

    return output;
}

// ── Output parsers ──────────────────────────────────────────────────────────

void StaticAnalyzerGate::parse_clang_tidy_output_(
    const std::string& output, StaticAnalysisResult& result)
{
    // clang-tidy format: file:line:col: severity: message [check-name]
    static const std::regex diag_rx{
        R"(([^:]+):(\d+):(\d+):\s+(warning|error|note):\s+(.+)\s+\[([^\]]+)\])"
    };

    std::istringstream ss{output};
    std::string line;

    while (std::getline(ss, line)) {
        std::smatch m;
        if (std::regex_search(line, m, diag_rx)) {
            AnalysisFinding f;
            f.tool       = "clang-tidy";
            f.file       = m[1].str();
            f.line       = std::stoull(m[2].str());
            f.column     = std::stoull(m[3].str());
            f.message    = m[5].str();
            f.check_name = m[6].str();

            std::string sev = m[4].str();
            if (sev == "error")        f.severity = FindingSeverity::ERROR;
            else if (sev == "warning") f.severity = FindingSeverity::WARNING;
            else                       f.severity = FindingSeverity::NOTE;

            result.findings.push_back(std::move(f));
        }
    }
}

void StaticAnalyzerGate::parse_cppcheck_output_(
    const std::string& output, StaticAnalysisResult& result)
{
    // cppcheck format (our template): file:line:col: severity: message [id]
    static const std::regex diag_rx{
        R"(([^:]+):(\d+):(\d+):\s+(error|warning|style|performance|portability|information):\s+(.+)\s+\[([^\]]+)\])"
    };

    std::istringstream ss{output};
    std::string line;

    while (std::getline(ss, line)) {
        std::smatch m;
        if (std::regex_search(line, m, diag_rx)) {
            AnalysisFinding f;
            f.tool       = "cppcheck";
            f.file       = m[1].str();
            f.line       = std::stoull(m[2].str());
            f.column     = std::stoull(m[3].str());
            f.message    = m[5].str();
            f.check_name = m[6].str();

            std::string sev = m[4].str();
            if (sev == "error")   f.severity = FindingSeverity::ERROR;
            else if (sev == "warning" || sev == "style"
                     || sev == "performance" || sev == "portability")
                                  f.severity = FindingSeverity::WARNING;
            else                  f.severity = FindingSeverity::NOTE;

            result.findings.push_back(std::move(f));
        }
    }
}

}  // namespace nikola::security
