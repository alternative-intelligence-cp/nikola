/**
 * @file aria/compile_validator.hpp
 * @brief Invoke ariac compiler as subprocess to validate Aria source code.
 *
 * Used by the SIE pipeline to gate generated code:
 *   specialist output → extract_code → compile_validate() → accept/reject
 *
 * The validator writes source to a tempfile, runs ariac, captures exit code
 * and error messages, then cleans up.  Thread-safe (no shared state).
 */
#pragma once

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

namespace nikola::aria {

// ============================================================================
// CompileResult
// ============================================================================

struct CompileResult {
    bool                     success{false};
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    std::string              raw_output;   ///< combined stdout+stderr
    double                   elapsed_ms{0.0};

    explicit operator bool() const noexcept { return success; }
};

// ============================================================================
// AriaCompileValidator
// ============================================================================

class AriaCompileValidator {
public:
    /**
     * @brief Construct with path to ariac binary.
     *
     * @param ariac_path  Absolute path to the ariac compiler binary.
     *                    Defaults to $ARIAC_BIN or ~/Workspace/REPOS/aria/build/ariac.
     * @param timeout_ms  Maximum compilation time before kill (default: 30s).
     */
    explicit AriaCompileValidator(
            std::string ariac_path = default_ariac_path(),
            int timeout_ms = 30000)
        : ariac_path_(std::move(ariac_path))
        , timeout_ms_(timeout_ms)
    {}

    /**
     * @brief Validate Aria source code by compiling with ariac.
     *
     * 1. Write source to /tmp/nikola_XXXXXX.aria
     * 2. Run: ariac <tmp>.aria -o <tmp>.out
     * 3. Parse stdout+stderr for error/warning lines
     * 4. Clean up temp files
     *
     * @param source_code  Aria source code to validate
     * @return             CompileResult with success flag, errors, warnings
     */
    CompileResult validate(const std::string& source_code) const {
        CompileResult result;

        // Check compiler exists
        if (!std::filesystem::exists(ariac_path_)) {
            result.errors.push_back("ariac not found at: " + ariac_path_);
            return result;
        }

        // Create temp file
        auto tmp_dir = std::filesystem::temp_directory_path();
        auto src_path = tmp_dir / ("nikola_validate_" +
            std::to_string(std::hash<std::string>{}(source_code) ^
                           std::hash<int>{}(static_cast<int>(
                               std::chrono::steady_clock::now().time_since_epoch().count())))
            + ".aria");
        auto out_path = std::filesystem::path(src_path.string() + ".out");

        // Write source
        {
            std::ofstream ofs(src_path);
            if (!ofs) {
                result.errors.push_back("Failed to create temp file: " + src_path.string());
                return result;
            }
            ofs << source_code;
        }

        // Build command: ariac <src> -o <out> 2>&1
        std::string cmd = ariac_path_ + " " + src_path.string()
                        + " -o " + out_path.string() + " 2>&1";

        // Execute with timeout via shell
        auto t0 = std::chrono::steady_clock::now();
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            result.errors.push_back("popen failed for ariac");
            cleanup(src_path, out_path);
            return result;
        }

        // Read output
        std::string output;
        char buf[512];
        while (fgets(buf, sizeof(buf), pipe)) {
            output += buf;
            // Rough timeout check
            auto elapsed = std::chrono::steady_clock::now() - t0;
            if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
                    > timeout_ms_) {
                pclose(pipe);
                result.errors.push_back("compile timeout (" +
                    std::to_string(timeout_ms_) + "ms)");
                result.raw_output = output;
                cleanup(src_path, out_path);
                return result;
            }
        }

        int status = pclose(pipe);
        auto t1 = std::chrono::steady_clock::now();
        result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.raw_output = output;

        // Parse exit code
#ifdef _WIN32
        result.success = (status == 0);
#else
        result.success = WIFEXITED(status) && (WEXITSTATUS(status) == 0);
#endif

        // Parse error/warning lines
        parse_output(output, result.errors, result.warnings);

        // Cleanup
        cleanup(src_path, out_path);
        return result;
    }

    /// Check if the ariac binary exists and is executable.
    [[nodiscard]] bool compiler_available() const {
        return std::filesystem::exists(ariac_path_);
    }

    /// Path to the ariac binary.
    [[nodiscard]] const std::string& ariac_path() const noexcept {
        return ariac_path_;
    }

    /// Default ariac path: $ARIAC_BIN or ~/Workspace/REPOS/aria/build/ariac
    static std::string default_ariac_path() {
        const char* env = std::getenv("ARIAC_BIN");
        if (env && *env) return env;
        auto home = std::filesystem::path(std::getenv("HOME") ? std::getenv("HOME") : "/root");
        return (home / "Workspace" / "REPOS" / "aria" / "build" / "ariac").string();
    }

private:
    std::string ariac_path_;
    int         timeout_ms_;

    static void cleanup(const std::filesystem::path& src,
                        const std::filesystem::path& out) {
        std::error_code ec;
        std::filesystem::remove(src, ec);
        std::filesystem::remove(out, ec);
    }

    static void parse_output(const std::string& output,
                             std::vector<std::string>& errors,
                             std::vector<std::string>& warnings) {
        std::istringstream stream(output);
        std::string line;
        while (std::getline(stream, line)) {
            // Case-insensitive check for "error" or "warning"
            std::string lower = line;
            for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (lower.find("error") != std::string::npos) {
                errors.push_back(line);
            } else if (lower.find("warning") != std::string::npos) {
                warnings.push_back(line);
            }
        }
    }
};

// ============================================================================
// extract_code_block — Parse Aria code from model response
// ============================================================================

/**
 * @brief Extract Aria source code from a model response string.
 *
 * Handles:
 *   1. Fenced ```aria\n...\n``` blocks
 *   2. Generic fenced ```\n...\n``` blocks
 *   3. Raw code (detected by Aria keywords)
 */
inline std::string extract_code_block(const std::string& response) {
    // Try fenced code block: ```aria\n...\n```
    static const std::regex fenced_re(R"(```(?:aria)?\s*\n([\s\S]*?)```)");
    std::smatch match;
    if (std::regex_search(response, match, fenced_re)) {
        return match[1].str();
    }

    // If response contains Aria keywords, treat as raw code
    if (response.find("func:") != std::string::npos ||
        response.find("use ") != std::string::npos ||
        response.find("extern ") != std::string::npos ||
        response.find("int32:") != std::string::npos ||
        response.find("string:") != std::string::npos) {
        return response;
    }

    return response;
}

} // namespace nikola::aria
