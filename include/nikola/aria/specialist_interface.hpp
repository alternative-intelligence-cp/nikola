/**
 * @file aria/specialist_interface.hpp
 * @brief C++ client for the Aria Specialist model (Python JSON-Lines server).
 *
 * Spawns scripts/server.py as a subprocess, communicates via JSON-Lines
 * over stdin/stdout pipes.  Protocol:
 *
 *   Ready signal: {"ready": true, "checkpoint": "..."}
 *   Request:      {"id": N, "instruction": "...", "context": "..."}
 *   Response:     {"id": N, "ok": true, "response": "..."}
 *   Error:        {"id": N, "ok": false, "error": "..."}
 *
 * Thread-safety: NOT thread-safe — use one instance per thread.
 */
#pragma once

#include <array>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace nikola::aria {

// ============================================================================
// SpecialistResult
// ============================================================================

struct SpecialistResult {
    bool        ok{false};
    std::string response;   ///< Model output (when ok=true)
    std::string error;      ///< Error message (when ok=false)

    explicit operator bool() const noexcept { return ok; }
};

// ============================================================================
// SpecialistInterface
// ============================================================================

class SpecialistInterface {
public:
    /**
     * @brief Construct (does NOT start the server yet).
     *
     * @param server_script_path  Absolute path to server.py.
     *   Default: $ARIA_SPECIALIST_SERVER or
     *            ~/Workspace/REPOS/aria-specialist/scripts/server.py
     * @param python_bin  Python interpreter to use.  Default: "python3".
     * @param startup_timeout_ms  Max time to wait for ready signal.
     */
    explicit SpecialistInterface(
            std::string server_script_path = default_server_path(),
            std::string python_bin = "python3",
            int startup_timeout_ms = 120000)
        : server_path_(std::move(server_script_path))
        , python_bin_(std::move(python_bin))
        , startup_timeout_ms_(startup_timeout_ms)
    {}

    ~SpecialistInterface() { stop(); }

    // Non-copyable, movable
    SpecialistInterface(const SpecialistInterface&) = delete;
    SpecialistInterface& operator=(const SpecialistInterface&) = delete;

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /**
     * @brief Start the specialist server subprocess.
     *
     * Forks, execs python3 server.py, then blocks until the ready signal
     * arrives on stdout (or startup_timeout_ms expires).
     *
     * @return true if server started and ready signal received.
     */
    bool start() {
        if (running_) return true;

        if (!std::filesystem::exists(server_path_)) {
            last_error_ = "server.py not found at: " + server_path_;
            return false;
        }

        // Create pipes: parent writes to child stdin, reads from child stdout
        int pipe_to_child[2];   // parent writes [1] → child reads [0]
        int pipe_from_child[2]; // child writes [1] → parent reads [0]

        if (::pipe(pipe_to_child) != 0 || ::pipe(pipe_from_child) != 0) {
            last_error_ = "pipe() failed";
            return false;
        }

        pid_ = ::fork();
        if (pid_ < 0) {
            last_error_ = "fork() failed";
            return false;
        }

        if (pid_ == 0) {
            // ---- Child process ----
            ::close(pipe_to_child[1]);
            ::close(pipe_from_child[0]);
            ::dup2(pipe_to_child[0], STDIN_FILENO);
            ::dup2(pipe_from_child[1], STDOUT_FILENO);
            ::close(pipe_to_child[0]);
            ::close(pipe_from_child[1]);

            ::execlp(python_bin_.c_str(), python_bin_.c_str(),
                     server_path_.c_str(), nullptr);
            ::_exit(127);  // exec failed
        }

        // ---- Parent process ----
        ::close(pipe_to_child[0]);
        ::close(pipe_from_child[1]);

        write_fd_ = pipe_to_child[1];
        read_fd_  = pipe_from_child[0];

        // Set read fd to non-blocking for timeout handling
        int flags = ::fcntl(read_fd_, F_GETFL, 0);
        ::fcntl(read_fd_, F_SETFL, flags | O_NONBLOCK);

        // Wait for ready signal
        auto t0 = std::chrono::steady_clock::now();
        std::string ready_line;
        while (true) {
            auto elapsed = std::chrono::steady_clock::now() - t0;
            if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
                    > startup_timeout_ms_) {
                stop();
                last_error_ = "startup timeout waiting for ready signal";
                return false;
            }

            char buf[1024];
            ssize_t n = ::read(read_fd_, buf, sizeof(buf) - 1);
            if (n > 0) {
                buf[n] = '\0';
                ready_line += buf;
                if (ready_line.find('\n') != std::string::npos) {
                    // Check if it contains "ready":true
                    if (ready_line.find("\"ready\"") != std::string::npos &&
                        ready_line.find("true") != std::string::npos) {
                        // Restore blocking mode for subsequent reads
                        ::fcntl(read_fd_, F_SETFL, flags);
                        running_ = true;
                        return true;
                    }
                }
            } else {
                // EAGAIN / EWOULDBLOCK — sleep briefly
                usleep(50000);  // 50ms
            }
        }
    }

    /**
     * @brief Stop the specialist server subprocess.
     */
    void stop() {
        if (write_fd_ >= 0) { ::close(write_fd_); write_fd_ = -1; }
        if (read_fd_ >= 0)  { ::close(read_fd_);  read_fd_ = -1; }
        if (pid_ > 0) {
            ::kill(pid_, SIGTERM);
            int status = 0;
            ::waitpid(pid_, &status, 0);
            pid_ = -1;
        }
        running_ = false;
    }

    /**
     * @brief Ask the specialist a question.
     *
     * @param instruction  The task instruction (e.g., "Write a function that...")
     * @param context      Optional context (existing code, etc.)
     * @param timeout_ms   Max time to wait for response (default: 60s)
     * @return             SpecialistResult with ok flag and response/error
     */
    SpecialistResult ask(const std::string& instruction,
                         const std::string& context = "",
                         int timeout_ms = 60000) {
        if (!running_) {
            return {false, {}, "specialist server not running"};
        }

        // Build JSON request (manual — no JSON library dependency)
        int req_id = next_id_++;
        std::string req = "{\"id\":" + std::to_string(req_id)
                        + ",\"instruction\":\"" + json_escape(instruction) + "\"";
        if (!context.empty()) {
            req += ",\"context\":\"" + json_escape(context) + "\"";
        }
        req += "}\n";

        // Write request
        ssize_t written = ::write(write_fd_, req.data(), req.size());
        if (written < 0 || static_cast<size_t>(written) != req.size()) {
            return {false, {}, "write to specialist pipe failed"};
        }

        // Read response line (blocking with timeout via poll-style)
        auto t0 = std::chrono::steady_clock::now();
        std::string line;
        char buf[4096];

        // Temporarily set non-blocking for timeout
        int flags = ::fcntl(read_fd_, F_GETFL, 0);
        ::fcntl(read_fd_, F_SETFL, flags | O_NONBLOCK);

        while (true) {
            auto elapsed = std::chrono::steady_clock::now() - t0;
            if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
                    > timeout_ms) {
                ::fcntl(read_fd_, F_SETFL, flags);
                return {false, {}, "response timeout (" + std::to_string(timeout_ms) + "ms)"};
            }

            ssize_t n = ::read(read_fd_, buf, sizeof(buf) - 1);
            if (n > 0) {
                buf[n] = '\0';
                line += buf;
                if (line.find('\n') != std::string::npos) {
                    break;
                }
            } else {
                usleep(10000);  // 10ms
            }
        }

        // Restore blocking mode
        ::fcntl(read_fd_, F_SETFL, flags);

        // Parse response — minimal JSON parsing
        return parse_response(line);
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    [[nodiscard]] bool running() const noexcept { return running_; }
    [[nodiscard]] const std::string& last_error() const noexcept { return last_error_; }
    [[nodiscard]] pid_t pid() const noexcept { return pid_; }

    static std::string default_server_path() {
        const char* env = std::getenv("ARIA_SPECIALIST_SERVER");
        if (env && *env) return env;
        auto home = std::filesystem::path(
            std::getenv("HOME") ? std::getenv("HOME") : "/root");
        return (home / "Workspace" / "REPOS" / "aria-specialist" / "scripts" / "server.py").string();
    }

private:
    std::string server_path_;
    std::string python_bin_;
    int         startup_timeout_ms_;

    pid_t       pid_      = -1;
    int         write_fd_ = -1;
    int         read_fd_  = -1;
    bool        running_  = false;
    int         next_id_  = 1;
    std::string last_error_;

    /// Escape a string for JSON embedding (minimal — handles \n, \r, \t, ", \\)
    static std::string json_escape(const std::string& s) {
        std::string out;
        out.reserve(s.size() + s.size() / 8);
        for (char c : s) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        // Control character — skip or hex-encode
                        char hex[8];
                        snprintf(hex, sizeof(hex), "\\u%04x", static_cast<unsigned>(c));
                        out += hex;
                    } else {
                        out += c;
                    }
            }
        }
        return out;
    }

    /// Parse a JSON-Lines response: {"id":N, "ok":bool, "response":"...", "error":"..."}
    static SpecialistResult parse_response(const std::string& line) {
        SpecialistResult result;

        // Check ok field
        result.ok = (line.find("\"ok\":true") != std::string::npos ||
                     line.find("\"ok\": true") != std::string::npos);

        // Extract "response" field value
        if (result.ok) {
            result.response = extract_json_string(line, "response");
        } else {
            result.error = extract_json_string(line, "error");
            if (result.error.empty()) {
                result.error = "unknown specialist error";
            }
        }
        return result;
    }

    /// Extract a JSON string field value (minimal parser — no nested objects)
    static std::string extract_json_string(const std::string& json,
                                            const std::string& key) {
        std::string needle = "\"" + key + "\":\"";
        auto pos = json.find(needle);
        if (pos == std::string::npos) {
            needle = "\"" + key + "\": \"";
            pos = json.find(needle);
        }
        if (pos == std::string::npos) return "";

        pos += needle.size();
        std::string value;
        bool escaped = false;
        for (size_t i = pos; i < json.size(); ++i) {
            char c = json[i];
            if (escaped) {
                switch (c) {
                    case '"':  value += '"';  break;
                    case '\\': value += '\\'; break;
                    case 'n':  value += '\n'; break;
                    case 'r':  value += '\r'; break;
                    case 't':  value += '\t'; break;
                    default:   value += c;    break;
                }
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                break;
            } else {
                value += c;
            }
        }
        return value;
    }
};

} // namespace nikola::aria
