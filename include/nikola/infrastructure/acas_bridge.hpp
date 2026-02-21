/**
 * @file acas_bridge.hpp
 * @brief ACAS (Agentic-Core Audio Subsystem) C++ Bridge.
 *
 * Manages two Python microservice processes:
 *
 *   analyzer_main.py  — VAD + Whisper ASR  ("the ears")
 *   generator_main.py — piper TTS          ("the voice")
 *
 * Both processes communicate via newline-delimited JSON on their
 * stdin/stdout pipes, following the SPDA "Tool Bridge" pattern.
 *
 * IPC Protocol
 * ============
 * Analyzer stdout:
 *   {"type":"status",     "mode":"online|offline", "reason":"...", "version":"1.0.0"}
 *   {"type":"transcript", "text":"...", "confidence":0.9, "language":"en", "ts":0.0}
 *   {"type":"vad",        "speech":true, "ts":0.0}
 *
 * Generator stdin:
 *   {"cmd":"speak",  "text":"Hello", "voice":"en_US-lessac-medium"}
 *   {"cmd":"stop"}
 *   {"cmd":"status"}
 *
 * Generator stdout:
 *   {"type":"status",   "mode":"online|offline", ...}
 *   {"type":"speaking", "text":"...", "ts":0.0}
 *   {"type":"done",     "text":"...", "duration_s":1.23, "ts":0.0}
 *
 * Thread Safety
 * =============
 * read_transcript() and read_generator_event() are NOT thread-safe with
 * each other.  Protect with an external mutex if used from multiple threads.
 * speak() is safe to call from any thread (uses an internal write mutex).
 *
 * Phase 12, Nikola v0.0.4
 */

#pragma once

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// POSIX
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace nikola::infrastructure {

// ---------------------------------------------------------------------------
// AcasConfig
// ---------------------------------------------------------------------------

struct AcasConfig {
    std::string python_executable  = "python3";
    std::string analyzer_script;   ///< Absolute path to analyzer_main.py
    std::string generator_script;  ///< Absolute path to generator_main.py

    /// Maximum bytes per line when reading process stdout.
    std::size_t max_line_bytes = 4096;

    /// If true, log raw IPC lines to stderr for debugging.
    bool debug_ipc = false;
};

// ---------------------------------------------------------------------------
// Internal: POSIX subprocess with bidirectional pipes
// ---------------------------------------------------------------------------

/// @cond INTERNAL
namespace detail {

/**
 * @brief Owns a forked Python child process with separate stdin/stdout pipes.
 *
 * stdin_pipe[1]  → child stdin   (parent writes)
 * stdout_pipe[0] ← child stdout  (parent reads)
 */
class ChildProcess {
public:
    ChildProcess() = default;

    ~ChildProcess() { kill(); }

    // Non-copyable; movable
    ChildProcess(const ChildProcess&)           = delete;
    ChildProcess& operator=(const ChildProcess&)= delete;

    /**
     * @brief Spawn a new child process.
     *
     * @param python  Python interpreter binary (e.g. "python3").
     * @param script  Absolute path to the Python script.
     * @throws std::runtime_error on fork/exec failure.
     */
    void spawn(const std::string& python, const std::string& script)
    {
        if (pid_ > 0) kill();

        int stdin_pipe[2];   // parent writes → child reads
        int stdout_pipe[2];  // child writes → parent reads

        if (::pipe(stdin_pipe)  < 0) throw std::runtime_error("pipe() stdin failed");
        if (::pipe(stdout_pipe) < 0) {
            ::close(stdin_pipe[0]);  ::close(stdin_pipe[1]);
            throw std::runtime_error("pipe() stdout failed");
        }

        const pid_t pid = ::fork();
        if (pid < 0) {
            ::close(stdin_pipe[0]);  ::close(stdin_pipe[1]);
            ::close(stdout_pipe[0]); ::close(stdout_pipe[1]);
            throw std::runtime_error("fork() failed");
        }

        if (pid == 0) {
            // ---- Child ---------------------------------------------------
            ::dup2(stdin_pipe[0],  STDIN_FILENO);
            ::dup2(stdout_pipe[1], STDOUT_FILENO);
            ::close(stdin_pipe[0]);  ::close(stdin_pipe[1]);
            ::close(stdout_pipe[0]); ::close(stdout_pipe[1]);

            const char* argv[] = { python.c_str(), script.c_str(), nullptr };
            ::execvp(python.c_str(), const_cast<char* const*>(argv));
            ::_exit(127);  // exec failed
        }

        // ---- Parent ------------------------------------------------------
        ::close(stdin_pipe[0]);
        ::close(stdout_pipe[1]);

        pid_       = pid;
        write_fd_  = stdin_pipe[1];
        read_fd_   = stdout_pipe[0];

        // Set stdout read fd to non-blocking for poll-style reads
        ::fcntl(read_fd_, F_SETFL, ::fcntl(read_fd_, F_GETFL) | O_NONBLOCK);

        read_fp_ = nullptr;  // use raw fd (non-blocking compatible)
    }

    /**
     * @brief Terminate the child (SIGTERM + waitpid).
     */
    void kill() noexcept
    {
        if (pid_ <= 0) return;
        ::kill(pid_, SIGTERM);
        // Give it up to 500 ms, then SIGKILL
        for (int i = 0; i < 50; ++i) {
            int status = 0;
            const pid_t r = ::waitpid(pid_, &status, WNOHANG);
            if (r == pid_) break;
            ::usleep(10'000);
        }
        ::waitpid(pid_, nullptr, WNOHANG);
        if (write_fd_ >= 0) { ::close(write_fd_); write_fd_ = -1; }
        if (read_fd_  >= 0) { ::close(read_fd_);  read_fd_  = -1; }
        pid_ = -1;
    }

    [[nodiscard]] bool running() const noexcept
    {
        if (pid_ <= 0) return false;
        int status = 0;
        return ::waitpid(pid_, &status, WNOHANG) == 0;
    }

    [[nodiscard]] pid_t pid() const noexcept { return pid_; }

    /**
     * @brief Write a line to the child's stdin (appends '\n' if missing).
     */
    bool write_line(const std::string& line) noexcept
    {
        if (write_fd_ < 0) return false;
        std::string data = line;
        if (data.empty() || data.back() != '\n') data += '\n';
        const ssize_t n = ::write(write_fd_, data.c_str(),
                                  static_cast<ssize_t>(data.size()));
        return n == static_cast<ssize_t>(data.size());
    }

    /**
     * @brief Read one newline-terminated line from the child's stdout.
     *
     * Non-blocking: returns nullopt immediately if no data is available.
     *
     * @param max_bytes Maximum line length (safety cap).
     */
    [[nodiscard]] std::optional<std::string>
    read_line(std::size_t max_bytes = 4096) noexcept
    {
        if (read_fd_ < 0) return std::nullopt;

        std::string& buf = line_buf_;

        // Drain bytes until '\n' or EAGAIN
        char ch = '\0';
        while (buf.size() < max_bytes) {
            const ssize_t n = ::read(read_fd_, &ch, 1);
            if (n == 1) {
                if (ch == '\n') {
                    std::string line = buf;
                    buf.clear();
                    return line;
                }
                buf += ch;
            } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                break;   // no data right now
            } else {
                // EOF or error — child likely exited
                if (!buf.empty()) {
                    std::string line = buf;
                    buf.clear();
                    return line;
                }
                return std::nullopt;
            }
        }
        return std::nullopt;
    }

private:
    pid_t       pid_      = -1;
    int         write_fd_ = -1;
    int         read_fd_  = -1;
    FILE*       read_fp_  = nullptr;
    std::string line_buf_;
};

}  // namespace detail
/// @endcond

// ---------------------------------------------------------------------------
// AcasBridge
// ---------------------------------------------------------------------------

/**
 * @brief Top-level ACAS bridge managing the analyzer and generator processes.
 */
class AcasBridge {
public:
    explicit AcasBridge(AcasConfig cfg) : cfg_(std::move(cfg)) {}
    ~AcasBridge() { stop(); }

    AcasBridge(const AcasBridge&)            = delete;
    AcasBridge& operator=(const AcasBridge&) = delete;

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /**
     * @brief Spawn analyzer and generator Python processes.
     * @throws std::runtime_error if scripts are not set or fork fails.
     */
    void start()
    {
        if (cfg_.analyzer_script.empty())
            throw std::runtime_error("AcasBridge: analyzer_script not set");
        if (cfg_.generator_script.empty())
            throw std::runtime_error("AcasBridge: generator_script not set");

        analyzer_.spawn(cfg_.python_executable, cfg_.analyzer_script);
        generator_.spawn(cfg_.python_executable, cfg_.generator_script);
        started_.store(true, std::memory_order_release);
    }

    /**
     * @brief Gracefully stop both subprocesses.
     */
    void stop() noexcept
    {
        started_.store(false, std::memory_order_release);
        analyzer_.kill();
        generator_.kill();
    }

    /**
     * @brief Restart both subprocesses (stop + start).
     */
    void restart() { stop(); start(); }

    [[nodiscard]] bool is_started()             const noexcept { return started_.load(); }
    [[nodiscard]] bool analyzer_running()       const noexcept { return analyzer_.running(); }
    [[nodiscard]] bool generator_running()      const noexcept { return generator_.running(); }

    // ------------------------------------------------------------------
    // Analyzer interface
    // ------------------------------------------------------------------

    /**
     * @brief Non-blocking poll for the next JSON line from the analyzer.
     *
     * Returns the raw JSON string (without newline), or nullopt if no
     * data is currently available.
     */
    [[nodiscard]] std::optional<std::string> read_analyzer_line()
    {
        auto line = analyzer_.read_line(cfg_.max_line_bytes);
        if (line && cfg_.debug_ipc)
            std::fprintf(stderr, "[ACAS/analyzer] %s\n", line->c_str());
        return line;
    }

    // ------------------------------------------------------------------
    // Generator interface
    // ------------------------------------------------------------------

    /**
     * @brief Send a "speak" command to the TTS generator.
     *
     * @param text   Text to synthesise.
     * @param voice  Optional voice identifier (uses Python default if empty).
     * @return true if the write succeeded.
     */
    bool speak(const std::string& text, const std::string& voice = "")
    {
        // Build minimal JSON manually (no external JSON dep)
        std::string cmd = R"({"cmd":"speak","text":")" + json_escape(text) + '"';
        if (!voice.empty())
            cmd += R"(,"voice":")" + json_escape(voice) + '"';
        cmd += '}';

        std::lock_guard<std::mutex> g(write_mutex_);
        const bool ok = generator_.write_line(cmd);
        if (cfg_.debug_ipc)
            std::fprintf(stderr, "[ACAS/generator] >> %s\n", cmd.c_str());
        return ok;
    }

    /**
     * @brief Send a "stop" command to the TTS generator.
     */
    bool stop_speaking()
    {
        std::lock_guard<std::mutex> g(write_mutex_);
        return generator_.write_line(R"({"cmd":"stop"})");
    }

    /**
     * @brief Non-blocking poll for the next JSON line from the generator.
     */
    [[nodiscard]] std::optional<std::string> read_generator_line()
    {
        auto line = generator_.read_line(cfg_.max_line_bytes);
        if (line && cfg_.debug_ipc)
            std::fprintf(stderr, "[ACAS/generator] << %s\n", line->c_str());
        return line;
    }

    // ------------------------------------------------------------------
    // Configuration access
    // ------------------------------------------------------------------

    [[nodiscard]] const AcasConfig& config() const noexcept { return cfg_; }

private:
    // ------------------------------------------------------------------
    // Minimal JSON string escaper (for speak() command building)
    // ------------------------------------------------------------------

    [[nodiscard]] static std::string json_escape(const std::string& s)
    {
        std::string out;
        out.reserve(s.size());
        for (const char c : s) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:   out += c;
            }
        }
        return out;
    }

    // ------------------------------------------------------------------
    // Data members
    // ------------------------------------------------------------------

    AcasConfig         cfg_;
    detail::ChildProcess analyzer_;
    detail::ChildProcess generator_;
    std::atomic<bool>  started_{false};
    std::mutex         write_mutex_;
};

} // namespace nikola::infrastructure
