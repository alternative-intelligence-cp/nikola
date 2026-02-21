/**
 * @file security/bootstrap_manager.hpp
 * @brief NIK-002 — Secure tiered bootstrap token management.
 *
 * Resolves the "Secret Zero Paradox" in headless/containerized deployments.
 *
 * Background:
 *   The original Nikola v0.0.4 spec generates a 256-bit bootstrap token and
 *   prints it to stdout with a 300-second expiry window.  In containerized
 *   environments (Kubernetes / Docker Swarm), log aggregation latency (Fluentd,
 *   ELK, Splunk), pod CrashLoopBackOff cycles, and journald buffering can
 *   render the stdout token inaccessible before the 300-second lockdown, causing
 *   administrative deadlock.
 *
 * Tiered strategy (priority order):
 *   Tier 1 — Environment variable NIKOLA_BOOTSTRAP_TOKEN
 *             Read at startup, immediately scrubbed from memory and unsetenv()
 *             to prevent leakage via /proc/self/environ or core dumps.
 *             Suitable for Docker/K8s env-var injection; use with Kubernetes
 *             Secrets or Docker Secrets (via secretKeyRef) — never raw YAML.
 *
 *   Tier 2 — File secret /run/secrets/nikola_bootstrap_token
 *             Docker Secrets / Kubernetes VolumeMount files.  The file path
 *             is configurable for testing via set_secret_file_path().
 *
 *   Tier 3 — Fallback: generate 256-bit random token from std::random_device
 *             and print it to stdout with a boxed ASCII border + std::flush.
 *             (Suitable for bare-metal interactive deployments.)
 *
 * Security properties:
 *   - Tokens are 256-bit (64 hex chars), generated from std::random_device
 *   - 300-second expiry window (configurable for tests)
 *   - Rate limiting: 5 failed validate() attempts per 60s → 5s delay ("Paranoid Mode")
 *   - Env var received pointer is volatile-scrubbed + unsetenv() immediately
 *
 * @see TASKS.md  NIK-002
 * @see TASK_BOOTSTRAP_TOKEN_SECURITY.txt (Round 4 spec)
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>      // getenv, unsetenv
#include <cstring>      // strlen
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>       // sleep_for (rate limit delay)
#include <unordered_map>

namespace nikola::security {

// ─────────────────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Bootstrap token expiry window (seconds).
inline constexpr int BOOTSTRAP_EXPIRY_SECONDS = 300;

/// Token entropy: 256 bits = 32 bytes = 64 hex chars.
inline constexpr std::size_t BOOTSTRAP_TOKEN_BYTES  = 32;
inline constexpr std::size_t BOOTSTRAP_TOKEN_LENGTH = 64;   // hex chars

/// Rate-limiting: max failures before paranoid mode (per window).
inline constexpr int RATE_LIMIT_MAX_FAILURES = 5;

/// Rate-limiting window (seconds).
inline constexpr int RATE_LIMIT_WINDOW_SECONDS = 60;

/// Delay imposed once rate limit exceeded (seconds).
inline constexpr int RATE_LIMIT_DELAY_SECONDS = 5;

/// Environment variable name for Tier 1 injection.
inline constexpr const char* BOOTSTRAP_ENV_VAR = "NIKOLA_BOOTSTRAP_TOKEN";

/// Default file path for Tier 2 Docker/K8s secrets.
inline constexpr const char* BOOTSTRAP_SECRET_FILE = "/run/secrets/nikola_bootstrap_token";

// ─────────────────────────────────────────────────────────────────────────────
//  BootstrapManager
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Manages secure bootstrap token acquisition and validation.
 *
 * Not thread-safe — drive from a single initialization thread.
 * Call reset() to restore fresh state for testing.
 */
class BootstrapManager {
public:
    // ── Types ──────────────────────────────────────────────────────────────

    enum class TokenSource : uint8_t {
        ENV_VAR,   ///< Tier 1 — environment variable
        FILE,      ///< Tier 2 — file secret
        GENERATED, ///< Tier 3 — generated from random_device
        NONE,      ///< No token acquired yet
    };

    // ── Configuration ──────────────────────────────────────────────────────

    /**
     * @brief Override the Tier 2 secret file path (for testing).
     *
     * Default: /run/secrets/nikola_bootstrap_token
     */
    void set_secret_file_path(std::string path) {
        secret_file_path_override_ = std::move(path);
    }

    /**
     * @brief Override the expiry window in seconds (for testing; default 300).
     */
    void set_expiry_seconds(int seconds) noexcept {
        expiry_seconds_ = seconds;
    }

    /**
     * @brief Override the rate-limit delay (for testing; default 5s).
     */
    void set_rate_limit_delay(int seconds) noexcept {
        rate_limit_delay_seconds_ = seconds;
    }

    /**
     * @brief Suppress stdout output when generating a Tier 3 token (for tests).
     */
    void set_silent(bool silent) noexcept {
        silent_ = silent;
    }

    // ── Primary API ────────────────────────────────────────────────────────

    /**
     * @brief Acquire a bootstrap token using the tiered strategy.
     *
     * Checks sources in priority order:
     *   1. NIKOLA_BOOTSTRAP_TOKEN env var (atomic scrub on read)
     *   2. Secret file (default /run/secrets/nikola_bootstrap_token)
     *   3. Random generation → printed to stdout
     *
     * After get_token() returns, the expiry timer starts.
     *
     * @return 64-character hex token string.
     */
    std::string get_token() {
        // --- Tier 1: Environment variable ---------------------------------
        {
            char* env_raw = std::getenv(BOOTSTRAP_ENV_VAR);
            if (env_raw && env_raw[0] != '\0') {
                std::string token(env_raw);

                // SECURITY: Overwrite the env var's memory immediately to
                // prevent leakage in /proc/self/environ, child process
                // inheritance, or core dumps.
                const std::size_t len = std::strlen(env_raw);
                volatile char* p = env_raw;
                for (std::size_t i = 0; i < len; ++i) p[i] = '\0';
                ::unsetenv(BOOTSTRAP_ENV_VAR);

                store_token(token, TokenSource::ENV_VAR);
                std::cerr << "[BootstrapManager] Token loaded from env var (SCRUBBED)\n";
                return token;
            }
        }

        // --- Tier 2: File secret ------------------------------------------
        {
            const std::filesystem::path fpath(secret_file_path());
            if (std::filesystem::exists(fpath)) {
                std::ifstream ifs(fpath);
                if (ifs) {
                    std::string token;
                    std::getline(ifs, token);
                    // Trim trailing whitespace / newline
                    const auto last = token.find_last_not_of(" \t\r\n");
                    if (last != std::string::npos)
                        token.erase(last + 1);
                    else
                        token.clear();

                    if (!token.empty()) {
                        store_token(token, TokenSource::FILE);
                        std::cerr << "[BootstrapManager] Token loaded from file: "
                                  << fpath.string() << '\n';
                        return token;
                    }
                }
            }
        }

        // --- Tier 3: Generate random token --------------------------------
        {
            const std::string token = generate_random_token();
            store_token(token, TokenSource::GENERATED);

            if (!silent_) {
                print_token_box(token);
            }

            return token;
        }
    }

    /**
     * @brief Validate a token supplied by a client.
     *
     * Checks:
     *   - Rate limiting: if the client has exceeded 5 failures in 60s, apply
     *     a 5-second delay and reject (returns false).
     *   - Expiry: if the token window has closed, reject.
     *   - Match: compare token_ against the supplied string.
     *
     * @param token      Token string supplied by the client.
     * @param client_id  Opaque client identifier for rate-limiting
     *                   (e.g., IP address, pod name).  Empty string → global.
     * @return           true if the token is valid and not expired.
     */
    bool validate(const std::string& token, const std::string& client_id = "") {
        // --- Rate limit check -------------------------------------------
        if (is_rate_limited(client_id)) {
            std::this_thread::sleep_for(
                std::chrono::seconds(rate_limit_delay_seconds_));
            record_failure(client_id);
            return false;
        }

        // --- Expiry check ------------------------------------------------
        if (token_.empty() || is_expired()) {
            record_failure(client_id);
            return false;
        }

        // --- Token comparison (constant-time style — avoid early exit) ---
        const bool match = constant_time_equal(token_, token);

        if (!match) {
            record_failure(client_id);
        }

        return match;
    }

    /**
     * @brief Return true if the bootstrap token has expired.
     *
     * Returns true if no token has been acquired yet, or if more than
     * expiry_seconds_ have elapsed since get_token() was called.
     */
    [[nodiscard]]
    bool is_expired() const noexcept {
        if (token_.empty()) return true;
        const auto now     = Clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                now - token_created_at_).count();
        return elapsed >= expiry_seconds_;
    }

    /**
     * @brief Return the token source (Tier 1/2/3 or NONE before get_token()).
     */
    [[nodiscard]]
    TokenSource source() const noexcept { return source_; }

    /**
     * @brief Return the current token (empty string if not yet acquired).
     *
     * @note  Exposed primarily for testing.  Do not log this in production.
     */
    [[nodiscard]]
    const std::string& token() const noexcept { return token_; }

    /**
     * @brief Return the number of failed validation attempts for a client_id.
     */
    [[nodiscard]]
    int failure_count(const std::string& client_id = "") const {
        auto it = rate_limits_.find(client_id);
        if (it == rate_limits_.end()) return 0;
        return it->second.count;
    }

    /**
     * @brief Reset all state: clear token, expiry, and rate-limit table.
     *
     * Primarily for unit tests.
     */
    void reset() {
        token_.clear();
        source_           = TokenSource::NONE;
        token_created_at_ = Clock::time_point{};
        rate_limits_.clear();
    }

    // ── Utility ────────────────────────────────────────────────────────────

    /**
     * @brief Generate a 256-bit random token using std::random_device.
     *
     * @return 64-character lowercase hex string.
     */
    static std::string generate_random_token() {
        std::random_device rdev;
        // 32 bytes = 256 bits; random_device gives 32-bit words → need 8
        std::array<uint32_t, 8> words{};
        for (auto& w : words) w = rdev();

        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (const auto w : words)
            oss << std::setw(8) << w;

        return oss.str();
    }

private:
    // ── Internal types ─────────────────────────────────────────────────────

    using Clock = std::chrono::steady_clock;

    struct RateLimitEntry {
        int         count      = 0;
        Clock::time_point first_failure{};
    };

    // ── State ──────────────────────────────────────────────────────────────

    std::string      token_;
    TokenSource      source_           = TokenSource::NONE;
    Clock::time_point token_created_at_{};
    int              expiry_seconds_          = BOOTSTRAP_EXPIRY_SECONDS;
    int              rate_limit_delay_seconds_= RATE_LIMIT_DELAY_SECONDS;
    bool             silent_                  = false;
    std::string      secret_file_path_override_;

    std::unordered_map<std::string, RateLimitEntry> rate_limits_;

    // ── Helpers ────────────────────────────────────────────────────────────

    const char* secret_file_path() const noexcept {
        if (!secret_file_path_override_.empty())
            return secret_file_path_override_.c_str();
        return BOOTSTRAP_SECRET_FILE;
    }

    void store_token(const std::string& t, TokenSource src) {
        token_            = t;
        source_           = src;
        token_created_at_ = Clock::now();
    }

    /// Returns true if client_id has hit the rate limit within the window.
    bool is_rate_limited(const std::string& client_id) {
        auto it = rate_limits_.find(client_id);
        if (it == rate_limits_.end()) return false;

        const auto& entry = it->second;
        if (entry.count < RATE_LIMIT_MAX_FAILURES) return false;

        // Check if the window has expired (reset if so)
        const auto now     = Clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                 now - entry.first_failure).count();
        if (elapsed >= RATE_LIMIT_WINDOW_SECONDS) {
            rate_limits_.erase(it);
            return false;
        }

        return true;
    }

    void record_failure(const std::string& client_id) {
        auto& entry = rate_limits_[client_id];
        if (entry.count == 0) {
            entry.first_failure = Clock::now();
        }
        ++entry.count;
    }

    /// Constant-time string comparison (avoids early-exit timing side-channels).
    static bool constant_time_equal(const std::string& a, const std::string& b) noexcept {
        if (a.size() != b.size()) return false;
        volatile unsigned diff = 0;
        for (std::size_t i = 0; i < a.size(); ++i)
            diff |= static_cast<unsigned>(
                static_cast<unsigned char>(a[i]) ^
                static_cast<unsigned char>(b[i]));
        return diff == 0u;
    }

    static void print_token_box(const std::string& token) {
        std::cout << "\n\n";
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║             NIKOLA MODEL v0.0.4 BOOTSTRAP TOKEN            ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  " << token << "  ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  EXPIRES IN " << BOOTSTRAP_EXPIRY_SECONDS
                  << " SECONDS                                     ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";
        std::cout << std::flush;  // CRITICAL: flush for journald buffering
    }
};

} // namespace nikola::security
