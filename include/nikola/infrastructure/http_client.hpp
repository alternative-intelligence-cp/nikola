#pragma once
/**
 * @file include/nikola/infrastructure/http_client.hpp
 * @brief Phase 32 — HTTP Client: lightweight libcurl wrapper for oracle agents.
 *
 * Provides a simple, safe C++ interface over libcurl for HTTP GET/POST requests
 * with JSON support, TLS, timeouts, and rate limiting.
 *
 * Design:
 *   · One HttpClient instance = one curl easy handle (not thread-safe).
 *   · Create one per thread or serialize access externally.
 *   · Rate limiting is per-instance via token bucket.
 *   · Responses include status code, headers, and body.
 *
 * Usage:
 *   HttpClient client;
 *   auto resp = client.get("https://api.example.com/data");
 *   if (resp.ok()) {
 *       std::cout << resp.body << "\n";
 *   }
 *
 *   auto resp2 = client.post_json("https://api.example.com/search",
 *                                 R"({"query":"hello"})");
 *
 * Phase: NIK-HTTP-01 (HTTP Client, Phase 32)
 */

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nikola::infrastructure {

// ============================================================================
// HttpResponse
// ============================================================================

/**
 * @brief Response from an HTTP request.
 */
struct HttpResponse {
    int         status_code = 0;       ///< HTTP status (200, 404, etc.); 0 on network error
    std::string body;                  ///< Response body
    std::string error;                 ///< Error message (empty on success)
    double      elapsed_seconds = 0.0; ///< Wall-clock time for the request

    /// True if status_code is in [200, 299].
    [[nodiscard]] bool ok() const noexcept {
        return status_code >= 200 && status_code < 300;
    }
};

// ============================================================================
// HttpClientConfig
// ============================================================================

/**
 * @brief Configuration for HttpClient.
 */
struct HttpClientConfig {
    /// Connection timeout in seconds.
    long connect_timeout_s = 10;

    /// Total request timeout in seconds (including transfer).
    long request_timeout_s = 30;

    /// Maximum response body size in bytes (0 = unlimited).
    size_t max_response_bytes = 10 * 1024 * 1024;  // 10 MB

    /// User-Agent header value.
    std::string user_agent = "NikolaHTTP/0.2.0";

    /// Verify TLS peer certificate (should be true in production).
    bool verify_tls = true;

    /// Minimum interval between requests (rate limiting). Zero = no limit.
    std::chrono::milliseconds min_request_interval{0};

    /// Maximum retry count on transient failures (5xx, timeout).
    int max_retries = 2;

    /// Base delay for exponential backoff between retries.
    std::chrono::milliseconds retry_base_delay{500};
};

// ============================================================================
// HttpClient
// ============================================================================

/**
 * @class HttpClient
 * @brief Lightweight HTTP client wrapping libcurl.
 *
 * Not thread-safe — create one per thread or serialize access.
 * RAII: curl handle is acquired in constructor, released in destructor.
 */
class HttpClient {
public:
    /// Construct with optional configuration.
    explicit HttpClient(const HttpClientConfig& config = {});

    /// Non-copyable, movable.
    HttpClient(const HttpClient&) = delete;
    HttpClient& operator=(const HttpClient&) = delete;
    HttpClient(HttpClient&& other) noexcept;
    HttpClient& operator=(HttpClient&& other) noexcept;

    ~HttpClient();

    // ── Requests ──────────────────────────────────────────────────────────────

    /**
     * @brief Perform an HTTP GET request.
     *
     * @param url     Full URL (must start with http:// or https://).
     * @param headers Optional additional headers (key: value).
     * @return HttpResponse with status, body, and error info.
     */
    [[nodiscard]] HttpResponse get(
        const std::string& url,
        const std::unordered_map<std::string, std::string>& headers = {});

    /**
     * @brief Perform an HTTP POST request with a JSON body.
     *
     * Sets Content-Type: application/json automatically.
     *
     * @param url      Full URL.
     * @param json_body JSON string to send as the request body.
     * @param headers   Optional additional headers.
     * @return HttpResponse.
     */
    [[nodiscard]] HttpResponse post_json(
        const std::string& url,
        const std::string& json_body,
        const std::unordered_map<std::string, std::string>& headers = {});

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Number of requests made since construction.
    [[nodiscard]] uint64_t request_count() const noexcept { return request_count_; }

    /// Access the current configuration.
    [[nodiscard]] const HttpClientConfig& config() const noexcept { return config_; }

private:
    /// Perform the actual curl request (shared by get/post).
    HttpResponse perform_(const std::string& url,
                          const std::string& method,
                          const std::string& body,
                          const std::unordered_map<std::string, std::string>& headers);

    /// Apply rate limiting (sleep if needed).
    void rate_limit_();

    /// Determine if an error/status is retryable.
    static bool is_retryable_(int status_code, const std::string& error);

    HttpClientConfig config_;
    void*            curl_ = nullptr;  ///< CURL* handle (void* to avoid leaking curl.h)
    uint64_t         request_count_ = 0;

    /// Timestamp of last request (for rate limiting).
    std::chrono::steady_clock::time_point last_request_time_{};
};

} // namespace nikola::infrastructure
