/**
 * @file src/infrastructure/http_client.cpp
 * @brief Phase 32 — HttpClient implementation (libcurl backend).
 */

#include <nikola/infrastructure/http_client.hpp>

#include <curl/curl.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace nikola::infrastructure {

// ============================================================================
// curl write callback
// ============================================================================

namespace {

struct WriteContext {
    std::string* buffer;
    size_t       max_bytes;
};

/// Called by curl when data arrives.  Appends to the WriteContext buffer.
size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    const size_t total = size * nmemb;
    auto* ctx = static_cast<WriteContext*>(userdata);

    // Enforce max response size
    if (ctx->max_bytes > 0 &&
        ctx->buffer->size() + total > ctx->max_bytes) {
        return 0;  // Signal curl to abort
    }

    ctx->buffer->append(ptr, total);
    return total;
}

} // anonymous namespace

// ============================================================================
// Global curl init/cleanup via static guard
// ============================================================================

namespace {

struct CurlGlobalInit {
    CurlGlobalInit()  { curl_global_init(CURL_GLOBAL_DEFAULT); }
    ~CurlGlobalInit() { curl_global_cleanup(); }
};

/// Ensures curl_global_init is called exactly once and cleaned up at exit.
void ensure_global_init() {
    static CurlGlobalInit guard;
}

} // anonymous namespace

// ============================================================================
// Construction / destruction / move
// ============================================================================

HttpClient::HttpClient(const HttpClientConfig& config)
    : config_(config)
{
    ensure_global_init();

    curl_ = curl_easy_init();
    if (!curl_) {
        throw std::runtime_error("HttpClient: curl_easy_init() failed");
    }
}

HttpClient::HttpClient(HttpClient&& other) noexcept
    : config_(std::move(other.config_))
    , curl_(other.curl_)
    , request_count_(other.request_count_)
    , last_request_time_(other.last_request_time_)
{
    other.curl_ = nullptr;
    other.request_count_ = 0;
}

HttpClient& HttpClient::operator=(HttpClient&& other) noexcept {
    if (this != &other) {
        if (curl_) {
            curl_easy_cleanup(static_cast<CURL*>(curl_));
        }
        config_ = std::move(other.config_);
        curl_ = other.curl_;
        request_count_ = other.request_count_;
        last_request_time_ = other.last_request_time_;

        other.curl_ = nullptr;
        other.request_count_ = 0;
    }
    return *this;
}

HttpClient::~HttpClient() {
    if (curl_) {
        curl_easy_cleanup(static_cast<CURL*>(curl_));
    }
}

// ============================================================================
// Public API
// ============================================================================

HttpResponse HttpClient::get(
    const std::string& url,
    const std::unordered_map<std::string, std::string>& headers)
{
    return perform_(url, "GET", "", headers);
}

HttpResponse HttpClient::post_json(
    const std::string& url,
    const std::string& json_body,
    const std::unordered_map<std::string, std::string>& headers)
{
    auto merged = headers;
    // Set Content-Type if not already provided
    if (merged.find("Content-Type") == merged.end()) {
        merged["Content-Type"] = "application/json";
    }
    return perform_(url, "POST", json_body, merged);
}

// ============================================================================
// perform_ — shared request execution
// ============================================================================

HttpResponse HttpClient::perform_(
    const std::string& url,
    const std::string& method,
    const std::string& body,
    const std::unordered_map<std::string, std::string>& headers)
{
    if (!curl_) {
        return { 0, "", "HttpClient: no curl handle (moved-from?)", 0.0 };
    }

    // URL validation — must start with http:// or https://
    if (url.rfind("http://", 0) != 0 && url.rfind("https://", 0) != 0) {
        return { 0, "", "HttpClient: URL must start with http:// or https://", 0.0 };
    }

    int attempts = 0;
    const int max_attempts = 1 + std::max(0, config_.max_retries);

    HttpResponse last_response;

    while (attempts < max_attempts) {
        if (attempts > 0) {
            // Exponential backoff: base * 2^(attempt-1)
            auto delay = config_.retry_base_delay * (1 << (attempts - 1));
            std::this_thread::sleep_for(delay);
        }

        rate_limit_();

        auto* handle = static_cast<CURL*>(curl_);
        curl_easy_reset(handle);

        // Response buffer
        std::string response_body;
        response_body.reserve(4096);

        WriteContext write_ctx{ &response_body, config_.max_response_bytes };

        // Core options
        curl_easy_setopt(handle, CURLOPT_URL, url.c_str());
        curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(handle, CURLOPT_WRITEDATA, &write_ctx);
        curl_easy_setopt(handle, CURLOPT_USERAGENT, config_.user_agent.c_str());
        curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT, config_.connect_timeout_s);
        curl_easy_setopt(handle, CURLOPT_TIMEOUT, config_.request_timeout_s);
        curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(handle, CURLOPT_MAXREDIRS, 5L);

        // TLS
        if (config_.verify_tls) {
            curl_easy_setopt(handle, CURLOPT_SSL_VERIFYPEER, 1L);
            curl_easy_setopt(handle, CURLOPT_SSL_VERIFYHOST, 2L);
        } else {
            curl_easy_setopt(handle, CURLOPT_SSL_VERIFYPEER, 0L);
            curl_easy_setopt(handle, CURLOPT_SSL_VERIFYHOST, 0L);
        }

        // Prevent file:// and other non-HTTP protocols
        curl_easy_setopt(handle, CURLOPT_PROTOCOLS_STR, "http,https");

        // Method + body
        if (method == "POST") {
            curl_easy_setopt(handle, CURLOPT_POST, 1L);
            curl_easy_setopt(handle, CURLOPT_POSTFIELDS, body.c_str());
            curl_easy_setopt(handle, CURLOPT_POSTFIELDSIZE,
                             static_cast<long>(body.size()));
        }

        // Headers
        struct curl_slist* header_list = nullptr;
        for (const auto& [key, value] : headers) {
            std::string header = key + ": " + value;
            header_list = curl_slist_append(header_list, header.c_str());
        }
        if (header_list) {
            curl_easy_setopt(handle, CURLOPT_HTTPHEADER, header_list);
        }

        // Execute
        const auto start = std::chrono::steady_clock::now();
        CURLcode res = curl_easy_perform(handle);
        const auto end = std::chrono::steady_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();

        // Clean up headers
        if (header_list) {
            curl_slist_free_all(header_list);
        }

        last_request_time_ = end;
        ++request_count_;

        HttpResponse response;
        response.elapsed_seconds = elapsed;

        if (res != CURLE_OK) {
            response.status_code = 0;
            response.error = curl_easy_strerror(res);
            last_response = std::move(response);

            if (is_retryable_(0, last_response.error)) {
                ++attempts;
                continue;
            }
            return last_response;
        }

        long http_code = 0;
        curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &http_code);

        response.status_code = static_cast<int>(http_code);
        response.body = std::move(response_body);
        last_response = std::move(response);

        if (is_retryable_(last_response.status_code, "") && attempts + 1 < max_attempts) {
            ++attempts;
            continue;
        }

        return last_response;
    }

    return last_response;
}

// ============================================================================
// rate_limit_
// ============================================================================

void HttpClient::rate_limit_() {
    if (config_.min_request_interval.count() <= 0) return;

    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - last_request_time_;

    if (elapsed < config_.min_request_interval) {
        std::this_thread::sleep_for(config_.min_request_interval - elapsed);
    }
}

// ============================================================================
// is_retryable_
// ============================================================================

bool HttpClient::is_retryable_(int status_code, const std::string& error) {
    // Network-level errors that are transient
    if (status_code == 0 && !error.empty()) {
        // Timeout or connection refused — retryable
        if (error.find("Timeout") != std::string::npos) return true;
        if (error.find("timed out") != std::string::npos) return true;
        if (error.find("Connection refused") != std::string::npos) return true;
        if (error.find("Could not resolve") != std::string::npos) return true;
        return false;
    }

    // HTTP status codes that are retryable
    if (status_code == 429) return true;  // Too Many Requests
    if (status_code == 502) return true;  // Bad Gateway
    if (status_code == 503) return true;  // Service Unavailable
    if (status_code == 504) return true;  // Gateway Timeout

    return false;
}

} // namespace nikola::infrastructure
