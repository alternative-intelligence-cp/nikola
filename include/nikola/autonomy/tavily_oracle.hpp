#pragma once
/**
 * @file include/nikola/autonomy/tavily_oracle.hpp
 * @brief Phase 32 — TavilyOracle: web search via the Tavily Search API.
 *
 * Implements the Oracle interface (oracle_pool.hpp) using Tavily's search API
 * to retrieve web search results and assess content credibility.
 *
 * Two modes:
 *   1. **Search mode** (search): given a query, returns aggregated web results
 *      as a single content string.  Used by LookupFulfillmentAgent.
 *   2. **Assess mode** (assess): given a query+content pair, searches for the
 *      query and compares the content against web results for credibility.
 *
 * The TavilyOracle also exposes a standalone search() method for direct use
 * outside the OraclePool, returning structured TavilyResult objects.
 *
 * API docs: https://docs.tavily.com/documentation/api-reference/search
 *
 * Phase: NIK-TAV-01 (Tavily Oracle, Phase 32)
 */

#include <nikola/autonomy/oracle_pool.hpp>
#include <nikola/infrastructure/http_client.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace nikola::autonomy {

// ============================================================================
// TavilyResult — one search result
// ============================================================================

/**
 * @brief A single result from a Tavily search.
 */
struct TavilyResult {
    std::string url;       ///< Source URL
    std::string title;     ///< Page title
    std::string content;   ///< Extracted text snippet
    float       score = 0; ///< Relevance score [0, 1] from Tavily
};

// ============================================================================
// TavilySearchResponse — full API response
// ============================================================================

/**
 * @brief Parsed response from a Tavily search API call.
 */
struct TavilySearchResponse {
    std::string                query;          ///< Echo of the query sent
    std::vector<TavilyResult>  results;        ///< Search results
    double                     response_time;  ///< API response time in seconds
    std::string                error;          ///< Error message (empty on success)

    /// True if the search completed without errors.
    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

// ============================================================================
// TavilyConfig
// ============================================================================

/**
 * @brief Configuration for TavilyOracle.
 */
struct TavilyConfig {
    /// Tavily API key.
    std::string api_key;

    /// API endpoint (override for testing with mock server).
    std::string endpoint = "https://api.tavily.com/search";

    /// Maximum number of search results to return.
    int max_results = 5;

    /// Search depth: "basic" (fast) or "advanced" (slower, more thorough).
    std::string search_depth = "basic";

    /// HTTP client config (timeouts, retries, etc.).
    infrastructure::HttpClientConfig http_config;
};

// ============================================================================
// TavilyOracle
// ============================================================================

/**
 * @class TavilyOracle
 * @brief Oracle that queries the Tavily Search API for web search results.
 *
 * Implements Oracle::assess() for the OraclePool, and exposes search()
 * for direct use by the LookupFulfillmentAgent.
 *
 * Thread safety: NOT thread-safe.  Create one instance per thread.
 */
class TavilyOracle final : public Oracle {
public:
    /// Construct with API key and optional configuration.
    explicit TavilyOracle(const std::string& api_key);

    /// Construct with full configuration.
    explicit TavilyOracle(const TavilyConfig& config);

    // ── Oracle interface ──────────────────────────────────────────────────────

    /**
     * @brief Assess content credibility by comparing against web search results.
     *
     * Searches Tavily for the query, then computes a similarity score between
     * the provided content and the search results.  Higher similarity = higher
     * credibility (the content agrees with web sources).
     *
     * @param query    The question that triggered the lookup.
     * @param content  The content to assess for credibility.
     * @return OracleVerdict with confidence in [0.0, 1.0].
     */
    OracleVerdict assess(const std::string& query,
                         const std::string& content) override;

    std::string name() const override { return "tavily"; }

    // ── Direct search API ─────────────────────────────────────────────────────

    /**
     * @brief Perform a Tavily web search and return structured results.
     *
     * @param query The search query string.
     * @return TavilySearchResponse with results or error.
     */
    [[nodiscard]] TavilySearchResponse search(const std::string& query);

    /**
     * @brief Convenience: search and return results as a single text string.
     *
     * Format: each result is "## Title\nURL\nContent\n\n".
     * Returns empty string on error or no results.
     */
    [[nodiscard]] std::string search_text(const std::string& query);

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Number of API calls made since construction.
    [[nodiscard]] uint64_t api_call_count() const noexcept { return api_calls_; }

    /// Access the configuration.
    [[nodiscard]] const TavilyConfig& config() const noexcept { return config_; }

    // ── Static JSON helpers (public for testing) ──────────────────────────────

    /// Build the search request JSON body.
    static std::string build_request_json(const std::string& api_key,
                                          const std::string& query,
                                          int max_results,
                                          const std::string& search_depth);

    /// Parse the Tavily search response JSON.
    static TavilySearchResponse parse_response_json(const std::string& json);

private:
    /// Compute content similarity: fraction of query words found in text.
    static float content_similarity_(const std::string& content,
                                     const std::string& reference);

    TavilyConfig                     config_;
    infrastructure::HttpClient       http_;
    uint64_t                         api_calls_ = 0;
};

// ============================================================================
// Credential loader — reads API key from a creds file
// ============================================================================

/**
 * @brief Load a Tavily API key from a credentials file.
 *
 * Expected file format (line 2 contains the key):
 *   api-key:
 *   tvly-...
 *
 * @param path Path to the credentials file.
 * @return The API key string, or empty string on failure.
 */
[[nodiscard]] std::string load_tavily_api_key(const std::string& path);

} // namespace nikola::autonomy
