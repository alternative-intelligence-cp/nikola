#pragma once
/**
 * @file include/nikola/autonomy/firecrawl_oracle.hpp
 * @brief Phase 32 — FirecrawlOracle: web page scraping via the Firecrawl API.
 *
 * Implements the Oracle interface (oracle_pool.hpp) using Firecrawl's scrape
 * API to retrieve clean markdown content from web pages.
 *
 * Two modes:
 *   1. **Scrape mode** (scrape): given a URL, returns the page content as
 *      structured markdown.  Used by LookupFulfillmentAgent for deep reads.
 *   2. **Assess mode** (assess): extracts URLs from the provided content,
 *      scrapes them, and compares the claimed content against the actual page
 *      text for source verification.
 *
 * The FirecrawlOracle complements TavilyOracle: Tavily finds relevant pages
 * via search, Firecrawl reads them thoroughly.
 *
 * API docs: https://docs.firecrawl.dev/api-reference/endpoint/scrape
 *
 * Phase: NIK-FCR-01 (Firecrawl Oracle, Phase 32)
 */

#include <nikola/autonomy/oracle_pool.hpp>
#include <nikola/infrastructure/http_client.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace nikola::autonomy {

// ============================================================================
// FirecrawlResult — scraped page data
// ============================================================================

/**
 * @brief The scraped content of a single web page.
 */
struct FirecrawlResult {
    std::string url;          ///< Source URL (after redirects)
    std::string title;        ///< Page title from metadata
    std::string description;  ///< Page description from metadata
    std::string markdown;     ///< Cleaned markdown content
    int         status_code = 0;  ///< HTTP status code of the scraped page
};

// ============================================================================
// FirecrawlScrapeResponse — full API response
// ============================================================================

/**
 * @brief Parsed response from a Firecrawl scrape API call.
 */
struct FirecrawlScrapeResponse {
    bool             success = false;  ///< Whether the API call succeeded
    FirecrawlResult  result;           ///< Scraped page data
    std::string      error;            ///< Error message (empty on success)

    /// True if the scrape completed without errors.
    [[nodiscard]] bool ok() const noexcept { return success && error.empty(); }
};

// ============================================================================
// FirecrawlConfig
// ============================================================================

/**
 * @brief Configuration for FirecrawlOracle.
 */
struct FirecrawlConfig {
    /// Firecrawl API key.
    std::string api_key;

    /// API endpoint (override for testing with mock server).
    std::string endpoint = "https://api.firecrawl.dev/v2/scrape";

    /// Only return main content (exclude headers, navs, footers).
    bool only_main_content = true;

    /// Timeout for the Firecrawl scrape in milliseconds.
    int timeout_ms = 60000;

    /// HTTP client config (timeouts, retries, etc.).
    infrastructure::HttpClientConfig http_config;
};

// ============================================================================
// FirecrawlOracle
// ============================================================================

/**
 * @class FirecrawlOracle
 * @brief Oracle that scrapes web pages via the Firecrawl API.
 *
 * Implements Oracle::assess() for the OraclePool, and exposes scrape()
 * for direct use by the LookupFulfillmentAgent.
 *
 * Thread safety: NOT thread-safe.  Create one instance per thread.
 */
class FirecrawlOracle final : public Oracle {
public:
    /// Construct with API key and default configuration.
    explicit FirecrawlOracle(const std::string& api_key);

    /// Construct with full configuration.
    explicit FirecrawlOracle(const FirecrawlConfig& config);

    // ── Oracle interface ──────────────────────────────────────────────────────

    /**
     * @brief Assess content credibility by scraping referenced URLs.
     *
     * Extracts URLs from the content, scrapes up to 3 of them, and compares
     * the content against actual page text.  Higher overlap = higher
     * credibility (the content accurately reflects its sources).
     *
     * If no URLs are found in the content, returns neutral (0.5).
     *
     * @param query    The question that triggered the lookup.
     * @param content  The content to assess for credibility.
     * @return OracleVerdict with confidence in [0.0, 1.0].
     */
    OracleVerdict assess(const std::string& query,
                         const std::string& content) override;

    std::string name() const override { return "firecrawl"; }

    // ── Direct scrape API ─────────────────────────────────────────────────────

    /**
     * @brief Scrape a URL and return structured results.
     *
     * @param url The URL to scrape.
     * @return FirecrawlScrapeResponse with markdown content or error.
     */
    [[nodiscard]] FirecrawlScrapeResponse scrape(const std::string& url);

    /**
     * @brief Convenience: scrape and return just the markdown content.
     *
     * Returns empty string on error.
     */
    [[nodiscard]] std::string scrape_markdown(const std::string& url);

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Number of API calls made since construction.
    [[nodiscard]] uint64_t api_call_count() const noexcept { return api_calls_; }

    /// Access the configuration.
    [[nodiscard]] const FirecrawlConfig& config() const noexcept { return config_; }

    // ── Static JSON helpers (public for testing) ──────────────────────────────

    /// Build the scrape request JSON body.
    static std::string build_request_json(const std::string& url,
                                          bool only_main_content,
                                          int timeout_ms);

    /// Parse the Firecrawl scrape response JSON.
    static FirecrawlScrapeResponse parse_response_json(const std::string& json);

    /// Extract http/https URLs from a text string.
    static std::vector<std::string> extract_urls(const std::string& text);

private:
    /// Compute content similarity: fraction of content words found in reference.
    static float content_similarity_(const std::string& content,
                                     const std::string& reference);

    FirecrawlConfig                  config_;
    infrastructure::HttpClient       http_;
    uint64_t                         api_calls_ = 0;
};

// ============================================================================
// Credential loader — reads API key from a creds file
// ============================================================================

/**
 * @brief Load a Firecrawl API key from a credentials file.
 *
 * Reads the file and finds the first line starting with "fc-".
 *
 * @param path Path to the credentials file.
 * @return The API key string, or empty string on failure.
 */
[[nodiscard]] std::string load_firecrawl_api_key(const std::string& path);

} // namespace nikola::autonomy
