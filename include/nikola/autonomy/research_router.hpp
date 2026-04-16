#pragma once
/**
 * @file include/nikola/autonomy/research_router.hpp
 * @brief Phase 32 — ResearchRouter: smart query routing across data oracles.
 *
 * Routes incoming research queries to the most appropriate data source:
 *   - URL queries   → FirecrawlOracle (deep page scraping)
 *   - Factual/search queries → TavilyOracle (web search)
 *   - Fallback chain: if the primary source fails, try the secondary
 *
 * The router produces a LookupFn-compatible interface so it can be wired
 * directly into the LookupFulfillmentAgent:
 *
 *   ResearchRouter router(tavily, firecrawl);
 *   agent.set_lookup_fn([&](const std::string& q) { return router.route(q); });
 *
 * API docs: N/A (internal routing layer)
 *
 * Phase: NIK-RTR-01 (Smart Router, Phase 32)
 */

#include <nikola/autonomy/tavily_oracle.hpp>
#include <nikola/autonomy/firecrawl_oracle.hpp>
#include <nikola/infrastructure/http_client.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace nikola::autonomy {

// ============================================================================
// QueryType — classification of incoming queries
// ============================================================================

/**
 * @brief Classified intent of a research query.
 */
enum class QueryType {
    URL_READ,    ///< Query is or contains a URL → scrape with Firecrawl
    FACTUAL,     ///< General knowledge question → search with Tavily
    RAW_HTTP     ///< Raw HTTP endpoint fetch → direct HttpClient
};

// ============================================================================
// RouteResult — result from routing a query
// ============================================================================

/**
 * @brief The output of a routed query, including metadata about how it was resolved.
 */
struct RouteResult {
    std::string content;        ///< The resolved content text
    QueryType   query_type;     ///< How the query was classified
    std::string source;         ///< Which oracle/source produced the result
    bool        used_fallback = false;  ///< Whether the fallback chain was triggered
    std::string error;          ///< Error message if resolution failed

    /// True if content was successfully retrieved.
    [[nodiscard]] bool ok() const noexcept { return error.empty() && !content.empty(); }
};

// ============================================================================
// ResearchRouterConfig
// ============================================================================

/**
 * @brief Configuration for ResearchRouter.
 */
struct ResearchRouterConfig {
    /// Enable fallback: if primary oracle fails, try secondary.
    bool enable_fallback = true;

    /// Enable result aggregation: combine Tavily search + Firecrawl scrape
    /// of the top result for richer answers on factual queries.
    bool enable_aggregation = false;

    /// Maximum number of Firecrawl scrapes for aggregation mode.
    int max_aggregation_scrapes = 1;

    /// HTTP client config for raw HTTP mode.
    infrastructure::HttpClientConfig http_config;
};

// ============================================================================
// ResearchRouter
// ============================================================================

/**
 * @class ResearchRouter
 * @brief Smart router that dispatches queries to the appropriate data oracle.
 *
 * Query classification rules:
 *   1. If the query starts with http:// or https:// → URL_READ (Firecrawl)
 *   2. If the query contains "read ", "scrape ", "fetch " + URL → URL_READ
 *   3. Otherwise → FACTUAL (Tavily search)
 *
 * Fallback chain (when enabled):
 *   - URL_READ fails → try Tavily search for the URL's domain/topic
 *   - FACTUAL fails → try Firecrawl on a well-known search fallback
 *
 * Thread safety: NOT thread-safe.  Create one instance per thread.
 */
class ResearchRouter {
public:
    /**
     * @brief Construct with both oracle references.
     *
     * Both oracles must outlive the router.
     */
    ResearchRouter(TavilyOracle& tavily, FirecrawlOracle& firecrawl);

    /**
     * @brief Construct with both oracles and custom configuration.
     */
    ResearchRouter(TavilyOracle& tavily, FirecrawlOracle& firecrawl,
                   const ResearchRouterConfig& config);

    // ── Main routing API ──────────────────────────────────────────────────────

    /**
     * @brief Route a query to the best data source and return content.
     *
     * This is the primary entry point.  Returns just the content string,
     * compatible with LookupFn.
     *
     * @param query The research query.
     * @return Content string, or empty on complete failure.
     */
    [[nodiscard]] std::string route(const std::string& query);

    /**
     * @brief Route with full metadata about how the query was resolved.
     *
     * @param query The research query.
     * @return RouteResult with content, source, fallback info, etc.
     */
    [[nodiscard]] RouteResult route_detailed(const std::string& query);

    /**
     * @brief Create a LookupFn that can be wired into LookupFulfillmentAgent.
     *
     * The returned function captures a reference to this router, so the
     * router must outlive the returned function.
     *
     * @return A LookupFn-compatible std::function.
     */
    [[nodiscard]] std::function<std::string(const std::string&)> as_lookup_fn();

    // ── Query classification ──────────────────────────────────────────────────

    /**
     * @brief Classify a query string into a QueryType.
     *
     * Public and static for testability.
     *
     * @param query The query string to classify.
     * @return The classified QueryType.
     */
    [[nodiscard]] static QueryType classify(const std::string& query);

    /**
     * @brief Extract a URL from a query that contains one.
     *
     * Returns the first http/https URL found in the query, or empty string.
     * Public and static for testability.
     */
    [[nodiscard]] static std::string extract_url(const std::string& query);

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Number of queries routed since construction.
    [[nodiscard]] uint64_t route_count() const noexcept { return route_count_; }

    /// Number of queries routed to Tavily.
    [[nodiscard]] uint64_t tavily_count() const noexcept { return tavily_count_; }

    /// Number of queries routed to Firecrawl.
    [[nodiscard]] uint64_t firecrawl_count() const noexcept { return firecrawl_count_; }

    /// Number of fallback attempts.
    [[nodiscard]] uint64_t fallback_count() const noexcept { return fallback_count_; }

    /// Access the configuration.
    [[nodiscard]] const ResearchRouterConfig& config() const noexcept { return config_; }

private:
    /// Execute a Tavily search and return content.
    std::string try_tavily_(const std::string& query);

    /// Execute a Firecrawl scrape and return content.
    std::string try_firecrawl_(const std::string& url);

    /// Aggregate: search with Tavily, then scrape top result with Firecrawl.
    std::string try_aggregated_(const std::string& query);

    TavilyOracle&                    tavily_;
    FirecrawlOracle&                 firecrawl_;
    ResearchRouterConfig             config_;
    uint64_t                         route_count_     = 0;
    uint64_t                         tavily_count_    = 0;
    uint64_t                         firecrawl_count_ = 0;
    uint64_t                         fallback_count_  = 0;
};

} // namespace nikola::autonomy
