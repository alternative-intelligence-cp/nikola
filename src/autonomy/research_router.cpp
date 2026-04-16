/**
 * @file src/autonomy/research_router.cpp
 * @brief Phase 32 — ResearchRouter implementation.
 *
 * Query classification and routing logic:
 *   1. URL detection: starts with http(s):// or contains "read/scrape/fetch" + URL
 *   2. Factual queries: everything else → Tavily search
 *   3. Fallback chain: primary fails → try secondary oracle
 *   4. Aggregation: search + scrape top result for richer answers
 */

#include <nikola/autonomy/research_router.hpp>

#include <algorithm>
#include <cctype>

namespace nikola::autonomy {

// ============================================================================
// Construction
// ============================================================================

ResearchRouter::ResearchRouter(TavilyOracle& tavily, FirecrawlOracle& firecrawl)
    : tavily_(tavily)
    , firecrawl_(firecrawl)
{}

ResearchRouter::ResearchRouter(TavilyOracle& tavily, FirecrawlOracle& firecrawl,
                               const ResearchRouterConfig& config)
    : tavily_(tavily)
    , firecrawl_(firecrawl)
    , config_(config)
{}

// ============================================================================
// route() — simple string return (LookupFn-compatible)
// ============================================================================

std::string ResearchRouter::route(const std::string& query) {
    auto result = route_detailed(query);
    return result.content;
}

// ============================================================================
// route_detailed() — full metadata
// ============================================================================

RouteResult ResearchRouter::route_detailed(const std::string& query) {
    RouteResult result;
    result.query_type = classify(query);
    ++route_count_;

    if (query.empty()) {
        result.error = "empty query";
        return result;
    }

    switch (result.query_type) {
        case QueryType::URL_READ: {
            std::string url = extract_url(query);
            if (url.empty()) {
                result.error = "classified as URL_READ but no URL found";
                break;
            }

            // Primary: Firecrawl scrape
            result.content = try_firecrawl_(url);
            result.source = "firecrawl";
            ++firecrawl_count_;

            // Fallback: Tavily search if scrape failed
            if (result.content.empty() && config_.enable_fallback) {
                result.content = try_tavily_(query);
                if (!result.content.empty()) {
                    result.source = "tavily";
                    result.used_fallback = true;
                    ++tavily_count_;
                    ++fallback_count_;
                }
            }
            break;
        }

        case QueryType::FACTUAL: {
            // Aggregation mode: search + scrape top result
            if (config_.enable_aggregation) {
                result.content = try_aggregated_(query);
                result.source = "tavily+firecrawl";
                ++tavily_count_;
                ++firecrawl_count_;

                if (!result.content.empty()) {
                    break;
                }
                // If aggregation fails, fall through to simple search
            }

            // Primary: Tavily search
            result.content = try_tavily_(query);
            result.source = "tavily";
            ++tavily_count_;

            // Fallback: if Tavily fails entirely, we can't really Firecrawl
            // without a URL, so just report failure
            if (result.content.empty() && config_.enable_fallback) {
                // No meaningful fallback for factual queries without URLs
                result.error = "tavily search returned no results";
            }
            break;
        }

        case QueryType::RAW_HTTP: {
            std::string url = extract_url(query);
            if (url.empty()) {
                result.error = "classified as RAW_HTTP but no URL found";
                break;
            }

            // Use HttpClient directly for raw HTTP
            infrastructure::HttpClient http(config_.http_config);
            auto resp = http.get(url);
            if (resp.ok()) {
                result.content = resp.body;
                result.source = "http";
            } else {
                result.error = "HTTP GET failed: " + resp.error;

                // Fallback: try Firecrawl
                if (config_.enable_fallback) {
                    result.content = try_firecrawl_(url);
                    if (!result.content.empty()) {
                        result.source = "firecrawl";
                        result.used_fallback = true;
                        ++firecrawl_count_;
                        ++fallback_count_;
                    }
                }
            }
            break;
        }
    }

    if (result.content.empty() && result.error.empty()) {
        result.error = "no content retrieved";
    }

    return result;
}

// ============================================================================
// as_lookup_fn() — LookupFn adapter
// ============================================================================

std::function<std::string(const std::string&)> ResearchRouter::as_lookup_fn() {
    return [this](const std::string& query) -> std::string {
        return route(query);
    };
}

// ============================================================================
// classify() — static query classification
// ============================================================================

QueryType ResearchRouter::classify(const std::string& query) {
    if (query.empty()) return QueryType::FACTUAL;

    // Skip leading whitespace for classification
    auto start = query.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return QueryType::FACTUAL;

    std::string trimmed = query.substr(start);

    // Direct URL: starts with http:// or https://
    if (trimmed.rfind("https://", 0) == 0 || trimmed.rfind("http://", 0) == 0) {
        return QueryType::URL_READ;
    }

    // Convert first ~20 chars to lowercase for command detection
    std::string lower_prefix;
    lower_prefix.reserve(std::min(trimmed.size(), size_t{30}));
    for (size_t i = 0; i < std::min(trimmed.size(), size_t{30}); ++i) {
        lower_prefix += static_cast<char>(
            std::tolower(static_cast<unsigned char>(trimmed[i])));
    }

    // Command-style queries: "read <url>", "scrape <url>", "fetch <url>"
    // "get <url>", "open <url>", "visit <url>"
    static const char* const url_commands[] = {
        "read ", "scrape ", "fetch ", "get ", "open ", "visit ",
        "read: ", "scrape: ", "fetch: ", "get: ", "open: ", "visit: "
    };

    for (const char* cmd : url_commands) {
        if (lower_prefix.rfind(cmd, 0) == 0) {
            // Check if the rest contains a URL
            if (!extract_url(trimmed).empty()) {
                return QueryType::URL_READ;
            }
        }
    }

    // Check if query contains "http://" or "https://" anywhere
    // (e.g. "what does https://example.com say about X")
    if (trimmed.find("https://") != std::string::npos ||
        trimmed.find("http://") != std::string::npos) {
        return QueryType::URL_READ;
    }

    // Default: factual search
    return QueryType::FACTUAL;
}

// ============================================================================
// extract_url() — static URL extraction
// ============================================================================

std::string ResearchRouter::extract_url(const std::string& query) {
    // Find first http:// or https://
    auto pos = query.find("https://");
    if (pos == std::string::npos) {
        pos = query.find("http://");
    }
    if (pos == std::string::npos) {
        return "";
    }

    // Extract URL until whitespace or delimiter
    auto i = pos;
    while (i < query.size()) {
        char c = query[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
            c == '"' || c == '\'' || c == '>' || c == '<' ||
            c == ')' || c == ']' || c == '}') {
            break;
        }
        ++i;
    }

    std::string url = query.substr(pos, i - pos);

    // Strip trailing punctuation
    while (!url.empty() && (url.back() == '.' || url.back() == ',' ||
                             url.back() == ';' || url.back() == ':' ||
                             url.back() == '!')) {
        url.pop_back();
    }

    // Must have something meaningful after the scheme
    if (url.size() <= 8) return "";  // "https://" alone isn't valid

    return url;
}

// ============================================================================
// Private helpers
// ============================================================================

std::string ResearchRouter::try_tavily_(const std::string& query) {
    return tavily_.search_text(query);
}

std::string ResearchRouter::try_firecrawl_(const std::string& url) {
    return firecrawl_.scrape_markdown(url);
}

std::string ResearchRouter::try_aggregated_(const std::string& query) {
    // Step 1: Search with Tavily
    auto search_result = tavily_.search(query);
    if (!search_result.ok() || search_result.results.empty()) {
        return "";
    }

    // Step 2: Start with search text
    std::string aggregated;
    for (const auto& r : search_result.results) {
        aggregated += "## " + r.title + "\n";
        aggregated += r.url + "\n";
        aggregated += r.content + "\n\n";
    }

    // Step 3: Scrape top N results with Firecrawl for deeper content
    int scrapes = 0;
    for (const auto& r : search_result.results) {
        if (scrapes >= config_.max_aggregation_scrapes) break;
        if (r.url.empty()) continue;

        auto scraped = firecrawl_.scrape_markdown(r.url);
        if (!scraped.empty()) {
            aggregated += "---\n## Deep Read: " + r.title + "\n";
            aggregated += scraped + "\n\n";
            ++scrapes;
        }
    }

    return aggregated;
}

} // namespace nikola::autonomy
