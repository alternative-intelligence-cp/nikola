/**
 * @file tests/unit/research_router_test.cpp
 * @brief Phase 32 — ResearchRouter unit tests (Catch2 v3).
 *
 * Offline tests validate query classification, URL extraction, routing logic,
 * fallback chains, and LookupFn integration.
 *
 * Live network tests are guarded by the [network] tag — run them explicitly
 * with: ./test_research_router "[network]"
 */

#include <catch2/catch_test_macros.hpp>

#include <nikola/autonomy/research_router.hpp>
#include <nikola/autonomy/tavily_oracle.hpp>
#include <nikola/autonomy/firecrawl_oracle.hpp>

#include <string>

using namespace nikola::autonomy;

// ─────────────────────────────────────────────────────────────────────────────
//  Query Classification — classify()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ResearchRouter — classify direct HTTPS URL", "[router][unit]") {
    CHECK(ResearchRouter::classify("https://example.com") == QueryType::URL_READ);
    CHECK(ResearchRouter::classify("https://en.wikipedia.org/wiki/C++") == QueryType::URL_READ);
}

TEST_CASE("ResearchRouter — classify direct HTTP URL", "[router][unit]") {
    CHECK(ResearchRouter::classify("http://example.com/path") == QueryType::URL_READ);
}

TEST_CASE("ResearchRouter — classify URL with leading whitespace", "[router][unit]") {
    CHECK(ResearchRouter::classify("  https://example.com") == QueryType::URL_READ);
    CHECK(ResearchRouter::classify("\thttps://example.com") == QueryType::URL_READ);
}

TEST_CASE("ResearchRouter — classify command + URL", "[router][unit]") {
    CHECK(ResearchRouter::classify("read https://example.com") == QueryType::URL_READ);
    CHECK(ResearchRouter::classify("scrape https://example.com/page") == QueryType::URL_READ);
    CHECK(ResearchRouter::classify("fetch https://api.example.com/data") == QueryType::URL_READ);
    CHECK(ResearchRouter::classify("get https://example.com") == QueryType::URL_READ);
    CHECK(ResearchRouter::classify("open https://example.com") == QueryType::URL_READ);
    CHECK(ResearchRouter::classify("visit https://example.com") == QueryType::URL_READ);
}

TEST_CASE("ResearchRouter — classify embedded URL in question", "[router][unit]") {
    CHECK(ResearchRouter::classify(
        "what does https://example.com say about testing?") == QueryType::URL_READ);
}

TEST_CASE("ResearchRouter — classify factual queries", "[router][unit]") {
    CHECK(ResearchRouter::classify("what is C++") == QueryType::FACTUAL);
    CHECK(ResearchRouter::classify("how does the Linux kernel work") == QueryType::FACTUAL);
    CHECK(ResearchRouter::classify("explain quantum computing") == QueryType::FACTUAL);
    CHECK(ResearchRouter::classify("capital of France") == QueryType::FACTUAL);
}

TEST_CASE("ResearchRouter — classify empty query as FACTUAL", "[router][unit]") {
    CHECK(ResearchRouter::classify("") == QueryType::FACTUAL);
    CHECK(ResearchRouter::classify("   ") == QueryType::FACTUAL);
}

TEST_CASE("ResearchRouter — classify does not trigger on 'http' without '://'", "[router][unit]") {
    CHECK(ResearchRouter::classify("http protocol history") == QueryType::FACTUAL);
    CHECK(ResearchRouter::classify("httponly cookies") == QueryType::FACTUAL);
}

// ─────────────────────────────────────────────────────────────────────────────
//  URL Extraction — extract_url()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ResearchRouter — extract_url from direct URL", "[router][unit]") {
    CHECK(ResearchRouter::extract_url("https://example.com") == "https://example.com");
    CHECK(ResearchRouter::extract_url("http://test.org/page") == "http://test.org/page");
}

TEST_CASE("ResearchRouter — extract_url from command + URL", "[router][unit]") {
    CHECK(ResearchRouter::extract_url("read https://example.com/docs") == "https://example.com/docs");
}

TEST_CASE("ResearchRouter — extract_url from sentence with URL", "[router][unit]") {
    auto url = ResearchRouter::extract_url("what does https://example.com say?");
    CHECK(url == "https://example.com");
}

TEST_CASE("ResearchRouter — extract_url strips trailing punctuation", "[router][unit]") {
    CHECK(ResearchRouter::extract_url("see https://example.com.") == "https://example.com");
    CHECK(ResearchRouter::extract_url("link: https://example.com,") == "https://example.com");
}

TEST_CASE("ResearchRouter — extract_url returns empty for no URL", "[router][unit]") {
    CHECK(ResearchRouter::extract_url("no url here").empty());
    CHECK(ResearchRouter::extract_url("").empty());
}

TEST_CASE("ResearchRouter — extract_url rejects bare scheme", "[router][unit]") {
    CHECK(ResearchRouter::extract_url("https://").empty());
}

// ─────────────────────────────────────────────────────────────────────────────
//  Construction & Configuration
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ResearchRouter — construction with defaults", "[router][unit]") {
    TavilyOracle tavily("tvly-test-key");
    FirecrawlOracle firecrawl("fc-test-key");
    ResearchRouter router(tavily, firecrawl);

    CHECK(router.route_count() == 0);
    CHECK(router.tavily_count() == 0);
    CHECK(router.firecrawl_count() == 0);
    CHECK(router.fallback_count() == 0);
    CHECK(router.config().enable_fallback == true);
    CHECK(router.config().enable_aggregation == false);
}

TEST_CASE("ResearchRouter — construction with custom config", "[router][unit]") {
    TavilyOracle tavily("tvly-test-key");
    FirecrawlOracle firecrawl("fc-test-key");

    ResearchRouterConfig cfg;
    cfg.enable_fallback = false;
    cfg.enable_aggregation = true;
    cfg.max_aggregation_scrapes = 3;

    ResearchRouter router(tavily, firecrawl, cfg);
    CHECK(router.config().enable_fallback == false);
    CHECK(router.config().enable_aggregation == true);
    CHECK(router.config().max_aggregation_scrapes == 3);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Routing — empty query
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ResearchRouter — route empty query returns error", "[router][unit]") {
    TavilyOracle tavily("tvly-test-key");
    FirecrawlOracle firecrawl("fc-test-key");
    ResearchRouter router(tavily, firecrawl);

    auto result = router.route_detailed("");
    CHECK_FALSE(result.ok());
    CHECK(result.error == "empty query");
    CHECK(router.route_count() == 1);
}

// ─────────────────────────────────────────────────────────────────────────────
//  as_lookup_fn()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ResearchRouter — as_lookup_fn returns callable", "[router][unit]") {
    TavilyOracle tavily("tvly-test-key");
    FirecrawlOracle firecrawl("fc-test-key");
    ResearchRouter router(tavily, firecrawl);

    auto fn = router.as_lookup_fn();
    // Calling with empty string exercises the route path (will return empty)
    auto result = fn("");
    CHECK(result.empty());
    CHECK(router.route_count() == 1);
}

// ─────────────────────────────────────────────────────────────────────────────
//  RouteResult ok() semantics
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("RouteResult — ok() semantics", "[router][unit]") {
    RouteResult r1;
    r1.content = "some content";
    CHECK(r1.ok());

    RouteResult r2;
    r2.content = "";
    r2.error = "failed";
    CHECK_FALSE(r2.ok());

    RouteResult r3;
    r3.content = "";
    CHECK_FALSE(r3.ok());  // empty content even without error

    RouteResult r4;
    r4.content = "data";
    r4.error = "partial failure";
    CHECK_FALSE(r4.ok());  // has error even with content
}

// ─────────────────────────────────────────────────────────────────────────────
//  Live network tests
// ─────────────────────────────────────────────────────────────────────────────

static std::pair<std::string, std::string> get_test_keys() {
    auto tavily_key = load_tavily_api_key(
        "/home/randy/Workspace/CREDS/creds/tavily.creds");
    auto firecrawl_key = load_firecrawl_api_key(
        "/home/randy/Workspace/CREDS/creds/firecrawl.creds");
    if (tavily_key.empty() || firecrawl_key.empty()) {
        SKIP("API keys not found");
    }
    return {tavily_key, firecrawl_key};
}

TEST_CASE("ResearchRouter — live route factual query to Tavily", "[router][network]") {
    auto [tkey, fkey] = get_test_keys();
    TavilyOracle tavily(tkey);
    FirecrawlOracle firecrawl(fkey);
    ResearchRouter router(tavily, firecrawl);

    auto result = router.route_detailed("What is the C++ programming language?");
    CHECK(result.ok());
    CHECK(result.query_type == QueryType::FACTUAL);
    CHECK(result.source == "tavily");
    CHECK_FALSE(result.content.empty());
    CHECK(router.tavily_count() >= 1);
}

TEST_CASE("ResearchRouter — live route URL to Firecrawl", "[router][network]") {
    auto [tkey, fkey] = get_test_keys();
    TavilyOracle tavily(tkey);
    FirecrawlOracle firecrawl(fkey);
    ResearchRouter router(tavily, firecrawl);

    auto result = router.route_detailed("https://example.com");
    CHECK(result.ok());
    CHECK(result.query_type == QueryType::URL_READ);
    CHECK(result.source == "firecrawl");
    CHECK(result.content.find("Example") != std::string::npos);
    CHECK(router.firecrawl_count() >= 1);
}

TEST_CASE("ResearchRouter — live route command + URL", "[router][network]") {
    auto [tkey, fkey] = get_test_keys();
    TavilyOracle tavily(tkey);
    FirecrawlOracle firecrawl(fkey);
    ResearchRouter router(tavily, firecrawl);

    auto result = router.route_detailed("read https://example.com");
    CHECK(result.ok());
    CHECK(result.query_type == QueryType::URL_READ);
    CHECK(result.source == "firecrawl");
}

TEST_CASE("ResearchRouter — live as_lookup_fn integration", "[router][network]") {
    auto [tkey, fkey] = get_test_keys();
    TavilyOracle tavily(tkey);
    FirecrawlOracle firecrawl(fkey);
    ResearchRouter router(tavily, firecrawl);

    auto fn = router.as_lookup_fn();
    auto content = fn("What is Linux?");
    CHECK_FALSE(content.empty());
    CHECK(router.route_count() == 1);
}

TEST_CASE("ResearchRouter — live aggregation mode", "[router][network]") {
    auto [tkey, fkey] = get_test_keys();
    TavilyOracle tavily(tkey);
    FirecrawlOracle firecrawl(fkey);

    ResearchRouterConfig cfg;
    cfg.enable_aggregation = true;
    cfg.max_aggregation_scrapes = 1;
    ResearchRouter router(tavily, firecrawl, cfg);

    auto result = router.route_detailed("What is example.com?");
    CHECK(result.ok());
    // Aggregation combines search + deep read
    CHECK(result.content.find("##") != std::string::npos);
}

TEST_CASE("ResearchRouter — counters increment correctly", "[router][network]") {
    auto [tkey, fkey] = get_test_keys();
    TavilyOracle tavily(tkey);
    FirecrawlOracle firecrawl(fkey);
    ResearchRouter router(tavily, firecrawl);

    (void)router.route("What is C++?");
    (void)router.route("https://example.com");

    CHECK(router.route_count() == 2);
    CHECK(router.tavily_count() >= 1);
    CHECK(router.firecrawl_count() >= 1);
}
