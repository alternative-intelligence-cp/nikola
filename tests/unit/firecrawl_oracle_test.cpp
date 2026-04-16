/**
 * @file tests/unit/firecrawl_oracle_test.cpp
 * @brief Phase 32 — FirecrawlOracle unit tests (Catch2 v3).
 *
 * Offline tests validate JSON building/parsing, URL extraction, credential
 * loading, and construction.
 *
 * Live network tests are guarded by the [network] tag — run them explicitly
 * with: ./test_firecrawl_oracle "[network]"
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/autonomy/firecrawl_oracle.hpp>
#include <nikola/autonomy/oracle_pool.hpp>

#include <cstdio>
#include <fstream>
#include <string>

using namespace nikola::autonomy;

// ─────────────────────────────────────────────────────────────────────────────
//  Construction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("FirecrawlOracle — construction with API key", "[firecrawl][unit]") {
    FirecrawlOracle oracle("fc-test-key-123");
    CHECK(oracle.name() == "firecrawl");
    CHECK(oracle.api_call_count() == 0);
    CHECK(oracle.config().api_key == "fc-test-key-123");
    CHECK(oracle.config().only_main_content == true);
    CHECK(oracle.config().timeout_ms == 60000);
    CHECK(oracle.config().endpoint == "https://api.firecrawl.dev/v2/scrape");
}

TEST_CASE("FirecrawlOracle — construction with full config", "[firecrawl][unit]") {
    FirecrawlConfig cfg;
    cfg.api_key = "fc-custom-key";
    cfg.only_main_content = false;
    cfg.timeout_ms = 30000;
    cfg.endpoint = "https://custom.endpoint/scrape";

    FirecrawlOracle oracle(cfg);
    CHECK(oracle.config().only_main_content == false);
    CHECK(oracle.config().timeout_ms == 30000);
    CHECK(oracle.config().endpoint == "https://custom.endpoint/scrape");
}

TEST_CASE("FirecrawlOracle — empty API key throws", "[firecrawl][unit]") {
    CHECK_THROWS_AS(FirecrawlOracle(""), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
//  JSON building — build_request_json()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("FirecrawlOracle — build_request_json basic", "[firecrawl][unit]") {
    auto json = FirecrawlOracle::build_request_json(
        "https://example.com", true, 60000);

    CHECK(json.find("\"url\":\"https://example.com\"") != std::string::npos);
    CHECK(json.find("\"formats\":[\"markdown\"]") != std::string::npos);
    CHECK(json.find("\"onlyMainContent\":true") != std::string::npos);
    CHECK(json.find("\"timeout\":60000") != std::string::npos);
}

TEST_CASE("FirecrawlOracle — build_request_json with options", "[firecrawl][unit]") {
    auto json = FirecrawlOracle::build_request_json(
        "https://example.com/path?q=test&x=1", false, 30000);

    CHECK(json.find("\"onlyMainContent\":false") != std::string::npos);
    CHECK(json.find("\"timeout\":30000") != std::string::npos);
    CHECK(json.find("q=test") != std::string::npos);
}

TEST_CASE("FirecrawlOracle — build_request_json escapes special chars", "[firecrawl][unit]") {
    auto json = FirecrawlOracle::build_request_json(
        "https://example.com/page?title=\"hello\"", true, 60000);

    CHECK(json.find("\\\"hello\\\"") != std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────────────
//  JSON parsing — parse_response_json()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("FirecrawlOracle — parse valid response JSON", "[firecrawl][unit]") {
    const std::string mock_json = R"({
        "success": true,
        "data": {
            "markdown": "# Example Domain\n\nThis is example content.",
            "metadata": {
                "title": "Example Domain",
                "description": "An example website",
                "sourceURL": "https://example.com",
                "url": "https://example.com/",
                "statusCode": 200
            }
        }
    })";

    auto resp = FirecrawlOracle::parse_response_json(mock_json);
    CHECK(resp.ok());
    CHECK(resp.success == true);
    CHECK(resp.result.markdown == "# Example Domain\n\nThis is example content.");
    CHECK(resp.result.title == "Example Domain");
    CHECK(resp.result.description == "An example website");
    CHECK(resp.result.url == "https://example.com");
    CHECK(resp.result.status_code == 200);
}

TEST_CASE("FirecrawlOracle — parse empty response", "[firecrawl][unit]") {
    auto resp = FirecrawlOracle::parse_response_json("");
    CHECK_FALSE(resp.ok());
    CHECK(resp.error == "empty response");
}

TEST_CASE("FirecrawlOracle — parse error response", "[firecrawl][unit]") {
    const std::string error_json = R"({"success":false,"error":"Invalid API key"})";
    auto resp = FirecrawlOracle::parse_response_json(error_json);
    CHECK_FALSE(resp.ok());
    CHECK(resp.error == "Invalid API key");
}

TEST_CASE("FirecrawlOracle — parse success false without error", "[firecrawl][unit]") {
    const std::string json = R"({"success":false})";
    auto resp = FirecrawlOracle::parse_response_json(json);
    CHECK_FALSE(resp.ok());
    CHECK(resp.error == "API returned success=false");
}

TEST_CASE("FirecrawlOracle — parse response with no metadata", "[firecrawl][unit]") {
    const std::string json = R"({
        "success": true,
        "data": {
            "markdown": "# Content only"
        }
    })";

    auto resp = FirecrawlOracle::parse_response_json(json);
    CHECK(resp.ok());
    CHECK(resp.result.markdown == "# Content only");
    CHECK(resp.result.title.empty());
    CHECK(resp.result.status_code == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  URL extraction — extract_urls()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("FirecrawlOracle — extract_urls finds http/https URLs", "[firecrawl][unit]") {
    std::string text = "Check out https://example.com and http://test.org/page for info.";
    auto urls = FirecrawlOracle::extract_urls(text);
    REQUIRE(urls.size() == 2);
    CHECK(urls[0] == "https://example.com");
    CHECK(urls[1] == "http://test.org/page");
}

TEST_CASE("FirecrawlOracle — extract_urls handles URLs in markdown", "[firecrawl][unit]") {
    std::string text = "See [link](https://example.com/path) and <https://other.com>.";
    auto urls = FirecrawlOracle::extract_urls(text);
    REQUIRE(urls.size() == 2);
    CHECK(urls[0] == "https://example.com/path");
    CHECK(urls[1] == "https://other.com");
}

TEST_CASE("FirecrawlOracle — extract_urls strips trailing punctuation", "[firecrawl][unit]") {
    std::string text = "Visit https://example.com/page. Also: https://other.com,";
    auto urls = FirecrawlOracle::extract_urls(text);
    REQUIRE(urls.size() == 2);
    CHECK(urls[0] == "https://example.com/page");
    CHECK(urls[1] == "https://other.com");
}

TEST_CASE("FirecrawlOracle — extract_urls deduplicates", "[firecrawl][unit]") {
    std::string text = "https://example.com https://example.com https://example.com";
    auto urls = FirecrawlOracle::extract_urls(text);
    CHECK(urls.size() == 1);
}

TEST_CASE("FirecrawlOracle — extract_urls returns empty for no URLs", "[firecrawl][unit]") {
    std::string text = "No URLs here, just plain text with some numbers 12345.";
    auto urls = FirecrawlOracle::extract_urls(text);
    CHECK(urls.empty());
}

TEST_CASE("FirecrawlOracle — extract_urls ignores partial http", "[firecrawl][unit]") {
    std::string text = "The http protocol was created. Also httponly cookies exist.";
    auto urls = FirecrawlOracle::extract_urls(text);
    CHECK(urls.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
//  Credential loader — load_firecrawl_api_key()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("FirecrawlOracle — load_firecrawl_api_key from file", "[firecrawl][unit]") {
    const std::string tmp_path = "/tmp/nikola_test_firecrawl_creds.txt";
    {
        std::ofstream f(tmp_path);
        f << "#API key:\n";
        f << "fc-test123abc\n";
        f << "\n";
        f << "#MCP Config\n";
    }

    auto key = load_firecrawl_api_key(tmp_path);
    CHECK(key == "fc-test123abc");

    std::remove(tmp_path.c_str());
}

TEST_CASE("FirecrawlOracle — load_firecrawl_api_key handles whitespace", "[firecrawl][unit]") {
    const std::string tmp_path = "/tmp/nikola_test_firecrawl_ws.txt";
    {
        std::ofstream f(tmp_path);
        f << "  \n";
        f << "  fc-with-spaces  \n";
    }

    auto key = load_firecrawl_api_key(tmp_path);
    CHECK(key == "fc-with-spaces");

    std::remove(tmp_path.c_str());
}

TEST_CASE("FirecrawlOracle — load_firecrawl_api_key returns empty for missing file", "[firecrawl][unit]") {
    auto key = load_firecrawl_api_key("/tmp/nonexistent_firecrawl_xyz.txt");
    CHECK(key.empty());
}

TEST_CASE("FirecrawlOracle — load_firecrawl_api_key returns empty for no key", "[firecrawl][unit]") {
    const std::string tmp_path = "/tmp/nikola_test_firecrawl_nokey.txt";
    {
        std::ofstream f(tmp_path);
        f << "some random content\n";
        f << "no api key here\n";
    }

    auto key = load_firecrawl_api_key(tmp_path);
    CHECK(key.empty());

    std::remove(tmp_path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
//  OraclePool integration
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("FirecrawlOracle — can be added to OraclePool", "[firecrawl][unit]") {
    OraclePool pool;
    pool.add_oracle(std::make_shared<FirecrawlOracle>("fc-pool-test-key"));
    CHECK(pool.size() == 1);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Live network tests — requires valid Firecrawl API key
// ─────────────────────────────────────────────────────────────────────────────

static std::string get_test_api_key() {
    auto key = load_firecrawl_api_key(
        "/home/randy/Workspace/CREDS/creds/firecrawl.creds");
    if (key.empty()) {
        SKIP("Firecrawl API key not found");
    }
    return key;
}

TEST_CASE("FirecrawlOracle — live scrape returns markdown", "[firecrawl][network]") {
    auto key = get_test_api_key();
    FirecrawlOracle oracle(key);

    auto response = oracle.scrape("https://example.com");
    REQUIRE(response.ok());
    CHECK_FALSE(response.result.markdown.empty());
    CHECK(response.result.markdown.find("Example Domain") != std::string::npos);
    CHECK(response.result.status_code == 200);
    CHECK(oracle.api_call_count() == 1);
}

TEST_CASE("FirecrawlOracle — live scrape_markdown convenience", "[firecrawl][network]") {
    auto key = get_test_api_key();
    FirecrawlOracle oracle(key);

    auto md = oracle.scrape_markdown("https://example.com");
    CHECK_FALSE(md.empty());
    CHECK(md.find("Example") != std::string::npos);
}

TEST_CASE("FirecrawlOracle — live scrape returns metadata", "[firecrawl][network]") {
    auto key = get_test_api_key();
    FirecrawlOracle oracle(key);

    auto response = oracle.scrape("https://example.com");
    REQUIRE(response.ok());
    CHECK_FALSE(response.result.title.empty());
    CHECK_FALSE(response.result.url.empty());
}

TEST_CASE("FirecrawlOracle — live assess with URLs in content", "[firecrawl][network]") {
    auto key = get_test_api_key();
    FirecrawlOracle oracle(key);

    auto verdict = oracle.assess(
        "What is example.com?",
        "Example Domain is a website at https://example.com used for "
        "documentation examples. You can use it without permission.");

    CHECK(verdict.confidence >= 0.0f);
    CHECK(verdict.confidence <= 1.0f);
    CHECK(verdict.confidence > 0.2f);
    CHECK_FALSE(verdict.rationale.empty());
}

TEST_CASE("FirecrawlOracle — assess with no URLs returns neutral", "[firecrawl][unit]") {
    FirecrawlOracle oracle("fc-test-key");

    auto verdict = oracle.assess("test query", "content with no URLs at all");
    CHECK(verdict.confidence == 0.5f);
    CHECK(verdict.rationale.find("no URLs") != std::string::npos);
}

TEST_CASE("FirecrawlOracle — assess empty content returns zero", "[firecrawl][unit]") {
    FirecrawlOracle oracle("fc-test-key");

    auto verdict = oracle.assess("test query", "");
    CHECK(verdict.confidence == 0.0f);
}
