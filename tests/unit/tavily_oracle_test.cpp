/**
 * @file tests/unit/tavily_oracle_test.cpp
 * @brief Phase 32 — TavilyOracle unit tests (Catch2 v3).
 *
 * Offline tests validate JSON building/parsing, credential loading,
 * construction, and content similarity.
 *
 * Live network tests are guarded by the [network] tag — run them explicitly
 * with: ./test_tavily_oracle "[network]"
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/autonomy/tavily_oracle.hpp>
#include <nikola/autonomy/oracle_pool.hpp>

#include <cstdio>
#include <fstream>
#include <string>

using namespace nikola::autonomy;

// ─────────────────────────────────────────────────────────────────────────────
//  Construction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("TavilyOracle — construction with API key", "[tavily][unit]") {
    TavilyOracle oracle("tvly-test-key-123");
    CHECK(oracle.name() == "tavily");
    CHECK(oracle.api_call_count() == 0);
    CHECK(oracle.config().api_key == "tvly-test-key-123");
    CHECK(oracle.config().max_results == 5);
    CHECK(oracle.config().search_depth == "basic");
    CHECK(oracle.config().endpoint == "https://api.tavily.com/search");
}

TEST_CASE("TavilyOracle — construction with full config", "[tavily][unit]") {
    TavilyConfig cfg;
    cfg.api_key = "tvly-custom-key";
    cfg.max_results = 10;
    cfg.search_depth = "advanced";
    cfg.endpoint = "https://custom.endpoint/search";

    TavilyOracle oracle(cfg);
    CHECK(oracle.config().max_results == 10);
    CHECK(oracle.config().search_depth == "advanced");
    CHECK(oracle.config().endpoint == "https://custom.endpoint/search");
}

TEST_CASE("TavilyOracle — empty API key throws", "[tavily][unit]") {
    CHECK_THROWS_AS(TavilyOracle(""), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
//  JSON building — build_request_json()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("TavilyOracle — build_request_json basic", "[tavily][unit]") {
    auto json = TavilyOracle::build_request_json(
        "tvly-key-abc", "what is C++", 5, "basic");

    CHECK(json.find("\"api_key\":\"tvly-key-abc\"") != std::string::npos);
    CHECK(json.find("\"query\":\"what is C++\"") != std::string::npos);
    CHECK(json.find("\"max_results\":5") != std::string::npos);
    CHECK(json.find("\"search_depth\":\"basic\"") != std::string::npos);
}

TEST_CASE("TavilyOracle — build_request_json escapes special chars", "[tavily][unit]") {
    auto json = TavilyOracle::build_request_json(
        "tvly-key", "query with \"quotes\" and\\backslash", 3, "advanced");

    // The query should be JSON-escaped
    CHECK(json.find("\\\"quotes\\\"") != std::string::npos);
    CHECK(json.find("\\\\backslash") != std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────────────
//  JSON parsing — parse_response_json()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("TavilyOracle — parse valid response JSON", "[tavily][unit]") {
    const std::string mock_json = R"({
        "query": "test query",
        "response_time": 0.42,
        "results": [
            {
                "url": "https://example.com/page1",
                "title": "Example Page",
                "content": "This is the first result content.",
                "score": 0.95
            },
            {
                "url": "https://example.com/page2",
                "title": "Another Page",
                "content": "Second result with different content.",
                "score": 0.87
            }
        ]
    })";

    auto resp = TavilyOracle::parse_response_json(mock_json);
    CHECK(resp.ok());
    CHECK(resp.query == "test query");
    CHECK(resp.response_time == Catch::Approx(0.42).margin(0.01));
    REQUIRE(resp.results.size() == 2);

    CHECK(resp.results[0].url == "https://example.com/page1");
    CHECK(resp.results[0].title == "Example Page");
    CHECK(resp.results[0].content == "This is the first result content.");
    CHECK(resp.results[0].score == Catch::Approx(0.95f).margin(0.01f));

    CHECK(resp.results[1].url == "https://example.com/page2");
    CHECK(resp.results[1].title == "Another Page");
    CHECK(resp.results[1].score == Catch::Approx(0.87f).margin(0.01f));
}

TEST_CASE("TavilyOracle — parse empty response", "[tavily][unit]") {
    auto resp = TavilyOracle::parse_response_json("");
    CHECK_FALSE(resp.ok());
    CHECK(resp.error == "empty response");
}

TEST_CASE("TavilyOracle — parse error response", "[tavily][unit]") {
    const std::string error_json = R"({"detail":"Invalid API key"})";
    auto resp = TavilyOracle::parse_response_json(error_json);
    CHECK_FALSE(resp.ok());
    CHECK(resp.error == "Invalid API key");
}

TEST_CASE("TavilyOracle — parse skips results with empty content", "[tavily][unit]") {
    const std::string json = R"({
        "query": "skip test",
        "response_time": 0.1,
        "results": [
            {"url": "https://a.com", "title": "A", "content": "", "score": 0.9},
            {"url": "https://b.com", "title": "B", "content": "has content", "score": 0.8}
        ]
    })";

    auto resp = TavilyOracle::parse_response_json(json);
    CHECK(resp.ok());
    REQUIRE(resp.results.size() == 1);
    CHECK(resp.results[0].url == "https://b.com");
}

TEST_CASE("TavilyOracle — parse response with no results array", "[tavily][unit]") {
    const std::string json = R"({"query": "nothing", "response_time": 0.01})";
    auto resp = TavilyOracle::parse_response_json(json);
    CHECK(resp.ok());
    CHECK(resp.results.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
//  Credential loader — load_tavily_api_key()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("TavilyOracle — load_tavily_api_key from file", "[tavily][unit]") {
    // Write a temp creds file
    const std::string tmp_path = "/tmp/nikola_test_tavily_creds.txt";
    {
        std::ofstream f(tmp_path);
        f << "api-key:\n";
        f << "tvly-test-abcdef123456\n";
        f << "\n";
        f << "mcp access:\n";
        f << "https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-test-abcdef123456\n";
    }

    auto key = load_tavily_api_key(tmp_path);
    CHECK(key == "tvly-test-abcdef123456");

    std::remove(tmp_path.c_str());
}

TEST_CASE("TavilyOracle — load_tavily_api_key handles whitespace", "[tavily][unit]") {
    const std::string tmp_path = "/tmp/nikola_test_tavily_ws.txt";
    {
        std::ofstream f(tmp_path);
        f << "  \n";
        f << "  tvly-with-spaces  \n";
    }

    auto key = load_tavily_api_key(tmp_path);
    CHECK(key == "tvly-with-spaces");

    std::remove(tmp_path.c_str());
}

TEST_CASE("TavilyOracle — load_tavily_api_key returns empty for missing file", "[tavily][unit]") {
    auto key = load_tavily_api_key("/tmp/nonexistent_creds_file_xyz.txt");
    CHECK(key.empty());
}

TEST_CASE("TavilyOracle — load_tavily_api_key returns empty for no key", "[tavily][unit]") {
    const std::string tmp_path = "/tmp/nikola_test_tavily_nokey.txt";
    {
        std::ofstream f(tmp_path);
        f << "some random content\n";
        f << "no api key here\n";
    }

    auto key = load_tavily_api_key(tmp_path);
    CHECK(key.empty());

    std::remove(tmp_path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
//  OraclePool integration
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("TavilyOracle — can be added to OraclePool", "[tavily][unit]") {
    OraclePool pool;
    pool.add_oracle(std::make_shared<TavilyOracle>("tvly-pool-test-key"));
    CHECK(pool.size() == 1);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Live network tests — requires valid Tavily API key
// ─────────────────────────────────────────────────────────────────────────────

static std::string get_test_api_key() {
    auto key = load_tavily_api_key("/home/randy/Workspace/CREDS/creds/tavily.creds");
    if (key.empty()) {
        SKIP("Tavily API key not found at /home/randy/Workspace/CREDS/creds/tavily.creds");
    }
    return key;
}

TEST_CASE("TavilyOracle — live search returns results", "[tavily][network]") {
    auto key = get_test_api_key();
    TavilyOracle oracle(key);

    auto response = oracle.search("What is the C++ programming language?");
    REQUIRE(response.ok());
    CHECK_FALSE(response.results.empty());
    CHECK(response.results.size() <= 5);
    CHECK(oracle.api_call_count() == 1);

    // Each result should have content
    for (const auto& r : response.results) {
        CHECK_FALSE(r.url.empty());
        CHECK_FALSE(r.content.empty());
        CHECK(r.score > 0.0f);
    }
}

TEST_CASE("TavilyOracle — live search_text returns formatted text", "[tavily][network]") {
    auto key = get_test_api_key();
    TavilyOracle oracle(key);

    auto text = oracle.search_text("Linux kernel");
    CHECK_FALSE(text.empty());
    CHECK(text.find("##") != std::string::npos);  // Has markdown headers
    CHECK(text.find("http") != std::string::npos); // Has URLs
}

TEST_CASE("TavilyOracle — live assess returns credibility score", "[tavily][network]") {
    auto key = get_test_api_key();
    TavilyOracle oracle(key);

    auto verdict = oracle.assess(
        "What is the capital of France?",
        "The capital of France is Paris. It is the largest city in France.");

    CHECK(verdict.confidence >= 0.0f);
    CHECK(verdict.confidence <= 1.0f);
    // Good content about Paris should score reasonably well
    CHECK(verdict.confidence > 0.2f);
    CHECK_FALSE(verdict.rationale.empty());
}

TEST_CASE("TavilyOracle — live assess with empty content", "[tavily][network]") {
    auto key = get_test_api_key();
    TavilyOracle oracle(key);

    auto verdict = oracle.assess("test query", "");
    // Empty content returns 0 confidence without making API call
    CHECK(verdict.confidence == 0.0f);
    CHECK(oracle.api_call_count() == 0);
}
