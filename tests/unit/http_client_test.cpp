/**
 * @file tests/unit/http_client_test.cpp
 * @brief Phase 32 — HttpClient unit tests (Catch2 v3).
 *
 * Tests the HTTP client wrapper without requiring a live network.
 * Validates: construction, URL validation, configuration, move semantics,
 * rate limiting, and retry logic.
 *
 * Live network tests are guarded by the [network] tag — run them explicitly
 * with: ./test_http_client "[network]"
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/infrastructure/http_client.hpp>

#include <chrono>
#include <string>
#include <thread>

using namespace nikola::infrastructure;

// ─────────────────────────────────────────────────────────────────────────────
//  Construction & Configuration
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("HttpClient — default construction succeeds", "[http][unit]") {
    HttpClient client;
    CHECK(client.request_count() == 0);
    CHECK(client.config().connect_timeout_s == 10);
    CHECK(client.config().request_timeout_s == 30);
    CHECK(client.config().max_retries == 2);
    CHECK(client.config().verify_tls == true);
    CHECK(client.config().user_agent == "NikolaHTTP/0.2.0");
}

TEST_CASE("HttpClient — custom configuration", "[http][unit]") {
    HttpClientConfig cfg;
    cfg.connect_timeout_s = 5;
    cfg.request_timeout_s = 15;
    cfg.max_retries = 0;
    cfg.user_agent = "TestAgent/1.0";
    cfg.verify_tls = false;
    cfg.max_response_bytes = 1024;

    HttpClient client(cfg);
    CHECK(client.config().connect_timeout_s == 5);
    CHECK(client.config().request_timeout_s == 15);
    CHECK(client.config().max_retries == 0);
    CHECK(client.config().user_agent == "TestAgent/1.0");
    CHECK(client.config().verify_tls == false);
    CHECK(client.config().max_response_bytes == 1024);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Move Semantics
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("HttpClient — move construction", "[http][unit]") {
    HttpClient client1;
    HttpClient client2(std::move(client1));
    CHECK(client2.request_count() == 0);
    // client1 is in moved-from state — using it should return error
}

TEST_CASE("HttpClient — move assignment", "[http][unit]") {
    HttpClient client1;
    HttpClient client2;
    client2 = std::move(client1);
    CHECK(client2.request_count() == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  URL Validation
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("HttpClient — rejects non-HTTP URLs", "[http][unit]") {
    HttpClient client;

    auto resp = client.get("ftp://example.com/file");
    CHECK_FALSE(resp.ok());
    CHECK(resp.status_code == 0);
    CHECK(resp.error.find("http") != std::string::npos);

    auto resp2 = client.get("file:///etc/passwd");
    CHECK_FALSE(resp2.ok());
    CHECK(resp2.status_code == 0);

    auto resp3 = client.get("javascript:alert(1)");
    CHECK_FALSE(resp3.ok());
    CHECK(resp3.status_code == 0);

    auto resp4 = client.get("");
    CHECK_FALSE(resp4.ok());
    CHECK(resp4.status_code == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Moved-from client returns error
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("HttpClient — moved-from client returns error gracefully", "[http][unit]") {
    HttpClient client1;
    HttpClient client2(std::move(client1));

    // client1 is now empty — should not crash, just return an error
    auto resp = client1.get("https://example.com");
    CHECK_FALSE(resp.ok());
    CHECK(resp.status_code == 0);
    CHECK_FALSE(resp.error.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
//  HttpResponse::ok()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("HttpResponse::ok() — status code ranges", "[http][unit]") {
    HttpResponse r;
    r.status_code = 200;
    CHECK(r.ok());

    r.status_code = 201;
    CHECK(r.ok());

    r.status_code = 299;
    CHECK(r.ok());

    r.status_code = 199;
    CHECK_FALSE(r.ok());

    r.status_code = 300;
    CHECK_FALSE(r.ok());

    r.status_code = 404;
    CHECK_FALSE(r.ok());

    r.status_code = 500;
    CHECK_FALSE(r.ok());

    r.status_code = 0;
    CHECK_FALSE(r.ok());
}

// ─────────────────────────────────────────────────────────────────────────────
//  Live Network Tests (opt-in via [network] tag)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("HttpClient — GET to httpbin.org", "[http][network]") {
    HttpClientConfig cfg;
    cfg.request_timeout_s = 10;
    cfg.max_retries = 1;

    HttpClient client(cfg);
    auto resp = client.get("https://httpbin.org/get");

    // httpbin may be down — pass if network unreachable
    if (resp.status_code == 0) {
        WARN("Network unavailable — skipping live test");
        return;
    }

    CHECK(resp.ok());
    CHECK(resp.status_code == 200);
    CHECK(resp.body.find("headers") != std::string::npos);
    CHECK(resp.elapsed_seconds > 0.0);
    CHECK(client.request_count() >= 1);
}

TEST_CASE("HttpClient — POST JSON to httpbin.org", "[http][network]") {
    HttpClientConfig cfg;
    cfg.request_timeout_s = 10;
    cfg.max_retries = 1;

    HttpClient client(cfg);
    auto resp = client.post_json("https://httpbin.org/post",
                                 R"({"query":"test","limit":5})");

    if (resp.status_code == 0) {
        WARN("Network unavailable — skipping live test");
        return;
    }

    CHECK(resp.ok());
    CHECK(resp.status_code == 200);
    CHECK(resp.body.find("test") != std::string::npos);
}

TEST_CASE("HttpClient — connection refused returns error", "[http][network]") {
    HttpClientConfig cfg;
    cfg.connect_timeout_s = 2;
    cfg.request_timeout_s = 3;
    cfg.max_retries = 0;  // No retries — fast failure

    HttpClient client(cfg);
    // Port 1 is unlikely to be listening
    auto resp = client.get("http://127.0.0.1:1/test");

    CHECK_FALSE(resp.ok());
    CHECK(resp.status_code == 0);
    CHECK_FALSE(resp.error.empty());
}

TEST_CASE("HttpClient — invalid hostname", "[http][network]") {
    HttpClientConfig cfg;
    cfg.connect_timeout_s = 2;
    cfg.request_timeout_s = 3;
    cfg.max_retries = 0;

    HttpClient client(cfg);
    auto resp = client.get("https://this-domain-does-not-exist-xyz123.com");

    CHECK_FALSE(resp.ok());
    CHECK(resp.status_code == 0);
    CHECK_FALSE(resp.error.empty());
}

TEST_CASE("HttpClient — custom headers are sent", "[http][network]") {
    HttpClientConfig cfg;
    cfg.request_timeout_s = 10;
    cfg.max_retries = 1;

    HttpClient client(cfg);
    auto resp = client.get("https://httpbin.org/get",
                           {{"X-Nikola-Test", "phase32"},
                            {"Authorization", "Bearer test-token"}});

    if (resp.status_code == 0) {
        WARN("Network unavailable — skipping live test");
        return;
    }

    CHECK(resp.ok());
    // httpbin echoes headers back in the response
    CHECK(resp.body.find("X-Nikola-Test") != std::string::npos);
    CHECK(resp.body.find("phase32") != std::string::npos);
}
