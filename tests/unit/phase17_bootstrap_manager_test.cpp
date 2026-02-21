/**
 * @file tests/unit/phase17_bootstrap_manager_test.cpp
 * @brief NIK-002 — Bootstrap Token Security test suite (Catch2 v3).
 *
 * Background:
 *   The original Nikola v0.0.4 spec generates a bootstrap token and prints it
 *   to stdout.  In containerized environments, log aggregation latency and
 *   CrashLoopBackOff cycles can render the token inaccessible (300-second
 *   lockdown triggers before admin can retrieve it).  BootstrapManager adds:
 *     - Tier 1: NIKOLA_BOOTSTRAP_TOKEN env var with atomic memory scrub
 *     - Tier 2: Docker/K8s file secret (/run/secrets/nikola_bootstrap_token)
 *     - Tier 3: Generated 256-bit random token → boxed stdout print
 *     - 300s expiry (configurable for tests via set_expiry_seconds())
 *     - Rate limiting: 5 failures / 60s → 5s delay ("Paranoid Mode")
 *
 * Tests:
 *   NIK-002-A  generate_random_token() produces 64-char hex string
 *   NIK-002-B  Tier 1 — env var path: token read, env var scrubbed
 *   NIK-002-C  Tier 2 — file path: token read from temp file
 *   NIK-002-D  Tier 3 — fallback: 64-char hex token generated (silent mode)
 *   NIK-002-E  validate() succeeds for correct token before expiry
 *   NIK-002-F  validate() rejects wrong token and records failure
 *   NIK-002-G  Token expires after configured window
 *   NIK-002-H  validate() rejects after expiry
 *   NIK-002-I  Rate limiter triggers after 5 failures per client_id
 *   NIK-002-J  rate_limit_delay=0 for test — no hang, still rejected
 *   NIK-002-K  reset() restores clean state
 *   NIK-002-L  source() returns correct tier
 *   NIK-002-M  constant-time comparison — same-length wrong token rejected
 *   NIK-002-N  validate() rejected when no token acquired
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/security/bootstrap_manager.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>

using nikola::security::BootstrapManager;
using nikola::security::BOOTSTRAP_TOKEN_LENGTH;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// True iff all 64 chars are lowercase hex digits.
static bool is_hex64(const std::string& s) {
    if (s.size() != 64) return false;
    return std::all_of(s.begin(), s.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

/// Write a known token to a temp file; return the path.
static std::string write_temp_token(const std::string& token) {
    const std::string path = "/tmp/nikola_test_bootstrap_token";
    std::ofstream ofs(path, std::ofstream::trunc);
    REQUIRE(ofs.is_open());
    ofs << token << '\n';
    return path;
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-002-A  generate_random_token()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-002-A — generate_random_token produces 64-char hex string",
          "[bootstrap_manager][nik002][gen]")
{
    const std::string tok = BootstrapManager::generate_random_token();
    CHECK(tok.size() == 64);
    CHECK(is_hex64(tok));
}

TEST_CASE("NIK-002-A — generate_random_token produces different tokens each call",
          "[bootstrap_manager][nik002][gen]")
{
    // Two consecutive calls should produce different tokens (256-bit random)
    const std::string a = BootstrapManager::generate_random_token();
    const std::string b = BootstrapManager::generate_random_token();
    CHECK(a != b);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-002-B  Tier 1 — environment variable
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-002-B — Tier 1: env var token is read correctly",
          "[bootstrap_manager][nik002][tier1]")
{
    constexpr const char* known = "aabb11223344556677889900aabb11223344556677889900aabb112233445566";
    // Note: 64-char hex string above
    REQUIRE(std::strlen(known) == 64);

    ::setenv(nikola::security::BOOTSTRAP_ENV_VAR, known, 1);

    BootstrapManager mgr;
    mgr.set_silent(true);
    const std::string tok = mgr.get_token();

    CHECK(tok == known);
    CHECK(mgr.source() == BootstrapManager::TokenSource::ENV_VAR);
}

TEST_CASE("NIK-002-B — Tier 1: env var is unset after get_token() (scrubbed)",
          "[bootstrap_manager][nik002][tier1][security]")
{
    constexpr const char* known = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef";
    ::setenv(nikola::security::BOOTSTRAP_ENV_VAR, known, 1);
    REQUIRE(std::getenv(nikola::security::BOOTSTRAP_ENV_VAR) != nullptr);

    BootstrapManager mgr;
    mgr.set_silent(true);
    mgr.get_token();

    // After get_token(), the env var should have been unset
    CHECK(std::getenv(nikola::security::BOOTSTRAP_ENV_VAR) == nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-002-C  Tier 2 — file secret
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-002-C — Tier 2: token read from file secret",
          "[bootstrap_manager][nik002][tier2]")
{
    // Make sure env var is clear
    ::unsetenv(nikola::security::BOOTSTRAP_ENV_VAR);

    const std::string known = "cafebabecafebabecafebabecafebabecafebabecafebabecafebabecafebabe";
    REQUIRE(known.size() == 64);
    const std::string path = write_temp_token(known);

    BootstrapManager mgr;
    mgr.set_silent(true);
    mgr.set_secret_file_path(path);

    const std::string tok = mgr.get_token();
    CHECK(tok == known);
    CHECK(mgr.source() == BootstrapManager::TokenSource::FILE);

    std::filesystem::remove(path);
}

TEST_CASE("NIK-002-C — Tier 2: trailing whitespace/newline is stripped",
          "[bootstrap_manager][nik002][tier2]")
{
    ::unsetenv(nikola::security::BOOTSTRAP_ENV_VAR);

    const std::string known = "1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff";
    // Write token with trailing \r\n\t
    const std::string path = "/tmp/nikola_test_token_whitespace";
    {
        std::ofstream ofs(path, std::ofstream::trunc);
        REQUIRE(ofs.is_open());
        ofs << known << "\r\n  \t";
    }

    BootstrapManager mgr;
    mgr.set_silent(true);
    mgr.set_secret_file_path(path);

    const std::string tok = mgr.get_token();
    CHECK(tok == known);

    std::filesystem::remove(path);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-002-D  Tier 3 — fallback generation
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-002-D — Tier 3: fallback generates valid 64-char hex token",
          "[bootstrap_manager][nik002][tier3]")
{
    ::unsetenv(nikola::security::BOOTSTRAP_ENV_VAR);

    BootstrapManager mgr;
    mgr.set_silent(true);
    // Set secret file path to something non-existent
    mgr.set_secret_file_path("/tmp/nikola_nonexistent_XXXXXX_secret");

    const std::string tok = mgr.get_token();
    CHECK(is_hex64(tok));
    CHECK(mgr.source() == BootstrapManager::TokenSource::GENERATED);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-002-E  validate() succeeds for correct token
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-002-E — validate() returns true for correct token",
          "[bootstrap_manager][nik002][validate]")
{
    ::unsetenv(nikola::security::BOOTSTRAP_ENV_VAR);

    BootstrapManager mgr;
    mgr.set_silent(true);
    mgr.set_secret_file_path("/tmp/nikola_nonexistent_secret");
    mgr.set_expiry_seconds(9999);

    const std::string tok = mgr.get_token();
    CHECK(mgr.validate(tok, "test_client"));
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-002-F  validate() rejects wrong token
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-002-F — validate() returns false for wrong token",
          "[bootstrap_manager][nik002][validate]")
{
    ::unsetenv(nikola::security::BOOTSTRAP_ENV_VAR);

    BootstrapManager mgr;
    mgr.set_silent(true);
    mgr.set_secret_file_path("/tmp/nikola_nonexistent_secret");
    mgr.set_expiry_seconds(9999);
    mgr.set_rate_limit_delay(0);

    const std::string tok = mgr.get_token();
    const std::string wrong(64, 'a');   // 64 'a' chars — different from generated token
    CHECK_FALSE(mgr.validate(wrong, "test_client"));
}

TEST_CASE("NIK-002-F — validate() records failure on wrong token",
          "[bootstrap_manager][nik002][validate]")
{
    ::unsetenv(nikola::security::BOOTSTRAP_ENV_VAR);

    BootstrapManager mgr;
    mgr.set_silent(true);
    mgr.set_secret_file_path("/tmp/nikola_nonexistent_secret");
    mgr.set_expiry_seconds(9999);
    mgr.set_rate_limit_delay(0);

    mgr.get_token();

    const std::string wrong(64, 'b');
    mgr.validate(wrong, "bad_client");
    CHECK(mgr.failure_count("bad_client") == 1);
    mgr.validate(wrong, "bad_client");
    CHECK(mgr.failure_count("bad_client") == 2);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-002-G / H  Token expiry
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-002-G — is_expired() is false within expiry window",
          "[bootstrap_manager][nik002][expiry]")
{
    ::unsetenv(nikola::security::BOOTSTRAP_ENV_VAR);

    BootstrapManager mgr;
    mgr.set_silent(true);
    mgr.set_secret_file_path("/tmp/nikola_nonexistent_secret");
    mgr.set_expiry_seconds(9999);
    mgr.get_token();

    CHECK_FALSE(mgr.is_expired());
}

TEST_CASE("NIK-002-G — is_expired() is true before get_token() is called",
          "[bootstrap_manager][nik002][expiry]")
{
    BootstrapManager mgr;
    CHECK(mgr.is_expired());
}

TEST_CASE("NIK-002-H — validate() fails after expiry",
          "[bootstrap_manager][nik002][expiry]")
{
    ::unsetenv(nikola::security::BOOTSTRAP_ENV_VAR);

    BootstrapManager mgr;
    mgr.set_silent(true);
    mgr.set_secret_file_path("/tmp/nikola_nonexistent_secret");
    mgr.set_expiry_seconds(0);   // immediately expired
    mgr.set_rate_limit_delay(0);

    const std::string tok = mgr.get_token();
    // With 0-second expiry, 1 second has already passed by now
    CHECK_FALSE(mgr.validate(tok, "test"));
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-002-I  Rate limiter after 5 failures
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-002-I — rate limiter blocks after 5 failures (delay=0 for test)",
          "[bootstrap_manager][nik002][rate_limit]")
{
    ::unsetenv(nikola::security::BOOTSTRAP_ENV_VAR);

    BootstrapManager mgr;
    mgr.set_silent(true);
    mgr.set_secret_file_path("/tmp/nikola_nonexistent_secret");
    mgr.set_expiry_seconds(9999);
    mgr.set_rate_limit_delay(0);   // 0-second delay — test won't hang

    mgr.get_token();

    const std::string wrong(64, 'f');

    // 5 failures to trigger paranoid mode
    for (int i = 0; i < 5; ++i)
        mgr.validate(wrong, "attacker");

    CHECK(mgr.failure_count("attacker") == 5);

    // 6th attempt — should be rate-limited (returns false)
    CHECK_FALSE(mgr.validate(wrong, "attacker"));
}

TEST_CASE("NIK-002-J — per-client rate limiting is isolated",
          "[bootstrap_manager][nik002][rate_limit]")
{
    ::unsetenv(nikola::security::BOOTSTRAP_ENV_VAR);

    BootstrapManager mgr;
    mgr.set_silent(true);
    mgr.set_secret_file_path("/tmp/nikola_nonexistent_secret");
    mgr.set_expiry_seconds(9999);
    mgr.set_rate_limit_delay(0);

    const std::string tok = mgr.get_token();
    const std::string wrong(64, 'e');

    // Exhaust rate limit for "attacker"
    for (int i = 0; i < 6; ++i)
        mgr.validate(wrong, "attacker");

    // "legit" client should still be able to validate correctly
    CHECK(mgr.validate(tok, "legit"));
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-002-K  reset()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-002-K — reset() clears token and rate limits",
          "[bootstrap_manager][nik002][reset]")
{
    ::unsetenv(nikola::security::BOOTSTRAP_ENV_VAR);

    BootstrapManager mgr;
    mgr.set_silent(true);
    mgr.set_secret_file_path("/tmp/nikola_nonexistent_secret");
    mgr.set_expiry_seconds(9999);
    mgr.set_rate_limit_delay(0);

    const std::string tok = mgr.get_token();
    REQUIRE(!tok.empty());

    const std::string wrong(64, 'c');
    for (int i = 0; i < 3; ++i)
        mgr.validate(wrong, "client");
    REQUIRE(mgr.failure_count("client") == 3);

    mgr.reset();

    CHECK(mgr.token().empty());
    CHECK(mgr.source() == BootstrapManager::TokenSource::NONE);
    CHECK(mgr.is_expired());
    CHECK(mgr.failure_count("client") == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-002-L  source() returns correct tier
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-002-L — source() returns NONE before get_token()",
          "[bootstrap_manager][nik002][source]")
{
    BootstrapManager mgr;
    CHECK(mgr.source() == BootstrapManager::TokenSource::NONE);
}

TEST_CASE("NIK-002-L — source() returns FILE for Tier 2",
          "[bootstrap_manager][nik002][source]")
{
    ::unsetenv(nikola::security::BOOTSTRAP_ENV_VAR);

    const std::string known = "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";
    REQUIRE(known.size() == 64);
    const std::string path = write_temp_token(known);

    BootstrapManager mgr;
    mgr.set_silent(true);
    mgr.set_secret_file_path(path);
    mgr.get_token();

    CHECK(mgr.source() == BootstrapManager::TokenSource::FILE);

    std::filesystem::remove(path);
}

TEST_CASE("NIK-002-L — source() returns GENERATED for Tier 3",
          "[bootstrap_manager][nik002][source]")
{
    ::unsetenv(nikola::security::BOOTSTRAP_ENV_VAR);

    BootstrapManager mgr;
    mgr.set_silent(true);
    mgr.set_secret_file_path("/tmp/nikola_nonexistent_secret");
    mgr.get_token();

    CHECK(mgr.source() == BootstrapManager::TokenSource::GENERATED);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-002-M  Constant-time comparison
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-002-M — wrong token of correct length is rejected",
          "[bootstrap_manager][nik002][constant_time]")
{
    ::unsetenv(nikola::security::BOOTSTRAP_ENV_VAR);

    BootstrapManager mgr;
    mgr.set_silent(true);
    mgr.set_secret_file_path("/tmp/nikola_nonexistent_secret");
    mgr.set_expiry_seconds(9999);
    mgr.set_rate_limit_delay(0);

    const std::string tok = mgr.get_token();

    // Build a wrong token of the same length (flip one bit)
    std::string wrong = tok;
    wrong[0] = (wrong[0] == '0') ? '1' : '0';

    CHECK_FALSE(mgr.validate(wrong, "test_ct"));
}

TEST_CASE("NIK-002-M — wrong-length token is rejected",
          "[bootstrap_manager][nik002][constant_time]")
{
    ::unsetenv(nikola::security::BOOTSTRAP_ENV_VAR);

    BootstrapManager mgr;
    mgr.set_silent(true);
    mgr.set_secret_file_path("/tmp/nikola_nonexistent_secret");
    mgr.set_expiry_seconds(9999);
    mgr.set_rate_limit_delay(0);

    mgr.get_token();
    CHECK_FALSE(mgr.validate("short", "test_ct"));
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-002-N  validate() with no token acquired
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-002-N — validate() rejects when no token acquired",
          "[bootstrap_manager][nik002][validate]")
{
    BootstrapManager mgr;
    mgr.set_rate_limit_delay(0);
    CHECK_FALSE(mgr.validate("anything", "test"));
}
