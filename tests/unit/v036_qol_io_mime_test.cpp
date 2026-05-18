#include <catch2/catch_test_macros.hpp>

#include <nikola/infrastructure/mime_detection_policy.hpp>
#include <nikola/security/io_guard.hpp>

#include <chrono>

using namespace nikola;

TEST_CASE("v0.3.6 §1 IOGuard default constants", "[v036][ioguard]") {
    CHECK(security::IOGUARD_DEFAULT_REFILL_BYTES_PER_SEC == 1024U * 1024U);
    CHECK(security::IOGUARD_DEFAULT_BURST_BYTES == 256U * 1024U);
}

TEST_CASE("v0.3.6 §2 IOGuard consumes burst and blocks overflow", "[v036][ioguard]") {
    security::IOGuard g(/*refill=*/1000, /*burst=*/200);

    REQUIRE(g.allow(150));
    CHECK_FALSE(g.allow(60));
    CHECK(g.allow(50));
}

TEST_CASE("v0.3.6 §3 IOGuard refills over elapsed time", "[v036][ioguard]") {
    using Clock = security::IOGuard::Clock;

    security::IOGuard g(/*refill=*/1000, /*burst=*/1000); // bytes/sec

    const auto t0 = Clock::now();
    REQUIRE(g.allow(1000, t0));
    CHECK_FALSE(g.allow(1, t0));

    const auto t1 = t0 + std::chrono::milliseconds(200); // +200 tokens
    CHECK(g.allow(200, t1));
    CHECK_FALSE(g.allow(1, t1));
}

TEST_CASE("v0.3.6 §4 IOGuard wait-time estimate", "[v036][ioguard]") {
    using Clock = security::IOGuard::Clock;

    security::IOGuard g(/*refill=*/1000, /*burst=*/1000);
    const auto t0 = Clock::now();

    REQUIRE(g.allow(900, t0)); // 100 remain
    const auto wait = g.time_until_available(300, t0);

    // Need ~200 bytes at 1000 B/s => ~200ms
    CHECK(wait >= std::chrono::milliseconds(200));
    CHECK(wait <= std::chrono::milliseconds(205));
}

TEST_CASE("v0.3.6 §5 MIME detects PDF magic regardless of extension", "[v036][mime]") {
    const std::string pdf = "%PDF-1.7\n1 0 obj\n";
    auto mt = infrastructure::resolve_mime("notes.txt", pdf);
    CHECK(mt == infrastructure::MimeType::APPLICATION_PDF);
    CHECK(std::string(infrastructure::mime_type_name(mt)) == "application/pdf");
}

TEST_CASE("v0.3.6 §6 MIME infers JSON from content for unknown extension", "[v036][mime]") {
    const std::string json = "  {\"hello\":\"world\"}";
    auto mt = infrastructure::resolve_mime("payload.bin", json);
    CHECK(mt == infrastructure::MimeType::APPLICATION_JSON);
    CHECK(infrastructure::detect_file_type("payload.bin", json) == infrastructure::FileType::JSON);
}

TEST_CASE("v0.3.6 §7 MIME keeps explicit code extension policy", "[v036][mime]") {
    const std::string code = "int main() { return 0; }\n";
    auto mt_cpp = infrastructure::resolve_mime("main.cpp", code);
    CHECK(mt_cpp == infrastructure::MimeType::TEXT_X_CPP);

    auto mt_aria = infrastructure::resolve_mime("kernel.aria", code);
    CHECK(mt_aria == infrastructure::MimeType::TEXT_X_ARIA);
}

TEST_CASE("v0.3.6 §8 MIME maps PDF to non-ingestible file type", "[v036][mime]") {
    const std::string pdf = "%PDF-2.0\n...";
    CHECK(infrastructure::detect_file_type("doc.pdf", pdf) == infrastructure::FileType::UNKNOWN);
}

TEST_CASE("v0.3.6 §9 MIME infers CSV from first line pattern", "[v036][mime]") {
    const std::string csv = "name,score\nalice,42\n";
    auto mt = infrastructure::resolve_mime("unknown.data", csv);
    CHECK(mt == infrastructure::MimeType::TEXT_CSV);
    CHECK(infrastructure::detect_file_type("unknown.data", csv) == infrastructure::FileType::CSV);
}
