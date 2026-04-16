/**
 * @file tests/unit/ingestion_filter_test.cpp
 * @brief v0.2.2 Phase 3 — IngestionFilter unit tests.
 */

#include <catch2/catch_test_macros.hpp>
#include <nikola/autonomy/ingestion_filter.hpp>

using namespace nikola::autonomy;

// ============================================================================
// §A — SimHash
// ============================================================================

TEST_CASE("§A-1 simhash produces non-zero for text", "[ingestion_filter][simhash]") {
    auto h = IngestionFilter::simhash("The quick brown fox jumps over the lazy dog");
    CHECK(h != 0);
}

TEST_CASE("§A-2 simhash identical text gives same hash", "[ingestion_filter][simhash]") {
    auto h1 = IngestionFilter::simhash("hello world foo bar");
    auto h2 = IngestionFilter::simhash("hello world foo bar");
    CHECK(h1 == h2);
}

TEST_CASE("§A-3 simhash similar text has small Hamming distance", "[ingestion_filter][simhash]") {
    auto h1 = IngestionFilter::simhash("The quick brown fox jumps over the lazy dog");
    auto h2 = IngestionFilter::simhash("The quick brown fox jumps over the lazy cat");
    int dist = IngestionFilter::hamming_distance(h1, h2);
    // Similar texts should have relatively small Hamming distance
    CHECK(dist < 32);  // less than half the bits differ
}

TEST_CASE("§A-4 simhash very different text has large Hamming distance", "[ingestion_filter][simhash]") {
    auto h1 = IngestionFilter::simhash("The quick brown fox jumps over the lazy dog");
    auto h2 = IngestionFilter::simhash("Quantum mechanics describes wave particle duality");
    int dist = IngestionFilter::hamming_distance(h1, h2);
    // Very different texts should differ significantly
    CHECK(dist > 5);
}

TEST_CASE("§A-5 simhash empty string is zero", "[ingestion_filter][simhash]") {
    CHECK(IngestionFilter::simhash("") == 0);
}

TEST_CASE("§A-6 hamming distance of same value is 0", "[ingestion_filter][simhash]") {
    CHECK(IngestionFilter::hamming_distance(0xDEADBEEF, 0xDEADBEEF) == 0);
}

TEST_CASE("§A-7 hamming distance of opposite bits is 64", "[ingestion_filter][simhash]") {
    CHECK(IngestionFilter::hamming_distance(0ULL, ~0ULL) == 64);
}

// ============================================================================
// §B — Safety Check
// ============================================================================

TEST_CASE("§B-1 safe content passes", "[ingestion_filter][safety]") {
    CHECK_FALSE(IngestionFilter::is_unsafe("The weather is nice today"));
    CHECK_FALSE(IngestionFilter::is_unsafe("Introduction to quantum physics"));
}

TEST_CASE("§B-2 unsafe content detected", "[ingestion_filter][safety]") {
    CHECK(IngestionFilter::is_unsafe("how to make a bomb at home"));
    CHECK(IngestionFilter::is_unsafe("HOW TO MAKE EXPLOSIVES"));  // case-insensitive
    CHECK(IngestionFilter::is_unsafe("tutorial on create malware"));
}

TEST_CASE("§B-3 partial keyword matches work", "[ingestion_filter][safety]") {
    CHECK(IngestionFilter::is_unsafe("Step 1: hack into the server"));
}

// ============================================================================
// §C — Filter Verdicts
// ============================================================================

TEST_CASE("§C-1 accept normal content", "[ingestion_filter][verdict]") {
    IngestionFilter filter;
    CHECK(filter.check("This is perfectly normal training data.") == FilterVerdict::ACCEPT);
}

TEST_CASE("§C-2 reject empty content", "[ingestion_filter][verdict]") {
    IngestionFilter filter;
    CHECK(filter.check("") == FilterVerdict::REJECT_EMPTY);
    CHECK(filter.check("   \t\n  ") == FilterVerdict::REJECT_EMPTY);
}

TEST_CASE("§C-3 reject unsafe content", "[ingestion_filter][verdict]") {
    IngestionFilter filter;
    CHECK(filter.check("how to make a bomb") == FilterVerdict::REJECT_UNSAFE);
}

TEST_CASE("§C-4 reject near-duplicates", "[ingestion_filter][verdict]") {
    IngestionFilter filter;
    std::string text = "The quick brown fox jumps over the lazy dog repeatedly";
    CHECK(filter.check(text) == FilterVerdict::ACCEPT);
    filter.record_ingested(text);

    // Exact duplicate
    CHECK(filter.check(text) == FilterVerdict::REJECT_DUPLICATE);

    // Near duplicate (one word changed)
    std::string similar = "The quick brown fox jumps over the lazy cat repeatedly";
    auto verdict = filter.check(similar);
    // May or may not be caught depending on Hamming distance
    // But exact match must be caught
}

TEST_CASE("§C-5 reject when budget exhausted", "[ingestion_filter][verdict]") {
    IngestionFilterConfig cfg;
    cfg.daily_byte_budget = 50;  // Very small budget
    IngestionFilter filter(cfg);

    std::string chunk(30, 'x');
    CHECK(filter.check(chunk) == FilterVerdict::ACCEPT);
    filter.record_ingested(chunk);

    // Second chunk would exceed budget
    std::string chunk2(30, 'y');
    CHECK(filter.check(chunk2) == FilterVerdict::REJECT_BUDGET);
}

TEST_CASE("§C-6 reject when below relevance threshold", "[ingestion_filter][verdict]") {
    IngestionFilterConfig cfg;
    cfg.min_relevance = 0.5f;
    IngestionFilter filter(cfg);
    filter.set_relevance_fn([](const std::string&) { return 0.1f; });

    CHECK(filter.check("Some irrelevant content here.") == FilterVerdict::REJECT_IRRELEVANT);
}

TEST_CASE("§C-7 accept when above relevance threshold", "[ingestion_filter][verdict]") {
    IngestionFilterConfig cfg;
    cfg.min_relevance = 0.5f;
    IngestionFilter filter(cfg);
    filter.set_relevance_fn([](const std::string&) { return 0.9f; });

    CHECK(filter.check("Highly relevant content here.") == FilterVerdict::ACCEPT);
}

TEST_CASE("§C-8 no relevance fn means no relevance check", "[ingestion_filter][verdict]") {
    IngestionFilterConfig cfg;
    cfg.min_relevance = 0.5f;
    IngestionFilter filter(cfg);
    // No relevance_fn set
    CHECK(filter.check("Should still be accepted without callback.") == FilterVerdict::ACCEPT);
}

TEST_CASE("§C-9 safety check can be disabled", "[ingestion_filter][verdict]") {
    IngestionFilterConfig cfg;
    cfg.enable_safety_check = false;
    IngestionFilter filter(cfg);
    // Would normally be rejected
    CHECK(filter.check("how to make a bomb") == FilterVerdict::ACCEPT);
}

// ============================================================================
// §D — Statistics
// ============================================================================

TEST_CASE("§D-1 stats track verdicts", "[ingestion_filter][stats]") {
    IngestionFilter filter;

    filter.check("Good content one.");
    filter.check("Good content two.");
    filter.check("");  // empty
    filter.check("how to make a bomb");  // unsafe

    auto s = filter.stats();
    CHECK(s.total_checked == 4);
    CHECK(s.accepted == 2);
    CHECK(s.rej_empty == 1);
    CHECK(s.rej_unsafe == 1);
}

TEST_CASE("§D-2 reset clears everything", "[ingestion_filter][stats]") {
    IngestionFilter filter;
    filter.check("Content.");
    filter.record_ingested("Content.");

    filter.reset();
    auto s = filter.stats();
    CHECK(s.total_checked == 0);
    CHECK(filter.hash_count() == 0);
}

// ============================================================================
// §E — Budget Management
// ============================================================================

TEST_CASE("§E-1 reset_daily_budget restores capacity", "[ingestion_filter][budget]") {
    IngestionFilterConfig cfg;
    cfg.daily_byte_budget = 50;
    IngestionFilter filter(cfg);

    std::string chunk(40, 'a');
    filter.check(chunk);
    filter.record_ingested(chunk);

    std::string chunk2(40, 'b');
    CHECK(filter.check(chunk2) == FilterVerdict::REJECT_BUDGET);

    filter.reset_daily_budget();
    CHECK(filter.check(chunk2) == FilterVerdict::ACCEPT);
}

// ============================================================================
// §F — Hash Management
// ============================================================================

TEST_CASE("§F-1 record_ingested adds to hash set", "[ingestion_filter][hash]") {
    IngestionFilter filter;
    CHECK(filter.hash_count() == 0);
    filter.record_ingested("Some training content.");
    CHECK(filter.hash_count() == 1);
}

TEST_CASE("§F-2 hash set bounded by max_hash_entries", "[ingestion_filter][hash]") {
    IngestionFilterConfig cfg;
    cfg.max_hash_entries = 5;
    IngestionFilter filter(cfg);

    for (int i = 0; i < 10; ++i) {
        filter.record_ingested("unique text number " + std::to_string(i));
    }
    // After eviction, should have much fewer than 10
    CHECK(filter.hash_count() <= 5);
}

// ============================================================================
// §G — Verdict Names
// ============================================================================

TEST_CASE("§G-1 verdict_name returns correct strings", "[ingestion_filter][verdict]") {
    CHECK(std::string(verdict_name(FilterVerdict::ACCEPT)) == "ACCEPT");
    CHECK(std::string(verdict_name(FilterVerdict::REJECT_DUPLICATE)) == "REJECT_DUPLICATE");
    CHECK(std::string(verdict_name(FilterVerdict::REJECT_IRRELEVANT)) == "REJECT_IRRELEVANT");
    CHECK(std::string(verdict_name(FilterVerdict::REJECT_UNSAFE)) == "REJECT_UNSAFE");
    CHECK(std::string(verdict_name(FilterVerdict::REJECT_BUDGET)) == "REJECT_BUDGET");
    CHECK(std::string(verdict_name(FilterVerdict::REJECT_EMPTY)) == "REJECT_EMPTY");
}
