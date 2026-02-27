/**
 * @file phase130_marketplace_test.cpp
 * @brief Phase 130 — NeuralMarketplace unit tests
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/economy/marketplace.hpp>

using namespace nikola::economy;
using Catch::Approx;

static ServiceListing make_listing(const std::string& id,
                                    const std::string& provider,
                                    const std::string& desc,
                                    uint64_t price = 100,
                                    double quality = 0.8) {
    ServiceListing s;
    s.service_id       = id;
    s.provider_address = provider;
    s.description      = desc;
    s.price_wei        = price;
    s.quality_score    = quality;
    return s;
}

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

TEST_CASE("NeuralMarketplace::make_tx_hash — deterministic", "[Phase130][static]") {
    const std::string h1 = NeuralMarketplace::make_tx_hash("svc_a", 100);
    const std::string h2 = NeuralMarketplace::make_tx_hash("svc_a", 100);
    REQUIRE(h1 == h2);
    REQUIRE(h1.substr(0, 2) == "0x");
}

TEST_CASE("NeuralMarketplace::make_tx_hash — different for different inputs",
          "[Phase130][static]") {
    const std::string h1 = NeuralMarketplace::make_tx_hash("svc_a", 100);
    const std::string h2 = NeuralMarketplace::make_tx_hash("svc_b", 100);
    const std::string h3 = NeuralMarketplace::make_tx_hash("svc_a", 101);
    REQUIRE(h1 != h2);
    REQUIRE(h1 != h3);
}

TEST_CASE("NeuralMarketplace::keyword_match — case-insensitive",
          "[Phase130][static]") {
    REQUIRE(NeuralMarketplace::keyword_match("Summarize PDF", "pdf")  == true);
    REQUIRE(NeuralMarketplace::keyword_match("Summarize PDF", "summarize") == true);
    REQUIRE(NeuralMarketplace::keyword_match("Physics engine", "xyz") == false);
    REQUIRE(NeuralMarketplace::keyword_match("Hello", "") == true);
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

TEST_CASE("NeuralMarketplace — empty on construction", "[Phase130][init]") {
    NeuralMarketplace m;
    REQUIRE(m.listing_count()     == 0);
    REQUIRE(m.transaction_count() == 0);
    const auto s = m.stats();
    REQUIRE(s.listing_count     == 0);
    REQUIRE(s.transaction_count == 0);
    REQUIRE(s.total_volume_wei  == 0);
}

// ---------------------------------------------------------------------------
// Listing CRUD
// ---------------------------------------------------------------------------

TEST_CASE("NeuralMarketplace::list_service — adds listing", "[Phase130][listing]") {
    NeuralMarketplace m;
    m.list_service(make_listing("svc:a", "alice", "image analysis"));
    REQUIRE(m.listing_count() == 1);
}

TEST_CASE("NeuralMarketplace::list_service — dedup by service_id",
          "[Phase130][listing]") {
    NeuralMarketplace m;
    m.list_service(make_listing("svc:a", "alice", "old description"));
    m.list_service(make_listing("svc:a", "alice", "new description"));
    REQUIRE(m.listing_count() == 1);
    // Updated description
    const auto found = m.find_service("svc:a");
    REQUIRE(found.has_value());
    REQUIRE(found->description == "new description");
}

TEST_CASE("NeuralMarketplace::delist_service — removes listing",
          "[Phase130][listing]") {
    NeuralMarketplace m;
    m.list_service(make_listing("svc:a", "alice", "test"));
    m.list_service(make_listing("svc:b", "bob",   "test2"));
    m.delist_service("svc:a");
    REQUIRE(m.listing_count() == 1);
    REQUIRE(m.find_service("svc:a").has_value() == false);
    REQUIRE(m.find_service("svc:b").has_value() == true);
}

TEST_CASE("NeuralMarketplace::delist_service — no-op on unknown id",
          "[Phase130][listing]") {
    NeuralMarketplace m;
    m.list_service(make_listing("svc:a", "alice", "test"));
    m.delist_service("does_not_exist");
    REQUIRE(m.listing_count() == 1);
}

TEST_CASE("NeuralMarketplace::find_service — returns nullopt if missing",
          "[Phase130][listing]") {
    NeuralMarketplace m;
    REQUIRE(m.find_service("nonexistent").has_value() == false);
}

// ---------------------------------------------------------------------------
// browse_services
// ---------------------------------------------------------------------------

TEST_CASE("NeuralMarketplace::browse_services — matches description",
          "[Phase130][browse]") {
    NeuralMarketplace m;
    m.list_service(make_listing("svc:summarize", "alice", "Summarize PDF documents"));
    m.list_service(make_listing("svc:physics",   "bob",   "Solve physics equations"));
    m.list_service(make_listing("svc:image",     "carol", "Image analysis using CV"));

    const auto pdf_results = m.browse_services("pdf");
    REQUIRE(pdf_results.size() == 1);
    REQUIRE(pdf_results[0].service_id == "svc:summarize");
}

TEST_CASE("NeuralMarketplace::browse_services — empty query returns all",
          "[Phase130][browse]") {
    NeuralMarketplace m;
    m.list_service(make_listing("svc:a", "alice", "alpha"));
    m.list_service(make_listing("svc:b", "bob",   "beta"));
    REQUIRE(m.browse_services("").size() == 2);
}

TEST_CASE("NeuralMarketplace::browse_services — sorted by quality desc",
          "[Phase130][browse]") {
    NeuralMarketplace m;
    m.list_service(make_listing("svc:low",    "p1", "compute service", 100, 0.3));
    m.list_service(make_listing("svc:high",   "p2", "compute service", 100, 0.9));
    m.list_service(make_listing("svc:medium", "p3", "compute service", 100, 0.6));

    const auto results = m.browse_services("compute");
    REQUIRE(results.size() == 3);
    REQUIRE(results[0].service_id == "svc:high");
    REQUIRE(results[2].service_id == "svc:low");
}

TEST_CASE("NeuralMarketplace::services_by_provider — filters correctly",
          "[Phase130][browse]") {
    NeuralMarketplace m;
    m.list_service(make_listing("svc:a", "alice", "x"));
    m.list_service(make_listing("svc:b", "bob",   "y"));
    m.list_service(make_listing("svc:c", "alice", "z"));

    const auto alice_svcs = m.services_by_provider("alice");
    REQUIRE(alice_svcs.size() == 2);
}

// ---------------------------------------------------------------------------
// Transactions
// ---------------------------------------------------------------------------

TEST_CASE("NeuralMarketplace::purchase_service — returns tx_hash",
          "[Phase130][tx]") {
    NeuralMarketplace m;
    m.list_service(make_listing("svc:a", "alice", "test"));
    const std::string hash = m.purchase_service("svc:a", "buyer1", 100, 10);
    REQUIRE(hash.empty() == false);
    REQUIRE(m.transaction_count() == 1);
}

TEST_CASE("NeuralMarketplace::purchase_service — fails for unknown service",
          "[Phase130][tx]") {
    NeuralMarketplace m;
    const std::string hash = m.purchase_service("nonexistent", "buyer1", 100, 0);
    REQUIRE(hash.empty() == true);
    REQUIRE(m.transaction_count() == 0);
}

TEST_CASE("NeuralMarketplace::execute_service — executes pending tx",
          "[Phase130][tx]") {
    NeuralMarketplace m;
    m.list_service(make_listing("svc:a", "alice", "my service"));
    const std::string hash = m.purchase_service("svc:a", "buyer", 50, 1);
    const std::string result = m.execute_service(hash, "my_input");

    REQUIRE(result.empty() == false);

    const Transaction* tx = m.find_transaction(hash);
    REQUIRE(tx != nullptr);
    REQUIRE(tx->status == TxStatus::EXECUTED);
    REQUIRE(tx->result == result);
}

TEST_CASE("NeuralMarketplace::execute_service — unknown hash returns empty",
          "[Phase130][tx]") {
    NeuralMarketplace m;
    REQUIRE(m.execute_service("0xdeadbeef", "input").empty() == true);
}

TEST_CASE("NeuralMarketplace — transaction stats", "[Phase130][stats]") {
    NeuralMarketplace m;
    m.list_service(make_listing("svc:a", "alice", "test"));
    m.list_service(make_listing("svc:b", "bob",   "test2"));

    const std::string h1 = m.purchase_service("svc:a", "buyer", 200, 1);
    const std::string h2 = m.purchase_service("svc:b", "buyer", 300, 2);
    m.execute_service(h1, "in1");

    const auto s = m.stats();
    REQUIRE(s.transaction_count == 2);
    REQUIRE(s.total_volume_wei  == 500);
    REQUIRE(s.executed_count    == 1);
}

// ---------------------------------------------------------------------------
// Quality rating
// ---------------------------------------------------------------------------

TEST_CASE("NeuralMarketplace::rate_service — EMA blends score",
          "[Phase130][quality]") {
    NeuralMarketplace m;
    m.list_service(make_listing("svc:a", "alice", "test", 100, 0.8));
    m.rate_service("svc:a", 0.0);  // bad rating
    const auto found = m.find_service("svc:a");
    REQUIRE(found.has_value());
    REQUIRE(found->quality_score < 0.8);
}

// ---------------------------------------------------------------------------
// Callback
// ---------------------------------------------------------------------------

TEST_CASE("NeuralMarketplace — on_transaction fires on purchase",
          "[Phase130][callback]") {
    NeuralMarketplace m;
    m.list_service(make_listing("svc:a", "alice", "test"));

    bool fired = false;
    m.on_transaction([&](const Transaction& tx) {
        fired = tx.status == TxStatus::PENDING;
    });

    m.purchase_service("svc:a", "buyer", 100, 0);
    REQUIRE(fired == true);
}

// ---------------------------------------------------------------------------
// clear
// ---------------------------------------------------------------------------

TEST_CASE("NeuralMarketplace::clear — empties everything", "[Phase130][clear]") {
    NeuralMarketplace m;
    m.list_service(make_listing("svc:a", "alice", "test"));
    m.purchase_service("svc:a", "buyer", 100, 0);
    m.clear();
    REQUIRE(m.listing_count()     == 0);
    REQUIRE(m.transaction_count() == 0);
}
