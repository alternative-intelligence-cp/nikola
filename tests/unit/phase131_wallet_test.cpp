/**
 * @file phase131_wallet_test.cpp
 * @brief Phase 131 — SimulatedWallet unit tests
 */

#include <catch2/catch_test_macros.hpp>
#include <nikola/economy/wallet.hpp>

using namespace nikola::economy;

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

TEST_CASE("SimulatedWallet::derive_address — deterministic", "[Phase131][static]") {
    const std::string a1 = SimulatedWallet::derive_address("seed_a");
    const std::string a2 = SimulatedWallet::derive_address("seed_a");
    REQUIRE(a1 == a2);
    REQUIRE(a1.substr(0, 2) == "0x");
    // 0x + 16 + 8 = 26 chars
    REQUIRE(a1.size() == 26);
}

TEST_CASE("SimulatedWallet::derive_address — different seeds give different addresses",
          "[Phase131][static]") {
    REQUIRE(SimulatedWallet::derive_address("seed_a") !=
            SimulatedWallet::derive_address("seed_b"));
}

TEST_CASE("SimulatedWallet::build_expected_sig — format check",
          "[Phase131][static]") {
    const std::string sig = SimulatedWallet::build_expected_sig("0xABCDEFGH", "hello_world");
    REQUIRE(sig.substr(0, 4) == "sig_");
    REQUIRE(sig.find("0xABCDEF") != std::string::npos);
    REQUIRE(sig.find("hello_wo") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

TEST_CASE("SimulatedWallet — no identity initially", "[Phase131][identity]") {
    SimulatedWallet w;
    REQUIRE(w.has_identity() == false);
    REQUIRE(w.get_address().empty() == true);
}

TEST_CASE("SimulatedWallet::derive_identity — sets address", "[Phase131][identity]") {
    SimulatedWallet w;
    const std::string pk = w.derive_identity("my_seed_value");
    REQUIRE(w.has_identity() == true);
    REQUIRE(w.get_address().empty() == false);
    REQUIRE(w.get_address().substr(0, 2) == "0x");
    REQUIRE(pk.empty() == false);
}

TEST_CASE("SimulatedWallet::derive_identity — same seed same address",
          "[Phase131][identity]") {
    SimulatedWallet w1, w2;
    w1.derive_identity("reproducible_seed");
    w2.derive_identity("reproducible_seed");
    REQUIRE(w1.get_address() == w2.get_address());
}

TEST_CASE("SimulatedWallet::derive_identity — different seeds → different addresses",
          "[Phase131][identity]") {
    SimulatedWallet w1, w2;
    w1.derive_identity("seed_alpha");
    w2.derive_identity("seed_beta");
    REQUIRE(w1.get_address() != w2.get_address());
}

// ---------------------------------------------------------------------------
// Signing
// ---------------------------------------------------------------------------

TEST_CASE("SimulatedWallet::sign — returns non-empty sig after identity",
          "[Phase131][sign]") {
    SimulatedWallet w;
    w.derive_identity("my_seed");
    const std::string sig = w.sign("hello_world");
    REQUIRE(sig.empty() == false);
    REQUIRE(w.sign_count() == 1);
}

TEST_CASE("SimulatedWallet::sign — deterministic for same data",
          "[Phase131][sign]") {
    SimulatedWallet w;
    w.derive_identity("seed");
    REQUIRE(w.sign("data_x") == w.sign("data_x"));
}

TEST_CASE("SimulatedWallet::sign — different data → different sigs",
          "[Phase131][sign]") {
    SimulatedWallet w;
    w.derive_identity("seed");
    REQUIRE(w.sign("data_x") != w.sign("data_y"));
}

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------

TEST_CASE("SimulatedWallet::verify — valid round-trip", "[Phase131][verify]") {
    SimulatedWallet w;
    w.derive_identity("seed");
    const std::string data  = "payload_data";
    const std::string sig   = w.sign(data);
    const std::string addr  = w.get_address();
    REQUIRE(w.verify(data, sig, addr) == true);
    REQUIRE(w.verify_count() == 1);
}

TEST_CASE("SimulatedWallet::verify — wrong signature → false",
          "[Phase131][verify]") {
    SimulatedWallet w;
    w.derive_identity("seed");
    REQUIRE(w.verify("data", "completely_wrong_sig", w.get_address()) == false);
}

TEST_CASE("SimulatedWallet::verify — wrong address → false",
          "[Phase131][verify]") {
    SimulatedWallet w;
    w.derive_identity("seed");
    const std::string sig = w.sign("data");
    REQUIRE(w.verify("data", sig, "0xwrongaddress00000000000000") == false);
}

// ---------------------------------------------------------------------------
// Balance
// ---------------------------------------------------------------------------

TEST_CASE("SimulatedWallet — starts with zero balance", "[Phase131][balance]") {
    SimulatedWallet w;
    REQUIRE(w.get_balance_wei() == 0);
}

TEST_CASE("SimulatedWallet::credit — increases balance", "[Phase131][balance]") {
    SimulatedWallet w;
    w.credit(1000);
    REQUIRE(w.get_balance_wei() == 1000);
    w.credit(500);
    REQUIRE(w.get_balance_wei() == 1500);
}

TEST_CASE("SimulatedWallet::debit — decreases balance", "[Phase131][balance]") {
    SimulatedWallet w;
    w.credit(1000);
    REQUIRE(w.debit(300) == true);
    REQUIRE(w.get_balance_wei() == 700);
    REQUIRE(w.debit_count() == 1);
}

TEST_CASE("SimulatedWallet::debit — fails if insufficient funds",
          "[Phase131][balance]") {
    SimulatedWallet w;
    w.credit(100);
    REQUIRE(w.debit(101) == false);
    REQUIRE(w.get_balance_wei() == 100);  // unchanged
    REQUIRE(w.debit_count() == 0);
}

TEST_CASE("SimulatedWallet::debit — exact amount succeeds",
          "[Phase131][balance]") {
    SimulatedWallet w;
    w.credit(500);
    REQUIRE(w.debit(500) == true);
    REQUIRE(w.get_balance_wei() == 0);
}
