/**
 * @file phase133_peer_registry_test.cpp
 * @brief Phase 133 — PeerRegistry unit tests
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/social/peer_registry.hpp>

using namespace nikola::social;
using Catch::Approx;

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

TEST_CASE("PeerRegistry — empty on construction", "[Phase133][init]") {
    PeerRegistry r;
    REQUIRE(r.peer_count() == 0);
    REQUIRE(r.get_all_peers().empty() == true);
    REQUIRE(r.most_trusted_peer().has_value() == false);
    REQUIRE(r.stats().total_peers == 0);
}

// ---------------------------------------------------------------------------
// add_peer
// ---------------------------------------------------------------------------

TEST_CASE("PeerRegistry::add_peer — registers peer", "[Phase133][add]") {
    PeerRegistry r;
    r.add_peer("peer_a", "pubkey_z85_a");
    REQUIRE(r.peer_count() == 1);
    REQUIRE(r.has_peer("peer_a") == true);
}

TEST_CASE("PeerRegistry::add_peer — multiple peers", "[Phase133][add]") {
    PeerRegistry r;
    r.add_peer("peer_a", "key_a");
    r.add_peer("peer_b", "key_b");
    r.add_peer("peer_c", "key_c");
    REQUIRE(r.peer_count() == 3);
}

TEST_CASE("PeerRegistry::add_peer — dedup: second add updates key",
          "[Phase133][add]") {
    PeerRegistry r;
    r.add_peer("peer_a", "old_key");
    r.add_peer("peer_a", "new_key");
    REQUIRE(r.peer_count() == 1);
    const auto* info = r.find_peer("peer_a");
    REQUIRE(info != nullptr);
    REQUIRE(info->public_key_z85 == "new_key");
}

TEST_CASE("PeerRegistry::add_peer — membrane created for each peer",
          "[Phase133][add]") {
    PeerRegistry r;
    r.add_peer("peer_a", "key_a");
    auto* m = r.get_membrane("peer_a");
    REQUIRE(m != nullptr);
}

// ---------------------------------------------------------------------------
// remove_peer
// ---------------------------------------------------------------------------

TEST_CASE("PeerRegistry::remove_peer — removes existing peer",
          "[Phase133][remove]") {
    PeerRegistry r;
    r.add_peer("peer_a", "key_a");
    r.remove_peer("peer_a");
    REQUIRE(r.peer_count() == 0);
    REQUIRE(r.has_peer("peer_a") == false);
}

TEST_CASE("PeerRegistry::remove_peer — no-op on unknown peer",
          "[Phase133][remove]") {
    PeerRegistry r;
    r.add_peer("peer_a", "key_a");
    r.remove_peer("nonexistent");
    REQUIRE(r.peer_count() == 1);
}

// ---------------------------------------------------------------------------
// find_peer / get_membrane
// ---------------------------------------------------------------------------

TEST_CASE("PeerRegistry::find_peer — returns null for unknown peer",
          "[Phase133][find]") {
    PeerRegistry r;
    REQUIRE(r.find_peer("unknown") == nullptr);
}

TEST_CASE("PeerRegistry::get_membrane — returns null for unknown peer",
          "[Phase133][membrane]") {
    PeerRegistry r;
    REQUIRE(r.get_membrane("unknown") == nullptr);
}

// ---------------------------------------------------------------------------
// record_interaction
// ---------------------------------------------------------------------------

TEST_CASE("PeerRegistry::record_interaction — increments interaction_count",
          "[Phase133][interaction]") {
    PeerRegistry r;
    r.add_peer("peer_a", "key_a");
    r.record_interaction("peer_a", 0.8, 10);
    const auto* info = r.find_peer("peer_a");
    REQUIRE(info != nullptr);
    REQUIRE(info->interaction_count == 1);
    REQUIRE(info->last_resonance == Approx(0.8));
}

TEST_CASE("PeerRegistry::record_interaction — no-op for unknown peer",
          "[Phase133][interaction]") {
    PeerRegistry r;
    r.record_interaction("nonexistent", 0.9, 0);
    REQUIRE(r.stats().total_interactions == 0);
}

TEST_CASE("PeerRegistry::record_interaction — positive resonance raises trust",
          "[Phase133][interaction]") {
    PeerRegistry r;
    r.add_peer("peer_a", "key_a");
    // Default membrane trust = 0.5
    const double trust_before = r.get_membrane("peer_a")->get_trust();
    // Resonance > threshold (0.5) → positive interaction
    r.record_interaction("peer_a", 0.9, 1);
    REQUIRE(r.get_membrane("peer_a")->get_trust() > trust_before);
}

TEST_CASE("PeerRegistry::record_interaction — low resonance lowers trust",
          "[Phase133][interaction]") {
    PeerRegistry r;
    r.add_peer("peer_a", "key_a");
    const double trust_before = r.get_membrane("peer_a")->get_trust();
    r.record_interaction("peer_a", 0.1, 1);
    REQUIRE(r.get_membrane("peer_a")->get_trust() < trust_before);
}

TEST_CASE("PeerRegistry::record_interaction — callback fires",
          "[Phase133][callback]") {
    PeerRegistry r;
    r.add_peer("peer_a", "key_a");

    bool fired       = false;
    double fired_res = 0.0;
    r.on_interaction([&]([[maybe_unused]] const std::string& id, double res) {
        fired     = true;
        fired_res = res;
    });

    r.record_interaction("peer_a", 0.75, 5);
    REQUIRE(fired     == true);
    REQUIRE(fired_res == Approx(0.75));
}

// ---------------------------------------------------------------------------
// most_trusted_peer / peers_by_trust
// ---------------------------------------------------------------------------

TEST_CASE("PeerRegistry::most_trusted_peer — returns highest-trust peer",
          "[Phase133][trust]") {
    PeerRegistry r;
    r.add_peer("peer_a", "key_a");
    r.add_peer("peer_b", "key_b");

    // Give peer_a many positive interactions
    for (int i = 0; i < 5; ++i) r.record_interaction("peer_a", 1.0, i);
    // Give peer_b many negative interactions
    for (int i = 0; i < 5; ++i) r.record_interaction("peer_b", 0.0, i);

    const auto best = r.most_trusted_peer();
    REQUIRE(best.has_value() == true);
    REQUIRE(best.value() == "peer_a");
}

TEST_CASE("PeerRegistry::peers_by_trust — ordered descending",
          "[Phase133][trust]") {
    PeerRegistry r;
    r.add_peer("peer_a", "key_a");
    r.add_peer("peer_b", "key_b");
    r.add_peer("peer_c", "key_c");

    for (int i = 0; i < 8; ++i) r.record_interaction("peer_a", 1.0, i); // highest
    for (int i = 0; i < 3; ++i) r.record_interaction("peer_b", 1.0, i); // mid
    // peer_c stays default

    const auto ranked = r.peers_by_trust();
    REQUIRE(ranked.size() == 3);
    REQUIRE(ranked[0] == "peer_a");
}

// ---------------------------------------------------------------------------
// stats
// ---------------------------------------------------------------------------

TEST_CASE("PeerRegistry::stats — reflects interactions",
          "[Phase133][stats]") {
    PeerRegistry r;
    r.add_peer("peer_a", "key_a");
    r.add_peer("peer_b", "key_b");
    r.record_interaction("peer_a", 0.8, 1);
    r.record_interaction("peer_b", 0.6, 2);

    const auto s = r.stats();
    REQUIRE(s.total_peers        == 2);
    REQUIRE(s.total_interactions == 2);
    REQUIRE(s.mean_resonance     == Approx(0.7));
}

// ---------------------------------------------------------------------------
// clear
// ---------------------------------------------------------------------------

TEST_CASE("PeerRegistry::clear — removes all peers and resets counter",
          "[Phase133][clear]") {
    PeerRegistry r;
    r.add_peer("peer_a", "key_a");
    r.record_interaction("peer_a", 0.9, 0);
    r.clear();
    REQUIRE(r.peer_count() == 0);
    REQUIRE(r.stats().total_interactions == 0);
}

// ---------------------------------------------------------------------------
// FIFO eviction at PEER_REGISTRY_MAX
// ---------------------------------------------------------------------------

TEST_CASE("PeerRegistry — FIFO eviction at PEER_REGISTRY_MAX",
          "[Phase133][eviction]") {
    PeerRegistry r;
    // Add exactly PEER_REGISTRY_MAX+1 peers, first one gets evicted
    for (size_t i = 0; i <= PEER_REGISTRY_MAX; ++i) {
        r.add_peer("peer_" + std::to_string(i), "key_" + std::to_string(i),
                   static_cast<uint64_t>(i));
    }
    REQUIRE(r.peer_count() <= PEER_REGISTRY_MAX);
}
