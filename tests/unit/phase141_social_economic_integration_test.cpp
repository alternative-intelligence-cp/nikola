// ============================================================
// Phase 141 — v0.0.14 Social/Economic Integration
// tests/unit/phase141_social_economic_integration_test.cpp
//
// End-to-end integration tests verifying the social and economic
// layers work together as a coherent system.
//
// Test domains:
//   §1  E2E economic transaction cycle
//   §2  Multi-wallet economy (buyer/seller fund transfer)
//   §3  Protobuf NES serialization roundtrip
//   §4  Protobuf IRSP serialization roundtrip
//   §5  SocialMembrane: high trust → full permeability
//   §6  SocialMembrane: high dissonance → near-zero permeability
//   §7  PeerRegistry trust dynamics over repeated interactions
//   §8  3-agent simulation: wallets + membranes + marketplace
//   §9  Multi-agent trust convergence
//   §10 Multi-agent economic flow (service purchase across agents)
//   §11 Peer handshake over CurveZMQ (2 threads, Ironhouse)
//   §12 PeerAnnouncement serialization + registry integration
// ============================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/economy/wallet.hpp>
#include <nikola/economy/marketplace.hpp>
#include <nikola/social/membrane.hpp>
#include <nikola/social/peer_registry.hpp>
#include <nikola/social/peer_handshake.hpp>
#include <nikola/security/ironhouse.hpp>

#include <nes.pb.h>
#include <irsp.pb.h>

#include <complex>
#include <thread>
#include <vector>
#include <string>
#include <cmath>

using Catch::Approx;
using namespace nikola::economy;
using namespace nikola::social;

// ── §1 E2E economic transaction cycle ─────────────────────────────────────────

TEST_CASE("§1 Full transaction cycle: list → discover → purchase → execute → balance",
          "[integration][economy]") {
    SimulatedWallet buyer_w, seller_w;
    buyer_w.derive_identity("buyer-seed-001");
    seller_w.derive_identity("seller-seed-001");

    // Fund the buyer
    buyer_w.credit(10'000);
    REQUIRE(buyer_w.get_balance_wei() == 10'000);

    NeuralMarketplace market;

    // Seller lists a service
    ServiceListing svc;
    svc.service_id        = "summarize:v1";
    svc.provider_address  = seller_w.get_address();
    svc.description       = "Text summarization service";
    svc.price_wei         = 500;
    svc.quality_score     = 0.8;
    market.list_service(svc);

    // Buyer discovers via keyword
    auto results = market.browse_services("summarization");
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].service_id == "summarize:v1");

    // Buyer purchases
    auto tx_hash = market.purchase_service(
        "summarize:v1", buyer_w.get_address(), 500, /*tick=*/1);
    REQUIRE_FALSE(tx_hash.empty());
    buyer_w.debit(500);

    // Execute the service
    auto result = market.execute_service(tx_hash, "Hello world document");
    REQUIRE_FALSE(result.empty());

    // Seller receives payment
    seller_w.credit(500);

    // Verify final balances
    REQUIRE(buyer_w.get_balance_wei()  == 9'500);
    REQUIRE(seller_w.get_balance_wei() == 500);

    // Verify transaction status
    auto* tx = market.find_transaction(tx_hash);
    REQUIRE(tx != nullptr);
    REQUIRE(tx->status == TxStatus::EXECUTED);
}

// ── §2 Multi-wallet economy ──────────────────────────────────────────────────

TEST_CASE("§2 Multi-wallet economy: 3 wallets, service exchange",
          "[integration][economy]") {
    SimulatedWallet w_a, w_b, w_c;
    w_a.derive_identity("agent-A");
    w_b.derive_identity("agent-B");
    w_c.derive_identity("agent-C");

    w_a.credit(5'000);
    w_b.credit(5'000);
    w_c.credit(5'000);

    NeuralMarketplace market;

    // A lists compute, B lists analysis
    ServiceListing svc_a{.service_id = "compute:A", .provider_address = w_a.get_address(),
                         .description = "GPU compute", .price_wei = 200};
    ServiceListing svc_b{.service_id = "analysis:B", .provider_address = w_b.get_address(),
                         .description = "Data analysis", .price_wei = 300};
    market.list_service(svc_a);
    market.list_service(svc_b);

    // C buys from A
    auto tx1 = market.purchase_service("compute:A", w_c.get_address(), 200, 1);
    REQUIRE_FALSE(tx1.empty());
    w_c.debit(200);
    w_a.credit(200);
    market.execute_service(tx1, "input-data");

    // C buys from B
    auto tx2 = market.purchase_service("analysis:B", w_c.get_address(), 300, 2);
    REQUIRE_FALSE(tx2.empty());
    w_c.debit(300);
    w_b.credit(300);
    market.execute_service(tx2, "analysis-input");

    REQUIRE(w_a.get_balance_wei() == 5'200);
    REQUIRE(w_b.get_balance_wei() == 5'300);
    REQUIRE(w_c.get_balance_wei() == 4'500);

    auto stats = market.stats();
    REQUIRE(stats.transaction_count == 2);
    REQUIRE(stats.executed_count == 2);
    REQUIRE(stats.total_volume_wei == 500);
}

// ── §3 Protobuf NES roundtrip ────────────────────────────────────────────────

TEST_CASE("§3 Protobuf NES: ServiceAdvertisement serialization roundtrip",
          "[integration][proto]") {
    ::nikola::economy::ServiceAdvertisement ad;
    ad.set_service_id("compute:gpu:rtx3090");
    ad.set_provider_address("0xdeadbeef12345678");
    ad.set_name("GPU Compute");
    ad.set_description("High-performance CUDA computation");
    ad.add_tags("gpu");
    ad.add_tags("cuda");
    ad.set_price_wei(1000);
    ad.set_quality_score(0.95);
    ad.set_estimated_time_ms(50);
    ad.set_execution_count(42);

    std::string wire = ad.SerializeAsString();
    REQUIRE_FALSE(wire.empty());

    ::nikola::economy::ServiceAdvertisement ad2;
    REQUIRE(ad2.ParseFromString(wire));
    REQUIRE(ad2.service_id() == "compute:gpu:rtx3090");
    REQUIRE(ad2.provider_address() == "0xdeadbeef12345678");
    REQUIRE(ad2.name() == "GPU Compute");
    REQUIRE(ad2.tags_size() == 2);
    REQUIRE(ad2.tags(0) == "gpu");
    REQUIRE(ad2.tags(1) == "cuda");
    REQUIRE(ad2.price_wei() == 1000);
    REQUIRE(ad2.quality_score() == Approx(0.95));
    REQUIRE(ad2.execution_count() == 42);
}

TEST_CASE("§3b Protobuf NES: PaymentChannel serialization roundtrip",
          "[integration][proto]") {
    ::nikola::economy::PaymentChannel ch;
    ch.set_channel_id("ch-001");
    ch.set_participant_a("0xAAAA");
    ch.set_participant_b("0xBBBB");
    ch.set_balance_a_wei(5000);
    ch.set_balance_b_wei(3000);
    ch.set_nonce(7);
    ch.set_closed(false);

    std::string wire = ch.SerializeAsString();
    ::nikola::economy::PaymentChannel ch2;
    REQUIRE(ch2.ParseFromString(wire));
    REQUIRE(ch2.channel_id() == "ch-001");
    REQUIRE(ch2.balance_a_wei() == 5000);
    REQUIRE(ch2.balance_b_wei() == 3000);
    REQUIRE(ch2.nonce() == 7);
    REQUIRE_FALSE(ch2.closed());
}

// ── §4 Protobuf IRSP roundtrip ──────────────────────────────────────────────

TEST_CASE("§4 Protobuf IRSP: ResonantPacket serialization roundtrip",
          "[integration][proto]") {
    ::nikola::social::ResonantPacket pkt;
    pkt.set_text_payload("Hello from Nikola");
    pkt.set_sender_id("nikola-alpha");
    pkt.set_session_key("sess-42");
    pkt.set_protocol_version("1.0");
    // 9 emitter components (golden ratio array)
    for (int i = 0; i < 9; ++i) {
        pkt.add_emitter_real(std::cos(i * 1.618));
        pkt.add_emitter_imag(std::sin(i * 1.618));
    }
    pkt.add_intent_vector(0.7);
    pkt.add_intent_vector(0.3);
    pkt.set_timestamp(1712345678000LL);

    std::string wire = pkt.SerializeAsString();
    REQUIRE_FALSE(wire.empty());

    ::nikola::social::ResonantPacket pkt2;
    REQUIRE(pkt2.ParseFromString(wire));
    REQUIRE(pkt2.sender_id() == "nikola-alpha");
    REQUIRE(pkt2.emitter_real_size() == 9);
    REQUIRE(pkt2.emitter_imag_size() == 9);
    REQUIRE(pkt2.intent_vector_size() == 2);
    REQUIRE(pkt2.timestamp() == 1712345678000LL);
}

TEST_CASE("§4b Protobuf IRSP: EmpathySignal serialization roundtrip",
          "[integration][proto]") {
    ::nikola::social::EmpathySignal sig;
    sig.set_session_key("sess-42");
    sig.set_responding_to_sender("nikola-alpha");
    sig.set_resonance_score(0.85);
    sig.set_dissonance_detected(0.05);
    sig.set_feedback_text("Strong resonance on shared concept");

    std::string wire = sig.SerializeAsString();
    ::nikola::social::EmpathySignal sig2;
    REQUIRE(sig2.ParseFromString(wire));
    REQUIRE(sig2.resonance_score() == Approx(0.85));
    REQUIRE(sig2.dissonance_detected() == Approx(0.05));
    REQUIRE(sig2.feedback_text() == "Strong resonance on shared concept");
}

// ── §5 High trust → full permeability ────────────────────────────────────────

TEST_CASE("§5 SocialMembrane: high trust, low dissonance → near-full pass-through",
          "[integration][social]") {
    SocialMembrane mem;
    mem.set_trust(1.0);
    mem.set_dissonance(0.01);

    // Permeability should be very high
    double p = SocialMembrane::compute_permeability(1.0, 0.01);
    REQUIRE(p == Approx(1.0).margin(0.01));  // clamped to MAX_PERMEABILITY

    // Filtering: friend wave should dominate
    std::complex<double> self_wave(1.0, 0.0);
    std::complex<double> friend_wave(0.0, 1.0);
    auto filtered = mem.filter_incoming(friend_wave, self_wave);

    // With p ≈ 1.0, result ≈ friend_wave
    REQUIRE(filtered.real() == Approx(0.0).margin(0.02));
    REQUIRE(filtered.imag() == Approx(1.0).margin(0.02));
}

// ── §6 High dissonance → filtering ──────────────────────────────────────────

TEST_CASE("§6 SocialMembrane: low trust, high dissonance → near-zero permeability",
          "[integration][social]") {
    SocialMembrane mem;
    mem.set_trust(0.05);
    mem.set_dissonance(1.0);

    double p = SocialMembrane::compute_permeability(0.05, 1.0);
    REQUIRE(p < 0.10);  // Very low permeability

    std::complex<double> self_wave(1.0, 0.0);
    std::complex<double> friend_wave(0.0, 1.0);
    auto filtered = mem.filter_incoming(friend_wave, self_wave);

    // With p ≈ 0, result ≈ self_wave
    REQUIRE(filtered.real() == Approx(1.0).margin(0.15));
    REQUIRE(filtered.imag() == Approx(0.0).margin(0.15));
}

// ── §7 PeerRegistry trust dynamics ──────────────────────────────────────────

TEST_CASE("§7 PeerRegistry: 50 positive interactions drive trust toward 1.0",
          "[integration][social]") {
    PeerRegistry reg;
    reg.add_peer("peer-alpha", "key-alpha-z85-0000000000000000000000000000", 0);

    double initial_trust = reg.get_membrane("peer-alpha")->get_trust();

    // 50 positive interactions (resonance > 0.5)
    for (int i = 1; i <= 50; ++i) {
        reg.record_interaction("peer-alpha", 0.9, static_cast<uint64_t>(i));
    }

    double final_trust = reg.get_membrane("peer-alpha")->get_trust();
    REQUIRE(final_trust > initial_trust);
    REQUIRE(final_trust > 0.90);  // Should be very high after 50 positives

    // Verify interaction count
    auto* info = reg.find_peer("peer-alpha");
    REQUIRE(info->interaction_count == 50);
}

// ── §8 3-agent simulation ───────────────────────────────────────────────────

struct SimAgent {
    std::string            id;
    SimulatedWallet        wallet;
    PeerRegistry           peers;
    NeuralMarketplace      market;  // Shared conceptually; local copy for test
};

TEST_CASE("§8 Three-agent simulation: wallets + membranes + marketplace",
          "[integration][multiagent]") {
    // Create 3 agents
    SimAgent agents[3];
    agents[0].id = "nikola-alpha";
    agents[1].id = "nikola-beta";
    agents[2].id = "nikola-gamma";

    for (auto& a : agents) {
        a.wallet.derive_identity(a.id + "-seed");
        a.wallet.credit(10'000);
    }

    // Each agent registers the other two as peers
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i == j) continue;
            agents[i].peers.add_peer(
                agents[j].id,
                "pubkey-" + agents[j].id,
                /*tick=*/0);
        }
    }

    // Verify peer registries
    for (auto& a : agents) {
        REQUIRE(a.peers.peer_count() == 2);
        REQUIRE(a.wallet.get_balance_wei() == 10'000);
    }

    // Shared marketplace (simulating network-visible listings)
    NeuralMarketplace shared_market;

    // Alpha lists a service
    ServiceListing svc_alpha;
    svc_alpha.service_id       = "compute:alpha";
    svc_alpha.provider_address = agents[0].wallet.get_address();
    svc_alpha.description      = "Alpha compute service";
    svc_alpha.price_wei        = 100;
    shared_market.list_service(svc_alpha);

    // Beta lists a service
    ServiceListing svc_beta;
    svc_beta.service_id       = "analysis:beta";
    svc_beta.provider_address = agents[1].wallet.get_address();
    svc_beta.description      = "Beta analysis service";
    svc_beta.price_wei        = 150;
    shared_market.list_service(svc_beta);

    REQUIRE(shared_market.listing_count() == 2);

    // Gamma purchases from both
    auto tx1 = shared_market.purchase_service(
        "compute:alpha", agents[2].wallet.get_address(), 100, 1);
    REQUIRE_FALSE(tx1.empty());
    agents[2].wallet.debit(100);
    agents[0].wallet.credit(100);
    shared_market.execute_service(tx1, "gamma-input-1");

    auto tx2 = shared_market.purchase_service(
        "analysis:beta", agents[2].wallet.get_address(), 150, 2);
    REQUIRE_FALSE(tx2.empty());
    agents[2].wallet.debit(150);
    agents[1].wallet.credit(150);
    shared_market.execute_service(tx2, "gamma-input-2");

    // Verify balances
    REQUIRE(agents[0].wallet.get_balance_wei() == 10'100);
    REQUIRE(agents[1].wallet.get_balance_wei() == 10'150);
    REQUIRE(agents[2].wallet.get_balance_wei() ==  9'750);

    // Record positive interactions (from successful transactions)
    agents[2].peers.record_interaction("nikola-alpha", 0.85, 1);
    agents[2].peers.record_interaction("nikola-beta",  0.80, 2);
    agents[0].peers.record_interaction("nikola-gamma", 0.85, 1);
    agents[1].peers.record_interaction("nikola-gamma", 0.80, 2);

    // Verify trust increased for gamma's peers
    auto* mem_alpha = agents[2].peers.get_membrane("nikola-alpha");
    auto* mem_beta  = agents[2].peers.get_membrane("nikola-beta");
    REQUIRE(mem_alpha->get_trust() > 0.5);
    REQUIRE(mem_beta->get_trust() > 0.5);
}

// ── §9 Multi-agent trust convergence ────────────────────────────────────────

TEST_CASE("§9 Multi-agent: repeated positive interactions → trust convergence",
          "[integration][multiagent]") {
    PeerRegistry reg_a, reg_b;
    reg_a.add_peer("B", "key-B", 0);
    reg_b.add_peer("A", "key-A", 0);

    // 30 rounds of mutual positive interactions
    for (uint64_t tick = 1; tick <= 30; ++tick) {
        reg_a.record_interaction("B", 0.85, tick);
        reg_b.record_interaction("A", 0.90, tick);
    }

    double trust_a_of_b = reg_a.get_membrane("B")->get_trust();
    double trust_b_of_a = reg_b.get_membrane("A")->get_trust();

    REQUIRE(trust_a_of_b > 0.85);
    REQUIRE(trust_b_of_a > 0.85);

    // Both should have high permeability
    double perm_a = reg_a.get_membrane("B")->get_permeability();
    double perm_b = reg_b.get_membrane("A")->get_permeability();
    REQUIRE(perm_a > 0.5);
    REQUIRE(perm_b > 0.5);
}

// ── §10 Multi-agent economic flow ───────────────────────────────────────────

TEST_CASE("§10 Multi-agent economic flow: 10 transactions, balance conservation",
          "[integration][multiagent]") {
    SimulatedWallet wallets[3];
    wallets[0].derive_identity("econ-A");
    wallets[1].derive_identity("econ-B");
    wallets[2].derive_identity("econ-C");

    const uint64_t initial = 10'000;
    for (auto& w : wallets) w.credit(initial);

    NeuralMarketplace market;

    ServiceListing svc;
    svc.service_id       = "svc:A";
    svc.provider_address = wallets[0].get_address();
    svc.description      = "Service A";
    svc.price_wei        = 100;
    market.list_service(svc);

    // 10 purchases from B→A
    for (int i = 1; i <= 10; ++i) {
        auto tx = market.purchase_service(
            "svc:A", wallets[1].get_address(), 100, static_cast<uint64_t>(i));
        REQUIRE_FALSE(tx.empty());
        wallets[1].debit(100);
        wallets[0].credit(100);
        market.execute_service(tx, "input-" + std::to_string(i));
    }

    // Balance conservation: total should still be 3 × initial
    uint64_t total = wallets[0].get_balance_wei()
                   + wallets[1].get_balance_wei()
                   + wallets[2].get_balance_wei();
    REQUIRE(total == 3 * initial);

    // A gained 1000, B lost 1000, C unchanged
    REQUIRE(wallets[0].get_balance_wei() == 11'000);
    REQUIRE(wallets[1].get_balance_wei() ==  9'000);
    REQUIRE(wallets[2].get_balance_wei() == 10'000);

    REQUIRE(market.stats().executed_count == 10);
}

// ── §11 Peer handshake over CurveZMQ ────────────────────────────────────────

TEST_CASE("§11 Peer handshake: two threads exchange PeerAnnouncements over CurveZMQ",
          "[integration][peer][zmq]") {
    auto server_kp = nikola::security::generate_ironhouse_keypair();
    auto client_kp = nikola::security::generate_ironhouse_keypair();
    REQUIRE(server_kp.valid());
    REQUIRE(client_kp.valid());

    zmq::context_t ctx(1);
    const std::string endpoint = "tcp://127.0.0.1:45199";

    PeerRegistry server_reg, client_reg;
    HandshakeResult server_result, client_result;

    // Server thread
    std::thread server_thread([&]() {
        server_result = accept_handshake(
            ctx, server_kp, endpoint, server_reg, "server-nikola", 10000);
    });

    // Small delay for bind
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Client thread (runs on main)
    client_result = initiate_handshake(
        ctx, client_kp, server_kp.pub(), endpoint, client_reg, "client-nikola", 10000);

    server_thread.join();

    // Both should succeed
    REQUIRE(server_result.success);
    REQUIRE(client_result.success);

    // Server sees client
    REQUIRE(server_result.peer_id == "client-nikola");
    REQUIRE(server_reg.has_peer("client-nikola"));

    // Client sees server
    REQUIRE(client_result.peer_id == "server-nikola");
    REQUIRE(client_reg.has_peer("server-nikola"));
}

// ── §12 PeerAnnouncement serialization + registry integration ───────────────

TEST_CASE("§12 PeerAnnouncement: serialize → deserialize → register in PeerRegistry",
          "[integration][peer]") {
    // Serialize
    auto wire = serialize_announcement(
        "nikola-42", "z85pubkey000000000000000000000000000000ab",
        "tcp://10.0.0.1:9876",
        {"compute", "analysis", "summarize"});
    REQUIRE_FALSE(wire.empty());

    // Deserialize
    std::string pid, pkey, ep;
    std::vector<std::string> caps;
    REQUIRE(deserialize_announcement(wire, pid, pkey, ep, caps));
    REQUIRE(pid == "nikola-42");
    REQUIRE(pkey == "z85pubkey000000000000000000000000000000ab");
    REQUIRE(ep == "tcp://10.0.0.1:9876");
    REQUIRE(caps.size() == 3);
    REQUIRE(caps[0] == "compute");
    REQUIRE(caps[1] == "analysis");
    REQUIRE(caps[2] == "summarize");

    // Register in PeerRegistry
    PeerRegistry reg;
    reg.add_peer(pid, pkey, 1);
    REQUIRE(reg.has_peer("nikola-42"));
    auto* info = reg.find_peer("nikola-42");
    REQUIRE(info->public_key_z85 == pkey);
}
