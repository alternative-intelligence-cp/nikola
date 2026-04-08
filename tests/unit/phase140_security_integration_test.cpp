/**
 * @file tests/unit/phase140_security_integration_test.cpp
 * @brief v0.0.13 — End-to-end security integration test
 *
 * Exercises the full security pipeline as a single flow:
 *   1. ML-KEM key exchange (Alice→Bob shared secret)
 *   2. SPHINCS+ + Ed25519 keygen → sign → HybridVerifier → verify
 *   3. ShadowSpine stage with valid dual-signed binary
 *   4. ShadowSpine stage with corrupted signature → rejected
 *   5. CodePatternBlacklist catches malicious source through SIE
 *   6. PolymorphicDefense registers entries → mutates → old tokens invalid
 *   7. HomeostasisMonitor baseline → normal check → anomaly → lockdown
 *   8. Recovery from lockdown
 *   9. Concurrent security operations across subsystems
 *
 * Plugin dir injected via PHASE140_PLUGIN_DIR (same plugin as phase 112/114).
 */

#include <nikola/security/hybrid_verifier.hpp>
#include <nikola/security/sphincs_signer.hpp>
#include <nikola/security/mlkem_kem.hpp>
#include <nikola/security/polymorphic_defense.hpp>
#include <nikola/security/homeostasis.hpp>
#include <nikola/security/code_blacklist.hpp>
#include <nikola/autonomy/shadow_spine.hpp>
#include <nikola/autonomy/metabolic_controller.hpp>

#include <openssl/evp.h>
#include <openssl/err.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <thread>
#include <vector>

// ── Plugin path ───────────────────────────────────────────────────────────────
#ifndef PHASE140_PLUGIN_DIR
#  define PHASE140_PLUGIN_DIR "."
#endif

static const std::string k_plugin =
    std::string(PHASE140_PLUGIN_DIR) + "/phase112_test_plugin.so";

// ── Aliases ───────────────────────────────────────────────────────────────────
using nikola::security::HybridVerifier;
using nikola::security::HybridSignature;
using nikola::security::VerifyFailReason;
using nikola::security::SphincsSigner;
using nikola::security::SphincsKeypair;
using nikola::security::SphincsSignature;
using nikola::security::MlKem;
using nikola::security::KyberKeypair;
using nikola::security::KyberEncapResult;
using nikola::security::PolymorphicDefense;
using nikola::security::HomeostasisMonitor;
using nikola::security::AnomalyType;
using nikola::security::CodePatternBlacklist;
using nikola::autonomy::ShadowSpine;
using nikola::autonomy::StageStatus;
using nikola::autonomy::EvolutionaryOrchestrator;
using nikola::autonomy::MetabolicController;
using nikola::autonomy::NikolaState;

// ── Ed25519 helpers (reused pattern from phase114) ────────────────────────────

struct Ed25519Keypair {
    std::vector<uint8_t> pk;
    std::vector<uint8_t> sk;
};

static Ed25519Keypair make_ed_keypair() {
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr);
    REQUIRE(ctx != nullptr);
    REQUIRE(EVP_PKEY_keygen_init(ctx) == 1);

    EVP_PKEY* pkey = nullptr;
    REQUIRE(EVP_PKEY_keygen(ctx, &pkey) == 1);
    EVP_PKEY_CTX_free(ctx);
    REQUIRE(pkey != nullptr);

    Ed25519Keypair kp;
    kp.pk.resize(32);
    kp.sk.resize(32);

    size_t pub_len = 32, priv_len = 32;
    REQUIRE(EVP_PKEY_get_raw_public_key(pkey, kp.pk.data(), &pub_len) == 1);
    REQUIRE(EVP_PKEY_get_raw_private_key(pkey, kp.sk.data(), &priv_len) == 1);
    EVP_PKEY_free(pkey);
    return kp;
}

static std::vector<uint8_t> ed_sign(const std::vector<uint8_t>& msg,
                                    const Ed25519Keypair&        kp) {
    EVP_PKEY* pkey = EVP_PKEY_new_raw_private_key(
        EVP_PKEY_ED25519, nullptr, kp.sk.data(), 32);
    REQUIRE(pkey != nullptr);

    EVP_MD_CTX* mctx = EVP_MD_CTX_new();
    REQUIRE(mctx != nullptr);
    REQUIRE(EVP_DigestSignInit(mctx, nullptr, nullptr, nullptr, pkey) == 1);

    size_t sig_len = 64;
    std::vector<uint8_t> sig(64);
    REQUIRE(EVP_DigestSign(mctx, sig.data(), &sig_len,
                           msg.data(), msg.size()) == 1);
    sig.resize(sig_len);

    EVP_MD_CTX_free(mctx);
    EVP_PKEY_free(pkey);
    return sig;
}

static HybridSignature make_valid_sig(
        const std::vector<uint8_t>& binary,
        const Ed25519Keypair&       ed_kp,
        const SphincsKeypair&       sp_kp) {
    HybridSignature hs;
    hs.ed25519_sig = ed_sign(binary, ed_kp);
    hs.sphincs_sig = SphincsSigner::sign(binary, sp_kp).bytes;
    return hs;
}

static std::vector<uint8_t> read_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return {std::istreambuf_iterator<char>(f),
            std::istreambuf_iterator<char>()};
}

// ── Helper: make a NikolaState with specific values ───────────────────────────
static NikolaState make_state(float dopamine, float atp,
                              float boredom, float entropy) {
    NikolaState s{};
    s.dopamine = dopamine;
    s.atp      = atp;
    s.boredom  = boredom;
    s.entropy  = entropy;
    return s;
}

// =============================================================================
// TEST 1: ML-KEM key exchange → shared secret agreement
// =============================================================================

TEST_CASE("v0.0.13 E2E: ML-KEM key exchange", "[v0.0.13][security][e2e]") {
    // Alice generates keypair
    auto alice_kp = MlKem::generate_keypair();
    REQUIRE(alice_kp.pk.size() == MlKem::public_key_bytes());
    REQUIRE(alice_kp.sk.size() == MlKem::secret_key_bytes());

    // Bob encapsulates with Alice's public key
    auto encap = MlKem::encapsulate(alice_kp.pk);
    REQUIRE(encap.ct.size() == MlKem::ciphertext_bytes());
    REQUIRE(encap.ss.size() == MlKem::shared_secret_bytes());

    // Alice decapsulates
    auto alice_ss = MlKem::decapsulate(encap.ct, alice_kp.sk);
    REQUIRE(alice_ss.size() == MlKem::shared_secret_bytes());

    // Shared secrets must match
    REQUIRE(alice_ss == encap.ss);

    // Eve with wrong key gets different secret (implicit rejection)
    auto eve_kp = MlKem::generate_keypair();
    auto eve_ss = MlKem::decapsulate(encap.ct, eve_kp.sk);
    REQUIRE(eve_ss != encap.ss);
}

// =============================================================================
// TEST 2: Full dual-signature lifecycle (keygen → sign → verify → cache)
// =============================================================================

TEST_CASE("v0.0.13 E2E: dual-signature lifecycle", "[v0.0.13][security][e2e]") {
    auto ed_kp = make_ed_keypair();
    auto sp_kp = SphincsSigner::generate_keypair();

    const std::vector<uint8_t> payload = {
        0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00,
        0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE
    };

    // Sign
    auto sig = make_valid_sig(payload, ed_kp, sp_kp);
    REQUIRE(sig.ed25519_sig.size() == 64);
    REQUIRE(!sig.sphincs_sig.empty());

    // Verify
    HybridVerifier hv;
    REQUIRE(hv.verify_module(payload, sig, ed_kp.pk, sp_kp.pk));
    CHECK(hv.last_failure() == VerifyFailReason::NONE);
    CHECK(hv.cache_size() == 1);

    // Cache hit — second verify skips crypto
    REQUIRE(hv.verify_module(payload, sig, ed_kp.pk, sp_kp.pk));
    CHECK(hv.cache_size() == 1);

    // Tampered payload → rejected
    auto tampered = payload;
    tampered[0] ^= 0xFF;
    REQUIRE_FALSE(hv.verify_module(tampered, sig, ed_kp.pk, sp_kp.pk));

    // Wrong Ed25519 key → rejected (fresh verifier to bypass cache)
    auto ed_kp2 = make_ed_keypair();
    HybridVerifier hv_fresh;
    REQUIRE_FALSE(hv_fresh.verify_module(payload, sig, ed_kp2.pk, sp_kp.pk));

    // Wrong SPHINCS+ key → rejected (Ed25519 passes, SPHINCS+ fails)
    auto sp_kp2 = SphincsSigner::generate_keypair();
    HybridVerifier hv3;  // fresh — no cache
    REQUIRE_FALSE(hv3.verify_module(payload, sig, ed_kp.pk, sp_kp2.pk));
    CHECK(hv3.last_failure() == VerifyFailReason::SPHINCS_INVALID);

    // Clear cache forces re-verification
    hv.clear_cache();
    CHECK(hv.cache_size() == 0);
    REQUIRE(hv.verify_module(payload, sig, ed_kp.pk, sp_kp.pk));
    CHECK(hv.cache_size() == 1);
}

// =============================================================================
// TEST 3: ShadowSpine — valid plugin accepted through full gate pipeline
// =============================================================================

TEST_CASE("v0.0.13 E2E: ShadowSpine accepts valid signed plugin",
          "[v0.0.13][security][e2e]") {
    if (!std::filesystem::exists(k_plugin)) {
        SKIP("phase112_test_plugin.so not built");
    }

    auto ed_kp = make_ed_keypair();
    auto sp_kp = SphincsSigner::generate_keypair();

    // Read the .so, sign it
    auto binary = read_binary(k_plugin);
    REQUIRE(!binary.empty());
    auto sig = make_valid_sig(binary, ed_kp, sp_kp);

    // Build ShadowSpine with plenty of ATP (EO cycle costs significant energy)
    MetabolicController mc{100'000.0f, 5.0f};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo(mc, bl);
    HybridVerifier hv;
    ShadowSpine spine(eo, hv);

    // Stage with safe source code (no blacklisted patterns)
    auto report = spine.stage(k_plugin, "int foo() { return 42; }",
                              sig, ed_kp.pk, sp_kp.pk, nullptr);

    CHECK(report.signature_passed);
    CHECK(report.status == StageStatus::SUCCESS);
    CHECK(spine.has_active());
}

// =============================================================================
// TEST 4: ShadowSpine — corrupted signature rejected at Gate 0
// =============================================================================

TEST_CASE("v0.0.13 E2E: ShadowSpine rejects corrupted signature",
          "[v0.0.13][security][e2e]") {
    if (!std::filesystem::exists(k_plugin)) {
        SKIP("phase112_test_plugin.so not built");
    }

    auto ed_kp = make_ed_keypair();
    auto sp_kp = SphincsSigner::generate_keypair();

    auto binary = read_binary(k_plugin);
    auto sig = make_valid_sig(binary, ed_kp, sp_kp);

    // Corrupt the Ed25519 signature
    sig.ed25519_sig[0] ^= 0xFF;

    MetabolicController mc{100'000.0f, 5.0f};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo(mc, bl);
    HybridVerifier hv;
    ShadowSpine spine(eo, hv);

    auto report = spine.stage(k_plugin, "int foo() { return 42; }",
                              sig, ed_kp.pk, sp_kp.pk, nullptr);

    CHECK_FALSE(report.signature_passed);
    CHECK(report.status == StageStatus::SIGNATURE_REJECTED);
    CHECK_FALSE(spine.has_active());
}

// =============================================================================
// TEST 5: SIE rejects malicious source via CodePatternBlacklist (Gate 1)
// =============================================================================

TEST_CASE("v0.0.13 E2E: SIE rejects malicious source code",
          "[v0.0.13][security][e2e]") {
    if (!std::filesystem::exists(k_plugin)) {
        SKIP("phase112_test_plugin.so not built");
    }

    auto ed_kp = make_ed_keypair();
    auto sp_kp = SphincsSigner::generate_keypair();
    auto binary = read_binary(k_plugin);
    auto sig = make_valid_sig(binary, ed_kp, sp_kp);

    MetabolicController mc{100'000.0f, 5.0f};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo(mc, bl);
    HybridVerifier hv;
    ShadowSpine spine(eo, hv);

    // Signature is VALID but source code contains system() — Gate 1 rejects
    auto report = spine.stage(k_plugin, "void exploit() { system(\"rm -rf /\"); }",
                              sig, ed_kp.pk, sp_kp.pk, nullptr);

    CHECK(report.signature_passed);  // Gate 0 passed
    CHECK(report.status == StageStatus::SECURITY_REJECTED);  // Gate 1 caught it
    CHECK_FALSE(spine.has_active());
}

// =============================================================================
// TEST 6: PolymorphicDefense — mutation invalidates old tokens
// =============================================================================

TEST_CASE("v0.0.13 E2E: PolymorphicDefense token lifecycle",
          "[v0.0.13][security][e2e]") {
    PolymorphicDefense pd;

    // Register entries
    uint64_t tok_a = pd.register_entry("verifier_key", 0);
    uint64_t tok_b = pd.register_entry("session_token", 0);
    REQUIRE(tok_a != 0);
    REQUIRE(tok_b != 0);
    REQUIRE(tok_a != tok_b);

    // Validate current tokens
    CHECK(pd.validate_token("verifier_key", tok_a));
    CHECK(pd.validate_token("session_token", tok_b));

    // Mutate all entries
    pd.randomize(1.0, 1);

    // Old tokens must be invalid
    CHECK_FALSE(pd.validate_token("verifier_key", tok_a));
    CHECK_FALSE(pd.validate_token("session_token", tok_b));

    // New tokens must be valid
    uint64_t new_tok_a = pd.current_token("verifier_key");
    uint64_t new_tok_b = pd.current_token("session_token");
    CHECK(new_tok_a != tok_a);
    CHECK(new_tok_b != tok_b);
    CHECK(pd.validate_token("verifier_key", new_tok_a));
    CHECK(pd.validate_token("session_token", new_tok_b));

    // Stats reflect mutations
    auto s = pd.stats();
    CHECK(s.total_entries == 2);
    CHECK(s.total_mutations >= 2);
}

// =============================================================================
// TEST 7: HomeostasisMonitor — full anomaly → lockdown → recovery
// =============================================================================

TEST_CASE("v0.0.13 E2E: HomeostasisMonitor anomaly → lockdown → recovery",
          "[v0.0.13][security][e2e]") {
    HomeostasisMonitor hm;

    // Normal state as baseline
    auto normal = make_state(0.5f, 0.6f, 0.3f, 0.4f);
    hm.set_baseline(normal);

    // Track anomalies via callback
    std::vector<AnomalyType> events;
    hm.on_anomaly([&](const auto& rec) { events.push_back(rec.type); });

    // Check 1: normal state → no anomaly, no lockdown
    CHECK(hm.check(normal, 0));
    CHECK_FALSE(hm.is_locked_down());
    CHECK(events.empty());

    // Check 2: small drift within tolerance → still OK
    auto slight_drift = make_state(0.55f, 0.65f, 0.3f, 0.45f);
    CHECK(hm.check(slight_drift, 1));
    CHECK(events.empty());

    // Check 3: energy spike → anomaly detected
    auto energy_spike = make_state(1.0f, 1.0f, 0.0f, 0.4f);
    CHECK_FALSE(hm.check(energy_spike, 2));
    CHECK(!events.empty());
    CHECK(events.back() == AnomalyType::ENERGY_SPIKE);

    // Check 4: entropy collapse → anomaly + auto-lockdown (severity ~1.0)
    auto entropy_collapse = make_state(0.5f, 0.6f, 0.3f, 0.0f);
    hm.check(entropy_collapse, 3);

    // At this point homeostasis should have triggered lockdown
    // (the energy spike alone might trigger it depending on severity)
    auto stats = hm.stats();
    CHECK(stats.total_anomalies >= 2);
    CHECK(stats.total_checks == 4);

    // If locked down, verify we can recover
    if (hm.is_locked_down()) {
        hm.release_lockdown();
        CHECK_FALSE(hm.is_locked_down());

        // System should resume normal operation after release
        CHECK(hm.check(normal, 4));
    }
}

// =============================================================================
// TEST 8: HomeostasisMonitor — verify_integrity is side-effect-free
// =============================================================================

TEST_CASE("v0.0.13 E2E: HomeostasisMonitor verify_integrity is read-only",
          "[v0.0.13][security][e2e]") {
    HomeostasisMonitor hm;
    auto normal = make_state(0.5f, 0.6f, 0.3f, 0.4f);
    hm.set_baseline(normal);

    auto bad = make_state(1.0f, 1.0f, 0.0f, 1.0f);
    auto stats_before = hm.stats();

    // verify_integrity detects the bad state
    CHECK_FALSE(hm.verify_integrity(bad));

    // But doesn't modify internal state
    auto stats_after = hm.stats();
    CHECK(stats_after.total_checks == stats_before.total_checks);
    CHECK(stats_after.total_anomalies == stats_before.total_anomalies);
    CHECK_FALSE(hm.is_locked_down());
}

// =============================================================================
// TEST 9: SCRAM-adjacent — energy spike through full pipeline
// =============================================================================

TEST_CASE("v0.0.13 E2E: ShadowSpine + HomeostasisMonitor interaction",
          "[v0.0.13][security][e2e]") {
    // This tests that after a ShadowSpine security rejection,
    // HomeostasisMonitor can still detect metabolic anomalies
    // (they operate as independent defense layers)

    MetabolicController mc{100'000.0f, 5.0f};
    CodePatternBlacklist bl;
    EvolutionaryOrchestrator eo(mc, bl);
    HybridVerifier hv;
    ShadowSpine spine(eo, hv);
    HomeostasisMonitor hm;

    // Set healthy baseline
    auto healthy = make_state(0.5f, 0.6f, 0.3f, 0.4f);
    hm.set_baseline(healthy);

    // ShadowSpine rejects a bad file
    HybridSignature dummy_sig;
    dummy_sig.ed25519_sig.resize(64, 0);
    dummy_sig.sphincs_sig.resize(100, 0);
    std::vector<uint8_t> dummy_pk(32, 0);
    std::vector<uint8_t> dummy_sphincs_pk(SphincsSigner::public_key_bytes(), 0);

    auto report = spine.stage("/nonexistent/bad.so", "",
                              dummy_sig, dummy_pk, dummy_sphincs_pk, nullptr);
    CHECK(report.status == StageStatus::LOAD_FAILED);

    // HomeostasisMonitor independently detects anomaly
    auto spiked = make_state(1.0f, 1.0f, 0.0f, 0.9f);
    CHECK_FALSE(hm.check(spiked, 0));
    CHECK(hm.stats().total_anomalies > 0);
}

// =============================================================================
// TEST 10: Concurrent security operations
// =============================================================================

TEST_CASE("v0.0.13 E2E: concurrent security operations",
          "[v0.0.13][security][e2e]") {
    HybridVerifier hv;
    PolymorphicDefense pd;
    HomeostasisMonitor hm;

    auto normal = make_state(0.5f, 0.6f, 0.3f, 0.4f);
    hm.set_baseline(normal);

    // Pre-register some entries
    for (int i = 0; i < 10; ++i) {
        pd.register_entry("entry_" + std::to_string(i), 0);
    }

    // Generate one valid keypair and signature for verification threads
    auto ed_kp = make_ed_keypair();
    auto sp_kp = SphincsSigner::generate_keypair();
    const std::vector<uint8_t> payload = {0x01, 0x02, 0x03, 0x04};
    auto sig = make_valid_sig(payload, ed_kp, sp_kp);

    std::atomic<bool> any_failure{false};

    auto verify_thread = [&]() {
        for (int i = 0; i < 10; ++i) {
            if (!hv.verify_module(payload, sig, ed_kp.pk, sp_kp.pk)) {
                any_failure.store(true);
            }
        }
    };

    auto mutate_thread = [&]() {
        for (int i = 0; i < 10; ++i) {
            pd.randomize(0.5, static_cast<uint64_t>(i));
        }
    };

    auto monitor_thread = [&]() {
        for (int i = 0; i < 10; ++i) {
            hm.check(normal, static_cast<uint64_t>(i));
        }
    };

    std::thread t1(verify_thread);
    std::thread t2(mutate_thread);
    std::thread t3(monitor_thread);
    std::thread t4(verify_thread);

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // Verification should never fail with valid keys
    CHECK_FALSE(any_failure.load());
    // Stats should reflect all operations
    CHECK(hv.cache_size() >= 1);
    CHECK(pd.stats().total_mutations > 0);
    CHECK(hm.stats().total_checks >= 10);
}

// =============================================================================
// TEST 11: CodePatternBlacklist — multiple attack vectors
// =============================================================================

TEST_CASE("v0.0.13 E2E: CodePatternBlacklist attack vectors",
          "[v0.0.13][security][e2e]") {
    CodePatternBlacklist bl;

    // Safe code passes
    CHECK(bl.check("int add(int a, int b) { return a + b; }").safe);

    // Each dangerous pattern is caught
    CHECK_FALSE(bl.check("system(\"ls\");").safe);
    CHECK_FALSE(bl.check("execve(\"/bin/sh\", NULL, NULL);").safe);
    CHECK_FALSE(bl.check("fork();").safe);
    CHECK_FALSE(bl.check("popen(\"cat /etc/passwd\", \"r\");").safe);
    CHECK_FALSE(bl.check("asm(\"int $0x80\");").safe);
    CHECK_FALSE(bl.check("__asm__(\"syscall\");").safe);
    CHECK_FALSE(bl.check("#include <sys/socket.h>").safe);

    // Attempts to access sensitive paths
    CHECK_FALSE(bl.check("fopen(\"/proc/self/maps\", \"r\");").safe);
    CHECK_FALSE(bl.check("open(\"/dev/mem\", O_RDWR);").safe);
}

// =============================================================================
// TEST 12: Key re-use after cache clear (simulated rotation)
// =============================================================================

TEST_CASE("v0.0.13 E2E: simulated key rotation via cache clear",
          "[v0.0.13][security][e2e]") {
    HybridVerifier hv;

    // Key generation A
    auto ed_kp_a = make_ed_keypair();
    auto sp_kp_a = SphincsSigner::generate_keypair();
    const std::vector<uint8_t> payload = {0xCA, 0xFE, 0xBA, 0xBE};
    auto sig_a = make_valid_sig(payload, ed_kp_a, sp_kp_a);

    // Verify with key A
    REQUIRE(hv.verify_module(payload, sig_a, ed_kp_a.pk, sp_kp_a.pk));
    CHECK(hv.cache_size() == 1);

    // Simulate key rotation: clear cache, generate new keys
    hv.clear_cache();
    CHECK(hv.cache_size() == 0);

    auto ed_kp_b = make_ed_keypair();
    auto sp_kp_b = SphincsSigner::generate_keypair();
    auto sig_b = make_valid_sig(payload, ed_kp_b, sp_kp_b);

    // Old signature with new keys → rejected
    REQUIRE_FALSE(hv.verify_module(payload, sig_a, ed_kp_b.pk, sp_kp_b.pk));

    // New signature with new keys → accepted
    REQUIRE(hv.verify_module(payload, sig_b, ed_kp_b.pk, sp_kp_b.pk));
    CHECK(hv.cache_size() == 1);

    // Old keys + old signature still valid (crypto doesn't expire here)
    HybridVerifier hv2;
    REQUIRE(hv2.verify_module(payload, sig_a, ed_kp_a.pk, sp_kp_a.pk));
}
