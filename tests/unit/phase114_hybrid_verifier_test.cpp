/**
 * @file tests/unit/phase114_hybrid_verifier_test.cpp
 * @brief Phase 114 — HybridVerifier + ShadowSpine test suite.
 *
 * Exercises:
 *   HybridVerifier — Ed25519 fast-path + SPHINCS+ slow-path + FNV cache
 *   ShadowSpine    — Gate 0 (signatures) + EO pipeline integration
 *
 * Uses real cryptographic operations:
 *   Ed25519 via libsodium (crypto_sign_keypair / crypto_sign_detached)
 *   SPHINCS+-shake-256f via SphincsSigner (nikola::security)
 *
 * Phase 112 plugin (.so) is reused as the candidate binary for ShadowSpine
 * tests — sign the file's bytes then feed to stage().
 * Plugin dir injected at compile-time via PHASE114_PLUGIN_DIR.
 */

#include <nikola/security/hybrid_verifier.hpp>
#include <nikola/security/sphincs_signer.hpp>
#include <nikola/autonomy/shadow_spine.hpp>
#include <nikola/autonomy/metabolic_controller.hpp>
#include <nikola/security/code_blacklist.hpp>

#include <openssl/evp.h>    // Ed25519 keypair gen + sign (no sodium clash)
#include <openssl/err.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ── Plugin path ───────────────────────────────────────────────────────────────
#ifndef PHASE114_PLUGIN_DIR
#  define PHASE114_PLUGIN_DIR "."
#endif

static const std::string k_plugin =
    std::string(PHASE114_PLUGIN_DIR) + "/phase112_test_plugin.so";
static const std::string k_bad_path = "/nonexistent/phase114/missing.so";

/// Safe source code that passes Gate 1 security scan.
static const std::string k_safe_source =
    "void* nikola_module_factory() { return nullptr; }";

// ── Aliases ───────────────────────────────────────────────────────────────────
using nikola::security::HybridVerifier;
using nikola::security::HybridSignature;
using nikola::security::VerifyFailReason;
using nikola::security::SphincsSigner;
using nikola::security::SphincsKeypair;
using nikola::security::SphincsSignature;
using nikola::autonomy::ShadowSpine;
using nikola::autonomy::StageStatus;
using nikola::autonomy::EvolutionaryOrchestrator;
using nikola::autonomy::MetabolicController;
using nikola::security::CodePatternBlacklist;

// ── Test constants ────────────────────────────────────────────────────────────
static constexpr float k_test_nap_threshold = 5.0f;

// ── Ed25519 helper types / functions (via OpenSSL EVP) ───────────────────────
// We use OpenSSL to avoid the symbol conflict between libsodium's
// crypto_sign_* (Ed25519) and sphincsplus_shake256f's crypto_sign_* (SPHINCS+).

struct Ed25519Keypair {
    std::vector<uint8_t> pk;  // 32 bytes raw public key
    std::vector<uint8_t> sk;  // 32 bytes raw private key
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

// ── SPHINCS+ helper ───────────────────────────────────────────────────────────

static HybridSignature make_valid_sig(
        const std::vector<uint8_t>& binary,
        const Ed25519Keypair&       ed_kp,
        const SphincsKeypair&       sp_kp) {
    HybridSignature hs;
    hs.ed25519_sig = ed_sign(binary, ed_kp);
    hs.sphincs_sig = SphincsSigner::sign(binary, sp_kp).bytes;
    return hs;
}

// ── Read file helper ──────────────────────────────────────────────────────────

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return {std::istreambuf_iterator<char>(f),
            std::istreambuf_iterator<char>()};
}

// ── Sample test binary ────────────────────────────────────────────────────────

static const std::vector<uint8_t> k_test_binary = {
    0x7f, 0x45, 0x4c, 0x46,  // ELF magic
    0x01, 0x02, 0x03, 0x04,
    0xde, 0xad, 0xbe, 0xef
};

static const std::vector<uint8_t> k_other_binary = {
    0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88
};

// =============================================================================
// SECTION 1: HybridVerifier — construction
// =============================================================================

TEST_CASE("HybridVerifier: construction", "[phase114][hybrid_verifier]") {
    SECTION("constructs without throwing") {
        REQUIRE_NOTHROW(HybridVerifier{});
    }

    SECTION("initial cache is empty") {
        HybridVerifier hv;
        CHECK(hv.cache_size() == 0u);
    }

    SECTION("is_cached returns false for any binary initially") {
        HybridVerifier hv;
        CHECK_FALSE(hv.is_cached(k_test_binary));
        CHECK_FALSE(hv.is_cached({}));
    }

    SECTION("initial last_failure is NONE") {
        HybridVerifier hv;
        CHECK(hv.last_failure() == VerifyFailReason::NONE);
    }
}

// =============================================================================
// SECTION 2: HybridVerifier — size validation (fast rejection)
// =============================================================================

TEST_CASE("HybridVerifier: size validation rejects before crypto",
          "[phase114][hybrid_verifier]") {

    HybridVerifier hv;
    const auto ed_kp = make_ed_keypair();
    const auto sp_kp = SphincsSigner::generate_keypair();

    HybridSignature good_sig = make_valid_sig(k_test_binary, ed_kp, sp_kp);

    SECTION("ed_pub too short → ED25519_BAD_PUBKEY") {
        std::vector<uint8_t> bad_pub(16, 0x01);
        CHECK_FALSE(hv.verify_module(k_test_binary, good_sig, bad_pub, sp_kp.pk));
        CHECK(hv.last_failure() == VerifyFailReason::ED25519_BAD_PUBKEY);
    }

    SECTION("ed_pub too long → ED25519_BAD_PUBKEY") {
        std::vector<uint8_t> bad_pub(64, 0x01);
        CHECK_FALSE(hv.verify_module(k_test_binary, good_sig, bad_pub, sp_kp.pk));
        CHECK(hv.last_failure() == VerifyFailReason::ED25519_BAD_PUBKEY);
    }

    SECTION("ed25519_sig wrong size → ED25519_BAD_SIG") {
        HybridSignature bad_sig = good_sig;
        bad_sig.ed25519_sig.resize(16, 0x00);  // too short
        CHECK_FALSE(hv.verify_module(k_test_binary, bad_sig, ed_kp.pk, sp_kp.pk));
        CHECK(hv.last_failure() == VerifyFailReason::ED25519_BAD_SIG);
    }

    SECTION("sphincs_pub wrong size → SPHINCS_BAD_PUBKEY") {
        std::vector<uint8_t> bad_pk(7, 0xAA);  // definitely wrong size
        CHECK_FALSE(hv.verify_module(k_test_binary, good_sig, ed_kp.pk, bad_pk));
        CHECK(hv.last_failure() == VerifyFailReason::SPHINCS_BAD_PUBKEY);
    }

    SECTION("sphincs_sig empty → SPHINCS_BAD_SIG") {
        HybridSignature bad_sig = good_sig;
        bad_sig.sphincs_sig.clear();
        CHECK_FALSE(hv.verify_module(k_test_binary, bad_sig, ed_kp.pk, sp_kp.pk));
        CHECK(hv.last_failure() == VerifyFailReason::SPHINCS_BAD_SIG);
    }
}

// =============================================================================
// SECTION 3: HybridVerifier — Ed25519 fast-path rejection
// =============================================================================

TEST_CASE("HybridVerifier: Ed25519 fast-path", "[phase114][hybrid_verifier]") {
    HybridVerifier hv;
    const auto ed_kp  = make_ed_keypair();
    const auto ed_kp2 = make_ed_keypair();   // different keypair
    const auto sp_kp  = SphincsSigner::generate_keypair();

    SECTION("wrong Ed25519 signature → ED25519_INVALID") {
        HybridSignature sig = make_valid_sig(k_test_binary, ed_kp2, sp_kp);
        // sig.ed25519_sig was made with ed_kp2, verify against ed_kp.pk
        CHECK_FALSE(hv.verify_module(k_test_binary, sig, ed_kp.pk, sp_kp.pk));
        CHECK(hv.last_failure() == VerifyFailReason::ED25519_INVALID);
    }

    SECTION("Ed25519 sig of different binary → ED25519_INVALID") {
        HybridSignature sig = make_valid_sig(k_other_binary, ed_kp, sp_kp);
        // sig is for k_other_binary; verify against k_test_binary
        CHECK_FALSE(hv.verify_module(k_test_binary, sig, ed_kp.pk, sp_kp.pk));
        CHECK(hv.last_failure() == VerifyFailReason::ED25519_INVALID);
    }

    SECTION("all-zeros Ed25519 sig → ED25519_INVALID") {
        HybridSignature sig = make_valid_sig(k_test_binary, ed_kp, sp_kp);
        std::fill(sig.ed25519_sig.begin(), sig.ed25519_sig.end(), 0x00);
        CHECK_FALSE(hv.verify_module(k_test_binary, sig, ed_kp.pk, sp_kp.pk));
        CHECK(hv.last_failure() == VerifyFailReason::ED25519_INVALID);
    }
}

// =============================================================================
// SECTION 4: HybridVerifier — SPHINCS+ slow-path rejection
// =============================================================================

TEST_CASE("HybridVerifier: SPHINCS+ slow-path", "[phase114][hybrid_verifier]") {
    HybridVerifier hv;
    const auto ed_kp  = make_ed_keypair();
    const auto sp_kp  = SphincsSigner::generate_keypair();
    const auto sp_kp2 = SphincsSigner::generate_keypair();  // different keypair

    SECTION("valid Ed25519, wrong SPHINCS+ signature → SPHINCS_INVALID") {
        HybridSignature sig = make_valid_sig(k_test_binary, ed_kp, sp_kp);
        // Use sp_kp2 public key for verification — mismatch
        CHECK_FALSE(hv.verify_module(k_test_binary, sig, ed_kp.pk, sp_kp2.pk));
        CHECK(hv.last_failure() == VerifyFailReason::SPHINCS_INVALID);
    }

    SECTION("valid Ed25519, SPHINCS+ sig of different binary → SPHINCS_INVALID") {
        // ed sig over k_test_binary with ed_kp
        HybridSignature sig;
        sig.ed25519_sig = ed_sign(k_test_binary, ed_kp);
        // sphincs sig over k_other_binary with sp_kp — mismatch
        sig.sphincs_sig = SphincsSigner::sign(k_other_binary, sp_kp).bytes;

        CHECK_FALSE(hv.verify_module(k_test_binary, sig, ed_kp.pk, sp_kp.pk));
        CHECK(hv.last_failure() == VerifyFailReason::SPHINCS_INVALID);
    }
}

// =============================================================================
// SECTION 5: HybridVerifier — Full success path + cache
// =============================================================================

TEST_CASE("HybridVerifier: full success path", "[phase114][hybrid_verifier]") {
    HybridVerifier hv;
    const auto ed_kp = make_ed_keypair();
    const auto sp_kp = SphincsSigner::generate_keypair();
    HybridSignature sig = make_valid_sig(k_test_binary, ed_kp, sp_kp);

    SECTION("valid Ed25519 + valid SPHINCS+ → true") {
        CHECK(hv.verify_module(k_test_binary, sig, ed_kp.pk, sp_kp.pk));
        CHECK(hv.last_failure() == VerifyFailReason::NONE);
    }

    SECTION("cache populated after success") {
        REQUIRE(hv.verify_module(k_test_binary, sig, ed_kp.pk, sp_kp.pk));
        CHECK(hv.cache_size() == 1u);
        CHECK(hv.is_cached(k_test_binary));
    }

    SECTION("cache hit returns true even with a bad sig (replay protection bypass)") {
        // First call populates the cache.
        REQUIRE(hv.verify_module(k_test_binary, sig, ed_kp.pk, sp_kp.pk));
        // Second call: corrupt the sig — should still return true via cache.
        HybridSignature bad_sig = sig;
        std::fill(bad_sig.ed25519_sig.begin(), bad_sig.ed25519_sig.end(), 0xFF);
        CHECK(hv.verify_module(k_test_binary, bad_sig, ed_kp.pk, sp_kp.pk));
        CHECK(hv.last_failure() == VerifyFailReason::NONE);
    }

    SECTION("two distinct binaries → separate cache entries") {
        HybridSignature sig2 = make_valid_sig(k_other_binary, ed_kp, sp_kp);
        REQUIRE(hv.verify_module(k_test_binary,  sig,  ed_kp.pk, sp_kp.pk));
        REQUIRE(hv.verify_module(k_other_binary, sig2, ed_kp.pk, sp_kp.pk));
        CHECK(hv.cache_size() == 2u);
    }
}

// =============================================================================
// SECTION 6: HybridVerifier — cache management
// =============================================================================

TEST_CASE("HybridVerifier: cache management", "[phase114][hybrid_verifier]") {
    HybridVerifier hv;
    const auto ed_kp = make_ed_keypair();
    const auto sp_kp = SphincsSigner::generate_keypair();
    HybridSignature sig = make_valid_sig(k_test_binary, ed_kp, sp_kp);

    SECTION("clear_cache removes all entries") {
        REQUIRE(hv.verify_module(k_test_binary, sig, ed_kp.pk, sp_kp.pk));
        REQUIRE(hv.cache_size() == 1u);
        hv.clear_cache();
        CHECK(hv.cache_size() == 0u);
        CHECK_FALSE(hv.is_cached(k_test_binary));
    }

    SECTION("after clear, re-verification works normally") {
        REQUIRE(hv.verify_module(k_test_binary, sig, ed_kp.pk, sp_kp.pk));
        hv.clear_cache();
        CHECK(hv.verify_module(k_test_binary, sig, ed_kp.pk, sp_kp.pk));
        CHECK(hv.cache_size() == 1u);
    }
}

// =============================================================================
// SECTION 7: HybridVerifier — thread safety
// =============================================================================

TEST_CASE("HybridVerifier: concurrent verify_module calls are safe",
          "[phase114][hybrid_verifier][thread_safety]") {
    HybridVerifier hv;
    const auto ed_kp = make_ed_keypair();
    const auto sp_kp = SphincsSigner::generate_keypair();

    // Pre-build N distinct binaries and signatures
    constexpr int k_threads = 4;
    std::vector<std::vector<uint8_t>> binaries(k_threads);
    std::vector<HybridSignature>      sigs(k_threads);
    for (int i = 0; i < k_threads; ++i) {
        binaries[i] = {static_cast<uint8_t>(0xA0 + i),
                       static_cast<uint8_t>(0xB0 + i),
                       static_cast<uint8_t>(0xC0 + i),
                       static_cast<uint8_t>(0xD0 + i)};
        sigs[i] = make_valid_sig(binaries[i], ed_kp, sp_kp);
    }

    std::vector<std::thread> threads;
    std::vector<bool>        results(k_threads, false);

    for (int i = 0; i < k_threads; ++i) {
        threads.emplace_back([&, i]() {
            results[i] = hv.verify_module(binaries[i], sigs[i],
                                          ed_kp.pk, sp_kp.pk);
        });
    }
    for (auto& t : threads) t.join();

    for (int i = 0; i < k_threads; ++i) {
        CHECK(results[i]);
    }
    CHECK(hv.cache_size() == static_cast<size_t>(k_threads));
}

// =============================================================================
// SECTION 8: HybridVerifier — verify_fail_str
// =============================================================================

TEST_CASE("verify_fail_str: all values return non-empty strings",
          "[phase114][hybrid_verifier]") {
    CHECK_FALSE(nikola::security::verify_fail_str(VerifyFailReason::NONE).empty());
    CHECK_FALSE(nikola::security::verify_fail_str(VerifyFailReason::ED25519_BAD_PUBKEY).empty());
    CHECK_FALSE(nikola::security::verify_fail_str(VerifyFailReason::ED25519_BAD_SIG).empty());
    CHECK_FALSE(nikola::security::verify_fail_str(VerifyFailReason::ED25519_INVALID).empty());
    CHECK_FALSE(nikola::security::verify_fail_str(VerifyFailReason::SPHINCS_BAD_PUBKEY).empty());
    CHECK_FALSE(nikola::security::verify_fail_str(VerifyFailReason::SPHINCS_BAD_SIG).empty());
    CHECK_FALSE(nikola::security::verify_fail_str(VerifyFailReason::SPHINCS_INVALID).empty());
}

// =============================================================================
// Helper: build EO + Spine fixtures
// =============================================================================

namespace {
    // Large initial ATP so tests aren't budget-starved.
    static constexpr float k_atp_large = 100'000.0f;

    struct SpineFixture {
        CodePatternBlacklist     blacklist;
        MetabolicController      controller{k_atp_large, k_test_nap_threshold};
        EvolutionaryOrchestrator eo{controller, blacklist};
        HybridVerifier           hv;
        ShadowSpine              spine{eo, hv};

        Ed25519Keypair  ed_kp  = make_ed_keypair();
        SphincsKeypair  sp_kp  = SphincsSigner::generate_keypair();
    };
}

// =============================================================================
// SECTION 9: ShadowSpine — construction
// =============================================================================

TEST_CASE("ShadowSpine: construction", "[phase114][shadow_spine]") {
    SECTION("constructs without throwing") {
        REQUIRE_NOTHROW(SpineFixture{});
    }

    SECTION("has_active() == false initially") {
        SpineFixture f;
        CHECK_FALSE(f.spine.has_active());
    }

    SECTION("last_report() == nullptr initially") {
        SpineFixture f;
        CHECK(f.spine.last_report() == nullptr);
    }

    SECTION("stats() returns zeros initially") {
        SpineFixture f;
        auto s = f.spine.stats();
        CHECK(s.total == 0u);
        CHECK(s.succeeded == 0u);
    }
}

// =============================================================================
// SECTION 10: ShadowSpine — stage() paths
// =============================================================================

TEST_CASE("ShadowSpine: stage LOAD_FAILED when file doesn't exist",
          "[phase114][shadow_spine]") {
    SpineFixture f;
    HybridSignature dummy_sig;
    dummy_sig.ed25519_sig.assign(64, 0x00);
    dummy_sig.sphincs_sig.assign(8, 0x00);

    auto rep = f.spine.stage(k_bad_path, "", dummy_sig,
                             f.ed_kp.pk, f.sp_kp.pk);
    CHECK(rep.status == StageStatus::LOAD_FAILED);
    CHECK_FALSE(rep.signature_passed);
    CHECK_FALSE(static_cast<bool>(rep));
}

TEST_CASE("ShadowSpine: stage SIGNATURE_REJECTED when sig is corrupt",
          "[phase114][shadow_spine]") {
    SpineFixture f;

    // Check that the plugin exists; skip gracefully if not (CI without build).
    if (!std::filesystem::exists(k_plugin)) {
        SKIP("Phase 112 plugin not found — skipping ShadowSpine stage test");
    }

    // Deliberately wrong signatures (correct size, all-zero — will fail crypto).
    HybridSignature bad_sig;
    bad_sig.ed25519_sig.assign(64, 0x00);
    bad_sig.sphincs_sig = SphincsSigner::sign(k_test_binary, f.sp_kp).bytes;

    auto rep = f.spine.stage(k_plugin, "", bad_sig, f.ed_kp.pk, f.sp_kp.pk);
    CHECK(rep.status == StageStatus::SIGNATURE_REJECTED);
    CHECK_FALSE(rep.signature_passed);
    CHECK_FALSE(static_cast<bool>(rep));
}

TEST_CASE("ShadowSpine: stage SUCCESS with valid dual signatures + plugin",
          "[phase114][shadow_spine]") {
    SpineFixture f;

    if (!std::filesystem::exists(k_plugin)) {
        SKIP("Phase 112 plugin not found — skipping ShadowSpine stage test");
    }

    // Read the .so binary and sign it with both algorithms.
    auto plugin_binary = read_file(k_plugin);
    REQUIRE_FALSE(plugin_binary.empty());

    HybridSignature sig = make_valid_sig(plugin_binary, f.ed_kp, f.sp_kp);

    auto rep = f.spine.stage(k_plugin, k_safe_source, sig, f.ed_kp.pk, f.sp_kp.pk);

    CHECK(rep.signature_passed);
    CHECK(rep.status == StageStatus::SUCCESS);
    CHECK(static_cast<bool>(rep));
    CHECK(f.spine.has_active());
}

TEST_CASE("ShadowSpine: last_report() reflects most recent stage result",
          "[phase114][shadow_spine]") {
    SpineFixture f;

    HybridSignature dummy_sig;
    dummy_sig.ed25519_sig.assign(64, 0x00);
    dummy_sig.sphincs_sig.assign(8, 0x00);

    (void)f.spine.stage(k_bad_path, "", dummy_sig, f.ed_kp.pk, f.sp_kp.pk);
    const auto* rep = f.spine.last_report();
    REQUIRE(rep != nullptr);
    CHECK(rep->status == StageStatus::LOAD_FAILED);
}

TEST_CASE("ShadowSpine: sig_fail field populated on SIGNATURE_REJECTED",
          "[phase114][shadow_spine]") {
    SpineFixture f;

    if (!std::filesystem::exists(k_plugin)) {
        SKIP("Phase 112 plugin not found");
    }

    auto plugin_binary = read_file(k_plugin);
    REQUIRE_FALSE(plugin_binary.empty());

    // Ed pub key wrong size — should give SIGNATURE_REJECTED + matching sig_fail
    HybridSignature dummy_sig = make_valid_sig(plugin_binary, f.ed_kp, f.sp_kp);
    std::vector<uint8_t> bad_ed_pub(16, 0xAA);  // wrong size

    auto rep = f.spine.stage(k_plugin, "", dummy_sig, bad_ed_pub, f.sp_kp.pk);
    CHECK(rep.status == StageStatus::SIGNATURE_REJECTED);
    CHECK(rep.sig_fail == VerifyFailReason::ED25519_BAD_PUBKEY);
}

// =============================================================================
// SECTION 11: ShadowSpine — rollback
// =============================================================================

TEST_CASE("ShadowSpine: rollback after first SUCCESS returns false (no previous)",
          "[phase114][shadow_spine]") {
    SpineFixture f;

    if (!std::filesystem::exists(k_plugin)) {
        SKIP("Phase 112 plugin not found");
    }

    auto plugin_binary = read_file(k_plugin);
    HybridSignature sig = make_valid_sig(plugin_binary, f.ed_kp, f.sp_kp);

    // First ever stage: active = new module, previous = empty (nothing to roll back to)
    auto rep = f.spine.stage(k_plugin, k_safe_source, sig, f.ed_kp.pk, f.sp_kp.pk);
    REQUIRE(rep.status == StageStatus::SUCCESS);
    REQUIRE(f.spine.has_active());

    // No previous slot was occupied before this load — rollback returns false
    CHECK_FALSE(f.spine.rollback());
}

TEST_CASE("ShadowSpine: rollback with nothing loaded returns false",
          "[phase114][shadow_spine]") {
    SpineFixture f;
    CHECK_FALSE(f.spine.rollback());
}

// =============================================================================
// SECTION 12: ShadowSpine — stats
// =============================================================================

TEST_CASE("ShadowSpine: stats reflect EO cycle counts",
          "[phase114][shadow_spine]") {
    SpineFixture f;

    if (!std::filesystem::exists(k_plugin)) {
        SKIP("Phase 112 plugin not found");
    }

    auto plugin_binary = read_file(k_plugin);
    HybridSignature sig = make_valid_sig(plugin_binary, f.ed_kp, f.sp_kp);

    (void)f.spine.stage(k_plugin, k_safe_source, sig, f.ed_kp.pk, f.sp_kp.pk);

    auto s = f.spine.stats();
    CHECK(s.total >= 1u);
}

// =============================================================================
// SECTION 13: StageStatus — stage_status_str
// =============================================================================

TEST_CASE("stage_status_str: all values return non-empty strings",
          "[phase114][shadow_spine]") {
    using nikola::autonomy::stage_status_str;
    CHECK_FALSE(stage_status_str(StageStatus::SUCCESS).empty());
    CHECK_FALSE(stage_status_str(StageStatus::SIGNATURE_REJECTED).empty());
    CHECK_FALSE(stage_status_str(StageStatus::ATP_DENIED).empty());
    CHECK_FALSE(stage_status_str(StageStatus::SECURITY_REJECTED).empty());
    CHECK_FALSE(stage_status_str(StageStatus::PHYSICS_REJECTED).empty());
    CHECK_FALSE(stage_status_str(StageStatus::LOAD_FAILED).empty());
    CHECK_FALSE(stage_status_str(StageStatus::SYMBOL_MISSING).empty());
    CHECK_FALSE(stage_status_str(StageStatus::SAME_MODULE).empty());
}

// =============================================================================
// SECTION 14: StageReport — operator bool
// =============================================================================

TEST_CASE("StageReport: operator bool reflects SUCCESS status",
          "[phase114][shadow_spine]") {
    using nikola::autonomy::StageReport;

    SECTION("SUCCESS → true") {
        StageReport r;
        r.status = StageStatus::SUCCESS;
        CHECK(static_cast<bool>(r));
    }

    SECTION("non-SUCCESS → false") {
        for (auto s : { StageStatus::SIGNATURE_REJECTED,
                        StageStatus::ATP_DENIED,
                        StageStatus::LOAD_FAILED,
                        StageStatus::PHYSICS_REJECTED }) {
            StageReport r;
            r.status = s;
            CHECK_FALSE(static_cast<bool>(r));
        }
    }
}
