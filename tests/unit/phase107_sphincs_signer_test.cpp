// =============================================================================
// Phase 107 / GAP-047 — SPHINCS+ Post-Quantum Signer Tests
// =============================================================================
// Tests the SphincsSigner class against the sphincs-shake-256f parameter set.
//
// Parameter set: sphincs-shake-256f
//   PK  = 64 bytes    SK  = 128 bytes    Sig = 49856 bytes max
//   Sign ~1-50ms (ref impl, no AVX2)    Verify ~1ms
//
// Test structure (9 test cases):
//   1. Algorithm name and constant sizes
//   2. Keypair generation — correct byte sizes, non-zero content
//   3. Sign produces non-empty signature with valid size
//   4. Roundtrip: sign + verify (hello world message)
//   5. Roundtrip: sign + verify (empty message)
//   6. Roundtrip: sign + verify (1024-byte message)
//   7. Verify fails for tampered signature (flip bit in sig)
//   8. Verify fails for tampered message (flip bit in message)
//   9. Cross-key failure + deterministic seed keypair roundtrip
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "nikola/security/sphincs_signer.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

using nikola::security::SphincsSigner;
using nikola::security::SphincsKeypair;
using nikola::security::SphincsSignature;

// ---------------------------------------------------------------------------
// Helper: simple LCG fill for deterministic test messages
// ---------------------------------------------------------------------------
static std::vector<uint8_t> make_msg(size_t n, uint32_t seed = 0xDEADBEEFu) {
    std::vector<uint8_t> m(n);
    uint32_t x = seed;
    for (auto& b : m) {
        x = x * 1664525u + 1013904223u;
        b = static_cast<uint8_t>(x >> 24);
    }
    return m;
}

// ---------------------------------------------------------------------------
// Shared keypair (generated once per process; sign/verify is deterministic)
// Avoid regenerating in every section to keep total test time down.
// ---------------------------------------------------------------------------
static SphincsKeypair make_once() {
    static SphincsKeypair kp = SphincsSigner::generate_keypair();
    return kp;
}

// ===========================================================================
// Test 1 — Algorithm name and constant sizes
// ===========================================================================
TEST_CASE("Phase107 SphincsSigner constants", "[phase107][sphincs][pqc]") {
    SECTION("algorithm_name is non-empty") {
        REQUIRE_FALSE(SphincsSigner::algorithm_name().empty());
    }

    SECTION("algorithm_name contains 'SPHINCS'") {
        auto name = SphincsSigner::algorithm_name();
        REQUIRE(name.find("SPHINCS") != std::string_view::npos);
    }

    SECTION("algorithm_name contains 'shake'") {
        auto name = SphincsSigner::algorithm_name();
        REQUIRE(name.find("shake") != std::string_view::npos);
    }

    SECTION("public_key_bytes == 64") {
        // shake-256f/s: PK = 2 * SPX_N = 2 * 32
        REQUIRE(SphincsSigner::public_key_bytes() == 64ULL);
    }

    SECTION("secret_key_bytes == 128") {
        // shake-256f/s: SK = 4 * SPX_N = 4 * 32
        REQUIRE(SphincsSigner::secret_key_bytes() == 128ULL);
    }

    SECTION("max_signature_bytes > 0") {
        REQUIRE(SphincsSigner::max_signature_bytes() > 0ULL);
    }

    SECTION("max_signature_bytes is plausible (>= 8KB, <= 65KB)") {
        auto bytes = SphincsSigner::max_signature_bytes();
        REQUIRE(bytes >= 8'000ULL);
        REQUIRE(bytes <= 65'536ULL);
    }
}

// ===========================================================================
// Test 2 — Keypair generation
// ===========================================================================
TEST_CASE("Phase107 SphincsSigner generate_keypair", "[phase107][sphincs][pqc]") {
    auto kp = SphincsSigner::generate_keypair();

    SECTION("public key has correct size") {
        REQUIRE(kp.pk.size() == SphincsSigner::public_key_bytes());
    }

    SECTION("secret key has correct size") {
        REQUIRE(kp.sk.size() == SphincsSigner::secret_key_bytes());
    }

    SECTION("public key is non-zero") {
        bool all_zero = true;
        for (auto b : kp.pk) if (b != 0) { all_zero = false; break; }
        REQUIRE_FALSE(all_zero);
    }

    SECTION("secret key is non-zero") {
        bool all_zero = true;
        for (auto b : kp.sk) if (b != 0) { all_zero = false; break; }
        REQUIRE_FALSE(all_zero);
    }

    SECTION("two keypairs differ (probabilistically)") {
        auto kp2 = SphincsSigner::generate_keypair();
        // Public keys should differ (collision prob negligible)
        REQUIRE(kp.pk != kp2.pk);
    }
}

// ===========================================================================
// Test 3 — Sign produces a valid-sized non-trivial signature
// ===========================================================================
TEST_CASE("Phase107 SphincsSigner sign produces signature",
          "[phase107][sphincs][pqc]") {
    const auto kp = make_once();
    const auto msg = make_msg(32);
    auto sig = SphincsSigner::sign(msg, kp);

    SECTION("signature bytes is non-empty") {
        REQUIRE_FALSE(sig.bytes.empty());
    }

    SECTION("signature size <= max_signature_bytes()") {
        REQUIRE(sig.bytes.size() <= SphincsSigner::max_signature_bytes());
    }

    SECTION("signature size > 0") {
        REQUIRE(sig.bytes.size() > 0);
    }

    SECTION("signature is non-zero") {
        bool all_zero = true;
        for (auto b : sig.bytes) if (b != 0) { all_zero = false; break; }
        REQUIRE_FALSE(all_zero);
    }
}

// ===========================================================================
// Test 4 — Roundtrip: sign + verify (small message)
// ===========================================================================
TEST_CASE("Phase107 SphincsSigner roundtrip small message",
          "[phase107][sphincs][pqc]") {
    const auto kp = make_once();

    SECTION("sign+verify 'hello nikola'") {
        const std::string text = "hello nikola";
        auto sig = SphincsSigner::sign(text, kp);
        REQUIRE(SphincsSigner::verify(sig, text, kp));
    }

    SECTION("sign+verify 16-byte message") {
        auto msg = make_msg(16);
        auto sig = SphincsSigner::sign(msg, kp);
        REQUIRE(SphincsSigner::verify(sig, msg, kp));
    }

    SECTION("sign+verify 32-byte message") {
        auto msg = make_msg(32, 0xCAFEBABEu);
        auto sig = SphincsSigner::sign(msg, kp);
        REQUIRE(SphincsSigner::verify(sig, msg, kp));
    }
}

// ===========================================================================
// Test 5 — Roundtrip: sign + verify (empty message)
// ===========================================================================
TEST_CASE("Phase107 SphincsSigner roundtrip empty message",
          "[phase107][sphincs][pqc]") {
    const auto kp = make_once();

    SECTION("empty message sign does not throw") {
        std::vector<uint8_t> empty;
        REQUIRE_NOTHROW(SphincsSigner::sign(empty, kp));
    }

    SECTION("empty message verify succeeds") {
        std::vector<uint8_t> empty;
        auto sig = SphincsSigner::sign(empty, kp);
        REQUIRE(SphincsSigner::verify(sig, empty, kp));
    }

    SECTION("empty message signature is non-empty (SPHINCS+ sigs are large)") {
        std::vector<uint8_t> empty;
        auto sig = SphincsSigner::sign(empty, kp);
        REQUIRE_FALSE(sig.bytes.empty());
    }
}

// ===========================================================================
// Test 6 — Roundtrip: sign + verify (1 KB message)
// ===========================================================================
TEST_CASE("Phase107 SphincsSigner roundtrip 1KB message",
          "[phase107][sphincs][pqc]") {
    const auto kp = make_once();
    auto msg = make_msg(1024, 0xABCD1234u);

    SECTION("1024-byte sign+verify succeeds") {
        auto sig = SphincsSigner::sign(msg, kp);
        REQUIRE(SphincsSigner::verify(sig, msg, kp));
    }

    SECTION("1024-byte signature size in valid range") {
        auto sig = SphincsSigner::sign(msg, kp);
        REQUIRE(sig.bytes.size() <= SphincsSigner::max_signature_bytes());
        REQUIRE(sig.bytes.size() > 0);
    }
}

// ===========================================================================
// Test 7 — Tampered signature fails verification
// ===========================================================================
TEST_CASE("Phase107 SphincsSigner tampered signature rejected",
          "[phase107][sphincs][pqc]") {
    const auto kp  = make_once();
    const auto msg = make_msg(64);
    auto sig        = SphincsSigner::sign(msg, kp);

    SECTION("flip first byte of signature => fail") {
        auto bad = sig;
        bad.bytes[0] ^= 0xFF;
        REQUIRE_FALSE(SphincsSigner::verify(bad, msg, kp));
    }

    SECTION("flip middle byte of signature => fail") {
        auto bad = sig;
        size_t mid = bad.bytes.size() / 2;
        bad.bytes[mid] ^= 0xA5;
        REQUIRE_FALSE(SphincsSigner::verify(bad, msg, kp));
    }

    SECTION("flip last byte of signature => fail") {
        auto bad = sig;
        bad.bytes.back() ^= 0x01;
        REQUIRE_FALSE(SphincsSigner::verify(bad, msg, kp));
    }

    SECTION("truncated signature => fail") {
        auto bad = sig;
        bad.bytes.pop_back();
        REQUIRE_FALSE(SphincsSigner::verify(bad, msg, kp));
    }
}

// ===========================================================================
// Test 8 — Tampered message fails verification
// ===========================================================================
TEST_CASE("Phase107 SphincsSigner tampered message rejected",
          "[phase107][sphincs][pqc]") {
    const auto kp  = make_once();
    const auto msg = make_msg(128);
    const auto sig = SphincsSigner::sign(msg, kp);

    SECTION("flip first byte of message => fail") {
        auto bad = msg;
        bad[0] ^= 0xFF;
        REQUIRE_FALSE(SphincsSigner::verify(sig, bad, kp));
    }

    SECTION("flip last byte of message => fail") {
        auto bad = msg;
        bad.back() ^= 0x01;
        REQUIRE_FALSE(SphincsSigner::verify(sig, bad, kp));
    }

    SECTION("appended byte changes message => fail") {
        auto bad = msg;
        bad.push_back(0x42);
        REQUIRE_FALSE(SphincsSigner::verify(sig, bad, kp));
    }

    SECTION("original message still verifies (sanity)") {
        REQUIRE(SphincsSigner::verify(sig, msg, kp));
    }
}

// ===========================================================================
// Test 9 — Cross-keypair rejection + deterministic seed roundtrip
// ===========================================================================
TEST_CASE("Phase107 SphincsSigner cross-keypair and seeded keypair",
          "[phase107][sphincs][pqc]") {
    SECTION("signature from key A does not verify under key B") {
        auto kp_a = SphincsSigner::generate_keypair();
        auto kp_b = SphincsSigner::generate_keypair();
        auto msg  = make_msg(32, 0x12345678u);
        auto sig  = SphincsSigner::sign(msg, kp_a);

        // Must fail under key B's public key
        REQUIRE_FALSE(SphincsSigner::verify(sig, msg, kp_b));
        // Must succeed under key A's public key (sanity)
        REQUIRE(SphincsSigner::verify(sig, msg, kp_a));
    }

    SECTION("deterministic seed produces fixed keypair") {
        // seed_len = 3 * SPX_N = 96 bytes for shake-256f
        std::vector<uint8_t> seed(96, 0x42);
        auto kp1 = SphincsSigner::generate_keypair_from_seed(seed.data(), seed.size());
        auto kp2 = SphincsSigner::generate_keypair_from_seed(seed.data(), seed.size());

        REQUIRE(kp1.pk == kp2.pk);
        REQUIRE(kp1.sk == kp2.sk);
    }

    SECTION("seeded keypair roundtrip sign+verify") {
        std::vector<uint8_t> seed(96, 0x7F);
        auto kp  = SphincsSigner::generate_keypair_from_seed(seed.data(), seed.size());
        auto msg = make_msg(48, 0xFACEFEEDu);
        auto sig = SphincsSigner::sign(msg, kp);
        REQUIRE(SphincsSigner::verify(sig, msg, kp));
    }

    SECTION("seed too short throws") {
        std::vector<uint8_t> short_seed(16, 0x01);  // 96 required
        REQUIRE_THROWS_AS(
            SphincsSigner::generate_keypair_from_seed(
                short_seed.data(), short_seed.size()),
            std::invalid_argument);
    }
}
