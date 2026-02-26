// =============================================================================
// NIKOLA — Phase 108 — ML-KEM (Kyber-768) KEM Test Suite
// =============================================================================
// Tests the MlKem post-quantum key encapsulation wrapper:
//   §1  Algorithm identity / size constants
//   §2  Keypair generation — correct sizes, non-zero, keys differ
//   §3  Two distinct keypairs are independent (different bytes)
//   §4  Encapsulate — correct output sizes, ss non-zero, ct non-zero
//   §5  Decapsulate roundtrip — ss from encap == ss from decap
//   §6  Two encapsulations of the same pk produce different ct/ss (randomized)
//   §7  Tamper: modified ciphertext → decap yields DIFFERENT ss (security property)
//   §8  Tamper: wrong secret key → decap yields DIFFERENT ss
//   §9  Invalid input sizes throw correct exceptions
//   §10 Constant-time compare utility
//   §11 Full Diffie-Hellman-like exchange simulation (Alice/Bob)

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/security/mlkem_kem.hpp>

#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace nikola::security;

// =============================================================================
// §1 — Algorithm identity / size constants
// =============================================================================
TEST_CASE("MlKem §1 algorithm identity and size constants", "[mlkem][phase108]") {
    REQUIRE(std::string(MlKem::algorithm_name()) == "Kyber768 (ML-KEM-768, FIPS 203)");

    // Kyber-768 (KYBER_K=3) sizes are well-specified
    CHECK(MlKem::public_key_bytes()    == 1184);
    CHECK(MlKem::secret_key_bytes()    == 2400);
    CHECK(MlKem::ciphertext_bytes()    == 1088);
    CHECK(MlKem::shared_secret_bytes() == 32);

    // Struct mirrors static methods
    CHECK(MlKemSizes::public_key_bytes   == 1184);
    CHECK(MlKemSizes::secret_key_bytes   == 2400);
    CHECK(MlKemSizes::ciphertext_bytes   == 1088);
    CHECK(MlKemSizes::shared_secret_bytes == 32);
}

// =============================================================================
// §2 — Keypair generation
// =============================================================================
TEST_CASE("MlKem §2 keypair generation sizes and non-zero", "[mlkem][phase108]") {
    auto kp = MlKem::generate_keypair();

    REQUIRE(kp.pk.size() == MlKem::public_key_bytes());
    REQUIRE(kp.sk.size() == MlKem::secret_key_bytes());

    // Keys should not be all-zero
    int pk_nonzero = 0, sk_nonzero = 0;
    for (auto b : kp.pk) pk_nonzero |= b;
    for (auto b : kp.sk) sk_nonzero |= b;
    CHECK(pk_nonzero != 0);
    CHECK(sk_nonzero != 0);

    // pk and sk should differ
    CHECK(kp.pk != std::vector<uint8_t>(kp.pk.size(), 0));
}

// =============================================================================
// §3 — Two distinct keypairs are independent
// =============================================================================
TEST_CASE("MlKem §3 two keypairs are independent", "[mlkem][phase108]") {
    auto kp1 = MlKem::generate_keypair();
    auto kp2 = MlKem::generate_keypair();

    // Public keys should differ (overwhelmingly likely with proper RNG)
    CHECK(kp1.pk != kp2.pk);
    // Secret keys should differ
    CHECK(kp1.sk != kp2.sk);
}

// =============================================================================
// §4 — Encapsulate output sizes and non-zero content
// =============================================================================
TEST_CASE("MlKem §4 encapsulate output sizes and non-zero", "[mlkem][phase108]") {
    auto kp = MlKem::generate_keypair();
    auto enc = MlKem::encapsulate(kp.pk);

    REQUIRE(enc.ct.size() == MlKem::ciphertext_bytes());
    REQUIRE(enc.ss.size() == MlKem::shared_secret_bytes());

    int ct_nonzero = 0, ss_nonzero = 0;
    for (auto b : enc.ct) ct_nonzero |= b;
    for (auto b : enc.ss) ss_nonzero |= b;
    CHECK(ct_nonzero != 0);
    CHECK(ss_nonzero != 0);
}

// =============================================================================
// §5 — Decapsulate roundtrip: sender ss == receiver ss
// =============================================================================
TEST_CASE("MlKem §5 encap/decap roundtrip shared secret matches", "[mlkem][phase108]") {
    auto kp  = MlKem::generate_keypair();
    auto enc = MlKem::encapsulate(kp.pk);
    auto ss_dec = MlKem::decapsulate(enc.ct, kp.sk);

    REQUIRE(ss_dec.size() == MlKem::shared_secret_bytes());
    // The fundamental KEM correctness property
    CHECK(MlKem::constant_time_equal(enc.ss, ss_dec));
}

// =============================================================================
// §6 — Two encapsulations of the same pk differ (ephemerality)
// =============================================================================
TEST_CASE("MlKem §6 two encapsulations of same pk produce different ct and ss", "[mlkem][phase108]") {
    auto kp   = MlKem::generate_keypair();
    auto enc1 = MlKem::encapsulate(kp.pk);
    auto enc2 = MlKem::encapsulate(kp.pk);

    // Both should still decapsulate correctly
    auto ss1 = MlKem::decapsulate(enc1.ct, kp.sk);
    auto ss2 = MlKem::decapsulate(enc2.ct, kp.sk);
    CHECK(MlKem::constant_time_equal(enc1.ss, ss1));
    CHECK(MlKem::constant_time_equal(enc2.ss, ss2));

    // Different randomness each time — ciphertexts and shared secrets differ
    CHECK(enc1.ct != enc2.ct);
    CHECK(enc1.ss != enc2.ss);
}

// =============================================================================
// §7 — Security: tampered ciphertext produces wrong shared secret
// =============================================================================
TEST_CASE("MlKem §7 tampered ciphertext yields different shared secret", "[mlkem][phase108]") {
    auto kp  = MlKem::generate_keypair();
    auto enc = MlKem::encapsulate(kp.pk);

    // Flip a byte in the middle of the ciphertext
    auto ct_bad = enc.ct;
    ct_bad[ct_bad.size() / 2] ^= 0xFF;

    auto ss_tampered = MlKem::decapsulate(ct_bad, kp.sk);

    // Kyber's implicit rejection means decap succeeds (no exception) but
    // the recovered ss should differ from the sender's ss
    REQUIRE(ss_tampered.size() == MlKem::shared_secret_bytes());
    CHECK_FALSE(MlKem::constant_time_equal(enc.ss, ss_tampered));
}

// =============================================================================
// §8 — Security: wrong secret key produces wrong shared secret
// =============================================================================
TEST_CASE("MlKem §8 wrong secret key yields different shared secret", "[mlkem][phase108]") {
    auto kp_correct = MlKem::generate_keypair();
    auto kp_wrong   = MlKem::generate_keypair();

    auto enc = MlKem::encapsulate(kp_correct.pk);

    // Decapsulate with a completely different key
    auto ss_wrong = MlKem::decapsulate(enc.ct, kp_wrong.sk);

    REQUIRE(ss_wrong.size() == MlKem::shared_secret_bytes());
    CHECK_FALSE(MlKem::constant_time_equal(enc.ss, ss_wrong));
}

// =============================================================================
// §9 — Invalid input sizes throw correct exceptions
// =============================================================================
TEST_CASE("MlKem §9 invalid input sizes throw std::invalid_argument", "[mlkem][phase108]") {
    auto kp = MlKem::generate_keypair();

    // Encapsulate with too-short pk
    std::vector<uint8_t> short_pk(100, 0x42);
    CHECK_THROWS_AS(MlKem::encapsulate(short_pk), std::invalid_argument);

    // Decapsulate with too-short ct
    auto enc = MlKem::encapsulate(kp.pk);
    std::vector<uint8_t> short_ct(10, 0x00);
    CHECK_THROWS_AS(MlKem::decapsulate(short_ct, kp.sk), std::invalid_argument);

    // Decapsulate with too-short sk
    std::vector<uint8_t> short_sk(100, 0x00);
    CHECK_THROWS_AS(MlKem::decapsulate(enc.ct, short_sk), std::invalid_argument);

    // Encapsulate with too-long pk (also wrong)
    std::vector<uint8_t> long_pk(2000, 0x01);
    CHECK_THROWS_AS(MlKem::encapsulate(long_pk), std::invalid_argument);
}

// =============================================================================
// §10 — constant_time_equal utility
// =============================================================================
TEST_CASE("MlKem §10 constant_time_equal utility", "[mlkem][phase108]") {
    std::vector<uint8_t> a = {0x01, 0x02, 0x03};
    std::vector<uint8_t> b = {0x01, 0x02, 0x03};
    std::vector<uint8_t> c = {0x01, 0x02, 0x04};
    std::vector<uint8_t> d = {0x01, 0x02};

    CHECK( MlKem::constant_time_equal(a, b));
    CHECK_FALSE(MlKem::constant_time_equal(a, c));
    CHECK_FALSE(MlKem::constant_time_equal(a, d));

    // Self-comparison
    auto kp = MlKem::generate_keypair();
    CHECK(MlKem::constant_time_equal(kp.pk, kp.pk));
    CHECK_FALSE(MlKem::constant_time_equal(kp.pk, kp.sk));
}

// =============================================================================
// §11 — Full Alice/Bob KEM exchange simulation
// =============================================================================
TEST_CASE("MlKem §11 full Alice-Bob KEM exchange simulation", "[mlkem][phase108]") {
    // --- Key generation (Bob publishes pk) ---
    auto bob_kp = MlKem::generate_keypair();

    // --- Encapsulation (Alice encapsulates to Bob) ---
    auto alice_enc = MlKem::encapsulate(bob_kp.pk);
    // Alice keeps alice_enc.ss, sends alice_enc.ct to Bob

    // --- Decapsulation (Bob recovers shared secret) ---
    auto bob_ss = MlKem::decapsulate(alice_enc.ct, bob_kp.sk);

    // --- Verification ---
    REQUIRE(alice_enc.ss.size() == 32);
    REQUIRE(bob_ss.size()       == 32);
    CHECK(MlKem::constant_time_equal(alice_enc.ss, bob_ss));

    // The shared secret is 32 bytes of entropy suitable for AES-256
    // Verify it doesn't degenerate to all-zeros
    int nonzero = 0;
    for (auto b : alice_enc.ss) nonzero |= b;
    CHECK(nonzero != 0);

    // --- Confirm Eve (third party with different key) cannot derive the same ss ---
    auto eve_kp  = MlKem::generate_keypair();
    auto eve_ss  = MlKem::decapsulate(alice_enc.ct, eve_kp.sk);
    CHECK_FALSE(MlKem::constant_time_equal(alice_enc.ss, eve_ss));
}
