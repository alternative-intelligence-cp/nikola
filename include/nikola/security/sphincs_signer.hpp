// =============================================================================
// NIKOLA — Phase 107 / GAP-047
// SPHINCS+ Signer — post-quantum digital signature wrapper
// =============================================================================
// Spec   : GAP-047  §8.5  "Implementation Guide for SPHINCS+"
//          docs/info/integration/sections/04_infrastructure/05_security_subsystem.md
// Author : Nikola Phase 107
// License: MIT
//
// Wraps the SPHINCS+ reference implementation (public-domain, PQClean lineage)
// with a clean C++ API suitable for the Nikola module-signing pipeline.
//
// Parameter set compiled: sphincs-shake-256f
//   Public key  : 64 bytes
//   Secret key  : 128 bytes
//   Signature   : ≤ 49856 bytes (exact size returned in SphincsSignature)
//   Sign latency: ~1 ms (ref impl)   Verify: ~1 ms (ref impl)
//
// Production note: For maximum security with smaller signatures, recompile the
// sphincsplus_shake256f target with PARAMS=sphincs-shake-256s (sign ~60 ms).
// For maximum throughput, use the avx2 build in third_party/sphincsplus/shake-avx2/.
//
// Usage:
//   auto kp  = nikola::security::SphincsSigner::generate_keypair();
//   auto sig = nikola::security::SphincsSigner::sign(msg.data(), msg.size(), kp);
//   bool ok  = nikola::security::SphincsSigner::verify(sig, msg.data(), msg.size(), kp);
// =============================================================================

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>

// ---------------------------------------------------------------------------
// SPHINCS+ C API — declared via extern "C" so the C headers compile cleanly
// under g++ with C++23.  The actual symbols are in libsphincsplus_shake256f.a.
// ---------------------------------------------------------------------------
extern "C" {
    unsigned long long crypto_sign_secretkeybytes(void);
    unsigned long long crypto_sign_publickeybytes(void);
    unsigned long long crypto_sign_bytes(void);
    unsigned long long crypto_sign_seedbytes(void);

    int crypto_sign_seed_keypair(unsigned char *pk, unsigned char *sk,
                                 const unsigned char *seed);
    int crypto_sign_keypair(unsigned char *pk, unsigned char *sk);

    int crypto_sign_signature(uint8_t *sig, size_t *siglen,
                              const uint8_t *m, size_t mlen,
                              const uint8_t *sk);
    int crypto_sign_verify(const uint8_t *sig, size_t siglen,
                           const uint8_t *m, size_t mlen,
                           const uint8_t *pk);
} // extern "C"

namespace nikola::security {

// ---------------------------------------------------------------------------
// SphincsKeypair
// ---------------------------------------------------------------------------

/// SPHINCS+ keypair produced by generate_keypair().
/// pk: public key (64 bytes for shake-256f)
/// sk: secret key (128 bytes for shake-256f; keep private!)
struct SphincsKeypair {
    std::vector<uint8_t> pk;   ///< public key
    std::vector<uint8_t> sk;   ///< secret key (private)
};

// ---------------------------------------------------------------------------
// SphincsSignature
// ---------------------------------------------------------------------------

/// Detached SPHINCS+ signature.  `bytes` is the raw signature bytes; its
/// length is <= max_signature_bytes() and is the actual signed length.
struct SphincsSignature {
    std::vector<uint8_t> bytes;
};

// ---------------------------------------------------------------------------
// SphincsSigner
// ---------------------------------------------------------------------------

/// Stateless SPHINCS+ signing utility.
///
/// All methods are static; no instance state is required.
/// Thread-safe as long as different keypairs are used on different threads
/// (the reference SPHINCS+ keygen/sign calls /dev/urandom which is safe).
class SphincsSigner {
public:
    // -----------------------------------------------------------------------
    // Static parameter constants
    // -----------------------------------------------------------------------

    /// Size of a public key in bytes (64 for shake-256f/s).
    [[nodiscard]] static unsigned long long public_key_bytes() noexcept {
        return crypto_sign_publickeybytes();
    }

    /// Size of a secret key in bytes (128 for shake-256f/s).
    [[nodiscard]] static unsigned long long secret_key_bytes() noexcept {
        return crypto_sign_secretkeybytes();
    }

    /// Maximum detached signature size in bytes.
    /// Actual signature bytes may differ — use SphincsSignature::bytes.size().
    [[nodiscard]] static unsigned long long max_signature_bytes() noexcept {
        return crypto_sign_bytes();
    }

    /// Human-readable algorithm identifier.
    [[nodiscard]] static std::string_view algorithm_name() noexcept {
        return "SPHINCS+-shake-256f";
    }

    // -----------------------------------------------------------------------
    // Key generation
    // -----------------------------------------------------------------------

    /// Generate a fresh random keypair.
    /// Throws std::runtime_error if the underlying call fails.
    [[nodiscard]] static SphincsKeypair generate_keypair() {
        SphincsKeypair kp;
        kp.pk.resize(crypto_sign_publickeybytes());
        kp.sk.resize(crypto_sign_secretkeybytes());
        int rc = crypto_sign_keypair(kp.pk.data(), kp.sk.data());
        if (rc != 0) {
            throw std::runtime_error(
                "SphincsSigner: keypair generation failed (rc=" +
                std::to_string(rc) + ")");
        }
        return kp;
    }

    /// Generate a keypair deterministically from a 96-byte seed.
    /// Throws std::runtime_error if seed is too short or the call fails.
    [[nodiscard]] static SphincsKeypair generate_keypair_from_seed(
            const uint8_t* seed, size_t seed_len) {
        if (seed_len < crypto_sign_seedbytes()) {
            throw std::invalid_argument(
                "SphincsSigner: seed too short (" +
                std::to_string(seed_len) + " < " +
                std::to_string(crypto_sign_seedbytes()) + " required)");
        }
        SphincsKeypair kp;
        kp.pk.resize(crypto_sign_publickeybytes());
        kp.sk.resize(crypto_sign_secretkeybytes());
        int rc = crypto_sign_seed_keypair(kp.pk.data(), kp.sk.data(), seed);
        if (rc != 0) {
            throw std::runtime_error(
                "SphincsSigner: seed-keypair generation failed (rc=" +
                std::to_string(rc) + ")");
        }
        return kp;
    }

    // -----------------------------------------------------------------------
    // Sign (detached)
    // -----------------------------------------------------------------------

    /// Sign `msg_len` bytes at `msg` using the secret key in `kp`.
    /// Returns a detached SphincsSignature.
    /// Throws std::runtime_error on failure.
    [[nodiscard]] static SphincsSignature sign(
            const uint8_t* msg, size_t msg_len,
            const SphincsKeypair& kp) {
        SphincsSignature sig;
        sig.bytes.resize(crypto_sign_bytes());
        size_t sig_len = 0;
        int rc = crypto_sign_signature(
            sig.bytes.data(), &sig_len,
            msg, msg_len,
            kp.sk.data());
        if (rc != 0) {
            throw std::runtime_error(
                "SphincsSigner: sign failed (rc=" + std::to_string(rc) + ")");
        }
        sig.bytes.resize(sig_len);
        return sig;
    }

    /// Convenience overload: sign a std::vector<uint8_t>.
    [[nodiscard]] static SphincsSignature sign(
            const std::vector<uint8_t>& msg,
            const SphincsKeypair& kp) {
        return sign(msg.data(), msg.size(), kp);
    }

    /// Convenience overload: sign a std::string (raw bytes).
    [[nodiscard]] static SphincsSignature sign(
            std::string_view msg,
            const SphincsKeypair& kp) {
        return sign(reinterpret_cast<const uint8_t*>(msg.data()),
                    msg.size(), kp);
    }

    // -----------------------------------------------------------------------
    // Verify (detached)
    // -----------------------------------------------------------------------

    /// Verify a detached signature against `msg_len` bytes at `msg`.
    /// Returns true iff the signature is valid under `kp.pk`.
    [[nodiscard]] static bool verify(
            const SphincsSignature& sig,
            const uint8_t* msg, size_t msg_len,
            const SphincsKeypair& kp) noexcept {
        int rc = crypto_sign_verify(
            sig.bytes.data(), sig.bytes.size(),
            msg, msg_len,
            kp.pk.data());
        return rc == 0;
    }

    /// Convenience overload: verify against a std::vector<uint8_t>.
    [[nodiscard]] static bool verify(
            const SphincsSignature& sig,
            const std::vector<uint8_t>& msg,
            const SphincsKeypair& kp) noexcept {
        return verify(sig, msg.data(), msg.size(), kp);
    }

    /// Convenience overload: verify against a std::string (raw bytes).
    [[nodiscard]] static bool verify(
            const SphincsSignature& sig,
            std::string_view msg,
            const SphincsKeypair& kp) noexcept {
        return verify(sig,
                      reinterpret_cast<const uint8_t*>(msg.data()),
                      msg.size(), kp);
    }
};

} // namespace nikola::security
