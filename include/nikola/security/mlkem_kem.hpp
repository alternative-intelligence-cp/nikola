// =============================================================================
// NIKOLA — Phase 108
// ML-KEM (Kyber-768) — post-quantum key encapsulation mechanism wrapper
// =============================================================================
// Spec   : FIPS 203 (ML-KEM) / CRYSTALS-Kyber spec v3.02
//          Parameter set: Kyber-768 (NIST Security Level 3)
// Source : third_party/kyber/ref  (CC0 public domain, pq-crystals)
// Author : Nikola Phase 108
// License: MIT
//
// Wraps the Kyber reference implementation with a clean C++ API for secure
// key exchange and ephemeral shared-secret derivation in the Nikola framework.
//
// Parameter set compiled: kyber768  (KYBER_K=3)
//   Public key  : 1184 bytes
//   Secret key  : 2400 bytes
//   Ciphertext  : 1088 bytes
//   Shared secret: 32 bytes  (suitable for use as symmetric AES-256 key)
//
// Security note: Kyber-768 provides approximately 180 bits of classical security
// and ~180 bits of quantum security. It is the primary NIST recommendation
// (ML-KEM-768) under FIPS 203. Use Kyber-1024 (KYBER_K=4) for Level 5 if needed.
//
// Usage:
//   auto kp = nikola::security::MlKem::generate_keypair();
//   auto enc = nikola::security::MlKem::encapsulate(kp.pk);
//   // enc.ss == shared secret (sender side),  enc.ct == ciphertext to send
//   auto ss  = nikola::security::MlKem::decapsulate(enc.ct, kp.sk);
//   // ss == same shared secret (receiver side — only if sk matches pk)
//   assert(ss == enc.ss);
// =============================================================================

#pragma once

#include <cstdint>
#include <cstring>
#include <array>
#include <vector>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Kyber C API — extern "C" declarations using the actual compiled symbol names.
// The Kyber ref implementation uses KYBER_NAMESPACE(s) = pqcrystals_kyber768_ref_##s
// for KYBER_K=3. We declare these directly so the header is self-contained.
// Symbols are in libkyber768.a linked by CMake.
// ---------------------------------------------------------------------------
extern "C" {
    // keypair: pk = public key (1184 B), sk = secret key (2400 B)
    int pqcrystals_kyber768_ref_keypair(uint8_t *pk, uint8_t *sk);

    // enc: ct = ciphertext (1088 B), ss = shared secret (32 B), pk = public key
    int pqcrystals_kyber768_ref_enc(uint8_t *ct, uint8_t *ss, const uint8_t *pk);

    // dec: ss = shared secret (32 B), ct = ciphertext (1088 B), sk = secret key
    int pqcrystals_kyber768_ref_dec(uint8_t *ss, const uint8_t *ct, const uint8_t *sk);
} // extern "C"

namespace nikola::security {

// ---------------------------------------------------------------------------
// Size constants for Kyber-768 (KYBER_K=3)
// ---------------------------------------------------------------------------

/// Byte sizes for the compiled parameter set (Kyber-768).
struct MlKemSizes {
    static constexpr size_t public_key_bytes  = 1184;   ///< KYBER_PUBLICKEYBYTES
    static constexpr size_t secret_key_bytes  = 2400;   ///< KYBER_SECRETKEYBYTES
    static constexpr size_t ciphertext_bytes  = 1088;   ///< KYBER_CIPHERTEXTBYTES
    static constexpr size_t shared_secret_bytes = 32;   ///< KYBER_SSBYTES
};

// ---------------------------------------------------------------------------
// KyberKeypair
// ---------------------------------------------------------------------------

/// ML-KEM keypair produced by MlKem::generate_keypair().
/// pk: public key — share with the encapsulator.
/// sk: secret key — keep private; needed for decapsulation.
struct KyberKeypair {
    std::vector<uint8_t> pk;   ///< public key  (1184 bytes for Kyber-768)
    std::vector<uint8_t> sk;   ///< secret key  (2400 bytes for Kyber-768; private!)
};

// ---------------------------------------------------------------------------
// KyberEncapResult
// ---------------------------------------------------------------------------

/// Result of MlKem::encapsulate().
/// ct: ciphertext — send to the owner of the corresponding secret key.
/// ss: shared secret — do NOT send; use as a symmetric key locally.
struct KyberEncapResult {
    std::vector<uint8_t> ct;   ///< ciphertext (1088 bytes for Kyber-768)
    std::vector<uint8_t> ss;   ///< shared secret (32 bytes; keep local!)
};

// ---------------------------------------------------------------------------
// MlKem — static facade over the Kyber-768 C API
// ---------------------------------------------------------------------------

/// Post-quantum Key Encapsulation Mechanism (FIPS 203 / ML-KEM-768).
///
/// All methods are static; no state — construct nothing.
///
/// Error handling: throws std::runtime_error on C-level failure (return != 0).
/// In practice the Kyber reference implementation never fails on valid inputs,
/// so errors indicate programmer error (null buffers, corrupt key material).
class MlKem {
public:
    // -----------------------------------------------------------------------
    // generate_keypair
    // -----------------------------------------------------------------------

    /// Generate a new Kyber-768 keypair using OS randomness.
    ///
    /// @returns KyberKeypair with pk (1184 B) and sk (2400 B).
    /// @throws  std::runtime_error if the C keygen call fails.
    static KyberKeypair generate_keypair() {
        KyberKeypair kp;
        kp.pk.resize(MlKemSizes::public_key_bytes);
        kp.sk.resize(MlKemSizes::secret_key_bytes);

        int rc = pqcrystals_kyber768_ref_keypair(kp.pk.data(), kp.sk.data());
        if (rc != 0) {
            throw std::runtime_error(
                "MlKem::generate_keypair — keypair returned " +
                std::to_string(rc));
        }
        return kp;
    }

    // -----------------------------------------------------------------------
    // encapsulate
    // -----------------------------------------------------------------------

    /// Encapsulate: produce a ciphertext and shared secret from a public key.
    ///
    /// Called by the *sender* / KEM initiator.
    ///
    /// @param pk  Recipient's public key (must be MlKemSizes::public_key_bytes).
    /// @returns KyberEncapResult { ct, ss }.
    ///          - ct: send to recipient over any channel (not secret).
    ///          - ss: local shared secret — do NOT transmit; use as symmetric key.
    /// @throws  std::invalid_argument if pk has wrong size.
    /// @throws  std::runtime_error    if the C encapsulation call fails.
    static KyberEncapResult encapsulate(const std::vector<uint8_t>& pk) {
        if (pk.size() != MlKemSizes::public_key_bytes) {
            throw std::invalid_argument(
                "MlKem::encapsulate — pk must be " +
                std::to_string(MlKemSizes::public_key_bytes) + " bytes, got " +
                std::to_string(pk.size()));
        }

        KyberEncapResult result;
        result.ct.resize(MlKemSizes::ciphertext_bytes);
        result.ss.resize(MlKemSizes::shared_secret_bytes);

        int rc = pqcrystals_kyber768_ref_enc(result.ct.data(), result.ss.data(), pk.data());
        if (rc != 0) {
            throw std::runtime_error(
                "MlKem::encapsulate — enc returned " +
                std::to_string(rc));
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // decapsulate
    // -----------------------------------------------------------------------

    /// Decapsulate: recover the shared secret from a ciphertext and secret key.
    ///
    /// Called by the *recipient* / KEM responder.
    ///
    /// @param ct  Ciphertext received from the encapsulator
    ///            (must be MlKemSizes::ciphertext_bytes).
    /// @param sk  Own secret key (must be MlKemSizes::secret_key_bytes).
    /// @returns   Shared secret (32 bytes).  Matches encapsulator's ss iff sk
    ///            corresponds to the pk used for encapsulation.
    /// @throws    std::invalid_argument if ct or sk have wrong sizes.
    /// @throws    std::runtime_error    if the C decapsulation call fails.
    static std::vector<uint8_t> decapsulate(const std::vector<uint8_t>& ct,
                                            const std::vector<uint8_t>& sk) {
        if (ct.size() != MlKemSizes::ciphertext_bytes) {
            throw std::invalid_argument(
                "MlKem::decapsulate — ct must be " +
                std::to_string(MlKemSizes::ciphertext_bytes) + " bytes, got " +
                std::to_string(ct.size()));
        }
        if (sk.size() != MlKemSizes::secret_key_bytes) {
            throw std::invalid_argument(
                "MlKem::decapsulate — sk must be " +
                std::to_string(MlKemSizes::secret_key_bytes) + " bytes, got " +
                std::to_string(sk.size()));
        }

        std::vector<uint8_t> ss(MlKemSizes::shared_secret_bytes);
        int rc = pqcrystals_kyber768_ref_dec(ss.data(), ct.data(), sk.data());
        if (rc != 0) {
            throw std::runtime_error(
                "MlKem::decapsulate — dec returned " +
                std::to_string(rc));
        }
        return ss;
    }

    // -----------------------------------------------------------------------
    // constant_time_compare (utility for tests / verification)
    // -----------------------------------------------------------------------

    /// Constant-time comparison of two byte vectors — avoids timing side channels.
    /// @returns true iff a.size() == b.size() and all bytes match.
    static bool constant_time_equal(const std::vector<uint8_t>& a,
                                    const std::vector<uint8_t>& b) noexcept {
        if (a.size() != b.size()) return false;
        uint8_t diff = 0;
        for (size_t i = 0; i < a.size(); ++i) diff |= (a[i] ^ b[i]);
        return diff == 0;
    }

    // -----------------------------------------------------------------------
    // algorithm_name / sizes (introspection)
    // -----------------------------------------------------------------------

    static constexpr const char* algorithm_name() noexcept { return "Kyber768 (ML-KEM-768, FIPS 203)"; }
    static constexpr size_t public_key_bytes()   noexcept { return MlKemSizes::public_key_bytes;   }
    static constexpr size_t secret_key_bytes()   noexcept { return MlKemSizes::secret_key_bytes;   }
    static constexpr size_t ciphertext_bytes()   noexcept { return MlKemSizes::ciphertext_bytes;   }
    static constexpr size_t shared_secret_bytes()noexcept { return MlKemSizes::shared_secret_bytes;}

    // Non-instantiable
    MlKem() = delete;
};

} // namespace nikola::security
