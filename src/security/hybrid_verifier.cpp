// =============================================================================
// NIKOLA — Phase 114 / GAP-047
// HybridVerifier implementation
// =============================================================================
// Spec   : §4.1 "Hybrid Signature Architecture"
//          docs/info/integration/sections/05_autonomous_systems/04_self_improvement.md
// Author : Nikola Phase 114
// License: MIT
//
// Dependencies:
//   OpenSSL (EVP) — Ed25519 detached verification; different API namespace
//                   from SPHINCS+ NaCl-style crypto_sign_* — no symbol clash
//   sphincsplus_shake256f — SPHINCS+-shake-256f reference implementation
//
// WHY OpenSSL instead of libsodium?
//   libsodium exports crypto_sign_verify() for Ed25519 using the NaCl naming
//   convention, which is identical to the names exported by the SPHINCS+
//   reference library.  Linking both in the same binary causes a symbol
//   collision.  OpenSSL uses the EVP_* naming family and has no overlap with
//   the SPHINCS+ NaCl API.
// =============================================================================

#include <nikola/security/hybrid_verifier.hpp>
#include <nikola/security/sphincs_signer.hpp>   // SphincsSigner (impl detail)

#include <openssl/evp.h>    // EVP_PKEY_new_raw_public_key, EVP_DigestVerify

#include <stdexcept>
#include <cstdint>

namespace nikola::security {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

HybridVerifier::HybridVerifier() {
    // No runtime initialisation needed for OpenSSL 3.x —
    // automatic initialisation via OPENSSL_init_crypto is performed
    // on first use.  Constructor is kept to preserve the API contract.
}

// ---------------------------------------------------------------------------
// FNV-1a 64-bit hash
// ---------------------------------------------------------------------------

uint64_t HybridVerifier::fnv1a(const uint8_t* data, size_t len) noexcept {
    // FNV-1a 64-bit: http://www.isthe.com/chongo/tech/comp/fnv/
    static constexpr uint64_t k_offset_basis = 14695981039346656037ULL;
    static constexpr uint64_t k_prime        = 1099511628211ULL;
    uint64_t h = k_offset_basis;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(data[i]);
        h *= k_prime;
    }
    return h;
}

// ---------------------------------------------------------------------------
// verify_module
// ---------------------------------------------------------------------------

bool HybridVerifier::verify_module(
        const std::vector<uint8_t>& binary,
        const HybridSignature&       sig,
        const std::vector<uint8_t>& ed_pub,
        const std::vector<uint8_t>& sphincs_pub) noexcept {

    // ── Step 1: cache look-up ─────────────────────────────────────────────
    {
        const uint64_t hash = fnv1a(binary.data(), binary.size());
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (verified_cache_.count(hash)) {
            last_failure_ = VerifyFailReason::NONE;
            return true;
        }
    }

    // ── Step 2: size validation ───────────────────────────────────────────

    // Ed25519 public key must be exactly 32 bytes.
    if (ed_pub.size() != ED25519_PUB_BYTES) {
        last_failure_ = VerifyFailReason::ED25519_BAD_PUBKEY;
        return false;
    }

    // Ed25519 signature must be exactly 64 bytes.
    if (sig.ed25519_sig.size() != ED25519_SIG_BYTES) {
        last_failure_ = VerifyFailReason::ED25519_BAD_SIG;
        return false;
    }

    // SPHINCS+ public key must match the compiled parameter set.
    const size_t expected_sphincs_pk = SphincsSigner::public_key_bytes();
    if (sphincs_pub.size() != expected_sphincs_pk) {
        last_failure_ = VerifyFailReason::SPHINCS_BAD_PUBKEY;
        return false;
    }

    // SPHINCS+ signature must be non-empty.
    if (sig.sphincs_sig.empty()) {
        last_failure_ = VerifyFailReason::SPHINCS_BAD_SIG;
        return false;
    }

    // ── Step 3: Ed25519 fast-path (OpenSSL EVP) ──────────────────────────
    // OpenSSL EVP uses EVP_PKEY_* / EVP_DigestVerify* API — no naming
    // conflict with SPHINCS+ NaCl-style crypto_sign_* symbols.
    {
        EVP_PKEY* pkey = EVP_PKEY_new_raw_public_key(
            EVP_PKEY_ED25519, nullptr,
            ed_pub.data(),
            static_cast<size_t>(ED25519_PUB_BYTES));

        if (!pkey) {
            last_failure_ = VerifyFailReason::ED25519_INVALID;
            return false;
        }

        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        if (!ctx) {
            EVP_PKEY_free(pkey);
            last_failure_ = VerifyFailReason::ED25519_INVALID;
            return false;
        }

        bool ed_ok = false;
        if (EVP_DigestVerifyInit(ctx, nullptr, nullptr, nullptr, pkey) == 1) {
            ed_ok = (EVP_DigestVerify(
                ctx,
                sig.ed25519_sig.data(), sig.ed25519_sig.size(),
                binary.data(), binary.size()) == 1);
        }

        EVP_MD_CTX_free(ctx);
        EVP_PKEY_free(pkey);

        if (!ed_ok) {
            last_failure_ = VerifyFailReason::ED25519_INVALID;
            return false;
        }
    }

    // ── Step 4: SPHINCS+ slow-path (sphincsplus_shake256f) ────────────────
    // Build a SphincsKeypair with only the public key set; the secret key
    // is not needed for verification.
    SphincsKeypair kp;
    kp.pk = sphincs_pub;

    const bool sphincs_ok = SphincsSigner::verify(
        SphincsSignature{sig.sphincs_sig},
        binary.data(),
        binary.size(),
        kp);

    if (!sphincs_ok) {
        last_failure_ = VerifyFailReason::SPHINCS_INVALID;
        return false;
    }

    // ── Step 5: both passed — add to cache and succeed ────────────────────
    {
        const uint64_t hash = fnv1a(binary.data(), binary.size());
        std::lock_guard<std::mutex> lock(cache_mutex_);
        verified_cache_.insert(hash);
        last_failure_ = VerifyFailReason::NONE;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Diagnostics / cache accessors
// ---------------------------------------------------------------------------

VerifyFailReason HybridVerifier::last_failure() const noexcept {
    return last_failure_;
}

bool HybridVerifier::is_cached(const std::vector<uint8_t>& binary) const noexcept {
    const uint64_t hash = fnv1a(binary.data(), binary.size());
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return verified_cache_.count(hash) > 0;
}

void HybridVerifier::clear_cache() noexcept {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    verified_cache_.clear();
}

size_t HybridVerifier::cache_size() const noexcept {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return verified_cache_.size();
}

} // namespace nikola::security
