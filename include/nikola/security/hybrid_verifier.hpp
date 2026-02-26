// =============================================================================
// NIKOLA — Phase 114 / GAP-047
// HybridVerifier — Ed25519 fast-path + SPHINCS+ quantum-safe verification
// =============================================================================
// Spec   : GAP-047 §4.1 "Hybrid Signature Architecture"
//          docs/info/integration/sections/05_autonomous_systems/04_self_improvement.md
// Author : Nikola Phase 114
// License: MIT
//
// Combines two independent signature schemes:
//   1. Ed25519 (libsodium)  — fast-path, classical security, ~50 µs verify
//   2. SPHINCS+-shake-256f  — slow-path, post-quantum security, ~1 ms verify
//
// BOTH signatures must be valid for a module to be accepted. If Ed25519 fails
// we reject immediately (DoS protection). If SPHINCS+ fails we reject.
//
// Successful verifications are cached by FNV-1a(binary) to avoid re-running
// the expensive SPHINCS+ path on repeated loads of the same module image.
//
// Key sizes (compile-time constants):
//   ED25519_SIG_BYTES = 64   (crypto_sign_BYTES)
//   ED25519_PUB_BYTES = 32   (crypto_sign_PUBLICKEYBYTES)
//   SPHINCS_PUB_BYTES = 64   (SphincsSigner::public_key_bytes())
//
// Thread-safety: verify_module() and the cache accessors are protected by an
// internal mutex — safe to call from multiple threads concurrently.
// =============================================================================
#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace nikola::security {

// ---------------------------------------------------------------------------
// HybridSignature
// ---------------------------------------------------------------------------

/// Dual-signature container for a module binary.
/// Both fields must be populated for verify_module() to succeed.
struct HybridSignature {
    std::vector<uint8_t> ed25519_sig;  ///< 64-byte Ed25519 detached signature
    std::vector<uint8_t> sphincs_sig;  ///< ≤49856-byte SPHINCS+-shake-256f signature
};

// ---------------------------------------------------------------------------
// VerifyFailReason
// ---------------------------------------------------------------------------

/// Detailed failure reason returned by last_failure().
enum class VerifyFailReason : int {
    NONE,                ///< No failure (last call succeeded or no call yet)
    ED25519_BAD_PUBKEY,  ///< ed_pub was the wrong size (must be 32 bytes)
    ED25519_BAD_SIG,     ///< ed25519_sig was the wrong size (must be 64 bytes)
    ED25519_INVALID,     ///< Ed25519 signature verification failed
    SPHINCS_BAD_PUBKEY,  ///< sphincs_pub was the wrong size
    SPHINCS_BAD_SIG,     ///< sphincs_sig was empty
    SPHINCS_INVALID,     ///< SPHINCS+ signature verification failed
};

/// Returns a human-readable string for a VerifyFailReason.
[[nodiscard]] constexpr std::string_view verify_fail_str(VerifyFailReason r) noexcept {
    switch (r) {
        case VerifyFailReason::NONE:               return "NONE";
        case VerifyFailReason::ED25519_BAD_PUBKEY: return "ED25519_BAD_PUBKEY";
        case VerifyFailReason::ED25519_BAD_SIG:    return "ED25519_BAD_SIG";
        case VerifyFailReason::ED25519_INVALID:    return "ED25519_INVALID";
        case VerifyFailReason::SPHINCS_BAD_PUBKEY: return "SPHINCS_BAD_PUBKEY";
        case VerifyFailReason::SPHINCS_BAD_SIG:    return "SPHINCS_BAD_SIG";
        case VerifyFailReason::SPHINCS_INVALID:    return "SPHINCS_INVALID";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// HybridVerifier
// ---------------------------------------------------------------------------

/// Stateful hybrid signature verifier (Ed25519 + SPHINCS+-shake-256f).
///
/// Intended to be long-lived — keep one instance per deployment pipeline.
/// The internal cache accumulates over the verifier's lifetime; call
/// clear_cache() if the trusted-module whitelist needs to be invalidated.
class HybridVerifier {
public:
    // -----------------------------------------------------------------------
    // Compile-time key / signature size constants
    // -----------------------------------------------------------------------

    /// Ed25519 detached signature size in bytes (crypto_sign_BYTES = 64).
    static constexpr size_t ED25519_SIG_BYTES = 64u;

    /// Ed25519 public key size in bytes (crypto_sign_PUBLICKEYBYTES = 32).
    static constexpr size_t ED25519_PUB_BYTES = 32u;

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /// Construct a HybridVerifier.
    /// Calls sodium_init() internally — safe to call multiple times.
    HybridVerifier();

    // Non-copyable, non-movable (std::mutex is neither copyable nor movable)
    HybridVerifier(const HybridVerifier&)            = delete;
    HybridVerifier& operator=(const HybridVerifier&) = delete;
    HybridVerifier(HybridVerifier&&)                 = delete;
    HybridVerifier& operator=(HybridVerifier&&)      = delete;

    // -----------------------------------------------------------------------
    // Core verification
    // -----------------------------------------------------------------------

    /// Verify a module binary against a HybridSignature.
    ///
    /// Verification flow:
    ///   1. Cache hit? → true immediately.
    ///   2. Validate key/sig sizes.
    ///   3. Ed25519 fast-path (reject on failure — DoS protection).
    ///   4. SPHINCS+ slow-path (reject on failure).
    ///   5. Add binary hash to cache.  Return true.
    ///
    /// Thread-safe: protected internally by a mutex.
    ///
    /// @param binary     Raw bytes of the module being verified (e.g., .so content).
    /// @param sig        Dual signature container.
    /// @param ed_pub     32-byte Ed25519 public key.
    /// @param sphincs_pub  64-byte SPHINCS+-shake-256f public key.
    /// @returns true iff both signatures verified successfully (or cache hit).
    [[nodiscard]] bool verify_module(
            const std::vector<uint8_t>& binary,
            const HybridSignature&       sig,
            const std::vector<uint8_t>& ed_pub,
            const std::vector<uint8_t>& sphincs_pub) noexcept;

    // -----------------------------------------------------------------------
    // Failure / diagnostics
    // -----------------------------------------------------------------------

    /// Reason for the most recent failed verify_module() call.
    /// Returns NONE if the last call succeeded or if no call has been made.
    [[nodiscard]] VerifyFailReason last_failure() const noexcept;

    // -----------------------------------------------------------------------
    // Cache management
    // -----------------------------------------------------------------------

    /// Returns true if the binary's hash is in the verified cache.
    [[nodiscard]] bool is_cached(const std::vector<uint8_t>& binary) const noexcept;

    /// Remove all entries from the verified cache.
    void clear_cache() noexcept;

    /// Current number of entries in the verified cache.
    [[nodiscard]] size_t cache_size() const noexcept;

private:
    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    /// FNV-1a 64-bit hash of an arbitrary byte buffer.
    [[nodiscard]] static uint64_t fnv1a(const uint8_t* data, size_t len) noexcept;

    mutable std::mutex           cache_mutex_;
    std::unordered_set<uint64_t> verified_cache_;
    VerifyFailReason             last_failure_{VerifyFailReason::NONE};
};

} // namespace nikola::security
