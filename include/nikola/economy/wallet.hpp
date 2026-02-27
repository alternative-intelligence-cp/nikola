#pragma once
/**
 * @file wallet.hpp
 * @brief Phase 131 — NeuralWallet: identity + simulated payment layer
 *
 * NeuralWallet is a pure-virtual interface for Nikola's economic identity.
 * In production it would wrap real ECC key-pairs and on-chain signing; for now
 * only SimulatedWallet is used — a deterministic mock that derives an
 * Ethereum-style hex address from a seed string (no TorusManifold required).
 * This layer is intentionally "wallet-shaped" so future real impls drop in.
 */

#include <string>
#include <cstdint>
#include <unordered_map>

namespace nikola::economy {

// ---------------------------------------------------------------------------
// NeuralWallet (abstract interface)
// ---------------------------------------------------------------------------

/**
 * @brief Abstract identity + signing interface for Nikola's economic layer.
 *
 * derive_identity() binds the wallet to a seed.  The seed can be anything
 * (a torus initialisation value, a UUID, a passphrase hash).
 */
class NeuralWallet {
public:
    virtual ~NeuralWallet() = default;

    /**
     * @brief Bind wallet identity to a seed string.
     * @param seed Arbitrary bytes encoded as hex or printable string.
     * @return Derived private-key string (representation only).
     */
    virtual std::string derive_identity(const std::string& seed) = 0;

    /** @return Public wallet address (Ethereum-style 0x… hex). */
    virtual std::string get_address() const = 0;

    /** @brief Sign arbitrary data; returns signature string. */
    virtual std::string sign(const std::string& data) = 0;

    /**
     * @brief Verify a signature.
     * @param data      Original signed data.
     * @param signature Signature returned by sign().
     * @param address   Claimed signer address.
     * @return true if valid.
     */
    virtual bool verify(const std::string& data,
                        const std::string& signature,
                        const std::string& address) = 0;

    /** @return Current balance in Wei-equivalent units. */
    virtual uint64_t get_balance_wei() const = 0;

    /** @brief Credit the wallet with amount_wei units. */
    virtual void credit(uint64_t amount_wei) = 0;

    /**
     * @brief Debit amount_wei.
     * @return true on success; false if insufficient funds.
     */
    virtual bool debit(uint64_t amount_wei) = 0;
};

// ---------------------------------------------------------------------------
// SimulatedWallet
// ---------------------------------------------------------------------------

/**
 * @brief Deterministic mock wallet for testing and local simulation.
 *
 * derive_identity(seed) computes a FNV-1a-inspired address from the seed.
 * sign(data) returns a deterministic string keyed on (address, data).
 * verify() checks the expected signature format rather than real ECDSA.
 */
class SimulatedWallet : public NeuralWallet {
public:
    explicit SimulatedWallet() = default;

    // -- NeuralWallet interface ---

    std::string derive_identity(const std::string& seed) override;
    std::string get_address() const override;
    std::string sign(const std::string& data) override;
    bool verify(const std::string& data,
                const std::string& signature,
                const std::string& address) override;

    uint64_t get_balance_wei() const override;
    void     credit(uint64_t amount_wei) override;
    bool     debit(uint64_t amount_wei)  override;

    // -- SimulatedWallet extras ---

    /** @return true if derive_identity() has been called. */
    bool has_identity() const;

    /** @return Number of successful sign() calls */
    size_t sign_count() const;

    /** @return Number of successful (returning true) verify() calls */
    size_t verify_count() const;

    /** @return Number of successful debit() calls */
    size_t debit_count() const;

    // -- Static helpers ---

    /**
     * @brief Derive a 40-char hex address from a seed string (FNV-1a based).
     * Deterministic: same seed → same address across runs.
     */
    static std::string derive_address(const std::string& seed);

    /**
     * @brief Build the expected signature token for (address, data).
     * Format: "sig_<addr[:8]>_<data[:8]>"
     */
    static std::string build_expected_sig(const std::string& address,
                                           const std::string& data);

private:
    std::string address_;         ///< Derived 0x… address (empty until bound)
    std::string private_key_;     ///< Derived private-key string
    uint64_t    balance_wei_ = 0; ///< Simulated balance
    size_t      sign_count_    = 0;
    size_t      verify_count_  = 0;
    size_t      debit_count_   = 0;
};

} // namespace nikola::economy
