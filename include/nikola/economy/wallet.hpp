#pragma once

#include <string>
#include <memory>

namespace nikola::economy {

// Forward declaration
class TorusManifold;

/**
 * @brief Neural Wallet interface for blockchain identity
 *
 * Derives identity from torus geometry seed (SHA-256 hash).
 * The same initialization seed that creates the 9D toroidal topology
 * also serves as the private key, ensuring identity is tied to memory structure.
 *
 * Benefits:
 * - No separate key management
 * - Identity persists across restarts (if torus seed is stored)
 * - Cryptographic proof of unique neural topology
 *
 * @status STUB - Implementation deferred to Phase 4
 */
class NeuralWallet {
public:
    virtual ~NeuralWallet() = default;

    /**
     * @brief Derive wallet identity from torus geometry
     * @param torus The 9D toroidal manifold
     * @return Private key (SHA-256 of seed)
     */
    virtual std::string derive_identity(const TorusManifold& torus) = 0;

    /**
     * @brief Get public wallet address
     * @return Ethereum-compatible address
     */
    virtual std::string get_address() const = 0;

    /**
     * @brief Sign data with private key
     * @param data Data to sign
     * @return ECDSA signature
     */
    virtual std::string sign(const std::string& data) = 0;

    /**
     * @brief Verify signature
     * @param data Original data
     * @param signature Signature to verify
     * @param address Claimed signer address
     * @return true if valid
     */
    virtual bool verify(const std::string& data, const std::string& signature,
                       const std::string& address) = 0;
};

/**
 * @brief Simulated wallet for testing (no real blockchain)
 *
 * Implements NeuralWallet interface with mock operations.
 * Used during Phase 0-3 before Polygon CDK integration.
 *
 * @status ACTIVE STUB
 */
class SimulatedWallet : public NeuralWallet {
public:
    SimulatedWallet() = default;

    std::string derive_identity(const TorusManifold& torus) override;
    std::string get_address() const override;
    std::string sign(const std::string& data) override;
    bool verify(const std::string& data, const std::string& signature,
               const std::string& address) override;

private:
    std::string mock_address_ = "0x0000000000000000000000000000000000000000";
    std::string mock_private_key_ = "mock_key";
};

} // namespace nikola::economy
