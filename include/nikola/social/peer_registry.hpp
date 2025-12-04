#pragma once

#include "nikola/social/membrane.hpp"
#include <string>
#include <unordered_map>
#include <memory>

namespace nikola::social {

struct PeerInfo {
    std::string peer_id;
    std::string public_key_z85;
    double last_resonance;
    int interaction_count;
    std::unique_ptr<SocialMembrane> membrane;
};

/**
 * @brief Registry of known peer instances
 *
 * Maintains list of authorized peers for IRSP communication.
 * Each peer has:
 * - Unique ID (derived from torus seed)
 * - Public key (CurveZMQ)
 * - Trust membrane
 * - Interaction history
 *
 * @status STUB - Implementation deferred to Phase 3
 */
class PeerRegistry {
public:
    /**
     * @brief Add a new peer to registry
     * @param peer_id Unique identifier
     * @param public_key_z85 CurveZMQ public key
     */
    void add_peer(const std::string& peer_id, const std::string& public_key_z85);

    /**
     * @brief Remove peer from registry
     * @param peer_id Peer to remove
     */
    void remove_peer(const std::string& peer_id);

    /**
     * @brief Get membrane for specific peer
     * @param peer_id Peer identifier
     * @return Pointer to membrane, or nullptr if not found
     */
    SocialMembrane* get_membrane(const std::string& peer_id);

    /**
     * @brief Record interaction with peer
     * @param peer_id Peer identifier
     * @param resonance Measured resonance value
     */
    void record_interaction(const std::string& peer_id, double resonance);

    /**
     * @brief Get list of all peer IDs
     * @return Vector of peer identifiers
     */
    std::vector<std::string> get_all_peers() const;

private:
    std::unordered_map<std::string, PeerInfo> peers_;
};

} // namespace nikola::social
