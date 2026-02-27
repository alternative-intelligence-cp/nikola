#pragma once
/**
 * @file peer_registry.hpp
 * @brief Phase 133 — PeerRegistry: roster of known Nikola peers
 *
 * Maintains a map of authorized peer instances for IRSP communication.
 * Each peer has: a unique ID, a public key (CurveZMQ z85), a trust membrane
 * (SocialMembrane), interaction history, and a last-seen resonance value.
 *
 * record_interaction(id, resonance) drives membrane trust — resonance > 0.5
 * is treated as a positive interaction, ≤ 0.5 as negative.
 */

#include "nikola/social/membrane.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <optional>
#include <functional>
#include <cstdint>

namespace nikola::social {

/// Maximum peers held in registry (oldest evicted on add when at cap)
inline constexpr size_t PEER_REGISTRY_MAX = 128;
/// Resonance threshold above which an interaction is treated as positive
inline constexpr double PEER_RESONANCE_THRESHOLD = 0.50;

// ---------------------------------------------------------------------------
// PeerInfo
// ---------------------------------------------------------------------------

struct PeerInfo {
    std::string                  peer_id;           ///< Unique peer identifier
    std::string                  public_key_z85;    ///< CurveZMQ public key (z85)
    double                       last_resonance = 0.0;
    int                          interaction_count  = 0;
    uint64_t                     first_seen_tick    = 0;
    uint64_t                     last_seen_tick     = 0;
    std::unique_ptr<SocialMembrane> membrane;
};

// ---------------------------------------------------------------------------
// PeerRegistry
// ---------------------------------------------------------------------------

class PeerRegistry {
public:
    PeerRegistry() = default;

    // non-copyable (contains unique_ptr)
    PeerRegistry(const PeerRegistry&)            = delete;
    PeerRegistry& operator=(const PeerRegistry&) = delete;
    PeerRegistry(PeerRegistry&&)                 = default;
    PeerRegistry& operator=(PeerRegistry&&)      = default;

    // -----------------------------------------------------------------------
    // Peer management
    // -----------------------------------------------------------------------

    /**
     * @brief Register a peer.
     * If the peer already exists, their public key is updated; the membrane
     * and interaction history are preserved.  If at PEER_REGISTRY_MAX,
     * the least-recently-seen peer is evicted first.
     * @param tick  Current simulation tick (recorded as first_seen or updated).
     */
    void add_peer(const std::string& peer_id,
                  const std::string& public_key_z85,
                  uint64_t tick = 0);

    /// Remove a peer; no-op if not found.
    void remove_peer(const std::string& peer_id);

    /// true if the peer_id exists in registry.
    [[nodiscard]] bool has_peer(const std::string& peer_id) const;

    /// Const pointer to PeerInfo; nullptr if not found.
    [[nodiscard]] const PeerInfo* find_peer(const std::string& peer_id) const;

    /// Raw pointer to membrane; nullptr if peer not found.
    [[nodiscard]] SocialMembrane* get_membrane(const std::string& peer_id);

    // -----------------------------------------------------------------------
    // Interaction recording
    // -----------------------------------------------------------------------

    /**
     * @brief Record a resonance measurement with a peer.
     *
     * - Updates last_resonance and last_seen_tick.
     * - Increments interaction_count.
     * - Calls membrane.update_trust(resonance > PEER_RESONANCE_THRESHOLD).
     * No-op if peer not found.
     */
    void record_interaction(const std::string& peer_id,
                             double resonance,
                             uint64_t tick = 0);

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// All registered peer IDs.
    [[nodiscard]] std::vector<std::string> get_all_peers() const;

    /// Number of registered peers.
    [[nodiscard]] size_t peer_count() const;

    /// Peer with the highest membrane trust_score (nullopt if empty).
    [[nodiscard]] std::optional<std::string> most_trusted_peer() const;

    /// Peers sorted by membrane trust (descending).
    [[nodiscard]] std::vector<std::string> peers_by_trust() const;

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    struct Stats {
        size_t  total_peers        = 0;
        size_t  total_interactions = 0;
        double  mean_resonance     = 0.0;
        double  mean_trust         = 0.0;
        std::string most_trusted;   ///< peer_id, empty if none
    };

    [[nodiscard]] Stats stats() const;

    // -----------------------------------------------------------------------
    // Management
    // -----------------------------------------------------------------------

    /// Remove all peers.
    void clear();

    // -----------------------------------------------------------------------
    // Callbacks
    // -----------------------------------------------------------------------

    using OnInteraction = std::function<void(const std::string& peer_id,
                                              double resonance)>;
    void on_interaction(OnInteraction cb) { on_interaction_cb_ = std::move(cb); }

private:
    std::unordered_map<std::string, PeerInfo> peers_;
    size_t total_interactions_ = 0;
    OnInteraction on_interaction_cb_;

    void evict_least_recently_seen();
};

} // namespace nikola::social
