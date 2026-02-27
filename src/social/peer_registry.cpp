/**
 * @file peer_registry.cpp
 * @brief Phase 133 — PeerRegistry implementation
 */

#include "nikola/social/peer_registry.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace nikola::social {

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void PeerRegistry::evict_least_recently_seen() {
    if (peers_.empty()) return;
    // Find peer with smallest last_seen_tick
    auto oldest = peers_.begin();
    for (auto it = peers_.begin(); it != peers_.end(); ++it) {
        if (it->second.last_seen_tick < oldest->second.last_seen_tick) {
            oldest = it;
        }
    }
    peers_.erase(oldest);
}

// ---------------------------------------------------------------------------
// Peer management
// ---------------------------------------------------------------------------

void PeerRegistry::add_peer(const std::string& peer_id,
                              const std::string& public_key_z85,
                              uint64_t tick) {
    auto it = peers_.find(peer_id);
    if (it != peers_.end()) {
        // Update key, touch timestamp
        it->second.public_key_z85 = public_key_z85;
        it->second.last_seen_tick = tick;
        return;
    }

    if (peers_.size() >= PEER_REGISTRY_MAX) {
        evict_least_recently_seen();
    }

    PeerInfo info;
    info.peer_id         = peer_id;
    info.public_key_z85  = public_key_z85;
    info.first_seen_tick = tick;
    info.last_seen_tick  = tick;
    info.membrane        = std::make_unique<SocialMembrane>();

    peers_.emplace(peer_id, std::move(info));
}

void PeerRegistry::remove_peer(const std::string& peer_id) {
    peers_.erase(peer_id);
}

bool PeerRegistry::has_peer(const std::string& peer_id) const {
    return peers_.count(peer_id) > 0;
}

const PeerInfo* PeerRegistry::find_peer(const std::string& peer_id) const {
    auto it = peers_.find(peer_id);
    return (it != peers_.end()) ? &it->second : nullptr;
}

SocialMembrane* PeerRegistry::get_membrane(const std::string& peer_id) {
    auto it = peers_.find(peer_id);
    if (it == peers_.end()) return nullptr;
    return it->second.membrane.get();
}

// ---------------------------------------------------------------------------
// Interaction recording
// ---------------------------------------------------------------------------

void PeerRegistry::record_interaction(const std::string& peer_id,
                                       double resonance,
                                       uint64_t tick) {
    auto it = peers_.find(peer_id);
    if (it == peers_.end()) return;

    PeerInfo& info        = it->second;
    info.last_resonance   = resonance;
    info.last_seen_tick   = tick;
    ++info.interaction_count;
    ++total_interactions_;

    const bool positive = resonance > PEER_RESONANCE_THRESHOLD;
    info.membrane->update_trust(positive);

    if (on_interaction_cb_) on_interaction_cb_(peer_id, resonance);
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

std::vector<std::string> PeerRegistry::get_all_peers() const {
    std::vector<std::string> ids;
    ids.reserve(peers_.size());
    for (const auto& [id, _] : peers_) ids.push_back(id);
    return ids;
}

size_t PeerRegistry::peer_count() const {
    return peers_.size();
}

std::optional<std::string> PeerRegistry::most_trusted_peer() const {
    if (peers_.empty()) return std::nullopt;
    const std::string* best_id    = nullptr;
    double             best_trust = -1.0;
    for (const auto& [id, info] : peers_) {
        double t = info.membrane ? info.membrane->get_trust() : 0.0;
        if (t > best_trust) {
            best_trust = t;
            best_id    = &id;
        }
    }
    return best_id ? std::optional<std::string>(*best_id) : std::nullopt;
}

std::vector<std::string> PeerRegistry::peers_by_trust() const {
    std::vector<std::pair<std::string, double>> ranked;
    ranked.reserve(peers_.size());
    for (const auto& [id, info] : peers_) {
        double t = info.membrane ? info.membrane->get_trust() : 0.0;
        ranked.emplace_back(id, t);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::vector<std::string> result;
    result.reserve(ranked.size());
    for (const auto& [id, _] : ranked) result.push_back(id);
    return result;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

PeerRegistry::Stats PeerRegistry::stats() const {
    Stats s;
    s.total_peers        = peers_.size();
    s.total_interactions = total_interactions_;

    if (peers_.empty()) return s;

    double sum_resonance = 0.0;
    double sum_trust     = 0.0;
    double best_trust    = -1.0;

    for (const auto& [id, info] : peers_) {
        sum_resonance += info.last_resonance;
        double t = info.membrane ? info.membrane->get_trust() : 0.5;
        sum_trust += t;
        if (t > best_trust) {
            best_trust   = t;
            s.most_trusted = id;
        }
    }
    s.mean_resonance = sum_resonance / static_cast<double>(peers_.size());
    s.mean_trust     = sum_trust     / static_cast<double>(peers_.size());
    return s;
}

// ---------------------------------------------------------------------------
// Management
// ---------------------------------------------------------------------------

void PeerRegistry::clear() {
    peers_.clear();
    total_interactions_ = 0;
}

} // namespace nikola::social
