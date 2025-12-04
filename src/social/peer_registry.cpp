#include "nikola/social/peer_registry.hpp"

namespace nikola::social {

void PeerRegistry::add_peer(const std::string& peer_id, const std::string& public_key_z85) {
    // STUB: Minimal implementation - deferred to Phase 3
    PeerInfo info;
    info.peer_id = peer_id;
    info.public_key_z85 = public_key_z85;
    info.last_resonance = 0.0;
    info.interaction_count = 0;
    info.membrane = std::make_unique<SocialMembrane>();

    peers_[peer_id] = std::move(info);
}

void PeerRegistry::remove_peer(const std::string& peer_id) {
    // STUB: Minimal implementation - deferred to Phase 3
    peers_.erase(peer_id);
}

SocialMembrane* PeerRegistry::get_membrane(const std::string& peer_id) {
    // STUB: Returns nullptr if not found - implementation deferred to Phase 3
    auto it = peers_.find(peer_id);
    if (it != peers_.end()) {
        return it->second.membrane.get();
    }
    return nullptr;
}

void PeerRegistry::record_interaction(const std::string& peer_id, double resonance) {
    // STUB: No-op - implementation deferred to Phase 3
    auto it = peers_.find(peer_id);
    if (it != peers_.end()) {
        it->second.last_resonance = resonance;
        it->second.interaction_count++;
    }
}

std::vector<std::string> PeerRegistry::get_all_peers() const {
    // STUB: Returns empty vector - implementation deferred to Phase 3
    std::vector<std::string> result;
    for (const auto& [peer_id, info] : peers_) {
        result.push_back(peer_id);
    }
    return result;
}

} // namespace nikola::social
