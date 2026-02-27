/**
 * @file membrane.cpp
 * @brief Phase 132 — SocialMembrane implementation
 */

#include "nikola/social/membrane.hpp"

#include <algorithm>
#include <cmath>

namespace nikola::social {

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

double SocialMembrane::compute_permeability(double trust, double dissonance) {
    const double raw = trust / (dissonance + MEMBRANE_EPSILON);
    return std::clamp(raw, MEMBRANE_MIN_PERMEABILITY, MEMBRANE_MAX_PERMEABILITY);
}

// ---------------------------------------------------------------------------
// Wave filtering
// ---------------------------------------------------------------------------

std::complex<double>
SocialMembrane::filter_incoming(const std::complex<double>& friend_wave,
                                 const std::complex<double>& self_wave) const {
    // Linear interpolation: result = self + permeability * (friend - self)
    // permeability=1.0 → full friend_wave; permeability=0.0 → self_wave
    return self_wave + permeability_ * (friend_wave - self_wave);
}

// ---------------------------------------------------------------------------
// Trust / dissonance updates
// ---------------------------------------------------------------------------

void SocialMembrane::update_trust(bool positive_interaction) {
    if (positive_interaction) {
        trust_score_ = std::min(1.0, trust_score_ + MEMBRANE_TRUST_STEP);
        ++positive_count_;
    } else {
        trust_score_ = std::max(0.0, trust_score_ - MEMBRANE_TRUST_STEP);
        ++negative_count_;
    }
    recalculate_permeability();
}

void SocialMembrane::update_dissonance(double delta) {
    dissonance_ = std::clamp(dissonance_ + delta, 0.0, 1.0);
    recalculate_permeability();
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

size_t SocialMembrane::positive_interaction_count() const { return positive_count_; }
size_t SocialMembrane::negative_interaction_count() const { return negative_count_; }
size_t SocialMembrane::interaction_count() const {
    return positive_count_ + negative_count_;
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

void SocialMembrane::set_permeability(double value) {
    permeability_ = std::clamp(value, MEMBRANE_MIN_PERMEABILITY, MEMBRANE_MAX_PERMEABILITY);
}

void SocialMembrane::set_trust(double value) {
    trust_score_ = std::clamp(value, 0.0, 1.0);
    recalculate_permeability();
}

void SocialMembrane::set_dissonance(double value) {
    dissonance_ = std::clamp(value, 0.0, 1.0);
    recalculate_permeability();
}

void SocialMembrane::reset() {
    permeability_   = 0.1;
    trust_score_    = 0.5;
    dissonance_     = 0.5;
    positive_count_ = 0;
    negative_count_ = 0;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

SocialMembrane::Stats SocialMembrane::stats() const {
    return Stats {
        trust_score_,
        dissonance_,
        permeability_,
        positive_count_,
        negative_count_
    };
}

// ---------------------------------------------------------------------------
// Private
// ---------------------------------------------------------------------------

void SocialMembrane::recalculate_permeability() {
    permeability_ = compute_permeability(trust_score_, dissonance_);
}

} // namespace nikola::social
