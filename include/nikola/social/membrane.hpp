#pragma once
/**
 * @file membrane.hpp
 * @brief Phase 132 — SocialMembrane: trust-based wave filtering
 *
 * A SocialMembrane governs how much of a peer's signal is allowed to
 * influence Nikola's internal state.  Permeability is computed as:
 *
 *   permeability = trust_score / (dissonance + MEMBRANE_EPSILON)
 *
 * High trust + low dissonance → high permeability (waves pass)
 * Low trust + high dissonance → low permeability  (waves blocked)
 *
 * filter_incoming() blends friend_wave into self_wave proportional to
 * permeability.  Trust is updated incrementally via update_trust().
 */

#include <complex>
#include <cstdint>
#include <functional>
#include <string>

namespace nikola::social {

/// Minimum denominator to prevent divide-by-zero in permeability calc
inline constexpr double MEMBRANE_EPSILON           = 1e-6;
/// Step size for trust updates
inline constexpr double MEMBRANE_TRUST_STEP        = 0.08;
/// Step size for dissonance updates
inline constexpr double MEMBRANE_DISSONANCE_STEP   = 0.05;
/// Upper permeability clamp
inline constexpr double MEMBRANE_MAX_PERMEABILITY  = 1.0;
/// Lower permeability clamp
inline constexpr double MEMBRANE_MIN_PERMEABILITY  = 0.0;

// ---------------------------------------------------------------------------
// SocialMembrane
// ---------------------------------------------------------------------------

class SocialMembrane {
public:
    explicit SocialMembrane() = default;

    // -----------------------------------------------------------------------
    // Core wave filtering
    // -----------------------------------------------------------------------

    /**
     * @brief Filter an incoming peer wave through this membrane.
     *
     * result = self_wave + permeability * (friend_wave - self_wave)
     *
     * When permeability==1.0: result = friend_wave (full pass-through)
     * When permeability==0.0: result = self_wave   (fully blocked)
     *
     * @param friend_wave  Complex wave received from peer.
     * @param self_wave    Current internal complex wave state.
     * @return Filtered wave.
     */
    [[nodiscard]] std::complex<double>
    filter_incoming(const std::complex<double>& friend_wave,
                    const std::complex<double>& self_wave) const;

    // -----------------------------------------------------------------------
    // Trust / dissonance management
    // -----------------------------------------------------------------------

    /**
     * @brief Adjust trust based on interaction outcome.
     * @param positive true → trust += MEMBRANE_TRUST_STEP;
     *                 false → trust -= MEMBRANE_TRUST_STEP.
     * Trust is clamped to [0,1]; permeability is recalculated.
     */
    void update_trust(bool positive_interaction);

    /**
     * @brief Adjust dissonance level (e.g. from semantic divergence).
     * @param delta Positive → more dissonance; negative → less.
     * Dissonance is clamped to [0,1]; permeability is recalculated.
     */
    void update_dissonance(double delta);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    [[nodiscard]] double get_permeability() const { return permeability_; }
    [[nodiscard]] double get_trust()        const { return trust_score_;  }
    [[nodiscard]] double get_dissonance()   const { return dissonance_;   }

    /// Number of update_trust() calls with positive=true on this membrane
    [[nodiscard]] size_t positive_interaction_count() const;
    /// Number of update_trust() calls with positive=false
    [[nodiscard]] size_t negative_interaction_count() const;
    /// Total interactions (positive + negative)
    [[nodiscard]] size_t interaction_count() const;

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Force-set permeability (used in tests / manual override)
    void set_permeability(double value);

    /// Force-set trust score and recompute permeability
    void set_trust(double value);

    /// Force-set dissonance and recompute permeability
    void set_dissonance(double value);

    /// Reset to default state (trust=0.5, dissonance=0.5, permeability=0.1)
    void reset();

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    struct Stats {
        double trust_score    = 0.5;
        double dissonance     = 0.5;
        double permeability   = 0.1;
        size_t positive_count = 0;
        size_t negative_count = 0;
    };

    [[nodiscard]] Stats stats() const;

    // -----------------------------------------------------------------------
    // Static helpers
    // -----------------------------------------------------------------------

    /**
     * @brief Compute permeability from trust and dissonance.
     * permeability = clamp(trust / (dissonance + epsilon), 0, 1)
     */
    static double compute_permeability(double trust, double dissonance);

private:
    double permeability_   = 0.1;
    double trust_score_    = 0.5;
    double dissonance_     = 0.5;
    size_t positive_count_ = 0;
    size_t negative_count_ = 0;

    void recalculate_permeability();
};

} // namespace nikola::social
