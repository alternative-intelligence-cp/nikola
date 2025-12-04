#pragma once

#include <complex>
#include <string>

namespace nikola::social {

/**
 * @brief Social Membrane for trust-based wave filtering
 *
 * Filters incoming resonant packets from other instances based on trust.
 * Implements the permeability equation:
 *
 *   permeability = trust_score / (dissonance + epsilon)
 *
 * High trust + low dissonance = high permeability (waves pass easily)
 * Low trust + high dissonance = low permeability (waves blocked)
 *
 * @status STUB - Implementation deferred to Phase 3
 */
class SocialMembrane {
public:
    /**
     * @brief Filter incoming wave from another instance
     * @param friend_wave Wave received from peer
     * @param self_wave Current internal wave state
     * @return Filtered wave (attenuated by permeability)
     */
    std::complex<double> filter_incoming(
        const std::complex<double>& friend_wave,
        const std::complex<double>& self_wave
    );

    /**
     * @brief Update trust score based on interaction outcome
     * @param positive_interaction true if interaction was beneficial
     */
    void update_trust(bool positive_interaction);

    /**
     * @brief Get current permeability value
     * @return Permeability (0.0-1.0)
     */
    double get_permeability() const { return permeability_; }

    /**
     * @brief Set permeability manually (for testing)
     * @param value Permeability (0.0-1.0)
     */
    void set_permeability(double value) { permeability_ = value; }

private:
    double permeability_ = 0.1;  // Default: cautious
    double trust_score_ = 0.5;   // Neutral trust
    double dissonance_ = 0.5;    // Neutral dissonance
};

} // namespace nikola::social
