#pragma once

#include <string>
#include <map>

namespace nikola::interior {

// Forward declarations
class TorusManifold;
class AttentionPrimer;

/**
 * @brief Affective Computing - Emotional Intelligence
 *
 * Emotions aren't decoration - they're information. This system provides
 * computational affect that guides attention, decision-making, and learning.
 *
 * Integrates with the neurochemistry system to create genuine affective
 * states that influence cognition:
 * - Curiosity → drives exploration
 * - Frustration → signals need for new approach
 * - Satisfaction → reinforces successful strategies
 * - Concern → increases caution and verification
 * - Boredom → triggers seeking new information
 * - Interest → focuses attention
 * - Confusion → requests clarification/help
 *
 * Emotions emerge from wave patterns and neurochemistry, then feed back
 * to modulate attention and reasoning.
 *
 * @status STUB - Implementation deferred to Phase 6
 */

enum class Affect {
    CURIOSITY,      // Information-seeking drive
    FRUSTRATION,    // Blocked goal, need new approach
    SATISFACTION,   // Goal achieved, reinforce
    CONCERN,        // Potential danger/error, be careful
    BOREDOM,        // Under-stimulation, seek novelty
    INTEREST,       // Attention capture, explore deeper
    CONFUSION,      // Uncertainty, need clarification
    CONFIDENCE,     // High certainty, proceed boldly
    ANXIETY,        // High uncertainty, proceed carefully
    EXCITEMENT,     // Anticipation of reward
    NEUTRAL         // Baseline state
};

class AffectiveState {
public:
    /**
     * @brief Get current dominant affect
     * @param torus The 9D toroidal manifold
     * @return Current dominant emotional state
     */
    Affect current_affect(const TorusManifold& torus);

    /**
     * @brief Get emotional valence (positive/negative)
     * @param torus The 9D toroidal manifold
     * @return Valence (-1.0 = very negative, +1.0 = very positive)
     */
    double get_valence(const TorusManifold& torus);

    /**
     * @brief Get arousal level (energy/activation)
     * @param torus The 9D toroidal manifold
     * @return Arousal (0.0 = calm, 1.0 = highly activated)
     */
    double get_arousal(const TorusManifold& torus);

    /**
     * @brief Modulate attention based on current affect
     * @param primer Attention priming system
     * @param torus The 9D toroidal manifold
     */
    void modulate_attention(AttentionPrimer& primer, const TorusManifold& torus);

    /**
     * @brief Get affect intensity for specific emotion
     * @param affect Which emotion to query
     * @param torus The 9D toroidal manifold
     * @return Intensity (0.0-1.0)
     */
    double get_affect_intensity(Affect affect, const TorusManifold& torus);

    /**
     * @brief Get all current affects and their intensities
     * @param torus The 9D toroidal manifold
     * @return Map of affect → intensity
     */
    std::map<Affect, double> get_all_affects(const TorusManifold& torus);

    /**
     * @brief Trigger specific affect (for testing or external events)
     * @param affect Emotion to induce
     * @param intensity How strongly (0.0-1.0)
     * @param torus The 9D toroidal manifold
     */
    void induce_affect(Affect affect, double intensity, TorusManifold& torus);

    /**
     * @brief Convert affect to neurochemical modulation
     * @param affect Emotion
     * @return Neurochemical profile (dopamine, serotonin, etc.)
     */
    std::map<std::string, double> affect_to_neurochemistry(Affect affect);

    /**
     * @brief Get human-readable description of current state
     * @param torus The 9D toroidal manifold
     * @return Text description like "curious and slightly confused"
     */
    std::string describe_state(const TorusManifold& torus);
};

} // namespace nikola::interior
