#pragma once

/**
 * @file affective_state.hpp
 * @brief Affective Computing - Emotional Intelligence (Phase 120)
 *
 * Emotions are information, not decoration.  AffectiveState derives
 * computational affect from the ENGS neurochemical state vector
 * [D, S_approx, N_approx, ATP, boredom, entropy] and exposes:
 *
 *  - Russell circumplex coordinates: valence in [-1,+1], arousal in [0,1]
 *  - Soft membership scores for 11 discrete Affect labels (0-1 each)
 *  - Dominant affect selection (argmax over intensities + induced weights)
 *  - External induction (transient overrides that decay each update)
 *  - Neurochemical consequence table
 *  - Human-readable state description
 *
 * Derivation of latent variables from NikolaState:
 *   D      = state.dopamine
 *   ATP    = state.atp
 *   B      = state.boredom
 *   H      = state.entropy  (Shannon entropy of torus field)
 *   td     = state.td_error (reward prediction error)
 *   N_eff  = clamp(H / ENTROPY_AROUSAL_CEILING, 0, 1)  (arousal proxy)
 *   S_eff  = clamp(ATP * 0.7 + D * 0.3, 0, 1)          (stability proxy)
 *
 * @namespace nikola::interior
 */

#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <string>
#include <stdexcept>

#include <nikola/autonomy/decision_loop.hpp>

namespace nikola::interior {

// ---------------------------------------------------------------------------
// Affect enum  (11 states)
// ---------------------------------------------------------------------------

enum class Affect : int {
    CURIOSITY    = 0,  ///< Information-seeking drive  (high boredom, moderate entropy)
    FRUSTRATION  = 1,  ///< Blocked goal               (low dopamine, negative td_error)
    SATISFACTION = 2,  ///< Goal achieved               (high dopamine, positive td_error)
    CONCERN      = 3,  ///< Potential error/danger      (negative td_error, high entropy)
    BOREDOM      = 4,  ///< Under-stimulation           (high boredom, low entropy)
    INTEREST     = 5,  ///< Attention capture           (moderate dopamine, high entropy)
    CONFUSION    = 6,  ///< Deep uncertainty            (low dopamine, very high entropy)
    CONFIDENCE   = 7,  ///< High certainty              (high dopamine, high ATP)
    ANXIETY      = 8,  ///< Resource threat             (low ATP, high entropy)
    EXCITEMENT   = 9,  ///< Anticipated reward          (high dopamine, high entropy, pos td)
    NEUTRAL      = 10  ///< Baseline equilibrium
};

static constexpr int AFFECT_COUNT = 11;

/// Convert Affect to human-readable label.
[[nodiscard]] inline const char* affect_name(Affect a) noexcept {
    static constexpr const char* kNames[AFFECT_COUNT] = {
        "curiosity", "frustration", "satisfaction", "concern",
        "boredom",   "interest",    "confusion",    "confidence",
        "anxiety",   "excitement",  "neutral"
    };
    auto idx = static_cast<int>(a);
    return (idx >= 0 && idx < AFFECT_COUNT) ? kNames[idx] : "unknown";
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Entropy ceiling at which arousal proxy saturates.
inline constexpr double ENTROPY_AROUSAL_CEILING = 3.0;
/// Induced affect decay rate per update() call (multiplicative).
inline constexpr double INDUCED_AFFECT_DECAY    = 0.85;
/// Minimum induced weight below which it is zeroed.
inline constexpr double INDUCED_AFFECT_MIN      = 0.01;
/// Homeostatic dopamine equilibrium.
inline constexpr double DOPAMINE_EQUILIBRIUM    = 0.5;
/// ATP floor below which anxiety scoring ramps sharply.
inline constexpr double ATP_ANXIETY_THRESHOLD   = 0.20;
/// Boredom level above which curiosity / boredom scoring activates.
inline constexpr double BOREDOM_THRESHOLD       = 0.50;

// ---------------------------------------------------------------------------
// AffectiveState
// ---------------------------------------------------------------------------

/**
 * @brief Maintains continuous affective state derived from ENGS neurochemistry.
 *
 * Thread-safety: NOT thread-safe.  Wire via DecisionLoop::on_tick from
 * a single owner thread.
 */
class AffectiveState {
public:
    // -- types ---------------------------------------------------------------

    /// Soft membership intensities for each Affect label, in [0, 1].
    using IntensityMap = std::array<double, AFFECT_COUNT>;

    /// Optional callback fired whenever dominant affect changes.
    using OnAffectChange = std::function<void(Affect prev, Affect next,
                                              double intensity)>;

    // -- construction --------------------------------------------------------

    AffectiveState()  noexcept;
    ~AffectiveState() noexcept = default;

    AffectiveState(const AffectiveState&)            = default;
    AffectiveState& operator=(const AffectiveState&) = default;
    AffectiveState(AffectiveState&&)                 = default;
    AffectiveState& operator=(AffectiveState&&)      = default;

    // -- primary interface ---------------------------------------------------

    /**
     * @brief Recompute all affective scores from a NikolaState snapshot.
     * Decays induced weights, updates valence/arousal, recomputes dominant.
     * Wire via DecisionLoop::on_tick for continuous affect tracking.
     */
    void update(const nikola::autonomy::NikolaState& s) noexcept;

    // -- circumplex coordinates ----------------------------------------------

    /** Hedonic valence derived from [D, td_error, ATP]. Range [-1, +1]. */
    [[nodiscard]] double valence() const noexcept { return valence_; }

    /** Activation arousal derived from [entropy, boredom]. Range [0, 1]. */
    [[nodiscard]] double arousal() const noexcept { return arousal_; }

    // -- discrete affect queries ---------------------------------------------

    /// Current dominant Affect (highest total intensity after induction).
    [[nodiscard]] Affect current_affect() const noexcept;

    /// Intensity of one specific Affect label, in [0, 1].
    [[nodiscard]] double get_affect_intensity(Affect a) const noexcept;

    /// Full intensity array - indexed by static_cast<int>(Affect).
    [[nodiscard]] const IntensityMap& intensities() const noexcept { return scores_; }

    /// std::map version (testing / scripting convenience).
    [[nodiscard]] std::map<Affect, double> get_all_affects() const;

    // -- external induction --------------------------------------------------

    /**
     * @brief Transiently induce an affect with given weight.
     *
     * The induced weight is added on top of the computed score and decays
     * each update() by INDUCED_AFFECT_DECAY until < INDUCED_AFFECT_MIN.
     *
     * @throws std::invalid_argument if intensity is not in [0, 1].
     */
    void induce_affect(Affect a, double intensity);

    // -- neurochemical consequence table -------------------------------------

    /**
     * @brief Neurochemical profile that each Affect implies.
     *
     * Returns {dopamine, serotonin, norepinephrine} modulation deltas that
     * would reinforce / satisfy this affect state.
     * Keys: "dopamine", "serotonin", "norepinephrine".
     */
    [[nodiscard]] static std::map<std::string, double>
        affect_to_neurochemistry(Affect a);

    // -- attention modulation ------------------------------------------------

    /**
     * @brief Compute attention focus weight for a region entropy level.
     *
     * Current affect modulates whether high-entropy or low-entropy regions
     * get more attention.  Returns multiplier in [0.5, 2.0].
     */
    [[nodiscard]] double attention_weight(double entropy) const noexcept;

    // -- callback ------------------------------------------------------------

    /// Called when dominant affect changes after update().
    OnAffectChange on_affect_change;

    // -- description ---------------------------------------------------------

    /**
     * @brief Human-readable description of the current affective state.
     * Example: "curious and slightly anxious (valence=-0.12, arousal=0.74)"
     */
    [[nodiscard]] std::string describe_state() const;

    // -- pure static helpers (testable without instance) ---------------------

    /**
     * @brief Compute the raw IntensityMap directly from a NikolaState.
     * Pure function -- does NOT modify any member state.
     */
    [[nodiscard]] static IntensityMap
        compute_scores(const nikola::autonomy::NikolaState& s) noexcept;

    /** Compute valence from scalar parameters.  Pure function. */
    [[nodiscard]] static double
        compute_valence(double d, double td_error, double atp) noexcept;

    /** Compute arousal from scalar parameters.  Pure function. */
    [[nodiscard]] static double
        compute_arousal(double entropy, double boredom) noexcept;

private:
    static double clamp01(double v) noexcept {
        return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
    }
    static double clamp(double v, double lo, double hi) noexcept {
        return v < lo ? lo : (v > hi ? hi : v);
    }
    static double soft_step(double x, double centre, double k) noexcept {
        return 0.5 + 0.5 * std::tanh(k * (x - centre));
    }

    IntensityMap scores_{};
    IntensityMap induced_{};
    double       valence_  = 0.0;
    double       arousal_  = 0.0;
    Affect       dominant_ = Affect::NEUTRAL;
};

} // namespace nikola::interior
