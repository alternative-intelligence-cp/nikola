#pragma once

/**
 * @file wave_mirror.hpp
 * @brief Phase 121 -- WaveMirror: introspective proprioception over NikolaState
 *
 * WaveMirror gives Nikola "proprioception" -- awareness of its own cognitive
 * state -- without depending on TorusManifold (also a stub).  Instead, it
 * derives six introspective metrics directly from the ENGS fields already
 * present in NikolaState:
 *
 *   dopamine  (D)   -- reward / prediction confidence
 *   td_error  (d)   -- prediction accuracy, sign matters
 *   atp       (A)   -- metabolic availability
 *   boredom   (B)   -- exploration pressure
 *   entropy   (H)   -- information complexity of current field
 *   torus_energy(E) -- total field activity level
 *
 * Derived metrics
 * ---------------
 *   confidence      in [0,1]   f(D, d, A)
 *   confusion       in [0,1]   f(H, d)
 *   cognitive_load  in [0,1]   f(H, E)
 *   coherence       in [0,1]   f(D, A, H)  -- smoothed over HISTORY_WINDOW ticks
 *   attention_focus            mode + salience derived from dominant signal
 *   spectral_signature[9]      9-band projection of ENGS fields
 *
 * A rolling history window of MIRROR_HISTORY_WINDOW ticks smooths the
 * coherence and confidence readouts, reducing tick-to-tick jitter.
 *
 * @status IMPLEMENTED -- Phase 121
 */

#include <nikola/autonomy/decision_loop.hpp>

#include <array>
#include <string>

namespace nikola::interior {

using nikola::autonomy::NikolaState;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Ticks retained in the rolling smoothing window
inline constexpr int    MIRROR_HISTORY_WINDOW   = 8;
/// Entropy ceiling to normalise H -> [0,1] for load and confusion
inline constexpr double MIRROR_ENTROPY_CEILING  = 3.0;
/// Coherence considered "high" above this threshold (used in describe())
inline constexpr double MIRROR_COHERENCE_HIGH   = 0.70;
/// Cognitive load considered "saturated" above this
inline constexpr double MIRROR_LOAD_SATURATED   = 0.80;
/// Minimum salience before reporting a non-IDLE focus
inline constexpr double MIRROR_MIN_SALIENCE     = 0.05;

// ---------------------------------------------------------------------------
// AttentionFocus
// ---------------------------------------------------------------------------

/**
 * @brief What the cognitive field is predominantly attending to
 *
 * Derived from the relative magnitudes of ENGS signals:
 *   REWARD   -- strong positive TD + high dopamine
 *   THREAT   -- strong negative TD with energy to process it
 *   FATIGUE  -- critically low ATP (system is exhausted)
 *   IDLE     -- high boredom + low entropy (underutilised)
 *   CURIOUS  -- moderate entropy with boredom drive; exploration mode
 */
struct AttentionFocus {
    enum class Mode {
        CURIOUS,   ///< Exploratory / seeking mode
        REWARD,    ///< Positive prediction event captured attention
        THREAT,    ///< Negative prediction error / aversive signal
        FATIGUE,   ///< Metabolic depletion dominates
        IDLE       ///< Bored, low activation
    };

    Mode   mode     = Mode::IDLE;
    double salience = 0.0;  ///< Strength of the dominant signal [0, 1]

    static const char* mode_name(Mode m) noexcept;
    const char* mode_name() const noexcept { return mode_name(mode); }
};

// ---------------------------------------------------------------------------
// MirrorSnapshot -- all introspective metrics at one point in time
// ---------------------------------------------------------------------------

struct MirrorSnapshot {
    double confidence     = 0.0;
    double confusion      = 0.0;
    double cognitive_load = 0.0;
    double coherence      = 0.0;
    double metacognitive  = 0.0;  ///< coherence * confidence * (1 - confusion)
    AttentionFocus focus;
    std::array<double, 9> spectral_signature{};
};

// ---------------------------------------------------------------------------
// WaveMirror
// ---------------------------------------------------------------------------

class WaveMirror {
public:
    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    WaveMirror() noexcept;

    /// Feed a new NikolaState tick -- updates all metrics and rolling window
    void update(const NikolaState& s) noexcept;

    // ------------------------------------------------------------------
    // Individual metric accessors
    // ------------------------------------------------------------------

    double confidence()     const noexcept;
    double confusion()      const noexcept;
    double cognitive_load() const noexcept;
    double coherence()      const noexcept;
    double metacognitive()  const noexcept;

    AttentionFocus        attention_focus()    const noexcept;
    std::array<double, 9> spectral_signature() const noexcept;
    MirrorSnapshot        snapshot()           const noexcept;

    // ------------------------------------------------------------------
    // Description
    // ------------------------------------------------------------------

    std::string describe() const;

    // ------------------------------------------------------------------
    // Pure static helpers -- no instance state, suitable for unit tests
    // ------------------------------------------------------------------

    /**
     * confidence = clamp(D*0.5 + max(d,0)*0.3 - max(-d,0)*0.2 + A*0.2, 0, 1)
     */
    static double compute_confidence(double dopamine,
                                     double td_error,
                                     double atp) noexcept;

    /**
     * h     = clamp(H / MIRROR_ENTROPY_CEILING, 0, 1)
     * neg_d = clamp(-td, 0, 1)
     * confusion = h*0.6 + neg_d*0.4
     */
    static double compute_confusion(double entropy,
                                    double td_error) noexcept;

    /**
     * h    = clamp(H / MIRROR_ENTROPY_CEILING, 0, 1)
     * e    = clamp(E, 0, 1)
     * load = h*0.7 + e*0.3
     */
    static double compute_cognitive_load(double entropy,
                                         double torus_energy) noexcept;

    /**
     * h_inv = 1 - clamp(H / MIRROR_ENTROPY_CEILING, 0, 1)
     * coh   = D*0.4 + A*0.4 + h_inv*0.2
     */
    static double compute_coherence(double dopamine,
                                    double atp,
                                    double entropy) noexcept;

    /**
     * 9-band ENGS spectral projection:
     *   [0] D                       dopamine
     *   [1] A*0.7 + D*0.3           serotonin proxy (stability)
     *   [2] H / ENTROPY_CEILING     norepinephrine proxy (arousal)
     *   [3] |td|                    prediction-error magnitude
     *   [4] B                       boredom
     *   [5] clamp(E, 0, 1)          torus activity
     *   [6] 1 - A                   depletion
     *   [7] max(-td, 0)             aversive signal (negative TD only)
     *   [8] conf*(1-conf)*(1-coh)   metacognitive uncertainty proxy
     */
    static std::array<double, 9> compute_spectral_signature(
        const NikolaState& s) noexcept;

    /**
     * Attention focus rule priority: FATIGUE > THREAT > REWARD > IDLE > CURIOUS
     *   FATIGUE  : atp < 0.15
     *   THREAT   : td < -0.10 AND atp >= 0.15
     *   REWARD   : td > +0.05 AND D > 0.55
     *   IDLE     : B > 0.70 AND H < 0.50
     *   CURIOUS  : default
     */
    static AttentionFocus compute_attention_focus(
        const NikolaState& s) noexcept;

private:
    MirrorSnapshot current_{};

    // Rolling history arrays for smoothed metrics
    std::array<double, MIRROR_HISTORY_WINDOW> conf_hist_{};
    std::array<double, MIRROR_HISTORY_WINDOW> coh_hist_{};
    int  hist_idx_   = 0;
    int  tick_count_ = 0;

    static double clamp01(double x) noexcept {
        return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
    }

    double smoothed_mean(
        const std::array<double, MIRROR_HISTORY_WINDOW>& h,
        int filled) const noexcept;
};

} // namespace nikola::interior
