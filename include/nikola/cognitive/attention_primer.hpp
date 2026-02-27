#pragma once
/**
 * @file attention_primer.hpp
 * @brief Phase 126 — AttentionPrimer: topic-priming attention-bias tracker
 *
 * The AttentionPrimer tracks which topics / feature-tags should receive
 * preferential cognitive processing on the next tick.  The original stub
 * was conceived as pre-warping toroidal geometry to shorten attention paths;
 * this implementation achieves the same effect through explicit activation
 * weights that decay exponentially — topics that were recently primed remain
 * biased until the weight falls below the culling threshold.
 *
 * Workflow:
 *  1. prime(tag, activation)  — boost a topic's weight (additive up to 1.0)
 *  2. decay_all()             — called once per tick; multiplies each weight
 *                               by its decay_rate; prunes if < MIN_WEIGHT
 *  3. weight_of(tag)          — query current activation for a tag
 *  4. most_primed()           — returns highest-activation active focus
 *  5. predict_focus(state)    — state-aware pick: dopamine boosts reward tags,
 *                               boredom boosts novel/exploration tags
 *
 * No Eigen / TorusManifold / Coord9D dependencies.
 * NikolaState used optionally for neurochemical context on entries.
 *
 * Key constants:
 *  ATTENTION_DECAY_RATE    0.85   default per-tick multiplicative decay
 *  ATTENTION_MIN_WEIGHT    0.05   culling threshold
 *  ATTENTION_MAX_TOPICS    64     FIFO cap on primed topics
 */

#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include <functional>

#include <nikola/autonomy/decision_loop.hpp>

namespace nikola::cognitive {

using nikola::autonomy::NikolaState;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

inline constexpr double   ATTENTION_DECAY_RATE  = 0.85;
inline constexpr double   ATTENTION_MIN_WEIGHT  = 0.05;
inline constexpr size_t   ATTENTION_MAX_TOPICS  = 64;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/**
 * @brief A single topic / feature-tag with its current activation weight.
 */
struct PrimedFocus {
    std::string  tag;
    double       activation    = 0.0;    ///< [0, 1] — current weight
    double       decay_rate    = ATTENTION_DECAY_RATE;
    uint64_t     prime_tick    = 0;      ///< tick at which it was last primed
    float        dopamine_ctx  = 0.f;
    float        entropy_ctx   = 0.f;
};

// ---------------------------------------------------------------------------
// AttentionPrimer
// ---------------------------------------------------------------------------

class AttentionPrimer {
public:
    AttentionPrimer() = default;

    // --- Priming ------------------------------------------------------------

    /**
     * @brief Prime a topic tag.
     *
     * If the tag already exists, its weight is boosted by `activation`
     * (clamped to 1.0).  New entries are created if the tag is absent.
     * If the pool exceeds ATTENTION_MAX_TOPICS the lowest-weight entry is
     * evicted to make room.
     *
     * @param tag        Arbitrary topic label (case-insensitive compare).
     * @param activation Boost to add, [0, 1].
     * @param decay_rate Per-tick decay multiplier for this entry (default:
     *                   ATTENTION_DECAY_RATE).
     * @param tick       Current tick (stored as prime_tick).
     * @param state      Optional NikolaState for neurochemical context.
     */
    void prime(const std::string& tag,
               double activation   = 0.5,
               double decay_rate   = ATTENTION_DECAY_RATE,
               uint64_t tick       = 0,
               const NikolaState*  state = nullptr);

    /**
     * @brief Apply one decay step to all primed topics.
     *
     * Each entry's activation is multiplied by its individual decay_rate.
     * Entries whose activation falls below ATTENTION_MIN_WEIGHT are removed.
     */
    void decay_all();

    /**
     * @brief Remove a specific tag from the primed pool.
     */
    void remove(const std::string& tag);

    /**
     * @brief Remove all primed topics.
     */
    void clear();

    // --- Queries ------------------------------------------------------------

    /**
     * @brief Get current activation weight for a tag.
     * Returns 0.0 if the tag is not in the pool.
     */
    double weight_of(const std::string& tag) const;

    /**
     * @brief Returns true if tag is active and weight >= threshold.
     */
    bool is_primed(const std::string& tag,
                   double threshold = ATTENTION_MIN_WEIGHT) const;

    /**
     * @brief Returns the PrimedFocus with the highest activation,
     *        or nullopt if the pool is empty.
     */
    std::optional<PrimedFocus> most_primed() const;

    /**
     * @brief Returns all active topics, sorted by activation descending.
     */
    std::vector<PrimedFocus> all_primed() const;

    size_t topic_count() const { return topics_.size(); }

    // --- State-aware prediction ---------------------------------------------

    /**
     * @brief Predict the most relevant focus given the current NikolaState.
     *
     * Scoring = activation * state_multiplier, where:
     *   dopamine > 0.6  → reward/goal/success tags get +0.15 bonus
     *   boredom  > 0.5  → novel/explore/question tags get +0.15 bonus
     *   entropy  > 0.6  → uncertainty/conflict/resolve tags get +0.10 bonus
     *
     * Returns nullopt if pool is empty.
     */
    std::optional<PrimedFocus> predict_focus(const NikolaState& state) const;

    // --- Stats --------------------------------------------------------------

    struct Stats {
        size_t  topic_count     = 0;
        double  mean_activation = 0.0;
        double  max_activation  = 0.0;
        double  min_activation  = 0.0;
    };

    Stats stats() const;

    // --- Callback -----------------------------------------------------------

    using PrimeCallback = std::function<void(const PrimedFocus&)>;
    void on_prime(PrimeCallback cb) { prime_cb_ = std::move(cb); }

    // --- Pure-static helpers ------------------------------------------------

    /**
     * @brief Case-insensitive Jaccard word-overlap, [0, 1].
     */
    static double topic_overlap(const std::string& a, const std::string& b);

    /**
     * @brief Merged activation: additive blend, clamped to [0, 1].
     */
    static double merged_activation(double existing, double incoming) {
        return std::min(1.0, existing + incoming);
    }

    /**
     * @brief State-based bonus for a tag given current NikolaState.
     * Returns additional activation score in [0, 0.25].
     */
    static double state_bonus(const std::string& tag, const NikolaState& state);

private:
    std::vector<PrimedFocus> topics_;
    PrimeCallback            prime_cb_;

    /// Case-insensitive tag compare.
    static std::string normalise_tag(const std::string& tag);

    /// Find index of tag in topics_; returns npos if absent.
    size_t find_index(const std::string& normalised) const;
};

} // namespace nikola::cognitive
