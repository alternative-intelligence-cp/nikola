#pragma once
/**
 * @file dream_engine.hpp
 * @brief Phase 123 — DreamEngine: memory consolidation and pattern synthesis
 *
 * Implements REM-sleep analogue for Nikola. During idle periods (boredom >=
 * DREAM_IDLE_THRESHOLD) the engine replays the experience buffer, finds
 * similarity-based connections between neurochemical snapshots, distils them
 * into ConsolidatedMemory entries, and processes nightmare experiences
 * (high-entropy / low-dopamine events) for failure-pattern extraction.
 *
 * No TorusManifold / Coord9D / QuantumScratchpad / AttentionPrimer
 * dependencies — operates purely on NikolaState snapshots and string tags.
 *
 * Key constants (tuneable):
 *  DREAM_IDLE_THRESHOLD        0.60  boredom needed to start a dream cycle
 *  DREAM_BUFFER_SIZE           256   max recorded experiences (FIFO)
 *  DREAM_SIMILARITY_THRESHOLD  0.55  min state_similarity() to form fragment
 *  DREAM_NIGHTMARE_ENTROPY     1.40  entropy above which marks a nightmare
 *  DREAM_NIGHTMARE_DOPAMINE    0.25  dopamine below which marks a nightmare
 *  DREAM_CONSOLIDATION_MIN     0.50  min fragment novelty_score to consolidate
 *  DREAM_MAX_RECALL            8     default max recall results
 */

#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <functional>

#include <nikola/autonomy/decision_loop.hpp>

namespace nikola::interior {

using nikola::autonomy::NikolaState;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

inline constexpr double DREAM_IDLE_THRESHOLD       = 0.60;
inline constexpr size_t DREAM_BUFFER_SIZE          = 256;
inline constexpr double DREAM_SIMILARITY_THRESHOLD = 0.55;
inline constexpr float  DREAM_NIGHTMARE_ENTROPY    = 1.40f;
inline constexpr float  DREAM_NIGHTMARE_DOPAMINE   = 0.25f;
inline constexpr double DREAM_CONSOLIDATION_MIN    = 0.50;
inline constexpr size_t DREAM_MAX_RECALL           = 8;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/**
 * @brief A single recorded experience — neurochemical snapshot + metadata.
 */
struct Experience {
    uint64_t    tick          = 0;
    NikolaState state         = {};
    std::string tag;              ///< brief label e.g. "reward_spike", "error"
    float       reward_signal = 0.f; ///< +/- reward magnitude at recording time
    bool        is_nightmare  = false;
};

/**
 * @brief A connection discovered between two experiences during dreaming.
 */
struct DreamFragment {
    size_t      exp_index_a   = 0;
    size_t      exp_index_b   = 0;
    double      similarity    = 0.0;  ///< state_similarity(a, b) in [0, 1]
    double      novelty_score = 0.0;  ///< how surprising this pairing is
    std::string insight;              ///< auto-generated human-readable note
};

/**
 * @brief A fully consolidated memory distilled from one or more fragments.
 */
struct ConsolidatedMemory {
    uint64_t    formation_tick = 0;
    std::string key_insight;
    double      confidence     = 0.0;  ///< mean novelty of contributing fragments
    std::vector<size_t> source_exp_indices;
    bool        from_nightmare = false;
};

/**
 * @brief Summary of one complete dream cycle.
 */
struct DreamCycle {
    uint64_t start_tick           = 0;
    uint64_t end_tick             = 0;
    size_t   fragments_found      = 0;
    size_t   memories_formed      = 0;
    size_t   nightmares_processed = 0;
    double   mean_novelty         = 0.0; ///< avg novelty across all fragments
};

// ---------------------------------------------------------------------------
// DreamEngine
// ---------------------------------------------------------------------------

class DreamEngine {
public:
    DreamEngine() = default;

    // --- Recording ----------------------------------------------------------

    /**
     * @brief Record an experience for future dreaming.
     *
     * Buffer is capped at DREAM_BUFFER_SIZE; oldest entries are evicted (FIFO).
     * is_nightmare is auto-set via is_nightmare_state(state).
     */
    void record_experience(const std::string& tag,
                           const NikolaState& state,
                           float reward = 0.f);

    // --- Dreaming -----------------------------------------------------------

    /**
     * @brief Run one full dream cycle across the experience buffer.
     *
     * Scans all unique pairs, builds DreamFragments for pairs with
     * similarity >= DREAM_SIMILARITY_THRESHOLD, then consolidates fragments
     * with novelty_score >= DREAM_CONSOLIDATION_MIN into ConsolidatedMemory
     * entries.  Nightmare experiences are also counted and processed.
     */
    DreamCycle dream(uint64_t tick);

    // --- Query --------------------------------------------------------------

    /**
     * @brief Recall consolidated memories whose insight overlaps query words.
     * @return Pointers into memories_, sorted descending by confidence.
     */
    std::vector<const ConsolidatedMemory*>
    recall(const std::string& query, size_t max = DREAM_MAX_RECALL) const;

    /**
     * @brief Extract failure patterns from nightmare experiences.
     * @return Human-readable descriptions of identified failure patterns.
     */
    std::vector<std::string> process_nightmares() const;

    // --- Accessors ----------------------------------------------------------

    const std::vector<Experience>&         experiences() const { return experiences_; }
    const std::vector<ConsolidatedMemory>& memories()    const { return memories_; }
    const std::vector<DreamCycle>&         dream_log()   const { return dream_log_; }

    size_t experience_count() const { return experiences_.size(); }
    size_t memory_count()     const { return memories_.size(); }
    size_t nightmare_count()  const;

    // --- Stats --------------------------------------------------------------

    struct Stats {
        size_t total_experiences      = 0;
        size_t total_nightmares       = 0;
        size_t total_fragments        = 0;   ///< cumulative across all cycles
        size_t total_memories         = 0;
        size_t total_dream_cycles     = 0;
        double mean_memory_confidence = 0.0;
    };

    Stats stats() const;

    // --- Callback -----------------------------------------------------------

    using DreamCallback = std::function<void(const DreamCycle&)>;
    void on_dream_complete(DreamCallback cb) { dream_cb_ = std::move(cb); }

    // --- Pure-static helpers ------------------------------------------------

    /**
     * @brief Similarity of two NikolaState snapshots in [0, 1].
     *
     * 1 - L2(dopamine, atp, entropy, torus_energy) / sqrt(4) clamped to [0,1].
     */
    static double state_similarity(const NikolaState& a, const NikolaState& b);

    /**
     * @brief True when entropy > DREAM_NIGHTMARE_ENTROPY
     *                AND dopamine < DREAM_NIGHTMARE_DOPAMINE.
     */
    static bool is_nightmare_state(const NikolaState& s);

    /**
     * @brief True when boredom >= DREAM_IDLE_THRESHOLD.
     */
    static bool is_idle_enough(const NikolaState& s);

    /**
     * @brief Auto-generate a human-readable insight string for a fragment.
     */
    static std::string generate_insight(const Experience& a,
                                        const Experience& b,
                                        double similarity);

    /**
     * @brief Score novelty of a fragment given the mean buffer similarity.
     *
     * novelty = (1 - similarity) * (1 - mean_similarity) clamped to [0, 1].
     * High when the pair is both dissimilar to each other AND the buffer
     * mean similarity is low (many diverse states -> surprising connection).
     */
    static double compute_novelty(double similarity, double mean_similarity);

private:
    std::vector<Experience>         experiences_;
    std::vector<ConsolidatedMemory> memories_;
    std::vector<DreamCycle>         dream_log_;
    size_t                          total_fragments_ = 0;
    DreamCallback                   dream_cb_;

    // word-overlap helper for recall scoring
    static double tag_overlap(const std::string& a, const std::string& b);
};

} // namespace nikola::interior
