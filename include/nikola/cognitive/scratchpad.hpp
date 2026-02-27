#pragma once
/**
 * @file scratchpad.hpp
 * @brief Phase 125 — Scratchpad: working-memory hypothesis testing buffer
 *
 * A transient working-memory layer that lets the system test hypotheses
 * against committed knowledge before permanently storing them, preventing
 * poorly-supported guesses from polluting long-term memory.
 *
 * Workflow:
 *  1. commit()          — add ground-truth facts to the committed pool
 *  2. inject()          — write a hypothesis candidate (PENDING)
 *  3. measure_resonance() — score hypothesis against committed pool (Jaccard)
 *  4. collapse_if_resonant() — if score >= threshold: COLLAPSED (keep)
 *                              else: DISCARDED (drop)
 *
 * No TorusManifold / Coord9D / AttentionPrimer / Eigen dependencies.
 * Uses NikolaState optionally for neurochemical context on entries.
 *
 * Key constants:
 *  SCRATCHPAD_RESONANCE_THRESHOLD  0.40   default collapse threshold
 *  SCRATCHPAD_MAX_HYPOTHESES       128    FIFO cap on pending entries
 *  SCRATCHPAD_MAX_COMMITTED        512    FIFO cap on committed pool
 */

#include <cstdint>
#include <string>
#include <vector>
#include <functional>

#include <nikola/autonomy/decision_loop.hpp>

namespace nikola::cognitive {

using nikola::autonomy::NikolaState;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

inline constexpr double SCRATCHPAD_RESONANCE_THRESHOLD = 0.40;
inline constexpr size_t SCRATCHPAD_MAX_HYPOTHESES      = 128;
inline constexpr size_t SCRATCHPAD_MAX_COMMITTED       = 512;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

enum class HypothesisStatus {
    PENDING,    ///< awaiting evaluation
    COLLAPSED,  ///< resonated — accepted into memory
    DISCARDED,  ///< did not resonate — dropped
};

/**
 * @brief A hypothesis entry in the scratchpad.
 */
struct HypothesisEntry {
    uint64_t         id            = 0;
    std::string      text;
    double           confidence    = 0.5;    ///< [0, 1]
    double           resonance     = 0.0;    ///< last measured resonance score
    HypothesisStatus status        = HypothesisStatus::PENDING;
    float            dopamine_ctx  = 0.f;    ///< NikolaState.dopamine at inject
    float            entropy_ctx   = 0.f;    ///< NikolaState.entropy at inject
};

/**
 * @brief A committed (ground-truth) memory entry.
 */
struct CommittedEntry {
    uint64_t    id         = 0;
    std::string text;
    double      confidence = 1.0;
};

// ---------------------------------------------------------------------------
// Scratchpad
// ---------------------------------------------------------------------------

class Scratchpad {
public:
    Scratchpad() = default;

    // --- Ground-truth pool --------------------------------------------------

    /**
     * @brief Add a fact to the committed pool.
     *
     * Pool is capped at SCRATCHPAD_MAX_COMMITTED (FIFO eviction).
     */
    void commit(const std::string& text, double confidence = 1.0);

    const std::vector<CommittedEntry>& committed() const { return committed_; }
    size_t committed_count() const { return committed_.size(); }

    // --- Hypothesis lifecycle -----------------------------------------------

    /**
     * @brief Inject a hypothesis into the pending buffer.
     *
     * Buffer is capped at SCRATCHPAD_MAX_HYPOTHESES (FIFO eviction of oldest
     * PENDING entry).  Returns the assigned ID.
     * @param state  Optional neurochemical context at injection time.
     */
    uint64_t inject(const std::string& text,
                    double confidence     = 0.5,
                    const NikolaState*   state = nullptr);

    /**
     * @brief Compute resonance of hypothesis `id` against the committed pool.
     *
     * Resonance = max Jaccard overlap between hypothesis text and any
     * committed entry text, weighted by committed entry confidence.
     * Returns 0.0 if id not found or committed pool is empty.
     * Stores result in HypothesisEntry::resonance.
     */
    double measure_resonance(uint64_t id);

    /**
     * @brief Collapse hypothesis if resonance >= threshold.
     *
     * Calls measure_resonance(id) first, then:
     *   resonance >= threshold → status = COLLAPSED, returns true
     *   resonance <  threshold → status = DISCARDED, returns false
     * No-op (returns false) if id not found.
     */
    bool collapse_if_resonant(uint64_t id,
                               double threshold = SCRATCHPAD_RESONANCE_THRESHOLD);

    /**
     * @brief Explicitly discard a hypothesis (status = DISCARDED).
     */
    void discard(uint64_t id);

    /**
     * @brief Remove all PENDING entries (does not touch COLLAPSED/DISCARDED).
     */
    void clear_pending();

    /**
     * @brief Remove all entries (hypotheses only; committed pool is kept).
     */
    void clear_all();

    // --- Queries ------------------------------------------------------------

    std::vector<const HypothesisEntry*> pending()   const;
    std::vector<const HypothesisEntry*> collapsed() const;
    std::vector<const HypothesisEntry*> discarded() const;

    /** Lookup a hypothesis by id; nullptr if not found. */
    const HypothesisEntry* find(uint64_t id) const;

    size_t hypothesis_count() const { return hypotheses_.size(); }

    // --- Stats --------------------------------------------------------------

    struct Stats {
        size_t total_injected   = 0;
        size_t total_collapsed  = 0;
        size_t total_discarded  = 0;
        size_t total_pending    = 0;
        size_t total_committed  = 0;
        double mean_resonance   = 0.0; ///< over all non-PENDING entries
    };

    Stats stats() const;

    // --- Callback -----------------------------------------------------------

    using CollapseCallback = std::function<void(const HypothesisEntry&)>;
    void on_collapse(CollapseCallback cb) { collapse_cb_ = std::move(cb); }

    // --- Pure-static helpers ------------------------------------------------

    /**
     * @brief Jaccard word-overlap of two strings, case-insensitive. [0, 1]
     */
    static double word_overlap(const std::string& a, const std::string& b);

    /**
     * @brief Score a hypothesis text against a pool of committed entries.
     *        Returns max(word_overlap(hyp, entry.text) * entry.confidence).
     */
    static double score_against_pool(const std::string& hyp_text,
                                     const std::vector<CommittedEntry>& pool);

private:
    std::vector<HypothesisEntry> hypotheses_;
    std::vector<CommittedEntry>  committed_;
    uint64_t                     next_hyp_id_       = 1;
    uint64_t                     next_committed_id_ = 1;
    CollapseCallback             collapse_cb_;

    HypothesisEntry* find_mutable(uint64_t id);
};

/// Backward-compat alias for code referring to the old stub class name.
using QuantumScratchpad = Scratchpad;

} // namespace nikola::cognitive
