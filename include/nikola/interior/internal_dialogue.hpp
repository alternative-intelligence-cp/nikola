#pragma once

/**
 * @file internal_dialogue.hpp
 * @brief Phase 122 -- InternalDialogue: persistent chain-of-thought reasoning
 *
 * InternalDialogue gives Nikola a self-directed reasoning loop -- the ability
 * to think through a problem in explicit, inspectable steps, question its own
 * assumptions, detect errors in its own reasoning, and recall how it solved
 * similar problems before.
 *
 * Design notes
 * ------------
 * Original stub coupled to TorusManifold/Coord9D/AttentionPrimer (all stubs).
 * Redesigned to be completely self-contained:
 *
 *   - NikolaState is optional context attached to each thought; it records the
 *     neurochemical state at the moment of formation for later analysis.
 *   - ThoughtEntry replaces ThoughtTrace: drops torus location, adds tick
 *     counter + neurochemical snapshot.
 *   - ReasoningChain is otherwise unchanged in structure.
 *   - Circular-reasoning detection uses word-overlap similarity.
 *   - Contradiction detection uses overlap + negation pattern matching.
 *   - Socratic questioning generates Who/What/Why/How/When variants.
 *   - Recall uses Jaccard word-overlap against problem + thought texts.
 *
 * @status IMPLEMENTED -- Phase 122
 */

#include <nikola/autonomy/decision_loop.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nikola::interior {

using nikola::autonomy::NikolaState;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Word-overlap threshold above which two thoughts are considered "circular"
inline constexpr double DIALOGUE_CIRCULAR_THRESHOLD    = 0.75;
/// Word-overlap threshold for contradiction candidate before negation check
inline constexpr double DIALOGUE_CONTRADICTION_OVERLAP = 0.35;
/// Maximum number of past chains returned by recall_similar()
inline constexpr size_t DIALOGUE_MAX_RECALL            = 5;
/// Maximum thoughts per chain before a forced synthesis is recommended
inline constexpr size_t DIALOGUE_CHAIN_LENGTH_WARN     = 32;

// ---------------------------------------------------------------------------
// ThoughtEntry
// ---------------------------------------------------------------------------

/**
 * @brief A single thought within a reasoning chain
 *
 * reasoning_type values:
 *   "observation"  -- direct reading of state/input
 *   "deduction"    -- logically derived from prior thoughts
 *   "induction"    -- generalised from examples
 *   "analogy"      -- by structural similarity to another domain
 *   "hypothesis"   -- tentative proposal to be tested
 *   "question"     -- self-generated query (Socratic step)
 */
struct ThoughtEntry {
    std::string text;
    uint64_t    tick          = 0;
    double      confidence    = 0.5;   ///< [0, 1]
    std::string reasoning_type = "observation";

    // Neurochemical context at formation (optional -- zeroed if no state given)
    double dopamine_context = 0.0;
    double entropy_context  = 0.0;
    double atp_context      = 0.0;
};

// ---------------------------------------------------------------------------
// ReasoningChain
// ---------------------------------------------------------------------------

struct ReasoningChain {
    uint64_t                chain_id       = 0;
    std::string             problem;
    std::vector<ThoughtEntry> thoughts;
    std::string             conclusion;
    double                  conclusion_confidence = 0.0;
    uint64_t                started_tick   = 0;
    uint64_t                concluded_tick = 0;  ///< 0 = not yet concluded

    bool     is_concluded()      const noexcept { return concluded_tick > 0; }
    bool     is_empty()          const noexcept { return thoughts.empty(); }
    size_t   length()            const noexcept { return thoughts.size(); }
    double   mean_confidence()   const noexcept;
    double   peak_confidence()   const noexcept;
};

// ---------------------------------------------------------------------------
// InternalDialogue
// ---------------------------------------------------------------------------

class InternalDialogue {
public:
    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    InternalDialogue() noexcept;

    // ------------------------------------------------------------------
    // Chain lifecycle
    // ------------------------------------------------------------------

    /// Begin a new reasoning chain.  Returns the chain_id.
    /// Concludes any currently active chain with an empty conclusion first.
    uint64_t start_chain(const std::string& problem);

    /// Add a thought to the active chain.
    /// If no chain is active, auto-starts one with problem="<unnamed>".
    void think(const std::string& text,
               double             confidence    = 0.5,
               const std::string& reasoning_type = "observation",
               const NikolaState* state          = nullptr);

    /// Conclude the active chain and store it in past_chains_.
    /// No-op if no chain is active.
    void conclude_chain(const std::string& conclusion,
                        double             confidence = 0.0);

    // ------------------------------------------------------------------
    // Current chain accessors
    // ------------------------------------------------------------------

    bool                    has_active_chain()  const noexcept;
    const ReasoningChain&   current_chain()     const noexcept;
    double                  chain_confidence()  const noexcept;
    size_t                  current_length()    const noexcept;

    // ------------------------------------------------------------------
    // Introspective analysis (operate on active chain)
    // ------------------------------------------------------------------

    /// Returns true if any two thoughts have word_overlap > CIRCULAR_THRESHOLD.
    bool detect_circular_reasoning() const;

    /**
     * Returns index pairs (i, j) where i < j and the thoughts are potential
     * contradictions (overlap > CONTRADICTION_OVERLAP AND one negates the other).
     */
    std::vector<std::pair<size_t, size_t>> detect_contradictions() const;

    /**
     * Synthesize a conclusion from the active chain.
     * Strategy: return the highest-confidence thought's text, prefixed with
     * "Synthesis: ".  Falls back to the most recent thought if all equal.
     * Returns empty string if the chain has no thoughts.
     */
    std::string synthesize_conclusion() const;

    /**
     * Generate Socratic questions about an assumption.
     * Produces up to 5 who/what/why/how/when variants.
     */
    std::vector<std::string> question_assumption(
        const std::string& assumption) const;

    /// Human-readable explanation of the reasoning chain (numbered steps).
    std::string explain_reasoning() const;

    // ------------------------------------------------------------------
    // Recall / past chains
    // ------------------------------------------------------------------

    /**
     * Return up to DIALOGUE_MAX_RECALL past chains whose problem text has
     * the highest word-overlap with @p query, sorted descending by overlap.
     * Includes at most @p max_results results.
     */
    std::vector<const ReasoningChain*> recall_similar(
        const std::string& query,
        size_t             max_results = DIALOGUE_MAX_RECALL) const;

    const std::vector<ReasoningChain>& all_chains() const noexcept;

    // ------------------------------------------------------------------
    // Stats
    // ------------------------------------------------------------------

    struct Stats {
        uint64_t total_thoughts      = 0;
        uint64_t total_chains        = 0;
        uint64_t completed_chains    = 0;
        double   mean_chain_confidence = 0.0;
        uint64_t circular_detections = 0;
    };

    Stats stats() const noexcept;

    // ------------------------------------------------------------------
    // Pure static helpers -- no instance state, suitable for unit tests
    // ------------------------------------------------------------------

    /**
     * Jaccard word-overlap between two strings.
     * Tokenises on whitespace + basic punctuation, lower-cases, returns
     *   |A intersect B| / |A union B|, or 0 for empty inputs.
     */
    static double word_overlap(const std::string& a,
                               const std::string& b) noexcept;

    /**
     * Returns true if one of the strings explicitly negates the other.
     * "Negation" = string b contains "not ", "no ", "never ", "cannot ",
     * "can't ", "isn't ", "don't ", "doesn't ", "won't " before a word
     * that appears in string a (case-insensitive).
     * Symmetric: also tests a negates b.
     */
    static bool contains_negation(const std::string& a,
                                  const std::string& b) noexcept;

    /**
     * Generate Socratic questions for an assumption string.
     * Returns a fixed set of 5 question variants.
     */
    static std::vector<std::string> generate_socratic_questions(
        const std::string& assumption);

private:
    ReasoningChain           current_;
    bool                     has_active_ = false;
    std::vector<ReasoningChain> past_;
    uint64_t                 chain_id_counter_ = 0;
    uint64_t                 tick_counter_     = 0;
    uint64_t                 circular_detections_ = 0;

    uint64_t next_tick()  noexcept { return ++tick_counter_;     }
    uint64_t next_chain() noexcept { return ++chain_id_counter_; }
};

} // namespace nikola::interior
