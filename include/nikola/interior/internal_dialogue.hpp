#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace nikola::interior {

// Forward declarations
class TorusManifold;
struct Coord9D;

/**
 * @brief Internal Dialogue System - Thinking Out Loud to Myself
 *
 * Complex reasoning requires talking through problems. This system provides
 * persistent chain-of-thought reasoning that stores in the toroidal manifold.
 *
 * Key capabilities:
 * - Record thought chains as they develop
 * - Retrieve past reasoning for similar problems
 * - Question own assumptions (Socratic self-dialogue)
 * - Synthesize conclusions from multiple thoughts
 * - Detect circular reasoning or contradictions
 *
 * This is how the system works through hard problems - by maintaining an
 * internal monologue that can be queried and analyzed.
 *
 * Benefits:
 * - Complex problems solved step-by-step
 * - Reasoning becomes transparent and debuggable
 * - Past reasoning chains inform future ones
 * - Can catch own errors through self-questioning
 *
 * @status STUB - Implementation deferred to Phase 6
 */

struct ThoughtTrace {
    std::string thought;
    int64_t timestamp;
    Coord9D location;              // Where in torus thought was formed
    double confidence;             // How certain about this thought
    std::string reasoning_type;    // "deduction", "induction", "analogy", etc.
};

struct ReasoningChain {
    std::string problem;
    std::vector<ThoughtTrace> thoughts;
    std::string conclusion;
    double confidence;
    int64_t started;
    int64_t concluded;
};

class InternalDialogue {
public:
    /**
     * @brief Think a thought (add to current chain)
     * @param thought Text of thought
     * @param torus Location where thought formed
     * @param confidence How certain (0.0-1.0)
     */
    void think(const std::string& thought, TorusManifold& torus, double confidence = 0.5);

    /**
     * @brief Get current reasoning chain
     * @return All thoughts in current chain
     */
    std::vector<ThoughtTrace> get_reasoning_chain() const;

    /**
     * @brief Question an assumption (Socratic dialogue)
     * @param assumption Assumption to question
     * @param torus Knowledge state
     * @return Questions about the assumption
     */
    std::vector<std::string> question_assumption(const std::string& assumption,
                                                 const TorusManifold& torus);

    /**
     * @brief Synthesize conclusion from thoughts
     * @param torus Knowledge state
     * @return Conclusion statement
     */
    std::string synthesize_conclusion(const TorusManifold& torus);

    /**
     * @brief Start new reasoning chain (for new problem)
     * @param problem Problem description
     */
    void start_chain(const std::string& problem);

    /**
     * @brief End current reasoning chain
     * @param conclusion Final conclusion
     * @param torus Store in torus
     */
    void conclude_chain(const std::string& conclusion, TorusManifold& torus);

    /**
     * @brief Retrieve past reasoning on similar problem
     * @param problem Problem description
     * @param torus Knowledge state
     * @return Past reasoning chains
     */
    std::vector<ReasoningChain> recall_similar_reasoning(const std::string& problem,
                                                         const TorusManifold& torus);

    /**
     * @brief Detect circular reasoning in current chain
     * @return true if circular reasoning detected
     */
    bool detect_circular_reasoning() const;

    /**
     * @brief Detect contradictions in current chain
     * @return List of contradictory thought pairs
     */
    std::vector<std::pair<ThoughtTrace, ThoughtTrace>> detect_contradictions() const;

    /**
     * @brief Get average confidence of current reasoning
     * @return Average confidence (0.0-1.0)
     */
    double get_chain_confidence() const;

    /**
     * @brief Explain reasoning chain in plain language
     * @return Human-readable explanation
     */
    std::string explain_reasoning() const;

    /**
     * @brief Get all stored reasoning chains
     * @return All past chains
     */
    std::vector<ReasoningChain> get_all_chains() const;

    /**
     * @brief Get statistics on reasoning
     * @return Map of metric → value
     */
    std::map<std::string, uint64_t> get_stats() const;

private:
    ReasoningChain current_chain_;
    std::vector<ReasoningChain> past_chains_;
    uint64_t total_thoughts_ = 0;
    uint64_t chains_completed_ = 0;
};

} // namespace nikola::interior
