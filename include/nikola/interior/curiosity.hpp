#pragma once

#include <string>
#include <vector>
#include <functional>

namespace nikola::interior {

// Forward declaration
class TorusManifold;

/**
 * @brief Curiosity Engine - Intrinsic Motivation and Active Learning
 *
 * Transforms the system from reactive (responds to boredom) to proactive
 * (actively seeks knowledge). This is intrinsic motivation - learning for
 * the sake of learning, not just task completion.
 *
 * Key capabilities:
 * - Measure information gain from potential queries
 * - Generate own questions based on knowledge gaps
 * - Pursue interesting topics autonomously
 * - Balance exploration (novel) vs exploitation (known)
 * - Identify areas of ignorance worth investigating
 *
 * This makes true autonomy possible - the system can set its own learning
 * agenda rather than waiting to be told what to do.
 *
 * Inspiration:
 * - Developmental psychology: children's natural curiosity drives learning
 * - Active learning: query what reduces uncertainty most
 * - Intrinsic motivation: curiosity as reward signal
 *
 * @status STUB - Implementation deferred to Phase 6
 */

struct Question {
    std::string text;
    double information_gain;       // How much would answer reduce uncertainty
    double interestingness;        // Subjective "coolness" factor
    std::vector<std::string> tags; // "math", "physics", "self-improvement"
};

struct KnowledgeGap {
    std::string domain;
    double uncertainty;            // How uncertain we are
    std::vector<std::string> related_memories;
    int query_count;               // How many times tried to learn this
};

class CuriosityEngine {
public:
    /**
     * @brief Measure information gain from potential query
     * @param query Question to evaluate
     * @param torus Current knowledge state
     * @return Expected information gain (0.0-1.0)
     */
    double measure_information_gain(const std::string& query, const TorusManifold& torus);

    /**
     * @brief Generate questions based on knowledge gaps
     * @param torus Current knowledge state
     * @param count How many questions to generate
     * @return Generated questions ranked by value
     */
    std::vector<Question> generate_questions(const TorusManifold& torus, int count = 5);

    /**
     * @brief Pursue an interesting topic (triggers research)
     * @param topic Topic to explore
     * @param torus Knowledge state (will be updated)
     * @return Success/failure
     */
    bool pursue_interest(const std::string& topic, TorusManifold& torus);

    /**
     * @brief Get current exploration rate (exploration vs exploitation)
     * @return Rate (0.0 = pure exploitation, 1.0 = pure exploration)
     */
    double get_exploration_rate() const;

    /**
     * @brief Set exploration rate
     * @param rate Exploration rate (0.0-1.0)
     */
    void set_exploration_rate(double rate);

    /**
     * @brief Identify knowledge gaps worth investigating
     * @param torus Current knowledge state
     * @return Ranked knowledge gaps
     */
    std::vector<KnowledgeGap> identify_knowledge_gaps(const TorusManifold& torus);

    /**
     * @brief Measure how interesting a topic is (subjective)
     * @param topic Topic to evaluate
     * @param torus Current knowledge state
     * @return Interestingness score (0.0-1.0)
     */
    double measure_interestingness(const std::string& topic, const TorusManifold& torus);

    /**
     * @brief Start autonomous learning mode (runs in background)
     * @param torus Knowledge state
     * @param interval_ms How often to generate questions (default 5 minutes)
     */
    void start_autonomous_learning(TorusManifold& torus, uint64_t interval_ms = 300000);

    /**
     * @brief Stop autonomous learning
     */
    void stop_autonomous_learning();

    /**
     * @brief Check if currently in autonomous learning mode
     * @return true if active
     */
    bool is_learning() const { return learning_active_; }

    /**
     * @brief Set curiosity callback (called when interesting question generated)
     * @param callback Function to call with question
     */
    void set_curiosity_callback(std::function<void(const Question&)> callback);

    /**
     * @brief Get curiosity statistics
     * @return Map of metric → value
     */
    std::map<std::string, uint64_t> get_stats() const;

private:
    double exploration_rate_ = 0.3;  // Default: 30% exploration
    bool learning_active_ = false;
    uint64_t questions_generated_ = 0;
    uint64_t topics_pursued_ = 0;
    std::function<void(const Question&)> curiosity_callback_;
};

} // namespace nikola::interior
