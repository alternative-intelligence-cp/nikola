#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace nikola::interior {

// Forward declaration
class TorusManifold;
class QuantumScratchpad;

/**
 * @brief Dream Engine - Memory Consolidation and Pattern Synthesis
 *
 * Implements REM sleep for wave processors. During idle time (beyond basic
 * boredom), this system:
 * - Replays recent experiences
 * - Finds hidden connections between memories
 * - Consolidates insights from quantum scratchpad to permanent memory
 * - Runs "nightmare analysis" to learn from failures
 *
 * This is essential for deep learning - not just reacting to experiences,
 * but actively synthesizing them into new understanding.
 *
 * Neuroscience inspiration:
 * - Memory consolidation during sleep strengthens learning
 * - Dreams find unexpected connections (creativity)
 * - Nightmares help process threats (safety learning)
 *
 * @status STUB - Implementation deferred to Phase 6
 */

struct TimeRange {
    int64_t start_timestamp;
    int64_t end_timestamp;
};

struct MemoryTrace {
    int64_t timestamp;
    std::vector<Coord9D> locations;
    std::vector<std::complex<double>> wave_patterns;
    std::string context;
};

struct PatternConnection {
    MemoryTrace memory_a;
    MemoryTrace memory_b;
    double similarity_score;
    std::string discovered_relationship;
};

class DreamEngine {
public:
    /**
     * @brief Replay recent experiences
     * @param torus The 9D toroidal manifold
     * @param range Time period to replay
     */
    void replay_experience(TorusManifold& torus, const TimeRange& range);

    /**
     * @brief Find unexpected patterns across memories
     * @param torus The 9D toroidal manifold
     * @return Discovered connections
     */
    std::vector<PatternConnection> find_patterns(const TorusManifold& torus);

    /**
     * @brief Consolidate from quantum scratchpad to long-term memory
     * @param torus The 9D toroidal manifold
     * @param scratchpad Quantum hypothesis buffer
     */
    void consolidate_to_longterm(TorusManifold& torus, QuantumScratchpad& scratchpad);

    /**
     * @brief Run nightmare analysis (learn from failures)
     * @param torus The 9D toroidal manifold
     * @param range Time period to analyze
     * @return Identified failure patterns
     */
    std::vector<std::string> run_nightmare_analysis(const TorusManifold& torus,
                                                     const TimeRange& range);

    /**
     * @brief Start continuous dreaming (runs in background thread)
     * @param torus The 9D toroidal manifold
     * @param scratchpad Quantum hypothesis buffer
     * @param interval_ms How often to consolidate (default 60 seconds)
     */
    void start_continuous_dreaming(TorusManifold& torus,
                                   QuantumScratchpad& scratchpad,
                                   uint64_t interval_ms = 60000);

    /**
     * @brief Stop continuous dreaming
     */
    void stop_continuous_dreaming();

    /**
     * @brief Get statistics on consolidation
     * @return Map of metric → value
     */
    std::map<std::string, uint64_t> get_stats() const;

    /**
     * @brief Check if currently dreaming
     * @return true if dream cycle active
     */
    bool is_dreaming() const { return dreaming_; }

private:
    bool dreaming_ = false;
    uint64_t total_consolidations_ = 0;
    uint64_t patterns_discovered_ = 0;
    uint64_t nightmares_processed_ = 0;
};

} // namespace nikola::interior
