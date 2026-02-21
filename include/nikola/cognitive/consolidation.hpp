/**
 * @file cognitive/consolidation.hpp
 * @brief Memory consolidation: homeostatic NAP cycles and replay-based strengthening.
 *
 * Implements Phase 3 cognitive-architecture — memory consolidation layer:
 *
 *   - NAP cycle:  bulk decay → LTP/prune consolidate pass → replay top-K memories
 *   - Replay:     load stored ψ back into WaveFunction, re-propagate N steps,
 *                 then re-store so the record reflects updated physics dynamics.
 *   - Homeostasis: target memory count range — prune below min strength
 *                  OR boost access threshold if count exceeds MAX_RECORDS.
 *
 * Architecture
 * ------------
 *
 *   ReplayCandidate
 *     Lightweight struct: {key, score=strength × log(1+access_count)}.
 *     The logarithmic access term prevents runaway dominance of single memories.
 *
 *   MemoryReplay
 *     Sorts SemanticMemory records by score, replays each into a WaveFunction
 *     via Propagator, then re-stores the result.  Updated dynamics are captured
 *     in persistent ψ snapshots (refreshed spectral content).
 *
 *   ConsolidationEngine
 *     Orchestrator.  Call nap_cycle() at the end of each "cognitive epoch"
 *     (typical: every 10–100 processing steps, or when the MetabolicLock
 *     signals low reward / downtime).
 *
 * Biological Analogy
 * ------------------
 *   ConsolidationEngine  ↔  Sleep / slow-wave consolidation phase.
 *   SemanticMemory.decay ↔  Synaptic down-scaling during NREM.
 *   SemanticMemory.LTP   ↔  Spike-timing-dependent potentiation.
 *   MemoryReplay         ↔  Hippocampal → cortical transfer / replay.
 *
 * Validation Criteria (Phase 3 gate)
 * -----------------------------------
 *   - Strong memories (strength ≥ 0.5 at cycle start) survive NAP.
 *   - Weak memories (strength < MIN_STRENGTH after decay) are pruned.
 *   - Replayed memories have strength = SemanticMemory::MAX_STRENGTH after replay.
 *
 * Reference:
 *   PROJECT_STATUS.md Phase 3 — "memory consolidation (neurogenesis/homeostasis)"
 *   engineering report §§ 8.9.3 (The Brain), Gap 3.3 (sequence / geometry memory)
 */
#pragma once

#include <nikola/cognitive/semantic_memory.hpp>
#include <nikola/cognitive/query_engine.hpp>
#include <nikola/physics/wave_function.hpp>
#include <nikola/physics/propagator.hpp>
#include <nikola/physics/hamiltonian.hpp>

#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdint>

namespace nikola::cognitive {

using physics::WaveFunction;
using physics::Propagator;

// ============================================================================
// Constants
// ============================================================================

/// Default number of physics steps per memory replay.
inline constexpr int   CONSOLIDATION_REPLAY_STEPS = 20;

/// Default number of top-K memories to replay per NAP cycle.
inline constexpr int   CONSOLIDATION_REPLAY_K     = 5;

/// Default time step (seconds) applied during NAP cycle decay.
inline constexpr float CONSOLIDATION_NAP_DT       = 60.f;   // one simulated minute

/// Maximum records before homeostatic pruning tightens threshold.
inline constexpr size_t CONSOLIDATION_MAX_RECORDS  = 1000;

// ============================================================================
// ReplayCandidate
// ============================================================================

/**
 * @brief Score-ranked memory candidate for the replay pass.
 *
 * score = strength × log₁₀(1 + access_count)
 *
 * The logarithm dampens runaway dominance from a single frequently accessed
 * memory, ensuring diverse replay across the top-K set.
 */
struct ReplayCandidate {
    MemoryKey key{0};
    float     strength{0.f};
    uint32_t  access_count{0};
    float     score{0.f};

    bool operator>(const ReplayCandidate& o) const noexcept {
        return score > o.score;
    }
};

// ============================================================================
// MemoryReplay
// ============================================================================

/// Configuration for MemoryReplay.
struct MemoryReplayConfig {
    int   replay_steps = CONSOLIDATION_REPLAY_STEPS;  ///< Steps per replay
    float replay_dt    = 0.001f;                       ///< Physics step size
    int   replay_k     = CONSOLIDATION_REPLAY_K;       ///< Memories to replay
};

/**
 * @brief Replays stored memories through the physics propagator.
 *
 * For each candidate:
 *   1. Load the stored ψ-field into the WaveFunction via SemanticMemory::load().
 *   2. Propagate for `replay_steps` physics steps.
 *   3. Re-store the result so the record reflects post-propagation dynamics.
 *
 * Post-replay, the record's strength is bumped to MAX_STRENGTH (the replay
 * is equivalent to reactivation-based consolidation in neuroscience).
 */
class MemoryReplay {
public:
    using Config = MemoryReplayConfig;

    explicit MemoryReplay(const Config& cfg = Config{}) : cfg_(cfg) {}

    /**
     * @brief Compute a sorted replay queue from the current memory store.
     *
     * Returns up to `replay_k` candidates ranked by descending score.
     *
     * @param mem  SemanticMemory to scan.
     * @return     Sorted vector of ReplayCandidate (highest score first).
     */
    [[nodiscard]]
    std::vector<ReplayCandidate> compute_replay_order(
            const SemanticMemory& mem) const
    {
        std::vector<ReplayCandidate> candidates;
        candidates.reserve(mem.size());

        for (const MemoryKey key : mem.all_keys()) {
            const MemoryRecord* rec = mem.get(key);
            if (!rec) continue;

            ReplayCandidate c;
            c.key          = key;
            c.strength     = rec->strength;
            c.access_count = rec->access_count;
            c.score        = rec->strength
                             * std::log10(1.f + static_cast<float>(rec->access_count));
            candidates.push_back(c);
        }

        // Sort descending by score
        std::sort(candidates.begin(), candidates.end(),
                  [](const ReplayCandidate& a, const ReplayCandidate& b) {
                      return a.score > b.score;
                  });

        // Truncate to replay_k
        const int k = std::min(cfg_.replay_k,
                               static_cast<int>(candidates.size()));
        candidates.resize(static_cast<size_t>(k));
        return candidates;
    }

    /**
     * @brief Execute one replay pass: load → propagate → re-store.
     *
     * @param candidates  Pre-sorted replay queue (from compute_replay_order).
     * @param mem         SemanticMemory (updated in place: old record overwritten
     *                    by post-propagation snapshot).
     * @param wf          WaveFunction used as scratch buffer (modified in place).
     * @return            Number of memories successfully replayed.
     */
    int replay(const std::vector<ReplayCandidate>& candidates,
               SemanticMemory&                     mem,
               WaveFunction&                       wf) const
    {
        Propagator prop;
        int replayed = 0;

        for (const ReplayCandidate& c : candidates) {
            // 1. Restore snapshot into wf
            if (!mem.load(c.key, wf)) continue;

            // 2. Propagate (re-activates the memory in the physics substrate)
            for (int s = 0; s < cfg_.replay_steps; ++s)
                prop.step(wf, cfg_.replay_dt);

            // 3. Re-store the post-propagation state (refreshes spectral content)
            mem.store(wf);   // This overwrites the existing key if dominant
                              // node maps to the same Hilbert position.

            ++replayed;
        }
        return replayed;
    }

    const MemoryReplayConfig& config() const noexcept { return cfg_; }
    MemoryReplayConfig&       config() noexcept       { return cfg_; }

private:
    MemoryReplayConfig cfg_;
};

// ============================================================================
// ConsolidationEngine
// ============================================================================

/// Configuration for ConsolidationEngine.
struct ConsolidationConfig {
    float  nap_dt         = CONSOLIDATION_NAP_DT;     ///< Simulated time per cycle (s)
    size_t max_records    = CONSOLIDATION_MAX_RECORDS; ///< Homeostasis ceiling
    MemoryReplayConfig replay_cfg{};                   ///< Replay configuration
};

/**
 * @brief Orchestrates homeostatic NAP cycles for memory consolidation.
 *
 * A NAP cycle consists of:
 *   1. Decay:        Update memory ages and decay strengths.
 *   2. Consolidate:  Apply LTP boosts and prune weak records.
 *   3. Homeostasis:  If record count exceeds MAX_RECORDS, tighten prune
 *                    by running an additional consolidate().
 *   4. Replay:       Load and re-propagate the top-K memories, re-store.
 *
 * Statistics from the most recent NAP cycle are captured for diagnostics.
 */
class ConsolidationEngine {
public:
    using Config = ConsolidationConfig;

    struct NapStats {
        size_t records_before{0};   ///< Record count before NAP
        size_t pruned{0};           ///< Records pruned during consolidate
        int    replayed{0};         ///< Memories successfully replayed
        size_t records_after{0};    ///< Record count after NAP
    };

    explicit ConsolidationEngine(const Config& cfg = Config{})
        : cfg_(cfg), replay_(cfg.replay_cfg)
    {}

    /**
     * @brief Execute one full NAP cycle.
     *
     * @param mem  SemanticMemory to consolidate (modified in place).
     * @param wf   WaveFunction scratch buffer used during replay.
     * @return     NapStats describing what happened this cycle.
     */
    NapStats nap_cycle(SemanticMemory& mem, WaveFunction& wf) {
        NapStats stats;
        stats.records_before = mem.size();

        // ---- Step 1: Decay
        mem.decay(cfg_.nap_dt);

        // ---- Step 2: Standard consolidate (LTP + prune)
        stats.pruned = mem.consolidate();

        // ---- Step 3: Homeostasis — extra prune if above ceiling
        if (mem.size() > cfg_.max_records) {
            stats.pruned += mem.consolidate();  // second pass tightens prune
        }

        // ---- Step 4: Replay top-K memories
        const auto candidates = replay_.compute_replay_order(mem);
        stats.replayed = replay_.replay(candidates, mem, wf);

        stats.records_after = mem.size();
        last_stats_ = stats;
        return stats;
    }

    /**
     * @brief Lightweight consolidation pass without replay (use during computation).
     *
     * Equivalent to a "micro-consolidation": decay + LTP/prune only.
     *
     * @param mem  SemanticMemory to consolidate.
     * @param dt   Simulated time in seconds.
     * @return     Number of records pruned.
     */
    size_t micro_consolidate(SemanticMemory& mem, float dt) {
        mem.decay(dt);
        return mem.consolidate();
    }

    /**
     * @brief Compute replay order only (non-modifying inspection).
     *
     * @param mem  SemanticMemory to inspect.
     * @return     Sorted replay candidates (highest score first).
     */
    [[nodiscard]]
    std::vector<ReplayCandidate> compute_replay_order(
            const SemanticMemory& mem) const
    {
        return replay_.compute_replay_order(mem);
    }

    /**
     * @brief Check homeostatic health of the memory store.
     *
     * Returns true if record count is within healthy range:
     *   (0, max_records].
     */
    [[nodiscard]]
    bool is_healthy(const SemanticMemory& mem) const noexcept {
        return mem.size() > 0 && mem.size() <= cfg_.max_records;
    }

    /// Statistics from the last nap_cycle() call.
    [[nodiscard]] const NapStats& last_stats() const noexcept { return last_stats_; }

    const ConsolidationConfig& config() const noexcept { return cfg_; }
    ConsolidationConfig&       config() noexcept       { return cfg_; }

private:
    ConsolidationConfig  cfg_;
    MemoryReplay         replay_;
    NapStats             last_stats_{};
};

} // namespace nikola::cognitive
