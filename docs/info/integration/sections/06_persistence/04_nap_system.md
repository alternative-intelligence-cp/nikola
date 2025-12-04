# NAP SYSTEM

## 22.1 Reduced State Processing

During nap, system enters low-power mode:
- Emitters slow down to 10% frequency
- Only critical background tasks run
- Neuroplastic updates deferred

## 22.2 Backlog Processing

**Backlog Queue:**

```cpp
class BacklogProcessor {
    std::queue<std::function<void()>> backlog;

public:
    void add_task(std::function<void()> task) {
        backlog.push(task);
    }

    void process_during_nap() {
        while (!backlog.empty()) {
            auto task = backlog.front();
            backlog.pop();

            task();  // Execute deferred task
        }
    }
};
```

## 22.3 State Saving

Already covered in Section 19 (DMC).

## 22.4 Implementation

**Nap Controller:**

```cpp
class NapController {
    bool in_nap = false;

public:
    void enter_nap(TorusManifold& torus, BacklogProcessor& backlog,
                   PersistenceManager& persistence, DreamWeaveEngine& dream_weave) {
        std::cout << "[NAP] Entering nap state..." << std::endl;

        in_nap = true;

        // 1. Slow emitters (reduce cognitive activity)
        torus.set_emitter_speed(0.1);

        // 2. Process backlog (handle deferred queries)
        backlog.process_during_nap();

        // 3. MEMORY CONSOLIDATION: Transfer high-resonance patterns to long-term storage
        //    This prevents RAM exhaustion and preserves critical context across restarts
        //    Implementation: Identify high-resonance nodes and serialize to LSM
        consolidate_memories(torus, persistence);

        // 4. DreamWeave: Run counterfactual simulations on high-loss interactions
        //    Reinforces pathways that could have led to better outcomes
        dream_weave.run_dream_cycle(torus, mamba, NUM_DREAM_SIMULATIONS);

        // 5. Save state (checkpoint entire torus to disk)
        persistence.trigger_nap(torus);

        // 6. Resume (restore full cognitive activity)
        torus.set_emitter_speed(1.0);

        in_nap = false;

        std::cout << "[NAP] Awake and refreshed." << std::endl;
    }

private:
    // Memory Consolidation: Transfer high-resonance short-term patterns to long-term storage
    // This implements the biological process of memory consolidation during sleep
    void consolidate_memories(TorusManifold& torus, PersistenceManager& persistence) {
        std::cout << "[CONSOLIDATION] Transferring short-term memories to long-term storage..." << std::endl;

        // Configuration
        const double HIGH_RESONANCE_THRESHOLD = 0.7;  // r > 0.7 indicates important memory
        const double MIN_AMPLITUDE_THRESHOLD = 0.5;   // Minimum amplitude to be worth saving
        const size_t MAX_CONSOLIDATE_PER_NAP = 1000;  // Prevent I/O overload

        // 1. Identify high-resonance nodes (important short-term memories)
        std::vector<std::pair<Coord9D, TorusNode>> consolidation_candidates;

        for (const auto& [coord, node] : torus.get_active_nodes()) {
            // Criteria for consolidation:
            // - High resonance (r > 0.7): Low damping → important pattern
            // - Significant amplitude: Not just noise
            // - Currently in RAM but not yet in LSM
            if (node.resonance_r > HIGH_RESONANCE_THRESHOLD &&
                std::abs(node.wavefunction) > MIN_AMPLITUDE_THRESHOLD &&
                !persistence.is_in_long_term_storage(coord)) {

                consolidation_candidates.push_back({coord, node});
            }
        }

        // 2. Sort by importance (amplitude × resonance)
        std::sort(consolidation_candidates.begin(), consolidation_candidates.end(),
                  [](const auto& a, const auto& b) {
                      double importance_a = std::abs(a.second.wavefunction) * a.second.resonance_r;
                      double importance_b = std::abs(b.second.wavefunction) * b.second.resonance_r;
                      return importance_a > importance_b;
                  });

        // 3. Transfer top N candidates to long-term storage (LSM)
        size_t num_consolidated = 0;
        for (const auto& [coord, node] : consolidation_candidates) {
            if (num_consolidated >= MAX_CONSOLIDATE_PER_NAP) {
                break;
            }

            // Serialize node state to LMDB (persistent key-value store)
            // Key: Hilbert curve index (uint64_t) for spatial locality
            // Value: Serialized TorusNode (metric tensor, wavefunction, resonance, etc.)
            uint64_t hilbert_key = HilbertMapper::encode(coord.to_array(), 10);

            persistence.write_to_lsm(hilbert_key, node);

            num_consolidated++;
        }

        // 4. Garbage collection: Prune low-resonance nodes from RAM
        //    These are temporary patterns that didn't consolidate to long-term memory
        size_t num_pruned = torus.prune_low_resonance_nodes(0.3);  // r < 0.3 → ephemeral

        std::cout << "[CONSOLIDATION] Complete: "
                  << num_consolidated << " patterns transferred to long-term storage, "
                  << num_pruned << " ephemeral patterns pruned from RAM" << std::endl;

        // Memory consolidation ensures:
        // - Critical patterns survive system restarts
        // - RAM usage remains bounded (prevents OOM)
        // - Distinction between short-term (RAM) and long-term (disk) memory
    }

    bool is_napping() const { return in_nap; }
};
```

## 22.5 Dream-Weave Counterfactual Simulation

**Status:** MANDATORY - Required for autonomous learning

### Concept

The base specification uses "Nap" cycles primarily for persistence (DMC flushing). This section extends the Nap state into an **active learning phase** where the system simulates counterfactual "what if" scenarios to learn from paths not taken.

### Mechanism

**Counterfactual Generation Algorithm:**

1. **Pause External I/O:** Decouple emitters from user queries
2. **Identify High-Loss Sequences:** Query recent history for interactions where prediction error was high
3. **Inject Quantum Noise:** Use the Quantum dimensions ($u, v, w$) as stochastic perturbation sources
4. **Replay with Variation:** Re-run the Mamba-9D scanner with perturbed initial conditions
5. **Resonance Evaluation:** Measure constructive interference in the alternate timeline
6. **Selective Reinforcement:** If counterfactual outcome > historical outcome, update metric tensor to favor that pathway

**Mathematical Formulation:**

Let $\mathcal{H}_{\text{actual}}$ be the historical sequence and $\mathcal{H}_{\text{cf}}$ be the counterfactual.

**Outcome Metric:**

$$Q(\mathcal{H}) = \sum_{t} |\Psi_t|^2 \cdot r_t$$

Where:
- $|\Psi_t|^2$ is the resonance strength at time $t$
- $r_t$ is the reward received

**Update Rule:**

If $Q(\mathcal{H}_{\text{cf}}) > Q(\mathcal{H}_{\text{actual}})$:

$$g_{ij} \leftarrow g_{ij} - \alpha \cdot \nabla_{g} Q(\mathcal{H}_{\text{cf}})$$

Where $\alpha$ is the counterfactual learning rate (default: 0.001).

### Implementation

**Enhanced Nap Controller:**

```cpp
// File: include/nikola/autonomy/dream_weave.hpp
#pragma once

#include "nikola/physics/torus_manifold.hpp"
#include "nikola/mamba/ssm_kernel.hpp"
#include <vector>
#include <random>

namespace nikola::autonomy {

struct InteractionRecord {
    std::vector<TorusNode> sequence;
    double prediction_error;
    double reward;
    uint64_t timestamp;
};

// Sum-tree data structure for O(log N) prioritized sampling
// Used in DreamWeave for efficient high-error experience replay
class SumTree {
private:
    std::vector<double> tree;     // Binary heap storing cumulative sums
    std::vector<InteractionRecord*> data;  // Leaf nodes (actual data)
    size_t capacity;
    size_t write_idx = 0;
    size_t size_ = 0;

public:
    explicit SumTree(size_t capacity) : capacity(capacity) {
        // Tree has 2*capacity-1 nodes (internal + leaves)
        tree.resize(2 * capacity - 1, 0.0);
        data.resize(capacity, nullptr);
    }

    // Add experience with priority (prediction error)
    void add(InteractionRecord* record, double priority) {
        size_t tree_idx = write_idx + capacity - 1;  // Leaf index in tree

        // Store data at leaf
        data[write_idx] = record;

        // Update tree with new priority
        update(tree_idx, priority);

        // Circular buffer
        write_idx = (write_idx + 1) % capacity;
        if (size_ < capacity) {
            size_++;
        }
    }

    // Update priority at specific tree index
    void update(size_t tree_idx, double priority) {
        double change = priority - tree[tree_idx];
        tree[tree_idx] = priority;

        // Propagate change up the tree
        while (tree_idx > 0) {
            tree_idx = (tree_idx - 1) / 2;  // Parent index
            tree[tree_idx] += change;
        }
    }

    // Sample index based on priority (O(log N))
    size_t sample(double value) const {
        size_t idx = 0;  // Start at root

        while (idx < capacity - 1) {  // Traverse to leaf
            size_t left = 2 * idx + 1;
            size_t right = left + 1;

            if (value <= tree[left]) {
                idx = left;
            } else {
                value -= tree[left];
                idx = right;
            }
        }

        return idx - (capacity - 1);  // Convert tree index to data index
    }

    // Get data at specific index
    InteractionRecord* get(size_t idx) const {
        return data[idx];
    }

    // Get priority at specific data index
    double get_priority(size_t idx) const {
        size_t tree_idx = idx + capacity - 1;
        return tree[tree_idx];
    }

    // Total sum of all priorities
    double total_priority() const {
        return tree[0];
    }

    size_t size() const { return size_; }
};

class DreamWeaveEngine {
    std::deque<InteractionRecord> recent_history;
    std::unique_ptr<SumTree> prioritized_buffer;
    std::mt19937_64 rng;

    const size_t MAX_HISTORY = 1000;
    const double HIGH_LOSS_THRESHOLD = 0.3;
    const int NUM_COUNTERFACTUALS = 5;
    const double PRIORITY_ALPHA = 0.6;  // Prioritization exponent

public:
    DreamWeaveEngine() : rng(std::random_device{}()) {
        // Initialize prioritized replay buffer with sum-tree
        prioritized_buffer = std::make_unique<SumTree>(MAX_HISTORY);
    }

    // Record interaction with priority based on TD-error
    void record_interaction(const std::vector<TorusNode>& sequence,
                           double error,
                           double reward) {
        InteractionRecord record;
        record.sequence = sequence;
        record.prediction_error = error;
        record.reward = reward;
        record.timestamp = std::chrono::system_clock::now().time_since_epoch().count();

        recent_history.push_back(record);

        // Calculate priority: |TD-error|^α (prioritized experience replay)
        // Higher error = higher priority for sampling during dreams
        double priority = std::pow(std::abs(error), PRIORITY_ALPHA);

        // Add to sum-tree with priority
        prioritized_buffer->add(&recent_history.back(), priority);

        // Maintain circular buffer
        if (recent_history.size() > MAX_HISTORY) {
            recent_history.pop_front();
        }
    }

    void run_dream_cycle(TorusManifold& torus,
                        Mamba9D& mamba,
                        int num_simulations = 10);

private:
    std::vector<TorusNode> generate_counterfactual(
        const std::vector<TorusNode>& original);

    double evaluate_outcome(const std::vector<TorusNode>& sequence,
                           TorusManifold& torus,
                           Mamba9D& mamba);

    void inject_quantum_noise(std::vector<TorusNode>& sequence);
};

} // namespace nikola::autonomy
```

**Core Implementation:**

```cpp
// File: src/autonomy/dream_weave.cpp

#include "nikola/autonomy/dream_weave.hpp"
#include <algorithm>

namespace nikola::autonomy {

void DreamWeaveEngine::run_dream_cycle(TorusManifold& torus,
                                       Mamba9D& mamba,
                                       int num_simulations) {
    if (prioritized_buffer->size() == 0) {
        return;  // No experiences to replay
    }

    // PRODUCTION: Prioritized sampling using sum-tree (O(log N) per sample)
    // Samples experiences with probability proportional to |TD-error|^α
    // High-error experiences are replayed more frequently → faster learning
    std::uniform_real_distribution<double> priority_dist(0.0, prioritized_buffer->total_priority());

    std::vector<InteractionRecord*> sampled_records;
    sampled_records.reserve(num_simulations);

    // Sample num_simulations experiences based on priority
    for (int i = 0; i < num_simulations && i < static_cast<int>(prioritized_buffer->size()); ++i) {
        // Sample from priority distribution
        double sample_value = priority_dist(rng);
        size_t idx = prioritized_buffer->sample(sample_value);

        InteractionRecord* record = prioritized_buffer->get(idx);
        if (record && record->prediction_error > HIGH_LOSS_THRESHOLD) {
            sampled_records.push_back(record);
        }
    }

    if (sampled_records.empty()) {
        return;  // No high-loss experiences
    }

    // Generate and evaluate counterfactuals
    for (const auto* record : sampled_records) {
        for (int cf = 0; cf < NUM_COUNTERFACTUALS; ++cf) {
            auto counterfactual = generate_counterfactual(record->sequence);

            double cf_outcome = evaluate_outcome(counterfactual, torus, mamba);
            double actual_outcome = record->reward;

            // Selective reinforcement: Update if counterfactual improved outcome
            if (cf_outcome > actual_outcome) {
                // Update metric tensor to favor this pathway
                std::cout << "[DREAM] Counterfactual improved outcome: "
                          << actual_outcome << " -> " << cf_outcome << std::endl;

                // Apply neuroplasticity update with counterfactual sequence
                torus.trigger_neuroplasticity_update_from_sequence(counterfactual);
            }
        }
    }

    std::cout << "[DREAM] Cycle complete: Sampled " << sampled_records.size()
              << " high-priority experiences (prioritized replay with sum-tree)" << std::endl;
}

std::vector<TorusNode> DreamWeaveEngine::generate_counterfactual(
    const std::vector<TorusNode>& original) {

    auto counterfactual = original;
    inject_quantum_noise(counterfactual);
    return counterfactual;
}

void DreamWeaveEngine::inject_quantum_noise(std::vector<TorusNode>& sequence) {
    std::normal_distribution<double> noise(0.0, 0.1);

    // Energy-bounded perturbation preserves resonance state hierarchy
    // Noise is multiplicative (scaled by existing energy) to respect vacuum states
    // This maintains the distinction between short-term and long-term memories
    for (auto& node : sequence) {
        // Perturb quantum dimensions (u, v, w)
        std::complex<double> u_noise(noise(rng), noise(rng));
        std::complex<double> v_noise(noise(rng), noise(rng));
        std::complex<double> w_noise(noise(rng), noise(rng));

        // Combined noise vector
        std::complex<double> total_noise = u_noise + v_noise + w_noise;

        // Multiplicative noise scaled by existing energy (preserves vacuum)
        // High-energy nodes (important memories) get larger perturbations
        // Low-energy nodes (weak memories) get proportionally smaller noise
        double current_energy = std::abs(node.wavefunction);

        // Apply multiplicative noise (10% of current amplitude)
        node.wavefunction += 0.1 * current_energy * total_noise;

        // Energy conservation: Clamp to maximum nonary amplitude (±4)
        // This respects the physical constraint from balanced nonary encoding
        // Max amplitude: 4.0 (maps to Nit::POS4 or Nit::NEG4)
        double amplitude = std::abs(node.wavefunction);
        if (amplitude > 4.0) {
            double phase = std::arg(node.wavefunction);
            node.wavefunction = std::polar(4.0, phase);  // Preserve phase, clamp to max Nit
        }

        // Additional resonance preservation:
        // The resonance_r dimension is NOT modified, preserving the damping hierarchy
        // High resonance nodes (r → 1.0) maintain low damping (long-term memory)
        // Low resonance nodes (r → 0.0) maintain high damping (temporary patterns)
    }

    // No normalization step - energy distribution is meaningful and must be preserved
    // The metric tensor g_ij will naturally balance energy distribution during propagation
}

double DreamWeaveEngine::evaluate_outcome(const std::vector<TorusNode>& sequence,
                                          TorusManifold& torus,
                                          Mamba9D& mamba) {
    // Run Mamba forward pass
    auto hidden_state = mamba.forward(sequence);

    // Measure resonance
    double resonance = 0.0;
    for (const auto& node : sequence) {
        resonance += std::norm(node.wavefunction) * node.resonance_r;
    }

    return resonance / sequence.size();
}

} // namespace nikola::autonomy
```

---

**Cross-References:**
- See Section 19 for DMC persistence mechanism
- See Section 14 for Neurochemistry triggers (dopamine, boredom)
- See Section 15 for Training Systems integration
- See Section 7 for Mamba-9D forward pass
- See Section 3 for Metric tensor neuroplasticity
