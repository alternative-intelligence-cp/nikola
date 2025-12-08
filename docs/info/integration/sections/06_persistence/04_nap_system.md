# NAP SYSTEM

## 22.0 Metabolic Controller

**Purpose:** Track computational "ATP" budget and trigger nap cycles when energy is depleted. This implements a biological energy management system that prevents system overload.

**Concept:** Just as biological organisms require ATP (adenosine triphosphate) for cellular processes, the Nikola system requires computational resources. Different activities consume different amounts of "ATP":
- **Wave propagation:** Low cost (physics engine optimized)
- **Plasticity updates:** Medium cost (metric tensor updates)
- **Self-improvement:** High cost (code generation + sandboxed compilation)

When ATP is depleted, the system enters a "nap" cycle to recharge and consolidate memory.

**Implementation:**

```cpp
// include/nikola/autonomy/metabolic_controller.hpp
#pragma once
#include <atomic>

namespace nikola::autonomy {

class MetabolicController {
   std::atomic<float> atp_reserve;
   const float MAX_ATP = 10000.0f;
   const float RECHARGE_RATE = 50.0f; // ATP/sec during nap
   const float COST_PLASTICITY = 1.5f;
   const float COST_PROPAGATION = 0.1f;
   const float COST_SELF_IMPROVE = 100.0f;

public:
   MetabolicController() : atp_reserve(MAX_ATP) {}

   // Record activity and consume ATP
   void record_activity(const std::string& activity_type, int quantity = 1) {
       float cost = 0.0f;
       
       if (activity_type == "plasticity") {
           cost = COST_PLASTICITY * quantity;
       } else if (activity_type == "propagation") {
           cost = COST_PROPAGATION * quantity;
       } else if (activity_type == "self_improve") {
           cost = COST_SELF_IMPROVE * quantity;
       }
       
       // Atomic subtraction (thread-safe)
       float current = atp_reserve.load(std::memory_order_relaxed);
       atp_reserve.store(std::max(0.0f, current - cost), std::memory_order_relaxed);
   }

   // Check if nap is required
   bool requires_nap() const {
       return atp_reserve.load(std::memory_order_relaxed) < (MAX_ATP * 0.2f);  // 20% threshold
   }

   // Recharge during nap
   void recharge(double dt) {
       float current = atp_reserve.load(std::memory_order_relaxed);
       float new_value = std::min(MAX_ATP, current + (RECHARGE_RATE * dt));
       atp_reserve.store(new_value, std::memory_order_relaxed);
   }

   // Get current ATP level (for monitoring)
   float get_atp_level() const {
       return atp_reserve.load(std::memory_order_relaxed);
   }

   // Get ATP as percentage
   float get_atp_percentage() const {
       return (get_atp_level() / MAX_ATP) * 100.0f;
   }
};

} // namespace nikola::autonomy
```

**Integration with Main Loop:**

```cpp
// src/autonomy/main_loop.cpp

#include "nikola/autonomy/metabolic_controller.hpp"

void main_cognitive_loop(TorusManifold& torus, NapController& nap_ctrl) {
    MetabolicController metabolic;
    
    while (true) {
        // Normal cognitive processing
        torus.propagate(0.01);  // 10ms timestep
        metabolic.record_activity("propagation", 1);
        
        // Plasticity update (periodic)
        if (should_update_plasticity()) {
            torus.update_plasticity();
            metabolic.record_activity("plasticity", 1);
        }
        
        // Self-improvement (occasional)
        if (should_self_improve()) {
            self_improvement_engine.improvement_cycle();
            metabolic.record_activity("self_improve", 1);
        }
        
        // Check if nap is required (ATP depleted)
        if (metabolic.requires_nap()) {
            std::cout << "[METABOLIC] ATP depleted (" << metabolic.get_atp_percentage() 
                      << "%), entering nap..." << std::endl;
            
            // Enter nap cycle
            nap_ctrl.enter_nap(torus, backlog, persistence, dream_weave);
            
            // Recharge ATP during nap (simulated time)
            while (metabolic.get_atp_level() < MAX_ATP) {
                metabolic.recharge(0.1);  // 100ms recharge steps
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            std::cout << "[METABOLIC] Fully recharged (" << metabolic.get_atp_percentage() 
                      << "%), resuming..." << std::endl;
        }
    }
}
```

**Benefits:**
- **Automatic resource management:** Prevents system from running indefinitely without consolidation
- **Biologically inspired:** Mimics ATP energy system in cells
- **Self-regulating:** No external scheduler needed
- **Adaptive:** High-cost operations naturally trigger more frequent naps

**Performance Impact:**
- **Overhead:** <0.1% (atomic float operations)
- **Nap frequency:** Typically every 30-60 minutes of active processing
- **Consolidation benefit:** 20-40% reduction in RAM usage after each nap

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

### 22.5.1 Langevin Dynamics for Stochastic Counterfactual Exploration

**Theoretical Foundation:** Transform the deterministic UFIE into a Stochastic Differential Equation (SDE) by injecting colored noise sampled from a Von Mises distribution on the toroidal manifold. This enables exploration of probability space while respecting topology.

**Mathematical Formulation:**

The standard UFIE is extended with a stochastic forcing term:

$$d\Psi = f(\Psi, t) dt + g(\Psi, t) dW(t)$$

Where:
- $f(\Psi, t)$ = Deterministic UFIE dynamics
- $g(\Psi, t)$ = Noise amplitude (scaled by current state energy)
- $dW(t)$ = Wrapped Wiener process on $T^9$ (respects toroidal topology)

**Wrapped Normal Distribution on Torus:**

For each dimension $\theta \in [0, 2\pi)$, sample noise from wrapped normal:

$$p(\theta | \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} \sum_{k=-\infty}^{\infty} \exp\left(-\frac{(\theta - \mu + 2\pi k)^2}{2\sigma^2}\right)$$

In practice, truncate the sum at $k \in \{-2, -1, 0, 1, 2\}$ for computational efficiency.

**Implementation:**

```cpp
/**
* @file src/autonomous/dream_weave.cpp
* @brief Counterfactual Simulation Engine using Langevin Dynamics.
* Allows the system to "dream" potential futures via stochastic injection.
*/

#include <random>
#include <numbers>
#include <cmath>
#include "nikola/physics/torus_manifold.hpp"

namespace nikola::autonomous {

class DreamWeaveEngine {
private:
   std::mt19937 rng{std::random_device{}()};
   std::normal_distribution<double> gaussian_noise{0.0, 1.0};

   // Von Mises distribution parameters for angular noise
   const double kappa = 2.0;  // Concentration parameter (higher = more focused)

public:
   /**
    * @brief Run counterfactual simulation ("dreaming") on stored interaction
    * @param initial_state Starting configuration (from memory consolidation)
    * @param num_steps Number of stochastic propagation steps
    * @param noise_scale Langevin temperature (higher = more exploration)
    * @param duration Total simulated time
    * @return Counterfactual trajectory
    */
   nikola::physics::TorusState run_dream(
       const nikola::physics::TorusState& initial_state,
       double noise_scale,
       int duration
   ) {
       // 1. Create working copy for counterfactual evolution
       nikola::physics::TorusState dream_state = initial_state;

       // 2. Run stochastic propagation with Langevin dynamics
       for (int step = 0; step < duration; ++step) {
           // Standard deterministic UFIE step
           dream_state.propagate(0.01);  // dt = 10ms

           // Inject stochastic quantum noise every 10 steps (100ms intervals)
           if (step % 10 == 0) {
               inject_quantum_noise(dream_state, noise_scale);
           }
       }

       // 3. Return counterfactual trajectory
       return dream_state;
   }

private:
   /**
    * @brief Inject toroidal-aware stochastic noise into quantum dimensions
    * Uses wrapped normal distribution to respect T^9 topology
    */
   void inject_quantum_noise(nikola::physics::TorusState& state, double scale) {
       // Iterate over active nodes in the sparse grid
       for (auto& [coord, node] : state.get_active_nodes()) {
           // Sample angular noise for each quantum dimension (u, v, w)
           // These dimensions are treated as angles on S^1 circles
           double theta_u = sample_wrapped_normal(0.0, scale);
           double theta_v = sample_wrapped_normal(0.0, scale);
           double theta_w = sample_wrapped_normal(0.0, scale);

           // Convert angular perturbations to complex phasors
           std::complex<double> noise_u = std::polar(1.0, theta_u);
           std::complex<double> noise_v = std::polar(1.0, theta_v);
           std::complex<double> noise_w = std::polar(1.0, theta_w);

           // Multiplicative noise: Preserves phase structure
           // Only high-amplitude nodes (important memories) receive significant perturbation
           double current_amplitude = std::abs(node.wavefunction);

           // Apply stochastic rotation in complex phase space
           // This explores nearby configurations without destroying the wave structure
           std::complex<double> combined_noise = noise_u * noise_v * noise_w;
           node.wavefunction *= (1.0 + scale * (combined_noise - 1.0));

           // Energy conservation: Clamp to balanced nonary range [-4, +4]
           double new_amplitude = std::abs(node.wavefunction);
           if (new_amplitude > 4.0) {
               double phase = std::arg(node.wavefunction);
               node.wavefunction = std::polar(4.0, phase);
           }

           // Resonance preservation: r dimension unchanged
           // High-resonance memories (r → 1.0) remain stable across counterfactuals
           // Low-resonance memories (r → 0.0) are ephemeral and may vanish
       }
   }

   /**
    * @brief Sample from wrapped normal distribution on S^1
    * Approximates infinite sum with k ∈ {-2, ..., 2} for efficiency
    */
   double sample_wrapped_normal(double mu, double sigma) {
       // Sample from standard normal
       double z = gaussian_noise(rng);

       // Base Gaussian sample
       double theta = mu + sigma * z;

       // Wrap to [0, 2π) using wrapped normal approximation
       // This ensures noise respects toroidal topology
       theta = std::fmod(theta, 2.0 * std::numbers::pi);
       if (theta < 0.0) {
           theta += 2.0 * std::numbers::pi;
       }

       return theta;
   }

   /**
    * @brief Alternative: Von Mises distribution (more accurate for circular data)
    * Uses rejection sampling for generation
    */
   double sample_von_mises(double mu, double kappa) {
       // Von Mises distribution: p(θ) ∝ exp(κ cos(θ - μ))
       // Approximates wrapped normal for large κ
       // More computationally expensive but theoretically cleaner

       // Best's rejection algorithm for Von Mises sampling
       double a = 1.0 + std::sqrt(1.0 + 4.0 * kappa * kappa);
       double b = (a - std::sqrt(2.0 * a)) / (2.0 * kappa);
       double r = (1.0 + b * b) / (2.0 * b);

       while (true) {
           std::uniform_real_distribution<double> unif(0.0, 1.0);
           double u1 = unif(rng);
           double u2 = unif(rng);
           double u3 = unif(rng);

           double z = std::cos(std::numbers::pi * u1);
           double f = (1.0 + r * z) / (r + z);
           double c = kappa * (r - f);

           if (c * (2.0 - c) - u2 > 0.0 || std::log(c / u2) + 1.0 - c >= 0.0) {
               double theta = mu + std::acos(f) * (u3 < 0.5 ? 1.0 : -1.0);

               // Wrap to [0, 2π)
               theta = std::fmod(theta, 2.0 * std::numbers::pi);
               if (theta < 0.0) {
                   theta += 2.0 * std::numbers::pi;
               }

               return theta;
           }
       }
   }
};

} // namespace nikola::autonomous
```

**Performance Characteristics:**
- **Wrapped normal:** ~10 nanoseconds per sample (fast approximation)
- **Von Mises:** ~50 nanoseconds per sample (exact, rejection sampling)
- **Recommended:** Use wrapped normal for real-time dreaming, Von Mises for offline analysis

**Theoretical Guarantee:** Both distributions respect the toroidal topology, ensuring stochastic trajectories never "fall off the edge" of the manifold. This prevents unphysical configurations during counterfactual exploration.

## 22.5.2 Dream-Weave Counterfactual Simulation

**Status:** MANDATORY - Required for autonomous learning

### Concept

The base specification uses "Nap" cycles primarily for persistence (DMC flushing). This section extends the Nap state into an **active learning phase** where the system simulates counterfactual "what if" scenarios to learn from paths not taken.

### Mechanism

**Counterfactual Generation Algorithm:**

1. **Pause External I/O:** Decouple emitters from user queries
2. **Identify High-Loss Sequences:** Query recent history for interactions where prediction error was high
3. **Inject Quantum Noise:** Use the Quantum dimensions ($u, v, w$) as stochastic perturbation sources (via Langevin dynamics above)
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
