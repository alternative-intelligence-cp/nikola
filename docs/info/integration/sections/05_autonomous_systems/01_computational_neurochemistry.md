# COMPUTATIONAL NEUROCHEMISTRY

## 14.1 Dopamine System

**Dopamine** ($D_t$) is a global scalar variable that modulates learning rate and exploration.

### Update Rule

$$D(t+1) = D(t) + \beta \cdot \delta_t - \lambda_{\text{decay}} \cdot (D(t) - D_{\text{baseline}})$$

Where:
- $\delta_t$: Reward prediction error (TD error)
- $\beta$: Dopamine sensitivity (typically 0.1)
- $\lambda_{\text{decay}}$: Decay constant (typically 0.01)
- $D_{\text{baseline}}$: Homeostatic baseline (typically 0.5)

### Reward Prediction Error

$$\delta_t = R_t + \gamma V(S_{t+1}) - V(S_t)$$

Where:
- $R_t$: Immediate reward (1 for success, -1 for failure, 0 otherwise)
- $\gamma$: Discount factor (0.99)
- $V(S_t)$: Value estimate of current state

### Effects of Dopamine

| Dopamine Level | Effect | Behavior |
|----------------|--------|----------|
| High ($> 0.7$) | ↑ Learning rate, ↑ Exploration | Risk-taking, rapid learning |
| Medium ($0.3-0.7$) | Balanced | Normal operation |
| Low ($< 0.3$) | ↓ Learning rate, ↑ Exploitation | Conservative, slow learning |

### Implementation

```cpp
class DopamineSystem {
    double level = 0.5;  // Baseline
    double baseline = 0.5;
    double beta = 0.1;
    double lambda_decay = 0.01;
    double gamma = 0.99;

public:
    void update(double reward, double value_current, double value_next) {
        // Compute TD error
        double delta = reward + gamma * value_next - value_current;

        // Update dopamine
        level += beta * delta - lambda_decay * (level - baseline);

        // Clamp to [0, 1]
        level = std::clamp(level, 0.0, 1.0);
    }

    double get_learning_rate(double base_lr = 0.001) const {
        // Modulate learning rate
        return base_lr * (1.0 + std::tanh(level - baseline));
    }

    double get_exploration_temp() const {
        // Higher dopamine → higher temperature → more exploration
        return 0.5 + level;
    }

    double get_level() const { return level; }
};
```

### 14.1.1 Neuro-Physical Coupling

**Critical Implementation:** Dopamine must physically modulate the physics engine learning rate.

The learning rate $\eta$ (plasticity) of the metric tensor is a function of Dopamine $D(t)$:

$$\eta(t) = \eta_{base} \cdot (1 + \tanh(D(t)))$$

**Behavior:**

- **High Dopamine** ($D > 0.8$): $\tanh$ approaches 1, doubling the learning rate. Metric tensor becomes highly plastic (rapid learning/encoding).
- **Low Dopamine** ($D < 0.2$): $\tanh$ approaches 0. System enters consolidation mode, resisting geometry changes to protect existing memories.

**Implementation Hook:** In `src/physics/torus_manifold.cpp`, the `update_metric_tensor()` function must query ENGS:

```cpp
// In Physics Engine Loop (Plasticity Update)
void apply_neuroplasticity(TorusGridSoA& grid, const ENGS_State& engs) {
    float learning_modulator = 1.0f + std::tanh(engs.dopamine);
    
    #pragma omp parallel for
    for (size_t i = 0; i < grid.num_nodes; ++i) {
        // Hebbian update weighted by dopamine
        float psi_magnitude = std::sqrt(
            grid.psi_real[i] * grid.psi_real[i] +
            grid.psi_imag[i] * grid.psi_imag[i]
        );
        
        // Update each metric tensor component
        for (int comp = 0; comp < 45; ++comp) {
            float delta = learning_modulator * psi_magnitude * 0.001f;  // Small step
            grid.metric_tensor[comp][i] += delta;
        }
    }
}
```

**Warning:** This coupling is NOT metadata—it's a control loop variable. Failure to implement this breaks the feedback loop between cognitive state (dopamine) and physical substrate (metric tensor).

## 14.2 Boredom and Curiosity

**Boredom** ($B_t$) accumulates when information entropy is low.

### Boredom Update

$$B(t+1) = B(t) + \frac{\alpha}{H(\Psi(t)) + \epsilon} - \kappa \cdot D(t)$$

Where:
- $H(\Psi)$: Shannon entropy of wavefunction distribution
- $\alpha$: Boredom accumulation rate (0.01)
- $\epsilon$: Small constant to prevent division by zero (0.001)
- $\kappa$: Dopamine suppression factor (0.05)

### Entropy Calculation

$$H(\Psi) = -\sum_i p_i \log_2 p_i$$

Where $p_i = \frac{|\Psi_i|^2}{\sum_j |\Psi_j|^2}$ (probability distribution from wavefunction amplitudes).

### Curiosity Trigger

When $B(t) > B_{\text{critical}}$ (typically 5.0), trigger curiosity routine:

1. Select random high-entropy topic from knowledge graph
2. Query Tavily for that topic
3. Ingest and embed results
4. Reset boredom: $B(t) \leftarrow 0$

### Implementation

```cpp
class BoredomCuriositySystem {
    double boredom = 0.0;
    double critical_threshold = 5.0;
    double alpha = 0.01;
    double kappa = 0.05;

public:
    // Update boredom with time-step scaling for frame-rate independence
    void update(const TorusManifold& torus, double dopamine, double dt) {
        // Compute entropy
        double entropy = compute_entropy(torus);

        // Update boredom with time-step scaling (frame-rate independent)
        boredom += (alpha / (entropy + 0.001) - kappa * dopamine) * dt;

        // Clamp
        boredom = std::max(0.0, boredom);
    }

    bool should_explore() const {
        return boredom > critical_threshold;
    }

    void reset_boredom() {
        boredom = 0.0;
    }

private:
    double compute_entropy(const TorusManifold& torus) {
        std::vector<double> probabilities;
        double total = 0.0;

        // Collect amplitudes
        for (const auto& [coord, node] : torus.get_active_nodes()) {
            double amp_sq = std::norm(node.wavefunction);
            probabilities.push_back(amp_sq);
            total += amp_sq;
        }

        // Normalize
        for (auto& p : probabilities) {
            p /= total;
        }

        // Compute entropy
        double entropy = 0.0;
        for (double p : probabilities) {
            if (p > 1e-10) {
                entropy -= p * std::log2(p);
            }
        }

        return entropy;
    }
};
```

## 14.3 Goal System

Goals are organized in a **Directed Acyclic Graph (DAG)** with three tiers:

```
Long-Term Goal
    ├── Mid-Term Goal 1
    │       ├── Short-Term Task 1.1
    │       └── Short-Term Task 1.2
    └── Mid-Term Goal 2
            ├── Short-Term Task 2.1
            └── Short-Term Task 2.2
```

### Goal Structure

```cpp
struct Goal {
    std::string id;
    std::string description;
    GoalTier tier;
    double reward_value;
    std::vector<std::string> prerequisites;  // Child goal IDs
    bool completed = false;
};

enum class GoalTier {
    SHORT_TERM,   // Minutes to hours
    MID_TERM,     // Hours to days
    LONG_TERM     // Days to weeks
};
```

### Goal Graph

```cpp
class GoalSystem {
    std::unordered_map<std::string, Goal> goals;
    std::string current_goal_id;

public:
    void add_goal(const Goal& goal) {
        goals[goal.id] = goal;
    }

    void complete_goal(const std::string& goal_id, DopamineSystem& dopamine) {
        auto& goal = goals.at(goal_id);
        goal.completed = true;

        // Release dopamine
        dopamine.update(goal.reward_value, 0.0, 0.0);

        // Check if parent goals can be completed
        propagate_completion(goal_id, dopamine);
    }

private:
    void propagate_completion(const std::string& child_id, DopamineSystem& dopamine) {
        // Find parent goals
        for (auto& [id, goal] : goals) {
            if (std::find(goal.prerequisites.begin(), goal.prerequisites.end(), child_id)
                != goal.prerequisites.end()) {

                // Check if all prerequisites completed
                bool all_done = true;
                for (const auto& prereq_id : goal.prerequisites) {
                    if (!goals.at(prereq_id).completed) {
                        all_done = false;
                        break;
                    }
                }

                if (all_done && !goal.completed) {
                    complete_goal(id, dopamine);  // Recursive
                }
            }
        }
    }
};
```

## 14.4 Reward Mechanisms

### Reward Sources

| Event | Reward | Trigger |
|-------|--------|---------|
| Query answered from memory | +0.5 | Resonance found |
| Query required external tool | +0.1 | Tool success |
| External tool failed | -0.3 | Tool error |
| Goal completed (short-term) | +0.5 | Goal system |
| Goal completed (mid-term) | +1.0 | Goal system |
| Goal completed (long-term) | +2.0 | Goal system |
| Prediction correct | +0.2 | Transformer training |
| Prediction wrong | -0.1 | Transformer training |
| Nap completed | +0.05 | Persistence system |

**IMPORTANT:** Negative rewards are ONLY for grave instances. Most feedback is positive or neutral.

## 14.5 Implementation

### Neurochemistry Manager

```cpp
class NeurochemistryManager {
    DopamineSystem dopamine;
    BoredomCuriositySystem boredom;
    GoalSystem goals;

public:
    void update(const TorusManifold& torus) {
        // Update boredom
        boredom.update(torus, dopamine.get_level());

        // Check if should explore
        if (boredom.should_explore()) {
            trigger_curiosity();
        }
    }

    double get_learning_rate() const {
        return dopamine.get_learning_rate();
    }

    void reward(double value) {
        dopamine.update(value, 0.0, 0.0);
    }

    void complete_goal(const std::string& goal_id) {
        goals.complete_goal(goal_id, dopamine);
    }

private:
    // PRODUCTION: Dynamic curiosity topic generation based on Knowledge Frontier
    // Analyzes high-entropy regions in the torus to identify unexplored conceptual spaces
    void trigger_curiosity() {
        // Query Knowledge Frontier from torus for high-entropy regions
        // High entropy indicates conceptual boundaries where new information is needed
        std::vector<KnowledgeFrontier> frontiers = identify_knowledge_frontiers();

        if (frontiers.empty()) {
            // Fallback: If no frontiers identified, use meta-learning strategy
            std::cout << "[CURIOSITY] No knowledge frontiers found, using meta-exploration" << std::endl;

            // Generate meta-level exploration queries
            std::vector<std::string> meta_topics = {
                "connections between existing knowledge domains",
                "contradictions in current understanding requiring resolution",
                "gaps in causal models based on observed patterns"
            };

            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, meta_topics.size() - 1);

            std::string topic = meta_topics[dis(gen)];
            tavily_query(topic);
        } else {
            // PRODUCTION: Query highest-entropy frontier
            std::sort(frontiers.begin(), frontiers.end(),
                     [](const auto& a, const auto& b) { return a.entropy > b.entropy; });

            std::string topic = frontiers[0].conceptual_description;
            std::cout << "[CURIOSITY] Exploring knowledge frontier: " << topic << std::endl;
            tavily_query(topic);
        }

        boredom.reset_boredom();
    }
};
```

## 14.6 Metabolic Energy Budget

**Critical Thermodynamic Regulation:** Biological brains consume metabolic energy (ATP) and require rest when depleted. The Nikola Model implements analogous virtual energy management to prevent runaway plasticity and enforce natural consolidation cycles.

**Implementation:**

```cpp
/**
* @file include/nikola/autonomy/metabolic_controller.hpp
* @brief Manages system energy budget and enforces rest cycles.
*/
#pragma once
#include <atomic>
#include <cmath>
#include <chrono>

namespace nikola::autonomy {

class MetabolicController {
private:
   std::atomic<float> atp_reserve;
   const float MAX_ATP = 10000.0f;
   const float RECHARGE_RATE = 50.0f; // ATP per second during rest
   
   // Cost constants
   const float COST_PROPAGATION_STEP = 0.1f;
   const float COST_PLASTICITY_UPDATE = 1.5f; // Expensive!
   const float COST_EXTERNAL_TOOL = 5.0f;     // Very expensive
   
public:
   MetabolicController() : atp_reserve(MAX_ATP) {}

   // Called by Physics Engine
   void record_activity(int num_nodes, bool plasticity_active) {
       float cost = num_nodes * COST_PROPAGATION_STEP;
       if (plasticity_active) {
           cost += num_nodes * COST_PLASTICITY_UPDATE;
       }
       consume(cost);
   }

   // Called by Orchestrator
   void record_tool_usage() {
       consume(COST_EXTERNAL_TOOL);
   }

   // Recharge function (called during "Nap" state)
   void recharge(double dt_seconds) {
       float current = atp_reserve.load();
       float new_val = std::min(MAX_ATP, current + (float)(RECHARGE_RATE * dt_seconds));
       atp_reserve.store(new_val);
   }

   // Returns a fatigue factor [0.0, 1.0]
   // 0.0 = Fresh, 1.0 = Exhausted
   float get_fatigue_level() const {
       float current = atp_reserve.load();
       return 1.0f - (current / MAX_ATP);
   }

   // Should the system enter forced nap mode?
   bool requires_nap() const {
       return atp_reserve.load() < (MAX_ATP * 0.15f); // 15% threshold
   }

private:
   void consume(float amount) {
       float current = atp_reserve.load();
       float new_val = std::max(0.0f, current - amount);
       atp_reserve.store(new_val);
   }
};

} // namespace nikola::autonomy
```

**Integration with Orchestrator:**

The controller forces scheduled "Nap" cycles when ATP reserves drop below 15%. During naps:
- External inputs are ignored
- System performs memory consolidation
- State is saved to disk via DMC
- Virtual energy recharges

This mechanism naturally regulates the pace of learning and prevents catastrophic forgetting associated with continuous unbounded plasticity.

**Integration with Physics Engine:**

```cpp
// In main physics loop
void PhysicsEngine::step(double dt, MetabolicController& metabolism) {
    // Record computational cost
    bool plasticity_active = (dopamine_level > 0.3);
    metabolism.record_activity(active_node_count, plasticity_active);
    
    // Check for forced rest
    if (metabolism.requires_nap()) {
        trigger_nap_cycle();
        return; // Skip this physics step
    }
    
    // Normal propagation
    propagate_wave_kernel<<<blocks, threads>>>(grid, dt);
}
            std::uniform_int_distribution<size_t> dist(0, meta_topics.size() - 1);
            std::string topic = meta_topics[dist(gen)];

            std::cout << "[CURIOSITY] Exploring meta-topic: " << topic << std::endl;
        } else {
            // Select frontier with highest entropy (maximum uncertainty)
            auto max_frontier = std::max_element(
                frontiers.begin(), frontiers.end(),
                [](const KnowledgeFrontier& a, const KnowledgeFrontier& b) {
                    return a.entropy < b.entropy;
                }
            );

            // Generate natural language query from frontier region
            std::string curiosity_query = generate_query_from_frontier(*max_frontier);

            std::cout << "[CURIOSITY] Exploring frontier region (entropy="
                      << max_frontier->entropy << "): " << curiosity_query << std::endl;

            // Mark frontier as being explored
            mark_frontier_explored(max_frontier->region_id);
        }

        boredom.reset_boredom();
    }

    // Knowledge Frontier: Region in manifold with high gradient/uncertainty
    struct KnowledgeFrontier {
        uint64_t region_id;           // Hilbert index of region center
        double entropy;                // Shannon entropy of local patterns
        std::vector<std::string> related_concepts;  // Known concepts nearby
        double exploration_recency;    // Time since last explored (for decay)
    };

    // Identify high-entropy regions indicating knowledge boundaries
    std::vector<KnowledgeFrontier> identify_knowledge_frontiers() {
        std::vector<KnowledgeFrontier> frontiers;

        // Scan active regions of torus for high-gradient boundaries
        // High gradient = sharp transition between learned and unknown
        for (const auto& [coord, node] : torus.get_active_nodes()) {
            // Calculate local entropy using neighbor divergence
            double local_entropy = calculate_local_entropy(coord);

            // Threshold for "frontier" status (high uncertainty)
            const double FRONTIER_THRESHOLD = 2.5;  // Bits of entropy

            if (local_entropy > FRONTIER_THRESHOLD) {
                KnowledgeFrontier frontier;
                frontier.region_id = HilbertMapper::encode(coord.to_array(), 10);
                frontier.entropy = local_entropy;

                // Query nearby concepts from database
                frontier.related_concepts = db.query_nearby_concepts(frontier.region_id);

                // Check if recently explored (avoid repetition)
                frontier.exploration_recency = db.get_exploration_recency(frontier.region_id);

                // Only include if not explored recently (decay > 24 hours)
                if (frontier.exploration_recency > 24.0) {
                    frontiers.push_back(frontier);
                }
            }
        }

        return frontiers;
    }

    // Calculate Shannon entropy of local neighborhood
    double calculate_local_entropy(const Coord9D& coord) {
        auto neighbors = torus.get_neighbors(coord);

        // Collect wavefunction amplitudes
        std::vector<double> amplitudes;
        for (const auto& neighbor : neighbors) {
            amplitudes.push_back(std::abs(neighbor.wavefunction));
        }

        // Normalize to probability distribution
        double total = std::accumulate(amplitudes.begin(), amplitudes.end(), 0.0);
        if (total < 1e-10) return 0.0;

        std::vector<double> probabilities;
        for (double amp : amplitudes) {
            probabilities.push_back(amp / total);
        }

        // Shannon entropy: H = -Σ(p * log2(p))
        double entropy = 0.0;
        for (double p : probabilities) {
            if (p > 1e-10) {
                entropy -= p * std::log2(p);
            }
        }

        return entropy;
    }

    // Generate natural language query from frontier context
    std::string generate_query_from_frontier(const KnowledgeFrontier& frontier) {
        // Use related concepts to formulate exploration query
        if (frontier.related_concepts.empty()) {
            return "explore unknown conceptual space at frontier region "
                   + std::to_string(frontier.region_id);
        }

        // Construct query connecting known concepts (knowledge gap)
        std::string query = "explore connections between ";
        for (size_t i = 0; i < std::min(frontier.related_concepts.size(), size_t(3)); ++i) {
            query += frontier.related_concepts[i];
            if (i < frontier.related_concepts.size() - 1) {
                query += " and ";
            }
        }

        return query;
    }

    // Mark frontier as explored to prevent redundant curiosity
    void mark_frontier_explored(uint64_t region_id) {
        db.update_exploration_timestamp(region_id, std::time(nullptr));
    }
};
```

## 14.6 Extended Neurochemical Gating System (ENGS)

**Status:** MANDATORY - Required for system stability

### Serotonin ($S_t$): Metric Elasticity Regulator

**Function:** Controls the stability/plasticity trade-off in the Riemannian manifold.

**Physical Mapping:**

$$S_t \rightarrow \lambda(t) = \lambda_{\text{base}} \cdot (0.5 + 0.5 \cdot \tanh(S_t - 0.5))$$

Where $\lambda$ is the elastic relaxation constant in the neuroplasticity equation:

$$\frac{\partial g_{ij}}{\partial t} = -\eta(D_t) \cdot \text{Re}(\Psi_i \cdot \Psi_j^*) + \lambda(S_t)(g_{ij} - \delta_{ij})$$

**Effect:**

- **High $S_t$ (> 0.7):** $\lambda$ increases → Metric tensor resists deformation → System "crystallizes" learned patterns → Exploitation mode
- **Low $S_t$ (< 0.3):** $\lambda$ decreases → Metric becomes highly plastic → Rapid restructuring → Exploration mode

### Norepinephrine ($N_t$): Global Arousal Regulator

**Function:** Controls the refractive index (State dimension $s$) globally, modulating "thinking speed."

**Physical Mapping:**

$$s_{\text{global}}(t) = s_{\text{local}} \cdot \frac{1}{1 + N_t}$$

**Effect:**

- **High $N_t$ (> 0.8):** Reduces $s$ globally → Increases wave velocity $c = c_0 / (1 + s)$ → Fast, shallow processing
- **Low $N_t$ (< 0.2):** $s$ remains high → Slow wave propagation → Deep, nuanced resonance

### Implementation

```cpp
class ExtendedNeurochemistry {
    DopamineSystem dopamine;
    BoredomCuriositySystem boredom;

    // Extended neurochemicals
    double serotonin = 0.5;       // Stability
    double norepinephrine = 0.5;  // Arousal

    // Baselines loaded from configuration file for tuning
    double dopamine_baseline;
    double serotonin_baseline;
    double norepinephrine_baseline;
    double boredom_baseline;

    // Decay rates (also configurable)
    double serotonin_decay_rate;
    double norepinephrine_decay_rate;

public:
    // Constructor loads baselines from configuration file
    ExtendedNeurochemistry(const Config& config) {
        // Load baselines from nikola.conf with sensible defaults
        dopamine_baseline = config.get_double("neurochemistry.dopamine_baseline", 0.5);
        serotonin_baseline = config.get_double("neurochemistry.serotonin_baseline", 0.5);
        norepinephrine_baseline = config.get_double("neurochemistry.norepinephrine_baseline", 0.5);
        boredom_baseline = config.get_double("neurochemistry.boredom_baseline", 0.0);

        // Load decay rates
        serotonin_decay_rate = config.get_double("neurochemistry.serotonin_decay", 0.01);
        norepinephrine_decay_rate = config.get_double("neurochemistry.norepinephrine_decay", 0.05);

        // Initialize neurochemical levels to baselines
        serotonin = serotonin_baseline;
        norepinephrine = norepinephrine_baseline;

        std::cout << "[NEUROCHEMISTRY] Loaded baselines: "
                  << "D=" << dopamine_baseline << " "
                  << "S=" << serotonin_baseline << " "
                  << "N=" << norepinephrine_baseline << std::endl;
    }

    void update(const TorusManifold& torus, double dt) {
        // Update base systems (dopamine uses its internal baseline)
        dopamine.update(...);
        boredom.update(torus, dopamine.get_level());

        // Serotonin homeostasis (slow decay to baseline)
        double S_decay = serotonin_decay_rate * (serotonin_baseline - serotonin);
        serotonin += S_decay * dt;
        serotonin = std::clamp(serotonin, 0.0, 1.0);

        // Norepinephrine homeostasis (faster decay)
        double N_decay = norepinephrine_decay_rate * (norepinephrine_baseline - norepinephrine);
        norepinephrine += N_decay * dt;
        norepinephrine = std::clamp(norepinephrine, 0.0, 1.0);
    }

    double get_metric_elasticity() const {
        double lambda_base = 0.01;
        return lambda_base * (0.5 + 0.5 * std::tanh(serotonin - 0.5));
    }

    double get_global_refractive_index() const {
        return 1.0 / (1.0 + norepinephrine);
    }

    void on_nap_complete() {
        serotonin += 0.2;
        serotonin = std::clamp(serotonin, 0.0, 1.0);
    }

    void on_security_alert() {
        norepinephrine = 1.0;  // Immediate spike
        serotonin -= 0.5;      // Emergency plasticity
    }
};
```

---

**Cross-References:**
- See Section 3.4 for Neuroplasticity mathematics
- See Section 15 for Training Systems that use these signals
- See Section 22 for Nap System integration
- See Section 17 for Self-Improvement triggers
