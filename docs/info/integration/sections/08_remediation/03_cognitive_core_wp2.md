# WORK PACKAGE 2: COGNITIVE CORE

## WP2.1 Overview

**Purpose:** Remediate cognitive subsystem defects and enhance autonomous learning.

**Status:** 1 CRITICAL FIXED, 1 MEDIUM specified

## WP2.2 Defect: AUTO-ENGS-01 - Zeno's Decay Bug

**Defect ID:** AUTO-ENGS-01
**Severity:** CRITICAL
**Status:** ✓ FIXED

### Impact

Emotional states (dopamine, serotonin, norepinephrine) persisted indefinitely; system could not achieve homeostasis, leading to runaway arousal or permanent depression states.

### Root Cause

Decay used incorrect formula: $C_{new} = C_{prev} - k \cdot dt$, which is frame-rate dependent and violates physics (Zeno's paradox analog).

### Resolution

Changed to exponential decay: $C(t) = C_{base} + (C_{prev} - C_{base}) \cdot e^{-k \cdot dt}$

Ensures time-step independence and proper return-to-baseline dynamics.

**Implementation:**

```cpp
// File: src/autonomy/engs.cpp

void ExtendedNeurochemistry::update(double dt) {
    const double DOPAMINE_DECAY_RATE = 0.05;
    const double SEROTONIN_DECAY_RATE = 0.01;
    const double NOREPINEPHRINE_DECAY_RATE = 0.10;

    // Exponential decay to baseline (time-step independent)
    dopamine = dopamine_baseline +
               (dopamine - dopamine_baseline) * std::exp(-DOPAMINE_DECAY_RATE * dt);

    serotonin = serotonin_baseline +
                (serotonin - serotonin_baseline) * std::exp(-SEROTONIN_DECAY_RATE * dt);

    norepinephrine = norepinephrine_baseline +
                     (norepinephrine - norepinephrine_baseline) * std::exp(-NOREPINEPHRINE_DECAY_RATE * dt);

    // Clamp to valid ranges
    dopamine = std::clamp(dopamine, 0.0, 1.0);
    serotonin = std::clamp(serotonin, 0.0, 1.0);
    norepinephrine = std::clamp(norepinephrine, 0.0, 1.0);
}
```

### Verification

- ✓ Decay reaches 99% baseline in ~5/k seconds regardless of dt
- ✓ Time-step independence verified
- ✓ Homeostasis achieved

### Location

Section 14.6.2 - [src/autonomy/engs.cpp](src/autonomy/engs.cpp) (Line 45-55)

## WP2.3 Defect: AUTO-DREAM-01 - Metric Tensor Unit Confusion

**Defect ID:** AUTO-DREAM-01
**Severity:** MEDIUM
**Status:** ✓ FIXED

### Impact

System biased towards hallucination over factual learning; compared dimensionally incompatible quantities (resonance [unitless] vs reward [arbitrary]).

### Root Cause

Hindsight learning condition: `if (dream_resonance > recorded_reward)` compared apples to oranges.

### Resolution

Implemented normalized comparison via z-scores using Welford's online algorithm for running statistics.

**Implementation:**

```cpp
// File: src/autonomy/dream_weave.cpp

class DreamWeaveEngine {
private:
    // Track historical distributions
    RunningStats resonance_stats;
    RunningStats reward_stats;

public:
    bool should_reinforce_counterfactual(double cf_resonance, double actual_reward) {
        // Convert to z-scores (standardized units)
        double z_resonance = resonance_stats.z_score(cf_resonance);
        double z_reward = reward_stats.z_score(actual_reward);

        // Compare in standardized space
        return (z_resonance > z_reward + 0.5);  // 0.5 sigma threshold
    }

    void update_stats(double resonance, double reward) {
        resonance_stats.add_sample(resonance);
        reward_stats.add_sample(reward);
    }
};

class RunningStats {
    double mean = 0.0;
    double m2 = 0.0;  // Sum of squared differences (Welford's algorithm)
    size_t count = 0;

public:
    // Add sample using Welford's online algorithm (numerically stable)
    void add_sample(double x) {
        count++;
        double delta = x - mean;
        mean += delta / count;
        double delta2 = x - mean;
        m2 += delta * delta2;
    }

    // Compute z-score (standardized value)
    double z_score(double x) const {
        if (count < 2) return 0.0;  // Not enough data
        double variance = m2 / (count - 1);  // Sample variance
        double stddev = std::sqrt(variance);
        if (stddev < 1e-10) return 0.0;  // Avoid division by zero
        return (x - mean) / stddev;
    }

    // Get current statistics
    double get_mean() const { return mean; }
    double get_stddev() const {
        if (count < 2) return 0.0;
        return std::sqrt(m2 / (count - 1));
    }
    size_t get_count() const { return count; }
};
```

### Integration with DreamWeaveEngine

The corrected comparison is integrated into the dream cycle:

```cpp
// File: src/autonomy/dream_weave.cpp (Updated)

void DreamWeaveEngine::run_dream_cycle(TorusManifold& torus,
                                       Mamba9D& mamba,
                                       int num_simulations) {
    // ... (existing code)

    for (const auto& record : high_loss_records) {
        for (int i = 0; i < NUM_COUNTERFACTUALS; ++i) {
            auto counterfactual = generate_counterfactual(record.sequence);
            double cf_resonance = evaluate_outcome(counterfactual, torus, mamba);

            // FIXED: Use z-score comparison instead of raw values
            if (should_reinforce_counterfactual(cf_resonance, record.reward)) {
                std::cout << "[DREAM] Reinforcing counterfactual (z-score improved)" << std::endl;
                torus.trigger_neuroplasticity_update_from_sequence(counterfactual);
            }

            // Update running statistics
            update_stats(cf_resonance, record.reward);
        }
    }
}
```

### Verification

- ✓ Z-score normalization eliminates unit confusion
- ✓ Welford's algorithm provides numerical stability
- ✓ Counterfactual learning no longer biased towards hallucination
- ✓ Both resonance and reward compared in standardized space

### Location

Section 22.5.3 - [src/autonomy/dream_weave.cpp](src/autonomy/dream_weave.cpp) (Line 88-131)

## WP2.4 Enhancement: Topological State Mapping (TSM)

**Status:** SPECIFIED

### Purpose

Establishes direct mathematical isomorphism where State Space Model (SSM) matrices ($A, B, C$) are dynamically derived from local metric tensor of the torus.

### Concept

"Layers ARE the toroid" - Mamba layer is not just reading the torus but is physically instantiated by the torus geometry.

**Mathematical Formulation:**

$$A_{ij} = \frac{1}{\sqrt{g_{ii}}} \cdot \delta_{ij} \cdot (1 - \alpha \cdot r_i)$$

Where:
- $g_{ii}$: Diagonal metric tensor component
- $r_i$: Local resonance value
- $\alpha$: Coupling constant (typically 0.1)

**Implementation:**

```cpp
// File: src/mamba/ssm_kernel.cpp

class Mamba9D {
    void update_ssm_from_torus(const TorusManifold& torus, const Coord9D& center) {
        // Extract local metric tensor
        const TorusNode& node = torus.get_node(center);

        // Build A matrix from metric diagonal
        for (int i = 0; i < 9; ++i) {
            int diag_idx = triangular_index(i, i);
            double g_ii = node.metric_tensor[diag_idx];

            // Topological mapping: geometry → dynamics
            A_matrix[i][i] = (1.0 - alpha * node.resonance_r) / std::sqrt(g_ii);
        }

        // B and C matrices derived from off-diagonal metric
        for (int i = 0; i < 9; ++i) {
            for (int j = i + 1; j < 9; ++j) {
                int idx = triangular_index(i, j);
                double g_ij = node.metric_tensor[idx];

                B_matrix[i][j] = g_ij;
                C_matrix[i][j] = g_ij;
            }
        }
    }
};
```

### Benefits

- **Stronger topological identity:** SSM structure reflects manifold geometry
- **Preserves local neighborhoods:** No information loss from linearization
- **Dynamic adaptation:** SSM parameters evolve with neuroplasticity

### Location

Section 7.3.1 - [src/mamba/ssm_kernel.cpp](src/mamba/ssm_kernel.cpp)

## WP2.5 Enhancement: Wave Correlation Attention

**Status:** IMPLEMENTED

### Purpose

Implement attention mechanism that respects wave physics rather than traditional dot-product attention.

**Mathematical Formulation:**

$$\text{WaveAttn}(Q, K, V) = \text{softmax}\left(\frac{|Q \star K|}{\sqrt{d_k}}\right) V$$

Where $\star$ denotes wave correlation (complex conjugate product).

**Implementation:**

```cpp
// File: src/transformer/wave_attention.cpp

Eigen::MatrixXcd WaveAttentionLayer::forward(
    const Eigen::MatrixXcd& Q,
    const Eigen::MatrixXcd& K,
    const Eigen::MatrixXcd& V) {

    int seq_len = Q.rows();
    int d_k = Q.cols();

    // Wave correlation: Q ⊗ K† (conjugate transpose)
    Eigen::MatrixXcd scores = Q * K.adjoint();

    // Take magnitude for softmax (real-valued)
    Eigen::MatrixXd scores_mag = scores.cwiseAbs();

    // Scale by sqrt(d_k)
    scores_mag /= std::sqrt(static_cast<double>(d_k));

    // Softmax
    Eigen::MatrixXd attn_weights = softmax(scores_mag);

    // Apply to values (keeping complex structure)
    Eigen::MatrixXcd output = attn_weights.cast<std::complex<double>>() * V;

    return output;
}
```

### Location

Section 8.1.1 - [src/transformer/wave_attention.cpp](src/transformer/wave_attention.cpp)

---

**Cross-References:**
- See Section 14 for Extended Neurochemical Gating System
- See Section 7 for Mamba-9D State Space Model
- See Section 8 for Neuroplastic Transformer
- See Section 22 for Dream-Weave Counterfactual Simulation
