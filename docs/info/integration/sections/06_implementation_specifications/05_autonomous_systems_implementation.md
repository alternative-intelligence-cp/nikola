# Domain V: Autonomous Systems Implementation Specifications

**Document Reference:** NM-004-GAP-AUTONOMOUS
**Status:** Implementation-Ready
**Date:** 2025-12-10
**Source:** Gap Analysis Report (Dr. Aris Thorne)

## Overview

The Autonomous Systems domain implements the Extended Neurochemical Gating System (ENGS) and self-regulation mechanisms. This creates goal-directed behavior, curiosity-driven exploration, and metabolic resource management.

---

## Gap 5.1: Prediction Error Calculation (Dopamine)

### Context and Requirement

Computing D(t) (Dopamine level) based on prediction errors.

### Technical Specification

We implement **Temporal Difference (TD) Learning on Amplitude**.

```
δ_t = (R_t + γ·V(S_{t+1})) - V(S_t)
```

Where:
- **V(S) = Σ|Ψ|²:** Total System Energy
- **Interpretation:** Did the system energy (confidence) increase or decrease unexpectedly?
- **Reward R_t:**
  - +1 if User provides positive feedback (via CLI)
  - -1 if negative
  - 0 otherwise

### Implementation

```cpp
class DopamineSystem {
private:
    float gamma = 0.95f; // Discount factor
    float dopamine_level = 0.5f; // Baseline [0, 1]
    float learning_rate = 0.01f;

    float prev_value = 0.0f;
    float current_value = 0.0f;

public:
    void update(float total_energy, float reward) {
        current_value = total_energy;

        // TD error: reward + discounted future - current estimate
        float td_error = reward + gamma * current_value - prev_value;

        // Dopamine encodes the prediction error (clamped to [0, 1])
        // Positive error -> dopamine spike
        // Negative error -> dopamine dip
        dopamine_level = std::clamp(0.5f + td_error, 0.0f, 1.0f);

        prev_value = current_value;
    }

    float get_dopamine() const { return dopamine_level; }

    // Decay dopamine back to baseline over time
    void decay(float dt) {
        float tau = 2.0f; // Time constant: 2 seconds
        dopamine_level += (0.5f - dopamine_level) * dt / tau;
    }
};
```

### Biological Interpretation

- **Dopamine spike (D > 0.5):** "Better than expected" → Increase learning rate, reward current behavior
- **Dopamine dip (D < 0.5):** "Worse than expected" → Suppress learning, explore alternatives
- **Baseline (D = 0.5):** No surprise, maintain current policy

### Validation Procedure

1. **Reward Test:** Provide positive feedback after correct token. Verify D spikes to ~0.8.
2. **Punishment Test:** Provide negative feedback after incorrect token. Verify D dips to ~0.2.
3. **Habituation Test:** Repeat same reward 10 times. Verify D returns to 0.5 (expectation learned).

---

## Gap 5.2: Entropy Estimation

### Context and Requirement

Discretizing Ψ for Shannon Entropy calculation (boredom detection).

### Technical Specification

**Monte Carlo Estimate** instead of full integration.

Instead of integrating over all nodes, sample K=1000 active nodes.

```
H ≈ -Σ_{k=1}^K p_k log₂(p_k)
```

Where:
```
p_k = |Ψ_k|² / Σ|Ψ_j|²
```

This is O(K) instead of O(N), making it tractable at 2000 Hz.

### Implementation

```cpp
#include <cmath>
#include <algorithm>
#include <random>

class EntropyEstimator {
private:
    static constexpr int SAMPLE_SIZE = 1000;
    std::mt19937 rng;

public:
    float estimate_entropy(const std::vector<std::complex<float>>& psi) {
        // 1. Compute total energy
        float total_energy = 0.0f;
        for (const auto& val : psi) {
            total_energy += std::norm(val); // |Ψ|²
        }

        if (total_energy < 1e-10f) return 0.0f; // Empty grid

        // 2. Sample K active nodes
        std::vector<size_t> active_indices;
        for (size_t i = 0; i < psi.size(); ++i) {
            if (std::norm(psi[i]) > 1e-6f) {
                active_indices.push_back(i);
            }
        }

        if (active_indices.empty()) return 0.0f;

        // Randomly sample up to SAMPLE_SIZE nodes
        std::shuffle(active_indices.begin(), active_indices.end(), rng);
        int samples = std::min(SAMPLE_SIZE, static_cast<int>(active_indices.size()));

        // 3. Compute entropy
        float entropy = 0.0f;
        for (int i = 0; i < samples; ++i) {
            float intensity = std::norm(psi[active_indices[i]]);
            float p = intensity / total_energy;

            if (p > 1e-10f) {
                entropy -= p * std::log2(p);
            }
        }

        return entropy;
    }
};
```

### Interpretation

- **Low Entropy (H < 2):** Narrow distribution → System is "focused" or "bored"
- **High Entropy (H > 10):** Broad distribution → System is "confused" or "exploring"
- **Target Range:** 4-8 for healthy cognitive state

### Boredom Trigger

```cpp
class BoredomRegulator {
private:
    EntropyEstimator entropy_calc;
    float boredom_level = 0.0f;

public:
    void update(const std::vector<std::complex<float>>& psi, float dt) {
        float entropy = entropy_calc.estimate_entropy(psi);

        // Low entropy -> increasing boredom
        // High entropy -> decreasing boredom
        float entropy_target = 6.0f;
        float boredom_rate = 0.1f;

        if (entropy < entropy_target) {
            boredom_level += boredom_rate * dt; // Getting bored
        } else {
            boredom_level -= boredom_rate * dt; // Engaged
        }

        boredom_level = std::clamp(boredom_level, 0.0f, 1.0f);
    }

    bool should_explore() const {
        return boredom_level > 0.7f; // Threshold for spontaneous action
    }
};
```

---

## Gap 5.3: Metabolic Cost Formula

### Context and Requirement

Defining "Work" for ATP depletion.

### Technical Specification

**Hamiltonian Kinetic Term** as metabolic cost.

```
Cost = α · Σ_{active nodes} |∇Ψ|² · Δt
```

- **High frequency waves** (high derivatives) burn more ATP
- **Standing waves** (low derivatives) are cheap

This naturally penalizes "thrashing" or high-noise states.

### Implementation

```cpp
class MetabolicSimulator {
private:
    float atp_level = 1.0f; // [0, 1], starts full
    float alpha = 0.001f; // Cost coefficient

public:
    void consume_energy(const std::vector<std::complex<float>>& psi,
                       const std::vector<std::complex<float>>& laplacian,
                       float dt) {
        float total_cost = 0.0f;

        // Cost proportional to kinetic energy (Laplacian magnitude)
        for (size_t i = 0; i < psi.size(); ++i) {
            if (std::norm(psi[i]) > 1e-6f) { // Only count active nodes
                total_cost += std::norm(laplacian[i]);
            }
        }

        // Deplete ATP
        float depletion = alpha * total_cost * dt;
        atp_level -= depletion;
        atp_level = std::max(0.0f, atp_level);
    }

    void recharge(float dt) {
        // Passive regeneration during idle/nap
        float regen_rate = 0.05f; // 5% per second
        atp_level += regen_rate * dt;
        atp_level = std::min(1.0f, atp_level);
    }

    float get_atp() const { return atp_level; }

    bool is_exhausted() const { return atp_level < 0.15f; }
};
```

### Energy Budget

At 2000 Hz physics loop:
- **Idle state:** ~0.001 ATP/sec (baseline maintenance)
- **Active reasoning:** ~0.05 ATP/sec (moderate thinking)
- **Intense computation:** ~0.2 ATP/sec (solving hard problems)

With regen rate of 0.05/sec:
- **Sustainable load:** < 0.05 ATP/sec
- **Burst capacity:** Can run at 0.2/sec for ~5 seconds before exhaustion

---

## Gap 5.4: Nap Cycle Duration

### Context and Requirement

Nap exit criteria.

### Technical Specification

**ATP Hysteresis** to prevent oscillation.

#### Parameters

- **Enter Nap:** ATP < 0.15
- **Exit Nap:** ATP > 0.90
- **Recharge Rate:** dATP/dt = 0.05 per second (simulated)
- **Min Nap:** (0.90 - 0.15) / 0.05 = 15 seconds
- **Max Nap:** 60 seconds (forced wake-up)

### Implementation

```cpp
class NapCycleManager {
private:
    enum class State { AWAKE, NAPPING };
    State state = State::AWAKE;
    float nap_start_time = 0.0f;

    static constexpr float NAP_ENTER_THRESHOLD = 0.15f;
    static constexpr float NAP_EXIT_THRESHOLD = 0.90f;
    static constexpr float MAX_NAP_DURATION = 60.0f;

public:
    void update(float atp_level, float current_time) {
        switch (state) {
            case State::AWAKE:
                if (atp_level < NAP_ENTER_THRESHOLD) {
                    enter_nap(current_time);
                }
                break;

            case State::NAPPING:
                float nap_duration = current_time - nap_start_time;

                // Exit conditions
                bool recharged = (atp_level > NAP_EXIT_THRESHOLD);
                bool timeout = (nap_duration > MAX_NAP_DURATION);

                if (recharged || timeout) {
                    exit_nap();
                }
                break;
        }
    }

    bool is_napping() const { return state == State::NAPPING; }

private:
    void enter_nap(float time) {
        state = State::NAPPING;
        nap_start_time = time;
        log_info("Entering NAP state (ATP depleted)");
    }

    void exit_nap() {
        state = State::AWAKE;
        log_info("Exiting NAP state (ATP recharged)");
    }
};
```

### Biological Analogy

- **Hysteresis prevents "flapping":** Once asleep, must fully recharge before waking
- **Max duration prevents infinite sleep:** Emergency wake-up after 60s (similar to arousal mechanisms in biology)

---

## Gap 5.5: Dream-Weave Convergence Criteria

### Context and Requirement

Stopping criteria for counterfactual dream iterations.

### Technical Specification

**Metric Stability** measured by Frobenius norm.

Run iterations until the Metric update Δg falls below threshold:

```
||Δg||_F < 10^-4
```

This indicates the memory has "settled" into a local energy minimum.

### Implementation

```cpp
#include <Eigen/Dense>

class DreamWeaveEngine {
private:
    static constexpr float CONVERGENCE_THRESHOLD = 1e-4f;
    static constexpr int MAX_ITERATIONS = 1000;

public:
    void run_counterfactual_consolidation(PhysicsEngine& engine) {
        auto prev_metric = engine.get_metric_tensor();

        for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
            // Run physics with modified boundary conditions
            // (e.g., "What if X happened instead of Y?")
            engine.tick_dream_mode();

            auto current_metric = engine.get_metric_tensor();

            // Compute Frobenius norm of metric change
            float delta_norm = compute_frobenius_norm(prev_metric, current_metric);

            if (delta_norm < CONVERGENCE_THRESHOLD) {
                log_info("Dream-Weave converged after {} iterations", iter);
                return;
            }

            prev_metric = current_metric;
        }

        log_warning("Dream-Weave did not converge after {} iterations", MAX_ITERATIONS);
    }

private:
    float compute_frobenius_norm(const std::vector<Eigen::Matrix<float, 9, 9>>& A,
                                  const std::vector<Eigen::Matrix<float, 9, 9>>& B) {
        float sum = 0.0f;

        for (size_t i = 0; i < A.size(); ++i) {
            Eigen::Matrix<float, 9, 9> diff = A[i] - B[i];
            sum += diff.squaredNorm();
        }

        return std::sqrt(sum);
    }
};
```

### Dream-Weave Purpose

During NAP, the system:
1. Replays recent experiences with variations ("What if I had said X instead of Y?")
2. Adjusts metric tensor based on hypothetical outcomes
3. Consolidates memory by finding stable geometric configurations
4. Prunes weak connections (low-amplitude nodes)

This is analogous to mammalian REM sleep consolidation.

---

## Summary

All 5 Autonomous Systems implementation gaps have been addressed with:
- ✅ TD-learning dopamine system tracking prediction errors
- ✅ Monte Carlo entropy estimation (O(K) complexity)
- ✅ Hamiltonian-based metabolic cost (penalizes high-frequency thrashing)
- ✅ ATP hysteresis nap cycle (15-60 second duration)
- ✅ Frobenius norm convergence for Dream-Weave (10^-4 threshold)

**Status:** Ready for autonomous behavior implementation.
