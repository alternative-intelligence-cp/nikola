/**
 * @file cognitive/plasticity.hpp
 * @brief Hebbian neuroplasticity and Equilibrium Propagation training.
 *
 * Implements Phase 3 cognitive-architecture gap from the engineering report:
 *
 *   Gap 3.6 — Equilibrium Propagation training (no backprop through physics)
 *
 * Also provides the supporting Hebbian co-activation machinery that couples
 * the ψ-field dynamics of the physics substrate to the Riemannian metric
 * tensor managed by TopologyManager (Phase 2, Gap 2.5).
 *
 * Architecture
 * ------------
 *
 *   CoactivationTracker
 *     Accumulates a running 9×9 outer-product matrix from wavefunction
 *     co-activations: C_ij += Re(Ψ_i* · Ψ_j).  Normalised by node count.
 *
 *   HebbianPlasticity
 *     Applies the CoactivationTracker output as a metric update via
 *     TopologyManager::update_metric() (which calls MetricLearner and
 *     validates positive-definiteness after every update).
 *
 *   EqPropTrainer
 *     Implements the two-phase Equilibrium Propagation loop:
 *       Positive phase: free evolution, capture E⁺ and co-activation C⁺.
 *       Negative phase: target token clamped, capture E⁻ and co-activation C⁻.
 *       Update:         Δg_ij ∝ −η · (E⁺ − E⁻) · (C⁺_ij − C⁻_ij)
 *     The energy difference makes the target the "path of least resistance".
 *
 *   PlasticityEngine
 *     Orchestrator class owning all sub-components.
 *
 * Biological Analogy
 * ------------------
 *   CoactivationTracker  ↔  NMDA receptor coincidence detection.
 *   HebbianPlasticity    ↔  Long-Term Potentiation (LTP) at synaptic weight.
 *   EqPropTrainer        ↔  Predictive coding / free-energy minimisation.
 *   Dopamine             ↔  Neuromodulatory gate on plasticity.
 *
 * Reference:
 *   docs/info/integration/sections/06_implementation_specifications/
 *   03_cognitive_architecture_implementation.md  § Gap 3.6
 *   engineering report §§ 8.9.3 (The Brain), 3.6 (EqProp)
 */
#pragma once

#include <nikola/physics/wave_function.hpp>
#include <nikola/physics/hamiltonian.hpp>
#include <nikola/physics/propagator.hpp>
#include <nikola/spatial/topology_manager.hpp>

#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <functional>
#include <cstring>

namespace nikola::cognitive {

using foundation::TORUS_DIMS;
using spatial::TopologyManager;
using physics::WaveFunction;
using physics::Hamiltonian;
using physics::Propagator;

// ============================================================================
// Constants
// ============================================================================

/// Saturation threshold: if max |Ψ| exceeds this, neurogenesis is triggered.
inline constexpr float PLASTICITY_SAT_THRESHOLD = 5.0f;

/// Default number of physics steps per EqProp phase.
inline constexpr int EQPROP_PHASE_STEPS = 100;

/// Default Equilibrium Propagation learning rate.
inline constexpr float EQPROP_ETA = 0.01f;

// ============================================================================
// CoactivationTracker
// ============================================================================

/**
 * @brief Computes the 9×9 Hebbian co-activation outer product.
 *
 * For a wavefunction on N grid nodes the co-activation matrix is:
 *
 *   C_ij  =  (1/N) · Σ_n  Re(ψ_n,i* · ψ_n,j)   (sum over nodes n)
 *
 * where "i, j" index the 9 torus *dimensions*, not individual nodes.
 * The dominant activation vector  a ∈ ℝ⁹ is computed as the mean absolute
 * amplitude across all nodes per dimension:
 *
 *   a_d  =  (1/N) · Σ_n  |ψ(n)|   (scalar, averaged over same-d nodes)
 *
 * Because node data is not structured per-dimension in the SoA layout,
 * we use the total field intensity ‖Ψ‖ as a proxy and compute a rank-1
 * outer product  a ⊗ a.  This is the biologically faithful formulation:
 * co-activation at the cell assembly level, not individual synapse level.
 *
 * The matrix is accumulated over multiple calls to accumulate() and
 * then normalised by get() or explicitly reset by reset().
 */
class CoactivationTracker {
public:
    CoactivationTracker() noexcept {
        reset();
    }

    /// Zero accumulated co-activation and sample count.
    void reset() noexcept {
        std::fill(accum_, accum_ + 81, 0.f);
        count_ = 0;
    }

    /**
     * @brief Accumulate one co-activation snapshot from a WaveFunction.
     *
     * Computes the mean |Ψ| per dimension bucket, then adds the outer product
     * a ⊗ a to the running accumulator.
     *
     * @param wf   Current wavefunction state.
     */
    void accumulate(const WaveFunction& wf) noexcept {
        const foundation::TorusGrid& g = wf.grid();
        const size_t N = g.num_active_nodes();
        if (N == 0) return;

        const float* pr = g.psi_real();
        const float* pi = g.psi_imag();

        // Compute mean amplitude over all nodes (scalar)
        float mean_amp = 0.f;
        for (size_t i = 0; i < N; ++i)
            mean_amp += std::sqrt(pr[i]*pr[i] + pi[i]*pi[i]);
        mean_amp /= static_cast<float>(N);

        // Rank-1 outer product: a ⊗ a where a = mean_amp * 1 (uniform)
        // For a richer signal we fill per-dimension using the first N/9 nodes
        // mapped to each dimension via round-robin.  Simple and consistent.
        std::array<float, TORUS_DIMS> a{};
        // Distribute nodes across dims: node i → dim i % 9, accumulate |ψ_i|
        std::array<float, TORUS_DIMS> dim_sum{};
        std::array<int,   TORUS_DIMS> dim_cnt{};
        for (size_t i = 0; i < N; ++i) {
            const int d = static_cast<int>(i % TORUS_DIMS);
            dim_sum[d] += std::sqrt(pr[i]*pr[i] + pi[i]*pi[i]);
            dim_cnt[d]++;
        }
        bool all_zero = true;
        for (int d = 0; d < TORUS_DIMS; ++d) {
            a[d] = (dim_cnt[d] > 0)
                   ? dim_sum[d] / static_cast<float>(dim_cnt[d])
                   : mean_amp;
            if (a[d] > 1e-9f) all_zero = false;
        }
        if (all_zero) return;  // nothing to learn from a zero field

        // L2-normalise a[] so the outer product captures activation *shape*
        // rather than magnitude.  Without this, raw amplitudes ≈ 1e-5 yield
        // outer products ≈ 1e-10, making EqProp metric deltas negligible.
        float norm2 = 0.f;
        for (int d = 0; d < TORUS_DIMS; ++d) norm2 += a[d] * a[d];
        const float norm = std::sqrt(norm2);
        if (norm > 1e-12f) {
            const float inv_norm = 1.f / norm;
            for (int d = 0; d < TORUS_DIMS; ++d) a[d] *= inv_norm;
        }

        // Outer product C_ij += a_i * a_j   (now O(1) magnitude)
        for (int i = 0; i < TORUS_DIMS; ++i)
            for (int j = 0; j < TORUS_DIMS; ++j)
                accum_[i * TORUS_DIMS + j] += a[i] * a[j];

        ++count_;
    }

    /**
     * @brief Copy normalised co-activation into the 81-element output array.
     *
     * Normalises by sample count.  If count == 0, fills with zeros.
     *
     * @param out  Output buffer, 81 elements (9×9 row-major).
     */
    void get(float out[81]) const noexcept {
        if (count_ == 0) {
            std::fill(out, out + 81, 0.f);
            return;
        }
        const float inv = 1.f / static_cast<float>(count_);
        for (int i = 0; i < 81; ++i) out[i] = accum_[i] * inv;
    }

    /// Number of accumulation samples taken since last reset().
    [[nodiscard]] int count() const noexcept { return count_; }

    /// Maximum off-diagonal element (proxy for co-activation strength).
    [[nodiscard]] float max_coactivation() const noexcept {
        float mx = 0.f;
        for (int i = 0; i < TORUS_DIMS; ++i)
            for (int j = 0; j < TORUS_DIMS; ++j)
                if (i != j) mx = std::max(mx, std::abs(accum_[i*TORUS_DIMS+j]));
        return mx;
    }

private:
    float accum_[81]{};
    int   count_{0};
};

// ============================================================================
// HebbianPlasticity
// ============================================================================

/**
 * @brief Applies Hebbian co-activation updates to a TopologyManager metric.
 *
 * HebbianPlasticity connects CoactivationTracker (which measures field
 * co-activations) to TopologyManager::update_metric() (which applies the
 * dopamine-modulated Δg update and validates positive-definiteness).
 *
 * Saturation check: if the peak wavefunction amplitude exceeds
 * PLASTICITY_SAT_THRESHOLD, apply_update() returns true (neurogenesis signal).
 */
class HebbianPlasticity {
public:
    explicit HebbianPlasticity(TopologyManager& topo) noexcept : topo_(topo) {}

    /**
     * @brief Apply accumulated co-activation as a Hebbian metric update.
     *
     *   Δg_ij = η(D, age) × C_ij   (C from CoactivationTracker)
     *
     * Calls TopologyManager::update_metric() which internally calls
     * MetricLearner and repairs positive-definiteness if needed.
     *
     * @param tracker       Co-activation tracker (must have ≥1 sample).
     * @param dopamine      Reward signal in [0, 1].
     * @param age_seconds   Node age in seconds (for learning rate annealing).
     * @param wf            Current wavefunction (for saturation check).
     * @return              true if wavefunction saturation detected (neurogenesis
     *                      hint); false if normal update occurred.
     */
    bool apply_update(const CoactivationTracker& tracker,
                      float                       dopamine,
                      float                       age_seconds,
                      const WaveFunction&         wf) noexcept
    {
        float corr[81];
        tracker.get(corr);

        topo_.update_metric(corr, dopamine, age_seconds);

        // Saturation check: scan ψ amplitudes for extremes
        const foundation::TorusGrid& g = wf.grid();
        const size_t N = g.num_active_nodes();
        if (N == 0) return false;
        const float* pr = g.psi_real();
        const float* pi = g.psi_imag();
        for (size_t i = 0; i < N; ++i) {
            const float amp = std::sqrt(pr[i]*pr[i] + pi[i]*pi[i]);
            if (amp > PLASTICITY_SAT_THRESHOLD) return true;  // neurogenesis signal
        }
        return false;
    }

    /// Direct metric query: is the current metric positive-definite?
    [[nodiscard]]
    bool metric_is_valid() noexcept {
        // Gerschgorin check via a copy: non-destructive
        float g[81];
        std::copy(topo_.metric(), topo_.metric() + 81, g);
        return spatial::MetricValidator::gerschgorin_check(g);
    }

    TopologyManager& topology() noexcept { return topo_; }

private:
    TopologyManager& topo_;
};

// ============================================================================
// EqPropTrainer
// ============================================================================

/// Configuration for EqPropTrainer.
struct EqPropConfig {
    int   phase_steps = EQPROP_PHASE_STEPS;  ///< Physics steps per phase
    float eta         = EQPROP_ETA;           ///< Learning rate
    float dt          = 0.01f;                ///< Integration time step
    float dopamine    = 1.f;                  ///< Neuromodulator level
    float age_seconds = 0.f;                  ///< Node age (for MetricLearner)
};

/**
 * @brief Equilibrium Propagation trainer (Gap 3.6).
 *
 * Two-phase training procedure:
 *
 *   Positive phase (free):
 *     - Run physics for EQPROP_PHASE_STEPS with input tokens injected.
 *     - Record total energy E⁺ and co-activation matrix C⁺.
 *
 *   Negative phase (clamped):
 *     - Reset wavefunction, re-inject input tokens.
 *     - Inject target token (clamp to correct output).
 *     - Run physics again for EQPROP_PHASE_STEPS.
 *     - Record E⁻ and C⁻.
 *
 *   Metric update:
 *     Δg_ij = −η · (E⁺ − E⁻) · (C⁺_ij − C⁻_ij)
 *
 * Convergence criterion:
 *     E⁻ < E⁺  (physics prefers the clamped/correct state)
 *
 * @note The WaveFunction is reset to vacuum state at the start of each phase
 *       using  wf.grid().fill_noise() equivalent.  The caller provides
 *       inject_fn and reset_fn callbacks to avoid coupling this class to a
 *       specific token-injection strategy.
 */
class EqPropTrainer {
public:
    /// Callback to inject one or more tokens into the wavefunction.
    using InjectFn = std::function<void(WaveFunction&)>;
    using Config   = EqPropConfig;

    explicit EqPropTrainer(TopologyManager& topo,
                           const Config&    cfg = Config{}) noexcept
        : topo_(topo), cfg_(cfg) {}

    /**
     * @brief Execute one EqProp training step.
     *
     * @param wf            Wavefunction (will be modified in-place; caller
     *                      should re-seed or reset after the call if needed).
     * @param inject_input  Callback: injects input tokens into wf.
     * @param inject_target Callback: injects target token (clamped phase).
     * @return              true if training decreased energy (E⁻ < E⁺).
     *
     * Algorithm:
     *   1. Phase+: reset wf, inject_input(wf), evolve, capture E⁺, C⁺.
     *   2. Phase−: reset wf, inject_input(wf), inject_target(wf), evolve, E⁻, C⁻.
     *   3. Δg = −η · (E⁺−E⁻) · (C⁺ − C⁻); apply via TopologyManager.
     */
    bool train_step(WaveFunction& wf,
                    InjectFn      inject_input,
                    InjectFn      inject_target)
    {
        Hamiltonian ham;
        ham.set_c0(1.f).set_beta(1.f);

        Propagator  prop;
        const float dt = cfg_.dt;

        // ------------------------------------------------------------------ Phase +
        zero_wf(wf);
        inject_input(wf);
        tracker_pos_.reset();
        for (int s = 0; s < cfg_.phase_steps; ++s) {
            prop.step(wf, dt);
            tracker_pos_.accumulate(wf);
        }
        const double E_pos = ham.compute(wf);

        // ------------------------------------------------------------------ Phase −
        zero_wf(wf);
        inject_input(wf);
        inject_target(wf);
        tracker_neg_.reset();
        for (int s = 0; s < cfg_.phase_steps; ++s) {
            prop.step(wf, dt);
            tracker_neg_.accumulate(wf);
        }
        const double E_neg = ham.compute(wf);

        // ------------------------------------------------------------------ Metric update
        float C_pos[81], C_neg[81], delta_C[81];
        tracker_pos_.get(C_pos);
        tracker_neg_.get(C_neg);

        const float energy_diff = static_cast<float>(E_pos - E_neg);
        for (int i = 0; i < 81; ++i)
            delta_C[i] = -cfg_.eta * energy_diff * (C_pos[i] - C_neg[i]);

        // Apply via the MetricLearner (bypasses η calculation — we already
        // folded in η — so we pass dopamine and age that yield factor ≈ 1)
        // We apply the precomputed delta directly:
        apply_delta_metric(delta_C);

        energy_pos_ = E_pos;
        energy_neg_ = E_neg;

        return E_neg < E_pos;
    }

    /// Energy recorded during the last positive phase.
    [[nodiscard]] double last_energy_positive() const noexcept { return energy_pos_; }
    /// Energy recorded during the last negative phase.
    [[nodiscard]] double last_energy_negative() const noexcept { return energy_neg_; }

    /// Energy difference E⁺ − E⁻ from last train_step (positive = converging).
    [[nodiscard]] double last_energy_diff() const noexcept {
        return energy_pos_ - energy_neg_;
    }

    const EqPropConfig& config() const noexcept { return cfg_; }
    EqPropConfig&       config() noexcept       { return cfg_; }

private:
    TopologyManager&   topo_;
    EqPropConfig       cfg_;
    CoactivationTracker tracker_pos_;
    CoactivationTracker tracker_neg_;
    double             energy_pos_{0.0};
    double             energy_neg_{0.0};

    /// Zero all ψ and velocity fields in the wavefunction.
    static void zero_wf(WaveFunction& wf) noexcept {
        foundation::TorusGrid& g = wf.grid();
        const size_t N = g.num_active_nodes();
        if (N == 0) return;
        float* pr = g.psi_real();
        float* pi = g.psi_imag();
        float* vr = g.vel_real();
        float* vi = g.vel_imag();
        std::fill(pr, pr + N, 0.f);
        std::fill(pi, pi + N, 0.f);
        std::fill(vr, vr + N, 0.f);
        std::fill(vi, vi + N, 0.f);
    }

    /// Apply a precomputed 81-element Δg directly to the topology metric.
    void apply_delta_metric(const float delta[81]) noexcept {
        // Read current metric, add delta, write back + validate
        float g[81];
        std::copy(topo_.metric(), topo_.metric() + 81, g);
        for (int i = 0; i < 81; ++i) g[i] += delta[i];
        topo_.set_metric(g);
        topo_.validate_metric();   // ensure positive-definiteness
    }
};

// ============================================================================
// PlasticityEngine — orchestrator
// ============================================================================

/**
 * @brief Top-level neuroplasticity controller for a Nikola agent.
 *
 * Owns a HebbianPlasticity and an EqPropTrainer, both operating on the same
 * TopologyManager.
 *
 * Typical usage:
 * @code
 *   TopologyManager topo;
 *   PlasticityEngine plasticity(topo);
 *
 *   // After each physics step: accumulate co-activations
 *   plasticity.tracker().accumulate(wf);
 *
 *   // At end of timestep: Hebbian update with dopamine signal
 *   bool saturation = plasticity.hebbian().apply_update(
 *       plasticity.tracker(), dopamine=0.8f, age=100.f, wf);
 *
 *   // For training: EqProp step
 *   bool converged = plasticity.eqprop().train_step(wf, inject_input, inject_target);
 * @endcode
 */
class PlasticityEngine {
public:
    explicit PlasticityEngine(TopologyManager& topo,
                              EqPropConfig eqprop_cfg = EqPropConfig{})
        : topo_(topo)
        , hebbian_(topo)
        , eqprop_(topo, eqprop_cfg)
    {}

          CoactivationTracker& tracker()  noexcept { return tracker_; }
    const CoactivationTracker& tracker()  const noexcept { return tracker_; }

          HebbianPlasticity& hebbian()    noexcept { return hebbian_; }
    const HebbianPlasticity& hebbian()    const noexcept { return hebbian_; }

          EqPropTrainer&    eqprop()     noexcept { return eqprop_; }
    const EqPropTrainer&    eqprop()     const noexcept { return eqprop_; }

          TopologyManager&  topology()   noexcept { return topo_; }
    const TopologyManager&  topology()   const noexcept { return topo_; }

    /// Reset co-activation accumulator (call at start of each learning window).
    void reset_tracker() noexcept { tracker_.reset(); }

private:
    TopologyManager&    topo_;
    CoactivationTracker tracker_;
    HebbianPlasticity   hebbian_;
    EqPropTrainer       eqprop_;
};

} // namespace nikola::cognitive
