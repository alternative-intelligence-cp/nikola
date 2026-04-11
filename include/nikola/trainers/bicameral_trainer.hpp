/**
 * @file include/nikola/trainers/bicameral_trainer.hpp
 * @brief BicameralTrainer — Joint coordination of Mamba + Transformer + EqProp
 *
 * Third and final training component (v0.1.15). Coordinates all three
 * Bicameral Autonomous Trainers (BATs) with phase scheduling:
 *
 *   1. MambaTrainer      — SSM dynamics (A, B, C)        [parameter-level]
 *   2. TransformerTrainer — Attention projections (Q,K,V) [parameter-level]
 *   3. EqPropTrainer     — Geometry (metric tensor g_ij)  [geometry-level]
 *
 * Phase scheduling strategies:
 *   ALL          — Run all three every epoch
 *   PARAMS_ONLY  — Only parameter-level trainers (Mamba + Transformer)
 *   GEOMETRY_ONLY — Only geometry trainer (EqProp)
 *   ALTERNATING  — Alternate params / geometry each epoch
 *
 * The key insight: parameter-level and geometry-level training are
 * complementary. Parameters tune the models; geometry tunes the space
 * they operate in. Running both together → better convergence.
 *
 * @see include/nikola/trainers/mamba_trainer.hpp
 * @see include/nikola/trainers/transformer_trainer.hpp
 * @see include/nikola/cognitive/plasticity.hpp
 */
#pragma once

#include <nikola/trainers/mamba_trainer.hpp>
#include <nikola/trainers/transformer_trainer.hpp>
#include <nikola/cognitive/plasticity.hpp>

#include <vector>
#include <cmath>

namespace nikola::trainers {

/// Phase scheduling policy.
enum class TrainingPhase {
    ALL,            ///< All three trainers every epoch
    PARAMS_ONLY,    ///< Mamba + Transformer only
    GEOMETRY_ONLY,  ///< EqProp only
    ALTERNATING     ///< Alternate params/geometry each epoch
};

/// Combined statistics from one joint training epoch.
struct JointTrainingStats {
    double mamba_loss         = 0.0;  ///< MambaTrainer batch loss (0 if skipped)
    double transformer_loss   = 0.0;  ///< TransformerTrainer batch loss (0 if skipped)
    double eqprop_energy_diff = 0.0;  ///< E⁺ - E⁻ from EqProp (0 if skipped)
    bool   eqprop_converged   = false;///< E⁻ < E⁺ in last EqProp step
    bool   ran_mamba          = false;
    bool   ran_transformer    = false;
    bool   ran_eqprop         = false;
    int    epoch              = 0;
};

/**
 * @brief BicameralTrainer — coordinates all three training subsystems.
 *
 * Owns MambaTrainer and TransformerTrainer directly. Takes a reference
 * to the PlasticityEngine (which owns EqPropTrainer and TopologyManager).
 *
 * MambaTrainer and TransformerTrainer operate on abstract 9D parameter
 * spaces via StaticComputeGraph (zero-allocation autodiff).
 *
 * EqPropTrainer operates on actual WaveFunctions through physics
 * simulation, updating the metric tensor g_ij in TopologyManager.
 *
 * The trainers are independent — parameter training doesn't read the
 * metric, and geometry training doesn't read SSM/attention parameters.
 * But they're complementary: better parameters → better inference →
 * better wavefunctions → better EqProp signal → better geometry.
 */
class BicameralTrainer {
public:
    /**
     * @brief Construct a BicameralTrainer.
     *
     * @param plasticity  PlasticityEngine managing EqProp + topology.
     * @param mamba_lr    MambaTrainer learning rate.
     * @param trans_lr    TransformerTrainer learning rate.
     * @param phase       Initial scheduling policy.
     */
    explicit BicameralTrainer(cognitive::PlasticityEngine& plasticity,
                              double mamba_lr  = 0.001,
                              double trans_lr  = 0.0001,
                              TrainingPhase phase = TrainingPhase::ALL)
        : plasticity_(plasticity)
        , mamba_(mamba_lr)
        , transformer_(trans_lr)
        , phase_(phase)
    {}

    // ── Joint training interface ───────────────────────────────────────

    /**
     * @brief Run one joint training epoch.
     *
     * Depending on the scheduling phase, runs some or all three trainers.
     * Parameter-level trainers take batch data; EqProp takes a wavefunction
     * and injection callbacks.
     *
     * @param mamba_data     SSM training samples (state, input, next_state).
     * @param attn_data      Attention training samples (x1, x2, y1, y2).
     * @param wf             WaveFunction for EqProp (modified in-place).
     * @param inject_input   EqProp input injection callback.
     * @param inject_target  EqProp target injection callback.
     * @return               Combined statistics from all trainers that ran.
     */
    JointTrainingStats train_joint_step(
        const std::vector<TrainingSample>& mamba_data,
        const std::vector<AttentionSample>& attn_data,
        physics::WaveFunction& wf,
        cognitive::EqPropTrainer::InjectFn inject_input,
        cognitive::EqPropTrainer::InjectFn inject_target)
    {
        JointTrainingStats stats;
        stats.epoch = epoch_;

        bool do_params   = should_run_params();
        bool do_geometry  = should_run_geometry();

        // ── Parameter-level training ──────────────────────────────
        if (do_params && !mamba_data.empty()) {
            auto ms = mamba_.train_batch(mamba_data);
            stats.mamba_loss = ms.loss;
            stats.ran_mamba  = true;
        }

        if (do_params && !attn_data.empty()) {
            auto ts = transformer_.train_batch(attn_data);
            stats.transformer_loss = ts.loss;
            stats.ran_transformer  = true;
        }

        // ── Geometry-level training ───────────────────────────────
        if (do_geometry && inject_input && inject_target) {
            bool converged = plasticity_.eqprop().train_step(
                wf, inject_input, inject_target);
            stats.eqprop_energy_diff = plasticity_.eqprop().last_energy_diff();
            stats.eqprop_converged   = converged;
            stats.ran_eqprop         = true;
        }

        ++epoch_;
        return stats;
    }

    /**
     * @brief Run parameter-level training only (no EqProp).
     */
    JointTrainingStats train_params_only(
        const std::vector<TrainingSample>& mamba_data,
        const std::vector<AttentionSample>& attn_data)
    {
        JointTrainingStats stats;
        stats.epoch = epoch_;

        if (!mamba_data.empty()) {
            auto ms = mamba_.train_batch(mamba_data);
            stats.mamba_loss = ms.loss;
            stats.ran_mamba  = true;
        }
        if (!attn_data.empty()) {
            auto ts = transformer_.train_batch(attn_data);
            stats.transformer_loss = ts.loss;
            stats.ran_transformer  = true;
        }

        ++epoch_;
        return stats;
    }

    /**
     * @brief Run EqProp geometry training only (no parameter updates).
     */
    JointTrainingStats train_geometry_only(
        physics::WaveFunction& wf,
        cognitive::EqPropTrainer::InjectFn inject_input,
        cognitive::EqPropTrainer::InjectFn inject_target)
    {
        JointTrainingStats stats;
        stats.epoch = epoch_;

        if (inject_input && inject_target) {
            bool converged = plasticity_.eqprop().train_step(
                wf, inject_input, inject_target);
            stats.eqprop_energy_diff = plasticity_.eqprop().last_energy_diff();
            stats.eqprop_converged   = converged;
            stats.ran_eqprop         = true;
        }

        ++epoch_;
        return stats;
    }

    // ── Sub-trainer access ─────────────────────────────────────────────

    MambaTrainer&       mamba()       { return mamba_; }
    const MambaTrainer& mamba() const { return mamba_; }

    TransformerTrainer&       transformer()       { return transformer_; }
    const TransformerTrainer& transformer() const { return transformer_; }

    cognitive::EqPropTrainer&       eqprop()       { return plasticity_.eqprop(); }
    const cognitive::EqPropTrainer& eqprop() const { return plasticity_.eqprop(); }

    cognitive::PlasticityEngine&       plasticity()       { return plasticity_; }
    const cognitive::PlasticityEngine& plasticity() const { return plasticity_; }

    // ── Scheduling ─────────────────────────────────────────────────────

    TrainingPhase phase() const { return phase_; }
    void set_phase(TrainingPhase p) { phase_ = p; }

    int epoch() const { return epoch_; }

    /// Check if metric is still positive-definite.
    bool metric_valid() const {
        float g[81];
        std::copy(plasticity_.topology().metric(),
                  plasticity_.topology().metric() + 81, g);
        return spatial::MetricValidator::gerschgorin_check(g);
    }

private:
    cognitive::PlasticityEngine& plasticity_;
    MambaTrainer                 mamba_;
    TransformerTrainer           transformer_;
    TrainingPhase                phase_;
    int                          epoch_ = 0;

    bool should_run_params() const {
        switch (phase_) {
            case TrainingPhase::ALL:          return true;
            case TrainingPhase::PARAMS_ONLY:  return true;
            case TrainingPhase::GEOMETRY_ONLY: return false;
            case TrainingPhase::ALTERNATING:  return (epoch_ % 2 == 0);
        }
        return true;
    }

    bool should_run_geometry() const {
        switch (phase_) {
            case TrainingPhase::ALL:          return true;
            case TrainingPhase::PARAMS_ONLY:  return false;
            case TrainingPhase::GEOMETRY_ONLY: return true;
            case TrainingPhase::ALTERNATING:  return (epoch_ % 2 == 1);
        }
        return true;
    }
};

} // namespace nikola::trainers
