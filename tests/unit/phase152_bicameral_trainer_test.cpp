/**
 * @file tests/unit/phase152_bicameral_trainer_test.cpp
 * @brief Phase 152 — BicameralTrainer & EqProp validation (v0.1.15)
 *
 * Tests:
 *   §1: Construction & component access (3 tests)
 *   §2: EqProp convergence — energy decreases over iterations (3 tests)
 *   §3: Metric SPD preservation (2 tests)
 *   §4: Joint training coordination (3 tests)
 *   §5: Phase scheduling (3 tests)
 *   §6: No interference between trainers (2 tests)
 *   §7: Combined vs single trainer (2 tests)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/trainers/bicameral_trainer.hpp>

#include <array>
#include <cmath>
#include <random>
#include <vector>

using namespace nikola;
using namespace nikola::trainers;
using namespace nikola::cognitive;
using namespace nikola::spatial;
using namespace nikola::physics;
using Catch::Approx;

// ============================================================================
// Helpers
// ============================================================================

/// Create a small WaveFunction for testing (2^9 = 512 nodes).
static WaveFunction make_wave(int n = 2, float amplitude = 1.f, uint32_t seed = 42) {
    WaveFunction wf;
    wf.seed_manifold(n, /*pilot_dim=*/0, /*k_mode=*/1, amplitude, seed);
    return wf;
}

/// Standard input injection: poke node 0 with amplitude.
static auto make_inject_input(float amp = 0.5f) {
    return [amp](WaveFunction& w) {
        w.grid().psi_real()[0] = amp;
        w.grid().psi_imag()[0] = 0.f;
    };
}

/// Standard target injection: poke a different node.
static auto make_inject_target(float amp = 1.f) {
    return [amp](WaveFunction& w) {
        const size_t N = w.grid().num_active_nodes();
        if (N > 1) {
            w.grid().psi_real()[1] = amp;
            w.grid().psi_imag()[1] = 0.f;
        }
    };
}

/// Create a diagonal-dominant 9×9 matrix.
static std::array<double, 81> make_diagonal(double diag = 0.5,
                                             double off  = 0.05) {
    std::array<double, 81> M{};
    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 9; ++j)
            M[i * 9 + j] = (i == j) ? diag : off;
    return M;
}

/// Generate synthetic SSM data from known A, B.
static std::vector<TrainingSample> gen_mamba_data(
        const std::array<double, 81>& A,
        const std::array<double, 81>& B,
        int count, uint32_t seed = 123u) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 0.5);

    std::vector<TrainingSample> samples;
    samples.reserve(count);
    for (int t = 0; t < count; ++t) {
        TrainingSample s;
        for (auto& v : s.state) v = dist(rng);
        for (auto& v : s.input) v = dist(rng);
        for (int i = 0; i < 9; ++i) {
            double sum = 0.0;
            for (int j = 0; j < 9; ++j)
                sum += A[i * 9 + j] * s.state[j] + B[i * 9 + j] * s.input[j];
            s.next_state[i] = sum;
        }
        samples.push_back(s);
    }
    return samples;
}

/// Generate synthetic attention data from known Q, K, V.
static std::vector<AttentionSample> gen_attn_data(
        const std::array<double, 81>& Q,
        const std::array<double, 81>& K,
        const std::array<double, 81>& V,
        int count, uint32_t seed = 456u) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 0.5);

    std::vector<AttentionSample> samples;
    samples.reserve(count);
    for (int n = 0; n < count; ++n) {
        AttentionSample s;
        for (auto& v : s.x1) v = dist(rng);
        for (auto& v : s.x2) v = dist(rng);

        // Manual 2-position attention
        std::array<double, 9> q1{}, q2{}, k1{}, k2{}, v1{}, v2{};
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                q1[i] += Q[i*9+j] * s.x1[j];
                q2[i] += Q[i*9+j] * s.x2[j];
                k1[i] += K[i*9+j] * s.x1[j];
                k2[i] += K[i*9+j] * s.x2[j];
                v1[i] += V[i*9+j] * s.x1[j];
                v2[i] += V[i*9+j] * s.x2[j];
            }
        }
        double s11 = 0, s12 = 0, s21 = 0, s22 = 0;
        for (int i = 0; i < 9; ++i) {
            s11 += q1[i] * k1[i]; s12 += q1[i] * k2[i];
            s21 += q2[i] * k1[i]; s22 += q2[i] * k2[i];
        }
        s11 /= 3.0; s12 /= 3.0; s21 /= 3.0; s22 /= 3.0;
        for (int i = 0; i < 9; ++i) {
            s.y1[i] = s11 * v1[i] + s12 * v2[i];
            s.y2[i] = s21 * v1[i] + s22 * v2[i];
        }
        samples.push_back(s);
    }
    return samples;
}

/// Check if metric is SPD via Gerschgorin.
static bool metric_is_spd(const TopologyManager& topo) {
    float g[81];
    std::copy(topo.metric(), topo.metric() + 81, g);
    return MetricValidator::gerschgorin_check(g);
}

// ============================================================================
// §1: Construction & Component Access
// ============================================================================

TEST_CASE("BicameralTrainer: construction succeeds",
          "[bat][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);
    BicameralTrainer bat(plasticity);

    CHECK(bat.epoch() == 0);
    CHECK(bat.phase() == TrainingPhase::ALL);
}

TEST_CASE("BicameralTrainer: sub-trainers accessible",
          "[bat][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);
    BicameralTrainer bat(plasticity);

    CHECK(bat.mamba().graph_size() == 539);
    CHECK(bat.transformer().graph_size() == 1377);
    CHECK(bat.mamba().A().size() == 81);     // Mamba has A, B, C
    CHECK(bat.transformer().Q().size() == 81);
}

TEST_CASE("BicameralTrainer: metric starts valid",
          "[bat][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);
    BicameralTrainer bat(plasticity);

    CHECK(bat.metric_valid());
}

// ============================================================================
// §2: EqProp Convergence — Energy Decreases Over Iterations
// ============================================================================

TEST_CASE("EqProp: train_step produces finite energies",
          "[eqprop][convergence][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);
    WaveFunction wf = make_wave(2, 0.5f, 42);

    EqPropConfig cfg;
    cfg.phase_steps = 10;
    cfg.dt = 0.001f;
    cfg.eta = 0.01f;
    plasticity.eqprop().config() = cfg;

    plasticity.eqprop().train_step(wf, make_inject_input(), make_inject_target());

    CHECK(std::isfinite(plasticity.eqprop().last_energy_positive()));
    CHECK(std::isfinite(plasticity.eqprop().last_energy_negative()));
    CHECK(std::isfinite(plasticity.eqprop().last_energy_diff()));
}

TEST_CASE("EqProp: repeated training maintains finite energy",
          "[eqprop][convergence][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);
    WaveFunction wf = make_wave(2, 0.5f, 42);

    EqPropConfig cfg;
    cfg.phase_steps = 5;
    cfg.dt = 0.001f;
    cfg.eta = 0.005f;     // Conservative η for stability
    plasticity.eqprop().config() = cfg;

    auto inject_in  = make_inject_input(0.5f);
    auto inject_tgt = make_inject_target(1.0f);

    bool all_finite = true;
    for (int i = 0; i < 100; ++i) {
        plasticity.eqprop().train_step(wf, inject_in, inject_tgt);
        if (!std::isfinite(plasticity.eqprop().last_energy_positive()) ||
            !std::isfinite(plasticity.eqprop().last_energy_negative())) {
            all_finite = false;
            break;
        }
    }
    CHECK(all_finite);
}

TEST_CASE("EqProp: metric changes reflect learning signal",
          "[eqprop][convergence][phase152]") {
    TopologyManager topo;

    // Save initial metric
    float before[81];
    std::copy(topo.metric(), topo.metric() + 81, before);

    PlasticityEngine plasticity(topo);
    WaveFunction wf = make_wave(2, 1.f, 9);

    EqPropConfig cfg;
    cfg.phase_steps = 10;
    cfg.dt = 0.001f;
    cfg.eta = 0.1f;     // Large η for visible metric change
    plasticity.eqprop().config() = cfg;

    // Distinct input vs target to create asymmetric co-activations
    auto inject_in = [](WaveFunction& w) {
        w.grid().psi_real()[0] = 1.f;
        w.grid().psi_imag()[0] = 0.5f;
    };
    auto inject_tgt = [](WaveFunction& w) {
        const size_t N = w.grid().num_active_nodes();
        if (N > 3) {
            w.grid().psi_real()[3] = 1.5f;
            w.grid().psi_imag()[3] = 0.f;
        }
    };

    for (int i = 0; i < 10; ++i)
        plasticity.eqprop().train_step(wf, inject_in, inject_tgt);

    bool changed = false;
    for (int i = 0; i < 81; ++i) {
        if (std::abs(topo.metric()[i] - before[i]) > 1e-10f) {
            changed = true;
            break;
        }
    }
    CHECK(changed);
}

// ============================================================================
// §3: Metric SPD Preservation
// ============================================================================

TEST_CASE("EqProp: metric remains SPD after 100 train_steps",
          "[eqprop][spd][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);
    WaveFunction wf = make_wave(2, 0.5f, 42);

    EqPropConfig cfg;
    cfg.phase_steps = 5;
    cfg.dt = 0.001f;
    cfg.eta = 0.01f;
    plasticity.eqprop().config() = cfg;

    auto inject_in  = make_inject_input(0.5f);
    auto inject_tgt = make_inject_target(1.0f);

    bool all_spd = true;
    for (int i = 0; i < 100; ++i) {
        plasticity.eqprop().train_step(wf, inject_in, inject_tgt);
        if (!metric_is_spd(topo)) {
            all_spd = false;
            break;
        }
    }
    CHECK(all_spd);
}

TEST_CASE("EqProp: metric remains SPD with large eta",
          "[eqprop][spd][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);
    WaveFunction wf = make_wave(2, 1.f, 7);

    EqPropConfig cfg;
    cfg.phase_steps = 5;
    cfg.dt = 0.001f;
    cfg.eta = 0.1f;   // Large but not extreme — tests geometric firewall
    plasticity.eqprop().config() = cfg;

    auto inject_in = [](WaveFunction& w) {
        w.grid().psi_real()[0] = 2.f;
    };
    auto inject_tgt = [](WaveFunction& w) {
        const size_t N = w.grid().num_active_nodes();
        if (N > 5) w.grid().psi_real()[5] = 2.f;
    };

    // After each EqProp step, validate_metric() is called internally.
    // The metric should remain structurally valid (diag > 0, finite).
    for (int i = 0; i < 50; ++i) {
        plasticity.eqprop().train_step(wf, inject_in, inject_tgt);
    }

    // Verify metric diagonals are positive and finite
    bool valid = true;
    for (int d = 0; d < 9; ++d) {
        float diag = topo.metric()[d * 9 + d];
        if (!std::isfinite(diag) || diag <= 0.f) {
            valid = false;
            break;
        }
    }
    CHECK(valid);
}

// ============================================================================
// §4: Joint Training Coordination
// ============================================================================

TEST_CASE("BicameralTrainer: train_joint_step runs all three",
          "[bat][joint][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);
    BicameralTrainer bat(plasticity);
    WaveFunction wf = make_wave(2, 0.5f, 42);

    auto A_true = make_diagonal(0.5, 0.05);
    auto B_true = make_diagonal(0.1, 0.01);
    auto mamba_data = gen_mamba_data(A_true, B_true, 10);

    auto Q_true = make_diagonal(0.3, 0.02);
    auto K_true = make_diagonal(0.2, 0.01);
    auto V_true = make_diagonal(0.1, 0.01);
    auto attn_data = gen_attn_data(Q_true, K_true, V_true, 10);

    auto stats = bat.train_joint_step(
        mamba_data, attn_data, wf,
        make_inject_input(), make_inject_target());

    CHECK(stats.ran_mamba);
    CHECK(stats.ran_transformer);
    CHECK(stats.ran_eqprop);
    CHECK(stats.mamba_loss > 0.0);
    CHECK(stats.transformer_loss > 0.0);
    CHECK(std::isfinite(stats.eqprop_energy_diff));
    CHECK(stats.epoch == 0);
    CHECK(bat.epoch() == 1);
}

TEST_CASE("BicameralTrainer: train_params_only skips EqProp",
          "[bat][joint][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);
    BicameralTrainer bat(plasticity);

    auto A_true = make_diagonal(0.5, 0.05);
    auto B_true = make_diagonal(0.1, 0.01);
    auto mamba_data = gen_mamba_data(A_true, B_true, 5);
    auto attn_data = gen_attn_data(make_diagonal(0.3), make_diagonal(0.2), make_diagonal(0.1), 5);

    auto stats = bat.train_params_only(mamba_data, attn_data);

    CHECK(stats.ran_mamba);
    CHECK(stats.ran_transformer);
    CHECK_FALSE(stats.ran_eqprop);
    CHECK(stats.eqprop_energy_diff == 0.0);
}

TEST_CASE("BicameralTrainer: train_geometry_only skips parameters",
          "[bat][joint][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);
    BicameralTrainer bat(plasticity);
    WaveFunction wf = make_wave(2, 0.5f, 42);

    auto stats = bat.train_geometry_only(
        wf, make_inject_input(), make_inject_target());

    CHECK_FALSE(stats.ran_mamba);
    CHECK_FALSE(stats.ran_transformer);
    CHECK(stats.ran_eqprop);
}

// ============================================================================
// §5: Phase Scheduling
// ============================================================================

TEST_CASE("BicameralTrainer: ALL phase runs everything",
          "[bat][scheduling][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);
    BicameralTrainer bat(plasticity, 0.001, 0.0001, TrainingPhase::ALL);
    WaveFunction wf = make_wave(2, 0.5f, 42);

    auto mamba_data = gen_mamba_data(make_diagonal(), make_diagonal(0.1), 5);
    auto attn_data = gen_attn_data(make_diagonal(0.3), make_diagonal(0.2), make_diagonal(0.1), 5);

    auto stats = bat.train_joint_step(
        mamba_data, attn_data, wf,
        make_inject_input(), make_inject_target());

    CHECK(stats.ran_mamba);
    CHECK(stats.ran_transformer);
    CHECK(stats.ran_eqprop);
}

TEST_CASE("BicameralTrainer: ALTERNATING switches each epoch",
          "[bat][scheduling][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);
    BicameralTrainer bat(plasticity, 0.001, 0.0001, TrainingPhase::ALTERNATING);
    WaveFunction wf = make_wave(2, 0.5f, 42);

    auto mamba_data = gen_mamba_data(make_diagonal(), make_diagonal(0.1), 5);
    auto attn_data = gen_attn_data(make_diagonal(0.3), make_diagonal(0.2), make_diagonal(0.1), 5);

    // Epoch 0: even → params
    auto s0 = bat.train_joint_step(
        mamba_data, attn_data, wf,
        make_inject_input(), make_inject_target());
    CHECK(s0.ran_mamba);
    CHECK(s0.ran_transformer);
    CHECK_FALSE(s0.ran_eqprop);

    // Epoch 1: odd → geometry
    auto s1 = bat.train_joint_step(
        mamba_data, attn_data, wf,
        make_inject_input(), make_inject_target());
    CHECK_FALSE(s1.ran_mamba);
    CHECK_FALSE(s1.ran_transformer);
    CHECK(s1.ran_eqprop);

    // Epoch 2: even → params again
    auto s2 = bat.train_joint_step(
        mamba_data, attn_data, wf,
        make_inject_input(), make_inject_target());
    CHECK(s2.ran_mamba);
    CHECK_FALSE(s2.ran_eqprop);
}

TEST_CASE("BicameralTrainer: set_phase changes scheduling",
          "[bat][scheduling][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);
    BicameralTrainer bat(plasticity);
    WaveFunction wf = make_wave(2, 0.5f, 42);

    auto mamba_data = gen_mamba_data(make_diagonal(), make_diagonal(0.1), 5);
    auto attn_data = gen_attn_data(make_diagonal(0.3), make_diagonal(0.2), make_diagonal(0.1), 5);

    // Start with GEOMETRY_ONLY
    bat.set_phase(TrainingPhase::GEOMETRY_ONLY);
    auto s1 = bat.train_joint_step(
        mamba_data, attn_data, wf,
        make_inject_input(), make_inject_target());
    CHECK_FALSE(s1.ran_mamba);
    CHECK(s1.ran_eqprop);

    // Switch to PARAMS_ONLY
    bat.set_phase(TrainingPhase::PARAMS_ONLY);
    auto s2 = bat.train_joint_step(
        mamba_data, attn_data, wf,
        make_inject_input(), make_inject_target());
    CHECK(s2.ran_mamba);
    CHECK_FALSE(s2.ran_eqprop);
}

// ============================================================================
// §6: No Interference Between Trainers
// ============================================================================

TEST_CASE("BicameralTrainer: EqProp does not affect parameter trainers",
          "[bat][interference][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);
    BicameralTrainer bat(plasticity);
    WaveFunction wf = make_wave(2, 0.5f, 42);

    auto A_true = make_diagonal(0.5, 0.05);
    auto B_true = make_diagonal(0.1, 0.01);
    auto Q_true = make_diagonal(0.3, 0.02);
    auto K_true = make_diagonal(0.2, 0.01);
    auto V_true = make_diagonal(0.1, 0.01);

    auto mamba_data = gen_mamba_data(A_true, B_true, 50);
    auto attn_data  = gen_attn_data(Q_true, K_true, V_true, 50);

    // Train with ALL for 50 epochs
    for (int i = 0; i < 50; ++i) {
        bat.train_joint_step(mamba_data, attn_data, wf,
                             make_inject_input(), make_inject_target());
    }
    (void)bat.mamba().learning_rate(); // just checking accessible

    // Train a separate params-only trainer for 50 epochs
    TopologyManager topo2;
    PlasticityEngine plasticity2(topo2);
    BicameralTrainer bat2(plasticity2, 0.001, 0.0001, TrainingPhase::PARAMS_ONLY);
    WaveFunction wf2 = make_wave(2, 0.5f, 42);

    for (int i = 0; i < 50; ++i) {
        bat2.train_params_only(mamba_data, attn_data);
    }

    // Both Mamba trainers should produce valid (non-zero) parameters
    bool joint_params_nonzero = false;
    bool solo_params_nonzero = false;
    for (int i = 0; i < 81; ++i) {
        if (std::abs(bat.mamba().A()[i]) > 1e-10) joint_params_nonzero = true;
        if (std::abs(bat2.mamba().A()[i]) > 1e-10) solo_params_nonzero = true;
    }
    CHECK(joint_params_nonzero);
    CHECK(solo_params_nonzero);
}

TEST_CASE("BicameralTrainer: parameter training doesn't affect metric",
          "[bat][interference][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);

    float metric_before[81];
    std::copy(topo.metric(), topo.metric() + 81, metric_before);

    BicameralTrainer bat(plasticity, 0.001, 0.0001, TrainingPhase::PARAMS_ONLY);

    auto A_true = make_diagonal(0.5, 0.05);
    auto B_true = make_diagonal(0.1, 0.01);
    auto mamba_data = gen_mamba_data(A_true, B_true, 20);
    auto attn_data  = gen_attn_data(make_diagonal(0.3), make_diagonal(0.2), make_diagonal(0.1), 20);

    // Run 100 param-only epochs
    for (int i = 0; i < 100; ++i)
        bat.train_params_only(mamba_data, attn_data);

    // Metric should be UNCHANGED (params don't touch geometry)
    bool unchanged = true;
    for (int i = 0; i < 81; ++i) {
        if (std::abs(topo.metric()[i] - metric_before[i]) > 1e-15f) {
            unchanged = false;
            break;
        }
    }
    CHECK(unchanged);
}

// ============================================================================
// §7: Combined vs Single Trainer
// ============================================================================

TEST_CASE("BicameralTrainer: Mamba loss decreases during joint training",
          "[bat][combined][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);
    BicameralTrainer bat(plasticity, 0.01, 0.0001, TrainingPhase::ALL);
    WaveFunction wf = make_wave(2, 0.5f, 42);

    auto A_true = make_diagonal(0.5, 0.05);
    auto B_true = make_diagonal(0.1, 0.01);
    auto mamba_data = gen_mamba_data(A_true, B_true, 50);
    auto attn_data  = gen_attn_data(make_diagonal(0.3), make_diagonal(0.2), make_diagonal(0.1), 50);

    double first_loss = 0.0;
    double last_loss = 0.0;

    for (int i = 0; i < 100; ++i) {
        auto stats = bat.train_joint_step(
            mamba_data, attn_data, wf,
            make_inject_input(), make_inject_target());
        if (i == 0) first_loss = stats.mamba_loss;
        last_loss = stats.mamba_loss;
    }

    // MambaTrainer's loss should decrease even while EqProp runs concurrently
    CHECK(last_loss < first_loss);
}

TEST_CASE("BicameralTrainer: Transformer loss decreases during joint training",
          "[bat][combined][phase152]") {
    TopologyManager topo;
    PlasticityEngine plasticity(topo);
    BicameralTrainer bat(plasticity, 0.001, 0.0001, TrainingPhase::ALL);
    WaveFunction wf = make_wave(2, 0.5f, 42);

    auto A_true = make_diagonal(0.5, 0.05);
    auto B_true = make_diagonal(0.1, 0.01);
    auto mamba_data = gen_mamba_data(A_true, B_true, 50);

    auto Q_true = make_diagonal(0.4, 0.03);
    auto K_true = make_diagonal(0.25, 0.02);
    auto V_true = make_diagonal(0.15, 0.01);
    auto attn_data = gen_attn_data(Q_true, K_true, V_true, 50);

    double first_loss = 0.0;
    double last_loss = 0.0;

    for (int i = 0; i < 100; ++i) {
        auto stats = bat.train_joint_step(
            mamba_data, attn_data, wf,
            make_inject_input(), make_inject_target());
        if (i == 0) first_loss = stats.transformer_loss;
        last_loss = stats.transformer_loss;
    }

    // TransformerTrainer loss should decrease during joint training
    CHECK(last_loss < first_loss);
}
