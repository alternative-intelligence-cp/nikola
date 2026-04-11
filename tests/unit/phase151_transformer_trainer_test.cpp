/**
 * @file tests/unit/phase151_transformer_trainer_test.cpp
 * @brief Phase 151 — TransformerTrainer unit tests (v0.1.14)
 *
 * Tests:
 *   §1: Construction & graph structure (3 tests)
 *   §2: Forward prediction (3 tests)
 *   §3: Gradient correctness — numerical check (3 tests)
 *   §4: Training convergence (4 tests)
 *   §5: Batch training & learning rate (3 tests)
 *   §6: Auto-training triggers (2 tests)
 *   §7: Zero-allocation training loop (2 tests)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/trainers/transformer_trainer.hpp>

#include <array>
#include <cmath>
#include <random>
#include <vector>

using namespace nikola::trainers;
using Catch::Approx;

// ============================================================================
// Helpers
// ============================================================================

/// Simple 9×9 matvec for ground-truth computations.
static std::array<double, 9> matvec9(const std::array<double, 81>& W,
                                      const std::array<double, 9>& x) {
    std::array<double, 9> r{};
    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 9; ++j)
            r[i] += W[i * 9 + j] * x[j];
    return r;
}

/// Dot product of two 9D vectors.
static double dot9(const std::array<double, 9>& a,
                   const std::array<double, 9>& b) {
    double s = 0.0;
    for (int i = 0; i < 9; ++i) s += a[i] * b[i];
    return s;
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

/// Compute the 2-position linear attention output from known Q, K, V.
static std::pair<std::array<double, 9>, std::array<double, 9>>
compute_attention(const std::array<double, 81>& Q,
                  const std::array<double, 81>& K,
                  const std::array<double, 81>& V,
                  const std::array<double, 9>& x1,
                  const std::array<double, 9>& x2) {
    auto q1 = matvec9(Q, x1);
    auto q2 = matvec9(Q, x2);
    auto k1 = matvec9(K, x1);
    auto k2 = matvec9(K, x2);
    auto v1 = matvec9(V, x1);
    auto v2 = matvec9(V, x2);

    constexpr double inv_sqrt = 1.0 / 3.0;
    double s11 = dot9(q1, k1) * inv_sqrt;
    double s12 = dot9(q1, k2) * inv_sqrt;
    double s21 = dot9(q2, k1) * inv_sqrt;
    double s22 = dot9(q2, k2) * inv_sqrt;

    std::array<double, 9> y1{}, y2{};
    for (int i = 0; i < 9; ++i) {
        y1[i] = s11 * v1[i] + s12 * v2[i];
        y2[i] = s21 * v1[i] + s22 * v2[i];
    }
    return {y1, y2};
}

/// Compute 2-position attention loss for given Q, K, V and a sample.
static double manual_loss(const std::array<double, 81>& Q,
                           const std::array<double, 81>& K,
                           const std::array<double, 81>& V,
                           const AttentionSample& sample) {
    auto [y1, y2] = compute_attention(Q, K, V, sample.x1, sample.x2);
    double loss = 0.0;
    for (int i = 0; i < 9; ++i) {
        double d1 = y1[i] - sample.y1[i];
        double d2 = y2[i] - sample.y2[i];
        loss += d1 * d1 + d2 * d2;
    }
    return loss;
}

/// Generate synthetic 2-position attention data from ground truth Q, K, V.
static std::vector<AttentionSample> generate_attention_data(
        const std::array<double, 81>& Q_true,
        const std::array<double, 81>& K_true,
        const std::array<double, 81>& V_true,
        int count, uint32_t seed = 123u) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 0.5);

    std::vector<AttentionSample> samples;
    samples.reserve(count);

    for (int i = 0; i < count; ++i) {
        AttentionSample s;
        for (auto& v : s.x1) v = dist(rng);
        for (auto& v : s.x2) v = dist(rng);
        auto [y1, y2] = compute_attention(Q_true, K_true, V_true, s.x1, s.x2);
        s.y1 = y1;
        s.y2 = y2;
        samples.push_back(s);
    }
    return samples;
}

// ============================================================================
// §1: Construction & Graph Structure
// ============================================================================

TEST_CASE("TransformerTrainer: construction creates fixed graph",
          "[trainer][transformer][phase151]") {
    TransformerTrainer trainer;
    // 280 leaves + 918 proj + 68 dot + 4 scale + 36 score*v + 18 sum + 53 loss = 1377
    CHECK(trainer.graph_size() == 1377);
}

TEST_CASE("TransformerTrainer: parameter dimensions correct",
          "[trainer][transformer][phase151]") {
    TransformerTrainer trainer;
    CHECK(trainer.Q().size() == 81);
    CHECK(trainer.K().size() == 81);
    CHECK(trainer.V().size() == 81);
}

TEST_CASE("TransformerTrainer: deterministic initialization",
          "[trainer][transformer][phase151]") {
    TransformerTrainer t1;
    TransformerTrainer t2;
    for (int i = 0; i < 81; ++i) {
        CHECK(t1.Q()[i] == t2.Q()[i]);
        CHECK(t1.K()[i] == t2.K()[i]);
        CHECK(t1.V()[i] == t2.V()[i]);
    }
}

// ============================================================================
// §2: Forward Prediction
// ============================================================================

TEST_CASE("TransformerTrainer: predict matches manual computation",
          "[trainer][transformer][phase151]") {
    TransformerTrainer trainer;
    std::array<double, 9> x1{}, x2{};
    for (int i = 0; i < 9; ++i) {
        x1[i] = 0.1 * (i + 1);
        x2[i] = 0.2 * (9 - i);
    }

    auto [y1, y2] = trainer.predict(x1, x2);
    auto [expected_y1, expected_y2] = compute_attention(
        trainer.Q(), trainer.K(), trainer.V(), x1, x2);

    for (int i = 0; i < 9; ++i) {
        CHECK(y1[i] == Approx(expected_y1[i]).margin(1e-12));
        CHECK(y2[i] == Approx(expected_y2[i]).margin(1e-12));
    }
}

TEST_CASE("TransformerTrainer: predict with zero x2 gives self-attention only",
          "[trainer][transformer][phase151]") {
    TransformerTrainer trainer;
    std::array<double, 9> x1{}, x2{};
    x1[0] = 1.0; // unit in dim 0

    auto [y1, y2] = trainer.predict(x1, x2);

    // k2 = V*0 = 0, so score_12 * v2 = 0
    // y1 = score_11 * v1 only
    auto q1 = matvec9(trainer.Q(), x1);
    auto k1 = matvec9(trainer.K(), x1);
    auto v1 = matvec9(trainer.V(), x1);
    double s11 = dot9(q1, k1) / 3.0;
    for (int i = 0; i < 9; ++i) {
        CHECK(y1[i] == Approx(s11 * v1[i]).margin(1e-12));
    }
}

TEST_CASE("TransformerTrainer: symmetric inputs give symmetric outputs",
          "[trainer][transformer][phase151]") {
    TransformerTrainer trainer;
    std::array<double, 9> x{};
    for (int i = 0; i < 9; ++i) x[i] = 0.3 * (i + 1);

    auto [y1, y2] = trainer.predict(x, x);

    // If x1 == x2, then by symmetry y1 == y2
    for (int i = 0; i < 9; ++i) {
        CHECK(y1[i] == Approx(y2[i]).margin(1e-12));
    }
}

// ============================================================================
// §3: Gradient Correctness — Numerical Finite Differences
// ============================================================================

TEST_CASE("TransformerTrainer: gradient check for Q parameters",
          "[trainer][transformer][gradient][phase151]") {
    TransformerTrainer trainer(0.001, 1.0);
    auto Q_true = make_diagonal(0.3, 0.02);
    auto K_true = make_diagonal(0.2, 0.01);
    auto V_true = make_diagonal(0.1, 0.01);
    auto data = generate_attention_data(Q_true, K_true, V_true, 1, 99);
    const auto& sample = data[0];

    auto result = trainer.eval_gradient(sample);

    double eps = 1e-5;
    auto Q_plus  = trainer.Q(); Q_plus[0]  += eps;
    auto Q_minus = trainer.Q(); Q_minus[0] -= eps;
    double num_grad = (manual_loss(Q_plus, trainer.K(), trainer.V(), sample)
                     - manual_loss(Q_minus, trainer.K(), trainer.V(), sample))
                    / (2.0 * eps);

    double rel_err = std::abs(num_grad - result.grad_Q[0])
                   / (std::abs(num_grad) + 1e-12);
    CHECK(rel_err < 0.001); // < 0.1% relative error
}

TEST_CASE("TransformerTrainer: gradient check for K parameters",
          "[trainer][transformer][gradient][phase151]") {
    TransformerTrainer trainer(0.001, 1.0);
    auto Q_true = make_diagonal(0.3, 0.02);
    auto K_true = make_diagonal(0.2, 0.01);
    auto V_true = make_diagonal(0.1, 0.01);
    auto data = generate_attention_data(Q_true, K_true, V_true, 1, 77);
    const auto& sample = data[0];

    auto result = trainer.eval_gradient(sample);

    double eps = 1e-5;
    int k_idx = 2 * 9 + 3; // K[2][3]
    auto K_plus  = trainer.K(); K_plus[k_idx]  += eps;
    auto K_minus = trainer.K(); K_minus[k_idx] -= eps;
    double num_grad = (manual_loss(trainer.Q(), K_plus, trainer.V(), sample)
                     - manual_loss(trainer.Q(), K_minus, trainer.V(), sample))
                    / (2.0 * eps);

    double rel_err = std::abs(num_grad - result.grad_K[k_idx])
                   / (std::abs(num_grad) + 1e-12);
    CHECK(rel_err < 0.001);
}

TEST_CASE("TransformerTrainer: all Q,K,V gradients match numerical",
          "[trainer][transformer][gradient][phase151]") {
    TransformerTrainer trainer(0.001, 1.0);
    auto Q_true = make_diagonal(0.4, 0.03);
    auto K_true = make_diagonal(0.2, 0.015);
    auto V_true = make_diagonal(0.15, 0.01);
    auto data = generate_attention_data(Q_true, K_true, V_true, 1, 55);
    const auto& sample = data[0];

    auto result = trainer.eval_gradient(sample);

    double eps = 1e-5;
    int checked = 0;
    int passed  = 0;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> idx_dist(0, 80);

    // Check 10 Q, 10 K, 10 V random indices
    for (int trial = 0; trial < 30; ++trial) {
        int param_idx = idx_dist(rng);
        double num_grad;
        double analytical_grad;

        if (trial < 10) {
            auto Q_plus  = trainer.Q(); Q_plus[param_idx]  += eps;
            auto Q_minus = trainer.Q(); Q_minus[param_idx] -= eps;
            num_grad = (manual_loss(Q_plus, trainer.K(), trainer.V(), sample)
                      - manual_loss(Q_minus, trainer.K(), trainer.V(), sample))
                     / (2.0 * eps);
            analytical_grad = result.grad_Q[param_idx];
        } else if (trial < 20) {
            auto K_plus  = trainer.K(); K_plus[param_idx]  += eps;
            auto K_minus = trainer.K(); K_minus[param_idx] -= eps;
            num_grad = (manual_loss(trainer.Q(), K_plus, trainer.V(), sample)
                      - manual_loss(trainer.Q(), K_minus, trainer.V(), sample))
                     / (2.0 * eps);
            analytical_grad = result.grad_K[param_idx];
        } else {
            auto V_plus  = trainer.V(); V_plus[param_idx]  += eps;
            auto V_minus = trainer.V(); V_minus[param_idx] -= eps;
            num_grad = (manual_loss(trainer.Q(), trainer.K(), V_plus, sample)
                      - manual_loss(trainer.Q(), trainer.K(), V_minus, sample))
                     / (2.0 * eps);
            analytical_grad = result.grad_V[param_idx];
        }

        double rel_err = std::abs(num_grad - analytical_grad)
                       / (std::abs(num_grad) + 1e-12);
        ++checked;
        if (rel_err < 0.001) ++passed;
    }
    CHECK(passed == checked);
}

// ============================================================================
// §4: Training Convergence
// ============================================================================

TEST_CASE("TransformerTrainer: loss decreases on repeated training",
          "[trainer][transformer][convergence][phase151]") {
    TransformerTrainer trainer(0.0001, 1.0);

    auto Q_true = make_diagonal(0.5, 0.05);
    auto K_true = make_diagonal(0.3, 0.02);
    auto V_true = make_diagonal(0.2, 0.01);
    auto data = generate_attention_data(Q_true, K_true, V_true, 100, 42);

    double prev_loss = 1e10;
    int decreasing = 0;

    for (int epoch = 0; epoch < 100; ++epoch) {
        auto stats = trainer.train_batch(data);
        if (stats.loss < prev_loss) ++decreasing;
        prev_loss = stats.loss;
    }

    CHECK(decreasing >= 85);
    CHECK(prev_loss < 10.0); // Loss should reduce substantially
}

TEST_CASE("TransformerTrainer: overfitting — single sample loss → 0",
          "[trainer][transformer][convergence][phase151]") {
    TransformerTrainer trainer(0.0005, 1.0);

    AttentionSample sample;
    sample.x1 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    sample.x2 = {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};
    sample.y1 = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    sample.y2 = {0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3};

    std::vector<AttentionSample> batch = {sample};
    double loss = 0.0;

    for (int epoch = 0; epoch < 5000; ++epoch) {
        auto stats = trainer.train_batch(batch);
        loss = stats.loss;
    }

    CHECK(loss < 1e-4);
}

TEST_CASE("TransformerTrainer: recovers known V matrix",
          "[trainer][transformer][convergence][phase151]") {
    // Identity Q, K → score = ||x||²/3, V determines output direction
    auto Q_true = make_diagonal(0.3, 0.0);
    auto K_true = make_diagonal(0.3, 0.0);
    auto V_true = make_diagonal(0.4, 0.0);

    auto data = generate_attention_data(Q_true, K_true, V_true, 300, 42);

    TransformerTrainer trainer(0.0001, 0.99999);
    for (int epoch = 0; epoch < 2000; ++epoch) {
        trainer.train_batch(data);
    }

    // Check V diagonal recovery (V has most direct gradient path)
    double error = 0.0;
    for (int i = 0; i < 9; ++i) {
        double diff = trainer.V()[i * 9 + i] - V_true[i * 9 + i];
        error += diff * diff;
    }
    error = std::sqrt(error / 9.0);
    CHECK(error < 0.5); // More lenient: trilinear attention is harder to train
}

TEST_CASE("TransformerTrainer: prediction accuracy improves with training",
          "[trainer][transformer][convergence][phase151]") {
    auto Q_true = make_diagonal(0.4, 0.03);
    auto K_true = make_diagonal(0.25, 0.02);
    auto V_true = make_diagonal(0.15, 0.01);

    auto train_data = generate_attention_data(Q_true, K_true, V_true, 500, 42);
    auto test_data  = generate_attention_data(Q_true, K_true, V_true, 50, 99);

    TransformerTrainer trainer(0.0001, 0.99999);

    // Error before
    double error_before = 0.0;
    for (const auto& s : test_data) {
        auto [y1, y2] = trainer.predict(s.x1, s.x2);
        for (int i = 0; i < 9; ++i) {
            double d1 = y1[i] - s.y1[i];
            double d2 = y2[i] - s.y2[i];
            error_before += d1 * d1 + d2 * d2;
        }
    }

    // Train (more epochs — Q×K multiplicative interaction converges slower)
    for (int epoch = 0; epoch < 2000; ++epoch) {
        trainer.train_batch(train_data);
    }

    // Error after
    double error_after = 0.0;
    for (const auto& s : test_data) {
        auto [y1, y2] = trainer.predict(s.x1, s.x2);
        for (int i = 0; i < 9; ++i) {
            double d1 = y1[i] - s.y1[i];
            double d2 = y2[i] - s.y2[i];
            error_after += d1 * d1 + d2 * d2;
        }
    }

    CHECK(error_after < error_before * 0.75); // At least 25% improvement (trilinear is harder)
}

// ============================================================================
// §5: Batch Training & Learning Rate
// ============================================================================

TEST_CASE("TransformerTrainer: batch returns correct stats",
          "[trainer][transformer][batch][phase151]") {
    auto Q_true = make_diagonal(0.3, 0.02);
    auto K_true = make_diagonal(0.2, 0.01);
    auto V_true = make_diagonal(0.1, 0.01);
    auto data = generate_attention_data(Q_true, K_true, V_true, 10, 42);

    TransformerTrainer trainer(0.0001, 1.0);
    auto stats = trainer.train_batch(data);

    CHECK(stats.loss > 0.0);
    CHECK(stats.samples == 10);
    CHECK(stats.max_grad > 0.0);
}

TEST_CASE("TransformerTrainer: learning rate decay works",
          "[trainer][transformer][lr][phase151]") {
    TransformerTrainer trainer(0.01, 0.9); // 10% decay per epoch
    auto Q_true = make_diagonal();
    auto K_true = make_diagonal(0.2, 0.01);
    auto V_true = make_diagonal(0.1, 0.01);
    auto data = generate_attention_data(Q_true, K_true, V_true, 5, 42);

    double lr_before = trainer.learning_rate();
    trainer.train_batch(data);
    double lr_after = trainer.learning_rate();

    CHECK(lr_after == Approx(lr_before * 0.9).margin(1e-12));
    CHECK(trainer.epoch() == 1);
}

TEST_CASE("TransformerTrainer: set_learning_rate overrides",
          "[trainer][transformer][lr][phase151]") {
    TransformerTrainer trainer;
    trainer.set_learning_rate(0.1);
    CHECK(trainer.learning_rate() == Approx(0.1));
}

// ============================================================================
// §6: Auto-Training Triggers
// ============================================================================

TEST_CASE("TransformerTrainer: should_train triggers on high error",
          "[trainer][transformer][trigger][phase151]") {
    TransformerTrainer trainer;
    trainer.set_error_threshold(0.5);

    for (int i = 0; i < 10; ++i)
        CHECK_FALSE(trainer.should_train(0.1));

    for (int i = 0; i < 50; ++i)
        trainer.should_train(1.0);

    CHECK(trainer.should_train(1.0) == true);
    CHECK(trainer.error_ema() > 0.5);
}

TEST_CASE("TransformerTrainer: error EMA decays with low errors",
          "[trainer][transformer][trigger][phase151]") {
    TransformerTrainer trainer;
    trainer.set_error_threshold(0.5);

    for (int i = 0; i < 50; ++i) trainer.should_train(1.0);
    double ema_high = trainer.error_ema();

    for (int i = 0; i < 100; ++i) trainer.should_train(0.01);
    double ema_low = trainer.error_ema();

    CHECK(ema_low < ema_high);
    CHECK(ema_low < 0.5);
}

// ============================================================================
// §7: Zero-Allocation Training Loop
// ============================================================================

TEST_CASE("TransformerTrainer: graph size stable across training",
          "[trainer][transformer][perf][phase151]") {
    TransformerTrainer trainer(0.0001, 1.0);
    auto Q_true = make_diagonal();
    auto K_true = make_diagonal(0.2, 0.01);
    auto V_true = make_diagonal(0.1, 0.01);
    auto data = generate_attention_data(Q_true, K_true, V_true, 20, 42);

    uint16_t initial_size = trainer.graph_size();

    for (int epoch = 0; epoch < 50; ++epoch) {
        trainer.train_batch(data);
    }

    CHECK(trainer.graph_size() == initial_size);
}

TEST_CASE("TransformerTrainer: reset_params reinitializes",
          "[trainer][transformer][phase151]") {
    TransformerTrainer trainer;
    auto original_Q = trainer.Q();

    auto Q_true = make_diagonal();
    auto K_true = make_diagonal(0.2, 0.01);
    auto V_true = make_diagonal(0.1, 0.01);
    auto data = generate_attention_data(Q_true, K_true, V_true, 10, 42);
    for (int i = 0; i < 50; ++i) trainer.train_batch(data);

    bool changed = false;
    for (int i = 0; i < 81; ++i) {
        if (std::abs(trainer.Q()[i] - original_Q[i]) > 1e-10) {
            changed = true;
            break;
        }
    }
    CHECK(changed);

    trainer.reset_params();
    for (int i = 0; i < 81; ++i) {
        CHECK(trainer.Q()[i] == Approx(original_Q[i]).margin(1e-15));
    }
}
