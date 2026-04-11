/**
 * @file include/nikola/trainers/mamba_trainer.hpp
 * @brief MambaTrainer — SSM parameter training via StaticComputeGraph
 *
 * First Bicameral Autonomous Trainer (BAT). Trains a reduced 9D SSM model
 * using gradient descent through the StaticComputeGraph autodiff engine.
 *
 * Architecture:
 *   - Reduced 9D parameter space matching the T⁹ manifold
 *   - A ∈ ℝ⁹ˣ⁹ (state transition), B ∈ ℝ⁹ˣ⁹ (input coupling), C ∈ ℝ⁹ (output)
 *   - Fixed graph topology built once at construction (539 nodes)
 *   - Training loop: set_value → forward → backward → SGD update
 *   - Zero allocations during training (StaticComputeGraph reuse)
 *
 * Training objective (teacher forcing):
 *   L = Σ_t ||A·s_t + B·x_t − s_{t+1}^{actual}||²
 *
 * The trained 9D parameters can later be mapped to the full 256-dim SSM
 * via the physics-derived expansion in Mamba9D::extract_ssm_params().
 *
 * @see docs/info/engineering/nikola_full.txt §5.2 (BAT — Mamba Trainer)
 * @see include/nikola/cognitive/mamba9d.hpp
 */
#pragma once

#include <nikola/core/static_autodiff.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numeric>
#include <vector>

namespace nikola::trainers {

/// Dimensions of the reduced T⁹ manifold SSM.
static constexpr int MAMBA_DIM = 9;
static constexpr int MAMBA_DIM_SQ = MAMBA_DIM * MAMBA_DIM; // 81

/// Training sample: one step of (state, input, next_state).
struct TrainingSample {
    std::array<double, 9> state;      ///< s_t (reduced hidden state)
    std::array<double, 9> input;      ///< x_t (torus input)
    std::array<double, 9> next_state; ///< s_{t+1} (actual next state)
};

/// Training statistics for one epoch or batch.
struct TrainingStats {
    double loss      = 0.0;  ///< Mean loss over samples
    double max_grad  = 0.0;  ///< Max gradient magnitude (for monitoring)
    int    samples   = 0;    ///< Number of samples processed
};

/**
 * @brief MambaTrainer — trains 9D SSM parameters via gradient descent.
 *
 * Uses a pre-built StaticComputeGraph with 539 nodes. The graph encodes
 * the SSM recurrence s_{t+1} = A·s + B·x and loss ||pred − actual||²
 * using scalar multiply/add operations so that A and B entries are
 * differentiable leaf nodes.
 *
 * Training loop (per sample):
 *   1. set_value() on leaves for current params, state, input, target
 *   2. forward() propagates through graph
 *   3. backward() computes gradients
 *   4. Accumulate gradients for batch
 *
 * After batch: SGD update on A, B, C parameters.
 */
class MambaTrainer {
public:
    explicit MambaTrainer(double learning_rate = 0.001,
                          double lr_decay      = 0.999)
        : learning_rate_(learning_rate)
        , lr_decay_(lr_decay)
    {
        init_params();
        build_graph();
    }

    // ── Training interface ─────────────────────────────────────────────

    /**
     * @brief Train on a batch of samples. Returns batch statistics.
     *
     * Processes all samples with teacher forcing: at each step, the actual
     * s_t is used as input (not the model's own prediction). Gradients
     * are accumulated over the batch, then a single SGD step is taken.
     */
    TrainingStats train_batch(const std::vector<TrainingSample>& batch) {
        TrainingStats stats;
        if (batch.empty()) return stats;

        // Zero gradient accumulators
        std::array<double, MAMBA_DIM_SQ> grad_A{};
        std::array<double, MAMBA_DIM_SQ> grad_B{};
        std::array<double, MAMBA_DIM>    grad_C{};

        double total_loss = 0.0;
        double max_grad   = 0.0;

        for (const auto& sample : batch) {
            // 1. Set leaf values for current parameters
            for (int i = 0; i < MAMBA_DIM; ++i) {
                for (int j = 0; j < MAMBA_DIM; ++j) {
                    graph_.set_value(a_ids_[i * MAMBA_DIM + j],
                                     {A_[i * MAMBA_DIM + j], 0.0});
                    graph_.set_value(b_ids_[i * MAMBA_DIM + j],
                                     {B_[i * MAMBA_DIM + j], 0.0});
                }
            }
            for (int i = 0; i < MAMBA_DIM; ++i) {
                graph_.set_value(c_ids_[i], {C_[i], 0.0});
            }

            // 2. Set leaf values for this sample's data
            for (int i = 0; i < MAMBA_DIM; ++i) {
                graph_.set_value(s_ids_[i], {sample.state[i], 0.0});
                graph_.set_value(x_ids_[i], {sample.input[i], 0.0});
                // Negate target for subtraction via addition
                graph_.set_value(neg_actual_ids_[i],
                                 {-sample.next_state[i], 0.0});
            }

            // 3. Forward propagation through existing graph
            graph_.zero_gradients();
            graph_.forward();

            // 4. Backward pass
            graph_.backward(loss_id_);

            // 5. Read loss value
            double sample_loss = graph_.get_value(loss_id_).real();
            total_loss += sample_loss;

            // 6. Accumulate gradients
            for (int idx = 0; idx < MAMBA_DIM_SQ; ++idx) {
                double ga = graph_.get_gradient(a_ids_[idx]).real();
                double gb = graph_.get_gradient(b_ids_[idx]).real();
                grad_A[idx] += ga;
                grad_B[idx] += gb;
                max_grad = std::max(max_grad, std::max(std::abs(ga), std::abs(gb)));
            }
            for (int i = 0; i < MAMBA_DIM; ++i) {
                double gc = graph_.get_gradient(c_ids_[i]).real();
                grad_C[i] += gc;
                max_grad = std::max(max_grad, std::abs(gc));
            }
        }

        // 7. SGD update (average gradient over batch)
        double inv_batch = 1.0 / static_cast<double>(batch.size());
        for (int idx = 0; idx < MAMBA_DIM_SQ; ++idx) {
            A_[idx] -= learning_rate_ * grad_A[idx] * inv_batch;
            B_[idx] -= learning_rate_ * grad_B[idx] * inv_batch;
        }
        for (int i = 0; i < MAMBA_DIM; ++i) {
            C_[i] -= learning_rate_ * grad_C[i] * inv_batch;
        }

        // 8. Decay learning rate
        learning_rate_ *= lr_decay_;
        ++epoch_;

        stats.loss     = total_loss * inv_batch;
        stats.max_grad = max_grad;
        stats.samples  = static_cast<int>(batch.size());
        return stats;
    }

    /**
     * @brief Convenience: train on a single sample. Returns its loss.
     */
    double train_step(const TrainingSample& sample) {
        auto batch = std::vector<TrainingSample>{sample};
        auto stats = train_batch(batch);
        return stats.loss;
    }

    /**
     * @brief Predict next state given current state and input.
     *
     * Uses the current A, B parameters: s_{t+1} = A·s + B·x
     * Does NOT modify gradients or training state.
     */
    std::array<double, 9> predict(const std::array<double, 9>& state,
                                  const std::array<double, 9>& input) const {
        std::array<double, 9> result{};
        for (int i = 0; i < MAMBA_DIM; ++i) {
            double sum = 0.0;
            for (int j = 0; j < MAMBA_DIM; ++j) {
                sum += A_[i * MAMBA_DIM + j] * state[j]
                     + B_[i * MAMBA_DIM + j] * input[j];
            }
            result[i] = sum;
        }
        return result;
    }

    /**
     * @brief Compute output from hidden state: y = C^T · s (dot product).
     */
    double compute_output(const std::array<double, 9>& state) const {
        double y = 0.0;
        for (int i = 0; i < MAMBA_DIM; ++i)
            y += C_[i] * state[i];
        return y;
    }

    // ── Parameter access ───────────────────────────────────────────────

    const std::array<double, MAMBA_DIM_SQ>& A() const { return A_; }
    const std::array<double, MAMBA_DIM_SQ>& B() const { return B_; }
    const std::array<double, MAMBA_DIM>&    C() const { return C_; }

    std::array<double, MAMBA_DIM_SQ>& A() { return A_; }
    std::array<double, MAMBA_DIM_SQ>& B() { return B_; }
    std::array<double, MAMBA_DIM>&    C() { return C_; }

    double learning_rate() const { return learning_rate_; }
    void   set_learning_rate(double lr) { learning_rate_ = lr; }
    int    epoch() const { return epoch_; }

    /// Number of nodes in the compute graph (fixed after construction).
    uint16_t graph_size() const { return graph_.size(); }

    // ── Triggers (for autonomous training) ─────────────────────────────

    /**
     * @brief Check if training should be triggered based on prediction error.
     *
     * Returns true if the moving average prediction error exceeds the
     * threshold, signaling that the SSM needs parameter updates.
     */
    bool should_train(double prediction_error) {
        error_ema_ = error_ema_alpha_ * prediction_error
                   + (1.0 - error_ema_alpha_) * error_ema_;
        return error_ema_ > error_threshold_;
    }

    void set_error_threshold(double t) { error_threshold_ = t; }
    double error_ema() const { return error_ema_; }

    // ── Gradient evaluation (for testing & diagnostics) ────────────────

    /// Result of evaluating gradients on a single sample.
    struct GradientResult {
        std::array<double, MAMBA_DIM_SQ> grad_A{};
        std::array<double, MAMBA_DIM_SQ> grad_B{};
        std::array<double, MAMBA_DIM>    grad_C{};
        double loss = 0.0;
    };

    /**
     * @brief Evaluate loss and gradients for a single sample WITHOUT
     *        updating parameters. Useful for gradient checking in tests.
     */
    GradientResult eval_gradient(const TrainingSample& sample) {
        // Set parameter leaf values
        for (int i = 0; i < MAMBA_DIM; ++i) {
            for (int j = 0; j < MAMBA_DIM; ++j) {
                graph_.set_value(a_ids_[i * MAMBA_DIM + j],
                                 {A_[i * MAMBA_DIM + j], 0.0});
                graph_.set_value(b_ids_[i * MAMBA_DIM + j],
                                 {B_[i * MAMBA_DIM + j], 0.0});
            }
        }
        for (int i = 0; i < MAMBA_DIM; ++i)
            graph_.set_value(c_ids_[i], {C_[i], 0.0});

        // Set data leaf values
        for (int i = 0; i < MAMBA_DIM; ++i) {
            graph_.set_value(s_ids_[i], {sample.state[i], 0.0});
            graph_.set_value(x_ids_[i], {sample.input[i], 0.0});
            graph_.set_value(neg_actual_ids_[i],
                             {-sample.next_state[i], 0.0});
        }

        graph_.zero_gradients();
        graph_.forward();
        graph_.backward(loss_id_);

        GradientResult result;
        result.loss = graph_.get_value(loss_id_).real();
        for (int i = 0; i < MAMBA_DIM_SQ; ++i) {
            result.grad_A[i] = graph_.get_gradient(a_ids_[i]).real();
            result.grad_B[i] = graph_.get_gradient(b_ids_[i]).real();
        }
        for (int i = 0; i < MAMBA_DIM; ++i)
            result.grad_C[i] = graph_.get_gradient(c_ids_[i]).real();

        return result;
    }

    /// Reset parameters to small random values (Xavier-like initialization).
    void reset_params(uint32_t seed = 42u) {
        init_params(seed);
    }

private:
    // ── Parameters ─────────────────────────────────────────────────────

    std::array<double, MAMBA_DIM_SQ> A_{};  ///< 9×9 state transition
    std::array<double, MAMBA_DIM_SQ> B_{};  ///< 9×9 input coupling
    std::array<double, MAMBA_DIM>    C_{};  ///< 9 output weights

    // ── Hyperparameters ────────────────────────────────────────────────

    double learning_rate_;
    double lr_decay_;
    int    epoch_ = 0;

    // ── Auto-training triggers ─────────────────────────────────────────

    double error_ema_       = 0.0;
    double error_ema_alpha_ = 0.1;
    double error_threshold_ = 0.01;

    // ── Compute graph (fixed topology) ─────────────────────────────────

    autodiff::StaticComputeGraph<8192> graph_;

    // Pre-allocated leaf node IDs
    std::array<uint16_t, MAMBA_DIM_SQ> a_ids_{};  ///< A[i][j] params
    std::array<uint16_t, MAMBA_DIM_SQ> b_ids_{};  ///< B[i][j] params
    std::array<uint16_t, MAMBA_DIM>    c_ids_{};   ///< C[i] params
    std::array<uint16_t, MAMBA_DIM>    s_ids_{};   ///< s_t[i] inputs
    std::array<uint16_t, MAMBA_DIM>    x_ids_{};   ///< x_t[i] inputs
    std::array<uint16_t, MAMBA_DIM>    neg_actual_ids_{}; ///< -s_{t+1}^actual

    // Internal node IDs (for verification only)
    uint16_t loss_id_ = 0;

    // ── Initialization ─────────────────────────────────────────────────

    void init_params(uint32_t seed = 42u) {
        // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
        // For 9×9: scale = sqrt(2/18) ≈ 0.333
        double scale = std::sqrt(2.0 / (MAMBA_DIM + MAMBA_DIM));

        // Simple LCG for deterministic initialization
        uint64_t rng = seed;
        auto next_rand = [&rng]() -> double {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            // Map to [-1, 1]
            return static_cast<double>(static_cast<int64_t>(rng >> 33))
                   / static_cast<double>(1LL << 31);
        };

        for (int i = 0; i < MAMBA_DIM_SQ; ++i)
            A_[i] = scale * next_rand();
        for (int i = 0; i < MAMBA_DIM_SQ; ++i)
            B_[i] = scale * next_rand();
        for (int i = 0; i < MAMBA_DIM; ++i)
            C_[i] = scale * next_rand();
    }

    // ── Graph construction (called once) ───────────────────────────────

    /**
     * @brief Build the fixed compute graph topology.
     *
     * Graph encodes: pred[i] = Σ_j A[i][j]*s[j] + Σ_j B[i][j]*x[j]
     *                loss = Σ_i |pred[i] - actual[i]|²
     *
     * Node layout:
     *   [0..80]    A param leaves (81)
     *   [81..161]  B param leaves (81)
     *   [162..170] C param leaves (9)
     *   [171..179] s_t input leaves (9)
     *   [180..188] x_t input leaves (9)
     *   [189..197] neg_actual leaves (9)
     *   [198..]    internal: mul_a, sum_a, mul_b, sum_b, pred, diff, sq, loss
     */
    void build_graph() {
        // Create parameter leaves
        for (int i = 0; i < MAMBA_DIM_SQ; ++i)
            a_ids_[i] = graph_.create_leaf({A_[i], 0.0});
        for (int i = 0; i < MAMBA_DIM_SQ; ++i)
            b_ids_[i] = graph_.create_leaf({B_[i], 0.0});
        for (int i = 0; i < MAMBA_DIM; ++i)
            c_ids_[i] = graph_.create_leaf({C_[i], 0.0});

        // Create input/target leaves (values set per sample)
        for (int i = 0; i < MAMBA_DIM; ++i)
            s_ids_[i] = graph_.create_leaf({0.0, 0.0});
        for (int i = 0; i < MAMBA_DIM; ++i)
            x_ids_[i] = graph_.create_leaf({0.0, 0.0});
        for (int i = 0; i < MAMBA_DIM; ++i)
            neg_actual_ids_[i] = graph_.create_leaf({0.0, 0.0});

        // Build SSM recurrence: pred[i] = Σ_j A[i][j]*s[j] + Σ_j B[i][j]*x[j]
        std::array<uint16_t, MAMBA_DIM> pred_ids;

        for (int i = 0; i < MAMBA_DIM; ++i) {
            // Compute A[i,:] · s = Σ_j A[i][j] * s[j]
            uint16_t sum_a = graph_.multiply(a_ids_[i * MAMBA_DIM], s_ids_[0]);
            for (int j = 1; j < MAMBA_DIM; ++j) {
                uint16_t prod = graph_.multiply(a_ids_[i * MAMBA_DIM + j], s_ids_[j]);
                sum_a = graph_.add(sum_a, prod);
            }

            // Compute B[i,:] · x = Σ_j B[i][j] * x[j]
            uint16_t sum_b = graph_.multiply(b_ids_[i * MAMBA_DIM], x_ids_[0]);
            for (int j = 1; j < MAMBA_DIM; ++j) {
                uint16_t prod = graph_.multiply(b_ids_[i * MAMBA_DIM + j], x_ids_[j]);
                sum_b = graph_.add(sum_b, prod);
            }

            // pred[i] = sum_a + sum_b
            pred_ids[i] = graph_.add(sum_a, sum_b);
        }

        // Compute loss: Σ_i |pred[i] - actual[i]|²
        // diff[i] = pred[i] + (-actual[i])
        uint16_t first_sq = 0;
        for (int i = 0; i < MAMBA_DIM; ++i) {
            uint16_t diff = graph_.add(pred_ids[i], neg_actual_ids_[i]);
            uint16_t sq   = graph_.squared_norm(diff);
            if (i == 0) {
                first_sq = sq;
            } else {
                first_sq = graph_.add(first_sq, sq);
            }
        }
        loss_id_ = first_sq;
    }
};

} // namespace nikola::trainers
