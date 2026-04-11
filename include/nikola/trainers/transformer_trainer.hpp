/**
 * @file include/nikola/trainers/transformer_trainer.hpp
 * @brief TransformerTrainer — NPT QKV weight training via StaticComputeGraph
 *
 * Second Bicameral Autonomous Trainer (BAT). Trains reduced 9D self-attention
 * QKV projection matrices using gradient descent through the StaticComputeGraph
 * autodiff engine.
 *
 * Architecture:
 *   - Reduced 9D parameter space matching the T⁹ manifold
 *   - W_Q ∈ ℝ⁹ˣ⁹ (query), W_K ∈ ℝ⁹ˣ⁹ (key), W_V ∈ ℝ⁹ˣ⁹ (value)
 *   - 2-position linear attention (no softmax — unavailable in graph ops)
 *   - Fixed graph topology built once at construction (1377 nodes)
 *   - Training loop: set_value → forward → backward → SGD update
 *   - Zero allocations during training (StaticComputeGraph reuse)
 *
 * Training objective (teacher forcing, 2-position cross-attention):
 *   q_i = W_Q · x_i,  k_j = W_K · x_j,  v_j = W_V · x_j
 *   score_ij = dot(q_i, k_j) / sqrt(9)
 *   out_i = Σ_j score_ij · v_j
 *   L = ||out_1 - y_1||² + ||out_2 - y_2||²
 *
 * The trained 9D projections map to the full NPT's WaveFunction-based attention
 * via the wave correlation mechanism in NeuroplasticTransformer::forward().
 *
 * @see docs/info/engineering/nikola_full.txt §5.2 (BAT — Transformer Trainer)
 * @see include/nikola/cognitive/neuroplastic_transformer.hpp
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

/// Dimensions of the reduced T⁹ manifold attention.
static constexpr int ATTN_DIM    = 9;
static constexpr int ATTN_DIM_SQ = ATTN_DIM * ATTN_DIM; // 81

/// Training sample: 2-position cross-attention with targets.
struct AttentionSample {
    std::array<double, 9> x1;  ///< Position 1 input (9D torus vector)
    std::array<double, 9> x2;  ///< Position 2 input (9D torus vector)
    std::array<double, 9> y1;  ///< Position 1 target output
    std::array<double, 9> y2;  ///< Position 2 target output
};

/// Training statistics for one epoch or batch.
struct AttentionTrainingStats {
    double loss      = 0.0;  ///< Mean loss over samples
    double max_grad  = 0.0;  ///< Max gradient magnitude
    int    samples   = 0;    ///< Number of samples processed
};

/**
 * @brief TransformerTrainer — trains 9D QKV projection matrices via gradient descent.
 *
 * Uses a pre-built StaticComputeGraph with 1377 nodes encoding 2-position
 * linear self-attention. The graph topology is fixed at construction.
 *
 * Linear attention (vs softmax):
 *   The StaticComputeGraph only supports ADD, MULTIPLY, SQUARED_NORM.
 *   Softmax requires exp/div which are unavailable. Linear attention uses
 *   raw scaled dot-product scores, sufficient for training the projections.
 *   Full wave-correlation attention happens at inference in the NPT.
 */
class TransformerTrainer {
public:
    explicit TransformerTrainer(double learning_rate = 0.0001,
                                double lr_decay      = 0.999)
        : learning_rate_(learning_rate)
        , lr_decay_(lr_decay)
    {
        init_params();
        build_graph();
    }

    // ── Training interface ─────────────────────────────────────────────

    /**
     * @brief Train on a batch of 2-position attention samples.
     *
     * Each sample provides (x1, x2) → (y1, y2) with teacher forcing.
     * Gradients accumulated over batch, then single SGD step.
     */
    AttentionTrainingStats train_batch(const std::vector<AttentionSample>& batch) {
        AttentionTrainingStats stats;
        if (batch.empty()) return stats;

        // Zero gradient accumulators
        std::array<double, ATTN_DIM_SQ> grad_Q{};
        std::array<double, ATTN_DIM_SQ> grad_K{};
        std::array<double, ATTN_DIM_SQ> grad_V{};

        double total_loss = 0.0;
        double max_grad   = 0.0;

        for (const auto& sample : batch) {
            // 1. Set parameter leaf values
            for (int idx = 0; idx < ATTN_DIM_SQ; ++idx) {
                graph_.set_value(q_ids_[idx], {Q_[idx], 0.0});
                graph_.set_value(k_ids_[idx], {K_[idx], 0.0});
                graph_.set_value(v_ids_[idx], {V_[idx], 0.0});
            }

            // 2. Set data leaf values
            for (int i = 0; i < ATTN_DIM; ++i) {
                graph_.set_value(x1_ids_[i], {sample.x1[i], 0.0});
                graph_.set_value(x2_ids_[i], {sample.x2[i], 0.0});
                graph_.set_value(neg_y1_ids_[i], {-sample.y1[i], 0.0});
                graph_.set_value(neg_y2_ids_[i], {-sample.y2[i], 0.0});
            }
            // Scale leaf stays at 1/sqrt(9) = 1/3
            graph_.set_value(scale_id_, {1.0 / 3.0, 0.0});

            // 3. Forward + backward
            graph_.zero_gradients();
            graph_.forward();
            graph_.backward(loss_id_);

            // 4. Accumulate
            double sample_loss = graph_.get_value(loss_id_).real();
            total_loss += sample_loss;

            for (int idx = 0; idx < ATTN_DIM_SQ; ++idx) {
                double gq = graph_.get_gradient(q_ids_[idx]).real();
                double gk = graph_.get_gradient(k_ids_[idx]).real();
                double gv = graph_.get_gradient(v_ids_[idx]).real();
                grad_Q[idx] += gq;
                grad_K[idx] += gk;
                grad_V[idx] += gv;
                max_grad = std::max(max_grad,
                    std::max({std::abs(gq), std::abs(gk), std::abs(gv)}));
            }
        }

        // 5. SGD update (average gradient over batch)
        double inv_batch = 1.0 / static_cast<double>(batch.size());
        for (int idx = 0; idx < ATTN_DIM_SQ; ++idx) {
            Q_[idx] -= learning_rate_ * grad_Q[idx] * inv_batch;
            K_[idx] -= learning_rate_ * grad_K[idx] * inv_batch;
            V_[idx] -= learning_rate_ * grad_V[idx] * inv_batch;
        }

        // 6. Decay learning rate
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
    double train_step(const AttentionSample& sample) {
        auto batch = std::vector<AttentionSample>{sample};
        auto stats = train_batch(batch);
        return stats.loss;
    }

    /**
     * @brief Predict outputs for 2-position linear attention.
     *
     * out_i = Σ_j (dot(W_Q·x_i, W_K·x_j)/3) · (W_V·x_j)
     * Uses current Q, K, V parameters directly (no graph).
     */
    std::pair<std::array<double, 9>, std::array<double, 9>>
    predict(const std::array<double, 9>& x1,
            const std::array<double, 9>& x2) const {

        // Project all inputs
        auto q1 = matvec(Q_, x1);
        auto q2 = matvec(Q_, x2);
        auto k1 = matvec(K_, x1);
        auto k2 = matvec(K_, x2);
        auto v1 = matvec(V_, x1);
        auto v2 = matvec(V_, x2);

        // Attention scores (scaled dot product)
        constexpr double inv_sqrt_d = 1.0 / 3.0; // 1/sqrt(9)
        double s11 = dot9(q1, k1) * inv_sqrt_d;
        double s12 = dot9(q1, k2) * inv_sqrt_d;
        double s21 = dot9(q2, k1) * inv_sqrt_d;
        double s22 = dot9(q2, k2) * inv_sqrt_d;

        // Attended outputs
        std::array<double, 9> y1{}, y2{};
        for (int i = 0; i < ATTN_DIM; ++i) {
            y1[i] = s11 * v1[i] + s12 * v2[i];
            y2[i] = s21 * v1[i] + s22 * v2[i];
        }
        return {y1, y2};
    }

    // ── Parameter access ───────────────────────────────────────────────

    const std::array<double, ATTN_DIM_SQ>& Q() const { return Q_; }
    const std::array<double, ATTN_DIM_SQ>& K() const { return K_; }
    const std::array<double, ATTN_DIM_SQ>& V() const { return V_; }

    std::array<double, ATTN_DIM_SQ>& Q() { return Q_; }
    std::array<double, ATTN_DIM_SQ>& K() { return K_; }
    std::array<double, ATTN_DIM_SQ>& V() { return V_; }

    double learning_rate() const { return learning_rate_; }
    void   set_learning_rate(double lr) { learning_rate_ = lr; }
    int    epoch() const { return epoch_; }

    /// Number of nodes in the compute graph (fixed after construction).
    uint16_t graph_size() const { return graph_.size(); }

    // ── Triggers (for autonomous training) ─────────────────────────────

    /**
     * @brief Check if training should be triggered based on output error.
     */
    bool should_train(double output_error) {
        error_ema_ = error_ema_alpha_ * output_error
                   + (1.0 - error_ema_alpha_) * error_ema_;
        return error_ema_ > error_threshold_;
    }

    void set_error_threshold(double t) { error_threshold_ = t; }
    double error_ema() const { return error_ema_; }

    // ── Gradient evaluation (for testing & diagnostics) ────────────────

    struct GradientResult {
        std::array<double, ATTN_DIM_SQ> grad_Q{};
        std::array<double, ATTN_DIM_SQ> grad_K{};
        std::array<double, ATTN_DIM_SQ> grad_V{};
        double loss = 0.0;
    };

    /**
     * @brief Evaluate loss and gradients without updating parameters.
     */
    GradientResult eval_gradient(const AttentionSample& sample) {
        for (int idx = 0; idx < ATTN_DIM_SQ; ++idx) {
            graph_.set_value(q_ids_[idx], {Q_[idx], 0.0});
            graph_.set_value(k_ids_[idx], {K_[idx], 0.0});
            graph_.set_value(v_ids_[idx], {V_[idx], 0.0});
        }
        for (int i = 0; i < ATTN_DIM; ++i) {
            graph_.set_value(x1_ids_[i], {sample.x1[i], 0.0});
            graph_.set_value(x2_ids_[i], {sample.x2[i], 0.0});
            graph_.set_value(neg_y1_ids_[i], {-sample.y1[i], 0.0});
            graph_.set_value(neg_y2_ids_[i], {-sample.y2[i], 0.0});
        }
        graph_.set_value(scale_id_, {1.0 / 3.0, 0.0});

        graph_.zero_gradients();
        graph_.forward();
        graph_.backward(loss_id_);

        GradientResult result;
        result.loss = graph_.get_value(loss_id_).real();
        for (int idx = 0; idx < ATTN_DIM_SQ; ++idx) {
            result.grad_Q[idx] = graph_.get_gradient(q_ids_[idx]).real();
            result.grad_K[idx] = graph_.get_gradient(k_ids_[idx]).real();
            result.grad_V[idx] = graph_.get_gradient(v_ids_[idx]).real();
        }
        return result;
    }

    /// Reset parameters to small random values.
    void reset_params(uint32_t seed = 42u) {
        init_params(seed);
    }

private:
    // ── Parameters ─────────────────────────────────────────────────────

    std::array<double, ATTN_DIM_SQ> Q_{};  ///< 9×9 query projection
    std::array<double, ATTN_DIM_SQ> K_{};  ///< 9×9 key projection
    std::array<double, ATTN_DIM_SQ> V_{};  ///< 9×9 value projection

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

    // Leaf node IDs
    std::array<uint16_t, ATTN_DIM_SQ> q_ids_{};
    std::array<uint16_t, ATTN_DIM_SQ> k_ids_{};
    std::array<uint16_t, ATTN_DIM_SQ> v_ids_{};
    std::array<uint16_t, ATTN_DIM>    x1_ids_{};
    std::array<uint16_t, ATTN_DIM>    x2_ids_{};
    std::array<uint16_t, ATTN_DIM>    neg_y1_ids_{};
    std::array<uint16_t, ATTN_DIM>    neg_y2_ids_{};
    uint16_t scale_id_ = 0;

    uint16_t loss_id_ = 0;

    // ── Math helpers ───────────────────────────────────────────────────

    static std::array<double, 9> matvec(const std::array<double, 81>& W,
                                         const std::array<double, 9>& x) {
        std::array<double, 9> result{};
        for (int i = 0; i < ATTN_DIM; ++i) {
            double sum = 0.0;
            for (int j = 0; j < ATTN_DIM; ++j)
                sum += W[i * ATTN_DIM + j] * x[j];
            result[i] = sum;
        }
        return result;
    }

    static double dot9(const std::array<double, 9>& a,
                       const std::array<double, 9>& b) {
        double sum = 0.0;
        for (int i = 0; i < ATTN_DIM; ++i)
            sum += a[i] * b[i];
        return sum;
    }

    // ── Initialization ─────────────────────────────────────────────────

    void init_params(uint32_t seed = 42u) {
        // Xavier: scale = sqrt(2 / (9 + 9)) ≈ 0.333
        double scale = std::sqrt(2.0 / (ATTN_DIM + ATTN_DIM));

        uint64_t rng = seed;
        auto next_rand = [&rng]() -> double {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<double>(static_cast<int64_t>(rng >> 33))
                   / static_cast<double>(1LL << 31);
        };

        for (int i = 0; i < ATTN_DIM_SQ; ++i) Q_[i] = scale * next_rand();
        for (int i = 0; i < ATTN_DIM_SQ; ++i) K_[i] = scale * next_rand();
        for (int i = 0; i < ATTN_DIM_SQ; ++i) V_[i] = scale * next_rand();
    }

    // ── Graph construction (called once) ───────────────────────────────

    /**
     * @brief Build the fixed 2-position linear attention graph.
     *
     * For each query position i ∈ {1,2}, for each key position j ∈ {1,2}:
     *   q_i = W_Q · x_i,  k_j = W_K · x_j,  v_j = W_V · x_j
     *   score_ij = dot(q_i, k_j) * (1/3)
     *   out_i = Σ_j score_ij · v_j
     *   loss += ||out_i - y_i||²
     *
     * Total: 1377 nodes (280 leaves + 1097 internal).
     */
    void build_graph() {
        // ── Leaf nodes ─────────────────────────────────────────────
        for (int i = 0; i < ATTN_DIM_SQ; ++i)
            q_ids_[i] = graph_.create_leaf({Q_[i], 0.0});
        for (int i = 0; i < ATTN_DIM_SQ; ++i)
            k_ids_[i] = graph_.create_leaf({K_[i], 0.0});
        for (int i = 0; i < ATTN_DIM_SQ; ++i)
            v_ids_[i] = graph_.create_leaf({V_[i], 0.0});

        for (int i = 0; i < ATTN_DIM; ++i)
            x1_ids_[i] = graph_.create_leaf({0.0, 0.0});
        for (int i = 0; i < ATTN_DIM; ++i)
            x2_ids_[i] = graph_.create_leaf({0.0, 0.0});
        for (int i = 0; i < ATTN_DIM; ++i)
            neg_y1_ids_[i] = graph_.create_leaf({0.0, 0.0});
        for (int i = 0; i < ATTN_DIM; ++i)
            neg_y2_ids_[i] = graph_.create_leaf({0.0, 0.0});

        scale_id_ = graph_.create_leaf({1.0 / 3.0, 0.0});

        // ── Helper: build W · x (9D output) ───────────────────────
        auto build_proj = [&](const std::array<uint16_t, ATTN_DIM_SQ>& w_ids,
                              const std::array<uint16_t, ATTN_DIM>& x_ids)
                -> std::array<uint16_t, ATTN_DIM> {
            std::array<uint16_t, ATTN_DIM> out;
            for (int i = 0; i < ATTN_DIM; ++i) {
                uint16_t acc = graph_.multiply(
                    w_ids[i * ATTN_DIM], x_ids[0]);
                for (int j = 1; j < ATTN_DIM; ++j) {
                    uint16_t prod = graph_.multiply(
                        w_ids[i * ATTN_DIM + j], x_ids[j]);
                    acc = graph_.add(acc, prod);
                }
                out[i] = acc;
            }
            return out;
        };

        // ── Helper: dot product of two 9D node vectors → scalar ───
        auto build_dot = [&](const std::array<uint16_t, ATTN_DIM>& a,
                             const std::array<uint16_t, ATTN_DIM>& b)
                -> uint16_t {
            uint16_t acc = graph_.multiply(a[0], b[0]);
            for (int i = 1; i < ATTN_DIM; ++i) {
                uint16_t prod = graph_.multiply(a[i], b[i]);
                acc = graph_.add(acc, prod);
            }
            return acc;
        };

        // ── Projections ───────────────────────────────────────────
        auto q1 = build_proj(q_ids_, x1_ids_);  // W_Q · x1
        auto q2 = build_proj(q_ids_, x2_ids_);  // W_Q · x2
        auto k1 = build_proj(k_ids_, x1_ids_);  // W_K · x1
        auto k2 = build_proj(k_ids_, x2_ids_);  // W_K · x2
        auto v1 = build_proj(v_ids_, x1_ids_);  // W_V · x1
        auto v2 = build_proj(v_ids_, x2_ids_);  // W_V · x2

        // ── Attention scores (scaled dot products) ────────────────
        uint16_t score_11 = graph_.multiply(build_dot(q1, k1), scale_id_);
        uint16_t score_12 = graph_.multiply(build_dot(q1, k2), scale_id_);
        uint16_t score_21 = graph_.multiply(build_dot(q2, k1), scale_id_);
        uint16_t score_22 = graph_.multiply(build_dot(q2, k2), scale_id_);

        // ── Attended outputs ──────────────────────────────────────
        // out1[i] = score_11 * v1[i] + score_12 * v2[i]
        // out2[i] = score_21 * v1[i] + score_22 * v2[i]
        std::array<uint16_t, ATTN_DIM> out1, out2;
        for (int i = 0; i < ATTN_DIM; ++i) {
            uint16_t sv11 = graph_.multiply(score_11, v1[i]);
            uint16_t sv12 = graph_.multiply(score_12, v2[i]);
            out1[i] = graph_.add(sv11, sv12);

            uint16_t sv21 = graph_.multiply(score_21, v1[i]);
            uint16_t sv22 = graph_.multiply(score_22, v2[i]);
            out2[i] = graph_.add(sv21, sv22);
        }

        // ── Loss: Σ_i (|out1_i - y1_i|² + |out2_i - y2_i|²) ────
        uint16_t loss_acc = 0;
        bool first = true;
        for (int i = 0; i < ATTN_DIM; ++i) {
            uint16_t diff1 = graph_.add(out1[i], neg_y1_ids_[i]);
            uint16_t sq1   = graph_.squared_norm(diff1);
            uint16_t diff2 = graph_.add(out2[i], neg_y2_ids_[i]);
            uint16_t sq2   = graph_.squared_norm(diff2);

            uint16_t pair_sum = graph_.add(sq1, sq2);
            if (first) {
                loss_acc = pair_sum;
                first = false;
            } else {
                loss_acc = graph_.add(loss_acc, pair_sum);
            }
        }
        loss_id_ = loss_acc;
    }
};

} // namespace nikola::trainers
