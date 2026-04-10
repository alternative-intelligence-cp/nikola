/**
 * @file include/nikola/core/autodiff.hpp
 * @brief NikolaAutodiff — Complex-valued tape-based automatic differentiation
 *
 * Implements Wirtinger calculus for complex derivatives needed by wave mechanics
 * training. Supports: add, multiply, matrix-vector, squared_norm, UFIE step.
 *
 * Wirtinger derivatives for complex z = x + iy:
 *   ∂/∂z   = ½(∂/∂x − i∂/∂y)
 *   ∂/∂z̄  = ½(∂/∂x + i∂/∂y)
 *
 * For real-valued loss L(z, z̄), the gradient update is:
 *   ∂L/∂z̄ (the Wirtinger conjugate derivative)
 *
 * @see docs/info/gemini/compilation/part_4_of_9.txt lines 1187-1500
 */
#pragma once

#include <complex>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <Eigen/Dense>

namespace nikola::autodiff {

/// Operation types for the compute tape
enum class OpType : uint8_t {
    LEAF,           ///< Input variable or parameter
    ADD,            ///< z = x + y
    MULTIPLY,       ///< z = x * y (complex Wirtinger product rule)
    MATVEC,         ///< y_i = A[i,:] · x (single output dimension of Ax)
    SQUARED_NORM,   ///< L = |x|² (real-valued output)
    UFIE_STEP       ///< Ψ_{t+1} = (1 - iHdt - iβ|Ψ|²dt)Ψ_t
};

/// A node on the computation tape
struct TapeNode {
    std::complex<double> value{0.0, 0.0};
    std::complex<double> gradient{0.0, 0.0};
    OpType op = OpType::LEAF;
    size_t parent_a = SIZE_MAX;     ///< First parent index (or sentinel)
    size_t parent_b = SIZE_MAX;     ///< Second parent / output_dim / scalar_idx
    uint16_t op_data_idx = 0xFFFF;  ///< Index into matrix pool
};

/**
 * @brief Tape-based automatic differentiation with complex Wirtinger calculus
 *
 * Usage:
 *   NikolaAutodiff tape;
 *   auto x = tape.create_variable({1.0, 2.0});
 *   auto y = tape.create_variable({3.0, 0.5});
 *   auto z = tape.multiply(x, y);
 *   auto loss = tape.squared_norm(z);
 *   tape.backward(loss);
 *   auto grad_x = tape.get_gradient(x);  // Wirtinger ∂L/∂x̄
 */
class NikolaAutodiff {
public:
    /// Create a leaf variable (input or trainable parameter)
    size_t create_variable(std::complex<double> value) {
        TapeNode node;
        node.value = value;
        node.op = OpType::LEAF;
        tape_.push_back(node);
        return tape_.size() - 1;
    }

    /// z = x + y
    size_t add(size_t x_id, size_t y_id) {
        TapeNode node;
        node.value = tape_[x_id].value + tape_[y_id].value;
        node.op = OpType::ADD;
        node.parent_a = x_id;
        node.parent_b = y_id;
        tape_.push_back(node);
        return tape_.size() - 1;
    }

    /// z = x * y  (Wirtinger: ∂(xy)/∂x̄ = conj(y), ∂(xy)/∂ȳ = conj(x))
    size_t multiply(size_t x_id, size_t y_id) {
        TapeNode node;
        node.value = tape_[x_id].value * tape_[y_id].value;
        node.op = OpType::MULTIPLY;
        node.parent_a = x_id;
        node.parent_b = y_id;
        tape_.push_back(node);
        return tape_.size() - 1;
    }

    /**
     * @brief Single output dimension of matrix-vector: y[out_dim] = A[out_dim,:] · x
     * @param A      Complex matrix
     * @param x_id   Node id of scalar input
     * @param out_dim Which row of A to dot with x
     * @return Node id of the scalar output
     *
     * For multi-dimensional input, call once per output dimension.
     * Gradient: ∂L/∂x += conj(A[out_dim, 0]) * ∂L/∂y
     */
    size_t matvec(const Eigen::MatrixXcd& A, size_t x_id, int out_dim) {
        uint16_t mat_idx = store_matrix(A);

        TapeNode node;
        node.value = A(out_dim, 0) * tape_[x_id].value;
        node.op = OpType::MATVEC;
        node.parent_a = x_id;
        node.parent_b = static_cast<size_t>(out_dim);
        node.op_data_idx = mat_idx;
        tape_.push_back(node);
        return tape_.size() - 1;
    }

    /// L = |x|²   (Wirtinger: ∂|x|²/∂x̄ = 2·conj(x), but loss is real so
    ///              the conjugate derivative = x, giving update direction conj(x))
    size_t squared_norm(size_t x_id) {
        TapeNode node;
        node.value = {std::norm(tape_[x_id].value), 0.0};
        node.op = OpType::SQUARED_NORM;
        node.parent_a = x_id;
        tape_.push_back(node);
        return tape_.size() - 1;
    }

    /**
     * @brief UFIE wave propagation: Ψ_{t+1} = (1 − iH₀dt − iβ|Ψ|²dt)·Ψ_t
     *
     * Linearized Schrödinger + nonlinear soliton self-interaction.
     * Backward: total derivative via Wirtinger product + chain rule.
     */
    size_t ufie_step(size_t psi_id, const Eigen::MatrixXcd& H,
                     double dt, double beta = 0.1) {
        uint16_t mat_idx = store_matrix(H);
        uint16_t scalar_idx = store_scalars(dt, beta);

        std::complex<double> psi = tape_[psi_id].value;
        std::complex<double> i_unit(0.0, 1.0);
        std::complex<double> linear = 1.0 - i_unit * H(0, 0) * dt;
        double psi_norm_sq = std::norm(psi);
        std::complex<double> nonlinear = -i_unit * beta * psi_norm_sq * dt;

        TapeNode node;
        node.value = (linear + nonlinear) * psi;
        node.op = OpType::UFIE_STEP;
        node.parent_a = psi_id;
        node.parent_b = static_cast<size_t>(scalar_idx);
        node.op_data_idx = mat_idx;
        tape_.push_back(node);
        return tape_.size() - 1;
    }

    /// Reverse-mode backward pass from loss node
    void backward(size_t loss_id) {
        tape_[loss_id].gradient = {1.0, 0.0};

        for (int64_t i = static_cast<int64_t>(loss_id); i >= 0; --i) {
            const auto& node = tape_[i];
            std::complex<double> grad = node.gradient;
            if (std::abs(grad) < 1e-30) continue;

            switch (node.op) {
            case OpType::LEAF:
                break;

            case OpType::ADD:
                tape_[node.parent_a].gradient += grad;
                tape_[node.parent_b].gradient += grad;
                break;

            case OpType::MULTIPLY: {
                auto x_val = tape_[node.parent_a].value;
                auto y_val = tape_[node.parent_b].value;
                tape_[node.parent_a].gradient += grad * std::conj(y_val);
                tape_[node.parent_b].gradient += grad * std::conj(x_val);
                break;
            }

            case OpType::MATVEC: {
                int out_dim = static_cast<int>(node.parent_b);
                const auto& A = matrices_[node.op_data_idx];
                // Use parent_a for the actual input node
                // We need to find parent_a — but parent_b is overloaded as out_dim
                // parent_a is the x input node
                tape_[node.parent_a].gradient += grad * std::conj(A(out_dim, 0));
                break;
            }

            case OpType::SQUARED_NORM: {
                auto x_val = tape_[node.parent_a].value;
                tape_[node.parent_a].gradient += grad * 2.0 * x_val;
                break;
            }

            case OpType::UFIE_STEP: {
                uint16_t sc_idx = static_cast<uint16_t>(node.parent_b);
                double dt_val = scalars_[sc_idx];
                double beta_val = scalars_[sc_idx + 1];
                const auto& H = matrices_[node.op_data_idx];

                auto psi = tape_[node.parent_a].value;
                std::complex<double> i_unit(0.0, 1.0);
                std::complex<double> linear = 1.0 - i_unit * H(0, 0) * dt_val;
                double psi_norm_sq = std::norm(psi);

                // Full backward: linear (conj(M)) + nonlinear correction
                auto conj_M = std::conj(linear)
                    + i_unit * beta_val * psi_norm_sq * dt_val;
                auto linear_contrib = grad * conj_M;
                double nl_scalar = 2.0 * beta_val * dt_val
                    * std::imag(std::conj(grad) * psi);
                tape_[node.parent_a].gradient += linear_contrib + nl_scalar * psi;
                break;
            }
            } // switch
        }
    }

    // ── Accessors ──────────────────────────────────────────────────────

    std::complex<double> get_value(size_t id) const { return tape_[id].value; }
    std::complex<double> get_gradient(size_t id) const { return tape_[id].gradient; }
    size_t size() const { return tape_.size(); }

    void clear() {
        tape_.clear();
        matrices_.clear();
        scalars_.clear();
    }

    // ── Checkpoint support ─────────────────────────────────────────────

    /// Number of nodes on tape
    size_t get_tape_size() const { return tape_.size(); }

    /// Restore values from a checkpoint snapshot
    void restore_values(const std::vector<std::complex<double>>& values,
                        size_t count) {
        for (size_t i = 0; i < count && i < tape_.size(); ++i) {
            tape_[i].value = values[i];
            tape_[i].gradient = {0.0, 0.0};
        }
    }

    /// Discard tape nodes before a position (keep structure from pos onward)
    void clear_before(size_t /*pos*/) {
        // For checkpointing: zero out gradients of early nodes
        // We keep nodes to preserve parent indices but free backward data
        for (size_t i = 0; i < tape_.size(); ++i) {
            tape_[i].gradient = {0.0, 0.0};
        }
    }

private:
    std::vector<TapeNode> tape_;
    std::vector<Eigen::MatrixXcd> matrices_;
    std::vector<double> scalars_;

    uint16_t store_matrix(const Eigen::MatrixXcd& m) {
        uint16_t idx = static_cast<uint16_t>(matrices_.size());
        matrices_.push_back(m);
        return idx;
    }

    uint16_t store_scalars(double a, double b) {
        uint16_t idx = static_cast<uint16_t>(scalars_.size());
        scalars_.push_back(a);
        scalars_.push_back(b);
        return idx;
    }
};

} // namespace nikola::autodiff
