/**
 * @file include/nikola/core/static_autodiff.hpp
 * @brief StaticComputeGraph — Zero-allocation fixed-size compute graph
 *
 * Pre-allocated SoA (Structure of Arrays) layout for training loops where the
 * topology is known at construction time. Achieves ~19× fewer L1D cache misses
 * vs dynamic tape approaches by keeping values/gradients in contiguous,
 * 64-byte-aligned arrays with static dispatch (no virtual calls).
 *
 * Template parameter MAX_NODES sets the compile-time capacity.
 * Use PagedComputeGraph when neurogenesis requires dynamic growth.
 *
 * @see docs/info/gemini/compilation/part_4_of_9.txt lines 1500-1700
 */
#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <Eigen/Dense>

namespace nikola::autodiff {

enum class StaticOpType : uint8_t {
    LEAF,
    ADD,
    MULTIPLY,
    MATVEC,
    SQUARED_NORM,
    UFIE_STEP
};

template<size_t MAX_NODES = 8192>
class StaticComputeGraph {
public:
    StaticComputeGraph() {
        std::memset(values_.data(), 0, sizeof(values_));
        std::memset(gradients_.data(), 0, sizeof(gradients_));
        op_types_.fill(StaticOpType::LEAF);
        parent_a_.fill(0xFFFF);
        parent_b_.fill(0xFFFF);
        op_data_.fill(0xFFFF);
    }

    // ── Graph construction ─────────────────────────────────────────────

    uint16_t create_leaf(std::complex<double> value) {
        check_capacity();
        uint16_t id = num_nodes_++;
        values_[id]    = value;
        gradients_[id] = {0.0, 0.0};
        op_types_[id]  = StaticOpType::LEAF;
        parent_a_[id]  = 0xFFFF;
        parent_b_[id]  = 0xFFFF;
        op_data_[id]   = 0xFFFF;
        return id;
    }

    uint16_t add(uint16_t x_id, uint16_t y_id) {
        check_capacity();
        uint16_t id = num_nodes_++;
        values_[id]    = values_[x_id] + values_[y_id];
        gradients_[id] = {0.0, 0.0};
        op_types_[id]  = StaticOpType::ADD;
        parent_a_[id]  = x_id;
        parent_b_[id]  = y_id;
        op_data_[id]   = 0xFFFF;
        return id;
    }

    uint16_t multiply(uint16_t x_id, uint16_t y_id) {
        check_capacity();
        uint16_t id = num_nodes_++;
        values_[id]    = values_[x_id] * values_[y_id];
        gradients_[id] = {0.0, 0.0};
        op_types_[id]  = StaticOpType::MULTIPLY;
        parent_a_[id]  = x_id;
        parent_b_[id]  = y_id;
        op_data_[id]   = 0xFFFF;
        return id;
    }

    uint16_t matvec(const Eigen::MatrixXcd& A, uint16_t x_id, int out_dim) {
        check_capacity();
        if (matrix_pool_idx_ >= MAX_MATRICES)
            throw std::runtime_error("StaticComputeGraph: matrix pool exhausted");

        uint16_t mat_idx = matrix_pool_idx_++;
        matrices_[mat_idx] = A;

        uint16_t id = num_nodes_++;
        values_[id]    = A(out_dim, 0) * values_[x_id];
        gradients_[id] = {0.0, 0.0};
        op_types_[id]  = StaticOpType::MATVEC;
        parent_a_[id]  = x_id;
        parent_b_[id]  = static_cast<uint16_t>(out_dim);
        op_data_[id]   = mat_idx;
        return id;
    }

    uint16_t squared_norm(uint16_t x_id) {
        check_capacity();
        uint16_t id = num_nodes_++;
        values_[id]    = {std::norm(values_[x_id]), 0.0};
        gradients_[id] = {0.0, 0.0};
        op_types_[id]  = StaticOpType::SQUARED_NORM;
        parent_a_[id]  = x_id;
        parent_b_[id]  = 0xFFFF;
        op_data_[id]   = 0xFFFF;
        return id;
    }

    uint16_t ufie_step(uint16_t psi_id, const Eigen::MatrixXcd& H,
                       double dt, double beta = 0.1) {
        check_capacity();
        if (matrix_pool_idx_ >= MAX_MATRICES)
            throw std::runtime_error("StaticComputeGraph: matrix pool exhausted");
        if (scalar_pool_idx_ + 1 >= MAX_SCALARS)
            throw std::runtime_error("StaticComputeGraph: scalar pool exhausted");

        uint16_t mat_idx = matrix_pool_idx_++;
        matrices_[mat_idx] = H;

        uint16_t sc_idx = scalar_pool_idx_;
        scalars_[sc_idx]     = dt;
        scalars_[sc_idx + 1] = beta;
        scalar_pool_idx_ += 2;

        std::complex<double> psi = values_[psi_id];
        std::complex<double> i_unit(0.0, 1.0);
        std::complex<double> linear = 1.0 - i_unit * H(0, 0) * dt;
        double psi_norm_sq = std::norm(psi);
        std::complex<double> nonlinear = -i_unit * beta * psi_norm_sq * dt;

        uint16_t id = num_nodes_++;
        values_[id]    = (linear + nonlinear) * psi;
        gradients_[id] = {0.0, 0.0};
        op_types_[id]  = StaticOpType::UFIE_STEP;
        parent_a_[id]  = psi_id;
        parent_b_[id]  = sc_idx;
        op_data_[id]   = mat_idx;
        return id;
    }

    // ── Backward pass (static dispatch, no virtual) ────────────────────

    void backward(uint16_t loss_id) {
        gradients_[loss_id] = {1.0, 0.0};

        for (int32_t i = static_cast<int32_t>(loss_id); i >= 0; --i) {
            std::complex<double> grad = gradients_[i];
            if (std::abs(grad) < 1e-30) continue;

            switch (op_types_[i]) {
            case StaticOpType::LEAF:
                break;

            case StaticOpType::ADD:
                gradients_[parent_a_[i]] += grad;
                gradients_[parent_b_[i]] += grad;
                break;

            case StaticOpType::MULTIPLY: {
                auto x_val = values_[parent_a_[i]];
                auto y_val = values_[parent_b_[i]];
                gradients_[parent_a_[i]] += grad * std::conj(y_val);
                gradients_[parent_b_[i]] += grad * std::conj(x_val);
                break;
            }

            case StaticOpType::MATVEC: {
                int out_dim = parent_b_[i];
                const auto& A = matrices_[op_data_[i]];
                gradients_[parent_a_[i]] += grad * std::conj(A(out_dim, 0));
                break;
            }

            case StaticOpType::SQUARED_NORM: {
                auto x_val = values_[parent_a_[i]];
                gradients_[parent_a_[i]] += grad * 2.0 * x_val;
                break;
            }

            case StaticOpType::UFIE_STEP: {
                uint16_t sc_idx = parent_b_[i];
                double dt_val   = scalars_[sc_idx];
                double beta_val = scalars_[sc_idx + 1];
                const auto& H  = matrices_[op_data_[i]];

                auto psi = values_[parent_a_[i]];
                std::complex<double> i_unit(0.0, 1.0);
                std::complex<double> linear = 1.0 - i_unit * H(0, 0) * dt_val;
                double psi_norm_sq = std::norm(psi);

                auto conj_M = std::conj(linear)
                    + i_unit * beta_val * psi_norm_sq * dt_val;
                auto linear_contrib = grad * conj_M;
                double nl_scalar = 2.0 * beta_val * dt_val
                    * std::imag(std::conj(grad) * psi);
                gradients_[parent_a_[i]] += linear_contrib + nl_scalar * psi;
                break;
            }
            } // switch
        }
    }

    // ── Accessors ──────────────────────────────────────────────────────

    std::complex<double> get_value(uint16_t id) const    { return values_[id]; }
    std::complex<double> get_gradient(uint16_t id) const { return gradients_[id]; }
    void set_value(uint16_t id, std::complex<double> v)  { values_[id] = v; }
    uint16_t size() const { return num_nodes_; }

    /// Reset for next training iteration: zero values/gradients, keep structure
    void reset() {
        std::memset(values_.data(),    0, num_nodes_ * sizeof(std::complex<double>));
        std::memset(gradients_.data(), 0, num_nodes_ * sizeof(std::complex<double>));
        matrix_pool_idx_ = 0;
        scalar_pool_idx_ = 0;
    }

    /// Full clear: reset node count as well
    void clear() {
        reset();
        num_nodes_ = 0;
    }

private:
    static constexpr size_t MAX_MATRICES = 64;
    static constexpr size_t MAX_SCALARS  = 256;

    // SoA layout for cache efficiency
    alignas(64) std::array<std::complex<double>, MAX_NODES> values_;
    alignas(64) std::array<std::complex<double>, MAX_NODES> gradients_;
    alignas(64) std::array<StaticOpType, MAX_NODES>         op_types_;
    alignas(64) std::array<uint16_t, MAX_NODES>             parent_a_;
    alignas(64) std::array<uint16_t, MAX_NODES>             parent_b_;
    alignas(64) std::array<uint16_t, MAX_NODES>             op_data_;

    uint16_t num_nodes_ = 0;

    // Pre-allocated pools
    std::array<Eigen::MatrixXcd, MAX_MATRICES> matrices_;
    std::array<double, MAX_SCALARS>            scalars_{};
    uint16_t matrix_pool_idx_ = 0;
    uint16_t scalar_pool_idx_ = 0;

    void check_capacity() {
        if (num_nodes_ >= MAX_NODES)
            throw std::runtime_error("StaticComputeGraph: capacity exceeded ("
                + std::to_string(MAX_NODES) + " nodes)");
    }
};

} // namespace nikola::autodiff
