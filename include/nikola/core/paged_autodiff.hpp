/**
 * @file include/nikola/core/paged_autodiff.hpp
 * @brief PagedComputeGraph — Dynamic-growth compute graph for neurogenesis
 *
 * Solves TRN-01: StaticComputeGraph's fixed MAX_NODES cannot accommodate
 * neurogenesis during Dream-Weave training cycles. This paged variant:
 *
 *   - Allocates 4096-node pages on demand (each ~128 KB, fits in L2 cache)
 *   - Pointer-stable: page addresses never move (vector<unique_ptr<Page>>)
 *   - Same API as StaticComputeGraph — drop-in replacement
 *   - Global ID indexing: page_idx = id / 4096, offset = id % 4096
 *
 * @see docs/info/gemini/compilation/part_4_of_9.txt lines 2182-2688
 */
#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <memory>
#include <vector>
#include <stdexcept>
#include <Eigen/Dense>

namespace nikola::autodiff {

/// Operation type — shared enum with static graph for interop
enum class PagedOpType : uint8_t {
    LEAF,
    ADD,
    MULTIPLY,
    MATVEC,
    SQUARED_NORM,
    UFIE_STEP
};

/// SoA page holding PAGE_SIZE compute nodes
template<size_t PAGE_SIZE = 4096>
struct ComputePage {
    alignas(64) std::array<std::complex<double>, PAGE_SIZE> values;
    alignas(64) std::array<std::complex<double>, PAGE_SIZE> gradients;
    alignas(64) std::array<PagedOpType, PAGE_SIZE>          op_types;
    alignas(64) std::array<uint32_t, PAGE_SIZE>             parent_a;
    alignas(64) std::array<uint32_t, PAGE_SIZE>             parent_b;
    alignas(64) std::array<uint16_t, PAGE_SIZE>             op_data_idx;

    ComputePage() {
        values.fill({0.0, 0.0});
        gradients.fill({0.0, 0.0});
        op_types.fill(PagedOpType::LEAF);
        parent_a.fill(0xFFFFFFFF);
        parent_b.fill(0xFFFFFFFF);
        op_data_idx.fill(0xFFFF);
    }
};

class PagedComputeGraph {
public:
    static constexpr size_t PAGE_SIZE = 4096;

    PagedComputeGraph() {
        grow();   // Start with one page
        matrices_.reserve(64);
        scalars_.reserve(256);
    }

    // ── Graph construction ─────────────────────────────────────────────

    uint32_t create_leaf(std::complex<double> value) {
        if (num_nodes_ == capacity_) grow();
        uint32_t id = num_nodes_++;
        auto [pi, off] = resolve(id);
        auto& p = *pages_[pi];
        p.values[off]   = value;
        p.gradients[off] = {0.0, 0.0};
        p.op_types[off]  = PagedOpType::LEAF;
        return id;
    }

    uint32_t add(uint32_t x_id, uint32_t y_id) {
        if (num_nodes_ == capacity_) grow();
        uint32_t id = num_nodes_++;
        auto [pi, off] = resolve(id);
        auto& p = *pages_[pi];
        p.values[off]   = get_value(x_id) + get_value(y_id);
        p.gradients[off] = {0.0, 0.0};
        p.op_types[off]  = PagedOpType::ADD;
        p.parent_a[off]  = x_id;
        p.parent_b[off]  = y_id;
        return id;
    }

    uint32_t multiply(uint32_t x_id, uint32_t y_id) {
        if (num_nodes_ == capacity_) grow();
        uint32_t id = num_nodes_++;
        auto [pi, off] = resolve(id);
        auto& p = *pages_[pi];
        p.values[off]   = get_value(x_id) * get_value(y_id);
        p.gradients[off] = {0.0, 0.0};
        p.op_types[off]  = PagedOpType::MULTIPLY;
        p.parent_a[off]  = x_id;
        p.parent_b[off]  = y_id;
        return id;
    }

    uint32_t matvec(const Eigen::MatrixXcd& A, uint32_t x_id, int out_dim) {
        if (num_nodes_ == capacity_) grow();
        uint16_t mat_idx = store_matrix(A);

        uint32_t id = num_nodes_++;
        auto [pi, off] = resolve(id);
        auto& p = *pages_[pi];
        p.values[off]      = A(out_dim, 0) * get_value(x_id);
        p.gradients[off]   = {0.0, 0.0};
        p.op_types[off]    = PagedOpType::MATVEC;
        p.parent_a[off]    = x_id;
        p.parent_b[off]    = static_cast<uint32_t>(out_dim);
        p.op_data_idx[off] = mat_idx;
        return id;
    }

    uint32_t squared_norm(uint32_t x_id) {
        if (num_nodes_ == capacity_) grow();
        uint32_t id = num_nodes_++;
        auto [pi, off] = resolve(id);
        auto& p = *pages_[pi];
        auto x_val = get_value(x_id);
        p.values[off]   = {std::norm(x_val), 0.0};
        p.gradients[off] = {0.0, 0.0};
        p.op_types[off]  = PagedOpType::SQUARED_NORM;
        p.parent_a[off]  = x_id;
        p.parent_b[off]  = 0xFFFFFFFF;
        return id;
    }

    uint32_t ufie_step(uint32_t psi_id, const Eigen::MatrixXcd& H,
                       double dt, double beta = 0.1) {
        if (num_nodes_ == capacity_) grow();
        uint16_t mat_idx = store_matrix(H);
        uint16_t sc_idx  = store_scalars(dt, beta);

        auto psi = get_value(psi_id);
        std::complex<double> i_unit(0.0, 1.0);
        auto linear = 1.0 - i_unit * H(0, 0) * dt;
        double psi_norm_sq = std::norm(psi);
        auto nonlinear = -i_unit * beta * psi_norm_sq * dt;

        uint32_t id = num_nodes_++;
        auto [pi, off] = resolve(id);
        auto& p = *pages_[pi];
        p.values[off]      = (linear + nonlinear) * psi;
        p.gradients[off]   = {0.0, 0.0};
        p.op_types[off]    = PagedOpType::UFIE_STEP;
        p.parent_a[off]    = psi_id;
        p.parent_b[off]    = static_cast<uint32_t>(sc_idx);
        p.op_data_idx[off] = mat_idx;
        return id;
    }

    // ── Backward pass ──────────────────────────────────────────────────

    void backward(uint32_t loss_id) {
        {
            auto [pi, off] = resolve(loss_id);
            pages_[pi]->gradients[off] = {1.0, 0.0};
        }

        for (int64_t i = static_cast<int64_t>(loss_id); i >= 0; --i) {
            auto [pi, off] = resolve(static_cast<uint32_t>(i));
            auto& page = *pages_[pi];

            std::complex<double> grad = page.gradients[off];
            if (std::abs(grad) < 1e-30) continue;

            switch (page.op_types[off]) {
            case PagedOpType::LEAF:
                break;

            case PagedOpType::ADD: {
                auto [xa, xo] = resolve(page.parent_a[off]);
                auto [ya, yo] = resolve(page.parent_b[off]);
                pages_[xa]->gradients[xo] += grad;
                pages_[ya]->gradients[yo] += grad;
                break;
            }

            case PagedOpType::MULTIPLY: {
                auto x_val = get_value(page.parent_a[off]);
                auto y_val = get_value(page.parent_b[off]);
                auto [xa, xo] = resolve(page.parent_a[off]);
                auto [ya, yo] = resolve(page.parent_b[off]);
                pages_[xa]->gradients[xo] += grad * std::conj(y_val);
                pages_[ya]->gradients[yo] += grad * std::conj(x_val);
                break;
            }

            case PagedOpType::MATVEC: {
                int out_dim = static_cast<int>(page.parent_b[off]);
                const auto& A = matrices_[page.op_data_idx[off]];
                auto [xa, xo] = resolve(page.parent_a[off]);
                pages_[xa]->gradients[xo] += grad * std::conj(A(out_dim, 0));
                break;
            }

            case PagedOpType::SQUARED_NORM: {
                auto x_val = get_value(page.parent_a[off]);
                auto [xa, xo] = resolve(page.parent_a[off]);
                pages_[xa]->gradients[xo] += grad * 2.0 * x_val;
                break;
            }

            case PagedOpType::UFIE_STEP: {
                uint16_t sc_idx = static_cast<uint16_t>(page.parent_b[off]);
                double dt_val   = scalars_[sc_idx];
                double beta_val = scalars_[sc_idx + 1];
                const auto& H  = matrices_[page.op_data_idx[off]];

                auto psi = get_value(page.parent_a[off]);
                std::complex<double> i_unit(0.0, 1.0);
                auto linear = 1.0 - i_unit * H(0, 0) * dt_val;
                double psi_norm_sq = std::norm(psi);

                auto conj_M = std::conj(linear)
                    + i_unit * beta_val * psi_norm_sq * dt_val;
                auto linear_contrib = grad * conj_M;
                double nl_scalar = 2.0 * beta_val * dt_val
                    * std::imag(std::conj(grad) * psi);

                auto [xa, xo] = resolve(page.parent_a[off]);
                pages_[xa]->gradients[xo] += linear_contrib + nl_scalar * psi;
                break;
            }
            } // switch
        }
    }

    // ── Accessors ──────────────────────────────────────────────────────

    std::complex<double> get_value(uint32_t id) const {
        auto [pi, off] = resolve(id);
        return pages_[pi]->values[off];
    }

    std::complex<double> get_gradient(uint32_t id) const {
        auto [pi, off] = resolve(id);
        return pages_[pi]->gradients[off];
    }

    void set_value(uint32_t id, std::complex<double> value) {
        auto [pi, off] = resolve(id);
        pages_[pi]->values[off] = value;
    }

    uint32_t size() const { return num_nodes_; }
    size_t page_count() const { return pages_.size(); }
    size_t capacity() const { return capacity_; }

    /// Clear for next training step: reset node count, keep pages allocated
    void clear() {
        num_nodes_ = 0;
        matrix_count_ = 0;
        scalar_count_ = 0;
    }

private:
    std::vector<std::unique_ptr<ComputePage<PAGE_SIZE>>> pages_;
    size_t num_nodes_ = 0;
    size_t capacity_  = 0;

    std::vector<Eigen::MatrixXcd> matrices_;
    std::vector<double>           scalars_;
    size_t matrix_count_ = 0;
    size_t scalar_count_ = 0;

    void grow() {
        pages_.push_back(std::make_unique<ComputePage<PAGE_SIZE>>());
        capacity_ += PAGE_SIZE;
    }

    std::pair<size_t, size_t> resolve(uint32_t id) const {
        return {id / PAGE_SIZE, id % PAGE_SIZE};
    }

    uint16_t store_matrix(const Eigen::MatrixXcd& m) {
        if (matrix_count_ >= matrices_.size())
            matrices_.resize(std::max(matrices_.size() * 2, size_t(64)));
        uint16_t idx = static_cast<uint16_t>(matrix_count_++);
        matrices_[idx] = m;
        return idx;
    }

    uint16_t store_scalars(double a, double b) {
        if (scalar_count_ + 2 >= scalars_.size())
            scalars_.resize(std::max(scalars_.size() * 2, size_t(256)));
        uint16_t idx = static_cast<uint16_t>(scalar_count_);
        scalars_[idx]     = a;
        scalars_[idx + 1] = b;
        scalar_count_ += 2;
        return idx;
    }
};

} // namespace nikola::autodiff
