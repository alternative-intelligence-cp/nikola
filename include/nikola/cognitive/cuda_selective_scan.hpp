#pragma once
/**
 * @file cognitive/cuda_selective_scan.hpp
 * @brief CUDA-accelerated Mamba S6 parallel selective scan.
 *
 * v0.1.6 Phase 3: GPU kernel for the selective scan recurrence:
 *   h_t = Ā_t · h_{t-1} + B̄_t
 * parallelized using the Blelloch work-efficient prefix scan with the
 * associative operator (a, b) ⊕ (c, d) = (a·c, a·d + b).
 *
 * The kernel processes T inputs × H hidden units in parallel.
 * Two phases:
 *   1. Pre-computation: compute Ā_t, B̄_t from inputs, A, W_delta, W_Bsel
 *   2. Parallel scan: Blelloch prefix scan over the T dimension per hidden unit
 *
 * PIMPL pattern: no CUDA headers leak into this header.
 *
 * Target: 1M nodes/ms throughput on RTX 3090 (sm_86).
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace nikola::cognitive {

// ============================================================================
// CudaSelectiveScan — GPU-accelerated parallel selective scan
// ============================================================================

class CudaSelectiveScan {
public:
    /**
     * @brief Construct with SSM dimensions.
     *
     * @param hidden_dim  H — hidden state size (default 256).
     * @param input_dim   I — input dimension (default 9).
     */
    explicit CudaSelectiveScan(int hidden_dim = 256, int input_dim = 9);

    ~CudaSelectiveScan();

    // Non-copyable, moveable
    CudaSelectiveScan(const CudaSelectiveScan&)            = delete;
    CudaSelectiveScan& operator=(const CudaSelectiveScan&) = delete;
    CudaSelectiveScan(CudaSelectiveScan&&)                 noexcept;
    CudaSelectiveScan& operator=(CudaSelectiveScan&&)      noexcept;

    // ----------------------------------------------------------------- upload

    /**
     * @brief Upload SSM weight matrices to GPU.
     *
     * Must be called once before scan(), and again if weights change.
     *
     * @param A        Diagonal A values, length H.
     * @param W_delta  Input→Δ projection, H×I row-major.
     * @param W_Bsel   Input→selective-B projection, H×I row-major.
     */
    void upload_weights(const float* A, const float* W_delta,
                        const float* W_Bsel);

    // ----------------------------------------------------------------- scan

    /**
     * @brief Run parallel selective scan over a sequence of T inputs.
     *
     * @param inputs     T × I row-major inputs (host memory).
     * @param h_init     Initial hidden state, length H (host memory).
     * @param T          Sequence length.
     * @param h_out      Output: final hidden state after T steps, length H
     *                   (host memory, written by this call).
     * @param all_states Optional: if non-null, receives all T hidden states
     *                   (T × H row-major, host memory). Pass nullptr to skip.
     */
    void scan(const float* inputs, const float* h_init, int T,
              float* h_out, float* all_states = nullptr);

    /**
     * @brief Throughput benchmark: scan T steps, return elapsed microseconds.
     */
    float benchmark_scan(const float* inputs, const float* h_init, int T,
                         float* h_out, int warmup = 5, int repeats = 20);

    // ----------------------------------------------------------------- info

    int hidden_dim()  const noexcept;
    int input_dim()   const noexcept;

private:
    struct Impl;
    struct ImplDeleter { void operator()(Impl* p) const noexcept; };
    std::unique_ptr<Impl, ImplDeleter> impl_;
};

} // namespace nikola::cognitive
