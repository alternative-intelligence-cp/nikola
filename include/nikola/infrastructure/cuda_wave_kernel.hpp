/**
 * @file include/nikola/infrastructure/cuda_wave_kernel.hpp
 * @brief Live CUDA wavefunction kernel — Phase 105 / GAP-046.
 *
 * Provides actual GPU-executed CUDA kernels for two fundamental wavefunction
 * operations used throughout the Nikola physics engine:
 *
 *   1. `psi_squared_kernel`  — per-element amplitude-squared |Ψᵢ|² = psi_r² + psi_i²
 *   2. `scale_field_kernel`  — in-place element-wise scaling  Ψᵢ *= α
 *
 * Both kernels are embarrassingly parallel (zero inter-thread communication),
 * block size 256, mapping one CUDA thread to one grid node — identical to the
 * block topology described in the Phase 22 + Phase 77 CUDA spec:
 *
 *   block_size = 256
 *   num_blocks = (N + 255) / 256
 *
 * The host-side class `CudaWaveKernel` wraps device buffer lifecycle
 * (cudaMalloc / cudaFree), host↔device transfers (cudaMemcpy), and kernel
 * launch.  All calls check CUDA error codes and store the last error for
 * inspection by the test harness.
 *
 * @code
 *   CudaWaveKernel k;
 *   k.allocate(64);
 *
 *   std::vector<float> r(64, 1.0f), i(64, 0.0f);
 *   k.upload(r.data(), i.data(), 64);
 *   k.launch_psi_squared();          // |Ψ|² = 1 per element
 *
 *   std::vector<float> out(64);
 *   k.download_output(out.data(), 64);
 *   // out[*] == 1.0f
 * @endcode
 *
 * Requires: CUDA 12.0, RTX 3090 (sm_86).
 * Link:     `nikola_cuda_kernels` CMake target (compiled with nvcc).
 *
 * Phase  : 105
 * GAP ID : GAP-046 (live)
 * Spec   : docs/info/integration/sections/10_appendices/04_hardware_optimization.md
 *          §D.2 CUDA Kernel Implementations; docs §05_autonomous_systems §Adversarial
 */
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>

namespace nikola::infrastructure {

// ============================================================================
// CudaWaveKernel — host-side lifecycle + launch API
// ============================================================================

/**
 * @brief RAII manager for GPU wavefunction kernel execution.
 *
 * Allocation lifecycle:
 *   - constructor: no allocation (lazy)
 *   - allocate(n): cudaMalloc for d_psi_r, d_psi_i, d_out (3 × n floats)
 *   - destructor: cudaFree for all three buffers
 *
 * Thread safety: not thread-safe — use one instance per CUDA stream.
 */
class CudaWaveKernel {
public:
    CudaWaveKernel() noexcept = default;

    /**
     * @brief Destructor frees all device allocations.
     *
     * Safe to call with n_nodes_ == 0 (no-op).
     */
    ~CudaWaveKernel() { free_device(); }

    // Non-copyable — device pointers can't be trivially copied
    CudaWaveKernel(const CudaWaveKernel&)            = delete;
    CudaWaveKernel& operator=(const CudaWaveKernel&) = delete;

    // Moveable
    CudaWaveKernel(CudaWaveKernel&& o) noexcept
        : d_psi_r_(o.d_psi_r_), d_psi_i_(o.d_psi_i_),
          d_out_(o.d_out_), n_nodes_(o.n_nodes_), last_err_(o.last_err_)
    {
        o.d_psi_r_ = o.d_psi_i_ = o.d_out_ = nullptr;
        o.n_nodes_ = 0;
    }

    // ---------------------------------------------------------------------- allocation

    /**
     * @brief Allocate device buffers for @p n nodes.
     *
     * Frees any pre-existing allocation before re-allocating.
     *
     * @throws std::runtime_error if any cudaMalloc fails.
     */
    void allocate(std::size_t n) {
        free_device();
        if (n == 0) return;
        last_err_ = cudaMalloc(reinterpret_cast<void**>(&d_psi_r_), n * sizeof(float));
        check_err("cudaMalloc d_psi_r");
        last_err_ = cudaMalloc(reinterpret_cast<void**>(&d_psi_i_), n * sizeof(float));
        check_err("cudaMalloc d_psi_i");
        last_err_ = cudaMalloc(reinterpret_cast<void**>(&d_out_),   n * sizeof(float));
        check_err("cudaMalloc d_out");
        n_nodes_ = n;
    }

    // ---------------------------------------------------------------------- transfers

    /**
     * @brief Copy wavefunction arrays from host to device.
     *
     * @param h_psi_r  Host array of Re(Ψ), @p n floats.
     * @param h_psi_i  Host array of Im(Ψ), @p n floats.
     * @param n        Number of elements.  Must equal n_nodes().
     * @throws std::runtime_error on size mismatch or cudaMemcpy failure.
     */
    void upload(const float* h_psi_r, const float* h_psi_i, std::size_t n) {
        if (n != n_nodes_)
            throw std::runtime_error("CudaWaveKernel::upload: size mismatch");
        last_err_ = cudaMemcpy(d_psi_r_, h_psi_r, n * sizeof(float), cudaMemcpyHostToDevice);
        check_err("cudaMemcpy H→D psi_r");
        last_err_ = cudaMemcpy(d_psi_i_, h_psi_i, n * sizeof(float), cudaMemcpyHostToDevice);
        check_err("cudaMemcpy H→D psi_i");
    }

    /**
     * @brief Copy output buffer from device to host.
     *
     * @param h_out  Destination host array, @p n floats.
     * @param n      Must equal n_nodes().
     * @throws std::runtime_error on size mismatch or cudaMemcpy failure.
     */
    void download_output(float* h_out, std::size_t n) {
        if (n != n_nodes_)
            throw std::runtime_error("CudaWaveKernel::download_output: size mismatch");
        last_err_ = cudaMemcpy(h_out, d_out_, n * sizeof(float), cudaMemcpyDeviceToHost);
        check_err("cudaMemcpy D→H out");
    }

    // ---------------------------------------------------------------------- kernel launches

    /**
     * @brief Launch psi_squared_kernel: d_out[i] = d_psi_r[i]² + d_psi_i[i]²
     *
     * Block size: 256 threads.  Grid size: (N + 255) / 256 blocks.
     * Synchronises the stream before returning (cudaDeviceSynchronize).
     *
     * @throws std::runtime_error if n_nodes() == 0, or on launch/sync error.
     */
    void launch_psi_squared();

    /**
     * @brief Launch scale_field_kernel: d_psi_r[i] *= alpha, d_psi_i[i] *= alpha
     *
     * Modifies d_psi_r and d_psi_i in-place.  d_out is not affected.
     * Synchronises the stream before returning.
     *
     * @param alpha  Scale factor applied to every element.
     * @throws std::runtime_error if n_nodes() == 0, or on launch/sync error.
     */
    void launch_scale(float alpha);

    // ---------------------------------------------------------------------- status

    /// Number of nodes allocated (0 if allocate() has not been called).
    [[nodiscard]] std::size_t n_nodes() const noexcept { return n_nodes_; }

    /// Last CUDA error code.  cudaSuccess (0) when all operations succeeded.
    [[nodiscard]] cudaError_t last_error() const noexcept { return last_err_; }

    /// True iff last_error() == cudaSuccess.
    [[nodiscard]] bool ok() const noexcept { return last_err_ == cudaSuccess; }

private:
    float*     d_psi_r_   = nullptr;
    float*     d_psi_i_   = nullptr;
    float*     d_out_     = nullptr;
    std::size_t n_nodes_  = 0;
    cudaError_t last_err_ = cudaSuccess;

    void free_device() noexcept {
        if (d_psi_r_) { cudaFree(d_psi_r_); d_psi_r_ = nullptr; }
        if (d_psi_i_) { cudaFree(d_psi_i_); d_psi_i_ = nullptr; }
        if (d_out_)   { cudaFree(d_out_);   d_out_   = nullptr; }
        n_nodes_ = 0;
    }

    void check_err(const char* ctx) {
        if (last_err_ != cudaSuccess)
            throw std::runtime_error(
                std::string("CudaWaveKernel: ") + ctx + " failed: "
                + cudaGetErrorString(last_err_));
    }
};

} // namespace nikola::infrastructure
