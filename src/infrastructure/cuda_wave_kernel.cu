/**
 * @file src/infrastructure/cuda_wave_kernel.cu
 * @brief CUDA kernel implementations — Phase 105 / GAP-046 (live).
 *
 * Compiled with nvcc (CUDA 12.0, sm_86 / RTX 3090).
 *
 * Kernels:
 *   psi_squared_kernel  — per-element |Ψ|² = psi_r² + psi_i²
 *   scale_field_kernel  — per-element Ψ *= alpha (modifies both components)
 *
 * Both are embarrassingly parallel, one thread per node, block size 256.
 */

#include <nikola/infrastructure/cuda_wave_kernel.hpp>

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace nikola::infrastructure {

// ============================================================================
// Device kernels
// ============================================================================

/**
 * @brief Compute per-element amplitude-squared.
 *
 * d_out[i] = d_psi_r[i]² + d_psi_i[i]²
 */
__global__ void psi_squared_kernel(
        const float* __restrict__ d_psi_r,
        const float* __restrict__ d_psi_i,
        float*       __restrict__ d_out,
        int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float r = d_psi_r[i];
    const float im = d_psi_i[i];
    d_out[i] = r * r + im * im;
}

/**
 * @brief In-place scale both wavefunction components by alpha.
 *
 * d_psi_r[i] *= alpha
 * d_psi_i[i] *= alpha
 */
__global__ void scale_field_kernel(
        float* __restrict__ d_psi_r,
        float* __restrict__ d_psi_i,
        float alpha,
        int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    d_psi_r[i] *= alpha;
    d_psi_i[i] *= alpha;
}

// ============================================================================
// Host-side wrappers (defined here to access CUDA launch syntax)
// ============================================================================

void CudaWaveKernel::launch_psi_squared()
{
    if (n_nodes_ == 0)
        throw std::runtime_error("CudaWaveKernel::launch_psi_squared: no nodes allocated");

    constexpr int BLOCK = 256;
    const int grid = (static_cast<int>(n_nodes_) + BLOCK - 1) / BLOCK;

    psi_squared_kernel<<<grid, BLOCK>>>(d_psi_r_, d_psi_i_, d_out_, static_cast<int>(n_nodes_));

    last_err_ = cudaGetLastError();
    check_err("launch psi_squared_kernel");

    last_err_ = cudaDeviceSynchronize();
    check_err("cudaDeviceSynchronize after psi_squared");
}

void CudaWaveKernel::launch_scale(float alpha)
{
    if (n_nodes_ == 0)
        throw std::runtime_error("CudaWaveKernel::launch_scale: no nodes allocated");

    constexpr int BLOCK = 256;
    const int grid = (static_cast<int>(n_nodes_) + BLOCK - 1) / BLOCK;

    scale_field_kernel<<<grid, BLOCK>>>(d_psi_r_, d_psi_i_, alpha, static_cast<int>(n_nodes_));

    last_err_ = cudaGetLastError();
    check_err("launch scale_field_kernel");

    last_err_ = cudaDeviceSynchronize();
    check_err("cudaDeviceSynchronize after scale_field");
}

} // namespace nikola::infrastructure
