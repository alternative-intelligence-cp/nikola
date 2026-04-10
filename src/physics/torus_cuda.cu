/**
 * @file src/physics/torus_cuda.cu
 * @brief GPU Hamiltonian density kernel — Phase 110 / GAP-034 device path.
 *
 * Implements `compute_hamiltonian_device()`, the CUDA counterpart to the
 * host-side `compute_hamiltonian_host()` in gpu_hamiltonian.hpp.
 *
 * Algorithm (per thread):
 *   kin_i  = 0.5  · |V_i|²
 *   grd_i  = c²   · (-0.5) · Re(Ψ_i* · ∇²Ψ_i)   [IBP gradient energy]
 *   nl_i   = β/4  · |Ψ_i|⁴                        [nonlinear self-interaction]
 *
 * Reduction strategy (two-level):
 *   1. Per-warp:  __shfl_down_sync reduction (5 stages, no shared-mem needed).
 *   2. Per-block: warp-0 reduces the per-warp sums from shared memory.
 *   3. Cross-block: atomicAdd on three device double accumulators.
 *
 * Precision notes:
 *   - Thread-local accumulation in double (avoids FP32 catastrophic cancellation
 *     on the gradient term when Re(Ψ* · lap) may be near zero).
 *   - atomicAdd(double*) requires CC ≥ 6.0; RTX 3090 is CC 8.6 — safe.
 *   - Final result matches compute_hamiltonian_host() within ≈ 1 ULP for
 *     N ≤ 20 000 nodes (validated in phase110 tests).
 *
 * Hardware target  : NVIDIA RTX 3090, sm_86 (CC 8.6)
 * CUDA version     : 12.0+
 * Compile options  : nvcc -arch=sm_86 -std=c++17
 *
 * Phase  : 110
 * GAP ID : GAP-034 (device path)
 * Spec   : docs/info/integration/sections/02_foundations/04_energy_conservation.md
 *          §4.2.1 "Total Hamiltonian — parallel reduction over the grid"
 */

#include <nikola/physics/gpu_hamiltonian.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// CUDA error macro
// ============================================================================

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            throw std::runtime_error(                                           \
                std::string("CUDA error [torus_cuda] at " __FILE__ ":") +      \
                std::to_string(__LINE__) + " — " +                             \
                cudaGetErrorString(_e));                                        \
        }                                                                       \
    } while (0)

// ============================================================================
// Constants
// ============================================================================

static constexpr int BLOCK_SZ = 256;

// ============================================================================
// Device kernel: hamiltonian_density_kernel
// ============================================================================

/**
 * @brief One-pass UFIE Hamiltonian density with two-level reduction.
 *
 * @param d_psi_r     Re(Ψ) device array, length N
 * @param d_psi_i     Im(Ψ) device array, length N
 * @param d_vel_r     Re(∂Ψ/∂t) device array, length N
 * @param d_vel_i     Im(∂Ψ/∂t) device array, length N
 * @param d_lap_r     Re(∇²Ψ) device array, length N  (pre-computed)
 * @param d_lap_i     Im(∇²Ψ) device array, length N  (pre-computed)
 * @param d_kinetic   Device accumulator: Σ kin_i    (must be zero-initialised)
 * @param d_gradient  Device accumulator: Σ grd_i    (must be zero-initialised)
 * @param d_nonlinear Device accumulator: Σ nl_i     (must be zero-initialised)
 * @param beta        Nonlinear coefficient β
 * @param c2          Wave speed squared c₀²
 * @param N           Number of nodes
 */
__global__ static void hamiltonian_density_kernel(
        const float* __restrict__ d_psi_r,
        const float* __restrict__ d_psi_i,
        const float* __restrict__ d_vel_r,
        const float* __restrict__ d_vel_i,
        const float* __restrict__ d_lap_r,
        const float* __restrict__ d_lap_i,
        double* __restrict__      d_kinetic,
        double* __restrict__      d_gradient,
        double* __restrict__      d_nonlinear,
        float beta, float c2, int N)
{
    // ----------------------------------------------------------------
    // Step 1: Per-thread energy density (double precision)
    // ----------------------------------------------------------------
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double kin = 0.0, grd = 0.0, nl = 0.0;

    if (i < N) {
        const double pr = static_cast<double>(d_psi_r[i]);
        const double pi = static_cast<double>(d_psi_i[i]);
        const double vr = static_cast<double>(d_vel_r[i]);
        const double vi = static_cast<double>(d_vel_i[i]);
        const double lr = static_cast<double>(d_lap_r[i]);
        const double li = static_cast<double>(d_lap_i[i]);

        // Kinetic: 0.5 |V|²
        kin = 0.5 * (vr * vr + vi * vi);

        // Gradient (IBP): c² · (-0.5) · Re(Ψ* · ∇²Ψ) = c² · (-0.5) · (pr·lr + pi·li)
        grd = static_cast<double>(c2) * (-0.5) * (pr * lr + pi * li);

        // Nonlinear: (β/4) |Ψ|⁴
        const double psi_sq = pr * pr + pi * pi;
        nl = static_cast<double>(beta) * 0.25 * psi_sq * psi_sq;
    }

    // ----------------------------------------------------------------
    // Step 2: Warp-level reduction via __shfl_down_sync, then
    //         shared-memory cross-warp reduction, then atomicAdd.
    // ----------------------------------------------------------------
    //
    // Each warp (32 threads) reduces its own partial sums with warp
    // shuffles (no __syncthreads needed). Then lane 0 of each warp
    // writes its result into shared memory. A final warp reduces
    // the per-warp sums and atomicAdds to the global accumulators.
    // ----------------------------------------------------------------

    constexpr unsigned FULL_MASK = 0xFFFFFFFF;

    // Intra-warp reduction via shuffle (5 stages: 16→8→4→2→1)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        kin += __shfl_down_sync(FULL_MASK, kin, offset);
        grd += __shfl_down_sync(FULL_MASK, grd, offset);
        nl  += __shfl_down_sync(FULL_MASK, nl,  offset);
    }

    // Write per-warp results to shared memory (one slot per warp)
    constexpr int WARPS_PER_BLOCK = BLOCK_SZ / 32;
    __shared__ double s_kin[WARPS_PER_BLOCK];
    __shared__ double s_grd[WARPS_PER_BLOCK];
    __shared__ double s_nl [WARPS_PER_BLOCK];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        s_kin[warp_id] = kin;
        s_grd[warp_id] = grd;
        s_nl [warp_id] = nl;
    }

    __syncthreads();

    // Final reduction: first warp reduces the WARPS_PER_BLOCK partial sums
    if (warp_id == 0) {
        kin = (lane_id < WARPS_PER_BLOCK) ? s_kin[lane_id] : 0.0;
        grd = (lane_id < WARPS_PER_BLOCK) ? s_grd[lane_id] : 0.0;
        nl  = (lane_id < WARPS_PER_BLOCK) ? s_nl [lane_id] : 0.0;

        #pragma unroll
        for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1) {
            kin += __shfl_down_sync(FULL_MASK, kin, offset);
            grd += __shfl_down_sync(FULL_MASK, grd, offset);
            nl  += __shfl_down_sync(FULL_MASK, nl,  offset);
        }
    }

    // ----------------------------------------------------------------
    // Step 3: Atomic accumulation across blocks (only lane 0 of warp 0)
    // ----------------------------------------------------------------
    if (threadIdx.x == 0) {
        atomicAdd(d_kinetic,   kin);
        atomicAdd(d_gradient,  grd);
        atomicAdd(d_nonlinear, nl);
    }
}

// ============================================================================
// Host wrapper: compute_hamiltonian_device
// ============================================================================

namespace nikola::physics {

/**
 * @brief GPU-resident Hamiltonian reduction — device path for GAP-034.
 *
 * Uploads @p buf to the device, runs `hamiltonian_density_kernel` over all N
 * nodes, waits for the kernel to finish, then downloads the three accumulated
 * sums.  Multiplies by dV and reconstructs GpuHamiltonianTerms.
 *
 * @throws std::invalid_argument if buf.consistent() is false or cfg.dV ≤ 0.
 * @throws std::runtime_error    on any CUDA error.
 */
GpuHamiltonianTerms compute_hamiltonian_device(
        const GpuFieldBuffer&       buf,
        const GpuHamiltonianConfig& cfg)
{
    if (!buf.consistent())
        throw std::invalid_argument(
            "compute_hamiltonian_device: GpuFieldBuffer component size mismatch");
    if (cfg.dV <= 0.0f)
        throw std::invalid_argument(
            "compute_hamiltonian_device: cfg.dV must be > 0");

    const int N = static_cast<int>(buf.size());

    // ---------------------------------------------------------------
    // Allocate device arrays
    // ---------------------------------------------------------------
    float*  d_psi_r = nullptr;
    float*  d_psi_i = nullptr;
    float*  d_vel_r = nullptr;
    float*  d_vel_i = nullptr;
    float*  d_lap_r = nullptr;
    float*  d_lap_i = nullptr;
    double* d_kin   = nullptr;  // scalar accumulator
    double* d_grd   = nullptr;
    double* d_nl    = nullptr;

    const std::size_t nbytes = static_cast<std::size_t>(N) * sizeof(float);

    try {
        CUDA_CHECK(cudaMalloc(&d_psi_r, nbytes));
        CUDA_CHECK(cudaMalloc(&d_psi_i, nbytes));
        CUDA_CHECK(cudaMalloc(&d_vel_r, nbytes));
        CUDA_CHECK(cudaMalloc(&d_vel_i, nbytes));
        CUDA_CHECK(cudaMalloc(&d_lap_r, nbytes));
        CUDA_CHECK(cudaMalloc(&d_lap_i, nbytes));
        CUDA_CHECK(cudaMalloc(&d_kin,   sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_grd,   sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_nl,    sizeof(double)));

        // ---------------------------------------------------------------
        // Upload SoA arrays  (H→D)
        // ---------------------------------------------------------------
        CUDA_CHECK(cudaMemcpy(d_psi_r, buf.psi_real.data(), nbytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_psi_i, buf.psi_imag.data(), nbytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel_r, buf.vel_real.data(), nbytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel_i, buf.vel_imag.data(), nbytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lap_r, buf.lap_real.data(), nbytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lap_i, buf.lap_imag.data(), nbytes, cudaMemcpyHostToDevice));

        // ---------------------------------------------------------------
        // Zero accumulators
        // ---------------------------------------------------------------
        CUDA_CHECK(cudaMemset(d_kin, 0, sizeof(double)));
        CUDA_CHECK(cudaMemset(d_grd, 0, sizeof(double)));
        CUDA_CHECK(cudaMemset(d_nl,  0, sizeof(double)));

        // ---------------------------------------------------------------
        // Launch kernel
        // ---------------------------------------------------------------
        const int grid = (N + BLOCK_SZ - 1) / BLOCK_SZ;

        hamiltonian_density_kernel<<<grid, BLOCK_SZ>>>(
            d_psi_r, d_psi_i,
            d_vel_r, d_vel_i,
            d_lap_r, d_lap_i,
            d_kin, d_grd, d_nl,
            cfg.beta, cfg.c2, N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---------------------------------------------------------------
        // Download accumulated sums  (D→H)
        // ---------------------------------------------------------------
        double h_kin = 0.0, h_grd = 0.0, h_nl = 0.0;
        CUDA_CHECK(cudaMemcpy(&h_kin, d_kin, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_grd, d_grd, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_nl,  d_nl,  sizeof(double), cudaMemcpyDeviceToHost));

        // ---------------------------------------------------------------
        // Free device memory
        // ---------------------------------------------------------------
        cudaFree(d_psi_r); cudaFree(d_psi_i);
        cudaFree(d_vel_r); cudaFree(d_vel_i);
        cudaFree(d_lap_r); cudaFree(d_lap_i);
        cudaFree(d_kin);   cudaFree(d_grd);   cudaFree(d_nl);

        // ---------------------------------------------------------------
        // Assemble result (multiply by dV)
        // ---------------------------------------------------------------
        const double dV = static_cast<double>(cfg.dV);
        GpuHamiltonianTerms out;
        out.kinetic   = h_kin * dV;
        out.gradient  = h_grd * dV;
        out.nonlinear = h_nl  * dV;
        out.total     = out.kinetic + out.gradient + out.nonlinear;
        return out;

    } catch (...) {
        // Release all device memory on any exception
        if (d_psi_r) cudaFree(d_psi_r);
        if (d_psi_i) cudaFree(d_psi_i);
        if (d_vel_r) cudaFree(d_vel_r);
        if (d_vel_i) cudaFree(d_vel_i);
        if (d_lap_r) cudaFree(d_lap_r);
        if (d_lap_i) cudaFree(d_lap_i);
        if (d_kin)   cudaFree(d_kin);
        if (d_grd)   cudaFree(d_grd);
        if (d_nl)    cudaFree(d_nl);
        throw;
    }
}

} // namespace nikola::physics
