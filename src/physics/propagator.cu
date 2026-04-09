/**
 * @file src/physics/propagator.cu
 * @brief CUDA kernels + CudaPropagator implementation.
 *
 * Implements the same 6-substep Strang split as the CPU Propagator:
 *
 *   D(dt/2)  · kick(dt/2) · drift(dt) · kick(dt/2) · NL(dt) · D(dt/2)
 *
 * All kernels are embarrassingly parallel over N nodes (block=256 threads).
 * Full working set (19,683 nodes) fits comfortably in the RTX 3090 L2 cache
 * (6 MB), so memory latency is hidden by the cache even on strided access.
 *
 * Adjacency encoding:
 *   Host adj_ stores size_t (8 bytes).  On upload we convert to uint32_t
 *   (4 bytes) using GPU_VACUUM = 0xFFFFFFFF for VACUUM_NODE entries.
 *   For the dense periodic 3^9 grid, no vacuum nodes exist; the check is
 *   compiled away by the branch predictor in practice.
 *
 * PML ghost boundary condition (matches CPU propagator.hpp):
 *   ghost(ψ_self) = ψ_self × 0.9   (absorbing boundary layer damping)
 *
 * NB: Kahan compensated Laplacian is used on CPU for numerical stability.
 *   On GPU we use plain FP32 summation over 9 terms — the rounding error
 *   is negligible for this problem size and adds ~50% register pressure
 *   for no observable difference in physics output.
 */

#include <nikola/physics/cuda_propagator.hpp>
#include <nikola/physics/wave_function.hpp>
#include <nikola/foundation/toroidal_grid.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// CUDA error checking
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            throw std::runtime_error(                                          \
                std::string("CUDA error at " __FILE__ ":") +                  \
                std::to_string(__LINE__) + " — " +                            \
                cudaGetErrorString(_e));                                       \
        }                                                                      \
    } while (0)

// ============================================================================
// Constants
// ============================================================================

static constexpr uint32_t GPU_VACUUM    = 0xFFFFFFFFu;
static constexpr float    PML_ALPHA_ABS = 0.9f;   // matches complex_field.hpp
static constexpr int      BLOCK_SZ      = 256;

// ============================================================================
// CUDA kernels
// ============================================================================

// ----------------------------------------------------------------------------
// k_damping — V ← V · exp(-α(1-r)τ)
// ----------------------------------------------------------------------------
__global__ static void k_damping(
        float* __restrict__ vel_r,
        float* __restrict__ vel_i,
        const float* __restrict__ resonance,
        float alpha, float tau,
        unsigned N)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float factor = expf(-alpha * (1.f - resonance[i]) * tau);
    vel_r[i] *= factor;
    vel_i[i] *= factor;
}

// ----------------------------------------------------------------------------
// k_drift — Ψ ← Ψ + V·τ
// ----------------------------------------------------------------------------
__global__ static void k_drift(
        float* __restrict__ psi_r,
        float* __restrict__ psi_i,
        const float* __restrict__ vel_r,
        const float* __restrict__ vel_i,
        float tau,
        unsigned N)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    psi_r[i] += vel_r[i] * tau;
    psi_i[i] += vel_i[i] * tau;
}

// ----------------------------------------------------------------------------
// k_nonlinear — V ← V + β|Ψ|²Ψ·τ
// ----------------------------------------------------------------------------
__global__ static void k_nonlinear(
        const float* __restrict__ psi_r,
        const float* __restrict__ psi_i,
        float* __restrict__ vel_r,
        float* __restrict__ vel_i,
        float beta_tau,
        unsigned N)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float pr = psi_r[i], pi = psi_i[i];
    const float psi_sq = pr * pr + pi * pi;
    vel_r[i] += beta_tau * psi_sq * pr;
    vel_i[i] += beta_tau * psi_sq * pi;
}

// ----------------------------------------------------------------------------
// k_kick — compute 9D discrete Laplacian and apply velocity kick in one pass.
//
//   V ← V + c_eff²(i) · ∇²Ψ(i) · τ
//
// Adjacency: adj[i*18 + 2d]   = +e_d neighbour index (or GPU_VACUUM)
//            adj[i*18 + 2d+1] = -e_d neighbour index (or GPU_VACUUM)
// inv_h2[d] = 1 / h_d²  (device pointer, 9 floats)
// ----------------------------------------------------------------------------
__global__ static void k_kick(
        const float* __restrict__ psi_r,
        const float* __restrict__ psi_i,
        float* __restrict__ vel_r,
        float* __restrict__ vel_i,
        const float* __restrict__ state_field,
        const uint32_t* __restrict__ adj,
        const float* __restrict__ inv_h2,   // 9 floats on device
        float c0, float tau,
        unsigned N)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float psi_ri = psi_r[i];
    const float psi_ii = psi_i[i];

    // PML ghost values for vacuum neighbours
    const float ghost_r = psi_ri * PML_ALPHA_ABS;
    const float ghost_i = psi_ii * PML_ALPHA_ABS;

    float lap_r = 0.f;
    float lap_i = 0.f;

    const uint32_t* nbrs = adj + (size_t)i * 18;

    #pragma unroll
    for (int d = 0; d < 9; ++d) {
        const uint32_t np_idx = nbrs[2 * d];
        const uint32_t nm_idx = nbrs[2 * d + 1];

        const float np_r = (np_idx != GPU_VACUUM) ? psi_r[np_idx] : ghost_r;
        const float np_i = (np_idx != GPU_VACUUM) ? psi_i[np_idx] : ghost_i;
        const float nm_r = (nm_idx != GPU_VACUUM) ? psi_r[nm_idx] : ghost_r;
        const float nm_i = (nm_idx != GPU_VACUUM) ? psi_i[nm_idx] : ghost_i;

        const float h2i = inv_h2[d];
        lap_r += (np_r + nm_r - 2.f * psi_ri) * h2i;
        lap_i += (np_i + nm_i - 2.f * psi_ii) * h2i;
    }

    const float c_eff   = c0 / (1.f + state_field[i]);
    const float c2_tau  = c_eff * c_eff * tau;

    vel_r[i] += c2_tau * lap_r;
    vel_i[i] += c2_tau * lap_i;
}

// ============================================================================
// CudaImpl — device buffer management (defined in nikola::physics to match
// the forward declaration in cuda_propagator.hpp)
// ============================================================================

namespace nikola::physics {

struct CudaImpl {
    float*    d_psi_r     = nullptr;
    float*    d_psi_i     = nullptr;
    float*    d_vel_r     = nullptr;
    float*    d_vel_i     = nullptr;
    float*    d_resonance = nullptr;
    float*    d_state     = nullptr;
    float*    d_inv_h2    = nullptr;  // 9 floats
    uint32_t* d_adj       = nullptr;  // N×18

    size_t node_count = 0;

    void free_all() noexcept {
        auto safe = [](void* p){ if (p) cudaFree(p); };
        safe(d_psi_r);   safe(d_psi_i);
        safe(d_vel_r);   safe(d_vel_i);
        safe(d_resonance); safe(d_state);
        safe(d_inv_h2);  safe(d_adj);
        d_psi_r = d_psi_i = d_vel_r = d_vel_i = nullptr;
        d_resonance = d_state = d_inv_h2 = nullptr;
        d_adj = nullptr;
        node_count = 0;
    }

    ~CudaImpl() { free_all(); }
};

// ============================================================================
// CudaImplDeleter (defined here, used by unique_ptr in cuda_propagator.hpp)
// ============================================================================

void CudaImplDeleter::operator()(CudaImpl* p) const noexcept {
    delete p;
}

} // namespace nikola::physics

// ============================================================================
// CudaPropagator implementation
// ============================================================================

namespace nikola::physics {

CudaPropagator::CudaPropagator()
    : impl_(new CudaImpl(), CudaImplDeleter{})
{}

// ------------------------------------------------------------------
// upload
// ------------------------------------------------------------------
void CudaPropagator::upload(const WaveFunction& wf)
{
    const nikola::foundation::TorusGrid& g = wf.grid();
    if (!g.adjacency_valid())
        throw std::runtime_error(
            "CudaPropagator::upload: precompute_adjacency() must be called first");

    const size_t N = g.num_active_nodes();
    impl_->free_all();
    impl_->node_count = N;

    // ---------- allocate ----------
    CUDA_CHECK(cudaMalloc(&impl_->d_psi_r,     N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&impl_->d_psi_i,     N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&impl_->d_vel_r,     N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&impl_->d_vel_i,     N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&impl_->d_resonance, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&impl_->d_state,     N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&impl_->d_inv_h2,    9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&impl_->d_adj,       N * 18 * sizeof(uint32_t)));

    // ---------- physics arrays (H→D) ----------
    CUDA_CHECK(cudaMemcpy(impl_->d_psi_r,     g.psi_real(),    N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl_->d_psi_i,     g.psi_imag(),    N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl_->d_vel_r,     g.vel_real(),    N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl_->d_vel_i,     g.vel_imag(),    N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl_->d_resonance, g.resonance(),   N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl_->d_state,     g.state_field(), N*sizeof(float), cudaMemcpyHostToDevice));

    // ---------- inv_h2[9] ----------
    float inv_h2[9];
    for (int d = 0; d < 9; ++d) {
        const float h = g.spacing(d);
        inv_h2[d] = 1.f / (h * h);
    }
    CUDA_CHECK(cudaMemcpy(impl_->d_inv_h2, inv_h2, 9*sizeof(float), cudaMemcpyHostToDevice));

    // ---------- adjacency: convert size_t → uint32_t ----------
    {
        const size_t total    = g.adjacency_table_size();   // N * 18
        const size_t* adj_src = g.adjacency_table();
        std::vector<uint32_t> adj32(total);
        const size_t VMAX = std::numeric_limits<size_t>::max();
        for (size_t k = 0; k < total; ++k)
            adj32[k] = (adj_src[k] == VMAX) ? GPU_VACUUM
                                             : static_cast<uint32_t>(adj_src[k]);
        CUDA_CHECK(cudaMemcpy(impl_->d_adj, adj32.data(),
                              total * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
}

// ------------------------------------------------------------------
// download
// ------------------------------------------------------------------
void CudaPropagator::download(WaveFunction& wf) const
{
    const size_t N = impl_->node_count;
    if (N == 0)
        throw std::runtime_error("CudaPropagator::download: nothing uploaded");

    nikola::foundation::TorusGrid& g = wf.grid();
    if (g.num_active_nodes() != N)
        throw std::runtime_error("CudaPropagator::download: node count mismatch");

    CUDA_CHECK(cudaMemcpy(g.psi_real(), impl_->d_psi_r, N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g.psi_imag(), impl_->d_psi_i, N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g.vel_real(), impl_->d_vel_r, N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g.vel_imag(), impl_->d_vel_i, N*sizeof(float), cudaMemcpyDeviceToHost));
}

// ------------------------------------------------------------------
// step (pure GPU)
// ------------------------------------------------------------------
void CudaPropagator::step(float dt)
{
    const unsigned N      = static_cast<unsigned>(impl_->node_count);
    const unsigned blocks = (N + BLOCK_SZ - 1) / BLOCK_SZ;
    const float half_dt   = 0.5f * dt;

    // Strang split: D(dt/2) · kick(dt/2) · drift(dt) · kick(dt/2) · NL(dt) · D(dt/2)

    k_damping<<<blocks, BLOCK_SZ>>>(
        impl_->d_vel_r, impl_->d_vel_i, impl_->d_resonance,
        alpha_, half_dt, N);

    k_kick<<<blocks, BLOCK_SZ>>>(
        impl_->d_psi_r, impl_->d_psi_i,
        impl_->d_vel_r, impl_->d_vel_i,
        impl_->d_state, impl_->d_adj, impl_->d_inv_h2,
        c0_, half_dt, N);

    k_drift<<<blocks, BLOCK_SZ>>>(
        impl_->d_psi_r, impl_->d_psi_i,
        impl_->d_vel_r, impl_->d_vel_i,
        dt, N);

    k_kick<<<blocks, BLOCK_SZ>>>(
        impl_->d_psi_r, impl_->d_psi_i,
        impl_->d_vel_r, impl_->d_vel_i,
        impl_->d_state, impl_->d_adj, impl_->d_inv_h2,
        c0_, half_dt, N);

    k_nonlinear<<<blocks, BLOCK_SZ>>>(
        impl_->d_psi_r, impl_->d_psi_i,
        impl_->d_vel_r, impl_->d_vel_i,
        beta_ * dt, N);

    k_damping<<<blocks, BLOCK_SZ>>>(
        impl_->d_vel_r, impl_->d_vel_i, impl_->d_resonance,
        alpha_, half_dt, N);
}

// ------------------------------------------------------------------
// run
// ------------------------------------------------------------------
void CudaPropagator::run(int n_steps, float dt)
{
    for (int i = 0; i < n_steps; ++i) step(dt);
}

// ------------------------------------------------------------------
// sync
// ------------------------------------------------------------------
void CudaPropagator::sync() const
{
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ------------------------------------------------------------------
// step_synced (CPU-API-compatible drop-in)
// ------------------------------------------------------------------
void CudaPropagator::step_synced(WaveFunction& wf, float dt)
{
    if (!wf.grid().adjacency_valid())
        wf.grid().precompute_adjacency();
    upload(wf);
    step(dt);
    sync();
    download(wf);
}

// ------------------------------------------------------------------
// max_stable_dt
// ------------------------------------------------------------------
float CudaPropagator::max_stable_dt(const WaveFunction& wf) const noexcept
{
    const nikola::foundation::TorusGrid& g = wf.grid();
    float min_h = g.spacing(0);
    for (int d = 1; d < 9; ++d)
        min_h = std::min(min_h, g.spacing(d));
    const float sqrt9 = 3.f;
    return 0.5f * min_h / (c0_ * sqrt9);
}

// ------------------------------------------------------------------
// device_node_count
// ------------------------------------------------------------------
size_t CudaPropagator::device_node_count() const noexcept
{
    return impl_->node_count;
}

// ------------------------------------------------------------------
// query_occupancy — check GPU utilisation for the k_kick kernel
// ------------------------------------------------------------------
float CudaPropagator::query_occupancy() noexcept
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        return -1.0f;

    int max_active_blocks = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        k_kick,                // heaviest kernel (9-dim Laplacian)
        BLOCK_SZ,              // 256 threads per block
        0);                    // no dynamic shared memory

    if (err != cudaSuccess || max_active_blocks <= 0)
        return -1.0f;

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
        return -1.0f;

    int max_warps_per_sm    = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    int active_warps_per_sm = max_active_blocks * (BLOCK_SZ / prop.warpSize);

    return static_cast<float>(active_warps_per_sm) /
           static_cast<float>(max_warps_per_sm);
}

} // namespace nikola::physics
