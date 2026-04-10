#pragma once
/**
 * @file physics/cuda_propagator.hpp
 * @brief GPU-accelerated Strang-split UFIE propagator — RTX 3090 / sm_86.
 *
 * Drop-in GPU counterpart to the CPU Propagator.  Implements the identical
 * Strang–Verlet split-operator scheme on CUDA, achieving <1 ms per step on a
 * 3^9 = 19,683-node grid (original H100 spec target, cleared on RTX 3090).
 *
 * Usage model — minimise host↔device transfers:
 * @code
 *   WaveFunction wf;
 *   wf.seed_manifold(3, ...);
 *   wf.grid().precompute_adjacency();
 *
 *   CudaPropagator gpu_prop;
 *   gpu_prop.set_c0(1.f).set_beta(1.f).set_alpha(0.01f);
 *   gpu_prop.upload(wf);        // H→D once
 *
 *   gpu_prop.run(200, 0.01f);   // 200 steps, pure GPU
 *   gpu_prop.sync();            // wait for all kernels to finish
 *   gpu_prop.download(wf);      // D→H once
 * @endcode
 *
 * Or as a CPU-API-compatible drop-in (slower — H↔D per step):
 * @code
 *   gpu_prop.step_synced(wf, dt);
 * @endcode
 *
 * CUDA requirements:
 *   - CUDA 12.0+, compute capability 8.6 (RTX 3090)
 *   - Build with: cmake -DNVCC_ARCH=86 ..
 *   - Link: nikola_cuda (add_library nikola_cuda src/physics/propagator.cu)
 *
 * Thread hierarchy:
 *   Block size: 256 threads.  For N=19,683: 77 blocks (last block partially
 *   idle).  All kernels are embarrassingly parallel — zero inter-thread
 *   dependencies within any substep.
 *
 * Memory:
 *   Full working set ≈ 1.8 MB (fits in RTX 3090's 6 MB L2 cache).
 *   Adjacency table: N×18 uint32_t ≈ 1.4 MB (one-time upload).
 *   Physics SoA: 6 × N × 4 bytes ≈ 0.47 MB (re-uploaded per WF change).
 *
 * Phase: NIK-GPU-01 (performance milestone — GPU Phase 1 gate: <1 ms/step)
 */

#include <nikola/physics/wave_function.hpp>

#include <memory>
#include <stdexcept>

namespace nikola::physics {

// Forward declarations — defined only in propagator.cu (no CUDA headers here)
struct CudaImpl;
struct CudaImplDeleter {
    void operator()(CudaImpl* p) const noexcept;
};

// ============================================================================
// CudaPropagator
// ============================================================================

class CudaPropagator {
public:
    // Construction / destruction (impl_ allocated lazily on first upload)
    CudaPropagator();
    ~CudaPropagator() = default;

    // Non-copyable — device buffers cannot be cheaply copied
    CudaPropagator(const CudaPropagator&)            = delete;
    CudaPropagator& operator=(const CudaPropagator&) = delete;
    CudaPropagator(CudaPropagator&&)                 = default;
    CudaPropagator& operator=(CudaPropagator&&)      = default;

    // ------------------------------------------------------------------
    // Configuration (mirrors CPU Propagator API)
    // ------------------------------------------------------------------

    CudaPropagator& set_c0(float c0)     noexcept { c0_    = c0;    return *this; }
    CudaPropagator& set_beta(float beta) noexcept { beta_  = beta;  return *this; }
    CudaPropagator& set_alpha(float alpha) noexcept { alpha_ = alpha; return *this; }

    float c0()    const noexcept { return c0_; }
    float beta()  const noexcept { return beta_; }
    float alpha() const noexcept { return alpha_; }

    // ------------------------------------------------------------------
    // Host ↔ Device transfers
    // ------------------------------------------------------------------

    /**
     * @brief Upload WaveFunction to device memory.
     *
     * Allocates device buffers sized to wf.num_nodes() and copies all six
     * SoA arrays (psi_r/i, vel_r/i, resonance, state_field) plus the
     * precomputed adjacency table (converted to uint32_t for bandwidth).
     *
     * @pre  wf.grid().adjacency_valid() == true
     * @post Device state is a byte-for-byte copy of the host WaveFunction.
     */
    void upload(const WaveFunction& wf);

    /**
     * @brief Download psi and velocity arrays from device back to host.
     *
     * Only the physics arrays (psi_r/i, vel_r/i) are downloaded; resonance
     * and state_field are treated as static during a propagation run.
     *
     * @param wf  WaveFunction to receive device data (must have same node
     *            count as the last upload() call).
     */
    void download(WaveFunction& wf) const;

    // ------------------------------------------------------------------
    // GPU-side stepping (no H↔D transfers)
    // ------------------------------------------------------------------

    /**
     * @brief Advance by one Strang-split step on the GPU.
     *
     * Launches 6 CUDA kernels (D/2 · kick/2 · drift · kick/2 · NL · D/2).
     * Returns immediately; GPU work is queued asynchronously unless sync()
     * is called.
     *
     * @param dt  Timestep (CFL-safe: dt ≤ max_stable_dt(wf)).
     */
    void step(float dt);

    /**
     * @brief Advance N steps on GPU, then return.
     *
     * Queues N × 6 kernels.  Call sync() or download() afterwards.
     *
     * @param n_steps  Number of integration steps.
     * @param dt       Timestep per step.
     */
    void run(int n_steps, float dt);

    /**
     * @brief Block until all GPU kernels triggered by this propagator have
     *        completed.  Required before measuring elapsed time.
     */
    void sync() const;

    // ------------------------------------------------------------------
    // CPU-API-compatible drop-in (upload + step + download per call)
    // ------------------------------------------------------------------

    /**
     * @brief Fully synchronised step: H→D upload, GPU step, D→H download.
     *
     * Matches CPU Propagator::step() signature.  Incurs full H↔D transfer
     * overhead per call — use only for debugging or drop-in replacement in
     * code that cannot be refactored to batch steps.
     *
     * @param wf  WaveFunction to evolve in place.
     * @param dt  Timestep.
     */
    void step_synced(WaveFunction& wf, float dt);

    // ------------------------------------------------------------------
    // Utilities
    // ------------------------------------------------------------------

    /**
     * @brief CFL-safe maximum timestep (same formula as CPU Propagator).
     *
     *   dt_max = 0.5 · min_h / (c₀ · √9)
     *
     * @param wf  WaveFunction (only grid spacing is accessed; no upload required).
     */
    float max_stable_dt(const WaveFunction& wf) const noexcept;

    /// Number of nodes currently on device (0 before first upload).
    size_t device_node_count() const noexcept;

    // ------------------------------------------------------------------
    // Occupancy & diagnostics
    // ------------------------------------------------------------------

    /**
     * @brief Query GPU occupancy for the k_kick kernel (the most register-heavy).
     *
     * Returns the fraction of maximum active warps per SM, e.g. 0.75 for 75%.
     * Uses cudaOccupancyMaxActiveBlocksPerMultiprocessor to query the hardware.
     *
     * @return Occupancy as a fraction [0.0, 1.0], or -1.0 if no GPU present.
     */
    static float query_occupancy() noexcept;

private:
    float c0_    = 1.f;
    float beta_  = 1.f;
    float alpha_ = 0.01f;

    std::unique_ptr<CudaImpl, CudaImplDeleter> impl_;
};

} // namespace nikola::physics
