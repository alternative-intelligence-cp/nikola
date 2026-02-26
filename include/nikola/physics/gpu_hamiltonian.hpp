/**
 * @file include/nikola/physics/gpu_hamiltonian.hpp
 * @brief GPU-resident Hamiltonian oracle — Phase 104 / GAP-034.
 *
 * Implements the discrete UFIE Hamiltonian over a Structure-of-Arrays (SoA)
 * field buffer whose memory layout maps 1-to-1 onto CUDA device arrays:
 *
 *   H = Σᵢ [ (1/2)|Vᵢ|²                           (kinetic)
 *           + c²·(-1/2)·Re(Ψᵢ*·∇²Ψᵢ)              (gradient / IBP field)
 *           + (β/4)·|Ψᵢ|⁴                          (nonlinear self-interaction)
 *           ] · ΔV
 *
 * The Laplacian arrays (lap_real / lap_imag) must be pre-computed by the
 * propagator before calling compute(), matching the convention in the
 * Störmer–Verlet integrator.  This decouples adjacency topology from energy
 * accounting and allows the GPU Hamiltonian kernel to accept flat SoA pointers
 * without touching the neighbour list.
 *
 * Host-side path (this header):
 *   compile as plain C++17; links libcudart for device queries only.
 *
 * Device-side path (gpu_hamiltonian.cu, future):
 *   `__global__ void hamiltonian_density_kernel(…)` does one-pass per-element,
 *   followed by a warp-shuffle + block-level reduction into `d_accum[4]`, then
 *   a single cudaMemcpy to return kinetic/gradient/nonlinear/total.
 *
 * Reference: docs/info/integration/sections/02_foundations/04_energy_conservation.md
 *            §4.2.1 "Total Hamiltonian — parallel reduction over the grid".
 *
 * Phase  : 104
 * GAP ID : GAP-034
 * Spec   : §02_foundations/04_energy_conservation.md §4.2 + §4.3;
 *          hardware spec: RTX 3090, sm_86, CUDA 12.0
 */
#pragma once

#include <cuda_runtime_api.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>

namespace nikola::physics {

// ============================================================================
// GpuFieldBuffer — host-side SoA matching CUDA flat-array convention
// ============================================================================

/**
 * @brief Structure-of-Arrays wavefunction buffer.
 *
 * Each array must have identical length `n`.  Any element i represents one
 * active node on the 9-D toroidal manifold:
 *   (psi_real[i], psi_imag[i])  = Ψᵢ   (wave amplitude)
 *   (vel_real[i], vel_imag[i])  = Vᵢ   (∂Ψ/∂t velocity)
 *   (lap_real[i], lap_imag[i])  = ∇²Ψᵢ (18-pt discrete Laplacian)
 *
 * Alignment: each array is a separate allocation, keeping stride-1 access for
 * each component — identical to what cudaMalloc would give for a flat array.
 */
struct GpuFieldBuffer {
    std::vector<float> psi_real;   ///< Re(Ψ)
    std::vector<float> psi_imag;   ///< Im(Ψ)
    std::vector<float> vel_real;   ///< Re(∂Ψ/∂t)
    std::vector<float> vel_imag;   ///< Im(∂Ψ/∂t)
    std::vector<float> lap_real;   ///< Re(∇²Ψ) — pre-computed by propagator
    std::vector<float> lap_imag;   ///< Im(∇²Ψ) — pre-computed by propagator

    /// Number of nodes (all six arrays share this size).
    [[nodiscard]] std::size_t size() const noexcept { return psi_real.size(); }

    /// True when all six arrays have equal length ≥ 0.
    [[nodiscard]] bool consistent() const noexcept {
        const std::size_t n = psi_real.size();
        return psi_imag.size() == n
            && vel_real.size() == n
            && vel_imag.size() == n
            && lap_real.size() == n
            && lap_imag.size() == n;
    }

    /**
     * @brief Resize all six arrays to @p n elements, filling with @p val.
     *
     * Equivalent to calling assign(n, val) on each component.  Useful for
     * constructing trivial test cases (e.g. all-zero field).
     */
    void resize(std::size_t n, float val = 0.0f) {
        psi_real.assign(n, val);
        psi_imag.assign(n, val);
        vel_real.assign(n, val);
        vel_imag.assign(n, val);
        lap_real.assign(n, val);
        lap_imag.assign(n, val);
    }

    /**
     * @brief Zero all fields, retaining the current size.
     *
     * Equivalent to resize(size(), 0.f).
     */
    void zero() {
        const std::size_t n = size();
        resize(n, 0.0f);
    }
};

// ============================================================================
// GpuHamiltonianConfig
// ============================================================================

/**
 * @brief Physics parameters for the GPU Hamiltonian kernel.
 *
 * Defaults are tuned for natural-unit test runs:
 *   beta=1, c2=1, dV=1.
 *
 * RTX 3090 note: these are uploaded to constant memory once per batch.
 */
struct GpuHamiltonianConfig {
    float beta = 1.0f;   ///< Nonlinear self-interaction coefficient β
    float c2   = 1.0f;   ///< Wave speed squared c₀² (dimensionless natural units)
    float dV   = 1.0f;   ///< Volume element per node ΔV (product of grid spacings)
};

// ============================================================================
// GpuHamiltonianTerms — result of one energy reduction
// ============================================================================

/**
 * @brief Decomposed Hamiltonian returned by compute().
 *
 * The four fields satisfy:  total = (kinetic + gradient + nonlinear) * dV
 * where dV is already folded in via GpuHamiltonianConfig::dV.
 *
 * Using double precision for the reduced sum even when the device arrays hold
 * float32 data — this is the same strategy as `hamiltonian.hpp` (Kahan sum).
 */
struct GpuHamiltonianTerms {
    double kinetic   = 0.0;   ///< Σᵢ (1/2)|Vᵢ|² · ΔV
    double gradient  = 0.0;   ///< Σᵢ c²·(-1/2)·Re(Ψᵢ*·lap_i) · ΔV
    double nonlinear = 0.0;   ///< Σᵢ (β/4)|Ψᵢ|⁴ · ΔV
    double total     = 0.0;   ///< kinetic + gradient + nonlinear

    /// Recompute total from the three component terms.
    void recompute_total() noexcept { total = kinetic + gradient + nonlinear; }
};

// ============================================================================
// Free function: host-side parallel reduction (mirrors GPU kernel logic)
// ============================================================================

/**
 * @brief Compute the UFIE Hamiltonian over @p buf using the GPU kernel algorithm
 *        executed on the host CPU.
 *
 * This is the CPU reference implementation whose per-element computation is
 * identical to the planned CUDA kernel:
 *
 * @code
 * // (what the __global__ kernel does per thread)
 * float pr = psi_real[i], pi = psi_imag[i];
 * float vr = vel_real[i], vi = vel_imag[i];
 * float lr = lap_real[i], li = lap_imag[i];
 * float kin = 0.5f * (vr*vr + vi*vi);
 * float grd = cfg.c2 * (-0.5f) * (pr*lr + pi*li);   // Re(Ψ* · lap)
 * float nl  = (cfg.beta / 4.0f) * (pr*pr + pi*pi) * (pr*pr + pi*pi);
 * atomicAdd(d_kinetic,   (double)kin);
 * atomicAdd(d_gradient,  (double)grd);
 * atomicAdd(d_nonlinear, (double)nl );
 * @endcode
 *
 * Host reduction uses a plain loop; Kahan compensation is applied to the
 * totals to match the precision that a warp-shuffle reduction achieves on
 * float64 accumulators (CC≥8.0 supports `__reduce_add_sync` on fp32, but we
 * fold into double before summing to match Oracle precision requirements).
 *
 * @throws std::invalid_argument  if buf.consistent() is false.
 * @throws std::invalid_argument  if cfg.dV ≤ 0.
 */
[[nodiscard]]
inline GpuHamiltonianTerms compute_hamiltonian_host(
        const GpuFieldBuffer&    buf,
        const GpuHamiltonianConfig& cfg)
{
    if (!buf.consistent())
        throw std::invalid_argument("GpuFieldBuffer: component size mismatch");
    if (cfg.dV <= 0.0f)
        throw std::invalid_argument("GpuHamiltonianConfig: dV must be > 0");

    const std::size_t N = buf.size();

    // Kahan accumulation
    double kin  = 0.0, kin_c  = 0.0;
    double grd  = 0.0, grd_c  = 0.0;
    double nl   = 0.0, nl_c   = 0.0;

    for (std::size_t i = 0; i < N; ++i) {
        const double pr = buf.psi_real[i];
        const double pi = buf.psi_imag[i];
        const double vr = buf.vel_real[i];
        const double vi = buf.vel_imag[i];
        const double lr = buf.lap_real[i];
        const double li = buf.lap_imag[i];

        // Kinetic: 0.5 |V|²
        const double node_kin = 0.5 * (vr*vr + vi*vi);
        // Gradient (IBP): c² · (-0.5) · Re(Ψ* · lap(Ψ)) == c² · (-0.5)(pr·lr + pi·li)
        const double node_grd = static_cast<double>(cfg.c2) * (-0.5) * (pr*lr + pi*li);
        // Nonlinear: (β/4) |Ψ|⁴
        const double psi_sq   = pr*pr + pi*pi;
        const double node_nl  = static_cast<double>(cfg.beta) * 0.25 * (psi_sq * psi_sq);

        // Kahan add — kinetic
        {   double y = node_kin - kin_c; double t = kin + y; kin_c = (t - kin) - y; kin = t; }
        // Kahan add — gradient
        {   double y = node_grd - grd_c; double t = grd + y; grd_c = (t - grd) - y; grd = t; }
        // Kahan add — nonlinear
        {   double y = node_nl  - nl_c;  double t = nl  + y; nl_c  = (t - nl)  - y; nl  = t; }
    }

    const double dV = static_cast<double>(cfg.dV);
    GpuHamiltonianTerms out;
    out.kinetic   = kin  * dV;
    out.gradient  = grd  * dV;
    out.nonlinear = nl   * dV;
    out.total     = out.kinetic + out.gradient + out.nonlinear;
    return out;
}

// ============================================================================
// GpuHamiltonianOracle — GPU-aware class with CUDA runtime device queries
// ============================================================================

/**
 * @brief Stateful oracle for GPU-resident Hamiltonian computation.
 *
 * Wraps configuration, device metadata, and the compute interface.  On
 * systems where CUDA is available the static helpers query the first CUDA
 * device; the computation path is always the host kernel (GPU dispatch is
 * provided by the companion `gpu_hamiltonian.cu`, loaded as `nikola_cuda`).
 *
 * @code
 *   GpuFieldBuffer buf;
 *   buf.resize(1024, 0.0f);
 *   // ... fill buf from propagator ...
 *
 *   GpuHamiltonianOracle oracle;
 *   oracle.set_beta(1.0f).set_c2(1.0f).set_dV(0.001f);
 *
 *   auto terms = oracle.compute(buf);
 *   // terms.total  — total energy
 *   // terms.kinetic, .gradient, .nonlinear — components
 *
 *   double drift = GpuHamiltonianOracle::check_drift(H0, terms.total);
 * @endcode
 */
class GpuHamiltonianOracle {
public:
    // ------------------------------------------------------------------ construction

    GpuHamiltonianOracle() = default;

    explicit GpuHamiltonianOracle(GpuHamiltonianConfig cfg)
        : cfg_(cfg) {}

    // ------------------------------------------------------------------ configuration

    GpuHamiltonianOracle& set_beta(float b)  noexcept { cfg_.beta = b; return *this; }
    GpuHamiltonianOracle& set_c2  (float c2) noexcept { cfg_.c2   = c2; return *this; }
    GpuHamiltonianOracle& set_dV  (float dV) noexcept { cfg_.dV   = dV; return *this; }

    [[nodiscard]] float beta() const noexcept { return cfg_.beta; }
    [[nodiscard]] float c2()   const noexcept { return cfg_.c2; }
    [[nodiscard]] float dV()   const noexcept { return cfg_.dV; }
    [[nodiscard]] GpuHamiltonianConfig config() const noexcept { return cfg_; }

    // ------------------------------------------------------------------ compute

    /**
     * @brief Compute Hamiltonian over @p buf with current configuration.
     *
     * Delegates to compute_hamiltonian_host(); the GPU kernel dispatch path
     * (nikola_cuda) calls the identical reduction on device and returns the
     * same result struct via cudaMemcpy.
     *
     * @throws std::invalid_argument on inconsistent buffer or dV ≤ 0.
     */
    [[nodiscard]]
    GpuHamiltonianTerms compute(const GpuFieldBuffer& buf) const {
        return compute_hamiltonian_host(buf, cfg_);
    }

    // ------------------------------------------------------------------ drift check

    /**
     * @brief Fractional energy drift |H1 - H0| / |H0|.
     *
     * Returns 0.0 when H0 == 0 to avoid division by zero at startup.
     *
     * @param H0         Reference energy (baseline or previous step).
     * @param H1         Current energy after time evolution.
     * @param tolerance  If drift > tolerance the return value exceeds 1.0×tol
     *                   and the caller should trigger SCRAM.
     * @return           Fractional drift ∈ [0, ∞).
     */
    [[nodiscard]]
    static double check_drift(double H0, double H1,
                              [[maybe_unused]] double tolerance = 1e-4) noexcept {
        if (H0 == 0.0) return 0.0;
        return std::abs(H1 - H0) / std::abs(H0);
    }

    // ------------------------------------------------------------------ CUDA device info

    /**
     * @brief True if at least one CUDA-capable GPU is present.
     *
     * Uses cudaGetDeviceCount(); returns false if CUDA runtime is absent or
     * no GPU is installed.  Always thread-safe (read-only query).
     */
    [[nodiscard]]
    static bool has_gpu() noexcept {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return (err == cudaSuccess) && (count > 0);
    }

    /**
     * @brief Human-readable name of CUDA device 0 (e.g. "NVIDIA GeForce RTX 3090").
     *
     * Returns "N/A" if no GPU is present.
     */
    [[nodiscard]]
    static std::string device_name() {
        if (!has_gpu()) return "N/A";
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return "N/A";
        return std::string(prop.name);
    }

    /**
     * @brief Compute capability as an integer (e.g. 86 for sm_86 / RTX 3090).
     *
     * Returns -1 if no GPU is present.
     */
    [[nodiscard]]
    static int device_compute_capability() noexcept {
        if (!has_gpu()) return -1;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return -1;
        return prop.major * 10 + prop.minor;
    }

    /**
     * @brief Total global memory on device 0, in bytes.
     *
     * Useful for sizing device-side SoA buffers to fit in VRAM.
     * Returns 0 if no GPU is present.
     */
    [[nodiscard]]
    static std::size_t device_total_memory() noexcept {
        if (!has_gpu()) return 0u;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 0u;
        return prop.totalGlobalMem;
    }

private:
    GpuHamiltonianConfig cfg_{};
};

} // namespace nikola::physics
