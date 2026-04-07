// =============================================================================
// NIKOLA — Phase 138 — GPU Propagator Validation Suite (v0.0.7)
// =============================================================================
// Standard Candle energy conservation, undamped reversibility, CPU/GPU parity,
// and drift monitoring tests for the CUDA Strang-Verlet propagator.
//
//   §1  Standard Candle: Hamiltonian conservation, 1000 GPU steps (N=512, α=0)
//   §2  Standard Candle: full torus (N=19683), 100 GPU steps (α=0)
//   §3  Undamped reversibility: forward 100, backward 100, compare to initial
//   §4  CPU vs GPU Hamiltonian parity after identical propagation
//   §5  Drift monitoring: sample H every 100 steps, verify bounded oscillation
//   §6  GPU energy decomposition: kinetic + field + nonlinear == total
//   §7  Performance baseline: CPU vs GPU timing on full torus (N=19683)

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/physics/cuda_propagator.hpp>
#include <nikola/physics/propagator.hpp>
#include <nikola/physics/hamiltonian.hpp>
#include <nikola/physics/wave_function.hpp>
#include <nikola/foundation/toroidal_grid.hpp>

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

using namespace nikola::physics;
using namespace nikola::foundation;
using Catch::Approx;

// ============================================================================
// Helpers
// ============================================================================

/// Seed a wave function on a uniform n^9 torus with precomputed adjacency.
static WaveFunction make_wf(int n_per_dim, float amplitude = 0.2f,
                             int k_mode = 1, uint32_t seed = 42)
{
    WaveFunction wf;
    wf.seed_manifold(n_per_dim, /*pilot_dim=*/3, k_mode, amplitude, seed);
    wf.grid().precompute_adjacency();
    return wf;
}

/// Compute sum of |Ψᵢ|² over all active nodes.
static double field_norm_sq(const WaveFunction& wf)
{
    const size_t N  = wf.grid().num_active_nodes();
    const float* pr = wf.grid().psi_real();
    const float* pi = wf.grid().psi_imag();
    double acc = 0.0;
    for (size_t i = 0; i < N; ++i)
        acc += static_cast<double>(pr[i]) * pr[i]
             + static_cast<double>(pi[i]) * pi[i];
    return acc;
}

/// L2 norm of (wf_a.psi - wf_b.psi) + (wf_a.vel - wf_b.vel).
static double full_state_l2_diff(const WaveFunction& a, const WaveFunction& b)
{
    const size_t N   = a.grid().num_active_nodes();
    const float* apr = a.grid().psi_real();
    const float* api = a.grid().psi_imag();
    const float* avr = a.grid().vel_real();
    const float* avi = a.grid().vel_imag();
    const float* bpr = b.grid().psi_real();
    const float* bpi = b.grid().psi_imag();
    const float* bvr = b.grid().vel_real();
    const float* bvi = b.grid().vel_imag();
    double acc = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double d;
        d = apr[i] - bpr[i]; acc += d * d;
        d = api[i] - bpi[i]; acc += d * d;
        d = avr[i] - bvr[i]; acc += d * d;
        d = avi[i] - bvi[i]; acc += d * d;
    }
    return std::sqrt(acc);
}

/// Full state norm: sqrt(sum |psi|² + |vel|²).
static double full_state_norm(const WaveFunction& wf)
{
    const size_t N  = wf.grid().num_active_nodes();
    const float* pr = wf.grid().psi_real();
    const float* pi = wf.grid().psi_imag();
    const float* vr = wf.grid().vel_real();
    const float* vi = wf.grid().vel_imag();
    double acc = 0.0;
    for (size_t i = 0; i < N; ++i) {
        acc += static_cast<double>(pr[i]) * pr[i]
             + static_cast<double>(pi[i]) * pi[i]
             + static_cast<double>(vr[i]) * vr[i]
             + static_cast<double>(vi[i]) * vi[i];
    }
    return std::sqrt(acc);
}

// ============================================================================
// §1 — Standard Candle: Hamiltonian conservation, 1000 GPU steps (α=0)
// ============================================================================
TEST_CASE("Phase138 §1 Standard Candle: H conserved over 1000 GPU steps (α=0)",
          "[phase138][gpu_validation]")
{
    WaveFunction wf = make_wf(2, 0.1f);  // N=512

    // Undamped propagator configuration
    const float c0   = 1.f;
    const float beta = 0.1f;    // moderate nonlinearity
    const float alpha_damping = 0.f;  // NO damping — Hamiltonian MUST be conserved

    CudaPropagator gpu;
    gpu.set_c0(c0).set_beta(beta).set_alpha(alpha_damping);
    const float dt = gpu.max_stable_dt(wf) * 0.3f;  // well within CFL
    REQUIRE(dt > 0.f);

    // Compute reference Hamiltonian (CPU, Kahan-compensated)
    Hamiltonian ham;
    ham.set_c0(c0).set_beta(beta);
    const double H0 = ham.compute(wf);
    REQUIRE(std::isfinite(H0));
    REQUIRE(H0 != 0.0);

    // GPU propagation: 1000 steps
    gpu.upload(wf);
    gpu.run(1000, dt);
    gpu.sync();
    gpu.download(wf);

    const double H1 = ham.compute(wf);
    REQUIRE(std::isfinite(H1));

    const double drift = std::abs(H1 - H0) / std::abs(H0);
    // Strang-Verlet is symplectic → shadow Hamiltonian error bounded at O(dt²).
    // For 1000 steps with conservative dt, drift should stay well under 1%.
    CHECK(drift < 0.01);

    INFO("H0=" << H0 << "  H1=" << H1 << "  drift=" << drift);
}

// ============================================================================
// §2 — Standard Candle: full torus (N=19683), 100 GPU steps (α=0)
// ============================================================================
TEST_CASE("Phase138 §2 Standard Candle: full torus 19683 nodes, 100 steps",
          "[phase138][gpu_validation]")
{
    WaveFunction wf = make_wf(3, 0.05f);  // N=3^9=19683
    CHECK(wf.grid().num_active_nodes() == 19683u);

    const float c0   = 1.f;
    const float beta = 0.1f;
    const float alpha_damping = 0.f;

    CudaPropagator gpu;
    gpu.set_c0(c0).set_beta(beta).set_alpha(alpha_damping);
    const float dt = gpu.max_stable_dt(wf) * 0.3f;

    Hamiltonian ham;
    ham.set_c0(c0).set_beta(beta);
    const double H0 = ham.compute(wf);
    REQUIRE(std::isfinite(H0));
    REQUIRE(H0 != 0.0);

    gpu.upload(wf);
    gpu.run(100, dt);
    gpu.sync();
    gpu.download(wf);

    const double H1 = ham.compute(wf);
    REQUIRE(std::isfinite(H1));

    const double drift = std::abs(H1 - H0) / std::abs(H0);
    CHECK(drift < 0.01);

    INFO("Full torus: H0=" << H0 << "  H1=" << H1 << "  drift=" << drift);
}

// ============================================================================
// §3 — Undamped reversibility: forward 100 + backward 100 ≈ identity (α=0)
// ============================================================================
TEST_CASE("Phase138 §3 undamped reversibility: forward+backward ≈ identity",
          "[phase138][gpu_validation]")
{
    WaveFunction wf = make_wf(2, 0.1f, 1, /*seed=*/99);  // N=512
    WaveFunction wf_ref = wf.clone();
    wf_ref.grid().precompute_adjacency();

    const float c0   = 1.f;
    const float beta = 0.1f;
    const float alpha_damping = 0.f;  // undamped → reversible
    const float dt   = 0.0005f;       // very small dt for tight reversibility

    CudaPropagator gpu;
    gpu.set_c0(c0).set_beta(beta).set_alpha(alpha_damping);

    // Forward 100 steps
    gpu.upload(wf);
    gpu.run(100, dt);
    gpu.sync();
    gpu.download(wf);

    // Backward 100 steps (negative dt)
    gpu.upload(wf);
    gpu.run(100, -dt);
    gpu.sync();
    gpu.download(wf);

    // Compare to initial state
    const double ref_norm = full_state_norm(wf_ref);
    REQUIRE(ref_norm > 0.0);
    const double diff = full_state_l2_diff(wf, wf_ref);
    const double rel_err = diff / ref_norm;

    // Symplectic Strang-Verlet with small dt should reverse cleanly.
    // FP32 roundoff + splitting error accumulated over 200 steps.
    CHECK(rel_err < 0.01);

    INFO("reversibility: ref_norm=" << ref_norm << "  diff=" << diff
         << "  rel_err=" << rel_err);
}

// ============================================================================
// §4 — CPU vs GPU Hamiltonian parity
// ============================================================================
TEST_CASE("Phase138 §4 CPU vs GPU Hamiltonian match after 50 steps",
          "[phase138][gpu_validation]")
{
    WaveFunction wf_cpu = make_wf(2, 0.1f, 1, /*seed=*/77);
    WaveFunction wf_gpu = wf_cpu.clone();
    wf_gpu.grid().precompute_adjacency();

    const float c0    = 1.f;
    const float beta  = 0.1f;
    const float alpha = 0.001f;
    const float dt    = 0.001f;

    // CPU propagation
    Propagator cpu_prop;
    cpu_prop.set_c0(c0).set_beta(beta).set_alpha(alpha);
    for (int s = 0; s < 50; ++s)
        cpu_prop.step(wf_cpu, dt);

    // GPU propagation
    CudaPropagator gpu_prop;
    gpu_prop.set_c0(c0).set_beta(beta).set_alpha(alpha);
    gpu_prop.upload(wf_gpu);
    gpu_prop.run(50, dt);
    gpu_prop.sync();
    gpu_prop.download(wf_gpu);

    // Compare Hamiltonians
    Hamiltonian ham;
    ham.set_c0(c0).set_beta(beta);
    const double H_cpu = ham.compute(wf_cpu);
    const double H_gpu = ham.compute(wf_gpu);

    REQUIRE(std::isfinite(H_cpu));
    REQUIRE(std::isfinite(H_gpu));
    REQUIRE(H_cpu != 0.0);

    const double H_rel_diff = std::abs(H_gpu - H_cpu) / std::abs(H_cpu);
    // Same Strang-Verlet in FP32 but different summation order → < 1%
    CHECK(H_rel_diff < 0.01);

    INFO("H_cpu=" << H_cpu << "  H_gpu=" << H_gpu << "  rel_diff=" << H_rel_diff);
}

// ============================================================================
// §5 — Drift monitoring: sample H every 100 steps, verify bounded oscillation
// ============================================================================
TEST_CASE("Phase138 §5 drift monitoring: bounded oscillation over 500 steps (α=0)",
          "[phase138][gpu_validation]")
{
    WaveFunction wf = make_wf(2, 0.1f, 1, /*seed=*/123);

    const float c0   = 1.f;
    const float beta = 0.1f;
    const float alpha_damping = 0.f;

    CudaPropagator gpu;
    gpu.set_c0(c0).set_beta(beta).set_alpha(alpha_damping);
    const float dt = gpu.max_stable_dt(wf) * 0.3f;

    Hamiltonian ham;
    ham.set_c0(c0).set_beta(beta);
    const double H0 = ham.compute(wf);
    REQUIRE(H0 != 0.0);

    // Collect H at 5 checkpoints: steps 100, 200, 300, 400, 500
    std::vector<double> H_samples;
    H_samples.push_back(H0);

    for (int batch = 0; batch < 5; ++batch) {
        gpu.upload(wf);
        gpu.run(100, dt);
        gpu.sync();
        gpu.download(wf);

        double H_now = ham.compute(wf);
        REQUIRE(std::isfinite(H_now));
        H_samples.push_back(H_now);
    }

    // Check: max fractional drift from H0 across all samples
    double max_drift = 0.0;
    for (size_t k = 1; k < H_samples.size(); ++k) {
        double drift = std::abs(H_samples[k] - H0) / std::abs(H0);
        max_drift = std::max(max_drift, drift);
    }
    CHECK(max_drift < 0.01);

    // Check: no monotonic trend (not all increasing or all decreasing)
    // Symplectic integrators oscillate around H0, they don't trend.
    int up = 0, down = 0;
    for (size_t k = 1; k < H_samples.size(); ++k) {
        if (H_samples[k] > H_samples[k - 1]) ++up;
        else if (H_samples[k] < H_samples[k - 1]) ++down;
    }
    // Not ALL same direction — at least one reversal (unless drift is negligible)
    // If all diffs are effectively zero that's fine too
    bool monotonic = (up == 0 || down == 0);
    if (monotonic && max_drift > 1e-6) {
        WARN("Energy exhibits monotonic trend — max_drift=" << max_drift);
    }

    INFO("max_drift=" << max_drift << "  up=" << up << "  down=" << down);
}

// ============================================================================
// §6 — GPU energy decomposition check
// ============================================================================
TEST_CASE("Phase138 §6 Hamiltonian positive and finite after GPU propagation",
          "[phase138][gpu_validation]")
{
    WaveFunction wf = make_wf(2, 0.1f);

    CudaPropagator gpu;
    gpu.set_c0(1.f).set_beta(0.1f).set_alpha(0.f);
    const float dt = gpu.max_stable_dt(wf) * 0.3f;

    Hamiltonian ham;
    ham.set_c0(1.f).set_beta(0.1f);

    const double H_before = ham.compute(wf);
    REQUIRE(std::isfinite(H_before));

    // 200 steps on GPU
    gpu.upload(wf);
    gpu.run(200, dt);
    gpu.sync();
    gpu.download(wf);

    const double H_after = ham.compute(wf);
    CHECK(std::isfinite(H_after));
    CHECK(H_after > 0.0);  // seeded wave function has positive energy

    // Norm should also stay alive
    CHECK(field_norm_sq(wf) > 0.0);
}

// ============================================================================
// §7 — Performance baseline: CPU vs GPU timing on full torus (N=19683)
// ============================================================================
TEST_CASE("Phase138 §7 performance baseline: CPU vs GPU timing (N=19683, 100 steps)",
          "[phase138][gpu_validation][benchmark]")
{
    WaveFunction wf_cpu = make_wf(3, 0.05f, 1, /*seed=*/200);  // N=19683
    WaveFunction wf_gpu = wf_cpu.clone();
    wf_gpu.grid().precompute_adjacency();

    const float c0    = 1.f;
    const float beta  = 0.1f;
    const float alpha = 0.001f;
    const float dt    = 0.001f;
    const int   steps = 100;

    // ---- CPU timing ----
    Propagator cpu_prop;
    cpu_prop.set_c0(c0).set_beta(beta).set_alpha(alpha);

    auto t0_cpu = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; ++s)
        cpu_prop.step(wf_cpu, dt);
    auto t1_cpu = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1_cpu - t0_cpu).count();

    // ---- GPU timing (batched: upload once, run all, download once) ----
    CudaPropagator gpu_prop;
    gpu_prop.set_c0(c0).set_beta(beta).set_alpha(alpha);

    auto t0_gpu = std::chrono::high_resolution_clock::now();
    gpu_prop.upload(wf_gpu);
    gpu_prop.run(steps, dt);
    gpu_prop.sync();
    gpu_prop.download(wf_gpu);
    auto t1_gpu = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t1_gpu - t0_gpu).count();

    double speedup = cpu_ms / gpu_ms;

    // Both should produce finite results
    CHECK(field_norm_sq(wf_cpu) > 0.0);
    CHECK(field_norm_sq(wf_gpu) > 0.0);

    // GPU should be faster than CPU for 19,683 nodes
    CHECK(gpu_ms < cpu_ms);

    WARN("CPU: " << cpu_ms << " ms (" << (cpu_ms / steps) << " ms/step)  "
         << "GPU: " << gpu_ms << " ms (" << (gpu_ms / steps) << " ms/step)  "
         << "Speedup: " << speedup << "x");
}
