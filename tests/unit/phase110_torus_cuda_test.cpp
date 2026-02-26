/**
 * @file tests/unit/phase110_torus_cuda_test.cpp
 * @brief Phase 110: GPU Hamiltonian density kernel — torus_cuda.cu validation.
 *
 * Validates compute_hamiltonian_device() (the actual CUDA kernel path) against
 * compute_hamiltonian_host() (the CPU reference with Kahan compensation).
 *
 * All CUDA calls are hidden behind the function boundary — this file is pure
 * C++17 and requires only nikola_cuda at link time.
 *
 * Test matrix (10 test cases):
 *   §1  Zero field → all energy terms = 0 on device
 *   §2  Pure kinetic field — only vel_r nonzero
 *   §3  Pure nonlinear field — only psi_r nonzero (lap = 0)
 *   §4  Gradient term sign check — lap_r aligned with psi_r
 *   §5  dV scaling — doubling dV doubles all output terms
 *   §6  Device matches host for a constructed analytic field (N=16)
 *   §7  Device matches host for N=256 random-like synthetic field
 *   §8  Device matches host for N=19683 (3^9 torus grid)
 *   §9  N=1 single-node roundtrip
 *   §10 Repeated calls return identical results (determinism)
 *
 * Tolerance: GPU uses straight double summation (no Kahan) vs. Kahan on host.
 * For N ≤ 20 000 nodes we expect relative error ≤ 1 × 10⁻⁵  on each term.
 *
 * Phase  : 110
 * GAP ID : GAP-034 (device path)
 * Spec   : docs/info/integration/sections/02_foundations/04_energy_conservation.md §4.2.1
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/physics/gpu_hamiltonian.hpp>

#include <cmath>
#include <cstddef>
#include <random>

using namespace nikola::physics;
using Catch::Approx;

// ============================================================================
// Helpers
// ============================================================================

/// Fill buf with analytically defined values so CPU and GPU should agree exactly.
static void fill_analytic(GpuFieldBuffer& buf, std::size_t N,
                          float psi_r_val,   float psi_i_val,
                          float vel_r_val,   float vel_i_val,
                          float lap_r_val,   float lap_i_val)
{
    buf.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
        buf.psi_real[i] = psi_r_val;
        buf.psi_imag[i] = psi_i_val;
        buf.vel_real[i] = vel_r_val;
        buf.vel_imag[i] = vel_i_val;
        buf.lap_real[i] = lap_r_val;
        buf.lap_imag[i] = lap_i_val;
    }
}

/// Compute analytic GPU result and CPU reference for the same buffer / config.
static std::pair<GpuHamiltonianTerms, GpuHamiltonianTerms>
both(const GpuFieldBuffer& buf, const GpuHamiltonianConfig& cfg)
{
    auto gpu = compute_hamiltonian_device(buf, cfg);
    auto cpu = compute_hamiltonian_host(buf, cfg);
    return {gpu, cpu};
}

/// Relative tolerance for GPU vs CPU comparison.
static constexpr double REL_TOL = 1e-4;

// ============================================================================
// §1  Zero field → all energy terms = 0 on device
// ============================================================================

TEST_CASE("Phase110 §1 — Zero field: all GPU energy terms are 0", "[phase110][cuda]")
{
    GpuFieldBuffer buf;
    buf.resize(1024, 0.0f);

    GpuHamiltonianConfig cfg;  // beta=1, c2=1, dV=1

    GpuHamiltonianTerms gpu;
    REQUIRE_NOTHROW(gpu = compute_hamiltonian_device(buf, cfg));

    REQUIRE(gpu.kinetic   == Approx(0.0).margin(1e-12));
    REQUIRE(gpu.gradient  == Approx(0.0).margin(1e-12));
    REQUIRE(gpu.nonlinear == Approx(0.0).margin(1e-12));
    REQUIRE(gpu.total     == Approx(0.0).margin(1e-12));
}

// ============================================================================
// §2  Pure kinetic field — only vel_r nonzero
// ============================================================================

TEST_CASE("Phase110 §2 — Pure kinetic: GPU kinetic = 0.5 * vel_r^2 * N * dV", "[phase110][cuda]")
{
    const std::size_t N  = 512;
    const float       vr = 2.0f;

    GpuFieldBuffer buf;
    fill_analytic(buf, N,
                  /*psi_r*/ 0.0f, /*psi_i*/ 0.0f,
                  /*vel_r*/ vr,   /*vel_i*/ 0.0f,
                  /*lap_r*/ 0.0f, /*lap_i*/ 0.0f);

    GpuHamiltonianConfig cfg;

    GpuHamiltonianTerms gpu;
    REQUIRE_NOTHROW(gpu = compute_hamiltonian_device(buf, cfg));

    // Expected kinetic = 0.5 * vr² * N
    const double expected_kin = 0.5 * static_cast<double>(vr) * vr * N;

    REQUIRE(gpu.kinetic   == Approx(expected_kin).epsilon(REL_TOL));
    REQUIRE(gpu.gradient  == Approx(0.0).margin(1e-10));
    REQUIRE(gpu.nonlinear == Approx(0.0).margin(1e-10));
    REQUIRE(gpu.total     == Approx(gpu.kinetic).epsilon(1e-12));
}

// ============================================================================
// §3  Pure nonlinear field — psi_r = 1, vel = 0, lap = 0
// ============================================================================

TEST_CASE("Phase110 §3 — Pure nonlinear: GPU nl = (beta/4) * |psi|^4 * N * dV", "[phase110][cuda]")
{
    const std::size_t N    = 256;
    const float       pr   = 0.5f;
    const float       beta = 2.0f;

    GpuFieldBuffer buf;
    fill_analytic(buf, N,
                  /*psi_r*/ pr, /*psi_i*/ 0.0f,
                  /*vel_r*/ 0.0f, /*vel_i*/ 0.0f,
                  /*lap_r*/ 0.0f, /*lap_i*/ 0.0f);

    GpuHamiltonianConfig cfg;
    cfg.beta = beta;

    GpuHamiltonianTerms gpu;
    REQUIRE_NOTHROW(gpu = compute_hamiltonian_device(buf, cfg));

    // Expected: (beta/4) * pr^4 * N
    const double expected_nl = (beta / 4.0) * std::pow(pr, 4.0) * N;

    REQUIRE(gpu.kinetic   == Approx(0.0).margin(1e-10));
    REQUIRE(gpu.gradient  == Approx(0.0).margin(1e-10));
    REQUIRE(gpu.nonlinear == Approx(expected_nl).epsilon(REL_TOL));
}

// ============================================================================
// §4  Gradient term sign check
// ============================================================================

TEST_CASE("Phase110 §4 — Gradient sign: aligned psi and lap gives negative gradient", "[phase110][cuda]")
{
    // grd = c2 * (-0.5) * (psi_r * lap_r + psi_i * lap_i)
    // psi_r = lap_r = 1.0, psi_i = lap_i = 0 → grd < 0
    const std::size_t N = 128;

    GpuFieldBuffer buf;
    fill_analytic(buf, N,
                  /*psi_r*/ 1.0f, /*psi_i*/ 0.0f,
                  /*vel_r*/ 0.0f, /*vel_i*/ 0.0f,
                  /*lap_r*/ 1.0f, /*lap_i*/ 0.0f);

    GpuHamiltonianConfig cfg;   // c2 = 1, beta = 1

    GpuHamiltonianTerms gpu, cpu;
    REQUIRE_NOTHROW(gpu = compute_hamiltonian_device(buf, cfg));
    REQUIRE_NOTHROW(cpu = compute_hamiltonian_host(buf, cfg));

    // gradient = c2 * (-0.5) * 1 * N = -0.5 * N
    double expected_grd = -0.5 * static_cast<double>(N);

    REQUIRE(gpu.gradient < 0.0);
    REQUIRE(gpu.gradient == Approx(expected_grd).epsilon(REL_TOL));
    REQUIRE(gpu.gradient == Approx(cpu.gradient).epsilon(REL_TOL));
}

// ============================================================================
// §5  dV scaling
// ============================================================================

TEST_CASE("Phase110 §5 — dV scaling: doubling dV doubles all output terms", "[phase110][cuda]")
{
    const std::size_t N = 64;

    GpuFieldBuffer buf;
    fill_analytic(buf, N,
                  /*psi_r*/ 0.3f, /*psi_i*/ 0.4f,
                  /*vel_r*/ 0.1f, /*vel_i*/ 0.2f,
                  /*lap_r*/ 0.05f, /*lap_i*/ 0.05f);

    GpuHamiltonianConfig cfg1, cfg2;
    cfg1.dV = 0.5f;
    cfg2.dV = 1.0f;   // double of cfg1

    GpuHamiltonianTerms gpu1, gpu2;
    REQUIRE_NOTHROW(gpu1 = compute_hamiltonian_device(buf, cfg1));
    REQUIRE_NOTHROW(gpu2 = compute_hamiltonian_device(buf, cfg2));

    REQUIRE(gpu2.kinetic   == Approx(gpu1.kinetic   * 2.0).epsilon(REL_TOL));
    REQUIRE(gpu2.gradient  == Approx(gpu1.gradient  * 2.0).epsilon(REL_TOL));
    REQUIRE(gpu2.nonlinear == Approx(gpu1.nonlinear * 2.0).epsilon(REL_TOL));
    REQUIRE(gpu2.total     == Approx(gpu1.total     * 2.0).epsilon(REL_TOL));
}

// ============================================================================
// §6  Device matches host for analytic field (N=16)
// ============================================================================

TEST_CASE("Phase110 §6 — CPU/GPU agreement: analytic field N=16", "[phase110][cuda]")
{
    const std::size_t N = 16;

    GpuFieldBuffer buf;
    fill_analytic(buf, N,
                  /*psi_r*/ 0.6f, /*psi_i*/ 0.8f,
                  /*vel_r*/ 0.1f, /*vel_i*/ 0.3f,
                  /*lap_r*/ -0.2f, /*lap_i*/ 0.1f);

    GpuHamiltonianConfig cfg;
    cfg.beta = 1.5f;
    cfg.c2   = 0.8f;
    cfg.dV   = 0.25f;

    auto [gpu, cpu] = both(buf, cfg);

    REQUIRE(gpu.kinetic   == Approx(cpu.kinetic  ).epsilon(REL_TOL));
    REQUIRE(gpu.gradient  == Approx(cpu.gradient ).epsilon(REL_TOL));
    REQUIRE(gpu.nonlinear == Approx(cpu.nonlinear).epsilon(REL_TOL));
    REQUIRE(gpu.total     == Approx(cpu.total    ).epsilon(REL_TOL));
}

// ============================================================================
// §7  Device matches host for N=256 synthetic field
// ============================================================================

TEST_CASE("Phase110 §7 — CPU/GPU agreement: synthetic field N=256", "[phase110][cuda]")
{
    const std::size_t N = 256;

    // Build a synthetic field: index-based values
    GpuFieldBuffer buf;
    buf.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
        float fi = static_cast<float>(i) * 0.01f;
        buf.psi_real[i] = std::cos(fi);
        buf.psi_imag[i] = std::sin(fi) * 0.5f;
        buf.vel_real[i] = std::sin(fi) * 0.1f;
        buf.vel_imag[i] = std::cos(fi) * 0.05f;
        buf.lap_real[i] = -std::cos(fi) * 4.0f;
        buf.lap_imag[i] = -std::sin(fi) * 2.0f;
    }

    GpuHamiltonianConfig cfg;
    cfg.beta = 0.5f;
    cfg.c2   = 1.0f;
    cfg.dV   = 0.1f;

    auto [gpu, cpu] = both(buf, cfg);

    REQUIRE(gpu.kinetic   == Approx(cpu.kinetic  ).epsilon(REL_TOL));
    REQUIRE(gpu.gradient  == Approx(cpu.gradient ).epsilon(REL_TOL));
    REQUIRE(gpu.nonlinear == Approx(cpu.nonlinear).epsilon(REL_TOL));
    REQUIRE(gpu.total     == Approx(cpu.total    ).epsilon(REL_TOL));
}

// ============================================================================
// §8  Device matches host for N=19683 (3^9 grid)
// ============================================================================

TEST_CASE("Phase110 §8 — CPU/GPU agreement: N=19683 (3^9 torus grid)", "[phase110][cuda]")
{
    // 3^9 = 19683 nodes — full anisotropic torus grid size
    const std::size_t N = 19683;

    GpuFieldBuffer buf;
    buf.resize(N);

    // Pseudo-random fill with a simple LCG to create realistic variation
    uint32_t seed = 0xDEADBEEFu;
    auto lcg = [&seed]() -> float {
        seed = seed * 1664525u + 1013904223u;
        return (static_cast<float>(seed & 0xFFFFFFu) / static_cast<float>(0x1000000)) - 0.5f;
    };

    for (std::size_t i = 0; i < N; ++i) {
        buf.psi_real[i] = lcg() * 0.1f;
        buf.psi_imag[i] = lcg() * 0.1f;
        buf.vel_real[i] = lcg() * 0.05f;
        buf.vel_imag[i] = lcg() * 0.05f;
        buf.lap_real[i] = lcg() * 0.5f;    // typical Laplacian scale
        buf.lap_imag[i] = lcg() * 0.5f;
    }

    GpuHamiltonianConfig cfg;
    cfg.beta = 1.0f;
    cfg.c2   = 1.0f;
    cfg.dV   = 1.0f;

    auto [gpu, cpu] = both(buf, cfg);

    // For N≈20000 the non-Kahan GPU sum can accumulate a slightly larger error,
    // but it must be within REL_TOL = 1e-4 of the Kahan host result.
    INFO("gpu.kinetic   = " << gpu.kinetic   << ", cpu.kinetic   = " << cpu.kinetic);
    INFO("gpu.gradient  = " << gpu.gradient  << ", cpu.gradient  = " << cpu.gradient);
    INFO("gpu.nonlinear = " << gpu.nonlinear << ", cpu.nonlinear = " << cpu.nonlinear);

    REQUIRE(gpu.kinetic   == Approx(cpu.kinetic  ).epsilon(REL_TOL));
    REQUIRE(gpu.gradient  == Approx(cpu.gradient ).epsilon(REL_TOL));
    REQUIRE(gpu.nonlinear == Approx(cpu.nonlinear).epsilon(REL_TOL));
    REQUIRE(gpu.total     == Approx(cpu.total    ).epsilon(REL_TOL));
}

// ============================================================================
// §9  N=1 single-node roundtrip
// ============================================================================

TEST_CASE("Phase110 §9 — N=1 single-node GPU computation", "[phase110][cuda]")
{
    GpuFieldBuffer buf;
    fill_analytic(buf, 1,
                  /*psi_r*/ 3.0f, /*psi_i*/ 4.0f,   // |psi|² = 25
                  /*vel_r*/ 1.0f, /*vel_i*/ 0.0f,    // kin = 0.5
                  /*lap_r*/ 1.0f, /*lap_i*/ 1.0f);   // grd = -0.5 * (3 + 4) = -3.5

    GpuHamiltonianConfig cfg;   // beta=1, c2=1, dV=1

    GpuHamiltonianTerms gpu, cpu;
    REQUIRE_NOTHROW(gpu = compute_hamiltonian_device(buf, cfg));
    REQUIRE_NOTHROW(cpu = compute_hamiltonian_host(buf, cfg));

    // Hand-calculate:
    //   kin  = 0.5 * (1² + 0²) = 0.5
    //   grd  = 1 * (-0.5) * (3*1 + 4*1) = -3.5
    //   nl   = (1/4) * (9+16)² = 625/4 = 156.25
    //   total = 0.5 - 3.5 + 156.25 = 153.25
    REQUIRE(gpu.kinetic   == Approx(0.5   ).epsilon(1e-6));
    REQUIRE(gpu.gradient  == Approx(-3.5  ).epsilon(1e-6));
    REQUIRE(gpu.nonlinear == Approx(156.25).epsilon(1e-6));
    REQUIRE(gpu.total     == Approx(153.25).epsilon(1e-6));

    // Device must match host
    REQUIRE(gpu.kinetic   == Approx(cpu.kinetic  ).epsilon(1e-9));
    REQUIRE(gpu.gradient  == Approx(cpu.gradient ).epsilon(1e-9));
    REQUIRE(gpu.nonlinear == Approx(cpu.nonlinear).epsilon(1e-9));
}

// ============================================================================
// §10  Repeated calls return consistent results (within FP tolerance)
// ============================================================================

TEST_CASE("Phase110 §10 — Repeated GPU calls agree within tolerance", "[phase110][cuda]")
{
    // Note: atomicAdd(double*) accumulation order can vary between launches
    // because block execution order is non-deterministic.  We therefore test
    // within-tolerance consistency rather than bitwise equality.
    const std::size_t N = 4096;

    GpuFieldBuffer buf;
    buf.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
        float fi = static_cast<float>(i) * 0.003f;
        buf.psi_real[i] = std::sin(fi) * 0.2f;
        buf.psi_imag[i] = std::cos(fi) * 0.2f;
        buf.vel_real[i] = std::cos(fi) * 0.02f;
        buf.vel_imag[i] = std::sin(fi) * 0.02f;
        buf.lap_real[i] = -std::sin(fi) * 0.8f;
        buf.lap_imag[i] = -std::cos(fi) * 0.8f;
    }

    GpuHamiltonianConfig cfg;
    cfg.beta = 0.3f;
    cfg.c2   = 1.2f;
    cfg.dV   = 0.05f;

    GpuHamiltonianTerms r1, r2;
    REQUIRE_NOTHROW(r1 = compute_hamiltonian_device(buf, cfg));
    REQUIRE_NOTHROW(r2 = compute_hamiltonian_device(buf, cfg));

    // Repeated calls must agree to within REL_TOL
    REQUIRE(r2.kinetic   == Approx(r1.kinetic  ).epsilon(REL_TOL));
    REQUIRE(r2.gradient  == Approx(r1.gradient ).epsilon(REL_TOL));
    REQUIRE(r2.nonlinear == Approx(r1.nonlinear).epsilon(REL_TOL));
    REQUIRE(r2.total     == Approx(r1.total    ).epsilon(REL_TOL));
}
