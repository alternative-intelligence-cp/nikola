/**
 * @file tests/unit/phase104_gpu_hamiltonian_test.cpp
 * @brief Phase 104 / GAP-034: GPU-resident Hamiltonian oracle — unit tests.
 *
 * Validates:
 *   1.  GpuFieldBuffer — sizing, consistency, zero helper
 *   2.  GpuHamiltonianConfig — default values and setters
 *   3.  GpuHamiltonianTerms — zero-init and recompute_total
 *   4.  Zero field → zero Hamiltonian for all terms
 *   5.  Pure kinetic energy (vel≠0, Ψ=0, lap=0)
 *   6.  Pure nonlinear energy (Ψ≠0, vel=0, lap=0)
 *   7.  Gradient term (Ψ≠0, lap≠0, vel=0)
 *   8.  Configuration scaling laws (dV, β, c²)
 *   9.  CUDA device query (has_gpu, device_name, compute_capability)
 *
 * All tests compile as pure C++17 — no nvcc required.  CUDA is accessed only
 * through cuda_runtime_api.h (host-side query functions), which is linked via
 * the CUDA::cudart import target in CMakeLists.txt.
 *
 * Phase  : 104
 * GAP ID : GAP-034
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/physics/gpu_hamiltonian.hpp>

#include <cmath>
#include <string>

using Catch::Approx;

using nikola::physics::GpuFieldBuffer;
using nikola::physics::GpuHamiltonianConfig;
using nikola::physics::GpuHamiltonianOracle;
using nikola::physics::GpuHamiltonianTerms;
using nikola::physics::compute_hamiltonian_host;

// ============================================================================
// Helpers
// ============================================================================

/// Make a buffer of size N with all fields zero.
static GpuFieldBuffer make_zero_buf(std::size_t n) {
    GpuFieldBuffer b;
    b.resize(n, 0.0f);
    return b;
}

/// Make a buffer where only vel_real[i] = val (pure kinetic, real component).
static GpuFieldBuffer make_kinetic_buf(std::size_t n, float vel_r = 1.0f) {
    GpuFieldBuffer b = make_zero_buf(n);
    for (std::size_t i = 0; i < n; ++i)
        b.vel_real[i] = vel_r;
    return b;
}

/// Make a buffer where only psi_real[i] = val (pure nonlinear, no grad, no vel).
static GpuFieldBuffer make_nonlinear_buf(std::size_t n, float psi_r = 1.0f) {
    GpuFieldBuffer b = make_zero_buf(n);
    for (std::size_t i = 0; i < n; ++i)
        b.psi_real[i] = psi_r;
    return b;
}

/// Make a buffer for gradient testing: psi_real=1, lap_real=1, vel/psi_imag=0.
/// Re(Ψ* · lap) = psi_r * lap_r + psi_i * lap_i = 1*1 + 0*0 = 1
/// gradient term per node = c2 * (-0.5) * 1 = -c2/2  (negative gradient energy)
static GpuFieldBuffer make_gradient_buf(std::size_t n, float psi_r = 1.0f, float lap_r = 1.0f) {
    GpuFieldBuffer b = make_zero_buf(n);
    for (std::size_t i = 0; i < n; ++i) {
        b.psi_real[i] = psi_r;
        b.lap_real[i] = lap_r;
    }
    return b;
}

// ============================================================================
// Test 1 — GpuFieldBuffer sizing and consistency
// ============================================================================

TEST_CASE("Phase104 GpuFieldBuffer sizing and consistency", "[Phase104][gpu_hamiltonian]")
{
    // Default construction — all arrays empty
    GpuFieldBuffer empty;
    REQUIRE(empty.size() == 0u);
    REQUIRE(empty.consistent());

    // resize fills all six arrays
    GpuFieldBuffer b;
    b.resize(8, 0.0f);
    REQUIRE(b.size() == 8u);
    REQUIRE(b.consistent());
    REQUIRE(b.psi_real.size() == 8u);
    REQUIRE(b.psi_imag.size() == 8u);
    REQUIRE(b.vel_real.size() == 8u);
    REQUIRE(b.vel_imag.size() == 8u);
    REQUIRE(b.lap_real.size() == 8u);
    REQUIRE(b.lap_imag.size() == 8u);

    // After resize the values are the fill value
    for (std::size_t i = 0; i < 8; ++i) {
        REQUIRE(b.psi_real[i] == Approx(0.0f));
        REQUIRE(b.vel_real[i] == Approx(0.0f));
    }

    // Resize to different size
    b.resize(4, 1.0f);
    REQUIRE(b.size() == 4u);
    REQUIRE(b.consistent());

    // Manual inconsistency — pushing to one array
    b.psi_real.push_back(99.0f);
    REQUIRE_FALSE(b.consistent());   // psi_real is now 5, others 4

    // zero() restores values but NOT the mismatched size → inconsistency remains
    // Re-resize makes it consistent again
    b.resize(4, 2.0f);
    REQUIRE(b.consistent());

    // zero() clears values while keeping size
    b.zero();
    REQUIRE(b.size() == 4u);
    REQUIRE(b.consistent());
    REQUIRE(b.psi_real[0] == Approx(0.0f));
}

// ============================================================================
// Test 2 — GpuHamiltonianConfig defaults and setters
// ============================================================================

TEST_CASE("Phase104 GpuHamiltonianConfig defaults and setters", "[Phase104][gpu_hamiltonian]")
{
    // Default construction
    GpuHamiltonianConfig cfg;
    REQUIRE(cfg.beta == Approx(1.0f));
    REQUIRE(cfg.c2   == Approx(1.0f));
    REQUIRE(cfg.dV   == Approx(1.0f));

    // Oracle wrappers
    GpuHamiltonianOracle oracle;
    REQUIRE(oracle.beta() == Approx(1.0f));
    REQUIRE(oracle.c2()   == Approx(1.0f));
    REQUIRE(oracle.dV()   == Approx(1.0f));

    // Chained setters
    oracle.set_beta(2.5f).set_c2(0.5f).set_dV(0.1f);
    REQUIRE(oracle.beta() == Approx(2.5f));
    REQUIRE(oracle.c2()   == Approx(0.5f));
    REQUIRE(oracle.dV()   == Approx(0.1f));

    // config() returns a copy
    GpuHamiltonianConfig copy = oracle.config();
    REQUIRE(copy.beta == Approx(2.5f));
    REQUIRE(copy.c2   == Approx(0.5f));
    REQUIRE(copy.dV   == Approx(0.1f));
}

// ============================================================================
// Test 3 — GpuHamiltonianTerms defaults and recompute_total
// ============================================================================

TEST_CASE("Phase104 GpuHamiltonianTerms defaults and recompute_total", "[Phase104][gpu_hamiltonian]")
{
    GpuHamiltonianTerms t;
    REQUIRE(t.kinetic   == Approx(0.0));
    REQUIRE(t.gradient  == Approx(0.0));
    REQUIRE(t.nonlinear == Approx(0.0));
    REQUIRE(t.total     == Approx(0.0));

    // Manual assignment + recompute
    t.kinetic   = 1.0;
    t.gradient  = 2.0;
    t.nonlinear = 3.0;
    t.recompute_total();
    REQUIRE(t.total == Approx(6.0));

    // Negative gradient is valid (IBP sign convention)
    t.gradient = -1.0;
    t.recompute_total();
    REQUIRE(t.total == Approx(3.0));
}

// ============================================================================
// Test 4 — Zero field gives zero Hamiltonian
// ============================================================================

TEST_CASE("Phase104 Zero field gives zero Hamiltonian", "[Phase104][gpu_hamiltonian]")
{
    GpuFieldBuffer buf = make_zero_buf(64);
    GpuHamiltonianConfig cfg;   // beta=1, c2=1, dV=1

    auto terms = compute_hamiltonian_host(buf, cfg);

    REQUIRE(terms.kinetic   == Approx(0.0).margin(1e-15));
    REQUIRE(terms.gradient  == Approx(0.0).margin(1e-15));
    REQUIRE(terms.nonlinear == Approx(0.0).margin(1e-15));
    REQUIRE(terms.total     == Approx(0.0).margin(1e-15));

    // Oracle interface gives same result
    GpuHamiltonianOracle oracle;
    auto terms2 = oracle.compute(buf);
    REQUIRE(terms2.total == Approx(0.0).margin(1e-15));
}

// ============================================================================
// Test 5 — Pure kinetic energy
// ============================================================================

TEST_CASE("Phase104 Pure kinetic energy (vel != 0, psi = 0, lap = 0)", "[Phase104][gpu_hamiltonian]")
{
    // N nodes, vel_real = v0, all other fields = 0
    // Per node kinetic = 0.5 * v0^2
    // Total kinetic = N * 0.5 * v0^2 * dV
    constexpr std::size_t N  = 100;
    constexpr float       v0 = 2.0f;
    constexpr float       dV = 0.5f;

    GpuFieldBuffer buf = make_kinetic_buf(N, v0);

    GpuHamiltonianConfig cfg;
    cfg.dV = dV;

    auto t = compute_hamiltonian_host(buf, cfg);

    const double expected_kinetic = static_cast<double>(N) * 0.5 * v0 * v0 * dV;

    REQUIRE(t.kinetic   == Approx(expected_kinetic).epsilon(1e-6));
    REQUIRE(t.gradient  == Approx(0.0).margin(1e-12));
    REQUIRE(t.nonlinear == Approx(0.0).margin(1e-12));
    REQUIRE(t.total     == Approx(expected_kinetic).epsilon(1e-6));

    // Kinetic must be positive
    REQUIRE(t.kinetic > 0.0);

    // Complex velocity — add vel_imag contribution
    GpuFieldBuffer buf2 = make_zero_buf(N);
    for (std::size_t i = 0; i < N; ++i) {
        buf2.vel_real[i] = 1.0f;
        buf2.vel_imag[i] = 1.0f;   // |V|² = 1² + 1² = 2  → node_kin = 1.0
    }
    auto t2 = compute_hamiltonian_host(buf2, cfg);
    const double expected2 = static_cast<double>(N) * 1.0 * dV;  // 0.5 * 2 = 1.0 per node
    REQUIRE(t2.kinetic == Approx(expected2).epsilon(1e-6));
    REQUIRE(t2.total   == Approx(expected2).epsilon(1e-6));
}

// ============================================================================
// Test 6 — Pure nonlinear energy
// ============================================================================

TEST_CASE("Phase104 Pure nonlinear energy (psi != 0, vel = 0, lap = 0)", "[Phase104][gpu_hamiltonian]")
{
    // Per node: |Ψ|² = psi_r² + psi_i²
    // nonlinear = (β/4)|Ψ|⁴ = (β/4)(psi_r² + psi_i²)²
    // With psi_r=1, psi_i=0 → nonlinear = β/4 per node
    constexpr std::size_t N     = 50;
    constexpr float       psi_r = 1.0f;
    constexpr float       beta  = 2.0f;
    constexpr float       dV    = 1.0f;

    GpuFieldBuffer buf = make_nonlinear_buf(N, psi_r);

    GpuHamiltonianConfig cfg;
    cfg.beta = beta;
    cfg.dV   = dV;

    auto t = compute_hamiltonian_host(buf, cfg);

    const double expected_nl = static_cast<double>(N) * (beta / 4.0) * 1.0 * dV;

    REQUIRE(t.kinetic   == Approx(0.0).margin(1e-12));
    REQUIRE(t.gradient  == Approx(0.0).margin(1e-12));
    REQUIRE(t.nonlinear == Approx(expected_nl).epsilon(1e-6));
    REQUIRE(t.total     == Approx(expected_nl).epsilon(1e-6));
    REQUIRE(t.nonlinear > 0.0);

    // Complex Ψ: psi_r² + psi_i² = √2² = 2 → |Ψ|⁴ = 4
    GpuFieldBuffer buf3 = make_zero_buf(N);
    for (std::size_t i = 0; i < N; ++i) {
        buf3.psi_real[i] = 1.0f;
        buf3.psi_imag[i] = 1.0f;
    }
    auto t3 = compute_hamiltonian_host(buf3, cfg);
    const double expected_nl3 = static_cast<double>(N) * (beta / 4.0) * 4.0 * dV;
    REQUIRE(t3.nonlinear == Approx(expected_nl3).epsilon(1e-6));
}

// ============================================================================
// Test 7 — Gradient term
// ============================================================================

TEST_CASE("Phase104 Gradient term (psi != 0, lap != 0, vel = 0)", "[Phase104][gpu_hamiltonian]")
{
    // With psi_real=1, lap_real=1, vel=0, psi_imag=lap_imag=0:
    // Re(Ψ* · lap) = psi_r * lap_r + psi_i * lap_i = 1
    // gradient per node = c2 * (-0.5) * 1 = -c2/2  (negative)
    // Total = N * (-c2/2) * dV
    constexpr std::size_t N    = 32;
    constexpr float       c2   = 4.0f;
    constexpr float       dV   = 0.25f;

    GpuFieldBuffer buf = make_gradient_buf(N, /*psi_r=*/1.0f, /*lap_r=*/1.0f);

    GpuHamiltonianConfig cfg;
    cfg.c2 = c2;
    cfg.dV = dV;

    auto t = compute_hamiltonian_host(buf, cfg);

    const double expected_grd = static_cast<double>(N) * c2 * (-0.5) * 1.0 * dV;

    REQUIRE(t.kinetic   == Approx(0.0).margin(1e-12));
    REQUIRE(t.gradient  == Approx(expected_grd).epsilon(1e-6));
    // Nonlinear is NOT zero here (beta=1, psi_real=1 → (β/4)|Ψ|⁴ = 0.25 per node)
    // Gradient magnitude is still correct; isolate it via beta=0 below.

    GpuHamiltonianConfig cfg2;
    cfg2.beta = 0.0f;
    cfg2.c2   = c2;
    cfg2.dV   = dV;

    auto t2 = compute_hamiltonian_host(buf, cfg2);
    REQUIRE(t2.kinetic   == Approx(0.0).margin(1e-12));
    REQUIRE(t2.gradient  == Approx(expected_grd).epsilon(1e-6));
    REQUIRE(t2.nonlinear == Approx(0.0).margin(1e-12));

    // Gradient is negative for psi aligned with laplacian (typical for standing wave)
    REQUIRE(t2.gradient < 0.0);

    // Opposite sign of laplacian → positive gradient
    GpuFieldBuffer buf_neg = make_gradient_buf(N, /*psi_r=*/1.0f, /*lap_r=*/-1.0f);
    auto t3 = compute_hamiltonian_host(buf_neg, cfg2);
    REQUIRE(t3.gradient > 0.0);
}

// ============================================================================
// Test 8 — Configuration scaling laws
// ============================================================================

TEST_CASE("Phase104 Configuration scaling: dV, beta, c2", "[Phase104][gpu_hamiltonian]")
{
    constexpr std::size_t N = 64;

    // --- dV doubling doubles total H ---
    {
        GpuFieldBuffer buf = make_kinetic_buf(N, 1.0f);
        GpuHamiltonianConfig cfg1; cfg1.dV = 1.0f;
        GpuHamiltonianConfig cfg2; cfg2.dV = 2.0f;
        auto t1 = compute_hamiltonian_host(buf, cfg1);
        auto t2 = compute_hamiltonian_host(buf, cfg2);
        REQUIRE(t2.total == Approx(2.0 * t1.total).epsilon(1e-6));
        REQUIRE(t2.kinetic == Approx(2.0 * t1.kinetic).epsilon(1e-6));
    }

    // --- beta doubling doubles nonlinear term ---
    {
        GpuFieldBuffer buf = make_nonlinear_buf(N, 1.0f);
        GpuHamiltonianConfig cfg1; cfg1.beta = 1.0f;
        GpuHamiltonianConfig cfg2; cfg2.beta = 2.0f;
        auto t1 = compute_hamiltonian_host(buf, cfg1);
        auto t2 = compute_hamiltonian_host(buf, cfg2);
        REQUIRE(t2.nonlinear == Approx(2.0 * t1.nonlinear).epsilon(1e-6));
    }

    // --- c2 doubling doubles magnitude of gradient term ---
    {
        GpuFieldBuffer buf = make_gradient_buf(N, 1.0f, 1.0f);
        GpuHamiltonianConfig cfg1; cfg1.beta = 0.0f; cfg1.c2 = 1.0f;
        GpuHamiltonianConfig cfg2; cfg2.beta = 0.0f; cfg2.c2 = 2.0f;
        auto t1 = compute_hamiltonian_host(buf, cfg1);
        auto t2 = compute_hamiltonian_host(buf, cfg2);
        REQUIRE(t2.gradient == Approx(2.0 * t1.gradient).epsilon(1e-6));
    }

    // --- zero beta → nonlinear == 0 even when psi != 0 ---
    {
        GpuFieldBuffer buf = make_nonlinear_buf(N, 3.0f);
        GpuHamiltonianConfig cfg; cfg.beta = 0.0f;
        auto t = compute_hamiltonian_host(buf, cfg);
        REQUIRE(t.nonlinear == Approx(0.0).margin(1e-12));
    }

    // --- invalid dV throws ---
    {
        GpuFieldBuffer buf = make_zero_buf(N);
        GpuHamiltonianConfig bad_cfg; bad_cfg.dV = 0.0f;
        REQUIRE_THROWS_AS(compute_hamiltonian_host(buf, bad_cfg),
                          std::invalid_argument);
        bad_cfg.dV = -1.0f;
        REQUIRE_THROWS_AS(compute_hamiltonian_host(buf, bad_cfg),
                          std::invalid_argument);
    }

    // --- inconsistent buffer throws ---
    {
        GpuFieldBuffer bad_buf;
        bad_buf.psi_real.assign(4, 0.0f);   // intentionally mismatched sizes
        REQUIRE_THROWS_AS(compute_hamiltonian_host(bad_buf, GpuHamiltonianConfig{}),
                          std::invalid_argument);
    }
}

// ============================================================================
// Test 9 — CUDA device query (RTX 3090 expected)
// ============================================================================

TEST_CASE("Phase104 CUDA device query — RTX 3090", "[Phase104][gpu_hamiltonian][cuda]")
{
    // System under test: RTX 3090, sm_86, CUDA 12.0 (confirmed by build env)
    bool gpu = GpuHamiltonianOracle::has_gpu();
    REQUIRE(gpu);   // RTX 3090 must be present

    std::string name = GpuHamiltonianOracle::device_name();
    REQUIRE_FALSE(name.empty());
    REQUIRE(name != "N/A");
    // Name should identify NVIDIA hardware
    bool is_nvidia = (name.find("NVIDIA") != std::string::npos)
                  || (name.find("GeForce")  != std::string::npos)
                  || (name.find("RTX")      != std::string::npos)
                  || (name.find("Quadro")   != std::string::npos)
                  || (name.find("Tesla")    != std::string::npos);
    REQUIRE(is_nvidia);

    int cc = GpuHamiltonianOracle::device_compute_capability();
    REQUIRE(cc > 0);
    REQUIRE(cc >= 70);   // minimum CC 7.0 per spec

    std::size_t mem = GpuHamiltonianOracle::device_total_memory();
    REQUIRE(mem > 0u);
    REQUIRE(mem >= 1024u * 1024u * 1024u);   // ≥ 1 GB  (RTX 3090 has 24 GB)

    // check_drift edge cases
    REQUIRE(GpuHamiltonianOracle::check_drift(0.0, 5.0) == Approx(0.0));
    REQUIRE(GpuHamiltonianOracle::check_drift(10.0, 10.0) == Approx(0.0).margin(1e-15));
    REQUIRE(GpuHamiltonianOracle::check_drift(10.0, 11.0) == Approx(0.1).epsilon(1e-9));
}
