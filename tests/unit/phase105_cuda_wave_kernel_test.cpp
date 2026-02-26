/**
 * @file tests/unit/phase105_cuda_wave_kernel_test.cpp
 * @brief Phase 105 / GAP-046 (live): CUDA wavefunction kernel — unit tests.
 *
 * Tests ACTUAL GPU execution on the RTX 3090 (sm_86).  All CUDA calls go
 * through CudaWaveKernel's host-side API; this .cpp file requires no CUDA
 * compilation — it is compiled by g++ and linked against the nvcc-compiled
 * object from cuda_wave_kernel.cu.
 *
 * Coverage:
 *   1.  Default construction — n_nodes=0, ok=true, last_error=cudaSuccess
 *   2.  allocate() + n_nodes() / ok() / last_error()
 *   3.  psi_squared_kernel: zero input → zero output
 *   4.  psi_squared_kernel: unit real input → 1.0 per element
 *   5.  psi_squared_kernel: general complex input — |3+4i|²=25, |1+1i|²=2
 *   6.  scale_field_kernel: scale by 0 → zero field
 *   7.  scale_field_kernel: scale by 2 then re-upload → doubled wavefunction
 *   8.  error state: last_error()==cudaSuccess after successful round-trip
 *   9.  Round-trip accuracy: CPU↔GPU cross-check on mixed-sign random input
 *
 * Phase  : 105
 * GAP ID : GAP-046 (live)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/infrastructure/cuda_wave_kernel.hpp>

#include <cmath>
#include <vector>
#include <numeric>   // std::iota

using Catch::Approx;
using nikola::infrastructure::CudaWaveKernel;

// ============================================================================
// Test 1 — Default construction and default state
// ============================================================================

TEST_CASE("Phase105 CudaWaveKernel default construction", "[Phase105][cuda]")
{
    CudaWaveKernel k;

    // Before allocate(): nothing allocated
    REQUIRE(k.n_nodes() == 0u);
    REQUIRE(k.ok());
    REQUIRE(k.last_error() == cudaSuccess);
}

// ============================================================================
// Test 2 — allocate(), n_nodes(), ok(), last_error()
// ============================================================================

TEST_CASE("Phase105 CudaWaveKernel allocate and status", "[Phase105][cuda]")
{
    CudaWaveKernel k;
    k.allocate(64);

    REQUIRE(k.n_nodes() == 64u);
    REQUIRE(k.ok());
    REQUIRE(k.last_error() == cudaSuccess);

    // Re-allocate to different size — should re-allocate cleanly
    k.allocate(128);
    REQUIRE(k.n_nodes() == 128u);
    REQUIRE(k.ok());

    // allocate(0) releases buffers
    k.allocate(0);
    REQUIRE(k.n_nodes() == 0u);
}

// ============================================================================
// Test 3 — psi_squared kernel: zero input → zero output
// ============================================================================

TEST_CASE("Phase105 psi_squared: zero Psi gives zero amplitude squared",
          "[Phase105][cuda]")
{
    constexpr std::size_t N = 64;
    CudaWaveKernel k;
    k.allocate(N);
    REQUIRE(k.ok());

    std::vector<float> zero_r(N, 0.0f), zero_i(N, 0.0f);
    k.upload(zero_r.data(), zero_i.data(), N);
    REQUIRE(k.ok());

    k.launch_psi_squared();
    REQUIRE(k.ok());

    std::vector<float> out(N, 99.0f);   // fill with sentinel to confirm overwrite
    k.download_output(out.data(), N);
    REQUIRE(k.ok());

    for (std::size_t i = 0; i < N; ++i)
        REQUIRE(out[i] == Approx(0.0f).margin(1e-7f));
}

// ============================================================================
// Test 4 — psi_squared kernel: unit real Ψ = 1+0i → |Ψ|² = 1
// ============================================================================

TEST_CASE("Phase105 psi_squared: unit real Psi gives amplitude squared = 1",
          "[Phase105][cuda]")
{
    constexpr std::size_t N = 128;
    CudaWaveKernel k;
    k.allocate(N);

    std::vector<float> ones(N, 1.0f), zeros(N, 0.0f);
    k.upload(ones.data(), zeros.data(), N);
    k.launch_psi_squared();

    std::vector<float> out(N, 0.0f);
    k.download_output(out.data(), N);

    REQUIRE(k.ok());
    REQUIRE(k.last_error() == cudaSuccess);
    for (std::size_t i = 0; i < N; ++i)
        REQUIRE(out[i] == Approx(1.0f).epsilon(1e-6f));
}

// ============================================================================
// Test 5 — psi_squared kernel: general complex input
// ============================================================================

TEST_CASE("Phase105 psi_squared: complex input |3+4i|²=25, |1+1i|²=2",
          "[Phase105][cuda]")
{
    constexpr std::size_t N = 4;
    CudaWaveKernel k;
    k.allocate(N);

    // Two Ψ values interleaved: [3+4i, 1+1i, 0+0i, 2+0i]
    std::vector<float> psi_r = {3.0f, 1.0f, 0.0f, 2.0f};
    std::vector<float> psi_i = {4.0f, 1.0f, 0.0f, 0.0f};
    k.upload(psi_r.data(), psi_i.data(), N);
    k.launch_psi_squared();

    std::vector<float> out(N, 0.0f);
    k.download_output(out.data(), N);

    REQUIRE(k.ok());
    REQUIRE(out[0] == Approx(25.0f).epsilon(1e-6f));   // |3+4i|² = 9+16 = 25
    REQUIRE(out[1] == Approx(2.0f).epsilon(1e-6f));    // |1+1i|² = 1+1  = 2
    REQUIRE(out[2] == Approx(0.0f).margin(1e-7f));     // |0+0i|² = 0
    REQUIRE(out[3] == Approx(4.0f).epsilon(1e-6f));    // |2+0i|² = 4
}

// ============================================================================
// Test 6 — scale_field kernel: scale by 0.0f → zero field
// ============================================================================

TEST_CASE("Phase105 scale_field: scale by 0 zeroes the wavefunction",
          "[Phase105][cuda]")
{
    constexpr std::size_t N = 64;
    CudaWaveKernel k;
    k.allocate(N);

    std::vector<float> r(N, 3.0f), im(N, 4.0f);
    k.upload(r.data(), im.data(), N);

    k.launch_scale(0.0f);   // Ψ *= 0 → all zero
    REQUIRE(k.ok());

    // Verify via psi_squared: |0|² = 0
    k.launch_psi_squared();
    std::vector<float> out(N, 99.0f);
    k.download_output(out.data(), N);

    for (std::size_t i = 0; i < N; ++i)
        REQUIRE(out[i] == Approx(0.0f).margin(1e-7f));
}

// ============================================================================
// Test 7 — scale_field kernel: scale by 2.0f → doubled wavefunction
// ============================================================================

TEST_CASE("Phase105 scale_field: scale by 2 doubles amplitude squared by 4",
          "[Phase105][cuda]")
{
    // |2Ψ|² = 4|Ψ|²
    constexpr std::size_t N = 32;
    CudaWaveKernel k;
    k.allocate(N);

    // Ψ = 1 + i   →   |Ψ|² = 2
    // 2Ψ = 2+2i   →   |2Ψ|² = 8
    std::vector<float> r(N, 1.0f), im(N, 1.0f);
    k.upload(r.data(), im.data(), N);

    k.launch_scale(2.0f);   // Ψ *= 2 → 2+2i
    REQUIRE(k.ok());

    k.launch_psi_squared();
    std::vector<float> out(N, 0.0f);
    k.download_output(out.data(), N);
    REQUIRE(k.ok());

    for (std::size_t i = 0; i < N; ++i)
        REQUIRE(out[i] == Approx(8.0f).epsilon(1e-5f));

    // Also verify that unit scale_field(1.0f) is identity
    k.upload(r.data(), im.data(), N);   // reset to 1+1i
    k.launch_scale(1.0f);
    k.launch_psi_squared();
    k.download_output(out.data(), N);
    for (std::size_t i = 0; i < N; ++i)
        REQUIRE(out[i] == Approx(2.0f).epsilon(1e-5f));   // |1+1i|² = 2
}

// ============================================================================
// Test 8 — error state: last_error() == cudaSuccess after correct sequence
// ============================================================================

TEST_CASE("Phase105 error state: all-success sequence leaves ok=true",
          "[Phase105][cuda]")
{
    CudaWaveKernel k;
    REQUIRE(k.ok());

    k.allocate(16);
    REQUIRE(k.ok());

    std::vector<float> r(16, 0.5f), im(16, 0.5f);
    k.upload(r.data(), im.data(), 16);
    REQUIRE(k.ok());

    k.launch_psi_squared();
    REQUIRE(k.ok());

    std::vector<float> out(16);
    k.download_output(out.data(), 16);
    REQUIRE(k.ok());
    REQUIRE(k.last_error() == cudaSuccess);

    k.launch_scale(1.5f);
    REQUIRE(k.ok());
    REQUIRE(k.last_error() == cudaSuccess);
}

// ============================================================================
// Test 9 — Round-trip accuracy: GPU matches CPU reference for mixed-sign data
// ============================================================================

TEST_CASE("Phase105 round-trip: GPU psi_squared matches CPU for rand input",
          "[Phase105][cuda]")
{
    constexpr std::size_t N = 256;
    CudaWaveKernel k;
    k.allocate(N);
    REQUIRE(k.ok());

    // Deterministic pseudo-random input using simple LCG
    std::vector<float> h_r(N), h_i(N);
    uint32_t state = 0xDEADBEEFu;
    for (std::size_t i = 0; i < N; ++i) {
        state = state * 1664525u + 1013904223u;
        h_r[i] = static_cast<float>(static_cast<int32_t>(state)) / 2147483648.0f;  // ∈ [-1, 1)
        state = state * 1664525u + 1013904223u;
        h_i[i] = static_cast<float>(static_cast<int32_t>(state)) / 2147483648.0f;
    }

    // CPU reference
    std::vector<float> cpu_ref(N);
    for (std::size_t i = 0; i < N; ++i)
        cpu_ref[i] = h_r[i] * h_r[i] + h_i[i] * h_i[i];

    // GPU path
    k.upload(h_r.data(), h_i.data(), N);
    k.launch_psi_squared();
    REQUIRE(k.ok());

    std::vector<float> gpu_out(N, 0.0f);
    k.download_output(gpu_out.data(), N);
    REQUIRE(k.ok());

    // Compare — within float32 rounding tolerance
    bool all_match = true;
    for (std::size_t i = 0; i < N; ++i) {
        if (std::abs(gpu_out[i] - cpu_ref[i]) > 1e-5f) {
            all_match = false;
            break;
        }
    }
    REQUIRE(all_match);

    // Sanity check: all outputs are non-negative (|Ψ|² ≥ 0)
    for (std::size_t i = 0; i < N; ++i)
        REQUIRE(gpu_out[i] >= 0.0f);
}
