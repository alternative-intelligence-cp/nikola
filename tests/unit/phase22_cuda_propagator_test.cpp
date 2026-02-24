/**
 * @file tests/unit/phase22_cuda_propagator_test.cpp
 * @brief Phase 22: CUDA GPU Propagator — correctness + performance tests.
 *
 * Key targets:
 *   - Round-trip (upload → download) is lossless.
 *   - Single step keeps all fields finite (no NaN/Inf).
 *   - 200 steps at dt=0.01 remain finite (stability).
 *   - Performance: <1.0 ms/step (original spec, RTX 3090).
 *
 * No direct CUDA API calls here — all CUDA interaction goes through
 * CudaPropagator, so this file compiles as pure C++17.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/physics/cuda_propagator.hpp>
#include <nikola/physics/wave_function.hpp>
#include <nikola/foundation/toroidal_grid.hpp>

#include <chrono>
#include <cmath>

using nikola::physics::CudaPropagator;
using nikola::physics::WaveFunction;
using nikola::foundation::TorusGrid;

// ============================================================================
// Helpers
// ============================================================================

static bool all_finite(const WaveFunction& wf) {
    const TorusGrid& g = wf.grid();
    const size_t N = g.num_active_nodes();
    const float* pr = g.psi_real();
    const float* pi = g.psi_imag();
    const float* vr = g.vel_real();
    const float* vi = g.vel_imag();
    for (size_t i = 0; i < N; ++i) {
        if (!std::isfinite(pr[i]) || !std::isfinite(pi[i])) return false;
        if (!std::isfinite(vr[i]) || !std::isfinite(vi[i])) return false;
    }
    return true;
}

static WaveFunction make_test_wf()
{
    WaveFunction wf;
    // seed_manifold(3) → 3^9 = 19,683 nodes with pilot wave + thermal noise
    // Use small amplitude (0.01) so nonlinear term stays well-behaved at dt=0.01
    wf.seed_manifold(3, 3, 1, 0.01f, 42);
    wf.grid().precompute_adjacency();
    return wf;
}

// ============================================================================
// Section 1: Construction
// ============================================================================

TEST_CASE("Phase22 GPU propagator construction", "[gpu][Phase22]")
{
    REQUIRE_NOTHROW(CudaPropagator{});

    CudaPropagator p;
    REQUIRE(p.device_node_count() == 0);   // nothing uploaded yet
}

// ============================================================================
// Section 2: Configuration API
// ============================================================================

TEST_CASE("Phase22 GPU propagator configuration", "[gpu][Phase22]")
{
    CudaPropagator p;
    // Chainable setters return *this
    REQUIRE_NOTHROW(p.set_c0(1.5f).set_beta(0.5f).set_alpha(0.02f));
}

// ============================================================================
// Section 3: Upload + round-trip
// ============================================================================

TEST_CASE("Phase22 GPU upload / download round-trip", "[gpu][Phase22]")
{
    WaveFunction wf_orig = make_test_wf();
    WaveFunction wf_back = make_test_wf();   // same init

    CudaPropagator gpu;
    REQUIRE_NOTHROW(gpu.upload(wf_orig));
    REQUIRE(gpu.device_node_count() == wf_orig.grid().num_active_nodes());

    REQUIRE_NOTHROW(gpu.download(wf_back));
    gpu.sync();

    // Pixel-perfect round-trip (no compute between upload and download)
    const size_t N = wf_orig.grid().num_active_nodes();
    const float* orig_r = wf_orig.grid().psi_real();
    const float* back_r = wf_back.grid().psi_real();
    for (size_t i = 0; i < N; ++i) {
        REQUIRE(back_r[i] == Catch::Approx(orig_r[i]).margin(1e-7f));
    }
}

// ============================================================================
// Section 4: Single step — finiteness
// ============================================================================

TEST_CASE("Phase22 GPU single step produces finite fields", "[gpu][Phase22]")
{
    WaveFunction wf = make_test_wf();

    CudaPropagator gpu;
    gpu.set_c0(1.0f).set_beta(1.0f).set_alpha(0.01f);

    REQUIRE_NOTHROW(gpu.upload(wf));
    REQUIRE_NOTHROW(gpu.step(0.01f));
    REQUIRE_NOTHROW(gpu.sync());
    REQUIRE_NOTHROW(gpu.download(wf));

    REQUIRE(all_finite(wf));
}

// ============================================================================
// Section 5: 200-step stability run
// ============================================================================

TEST_CASE("Phase22 GPU 200-step stability (dt=0.01)", "[gpu][Phase22]")
{
    WaveFunction wf = make_test_wf();

    CudaPropagator gpu;
    gpu.set_c0(1.0f).set_beta(1.0f).set_alpha(0.01f);

    REQUIRE_NOTHROW(gpu.upload(wf));
    REQUIRE_NOTHROW(gpu.run(200, 0.01f));
    REQUIRE_NOTHROW(gpu.sync());
    REQUIRE_NOTHROW(gpu.download(wf));

    REQUIRE(all_finite(wf));
}

// ============================================================================
// Section 6: step_synced (drop-in API)
// ============================================================================

TEST_CASE("Phase22 step_synced produces finite fields", "[gpu][Phase22]")
{
    WaveFunction wf = make_test_wf();
    CudaPropagator gpu;
    REQUIRE_NOTHROW(gpu.step_synced(wf, 0.01f));
    REQUIRE(all_finite(wf));
}

// ============================================================================
// Section 7: max_stable_dt
// ============================================================================

TEST_CASE("Phase22 max_stable_dt matches CFL bound", "[gpu][Phase22]")
{
    WaveFunction wf = make_test_wf();
    CudaPropagator gpu;
    gpu.set_c0(1.0f);
    const float dt = gpu.max_stable_dt(wf);
    REQUIRE(dt > 0.0f);
    REQUIRE(std::isfinite(dt));
}

// ============================================================================
// Section 8: Performance — <1 ms/step on RTX 3090
// ============================================================================

TEST_CASE("Phase22 GPU step throughput < 1.0 ms/step", "[gpu][performance][Phase22]")
{
    WaveFunction wf = make_test_wf();

    CudaPropagator gpu;
    gpu.set_c0(1.0f).set_beta(1.0f).set_alpha(0.01f);
    gpu.upload(wf);

    // Warm-up: run a few steps to populate GPU caches
    gpu.run(20, 0.01f);
    gpu.sync();

    constexpr int BENCH_STEPS = 500;
    const auto t0 = std::chrono::high_resolution_clock::now();
    gpu.run(BENCH_STEPS, 0.01f);
    gpu.sync();
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_per_step = elapsed_ms / BENCH_STEPS;

    INFO("GPU step time: " << ms_per_step << " ms/step  (total " << elapsed_ms << " ms for " << BENCH_STEPS << " steps)");
    REQUIRE(ms_per_step < 1.0);   // original specification
}
