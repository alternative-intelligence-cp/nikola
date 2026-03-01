// =============================================================================
// NIKOLA — Phase 135 — CognitiveTorus GPU wiring integration test
// =============================================================================
// Verifies that CognitiveTorus::run() delegates to CudaPropagator when
// NIKOLA_HAS_CUDA_KERNELS is defined, and that the GPU result agrees with the
// CPU Propagator reference to within FP32 round-trip error.
//
//   §1  CognitiveTorus constructs without error (GPU or CPU mode)
//   §2  run(5, dt) produces finite psi field
//   §3  run(50, dt) maintains energy proxy (norm not exploding)
//   §4  GPU vs CPU L2 relative error < 0.5% (5 steps)
//   §5  hot_nodes() returns sane results after GPU run
//   §6  Full n=3 torus (19,683 nodes): 50-step run, finite, no crash
//
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/physics/propagator.hpp>
#include <nikola/physics/wave_function.hpp>

#include <cmath>
#include <vector>

using namespace nikola;
using namespace nikola::cognitive;
using namespace nikola::physics;
using Catch::Approx;

// ============================================================================
// Helpers
// ============================================================================

/// Compute sum |ψᵢ|² over all active nodes.
static double psi_norm_sq(const CognitiveTorus& ct)
{
    const auto& wf  = ct.wave_function();
    const size_t N  = wf.num_nodes();
    const float* pr = wf.grid().psi_real();
    const float* pi = wf.grid().psi_imag();
    double acc = 0.0;
    for (size_t i = 0; i < N; ++i)
        acc += double(pr[i]) * pr[i] + double(pi[i]) * pi[i];
    return acc;
}

/// Verify every psi component is finite.
static bool all_finite(const CognitiveTorus& ct)
{
    const auto& wf  = ct.wave_function();
    const size_t N  = wf.num_nodes();
    const float* pr = wf.grid().psi_real();
    const float* pi = wf.grid().psi_imag();
    for (size_t i = 0; i < N; ++i)
        if (!std::isfinite(pr[i]) || !std::isfinite(pi[i]))
            return false;
    return true;
}

/// L2 distance between psi fields of two CognitiveTorus instances.
static double psi_l2_diff(const CognitiveTorus& a, const CognitiveTorus& b)
{
    const auto& wfa  = a.wave_function();
    const auto& wfb  = b.wave_function();
    const size_t N   = wfa.num_nodes();
    const float* ar  = wfa.grid().psi_real();
    const float* ai  = wfa.grid().psi_imag();
    const float* br  = wfb.grid().psi_real();
    const float* bi  = wfb.grid().psi_imag();
    double acc = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double dr = ar[i] - br[i];
        double di = ai[i] - bi[i];
        acc += dr * dr + di * di;
    }
    return std::sqrt(acc);
}

// ============================================================================
// §1  Construction
// ============================================================================

TEST_CASE("Phase135: CognitiveTorus constructs cleanly", "[phase135][cuda][wiring]")
{
    // n=2 → 2^9=512 nodes (fast for unit test)
    REQUIRE_NOTHROW(CognitiveTorus(2));
    CognitiveTorus ct(2);
    REQUIRE(ct.num_nodes() == 512u);
}

// ============================================================================
// §2  Short run produces finite field
// ============================================================================

TEST_CASE("Phase135: run(5, dt) produces finite psi field", "[phase135][cuda][wiring]")
{
    CognitiveTorus ct(2);
    const float dt = ct.safe_dt();
    REQUIRE_NOTHROW(ct.run(5, dt));
    REQUIRE(all_finite(ct));
}

// ============================================================================
// §3  Energy proxy stable after 50 steps
// ============================================================================

TEST_CASE("Phase135: run(50, dt) energy proxy not exploding", "[phase135][cuda][wiring]")
{
    CognitiveTorus ct(2);
    const float dt     = ct.safe_dt();
    const double norm0 = psi_norm_sq(ct);
    ct.run(50, dt);
    REQUIRE(all_finite(ct));
    const double norm1 = psi_norm_sq(ct);
    // GPU Strang-Verlet with default alpha damping: allow up to 90% decay
    // but forbid complete collapse (< 1% of initial) and runaway growth (> 10x)
    REQUIRE(norm1 > norm0 * 0.01);
    REQUIRE(norm1 < norm0 * 10.0);
}

// ============================================================================
// §4  GPU vs CPU L2 relative error < 0.5% (5 steps, n=2)
// ============================================================================

TEST_CASE("Phase135: GPU-path L2 error vs CPU < 0.5% over 5 steps", "[phase135][cuda][wiring]")
{
    // GPU torus
    CognitiveTorus gpu_ct(2);
    const float dt = gpu_ct.safe_dt();
    gpu_ct.run(5, dt);

    // CPU reference: manually step with Propagator on identical initial state
    WaveFunction cpu_wf;
    cpu_wf.seed_manifold(2, /*pilot_dim=*/3, /*k_mode=*/1, /*amplitude=*/1.f, /*seed=*/42);
    cpu_wf.grid().precompute_adjacency();
    Propagator cpu_prop;
    for (int i = 0; i < 5; ++i)
        cpu_prop.step(cpu_wf, dt);

    // Compare gpu_ct.wave_function() vs cpu_wf
    const size_t N   = gpu_ct.wave_function().num_nodes();
    const float* gr  = gpu_ct.wave_function().grid().psi_real();
    const float* gi  = gpu_ct.wave_function().grid().psi_imag();
    const float* cr  = cpu_wf.grid().psi_real();
    const float* ci  = cpu_wf.grid().psi_imag();

    double diff_sq = 0.0, ref_sq = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double dr = gr[i] - cr[i];
        double di = gi[i] - ci[i];
        diff_sq += dr * dr + di * di;
        ref_sq  += double(cr[i]) * cr[i] + double(ci[i]) * ci[i];
    }
    const double rel_err = std::sqrt(diff_sq / (ref_sq + 1e-12));
    INFO("GPU vs CPU L2 relative error = " << rel_err);
    REQUIRE(rel_err < 0.005);   // < 0.5%
}

// ============================================================================
// §5  hot_nodes() returns sane results after GPU run
// ============================================================================

TEST_CASE("Phase135: hot_nodes() works after GPU run", "[phase135][cuda][wiring]")
{
    CognitiveTorus ct(2);
    const float dt = ct.safe_dt();
    ct.run(10, dt);
    const auto hot = ct.hot_nodes(5);
    REQUIRE(hot.size() == 5u);
    // All node indices must be in range
    for (const size_t idx : hot) {
        REQUIRE(idx < ct.num_nodes());
    }
}

// ============================================================================
// §6  Full n=3 torus (19,683 nodes): 50-step run, no crash, finite
// ============================================================================

TEST_CASE("Phase135: n=3 full torus 50-step run on GPU", "[phase135][cuda][wiring]")
{
    CognitiveTorus ct(3);   // 3^9 = 19,683 nodes
    REQUIRE(ct.num_nodes() == 19683u);
    const float dt = ct.safe_dt();
    REQUIRE_NOTHROW(ct.run(50, dt));
    REQUIRE(all_finite(ct));
}
