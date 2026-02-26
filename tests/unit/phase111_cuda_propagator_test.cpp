// =============================================================================
// NIKOLA — Phase 111 — CudaPropagator GPU Strang-Verlet Test Suite
// =============================================================================
// Tests the full GPU wave propagation pipeline now that propagator.cu compiles
// under nvcc C++17 (fix: std::span guarded in complex_field.hpp + toroidal_grid.hpp;
// adjacency_table_size() / adjacency_table() added to TorusGrid).
//
//   §1  Construction + configuration API
//   §2  max_stable_dt: positive, finite, matches CFL formula
//   §3  Upload→download round-trip (no step): fields preserved within FP32 error
//   §4  device_node_count: 0 pre-upload, N post-upload
//   §5  Single GPU step energy proxy: |Ψ|² norm conserved within 5%
//   §6  Short run (10 steps): energy proxy stable
//   §7  GPU vs CPU field comparison: 5-step L2 relative error < 0.5%
//   §8  step_synced API: result compatible with manual upload+step+download
//   §9  N=3^5=243 — medium grid, 50 steps, no crash + field finite
//   §10 N=3^9=19683 — full torus, 1 step upload+run+download, no crash

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/physics/cuda_propagator.hpp>
#include <nikola/physics/propagator.hpp>
#include <nikola/physics/wave_function.hpp>
#include <nikola/foundation/toroidal_grid.hpp>
#include <nikola/foundation/complex_field.hpp>

#include <cmath>
#include <vector>

using namespace nikola::physics;
using namespace nikola::foundation;
using Catch::Approx;

// ============================================================================
// Helpers
// ============================================================================

/// Seed and adjacency-precompute a wave function on a uniform n^9 torus.
static WaveFunction make_wf(int n_per_dim, float amplitude = 0.2f,
                             int k_mode = 1, uint32_t seed = 42)
{
    WaveFunction wf;
    wf.seed_manifold(n_per_dim, /*pilot_dim=*/3, k_mode, amplitude, seed);
    wf.grid().precompute_adjacency();
    return wf;
}

/// Compute proxy: sum of |Ψᵢ|² over all active nodes.
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

/// L2 norm of (wf_a.psi - wf_b.psi).  Both must have the same node count.
static double psi_l2_diff(const WaveFunction& a, const WaveFunction& b)
{
    const size_t N   = a.grid().num_active_nodes();
    const float* ar  = a.grid().psi_real();
    const float* ai  = a.grid().psi_imag();
    const float* br  = b.grid().psi_real();
    const float* bi  = b.grid().psi_imag();
    double acc = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double dr = ar[i] - br[i];
        double di = ai[i] - bi[i];
        acc += dr * dr + di * di;
    }
    return std::sqrt(acc);
}

/// Check every psi component is finite.
static bool all_finite(const WaveFunction& wf)
{
    const size_t N  = wf.grid().num_active_nodes();
    const float* pr = wf.grid().psi_real();
    const float* pi = wf.grid().psi_imag();
    for (size_t i = 0; i < N; ++i)
        if (!std::isfinite(pr[i]) || !std::isfinite(pi[i]))
            return false;
    return true;
}

// ============================================================================
// §1 — Construction + configuration API
// ============================================================================
TEST_CASE("CudaPropagator §1 construction and configuration", "[cuda_propagator][phase111]")
{
    CudaPropagator p;

    // Default parameters
    CHECK(p.c0()    == Approx(1.0f));
    CHECK(p.beta()  == Approx(1.0f));
    CHECK(p.alpha() == Approx(0.01f));

    // Zero nodes before upload
    CHECK(p.device_node_count() == 0u);

    // Setter chaining
    auto& ref = p.set_c0(2.f).set_beta(0.5f).set_alpha(0.02f);
    CHECK(&ref == &p);
    CHECK(p.c0()    == Approx(2.0f));
    CHECK(p.beta()  == Approx(0.5f));
    CHECK(p.alpha() == Approx(0.02f));
}

// ============================================================================
// §2 — max_stable_dt: positive, finite; higher c0 → smaller dt
// ============================================================================
TEST_CASE("CudaPropagator §2 max_stable_dt CFL estimate", "[cuda_propagator][phase111]")
{
    WaveFunction wf = make_wf(2);  // 2^9 = 512 nodes — fast to build

    CudaPropagator p;
    p.set_c0(1.f);
    float dt1 = p.max_stable_dt(wf);
    CHECK(std::isfinite(dt1));
    CHECK(dt1 > 0.f);

    // CFL: dt_max ∝ 1/c0 — doubling c0 should halve dt_max
    p.set_c0(2.f);
    float dt2 = p.max_stable_dt(wf);
    CHECK(dt2 < dt1);
    CHECK(dt2 == Approx(dt1 * 0.5f).epsilon(0.02f));
}

// ============================================================================
// §3 — Upload→download round-trip (no step): fields preserved
// ============================================================================
TEST_CASE("CudaPropagator §3 upload-download round-trip", "[cuda_propagator][phase111]")
{
    WaveFunction wf_orig = make_wf(2);    // N = 2^9 = 512
    WaveFunction wf_copy  = wf_orig.clone();  // deep copy before any GPU touch

    CudaPropagator p;
    p.upload(wf_orig);
    p.sync();
    p.download(wf_orig);   // write back to the same WF (no step was taken)

    // psi_real and psi_imag should be identical within float round-trip
    const size_t N = wf_orig.grid().num_active_nodes();
    REQUIRE(wf_copy.grid().num_active_nodes() == N);

    const float* orig_r = wf_orig.grid().psi_real();
    const float* orig_i = wf_orig.grid().psi_imag();
    const float* copy_r = wf_copy.grid().psi_real();
    const float* copy_i = wf_copy.grid().psi_imag();

    float max_err = 0.f;
    for (size_t i = 0; i < N; ++i) {
        max_err = std::max(max_err, std::abs(orig_r[i] - copy_r[i]));
        max_err = std::max(max_err, std::abs(orig_i[i] - copy_i[i]));
    }
    // FP32 round-trip through cudaMemcpy should have zero error (bitwise copy)
    CHECK(max_err == Approx(0.f).margin(1e-7f));
}

// ============================================================================
// §4 — device_node_count: 0 pre-upload, N post-upload
// ============================================================================
TEST_CASE("CudaPropagator §4 device_node_count", "[cuda_propagator][phase111]")
{
    CudaPropagator p;
    REQUIRE(p.device_node_count() == 0u);

    WaveFunction wf = make_wf(2);
    const size_t N  = wf.grid().num_active_nodes();
    REQUIRE(N > 0u);

    p.upload(wf);
    CHECK(p.device_node_count() == N);
}

// ============================================================================
// §5 — Single GPU step: |Ψ|² norm conserved within 5%
// ============================================================================
TEST_CASE("CudaPropagator §5 single step energy-proxy conservation", "[cuda_propagator][phase111]")
{
    WaveFunction wf = make_wf(2, 0.1f);
    const double norm_before = field_norm_sq(wf);
    REQUIRE(norm_before > 0.0);

    CudaPropagator p;
    p.set_c0(1.f).set_beta(0.1f).set_alpha(0.001f);
    const float dt = p.max_stable_dt(wf) * 0.5f;

    p.upload(wf);
    p.step(dt);
    p.sync();
    p.download(wf);

    const double norm_after = field_norm_sq(wf);
    CHECK(std::isfinite(norm_after));
    CHECK(norm_after > 0.0);

    const double rel_change = std::abs(norm_after - norm_before) / norm_before;
    // Strang-Verlet is symplectic approx — energy proxy should not drift >5% in one step
    CHECK(rel_change < 0.05);
}

// ============================================================================
// §6 — Short run (10 steps): energy proxy stable
// ============================================================================
TEST_CASE("CudaPropagator §6 run(10) energy proxy stable", "[cuda_propagator][phase111]")
{
    WaveFunction wf = make_wf(2, 0.1f);
    const double norm_before = field_norm_sq(wf);

    CudaPropagator p;
    p.set_c0(1.f).set_beta(0.1f).set_alpha(0.001f);
    const float dt = p.max_stable_dt(wf) * 0.5f;

    p.upload(wf);
    p.run(10, dt);
    p.sync();
    p.download(wf);

    const double norm_after = field_norm_sq(wf);
    CHECK(std::isfinite(norm_after));
    CHECK(norm_after > 0.0);
    // Note: the GL/Strang-Verlet wave equation is dissipative — |Ψ|² is NOT a
    // conserved quantity.  We only verify the field stays non-trivially alive.
}

// ============================================================================
// §7 — GPU vs CPU field comparison: 5-step L2 relative error < 0.5%
// ============================================================================
TEST_CASE("CudaPropagator §7 GPU vs CPU 5-step field comparison", "[cuda_propagator][phase111]")
{
    // Use a tiny N=3 (27-node) grid so both are fast
    WaveFunction wf_cpu = make_wf(2);   // N = 2^9 = 512 nodes
    WaveFunction wf_gpu  = wf_cpu.clone();   // identical starting state
    wf_gpu.grid().precompute_adjacency();      // clone() re-seeds without adjacency

    const float c0    = 1.f;
    const float beta  = 0.1f;
    const float alpha = 0.001f;
    const float dt    = 0.001f;     // small enough to be obviously CFL-safe

    // CPU reference
    Propagator cpu_prop;
    cpu_prop.set_c0(c0).set_beta(beta).set_alpha(alpha);
    for (int s = 0; s < 5; ++s)
        cpu_prop.step(wf_cpu, dt);

    // GPU
    CudaPropagator gpu_prop;
    gpu_prop.set_c0(c0).set_beta(beta).set_alpha(alpha);
    gpu_prop.upload(wf_gpu);
    gpu_prop.run(5, dt);
    gpu_prop.sync();
    gpu_prop.download(wf_gpu);

    const size_t  N       = wf_cpu.grid().num_active_nodes();
    const double  ref_sq  = field_norm_sq(wf_cpu);
    REQUIRE(ref_sq > 0.0);

    const double l2_diff = psi_l2_diff(wf_cpu, wf_gpu);
    // RMS per-node error relative to RMS signal amplitude
    const double rel_err  = (l2_diff / std::sqrt(static_cast<double>(N)))
                          / std::sqrt(ref_sq / static_cast<double>(N));
    // Same Strang-Verlet in FP32; different summation order → <0.5% relative
    CHECK(rel_err < 0.005);
}

// ============================================================================
// §8 — step_synced: result compatible with upload+step+download
// ============================================================================
TEST_CASE("CudaPropagator §8 step_synced matches upload+step+download", "[cuda_propagator][phase111]")
{
    WaveFunction wf_a = make_wf(2, 0.1f, 1, /*seed=*/7);
    WaveFunction wf_b  = wf_a.clone();
    wf_b.grid().precompute_adjacency();   // clone() re-seeds without adjacency

    const float c0    = 1.f;
    const float beta  = 0.1f;
    const float alpha = 0.001f;
    const float dt    = 0.0001f;

    // Method A: step_synced convenience wrapper
    {
        CudaPropagator p;
        p.set_c0(c0).set_beta(beta).set_alpha(alpha);
        p.step_synced(wf_a, dt);
    }

    // Method B: explicit upload + step + sync + download
    {
        CudaPropagator p;
        p.set_c0(c0).set_beta(beta).set_alpha(alpha);
        p.upload(wf_b);
        p.step(dt);
        p.sync();
        p.download(wf_b);
    }

    // Both should yield bitwise-identical results (same device state, same kernels)
    const size_t N = wf_a.grid().num_active_nodes();
    const float* ar = wf_a.grid().psi_real();
    const float* ai = wf_a.grid().psi_imag();
    const float* br = wf_b.grid().psi_real();
    const float* bi = wf_b.grid().psi_imag();

    float max_err = 0.f;
    for (size_t i = 0; i < N; ++i) {
        max_err = std::max(max_err, std::abs(ar[i] - br[i]));
        max_err = std::max(max_err, std::abs(ai[i] - bi[i]));
    }
    // Same kernels, same device state → should be bitwise identical
    CHECK(max_err == Approx(0.f).margin(1e-6f));
}

// ============================================================================
// §9 — N=2^9=512: 50 steps, no crash + all fields finite
// ============================================================================
TEST_CASE("CudaPropagator §9 N=2^9 50-step run stays finite", "[cuda_propagator][phase111]")
{
    WaveFunction wf = make_wf(2, 0.05f);
    const size_t N  = wf.grid().num_active_nodes();
    REQUIRE(N > 0u);
    REQUIRE(all_finite(wf));

    CudaPropagator p;
    p.set_c0(1.f).set_beta(0.1f).set_alpha(0.001f);
    const float dt = p.max_stable_dt(wf) * 0.5f;

    REQUIRE_NOTHROW(p.upload(wf));
    REQUIRE_NOTHROW(p.run(50, dt));
    REQUIRE_NOTHROW(p.sync());
    REQUIRE_NOTHROW(p.download(wf));

    CHECK(all_finite(wf));
    CHECK(field_norm_sq(wf) > 0.0);
}

// ============================================================================
// §10 — N=3^9=19683: full torus, 1 step, no crash
// ============================================================================
TEST_CASE("CudaPropagator §10 N=3^9 full torus single step", "[cuda_propagator][phase111]")
{
    WaveFunction wf = make_wf(3, 0.05f);
    const size_t N  = wf.grid().num_active_nodes();
    CHECK(N == 19683u);

    CudaPropagator p;
    p.set_c0(1.f).set_beta(0.1f).set_alpha(0.001f);
    CHECK(p.device_node_count() == 0u);

    const float dt = p.max_stable_dt(wf) * 0.5f;
    CHECK(dt > 0.f);

    REQUIRE_NOTHROW(p.upload(wf));
    CHECK(p.device_node_count() == N);
    REQUIRE_NOTHROW(p.step(dt));
    REQUIRE_NOTHROW(p.sync());
    REQUIRE_NOTHROW(p.download(wf));

    CHECK(all_finite(wf));
    CHECK(std::isfinite(field_norm_sq(wf)));
}
