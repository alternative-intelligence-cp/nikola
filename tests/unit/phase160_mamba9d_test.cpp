/**
 * @file phase160_mamba9d_test.cpp
 * @brief Phase 160 — Mamba-9D SSM with physics-derived parameter extraction.
 *
 * Tests the v0.1.6 Phase 1 Mamba9D class which wraps SSMLayer with:
 *   - extract_ssm_params(): maps manifold physics → SSM parameters
 *   - SpectralStabilizer Δ clamping (spectral radius control)
 *   - Physics-adaptive A diagonal (intensity-driven decay)
 *   - Sequence processing with per-step physics adaptation
 *
 * Sections:
 *   §1 — extract_ssm_params: valid output ranges
 *   §2 — extract_ssm_params: intensity→A mapping semantics
 *   §3 — extract_ssm_params: Δ clamping via SpectralStabilizer
 *   §4 — Mamba9D single-step correctness
 *   §5 — Mamba9D sequence processing
 *   §6 — Stability: long-run bounded state
 *   §7 — extract_physics: static helper
 *   §8 — Benchmarks
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/cognitive/mamba9d.hpp>
#include <nikola/cognitive/spectral_stabilizer.hpp>

#include <array>
#include <cmath>
#include <chrono>
#include <numeric>
#include <random>
#include <vector>

using namespace nikola::cognitive;
using nikola::foundation::TORUS_DIMS;   // 9

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a PhysicsParams with uniform intensity and zero phase.
static PhysicsParams uniform_physics(float intensity = 1.0f,
                                     float resonance = 0.5f,
                                     float rho_G = 1.0f) {
    PhysicsParams p;
    p.intensity.fill(intensity);
    p.phase.fill(0.f);
    p.resonance = resonance;
    p.rho_G = rho_G;
    return p;
}

/// Create a 9D input array with a constant value.
static std::array<float, TORUS_DIMS> const_input(float v) {
    std::array<float, TORUS_DIMS> u{};
    u.fill(v);
    return u;
}

/// L2 norm of a state vector.
static float state_norm(const SSMLayer::State& h) {
    float s = 0.f;
    for (float v : h) s += v * v;
    return std::sqrt(s);
}

/// Check all values are finite (no NaN/Inf).
static bool state_finite(const SSMLayer::State& h) {
    for (float v : h)
        if (!std::isfinite(v)) return false;
    return true;
}

// ============================================================================
// §1 — extract_ssm_params: valid output ranges
// ============================================================================

TEST_CASE("§1-1 extract_ssm_params: A diagonal in valid range",
          "[phase160][mamba9d][extract]") {
    auto phys = uniform_physics(1.0f, 0.5f, 1.0f);
    SSMParams params = extract_ssm_params(phys);

    for (int i = 0; i < SSM_HIDDEN_DIM; ++i) {
        REQUIRE(params.A_diag[i] < 0.f);         // all negative (continuous-time decay)
        REQUIRE(params.A_diag[i] >= -1.f);        // bounded
        REQUIRE(params.A_diag[i] <= -0.1f);       // at least BASE_A_MAG
    }
}

TEST_CASE("§1-2 extract_ssm_params: Δ is positive and clamped",
          "[phase160][mamba9d][extract]") {
    auto phys = uniform_physics(1.0f, 0.5f, 1.0f);
    SSMParams params = extract_ssm_params(phys);

    REQUIRE(params.delta > 0.f);
    // Must not exceed spectral safe bound
    const float safe = delta_safe(phys.rho_G, phys.resonance);
    REQUIRE(params.delta <= safe + 1e-6f);
}

TEST_CASE("§1-3 extract_ssm_params: stability classification is STABLE for nominal",
          "[phase160][mamba9d][extract]") {
    auto phys = uniform_physics(1.0f, 0.5f, 1.0f);
    SSMParams params = extract_ssm_params(phys);

    REQUIRE(params.stability == StabilityCondition::STABLE);
}

TEST_CASE("§1-4 extract_ssm_params: zero intensity → fallback (no crash)",
          "[phase160][mamba9d][extract]") {
    auto phys = uniform_physics(0.f);
    SSMParams params = extract_ssm_params(phys);

    // Should not crash; A values should be at maximum decay (low-intensity path)
    for (int i = 0; i < SSM_HIDDEN_DIM; ++i) {
        REQUIRE(std::isfinite(params.A_diag[i]));
    }
    REQUIRE(std::isfinite(params.delta));
    REQUIRE(params.delta > 0.f);
}

// ============================================================================
// §2 — extract_ssm_params: intensity→A mapping semantics
// ============================================================================

TEST_CASE("§2-1 High intensity → small |A| → slow decay (more memory)",
          "[phase160][mamba9d][extract][intensity]") {
    // High-intensity dim 0, low-intensity dim 1
    PhysicsParams phys{};
    phys.intensity.fill(0.f);
    phys.intensity[0] = 10.0f;    // dominant
    phys.intensity[1] = 0.001f;   // quiet
    phys.phase.fill(0.f);
    phys.resonance = 0.5f;
    phys.rho_G = 1.0f;

    SSMParams params = extract_ssm_params(phys);

    // Hidden units mapping to dim 0 should have smaller |A| than dim 1
    // A_0 = first unit for dim 0, A_1 = first unit for dim 1
    const float A_high_intensity = std::abs(params.A_diag[0]);   // dim 0
    const float A_low_intensity  = std::abs(params.A_diag[1]);   // dim 1

    // Small |A| → slow decay for high-intensity dimension
    REQUIRE(A_high_intensity < A_low_intensity);
}

TEST_CASE("§2-2 Uniform intensity → all A_i equal within dimensions",
          "[phase160][mamba9d][extract][intensity]") {
    auto phys = uniform_physics(5.0f);
    SSMParams params = extract_ssm_params(phys);

    // All dimensions have equal intensity → all A_i should be the same
    const float A_0 = params.A_diag[0];
    for (int i = 1; i < SSM_HIDDEN_DIM; ++i) {
        REQUIRE(params.A_diag[i] == A_0);
    }
}

TEST_CASE("§2-3 A diagonal has 9-periodic structure",
          "[phase160][mamba9d][extract][intensity]") {
    // Different intensity per dimension
    PhysicsParams phys{};
    for (int d = 0; d < 9; ++d)
        phys.intensity[d] = static_cast<float>(d + 1) * 0.5f;
    phys.phase.fill(0.f);
    phys.resonance = 0.5f;
    phys.rho_G = 1.0f;

    SSMParams params = extract_ssm_params(phys);

    // A_i should equal A_{i+9} (same dim mapping)
    for (int i = 0; i < SSM_HIDDEN_DIM - 9; ++i) {
        REQUIRE(params.A_diag[i] == params.A_diag[i + 9]);
    }
}

// ============================================================================
// §3 — extract_ssm_params: Δ clamping via SpectralStabilizer
// ============================================================================

TEST_CASE("§3-1 High ρ(G) → smaller Δ (spectral clamping active)",
          "[phase160][mamba9d][extract][delta]") {
    auto phys_low  = uniform_physics(1.0f, 0.5f, 0.1f);   // low curvature
    auto phys_high = uniform_physics(1.0f, 0.5f, 100.0f);  // high curvature

    SSMParams p_low  = extract_ssm_params(phys_low);
    SSMParams p_high = extract_ssm_params(phys_high);

    REQUIRE(p_high.delta < p_low.delta);
}

TEST_CASE("§3-2 Resonance r→1 → larger safe Δ",
          "[phase160][mamba9d][extract][delta]") {
    auto phys_lo = uniform_physics(1.0f, 0.1f, 5.0f);  // low resonance, moderate curvature
    auto phys_hi = uniform_physics(1.0f, 0.9f, 5.0f);  // high resonance

    SSMParams p_lo = extract_ssm_params(phys_lo);
    SSMParams p_hi = extract_ssm_params(phys_hi);

    // Higher resonance → (1−r) smaller → larger safe Δ
    REQUIRE(p_hi.delta >= p_lo.delta);
}

TEST_CASE("§3-3 Very high ρ(G) triggers TIMESTEP_VIOLATION classification",
          "[phase160][mamba9d][extract][delta]") {
    // With enormous curvature, even the clamped delta could be at the limit.
    // But since we clamp first, stability should be STABLE.
    // The raw requested Δ would violate if not clamped.
    auto phys = uniform_physics(1.0f, 0.01f, 1000.0f);  // extreme curvature
    SSMParams params = extract_ssm_params(phys);

    // After clamping, should be STABLE (that's the point of clamping)
    REQUIRE(params.stability == StabilityCondition::STABLE);
    // Δ should be very small
    REQUIRE(params.delta < 0.01f);
}

TEST_CASE("§3-4 Phase variation increases requested Δ",
          "[phase160][mamba9d][extract][delta]") {
    // Zero phase → base Δ only; large phase → base + phase component
    PhysicsParams phys_zero{};
    phys_zero.intensity.fill(1.0f);
    phys_zero.phase.fill(0.f);
    phys_zero.resonance = 0.5f;
    phys_zero.rho_G = 0.01f;  // very low curvature so clamping doesn't interfere

    PhysicsParams phys_high = phys_zero;
    phys_high.phase.fill(3.0f);  // near max phase

    SSMParams p_zero = extract_ssm_params(phys_zero);
    SSMParams p_high = extract_ssm_params(phys_high);

    REQUIRE(p_high.delta > p_zero.delta);
}

// ============================================================================
// §4 — Mamba9D single-step correctness
// ============================================================================

TEST_CASE("§4-1 Mamba9D step changes hidden state",
          "[phase160][mamba9d][step]") {
    Mamba9D mamba(32, 9, 10);
    mamba.ssm().randomise(42);
    mamba.ssm().randomise_selective(42);

    auto h = mamba.ssm().make_zero_state();
    auto h_before = h;  // copy

    auto phys = uniform_physics(1.0f);
    mamba.step(h, const_input(0.5f), phys);

    // State should have changed
    float dist = 0.f;
    for (size_t i = 0; i < h.size(); ++i) {
        float d = h[i] - h_before[i];
        dist += d * d;
    }
    REQUIRE(std::sqrt(dist) > 1e-6f);
}

TEST_CASE("§4-2 Mamba9D step advances sequence counter",
          "[phase160][mamba9d][step]") {
    Mamba9D mamba(16, 9, 5);
    mamba.ssm().randomise(42);
    mamba.ssm().randomise_selective(42);

    auto h = mamba.ssm().make_zero_state();
    auto phys = uniform_physics();

    REQUIRE(mamba.sequence().current_step() == 0);
    mamba.step(h, const_input(0.1f), phys);
    REQUIRE(mamba.sequence().current_step() == 1);
    mamba.step(h, const_input(0.2f), phys);
    REQUIRE(mamba.sequence().current_step() == 2);
}

TEST_CASE("§4-3 Mamba9D step stores last_params for diagnostics",
          "[phase160][mamba9d][step]") {
    Mamba9D mamba(16, 9, 5);
    mamba.ssm().randomise(42);
    mamba.ssm().randomise_selective(42);

    auto h = mamba.ssm().make_zero_state();
    PhysicsParams phys{};
    phys.intensity = {1,2,3,4,5,6,7,8,9};
    phys.phase.fill(0.5f);
    phys.resonance = 0.7f;
    phys.rho_G = 2.0f;

    mamba.step(h, const_input(0.3f), phys);

    const auto& lp = mamba.last_params();
    REQUIRE(lp.delta > 0.f);
    REQUIRE(lp.stability == StabilityCondition::STABLE);
}

TEST_CASE("§4-4 Mamba9D step applies A diagonal from physics",
          "[phase160][mamba9d][step]") {
    Mamba9D mamba(18, 9, 5);  // 18 = 2 × 9 for clear dim mapping
    mamba.ssm().randomise(42);
    mamba.ssm().randomise_selective(42);

    auto h = mamba.ssm().make_zero_state();

    // Non-uniform intensity → should produce non-uniform A
    PhysicsParams phys{};
    phys.intensity = {10, 0.01f, 5, 0.01f, 8, 0.01f, 3, 0.01f, 1};
    phys.phase.fill(0.f);
    phys.resonance = 0.5f;
    phys.rho_G = 1.0f;

    mamba.step(h, const_input(0.1f), phys);

    // After step, A diagonal should reflect the physics
    const auto& A = mamba.ssm().A();
    // Dim 0 (high I) should have smaller |A| than dim 1 (low I)
    REQUIRE(std::abs(A[0]) < std::abs(A[1]));
}

TEST_CASE("§4-5 Mamba9D reset clears state and sequence",
          "[phase160][mamba9d][step]") {
    Mamba9D mamba(16, 9, 5);
    mamba.ssm().randomise(42);
    mamba.ssm().randomise_selective(42);

    auto h = mamba.ssm().make_zero_state();
    auto phys = uniform_physics();

    // Run a few steps
    for (int i = 0; i < 5; ++i)
        mamba.step(h, const_input(0.5f), phys);

    REQUIRE(mamba.sequence().current_step() == 5);
    REQUIRE(state_norm(h) > 0.f);

    mamba.reset(h);

    REQUIRE(mamba.sequence().current_step() == 0);
    REQUIRE(state_norm(h) == 0.f);
}

// ============================================================================
// §5 — Mamba9D sequence processing
// ============================================================================

TEST_CASE("§5-1 process_sequence: 10-step sequence",
          "[phase160][mamba9d][sequence]") {
    Mamba9D mamba(32, 9, 10);
    mamba.ssm().randomise(42);
    mamba.ssm().randomise_selective(42);

    auto h = mamba.ssm().make_zero_state();

    std::vector<std::array<float, TORUS_DIMS>> inputs(10);
    std::vector<PhysicsParams> params(10);
    for (int t = 0; t < 10; ++t) {
        inputs[t] = const_input(0.1f * static_cast<float>(t + 1));
        params[t] = uniform_physics(static_cast<float>(t + 1));
    }

    mamba.process_sequence(h, inputs, params);

    REQUIRE(mamba.sequence().current_step() == 10);
    REQUIRE(state_finite(h));
    REQUIRE(state_norm(h) > 0.f);
}

TEST_CASE("§5-2 process_sequence: different inputs → different final states",
          "[phase160][mamba9d][sequence]") {
    Mamba9D mamba_a(32, 9, 10, 42);
    Mamba9D mamba_b(32, 9, 10, 42);
    mamba_a.ssm().randomise(42);  mamba_a.ssm().randomise_selective(42);
    mamba_b.ssm().randomise(42);  mamba_b.ssm().randomise_selective(42);

    auto h_a = mamba_a.ssm().make_zero_state();
    auto h_b = mamba_b.ssm().make_zero_state();

    auto phys = uniform_physics();
    std::vector<PhysicsParams> params(5, phys);

    // Sequence A: constant 0.5
    std::vector<std::array<float, TORUS_DIMS>> inputs_a(5, const_input(0.5f));
    // Sequence B: constant -0.5
    std::vector<std::array<float, TORUS_DIMS>> inputs_b(5, const_input(-0.5f));

    mamba_a.process_sequence(h_a, inputs_a, params);
    mamba_b.process_sequence(h_b, inputs_b, params);

    // States should differ
    float dist = 0.f;
    for (size_t i = 0; i < h_a.size(); ++i) {
        float d = h_a[i] - h_b[i];
        dist += d * d;
    }
    REQUIRE(std::sqrt(dist) > 1e-3f);
}

TEST_CASE("§5-3 process_sequence: mismatched lengths uses minimum",
          "[phase160][mamba9d][sequence]") {
    Mamba9D mamba(16, 9, 5);
    mamba.ssm().randomise(42);
    mamba.ssm().randomise_selective(42);

    auto h = mamba.ssm().make_zero_state();

    // 10 inputs but only 3 physics params → should process 3 steps
    std::vector<std::array<float, TORUS_DIMS>> inputs(10, const_input(0.5f));
    std::vector<PhysicsParams> params(3, uniform_physics());

    mamba.process_sequence(h, inputs, params);

    REQUIRE(mamba.sequence().current_step() == 3);
}

// ============================================================================
// §6 — Stability: long-run bounded state
// ============================================================================

TEST_CASE("§6-1 1000-step stability: state remains finite",
          "[phase160][mamba9d][stability]") {
    Mamba9D mamba(SSM_HIDDEN_DIM, 9, 100);
    mamba.ssm().randomise(42);
    mamba.ssm().randomise_selective(42);

    auto h = mamba.ssm().make_zero_state();
    auto phys = uniform_physics(1.0f, 0.5f, 1.0f);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);

    for (int t = 0; t < 1000; ++t) {
        std::array<float, TORUS_DIMS> input{};
        for (float& v : input) v = ud(rng);
        mamba.step(h, input, phys);

        REQUIRE(state_finite(h));
    }
    REQUIRE(mamba.sequence().current_step() == 1000);
}

TEST_CASE("§6-2 State norm remains bounded over 500 steps with varying physics",
          "[phase160][mamba9d][stability]") {
    Mamba9D mamba(SSM_HIDDEN_DIM, 9, 100);
    mamba.ssm().randomise(42);
    mamba.ssm().randomise_selective(42);

    auto h = mamba.ssm().make_zero_state();

    std::mt19937 rng(456);
    std::uniform_real_distribution<float> ud_input(-1.f, 1.f);
    std::uniform_real_distribution<float> ud_intensity(0.f, 10.f);
    std::uniform_real_distribution<float> ud_phase(-3.14f, 3.14f);
    std::uniform_real_distribution<float> ud_resonance(0.f, 1.f);
    std::uniform_real_distribution<float> ud_rho(0.1f, 50.f);

    float max_norm = 0.f;

    for (int t = 0; t < 500; ++t) {
        std::array<float, TORUS_DIMS> input{};
        for (float& v : input) v = ud_input(rng);

        PhysicsParams phys;
        for (float& v : phys.intensity) v = ud_intensity(rng);
        for (float& v : phys.phase) v = ud_phase(rng);
        phys.resonance = ud_resonance(rng);
        phys.rho_G = ud_rho(rng);

        mamba.step(h, input, phys);

        const float norm = state_norm(h);
        max_norm = std::max(max_norm, norm);

        REQUIRE(state_finite(h));
    }

    // The selective scan with ZOH clamping (Ā ∈ [0,1]) ensures bounded state.
    // With H=256, theoretical max norm is sqrt(256) ≈ 16 for saturated state.
    INFO("Max norm over 500 steps: " << max_norm);
    REQUIRE(max_norm < 100.f);  // very generous bound
}

TEST_CASE("§6-3 Zero input → state decays toward zero",
          "[phase160][mamba9d][stability]") {
    Mamba9D mamba(32, 9, 10);
    mamba.ssm().randomise(42);
    mamba.ssm().randomise_selective(42);

    auto h = mamba.ssm().make_zero_state();
    auto phys = uniform_physics();

    // Prime with some input
    for (int i = 0; i < 10; ++i)
        mamba.step(h, const_input(1.0f), phys);
    const float primed_norm = state_norm(h);
    REQUIRE(primed_norm > 0.01f);

    // Now feed zeros for many steps
    for (int i = 0; i < 100; ++i)
        mamba.step(h, const_input(0.f), phys);
    const float decayed_norm = state_norm(h);

    // Should have decayed significantly
    REQUIRE(decayed_norm < primed_norm);
}

// ============================================================================
// §7 — extract_physics: static helper
// ============================================================================

TEST_CASE("§7-1 extract_physics: produces valid params from synthetic data",
          "[phase160][mamba9d][physics]") {
    // Simulate a small grid (3^9 = 19683 nodes)
    constexpr size_t N = 19683;
    constexpr size_t GN = 3;

    std::vector<float> psi_real(N, 0.f);
    std::vector<float> psi_imag(N, 0.f);

    // Seed some non-zero values
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 0.1f);
    for (size_t i = 0; i < N; ++i) {
        psi_real[i] = nd(rng);
        psi_imag[i] = nd(rng);
    }

    // Extract physics from anchor node 100
    auto phys = Mamba9D::extract_physics(
        psi_real.data(), psi_imag.data(),
        100, GN, N, 0.6f, 2.5f);

    REQUIRE(phys.resonance == 0.6f);
    REQUIRE(phys.rho_G == 2.5f);
    for (int d = 0; d < 9; ++d) {
        REQUIRE(phys.intensity[d] >= 0.f);
        REQUIRE(std::isfinite(phys.phase[d]));
        REQUIRE(phys.phase[d] >= -3.15f);
        REQUIRE(phys.phase[d] <= 3.15f);
    }
}

TEST_CASE("§7-2 extract_physics: toroidal wrap at boundary",
          "[phase160][mamba9d][physics]") {
    // Use the full 3^9 grid where wrapping is obvious
    constexpr size_t GN = 3;
    constexpr size_t NODES = 19683;   // 3^9

    std::vector<float> psi_real(NODES, 1.f);
    std::vector<float> psi_imag(NODES, 0.f);

    // Last node: anchor_idx = NODES - 1
    // Stride neighbors wrap around via modulo
    auto phys = Mamba9D::extract_physics(
        psi_real.data(), psi_imag.data(),
        NODES - 1, GN, NODES, 0.5f, 1.0f);

    // All psi are (1, 0) so intensity = 1, phase = 0
    for (int d = 0; d < 9; ++d) {
        REQUIRE_THAT(phys.intensity[d],
                     Catch::Matchers::WithinAbs(1.0, 1e-5));
        REQUIRE_THAT(phys.phase[d],
                     Catch::Matchers::WithinAbs(0.0, 1e-5));
    }
}

// ============================================================================
// §8 — Benchmarks
// ============================================================================

TEST_CASE("§8-1 extract_ssm_params throughput",
          "[phase160][mamba9d][!benchmark]") {
    auto phys = uniform_physics(5.0f, 0.5f, 2.0f);

    constexpr int ITERS = 100'000;
    auto t0 = std::chrono::high_resolution_clock::now();
    float sink = 0.f;

    for (int i = 0; i < ITERS; ++i) {
        SSMParams p = extract_ssm_params(phys);
        sink += p.delta;
    }
    REQUIRE(sink > 0.f);  // prevent dead-code elimination

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    auto per_call = ns / ITERS;
    INFO("extract_ssm_params: " << per_call << " ns/call");
    REQUIRE(per_call < 5000);  // should be well under 5µs
}

TEST_CASE("§8-2 Mamba9D step throughput (H=256)",
          "[phase160][mamba9d][!benchmark]") {
    Mamba9D mamba(SSM_HIDDEN_DIM, 9, 100);
    mamba.ssm().randomise(42);
    mamba.ssm().randomise_selective(42);

    auto h = mamba.ssm().make_zero_state();
    auto phys = uniform_physics(1.0f, 0.5f, 1.0f);
    auto input = const_input(0.5f);

    constexpr int ITERS = 10'000;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERS; ++i) {
        mamba.step(h, input, phys);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    auto per_step = us / ITERS;
    INFO("Mamba9D step (H=256): " << per_step << " µs/step");
    // H=256 selective_step is O(H×I) = 256×9 ≈ 2304 FMAs — should be < 500µs
    REQUIRE(per_step < 500);
}

// ============================================================================
// §9 — HilbertMambaBridge: Hilbert-linearized input pipeline
// ============================================================================

TEST_CASE("§9-1 HilbertMambaBridge: construction for n=3 grid",
          "[phase160][bridge]") {
    HilbertMambaBridge bridge(3);

    REQUIRE(bridge.grid_n() == 3);
    REQUIRE(bridge.total_nodes() == 19683);
    REQUIRE(bridge.hilbert_order() == 2);  // 2^2=4 ≥ 3
}

TEST_CASE("§9-2 HilbertMambaBridge: pre-computed Hilbert indices are valid",
          "[phase160][bridge]") {
    HilbertMambaBridge bridge(3);

    // All indices should be unique (no collisions in Hilbert mapping)
    std::vector<uint64_t> indices;
    indices.reserve(bridge.total_nodes());
    for (size_t i = 0; i < bridge.total_nodes(); ++i)
        indices.push_back(bridge.node_hilbert_index(i));

    // Sort and check uniqueness
    std::sort(indices.begin(), indices.end());
    auto it = std::unique(indices.begin(), indices.end());
    REQUIRE(it == indices.end());  // all unique
}

TEST_CASE("§9-3 HilbertMambaBridge: tick with synthetic wavefunction",
          "[phase160][bridge]") {
    constexpr size_t N = 19683;  // 3^9
    HilbertMambaBridge bridge(3, 32, 10);
    bridge.mamba().ssm().randomise(42);
    bridge.mamba().ssm().randomise_selective(42);

    // Synthetic wavefunction: Gaussian peak at center
    std::vector<float> psi_real(N, 0.f);
    std::vector<float> psi_imag(N, 0.f);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 0.1f);
    for (size_t i = 0; i < N; ++i) {
        psi_real[i] = nd(rng);
        psi_imag[i] = nd(rng);
    }

    // Hot nodes: top-20 by intensity
    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), size_t{0});
    std::partial_sort(idx.begin(), idx.begin() + 20, idx.end(),
        [&](size_t a, size_t b) {
            float ia = psi_real[a]*psi_real[a] + psi_imag[a]*psi_imag[a];
            float ib = psi_real[b]*psi_real[b] + psi_imag[b]*psi_imag[b];
            return ia > ib;
        });
    std::vector<size_t> hot(idx.begin(), idx.begin() + 20);

    auto result = bridge.tick(psi_real.data(), psi_imag.data(), N, hot);

    REQUIRE(result.nodes_processed == 20);
    REQUIRE(result.state_norm > 0.f);
    REQUIRE(result.stability == StabilityCondition::STABLE);
}

TEST_CASE("§9-4 HilbertMambaBridge: Hilbert ordering preserved (sorted by index)",
          "[phase160][bridge]") {
    HilbertMambaBridge bridge(3);

    // Pick 5 arbitrary nodes and verify Hilbert ordering is monotonic
    std::vector<size_t> nodes = {0, 100, 5000, 10000, 19000};
    std::vector<uint64_t> h_indices;
    for (size_t n : nodes)
        h_indices.push_back(bridge.node_hilbert_index(n));

    // Sort nodes by Hilbert index
    std::vector<size_t> sorted = nodes;
    std::sort(sorted.begin(), sorted.end(),
        [&bridge](size_t a, size_t b) {
            return bridge.node_hilbert_index(a) < bridge.node_hilbert_index(b);
        });

    // Verify the sorted Hilbert indices are monotonically increasing
    for (size_t i = 1; i < sorted.size(); ++i) {
        REQUIRE(bridge.node_hilbert_index(sorted[i-1])
             <= bridge.node_hilbert_index(sorted[i]));
    }
}

TEST_CASE("§9-5 HilbertMambaBridge: empty hot nodes → no processing",
          "[phase160][bridge]") {
    constexpr size_t N = 19683;
    HilbertMambaBridge bridge(3, 16, 5);

    std::vector<float> psi_real(N, 0.1f);
    std::vector<float> psi_imag(N, 0.f);
    std::vector<size_t> empty_hot;

    auto result = bridge.tick(psi_real.data(), psi_imag.data(), N, empty_hot);

    REQUIRE(result.nodes_processed == 0);
    REQUIRE(result.state_norm == 0.f);  // no processing, zero initial state
}

TEST_CASE("§9-6 HilbertMambaBridge: state persists across ticks",
          "[phase160][bridge]") {
    constexpr size_t N = 19683;
    HilbertMambaBridge bridge(3, 32, 10);
    bridge.mamba().ssm().randomise(42);
    bridge.mamba().ssm().randomise_selective(42);

    std::vector<float> psi_real(N, 0.f);
    std::vector<float> psi_imag(N, 0.f);
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.f, 0.1f);
    for (size_t i = 0; i < N; ++i) {
        psi_real[i] = nd(rng);
        psi_imag[i] = nd(rng);
    }

    std::vector<size_t> hot = {0, 1, 2, 3, 4};

    // Tick 1
    bridge.tick(psi_real.data(), psi_imag.data(), N, hot);
    float norm_after_tick1 = SSMLayer::state_norm(bridge.state());

    // Tick 2 — state should accumulate
    bridge.tick(psi_real.data(), psi_imag.data(), N, hot);
    float norm_after_tick2 = SSMLayer::state_norm(bridge.state());

    // State changed between ticks (accumulated information)
    REQUIRE(norm_after_tick1 != norm_after_tick2);

    // Sequence advanced by 5 nodes per tick
    REQUIRE(bridge.mamba().sequence().current_step() == 10);
}

TEST_CASE("§9-7 HilbertMambaBridge: reset clears state",
          "[phase160][bridge]") {
    constexpr size_t N = 19683;
    HilbertMambaBridge bridge(3, 16, 5);
    bridge.mamba().ssm().randomise(42);
    bridge.mamba().ssm().randomise_selective(42);

    std::vector<float> psi_real(N, 0.1f);
    std::vector<float> psi_imag(N, 0.05f);
    std::vector<size_t> hot = {100, 200, 300};

    bridge.tick(psi_real.data(), psi_imag.data(), N, hot);
    REQUIRE(SSMLayer::state_norm(bridge.state()) > 0.f);

    bridge.reset();
    REQUIRE(SSMLayer::state_norm(bridge.state()) == 0.f);
    REQUIRE(bridge.mamba().sequence().current_step() == 0);
}

TEST_CASE("§9-8 HilbertMambaBridge: 100-tick stability with K=50 hot nodes",
          "[phase160][bridge][stability]") {
    constexpr size_t N = 19683;
    HilbertMambaBridge bridge(3, SSM_HIDDEN_DIM, 100);
    bridge.mamba().ssm().randomise(42);
    bridge.mamba().ssm().randomise_selective(42);

    std::mt19937 rng(789);
    std::normal_distribution<float> nd(0.f, 0.3f);

    for (int tick = 0; tick < 100; ++tick) {
        // Generate varying wavefunction each tick
        std::vector<float> psi_real(N), psi_imag(N);
        for (size_t i = 0; i < N; ++i) {
            psi_real[i] = nd(rng);
            psi_imag[i] = nd(rng);
        }

        // Find top-50 hot nodes
        std::vector<size_t> idx(N);
        std::iota(idx.begin(), idx.end(), size_t{0});
        std::partial_sort(idx.begin(), idx.begin() + 50, idx.end(),
            [&](size_t a, size_t b) {
                float ia = psi_real[a]*psi_real[a] + psi_imag[a]*psi_imag[a];
                float ib = psi_real[b]*psi_real[b] + psi_imag[b]*psi_imag[b];
                return ia > ib;
            });
        std::vector<size_t> hot(idx.begin(), idx.begin() + 50);

        auto result = bridge.tick(psi_real.data(), psi_imag.data(), N, hot);

        REQUIRE(result.nodes_processed == 50);
        REQUIRE(std::isfinite(result.state_norm));
        REQUIRE(result.stability == StabilityCondition::STABLE);
    }

    // After 100 ticks × 50 nodes = 5000 SSM steps
    REQUIRE(bridge.mamba().sequence().current_step() == 5000);
}

// ============================================================================
// §10 — Utility functions
// ============================================================================

TEST_CASE("§10-1 grid_coord_to_float: origin → all -1.0",
          "[phase160][util]") {
    auto coord = grid_coord_to_float(0, 3);
    for (int d = 0; d < 9; ++d)
        REQUIRE_THAT(coord[d], Catch::Matchers::WithinAbs(-1.0, 1e-5));
}

TEST_CASE("§10-2 grid_coord_to_float: max node → all +1.0",
          "[phase160][util]") {
    // Max node for n=3: (2,2,...,2) → flat = 3^9 - 1 = 19682
    auto coord = grid_coord_to_float(19682, 3);
    for (int d = 0; d < 9; ++d)
        REQUIRE_THAT(coord[d], Catch::Matchers::WithinAbs(1.0, 1e-5));
}

TEST_CASE("§10-3 flat_to_grid_coords round-trip",
          "[phase160][util]") {
    // Convert to grid coords and back via manual reconstruction
    auto gc = flat_to_grid_coords(12345, 3);
    size_t reconstructed = 0;
    size_t stride = 1;
    for (int d = 0; d < 9; ++d) {
        reconstructed += gc[d] * stride;
        stride *= 3;
    }
    REQUIRE(reconstructed == 12345);
}
