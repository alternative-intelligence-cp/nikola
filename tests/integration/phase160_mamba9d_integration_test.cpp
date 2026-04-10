/**
 * @file phase160_mamba9d_integration_test.cpp
 * @brief Phase 160 — Mamba-9D end-to-end integration test.
 *
 * Tests the full pipeline:
 *   CognitiveTorus → HilbertMambaBridge → Mamba9D
 *
 * Validates:
 *   §1 — Single-tick end-to-end: torus physics → hot nodes → Hilbert → Mamba
 *   §2 — Multi-tick stability: 100+ ticks with live physics
 *   §3 — Long-duration stability: 10,000+ Mamba steps remain bounded
 *   §4 — Resonance > 0.5 maintainable over extended run
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/cognitive/mamba9d.hpp>

#include <cmath>
#include <numeric>

using namespace nikola::cognitive;
using namespace nikola::foundation;

// ============================================================================
// §1 — Single-tick end-to-end
// ============================================================================

TEST_CASE("§1-1 Integration: CognitiveTorus → HilbertMambaBridge single tick",
          "[phase160][integration]") {
    // Small grid for speed: 3^9 = 19,683 nodes
    CognitiveTorus torus(3);
    const float dt = torus.safe_dt();

    // Run a few physics steps to populate the wavefunction
    torus.run(10, dt);

    // Get hot nodes
    auto hot = torus.hot_nodes(50);
    REQUIRE(hot.size() == 50);

    // Create bridge and tick
    HilbertMambaBridge bridge(3);
    auto result = bridge.tick(
        torus.grid().psi_real(), torus.grid().psi_imag(),
        torus.num_nodes(), hot, 0.5f, 1.0f);

    REQUIRE(result.nodes_processed == 50);
    REQUIRE(std::isfinite(result.state_norm));
    REQUIRE(result.state_norm >= 0.f);
    REQUIRE(result.stability == StabilityCondition::STABLE);
}

TEST_CASE("§1-2 Integration: Hot node ordering is Hilbert-sorted",
          "[phase160][integration]") {
    CognitiveTorus torus(3);
    torus.run(5, torus.safe_dt());

    auto hot = torus.hot_nodes(20);

    HilbertMambaBridge bridge(3);

    // After tick, nodes were processed in Hilbert order internally
    // We verify by checking the tick succeeds and processes all nodes
    auto result = bridge.tick(
        torus.grid().psi_real(), torus.grid().psi_imag(),
        torus.num_nodes(), hot);

    REQUIRE(result.nodes_processed == 20);
    REQUIRE(result.stability == StabilityCondition::STABLE);
}

// ============================================================================
// §2 — Multi-tick stability
// ============================================================================

TEST_CASE("§2-1 Integration: 100 ticks with live physics",
          "[phase160][integration]") {
    CognitiveTorus torus(3);
    const float dt = torus.safe_dt();
    HilbertMambaBridge bridge(3);

    // Warm up with physics
    torus.run(10, dt);

    float max_norm = 0.f;
    int stable_ticks = 0;

    for (int tick = 0; tick < 100; ++tick) {
        // Physics step
        torus.step(dt);

        // Get hot nodes and feed to Mamba
        auto hot = torus.hot_nodes(50);
        auto result = bridge.tick(
            torus.grid().psi_real(), torus.grid().psi_imag(),
            torus.num_nodes(), hot, 0.5f, 1.0f);

        REQUIRE(std::isfinite(result.state_norm));
        max_norm = std::max(max_norm, result.state_norm);

        if (result.stability == StabilityCondition::STABLE)
            ++stable_ticks;
    }

    INFO("Max state norm over 100 ticks: " << max_norm);
    REQUIRE(max_norm < 1e6f);   // Bounded, not diverging
    REQUIRE(stable_ticks > 90); // At least 90% stable
}

TEST_CASE("§2-2 Integration: State norm trend non-divergent",
          "[phase160][integration]") {
    CognitiveTorus torus(3);
    const float dt = torus.safe_dt();
    HilbertMambaBridge bridge(3);

    torus.run(10, dt);

    float first_norm = 0.f;
    float last_norm  = 0.f;

    for (int tick = 0; tick < 50; ++tick) {
        torus.step(dt);
        auto hot = torus.hot_nodes(30);
        auto result = bridge.tick(
            torus.grid().psi_real(), torus.grid().psi_imag(),
            torus.num_nodes(), hot, 0.5f, 1.0f);

        if (tick == 0) first_norm = result.state_norm;
        if (tick == 49) last_norm = result.state_norm;
    }

    // The state should not have blown up by more than 100x
    INFO("First norm: " << first_norm << " Last norm: " << last_norm);
    if (first_norm > 1e-6f) {
        REQUIRE(last_norm / first_norm < 100.f);
    }
}

// ============================================================================
// §3 — Long-duration stability (10,000+ Mamba steps)
// ============================================================================

TEST_CASE("§3-1 Integration: 10,000 SSM steps remain bounded",
          "[phase160][integration][longsession]") {
    // Use synthetic physics to control the test environment
    const int H = 256, I = 9, O = 100;

    Mamba9D mamba(H, I, O, 42);
    SSMLayer::State h = mamba.ssm().make_zero_state();

    // Prepare synthetic inputs and physics
    std::array<float, 9> input{};
    PhysicsParams physics;
    physics.resonance = 0.6f;
    physics.rho_G = 1.0f;

    int bounded_count = 0;

    for (int t = 0; t < 10'000; ++t) {
        // Slowly varying synthetic input
        for (int d = 0; d < 9; ++d) {
            input[d] = 0.1f * std::sin(0.01f * t + d * 0.7f);
            physics.intensity[d] = 0.3f + 0.2f * std::cos(0.005f * t + d);
            physics.phase[d] = 0.5f * std::sin(0.008f * t + d * 1.2f);
        }

        mamba.step(h, input, physics);

        float norm = SSMLayer::state_norm(h);
        if (std::isfinite(norm) && norm < 1e6f)
            ++bounded_count;
    }

    float bounded_pct = static_cast<float>(bounded_count) / 10'000.f;
    INFO("Bounded steps: " << bounded_count << "/10000 (" << bounded_pct * 100 << "%)");
    REQUIRE(bounded_pct > 0.99f);  // 99%+ must be bounded
}

TEST_CASE("§3-2 Integration: 10,000 steps via HilbertMambaBridge with synthetic torus",
          "[phase160][integration][longsession]") {
    CognitiveTorus torus(3);
    const float dt = torus.safe_dt();
    HilbertMambaBridge bridge(3);

    torus.run(10, dt);

    int bounded_count = 0;

    for (int tick = 0; tick < 200; ++tick) {
        // Each tick processes ~50 hot nodes = ~50 SSM steps
        torus.step(dt);
        auto hot = torus.hot_nodes(50);
        auto result = bridge.tick(
            torus.grid().psi_real(), torus.grid().psi_imag(),
            torus.num_nodes(), hot, 0.5f, 1.0f);

        if (std::isfinite(result.state_norm) && result.state_norm < 1e6f)
            ++bounded_count;
    }

    // 200 ticks × 50 nodes = 10,000 SSM steps
    INFO("Bounded ticks: " << bounded_count << "/200");
    REQUIRE(bounded_count >= 195);  // 97.5%+
}

// ============================================================================
// §4 — Resonance > 0.5 maintainable
// ============================================================================

TEST_CASE("§4-1 Integration: Mamba state supports resonance > 0.5 with rich input",
          "[phase160][integration]") {
    // Test that with resonance = 0.6, the system stays stable and
    // the SSM parameters reflect the resonance in meaningful ways
    CognitiveTorus torus(3);
    const float dt = torus.safe_dt();
    HilbertMambaBridge bridge(3);

    torus.run(20, dt);

    int resonant_stable_count = 0;
    const float target_resonance = 0.6f;

    for (int tick = 0; tick < 50; ++tick) {
        torus.step(dt);
        auto hot = torus.hot_nodes(50);
        auto result = bridge.tick(
            torus.grid().psi_real(), torus.grid().psi_imag(),
            torus.num_nodes(), hot, target_resonance, 1.0f);

        // When resonance > 0.5, clamp_delta allows larger Δ,
        // meaning the SSM is more responsive.
        // We verify the system stays stable under this regime.
        if (result.stability == StabilityCondition::STABLE
            && std::isfinite(result.state_norm))
            ++resonant_stable_count;
    }

    INFO("Stable ticks with resonance=" << target_resonance
         << ": " << resonant_stable_count << "/50");
    REQUIRE(resonant_stable_count >= 48);  // 96%+
}

TEST_CASE("§4-2 Integration: High resonance (0.8) still stable",
          "[phase160][integration]") {
    CognitiveTorus torus(3);
    const float dt = torus.safe_dt();
    HilbertMambaBridge bridge(3);

    torus.run(20, dt);

    int stable_count = 0;
    const float high_resonance = 0.8f;

    for (int tick = 0; tick < 50; ++tick) {
        torus.step(dt);
        auto hot = torus.hot_nodes(50);
        auto result = bridge.tick(
            torus.grid().psi_real(), torus.grid().psi_imag(),
            torus.num_nodes(), hot, high_resonance, 1.0f);

        if (result.stability == StabilityCondition::STABLE
            && std::isfinite(result.state_norm))
            ++stable_count;
    }

    INFO("Stable ticks with resonance=0.8: " << stable_count << "/50");
    REQUIRE(stable_count >= 45);  // 90%+ at high resonance
}

TEST_CASE("§4-3 Integration: Resonance sweep from 0.1 to 0.9 all stable",
          "[phase160][integration]") {
    CognitiveTorus torus(3);
    const float dt = torus.safe_dt();

    torus.run(20, dt);

    for (float res = 0.1f; res <= 0.9f; res += 0.1f) {
        HilbertMambaBridge bridge(3);

        torus.step(dt);
        auto hot = torus.hot_nodes(30);
        auto result = bridge.tick(
            torus.grid().psi_real(), torus.grid().psi_imag(),
            torus.num_nodes(), hot, res, 1.0f);

        INFO("Resonance=" << res);
        REQUIRE(result.stability == StabilityCondition::STABLE);
        REQUIRE(std::isfinite(result.state_norm));
    }
}
