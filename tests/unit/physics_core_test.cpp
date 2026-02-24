/**
 * @file physics_core_test.cpp
 * @brief Unit tests for Phase 1: Core Physics Engine.
 *
 * Tests cover:
 *   - Morton128 encode/decode round-trip
 *   - TorusGrid node management and toroidal neighbour lookup
 *   - WaveFunction seeding (Manifold Seeder, IMP-03)
 *   - Hamiltonian energy computation
 *   - Propagator substep correctness (damping, kinetic, nonlinear)
 *   - Energy conservation: |ΔH/H| < 0.01% over 10,000 steps
 *   - Numerical stability over 10,000 steps
 *   - Performance: single physics step < 5ms on 3^9 = 19683-node grid
 *
 * Reference: nikola Phase 1 gate criteria.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/foundation/complex_field.hpp>
#include <nikola/foundation/toroidal_grid.hpp>
#include <nikola/physics/wave_function.hpp>
#include <nikola/physics/hamiltonian.hpp>
#include <nikola/physics/propagator.hpp>

#include <chrono>
#include <cmath>
#include <array>

using namespace nikola::foundation;
using namespace nikola::physics;

// ============================================================================
// Foundation: Morton128
// ============================================================================

TEST_CASE("Morton128 encode-decode round-trip", "[foundation][morton]") {

    SECTION("All-zero coordinates") {
        std::array<int, 9> coords{};
        const Morton128 code = encode_morton(coords);
        REQUIRE(code == 0);
        const auto dec = decode_morton(code);
        REQUIRE(dec == coords);
    }

    SECTION("All-one coordinates") {
        std::array<int, 9> coords;
        coords.fill(1);
        const Morton128 code = encode_morton(coords);
        const auto dec = decode_morton(code);
        REQUIRE(dec == coords);
    }

    SECTION("Distinct single-bit coordinates") {
        for (int d = 0; d < 9; ++d) {
            std::array<int, 9> c{};
            c[d] = 1;
            const auto dec = decode_morton(encode_morton(c));
            REQUIRE(dec == c);
        }
    }

    SECTION("Mixed coordinates") {
        std::array<int, 9> c = {5, 12, 3, 127, 7, 0, 31, 15, 63};
        const auto dec = decode_morton(encode_morton(c));
        REQUIRE(dec == c);
    }

    SECTION("Different coords produce different codes") {
        std::array<int, 9> c1{};
        std::array<int, 9> c2{};
        c2[0] = 1;
        REQUIRE(encode_morton(c1) != encode_morton(c2));
    }
}

// ============================================================================
// Foundation: TorusGrid
// ============================================================================

TEST_CASE("TorusGrid node management", "[foundation][torus_grid]") {

    SECTION("Add and find nodes") {
        TorusGrid g(GridConfig::uniform(4));
        std::array<int, 9> c{};
        const size_t idx = g.add_node(c);
        REQUIRE(idx == 0);
        REQUIRE(g.num_active_nodes() == 1);
        REQUIRE(g.find_node(c) == 0);
    }

    SECTION("Duplicate add returns same index") {
        TorusGrid g(GridConfig::uniform(4));
        std::array<int, 9> c{};
        const size_t idx0 = g.add_node(c);
        const size_t idx1 = g.add_node(c);
        REQUIRE(idx0 == idx1);
        REQUIRE(g.num_active_nodes() == 1);
    }

    SECTION("Missing node returns VACUUM_NODE") {
        TorusGrid g(GridConfig::uniform(4));
        std::array<int, 9> c{};
        REQUIRE(g.find_node(c) == VACUUM_NODE);
    }

    SECTION("Toroidal coordinate wrapping") {
        TorusGrid g(GridConfig::uniform(4));
        REQUIRE(g.wrap_coord(-1, 0)  == 3);
        REQUIRE(g.wrap_coord(4,  0)  == 0);
        REQUIRE(g.wrap_coord(5,  0)  == 1);
        REQUIRE(g.wrap_coord(0,  0)  == 0);
    }

    SECTION("Node state set/get round-trip") {
        TorusGrid g(GridConfig::uniform(4));
        std::array<int, 9> c{};
        TorusNode n;
        n.psi = {3.f, 4.f};
        n.vel = {1.f, 2.f};
        n.resonance = 0.7f;
        n.state_field = 0.1f;

        const size_t idx = g.add_node(c, n);
        const auto readback = g.get_node(idx);

        REQUIRE(readback.psi.real() == Catch::Approx(3.f));
        REQUIRE(readback.psi.imag() == Catch::Approx(4.f));
        REQUIRE(readback.vel.real() == Catch::Approx(1.f));
        REQUIRE(readback.resonance  == Catch::Approx(0.7f));
    }
}

TEST_CASE("TorusGrid neighbour lookup", "[foundation][torus_grid][neighbors]") {

    SECTION("3-node ring in 1D (embedded in 9D)") {
        // On a 3-node torus: node 0 neighbours are 1 and 2.
        TorusGrid g(GridConfig::uniform(3));
        for (int i = 0; i < 3; ++i) {
            std::array<int, 9> c{};
            c[0] = i;
            g.add_node(c);
        }

        const size_t idx0 = g.find_node({0,0,0,0,0,0,0,0,0});
        const size_t idxp = g.find_node({1,0,0,0,0,0,0,0,0});
        const size_t idxm = g.find_node({2,0,0,0,0,0,0,0,0});

        // Dimension 0 neighbours of node 0: +1→1, -1→2
        auto nbrs = g.get_neighbors(idx0);
        REQUIRE(nbrs[0] == idxp);   // +e_0
        REQUIRE(nbrs[1] == idxm);   // -e_0
    }

    SECTION("2-node torus: both neighbours are the same node") {
        // Resolution 2 in dim 0: node 0's +1 and -1 both wrap to node 1.
        TorusGrid g(GridConfig::uniform(2));
        g.add_node({0,0,0,0,0,0,0,0,0});
        g.add_node({1,0,0,0,0,0,0,0,0});

        const size_t idx0 = g.find_node({0,0,0,0,0,0,0,0,0});
        const size_t idx1 = g.find_node({1,0,0,0,0,0,0,0,0});

        auto nbrs = g.get_neighbors(idx0);
        // +e_0: coords {1,0,...} → idx1
        REQUIRE(nbrs[0] == idx1);
        // -e_0: wrap(-1,2)=1, coords {1,0,...} → idx1
        REQUIRE(nbrs[1] == idx1);
    }

    SECTION("fill_dense_cube creates correct count") {
        TorusGrid g(GridConfig::uniform(3));
        g.fill_dense_cube(3);
        // 3^9 = 19683 nodes
        REQUIRE(g.num_active_nodes() == 19683u);
    }
}

TEST_CASE("TorusGrid precomputed adjacency matches on-demand", "[foundation][torus_grid]") {
    TorusGrid g(GridConfig::uniform(3));
    g.fill_dense_cube(3);
    g.precompute_adjacency();
    REQUIRE(g.adjacency_valid());

    // Check first 10 nodes: precomputed must match get_neighbors()
    for (size_t i = 0; i < std::min(size_t(10), g.num_active_nodes()); ++i) {
        const auto expected = g.get_neighbors(i);   // uses hash-map
        const size_t* fast = g.get_neighbors_fast(i);
        for (int n = 0; n < 18; ++n) {
            REQUIRE(fast[n] == expected[n]);
        }
    }
}

// ============================================================================
// ComplexField utilities
// ============================================================================

TEST_CASE("ComplexField utilities", "[foundation][complex_field]") {

    SECTION("magnitude_sq") {
        Complex c{3.f, 4.f};
        REQUIRE(magnitude_sq(c) == Catch::Approx(25.f));
    }

    SECTION("kahan_sum cancellation") {
        // Sum: +1, +1000000, -1000000 should equal 1
        std::array<Complex, 3> vals = {Complex{1e6f, 0.f}, Complex{1.f, 0.f}, Complex{-1e6f, 0.f}};
        const Complex s = kahan_sum(vals);
        REQUIRE(std::abs(s.real() - 1.f) < 0.01f);
    }

    SECTION("pilot_wave periodic boundary") {
        // At x=N, pilot_wave wraps to x=0
        const float v0 = pilot_wave(0, 4).real();
        const float v4 = pilot_wave(4, 4).real();
        REQUIRE(v0 == Catch::Approx(v4).margin(1e-5f));
    }

    SECTION("pml_ghost damps amplitude") {
        const Complex c{1.f, 0.f};
        const Complex ghost = pml_ghost(c, 0.9f);
        REQUIRE(magnitude(ghost) < magnitude(c));
        REQUIRE(magnitude(ghost) == Catch::Approx(0.9f));
    }

    SECTION("thermal_sigma scales with metric trace") {
        const float sigma1 = thermal_sigma(9.f);   // flat metric
        const float sigma2 = thermal_sigma(36.f);  // 4× more curvature
        REQUIRE(sigma2 == Catch::Approx(sigma1 * 2.f).margin(1e-7f));
    }
}

// ============================================================================
// WaveFunction
// ============================================================================

TEST_CASE("WaveFunction seeding (Manifold Seeder IMP-03)", "[physics][wave_function]") {

    SECTION("seed_manifold: finite fields") {
        WaveFunction wf;
        wf.seed_manifold(2);   // 2^9 = 512 nodes
        REQUIRE(wf.is_finite());
    }

    SECTION("seed_manifold: pilot wave creates non-zero energy") {
        WaveFunction wf;
        wf.seed_manifold(2, /*pilot_dim=*/3, /*k_mode=*/1, /*amplitude=*/1.f);
        REQUIRE(wf.total_probability() > 0.0);
    }

    SECTION("seed_manifold: node count") {
        WaveFunction wf;
        wf.seed_manifold(2);
        REQUIRE(wf.num_nodes() == 512u);   // 2^9
    }

    SECTION("apply_emergency_damping reduces kinetic energy") {
        WaveFunction wf;
        wf.seed_manifold(2);
        const double ke_before = wf.total_kinetic_energy();
        wf.apply_emergency_damping(0.5f);
        const double ke_after = wf.total_kinetic_energy();
        REQUIRE(ke_after < ke_before);
    }

    SECTION("max_amplitude ≥ 0") {
        WaveFunction wf;
        wf.seed_manifold(2);
        REQUIRE(wf.max_amplitude() >= 0.f);
    }
}

// ============================================================================
// Hamiltonian
// ============================================================================

TEST_CASE("Hamiltonian energy computation", "[physics][hamiltonian]") {

    SECTION("Seeded state has finite, positive energy") {
        WaveFunction wf;
        wf.seed_manifold(2, 3, 1, 1.f, 42);
        Hamiltonian ham;
        ham.set_c0(1.f).set_beta(1.f);
        const double H = ham.compute(wf);
        REQUIRE(std::isfinite(H));
        REQUIRE(H > 0.0);
    }

    SECTION("verify_initial_conditions passes on valid state") {
        WaveFunction wf;
        wf.seed_manifold(2, 3, 1, 1.f, 42);
        Hamiltonian ham;
        REQUIRE_NOTHROW(ham.verify_initial_conditions(wf));
    }

    SECTION("check_drift: zero drift below tolerance is OK") {
        Hamiltonian ham;
        REQUIRE_NOTHROW(ham.check_drift(100.0, 100.0, 1e-4));
    }

    SECTION("check_drift: large drift triggers callback") {
        Hamiltonian ham;
        bool called = false;
        ham.check_drift(100.0, 200.0, 1e-4,
            [&](double, double, double){ called = true; });
        REQUIRE(called);
    }

    SECTION("Hamiltonian increases with injected amplitude") {
        // Larger amplitude → higher nonlinear and gradient energy
        WaveFunction wf_lo, wf_hi;
        wf_lo.seed_manifold(2, 3, 1, 0.5f);
        wf_hi.seed_manifold(2, 3, 1, 2.0f);
        Hamiltonian ham;
        REQUIRE(ham.compute(wf_hi) > ham.compute(wf_lo));
    }
}

// ============================================================================
// Propagator substeps
// ============================================================================

TEST_CASE("Propagator damping substep", "[physics][propagator]") {

    SECTION("Large alpha decays velocity toward zero") {
        WaveFunction wf;
        wf.seed_manifold(2, 3, 1, 0.f, 42);  // no pilot wave, just thermal vel

        // Manually set non-trivial velocity
        auto& g = wf.grid();
        for (size_t i = 0; i < g.num_active_nodes(); ++i) {
            auto n = g.get_node(i);
            n.vel = {1.f, 0.f};
            n.resonance = 0.f;   // r=0 → max damping
            g.set_node(i, n);
        }

        Propagator p;
        p.set_alpha(10.f);
        p.step_damping(wf, 0.5f);

        // After D(τ) with α=10, r=0: V *= exp(-10·1·0.5) = exp(-5) ≈ 0.0067
        for (size_t i = 0; i < g.num_active_nodes(); ++i) {
            REQUIRE(std::abs(g.vel_real()[i]) < 0.01f);
        }
    }

    SECTION("Zero alpha: damping is identity") {
        WaveFunction wf;
        wf.seed_manifold(2, 3, 1, 0.f, 42);
        auto& g = wf.grid();
        // Set initial velocity
        const float v0 = 1.5f;
        for (size_t i = 0; i < g.num_active_nodes(); ++i) {
            auto n = g.get_node(i);
            n.vel = {v0, 0.f};
            g.set_node(i, n);
        }

        Propagator p;
        p.set_alpha(0.f);
        p.step_damping(wf, 1.0f);

        for (size_t i = 0; i < g.num_active_nodes(); ++i) {
            REQUIRE(g.vel_real()[i] == Catch::Approx(v0));
        }
    }
}

TEST_CASE("Propagator nonlinear substep", "[physics][propagator]") {

    SECTION("Non-zero psi accelerates velocity") {
        WaveFunction wf;
        wf.seed_manifold(2, 3, 1, 1.f, 42);
        auto& g = wf.grid();

        const float vel_before = g.vel_real()[0];
        const float psi_r = g.psi_real()[0];
        const float psi_i = g.psi_imag()[0];
        const float psi_sq = psi_r*psi_r + psi_i*psi_i;

        Propagator prop;
        prop.set_beta(1.f);
        prop.step_nonlinear(wf, 0.01f);

        const float vel_after = g.vel_real()[0];
        const float expected_delta = 1.f * psi_sq * psi_r * 0.01f;
        REQUIRE(vel_after == Catch::Approx(vel_before + expected_delta).margin(1e-5f));
    }

    SECTION("Zero beta: nonlinear substep is identity on velocity") {
        WaveFunction wf;
        wf.seed_manifold(2, 3, 1, 1.f, 42);
        auto& g = wf.grid();
        const float vel_before = g.vel_real()[0];

        Propagator prop;
        prop.set_beta(0.f);
        prop.step_nonlinear(wf, 1.0f);

        REQUIRE(g.vel_real()[0] == Catch::Approx(vel_before));
    }
}

// ============================================================================
// Energy conservation  (Phase 1 gate criterion)
// ============================================================================

TEST_CASE("Energy conservation: free wave on 2^9 grid", "[physics][energy_conservation]") {
    // Free wave: no damping, no nonlinearity
    // Integration over 10,000 steps with dt = 0.01
    // Gate: |ΔH/H₀| < 0.01% (1e-4)
    //
    // Grid: 2 nodes per dimension → 2^9 = 512 nodes (fast)
    // CFL: dt < h/(c0·sqrt(9)) = 1/(1·3) = 0.333  → dt=0.01 is safe

    WaveFunction wf;
    wf.seed_manifold(2, /*pilot_dim=*/3, /*k_mode=*/1, /*amplitude=*/1.f, /*seed=*/42);

    Propagator prop;
    prop.set_c0(1.f).set_beta(0.f).set_alpha(0.f);   // pure free wave

    Hamiltonian ham;
    ham.set_c0(1.f).set_beta(0.f);

    const double H0 = ham.compute(wf);
    REQUIRE(H0 > 0.0);
    REQUIRE(std::isfinite(H0));

    double max_drift = 0.0;
    const float dt = 0.01f;
    const int steps = 10'000;

    for (int s = 0; s < steps; ++s) {
        prop.step(wf, dt);
        if ((s + 1) % 500 == 0) {
            const double H = ham.compute(wf);
            const double drift = std::abs(H - H0) / (H0 + 1e-30);
            if (drift > max_drift) max_drift = drift;
        }
    }

    INFO("Max energy drift over " << steps << " steps: "
         << max_drift * 100.0 << "% (limit 0.01%)");
    REQUIRE(max_drift < 1e-4);   // < 0.01%
}

// ============================================================================
// Stability
// ============================================================================

TEST_CASE("Propagator stability: field remains finite over 10,000 steps",
          "[physics][stability]") {

    // Small 2^9 grid, realistic physics (non-zero damping + nonlinearity)
    WaveFunction wf;
    wf.seed_manifold(2, 3, 1, 0.1f, 99);   // small amplitude to stay in linear regime

    Propagator prop;
    prop.set_c0(1.f).set_beta(0.01f).set_alpha(0.01f);

    for (int s = 0; s < 10'000; ++s) {
        prop.step(wf, 0.01f);
    }

    REQUIRE(wf.is_finite());
}

// ============================================================================
// Performance  (Phase 1 gate: single step < 5ms on 3^9 = 19,683 node grid)
// ============================================================================

TEST_CASE("Performance: physics step on 3^9 grid", "[physics][performance]") {
    // Allocate 3^9 = 19683 nodes
    WaveFunction wf;
    wf.seed_manifold(3, 3, 1, 1.f, 42);
    REQUIRE(wf.num_nodes() == 19683u);

    Propagator prop;
    prop.set_c0(1.f).set_beta(1.f).set_alpha(0.01f);

    // Warm-up (also precomputes adjacency)
    prop.step(wf, 0.001f);

    // Time 10 steps
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        prop.step(wf, 0.001f);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double ms_per_step = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10.0;

    INFO("Physics step time on 3^9 grid: " << ms_per_step << " ms (limit: 5ms)");
#ifdef NDEBUG
    // Release: hard gate at 5ms (CPU Phase 1 target; GPU target is <1ms)
    REQUIRE(ms_per_step < 5.0);
#else
    // Debug: no inlining / no optimisation — allow up to 30ms
    REQUIRE(ms_per_step < 30.0);
#endif
}

// ============================================================================
// Bootstrap validation  (IMP-03 Step 7: Oracle gates)
// ============================================================================

TEST_CASE("Manifold Seeder bootstrap validation", "[physics][bootstrap]") {
    WaveFunction wf;
    wf.seed_manifold(2, 3, 1, 1.f, 42);   // Pilot wave in time dimension

    Hamiltonian ham;
    ham.set_c0(1.f).set_beta(1.f);

    SECTION("Fields are finite") {
        REQUIRE(wf.is_finite());
    }

    SECTION("Total energy > 0") {
        REQUIRE(ham.compute(wf) > 0.0);
    }

    SECTION("verify_initial_conditions does not throw") {
        REQUIRE_NOTHROW(ham.verify_initial_conditions(wf));
    }

    SECTION("Warm-up stabilization does not blow up") {
        // Run 100 warm-up steps with heavy damping (stabilization phase)
        Propagator prop;
        prop.set_c0(1.f).set_beta(1.f).set_alpha(10.f * DEFAULT_ALPHA);  // 10× normal damping
        for (int i = 0; i < 100; ++i) {
            prop.step(wf, 0.001f);
        }
        REQUIRE(wf.is_finite());
        REQUIRE(ham.compute(wf) >= 0.0);
    }
}
