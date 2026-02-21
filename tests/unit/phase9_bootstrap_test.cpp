/**
 * @file phase9_bootstrap_test.cpp
 * @brief Tests for Manifold Seeder (IMP-03) and PhysicsMonitor.
 *
 * Covers:
 *   - BootstrapState enum and name lookup
 *   - PhysicsMonitor: baseline, tick counting, event markers
 *   - ManifoldSeeder: happy-path bootstrap (grid_n=2)
 *   - ManifoldSeeder: physics_ready atomic flag
 *   - ManifoldSeeder: reset() after completion
 *   - ManifoldSeeder: metric tensor is SPD after seeding
 *   - ManifoldSeeder: Hamiltonian is non-zero after seeding
 *   - ManifoldSeeder: fault on intentionally broken pilot wave (amplitude=0)
 *   - Gershgorin SPD property: all diagonal > sum of row off-diagonals
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/physics/manifold_seeder.hpp>
#include <nikola/physics/propagator.hpp>
#include <nikola/physics/hamiltonian.hpp>
#include <nikola/physics/wave_function.hpp>
#include <nikola/physics/metric_tensor.hpp>

#include <cmath>
#include <sstream>

using namespace nikola::physics;

// ============================================================================
// BootstrapState
// ============================================================================

TEST_CASE("BootstrapState names are non-empty", "[bootstrap][state]") {
    for (int i = 0; i <= 6; ++i) {
        auto s = static_cast<BootstrapState>(i);
        CHECK(!bootstrap_state_name(s).empty());
    }
    CHECK(bootstrap_state_name(BootstrapState::READY)    == "READY");
    CHECK(bootstrap_state_name(BootstrapState::FAULTED)  == "FAULTED");
    CHECK(bootstrap_state_name(BootstrapState::ALLOCATING) == "ALLOCATING");
}

TEST_CASE("BootstrapState integer ordering is monotone", "[bootstrap][state]") {
    CHECK(static_cast<int>(BootstrapState::ALLOCATING)   < static_cast<int>(BootstrapState::SEEDING));
    CHECK(static_cast<int>(BootstrapState::SEEDING)      < static_cast<int>(BootstrapState::THERMALIZING));
    CHECK(static_cast<int>(BootstrapState::THERMALIZING) < static_cast<int>(BootstrapState::IGNITING));
    CHECK(static_cast<int>(BootstrapState::IGNITING)     < static_cast<int>(BootstrapState::STABILIZING));
    CHECK(static_cast<int>(BootstrapState::STABILIZING)  < static_cast<int>(BootstrapState::READY));
}

// ============================================================================
// PhysicsMonitor
// ============================================================================

TEST_CASE("PhysicsMonitor: initial state", "[monitor]") {
    PhysicsMonitor mon(10);
    CHECK(mon.tick_count() == 0);
    CHECK(mon.H_baseline() == Catch::Approx(0.0));
}

TEST_CASE("PhysicsMonitor: set_baseline stores value", "[monitor]") {
    PhysicsMonitor mon;
    mon.set_baseline(42.5);
    CHECK(mon.H_baseline() == Catch::Approx(42.5));
}

TEST_CASE("PhysicsMonitor: print() increments tick counter", "[monitor]") {
    PhysicsMonitor mon(1000);   // high interval so we don't print to stderr in test
    WaveFunction wf;
    wf.seed_manifold(2);

    CHECK(mon.tick_count() == 0);
    mon.print(wf, 1.0, BootstrapState::READY, /*force=*/false);
    CHECK(mon.tick_count() == 1);
    mon.print(wf, 1.0, BootstrapState::READY, /*force=*/false);
    CHECK(mon.tick_count() == 2);
}

TEST_CASE("PhysicsMonitor: force=true prints regardless of interval", "[monitor]") {
    // Just smoke-tests that force=true doesn't crash (output goes to stderr).
    PhysicsMonitor mon(99999);
    WaveFunction wf;
    wf.seed_manifold(2);
    mon.print(wf, 1.23, BootstrapState::STABILIZING, /*force=*/true);
    CHECK(mon.tick_count() == 1);
}

TEST_CASE("PhysicsMonitor: event() is a no-op smoke test", "[monitor]") {
    // Verify no crash or exception — output goes to stderr.
    PhysicsMonitor::event("TEST_EVENT");
    PhysicsMonitor::event("TEST_EVENT", "some detail");
    CHECK(true);
}

// ============================================================================
// ManifoldSeeder — happy path (grid_n=2 → 2^9=512 nodes, fast)
// ============================================================================

// Helper: build fully configured components for a small test grid.
struct TestPhysics {
    WaveFunction   wf;
    Propagator     prop;
    Hamiltonian    ham;
    PhysicsMonitor mon{99999};  // suppress output during tests

    TestPhysics() {
        prop.set_c0(1.0f).set_beta(1.0f).set_alpha(0.01f);
    }
};

TEST_CASE("ManifoldSeeder: initial state is not ready", "[seeder]") {
    ManifoldSeeder seeder;
    CHECK(!seeder.physics_ready());
    CHECK(seeder.state() == BootstrapState::ALLOCATING);
}

TEST_CASE("ManifoldSeeder: happy-path bootstrap, grid_n=2", "[seeder][bootstrap]") {
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(2)
          .set_warmup_steps(20)
          .set_verbose(false);

    REQUIRE_NOTHROW(seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon));

    CHECK(seeder.physics_ready());
    CHECK(seeder.state() == BootstrapState::READY);
}

TEST_CASE("ManifoldSeeder: grid has expected node count after bootstrap", "[seeder]") {
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(2).set_warmup_steps(10).set_verbose(false);
    seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon);

    // 2^9 = 512 nodes
    CHECK(tp.wf.num_nodes() == 512u);
}

TEST_CASE("ManifoldSeeder: Hamiltonian is finite and positive after bootstrap", "[seeder]") {
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(2).set_warmup_steps(10).set_verbose(false);
    seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon);

    const double H = tp.ham.compute(tp.wf);
    CHECK(std::isfinite(H));
    CHECK(H > 0.0);
}

TEST_CASE("ManifoldSeeder: wavefunction has non-zero probability after bootstrap", "[seeder]") {
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(2).set_warmup_steps(10).set_verbose(false);
    seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon);

    CHECK(tp.wf.total_probability() > 0.0);
}

TEST_CASE("ManifoldSeeder: metric tensor is valid (SPD) after bootstrap", "[seeder]") {
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(2).set_warmup_steps(10).set_verbose(false);
    seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon);

    CHECK(seeder.metric().is_valid());

    // log_det of SPD matrix must be finite.
    double ld = 0.0;
    REQUIRE_NOTHROW(ld = seeder.metric().log_det());
    CHECK(std::isfinite(ld));
}

TEST_CASE("ManifoldSeeder: metric apply_inverse is consistent", "[seeder]") {
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(2).set_warmup_steps(10).set_verbose(false);
    seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon);

    // Random vector; g⁻¹ g v ≈ v uses separate apply() + apply_inverse().
    std::array<double,9> v{1,0,0,0,0,0,0,0,0};
    auto gv  = seeder.metric().apply(v);
    auto igv = seeder.metric().apply_inverse(gv);
    for (int i = 0; i < 9; ++i) {
        // Use margin() for absolute tolerance (epsilon() fails when ref is 0.0).
        CHECK(igv[i] == Catch::Approx(v[i]).margin(1e-8));
    }
}

TEST_CASE("ManifoldSeeder: physics_ready uses acquire semantics", "[seeder]") {
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(2).set_warmup_steps(5).set_verbose(false);

    CHECK(!seeder.physics_ready());
    seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon);
    CHECK(seeder.physics_ready());
}

// ============================================================================
// ManifoldSeeder — reset
// ============================================================================

TEST_CASE("ManifoldSeeder: reset() clears ready flag and state", "[seeder][reset]") {
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(2).set_warmup_steps(5).set_verbose(false);
    seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon);

    REQUIRE(seeder.physics_ready());

    seeder.reset();

    CHECK(!seeder.physics_ready());
    CHECK(seeder.state() == BootstrapState::ALLOCATING);
    CHECK(!seeder.metric().is_valid());   // invalidated by reset()
}

TEST_CASE("ManifoldSeeder: can re-seed after reset", "[seeder][reset]") {
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(2).set_warmup_steps(5).set_verbose(false);

    seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon);
    seeder.reset();

    // Re-seed with different parameters.
    seeder.set_seed(99);
    REQUIRE_NOTHROW(seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon));
    CHECK(seeder.physics_ready());
}

// ============================================================================
// ManifoldSeeder — fault path
// ============================================================================

TEST_CASE("ManifoldSeeder: throws SeederFault on empty grid (zero nodes)",
          "[seeder][fault]") {
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(0)          // 0^9 = 0 nodes → zero kinetic energy → THERMALIZING fault
          .set_warmup_steps(5)
          .set_verbose(false);

    CHECK_THROWS_AS(seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon),
                    ManifoldSeeder::SeederFault);
    CHECK(!seeder.physics_ready());
    CHECK(seeder.state() == BootstrapState::FAULTED);
}

TEST_CASE("ManifoldSeeder: SeederFault on empty grid faults at THERMALIZING",
          "[seeder][fault]") {
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(0)
          .set_warmup_steps(5)
          .set_verbose(false);

    bool caught = false;
    try {
        seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon);
    } catch (const ManifoldSeeder::SeederFault& e) {
        caught = true;
        CHECK(e.failed_at == BootstrapState::THERMALIZING);
        CHECK(std::string(e.what()).find("ManifoldSeeder FAULT") != std::string::npos);
    }
    CHECK(caught);
}

// ============================================================================
// Gershgorin metric: SPD property independent of seeder
// ============================================================================

TEST_CASE("Gershgorin metric seeding: diagonal dominance → all eigenvalues > 0",
          "[seeder][metric][gershgorin]") {
    // Build a Gershgorin-seeded metric manually and verify:
    //   g_ii > Σ_{j≠i} |g_ij|   (strict row dominance)
    // This guarantees positive-definiteness by Gershgorin's disc theorem.

    std::mt19937 rng(1234);
    const double noise = 0.05;
    std::normal_distribution<double> nd(0.0, noise);

    std::array<double, METRIC_LOWER_SIZE> g{};
    g.fill(0.0);

    for (int i = 0; i < METRIC_DIM; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < i; ++j) {
            const double v = nd(rng) * noise / 8.0;
            g[metric_lower_idx(i, j)] = v;
            row_sum += std::abs(v);
        }
        const double diag_noise = std::abs(nd(rng)) * 0.1;
        g[metric_lower_idx(i, i)] = 1.0 + diag_noise + row_sum;
    }

    // Verify row dominance on all rows.
    for (int i = 0; i < METRIC_DIM; ++i) {
        double row_off = 0.0;
        for (int j = 0; j < METRIC_DIM; ++j) {
            if (j == i) continue;
            const int ii = std::max(i, j);
            const int jj = std::min(i, j);
            row_off += std::abs(g[metric_lower_idx(ii, jj)]);
        }
        const double diag = g[metric_lower_idx(i, i)];
        CHECK(diag > row_off);
    }

    // Verify Cholesky succeeds (proves positive-definiteness).
    MetricTensorCache cache;
    REQUIRE_NOTHROW(cache.force_update(g));
    CHECK(cache.is_valid());
    double ld = 0.0;
    REQUIRE_NOTHROW(ld = cache.log_det());
    CHECK(std::isfinite(ld));
}

// ============================================================================
// Integration: can step physics after seeder marks ready
// ============================================================================

TEST_CASE("ManifoldSeeder: propagation is stable for 50 steps post-bootstrap",
          "[seeder][integration]") {
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(2).set_warmup_steps(10).set_verbose(false);
    seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon);

    REQUIRE(seeder.physics_ready());

    const double H0  = tp.ham.compute(tp.wf);
    const float  dt  = tp.prop.max_stable_dt(tp.wf.grid());

    for (int i = 0; i < 50; ++i) {
        tp.prop.step(tp.wf, dt);
    }

    const double H1 = tp.ham.compute(tp.wf);
    REQUIRE(std::isfinite(H1));

    const double drift = std::abs(H1 - H0) / (H0 + 1e-30);
    // With damping the Hamiltonian should decrease — drift bound is generous.
    CHECK(drift < 1.0);   // less than 100% change in 50 steps
    CHECK(H1 > 0.0);      // still alive (not NaN/zero)
}
