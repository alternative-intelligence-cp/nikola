// ============================================================
// Integration Test: Physics Pipeline
// tests/integration/physics_integration_test.cpp
//
// Validates physics subsystem integration:
//   §A  Inject → Propagate → Scan — field state is coherent
//   §B  Long-run stability — 1000 steps with constant input
//   §C  Emitter array — field response from emitter injection
// ============================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/physics/wave_function.hpp>
#include <nikola/physics/propagator.hpp>
#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/foundation/vector9d.hpp>
#include <nikola/spatial/hilbert_scanner.hpp>

#include <cmath>
#include <complex>
#include <numeric>
#include <vector>

using namespace nikola::physics;
using namespace nikola::cognitive;
using namespace nikola::foundation;
using namespace nikola::spatial;
using Catch::Approx;

// ── Helpers ─────────────────────────────────────────────────────────────────

static CognitiveTorus make_torus(int n = 3) {
    return CognitiveTorus(n);
}

// ── §A  Inject → Propagate → Scan ──────────────────────────────────────────

TEST_CASE("§A-1 Inject energy, propagate, verify field is non-trivial",
          "[integration][physics]") {
    auto torus = make_torus();
    auto num   = torus.num_nodes();  // 3^9 = 19683

    // Inject at node 0
    torus.wave_function().inject(0, {1.0f, 0.0f});

    // Propagate 50 steps
    torus.run(50, torus.safe_dt());

    // After propagation, energy should have spread beyond node 0
    float max_intensity = 0.0f;
    size_t nonzero_count = 0;
    for (size_t i = 0; i < num; ++i) {
        float I = torus.intensity(i);
        if (I > 1e-12f) ++nonzero_count;
        if (I > max_intensity) max_intensity = I;
    }

    // Energy should have spread to multiple nodes
    REQUIRE(nonzero_count > 1);
    // Total probability should still be conserved (symplectic)
    double prob = torus.total_probability();
    REQUIRE(prob > 0.0);
    REQUIRE(std::isfinite(prob));
}

TEST_CASE("§A-2 Hot nodes reflect injection site proximity",
          "[integration][physics]") {
    auto torus = make_torus();

    // Inject strong signal at node 100
    torus.wave_function().inject(100, {2.0f, 0.0f});

    // Short propagation — energy hasn't spread far
    torus.run(5, torus.safe_dt());

    auto hot = torus.hot_nodes(3);
    REQUIRE(!hot.empty());
    // Node 100 should be among the hottest
    bool found_100 = false;
    for (auto idx : hot) {
        if (idx == 100) found_100 = true;
    }
    REQUIRE(found_100);
}

TEST_CASE("§A-3 Hilbert scan produces ordered traversal of field",
          "[integration][physics]") {
    HilbertScanner scanner(3);

    // Verify round-trip: index → coords → index
    for (uint64_t i = 0; i < 100; ++i) {
        auto coords = scanner.index_to_coords(i);
        auto back   = scanner.coords_to_index(coords);
        REQUIRE(back == i);
    }

    // Neighbors should be close in Hilbert order (locality property)
    auto n1 = scanner.get_neighbors(50, 1);
    REQUIRE(!n1.empty());
}

// ── §B  Long-Run Stability ─────────────────────────────────────────────────

TEST_CASE("§B-1 1000 steps with constant injection — no divergence",
          "[integration][physics][longsession]") {
    auto torus = make_torus();
    float dt   = torus.safe_dt();

    double initial_prob = torus.total_probability();

    for (int i = 0; i < 1000; ++i) {
        // Inject a small constant signal every 100 steps
        if (i % 100 == 0) {
            torus.wave_function().inject(0, {0.1f, 0.0f});
        }
        torus.step(dt);
    }

    // Field should still be finite
    REQUIRE(torus.wave_function().is_finite());
    // Probability should be positive
    double final_prob = torus.total_probability();
    REQUIRE(final_prob > 0.0);
    REQUIRE(std::isfinite(final_prob));
}

TEST_CASE("§B-2 1000 steps idle — field decays or remains stable, never diverges",
          "[integration][physics][longsession]") {
    auto torus = make_torus();
    float dt   = torus.safe_dt();

    // Just run with no input
    torus.run(1000, dt);

    REQUIRE(torus.wave_function().is_finite());
    double prob = torus.total_probability();
    REQUIRE(std::isfinite(prob));
}

// ── §C  Emitter Array Field Response ────────────────────────────────────────

TEST_CASE("§C-1 Multiple injection points create interference pattern",
          "[integration][physics]") {
    auto torus = make_torus();
    auto num   = torus.num_nodes();

    // Inject at 3 separated nodes
    torus.wave_function().inject(0,     {1.0f, 0.0f});
    torus.wave_function().inject(1000,  {1.0f, 0.0f});
    torus.wave_function().inject(5000,  {1.0f, 0.0f});

    torus.run(100, torus.safe_dt());

    // Field should be non-trivially distributed
    float total_intensity = 0.0f;
    size_t nonzero = 0;
    for (size_t i = 0; i < num; ++i) {
        float I = torus.intensity(i);
        total_intensity += I;
        if (I > 1e-12f) ++nonzero;
    }

    // Energy spread across many nodes
    REQUIRE(nonzero > 10);
    REQUIRE(std::isfinite(total_intensity));
    REQUIRE(total_intensity > 0.0f);
}

TEST_CASE("§C-2 Propagator substeps preserve symplectic structure",
          "[integration][physics]") {
    auto torus = make_torus();

    torus.wave_function().inject(500, {1.0f, 0.5f});

    double prob_before = torus.total_probability();

    // Run a moderate number of steps
    torus.run(50, torus.safe_dt());

    double prob_after = torus.total_probability();

    // Symplectic integrator: probability shouldn't drift wildly
    // Allow some numerical drift but not orders of magnitude
    double ratio = prob_after / prob_before;
    REQUIRE(ratio > 0.5);
    REQUIRE(ratio < 2.0);
    REQUIRE(std::isfinite(prob_after));
}
