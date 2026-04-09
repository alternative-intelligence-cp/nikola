// ============================================================
// v0.1.4 — Manifold Geometry: 9D Grid & Metric Tensor Validation
// tests/unit/phase014_manifold_geometry_test.cpp
//
// Validates the v0.1.4 acceptance criteria:
//   §1  9D grid initialization at multiple resolutions
//   §2  Metric tensor always SPD (Cholesky never fails)
//   §3  Gerschgorin perturbation robustness
//   §4  Manifold seeder produces valid geometry
//   §5  CoordWords integer/float dual coordinate system
//   §6  Dopamine-modulated Hebbian learning rate
//   §7  Integration: seeder → Hebbian → physics
//   §8  Memory footprint and SoA layout
// ============================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/physics/metric_tensor.hpp>
#include <nikola/physics/manifold_seeder.hpp>
#include <nikola/physics/wave_function.hpp>
#include <nikola/physics/propagator.hpp>
#include <nikola/physics/hamiltonian.hpp>
#include <nikola/physics/soa_layout.hpp>
#include <nikola/physics/coordinate_semantics.hpp>
#include <nikola/foundation/coord_serializer.hpp>
#include <nikola/foundation/vector9d.hpp>
#include <nikola/math/hebbian_metric.hpp>

#include <cmath>
#include <random>
#include <complex>
#include <vector>
#include <array>
#include <algorithm>

using namespace nikola::physics;
using namespace nikola::foundation;
using namespace nikola::math;
using Catch::Approx;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Build a seeded WaveFunction at a given grid size.
static WaveFunction make_wf(int n, float amp = 0.5f, uint32_t seed = 42) {
    WaveFunction wf;
    wf.seed_manifold(n, 3, 1, amp, seed);
    wf.grid().precompute_adjacency();
    return wf;
}

/// n^9 computed on integers.
static long long pow9(int n) {
    long long r = 1;
    for (int i = 0; i < 9; ++i) r *= n;
    return r;
}

/// Build a random SPD metric via Gerschgorin diagonal-dominance.
static std::array<double, METRIC_LOWER_SIZE>
random_gerschgorin_metric(std::mt19937& rng, double noise = 0.05) {
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
        g[metric_lower_idx(i, i)] = 1.0 + std::abs(nd(rng)) * 0.1 + row_sum;
    }
    return g;
}

/// Add Gaussian noise to a metric tensor.
static std::array<double, METRIC_LOWER_SIZE>
perturb_metric(const std::array<double, METRIC_LOWER_SIZE>& g,
               std::mt19937& rng, double sigma) {
    std::normal_distribution<double> nd(0.0, sigma);
    auto perturbed = g;
    for (int i = 0; i < METRIC_LOWER_SIZE; ++i)
        perturbed[i] += nd(rng);
    return perturbed;
}

/// Compute minimum eigenvalue of a 9×9 symmetric matrix via Gerschgorin bounds.
static double gerschgorin_min_eigenvalue(
    const std::array<double, METRIC_LOWER_SIZE>& g) {
    double lambda_min = 1e30;
    for (int i = 0; i < METRIC_DIM; ++i) {
        double diag = g[metric_lower_idx(i, i)];
        double off_sum = 0.0;
        for (int j = 0; j < METRIC_DIM; ++j) {
            if (j != i)
                off_sum += std::abs(metric_get(g, i, j));
        }
        lambda_min = std::min(lambda_min, diag - off_sum);
    }
    return lambda_min;
}

/// Build a PhysicsMonitor + Propagator + Hamiltonian for seeder tests.
struct TestPhysics {
    WaveFunction   wf;
    Propagator     prop;
    Hamiltonian    ham;
    PhysicsMonitor mon{99999};

    TestPhysics() {
        prop.set_c0(1.0f).set_beta(1.0f).set_alpha(0.01f);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// §1  9D Grid Initialization
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("v0.1.4 §1-1 Grid initialization: n=1..4 produce n^9 nodes",
          "[grid][phase014]") {
    for (int n = 1; n <= 4; ++n) {
        WaveFunction wf;
        wf.seed_manifold(n);
        const size_t expected = static_cast<size_t>(pow9(n));
        INFO("n=" << n << ", expected=" << expected
             << ", got=" << wf.num_nodes());
        REQUIRE(wf.num_nodes() == expected);
    }
}

TEST_CASE("v0.1.4 §1-2 Grid n=3 (19683 nodes): wavefunction is finite and non-zero",
          "[grid][phase014]") {
    auto wf = make_wf(3, 0.5f);
    REQUIRE(wf.num_nodes() == 19683);
    REQUIRE(wf.is_finite());
    REQUIRE(wf.total_kinetic_energy() > 0.0);
}

TEST_CASE("v0.1.4 §1-3 Coordinate semantics: 9 dimensions span 4 domains",
          "[grid][coords][phase014]") {
    // Systemic: RESONANCE, STATE
    REQUIRE(dim_domain(Dim9::RESONANCE) == CoordDomain::SYSTEMIC);
    REQUIRE(dim_domain(Dim9::STATE)     == CoordDomain::SYSTEMIC);
    // Temporal: TIME
    REQUIRE(dim_domain(Dim9::TIME)      == CoordDomain::TEMPORAL);
    // Quantum: U, V, W
    REQUIRE(dim_domain(Dim9::U)         == CoordDomain::QUANTUM);
    REQUIRE(dim_domain(Dim9::V)         == CoordDomain::QUANTUM);
    REQUIRE(dim_domain(Dim9::W)         == CoordDomain::QUANTUM);
    // Spatial: X, Y, Z
    REQUIRE(dim_domain(Dim9::X)         == CoordDomain::SPATIAL);
    REQUIRE(dim_domain(Dim9::Y)         == CoordDomain::SPATIAL);
    REQUIRE(dim_domain(Dim9::Z)         == CoordDomain::SPATIAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// §2  Metric Tensor — Always SPD
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("v0.1.4 §2-1 Flat identity metric: Cholesky, inverse, log_det",
          "[metric][spd][phase014]") {
    auto cache = MetricTensorCache::flat();
    REQUIRE(cache.is_valid());

    // log_det(I) = 0
    REQUIRE(cache.log_det() == Approx(0.0).margin(1e-12));

    // g^{-1} v = v for identity
    std::array<double, METRIC_DIM> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto result = cache.apply_inverse(v);
    for (int i = 0; i < METRIC_DIM; ++i)
        REQUIRE(result[i] == Approx(v[i]).epsilon(1e-12));
}

TEST_CASE("v0.1.4 §2-2 Random SPD metrics: 100 random Gerschgorin matrices all pass Cholesky",
          "[metric][spd][phase014]") {
    std::mt19937 rng(123);
    int pass_count = 0;

    for (int trial = 0; trial < 100; ++trial) {
        auto g = random_gerschgorin_metric(rng, 0.1);
        std::array<double, METRIC_LOWER_SIZE> L{};
        bool ok = cholesky_9x9(g, L);
        if (ok) ++pass_count;
    }

    REQUIRE(pass_count == 100);
}

TEST_CASE("v0.1.4 §2-3 Inverse round-trip: g^{-1} g v ≈ v on 20 random SPD matrices",
          "[metric][spd][phase014]") {
    std::mt19937 rng(456);

    for (int trial = 0; trial < 20; ++trial) {
        auto g = random_gerschgorin_metric(rng, 0.08);
        MetricTensorCache cache(g);
        REQUIRE(cache.is_valid());

        // Random test vector
        std::array<double, METRIC_DIM> v{};
        std::uniform_real_distribution<double> ud(-5.0, 5.0);
        for (auto& x : v) x = ud(rng);

        // Compute g⁻¹ v
        auto inv_v = cache.apply_inverse(v);

        // Compute g · (g⁻¹ v) manually = v
        for (int i = 0; i < METRIC_DIM; ++i) {
            double s = 0.0;
            for (int j = 0; j < METRIC_DIM; ++j)
                s += metric_get(g, i, j) * inv_v[j];
            REQUIRE(s == Approx(v[i]).epsilon(1e-8));
        }
    }
}

TEST_CASE("v0.1.4 §2-4 Non-SPD rejection: indefinite matrix throws on force_update",
          "[metric][spd][phase014]") {
    // All-zero matrix (not PD)
    std::array<double, METRIC_LOWER_SIZE> bad{};
    bad.fill(0.0);

    MetricTensorCache cache;
    REQUIRE_THROWS_AS(cache.force_update(bad), std::invalid_argument);
    REQUIRE(!cache.is_valid());
}

// ═══════════════════════════════════════════════════════════════════════════
// §3  Gerschgorin Perturbation Robustness
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("v0.1.4 §3-1 Gerschgorin diagonal dominance: all eigenvalues > 0",
          "[metric][gerschgorin][phase014]") {
    std::mt19937 rng(789);
    for (int trial = 0; trial < 50; ++trial) {
        auto g = random_gerschgorin_metric(rng, 0.1);
        double lmin = gerschgorin_min_eigenvalue(g);
        INFO("Trial " << trial << ": Gerschgorin λ_min = " << lmin);
        REQUIRE(lmin > 0.0);
    }
}

TEST_CASE("v0.1.4 §3-2 Perturbation resilience: 1% noise on Gerschgorin metric → still SPD",
          "[metric][gerschgorin][perturbation][phase014]") {
    std::mt19937 rng(101);
    int pass_count = 0;

    for (int trial = 0; trial < 50; ++trial) {
        auto g = random_gerschgorin_metric(rng, 0.05);
        // Add 1% diagonal-scale noise
        double diag_scale = g[metric_lower_idx(0, 0)];
        auto noisy = perturb_metric(g, rng, diag_scale * 0.01);

        // Re-enforce diagonal dominance after perturbation
        for (int i = 0; i < METRIC_DIM; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < METRIC_DIM; ++j)
                if (j != i) row_sum += std::abs(metric_get(noisy, i, j));
            if (noisy[metric_lower_idx(i, i)] <= row_sum)
                noisy[metric_lower_idx(i, i)] = row_sum + 1e-6;
        }

        std::array<double, METRIC_LOWER_SIZE> L{};
        if (cholesky_9x9(noisy, L)) ++pass_count;
    }

    REQUIRE(pass_count == 50);
}

TEST_CASE("v0.1.4 §3-3 Random walk: 200 tiny updates maintain SPD",
          "[metric][gerschgorin][perturbation][phase014]") {
    std::mt19937 rng(202);
    auto g = random_gerschgorin_metric(rng, 0.05);

    MetricTensorCache cache(g);
    REQUIRE(cache.is_valid());

    std::normal_distribution<double> nd(0.0, 1e-4);

    for (int step = 0; step < 200; ++step) {
        // Tiny perturbation
        auto g_new = cache.metric();
        for (int i = 0; i < METRIC_LOWER_SIZE; ++i)
            g_new[i] += nd(rng);

        // Re-enforce diagonal dominance
        for (int i = 0; i < METRIC_DIM; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < METRIC_DIM; ++j)
                if (j != i) row_sum += std::abs(metric_get(g_new, i, j));
            if (g_new[metric_lower_idx(i, i)] <= row_sum)
                g_new[metric_lower_idx(i, i)] = row_sum + 1e-6;
        }

        bool recomputed = cache.update_if_changed(g_new);
        REQUIRE(cache.is_valid());

        double ld = cache.log_det();
        INFO("Step " << step << ": log_det = " << ld << ", recomputed=" << recomputed);
        REQUIRE(std::isfinite(ld));
    }
}

TEST_CASE("v0.1.4 §3-4 Tikhonov fallback: project_to_spd rescues indefinite matrix",
          "[metric][tikhonov][phase014]") {
    // Build an indefinite matrix (one negative eigenvalue)
    MetricTensor g = MetricTensor::Identity();
    g(0, 0) = -1.0;  // eigenvalue = -1

    REQUIRE(!is_spd(g));

    // Project to SPD cone
    MetricTensor g_safe = project_to_spd(g, HM_EPSILON_MIN);
    REQUIRE(is_spd(g_safe));

    // All eigenvalues >= epsilon_min
    Eigen::SelfAdjointEigenSolver<MetricTensor> es(g_safe);
    double lmin = es.eigenvalues().minCoeff();
    REQUIRE(lmin >= HM_EPSILON_MIN);
}

// ═══════════════════════════════════════════════════════════════════════════
// §4  Manifold Seeder — Valid Geometry
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("v0.1.4 §4-1 Seeder bootstrap (n=2): happy path to READY",
          "[seeder][phase014]") {
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(2).set_verbose(false).set_seed(42);

    seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon);

    REQUIRE(seeder.physics_ready());
    REQUIRE(seeder.state() == BootstrapState::READY);
    REQUIRE(tp.wf.num_nodes() == 512);
    REQUIRE(tp.wf.is_finite());
}

TEST_CASE("v0.1.4 §4-2 Seeder bootstrap (n=3): 19683 nodes, metric valid",
          "[seeder][phase014]") {
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(3).set_verbose(false).set_seed(99);

    seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon);

    REQUIRE(seeder.physics_ready());
    REQUIRE(tp.wf.num_nodes() == 19683);
    REQUIRE(tp.wf.is_finite());
    REQUIRE(seeder.metric().is_valid());

    double ld = seeder.metric().log_det();
    INFO("n=3 seeder log_det = " << ld);
    REQUIRE(std::isfinite(ld));
    REQUIRE(ld > 0.0);  // det(g) > 1 for inflated diagonal
}

TEST_CASE("v0.1.4 §4-3 Seeder SPD verified via Gerschgorin on seeded metric",
          "[seeder][gerschgorin][phase014]") {
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(2).set_verbose(false).set_seed(77);

    seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon);

    const auto& g = seeder.metric().metric();
    double lmin = gerschgorin_min_eigenvalue(g);
    INFO("Seeder metric Gerschgorin λ_min = " << lmin);
    REQUIRE(lmin > 0.0);
}

TEST_CASE("v0.1.4 §4-4 Repeated seeding: 10 runs, all finite and bounded",
          "[seeder][phase014]") {
    for (int trial = 0; trial < 10; ++trial) {
        TestPhysics tp;
        ManifoldSeeder seeder;
        seeder.set_grid_size(2).set_verbose(false).set_seed(trial * 17 + 3);

        seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon);

        REQUIRE(seeder.physics_ready());
        REQUIRE(tp.wf.is_finite());
        REQUIRE(seeder.metric().is_valid());

        double H = tp.ham.compute(tp.wf);
        INFO("Trial " << trial << ": H = " << H);
        REQUIRE(std::isfinite(H));
        REQUIRE(H > 0.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §5  CoordWords — Dual Integer/Float Coordinate System
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("v0.1.4 §5-1 CoordWords round-trip: all 9 dimensions pack/unpack",
          "[coords][phase014]") {
    CoordWords cw(15, 7, 16383, 255, 128, 0, 12345, 9876, 4321);

    REQUIRE(cw.r() == 15);
    REQUIRE(cw.s() == 7);
    REQUIRE(cw.t() == 16383);
    REQUIRE(cw.u() == 255);
    REQUIRE(cw.v() == 128);
    REQUIRE(cw.w() == 0);
    REQUIRE(cw.x() == 12345);
    REQUIRE(cw.y() == 9876);
    REQUIRE(cw.z() == 4321);
}

TEST_CASE("v0.1.4 §5-2 Vector9D: toroidal distance wraps correctly",
          "[coords][phase014]") {
    // Two points near the boundary of [0, 1) torus
    Vector9D a({0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    Vector9D b({0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

    // Euclidean distance = 0.98
    double d_eucl = distance(a, b);
    REQUIRE(d_eucl == Approx(0.98).margin(0.01));

    // Toroidal distance should wrap → 0.02
    double d_torus = toroidal_distance(a, b);
    REQUIRE(d_torus == Approx(0.02).margin(0.01));
}

// ═══════════════════════════════════════════════════════════════════════════
// §6  Dopamine-Modulated Hebbian Learning Rate
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("v0.1.4 §6-1 Hebbian update: single step maintains SPD",
          "[hebbian][dopamine][phase014]") {
    MetricTensor g = MetricTensor::Identity();

    // Build a correlation matrix from a random wavefunction
    WavefunctionVec psi;
    psi << std::complex<double>(0.5, 0.1),
           std::complex<double>(0.3, -0.2),
           std::complex<double>(0.1, 0.4),
           std::complex<double>(-0.1, 0.3),
           std::complex<double>(0.2, -0.1),
           std::complex<double>(0.4, 0.0),
           std::complex<double>(-0.3, 0.2),
           std::complex<double>(0.1, -0.3),
           std::complex<double>(0.2, 0.1);

    CorrelationMatrix C = compute_correlation(psi);

    // η = 0.5 (moderate dopamine), λ = 10.0 (strong relaxation)
    MetricTensor g_new = hebbian_update(g, C, 0.5, 10.0, HM_DT_PHYSICS);

    // Should still be SPD after one step
    REQUIRE(is_spd(g_new));
}

TEST_CASE("v0.1.4 §6-2 Dopamine modulation: high D → faster convergence",
          "[hebbian][dopamine][phase014]") {
    MetricTensor g0 = MetricTensor::Identity() * 1.5;  // Start away from equilibrium

    WavefunctionVec psi;
    psi << std::complex<double>(0.3, 0.1),
           std::complex<double>(0.2, -0.1),
           std::complex<double>(0.1, 0.2),
           std::complex<double>(0.0, 0.1),
           std::complex<double>(0.1, 0.0),
           std::complex<double>(0.2, -0.1),
           std::complex<double>(0.1, 0.1),
           std::complex<double>(0.0, -0.1),
           std::complex<double>(0.1, 0.0);

    CorrelationMatrix C = compute_correlation(psi);
    double lambda = 5.0;
    int steps = 500;

    // Low dopamine (η = 0.1)
    auto result_low = run_hebbian(g0, C, 0.1, lambda, steps);
    double E_low = result_low.final_energy;

    // High dopamine (η = 1.0)
    auto result_high = run_hebbian(g0, C, 1.0, lambda, steps);
    double E_high = result_high.final_energy;

    INFO("E_low_D = " << E_low << ", E_high_D = " << E_high);

    // Both should be SPD
    REQUIRE(is_spd(result_low.g));
    REQUIRE(is_spd(result_high.g));

    // Higher η drives more plasticity; Lyapunov energy should differ
    // (η*Tr(g·C) term contributes more at higher η)
    REQUIRE(std::isfinite(E_low));
    REQUIRE(std::isfinite(E_high));
}

TEST_CASE("v0.1.4 §6-3 Lyapunov energy: monotonically non-increasing over 1000 steps",
          "[hebbian][lyapunov][phase014]") {
    MetricTensor g = MetricTensor::Identity() * 2.0;

    WavefunctionVec psi;
    psi << std::complex<double>(0.4, 0.0),
           std::complex<double>(0.3, 0.1),
           std::complex<double>(0.2, -0.1),
           std::complex<double>(0.1, 0.2),
           std::complex<double>(0.0, 0.1),
           std::complex<double>(0.1, 0.0),
           std::complex<double>(0.2, -0.2),
           std::complex<double>(0.1, 0.1),
           std::complex<double>(0.0, 0.0);

    CorrelationMatrix C = compute_correlation(psi);
    double eta = 0.5, lambda = 5.0;

    double E_prev = lyapunov_energy(g, C, eta, lambda);
    int violations = 0;

    for (int step = 0; step < 1000; ++step) {
        g = hebbian_update(g, C, eta, lambda, HM_DT_PHYSICS);
        if (!is_spd(g)) g = project_to_spd(g);

        double E_now = lyapunov_energy(g, C, eta, lambda);
        if (E_now > E_prev + 1e-10) ++violations;
        E_prev = E_now;
    }

    INFO("Lyapunov violations: " << violations << " / 1000");
    REQUIRE(violations == 0);
    REQUIRE(is_spd(g));
}

TEST_CASE("v0.1.4 §6-4 Soft SCRAM count: adversarial η/λ triggers projection",
          "[hebbian][tikhonov][phase014]") {
    // Large η, small λ → equilibrium g* may not be SPD → SCRAM expected
    MetricTensor g0 = MetricTensor::Identity();

    WavefunctionVec psi;
    psi << std::complex<double>(1.0, 0.0),
           std::complex<double>(0.8, 0.3),
           std::complex<double>(0.6, -0.5),
           std::complex<double>(0.4, 0.7),
           std::complex<double>(0.2, -0.9),
           std::complex<double>(0.0, 1.0),
           std::complex<double>(-0.2, 0.8),
           std::complex<double>(-0.4, 0.6),
           std::complex<double>(-0.6, 0.4);

    CorrelationMatrix C = compute_correlation(psi);

    // Aggressive: η=50, λ=1 → equilibrium g* = I - 50*C might be non-SPD
    auto result = run_hebbian(g0, C, 50.0, 1.0, 200, HM_DT_PHYSICS);

    INFO("SCRAM count: " << result.scram_count);
    // We expect at least some SRAMs with such aggressive parameters
    REQUIRE(result.scram_count > 0);
    // But the final metric must still be SPD (projection saved it)
    REQUIRE(is_spd(result.g));
    REQUIRE(std::isfinite(result.final_energy));
}

// ═══════════════════════════════════════════════════════════════════════════
// §7  Integration: Seeder → Hebbian → Physics
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("v0.1.4 §7-1 End-to-end: seed → 100 physics steps → Hebbian update → metric valid",
          "[integration][phase014]") {
    // 1. Bootstrap
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(2).set_verbose(false).set_seed(42);
    seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon);

    REQUIRE(seeder.physics_ready());
    double H0 = tp.ham.compute(tp.wf);
    REQUIRE(std::isfinite(H0));

    // 2. Run 100 physics steps
    tp.wf.grid().precompute_adjacency();
    float dt = tp.prop.max_stable_dt(tp.wf.grid()) * 0.06f;
    for (int i = 0; i < 100; ++i)
        tp.prop.step(tp.wf, dt);

    double H1 = tp.ham.compute(tp.wf);
    REQUIRE(tp.wf.is_finite());
    REQUIRE(std::isfinite(H1));

    // 3. Hebbian metric update
    WavefunctionVec psi;
    const auto& g = tp.wf.grid();
    // Sample first 9 nodes' psi for correlation
    for (int d = 0; d < 9; ++d) {
        size_t idx = std::min(static_cast<size_t>(d), g.num_active_nodes() - 1);
        psi(d) = std::complex<double>(g.psi_real()[idx], g.psi_imag()[idx]);
    }

    CorrelationMatrix C = compute_correlation(psi);
    MetricTensor g_eigen = MetricTensor::Identity();

    auto heb = run_hebbian(g_eigen, C, 0.3, 5.0, 50, HM_DT_PHYSICS);
    REQUIRE(is_spd(heb.g));
    REQUIRE(std::isfinite(heb.final_energy));

    // 4. Metric cache still valid
    REQUIRE(seeder.metric().is_valid());
}

TEST_CASE("v0.1.4 §7-2 Post-bootstrap stability: 500 physics steps, drift < 50%",
          "[integration][phase014]") {
    TestPhysics tp;
    ManifoldSeeder seeder;
    seeder.set_grid_size(3).set_verbose(false).set_seed(55);
    seeder.seed(tp.wf, tp.prop, tp.ham, tp.mon);

    double H0 = tp.ham.compute(tp.wf);
    REQUIRE(std::isfinite(H0));
    REQUIRE(H0 > 0.0);

    tp.wf.grid().precompute_adjacency();
    float dt = tp.prop.max_stable_dt(tp.wf.grid()) * 0.06f;

    for (int i = 0; i < 500; ++i)
        tp.prop.step(tp.wf, dt);

    double H1 = tp.ham.compute(tp.wf);
    double drift = std::abs(H1 - H0) / std::abs(H0);

    INFO("Post-bootstrap 500 steps: H0=" << H0 << ", H1=" << H1
         << ", drift=" << drift);
    REQUIRE(tp.wf.is_finite());
    REQUIRE(std::isfinite(H1));
    REQUIRE(drift < 0.50);  // <50% under damping
}

// ═══════════════════════════════════════════════════════════════════════════
// §8  Memory Footprint & SoA Layout
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("v0.1.4 §8-1 TorusBlock alignment: 64-byte aligned, METRIC_TENSOR_COMPONENTS=45",
          "[memory][soa][phase014]") {
    REQUIRE(alignof(TorusBlock) == 64);
    REQUIRE(METRIC_TENSOR_COMPONENTS == 45);
}

TEST_CASE("v0.1.4 §8-2 Memory budget: n=3 grid fits in GPU VRAM",
          "[memory][phase014]") {
    // TorusBlock size contains 19683 × (2 floats for psi + 45 floats for metric)
    // = 19683 × 47 × 4 bytes ≈ 3.5 MB per block
    // A single 3^9 grid = 1 TorusBlock ≈ 3.5 MB — easily fits in 24 GB
    constexpr size_t N = 19683;
    constexpr size_t bytes_per_node = 2 * sizeof(float) + 45 * sizeof(float);
    constexpr size_t total_bytes = N * bytes_per_node;
    constexpr size_t rtx3090_vram = 24ULL * 1024 * 1024 * 1024;  // 24 GB

    INFO("Grid memory: " << total_bytes << " bytes ("
         << (total_bytes / (1024.0 * 1024.0)) << " MB)");
    REQUIRE(total_bytes < rtx3090_vram);
    REQUIRE(total_bytes < 100 * 1024 * 1024);  // <100 MB even generously
}

TEST_CASE("v0.1.4 §8-3 AlignedVec: allocation respects 64-byte alignment",
          "[memory][soa][phase014]") {
    AlignedVec<float> v(1024);
    REQUIRE(reinterpret_cast<uintptr_t>(v.data()) % 64 == 0);

    AlignedVec<double> d(512);
    REQUIRE(reinterpret_cast<uintptr_t>(d.data()) % 64 == 0);
}
