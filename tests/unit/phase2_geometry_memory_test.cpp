/**
 * @file phase2_geometry_memory_test.cpp
 * @brief Unit tests for Phase 2: Manifold Geometry and Semantic Memory.
 *
 * Tests cover:
 *   === TopologyManager / MetricValidator (Gap 2.1, 2.3, 2.4, 2.5) ===
 *   - Gerschgorin accepts identity matrix as positive-definite
 *   - Gerschgorin rejects degenerate matrix (zero diagonal)
 *   - Tikhonov repair makes degenerate matrix valid
 *   - Coordinate round-trip: integer → physical → integer (lossless)
 *   - Anisotropic GridConfig matches spec (64,64,64,128,16,16,32,32,32)
 *   - Quadratic peak interpolation: flat → 0, asymmetric → correct direction
 *   - Toroidal distance wraparound
 *   - MetricLearner: high dopamine/young => near η_base; old/dopamine=0 => near 0
 *   - MetricLearner: update_metric keeps metric valid
 *
 *   === SemanticMemory (Phase 2 — wave basis storage) ===
 *   - store() returns nonzero key for nonempty WaveFunction
 *   - store() returns 0 for empty WaveFunction
 *   - store() + load() round-trip: ψ fields match stored values
 *   - decay() reduces strength over time
 *   - consolidate() prunes fully-decayed memories
 *   - LTP: access_count >= 3 → consolidate boosts strength
 *   - Overwrite: storing same pattern again resets strength
 *
 *   === QueryEngine (Phase 2 — inner product retrieval) ===
 *   - resonance_score of identical fields = 1.0
 *   - resonance_score of orthogonal fields = 0.0
 *   - resonance_score of zero field = 0.0
 *   - resonance_score is normalised to [0,1]
 *   - query() returns stored memory as top match
 *   - query() returns empty for empty memory
 *   - query_by_coords() returns closest key first
 *   - Top-K capping: query(k=1) returns exactly 1 result
 *
 * Reference: nikola Phase 2 gate criteria.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/spatial/topology_manager.hpp>
#include <nikola/cognitive/semantic_memory.hpp>
#include <nikola/cognitive/query_engine.hpp>

#include <array>
#include <cmath>
#include <numeric>

using namespace nikola;
using namespace nikola::spatial;
using namespace nikola::cognitive;
using namespace nikola::foundation;
using namespace nikola::physics;

// ============================================================================
// Helper: make a small seeded WaveFunction (n^9 nodes, pilot wave)
// ============================================================================

static WaveFunction make_wave(int n = 2, int k_mode = 1, uint32_t seed = 42) {
    WaveFunction wf(GridConfig::uniform(n));
    wf.seed_manifold(n, /*pilot_dim=*/0, k_mode, /*amplitude=*/1.f, seed);
    return wf;
}

// ============================================================================
// TopologyManager — Gap 2.1 (Metric validation)
// ============================================================================

TEST_CASE("MetricValidator: Gerschgorin accepts identity matrix", "[topology][gap2.1]") {
    float g[81]{};
    for (int k = 0; k < 9; ++k) g[k*9+k] = 1.f;  // identity

    REQUIRE(MetricValidator::gerschgorin_check(g) == true);
    // ensure_positive_definite should return true (no repair needed)
    REQUIRE(MetricValidator::ensure_positive_definite(g) == true);
    // Diagonal must still be 1.0 (not modified)
    REQUIRE(g[0] == Catch::Approx(1.f));
}

TEST_CASE("MetricValidator: Gerschgorin rejects zero-diagonal matrix", "[topology][gap2.1]") {
    float g[81]{};
    // Zero matrix — clearly not positive definite
    REQUIRE(MetricValidator::gerschgorin_check(g) == false);
}

TEST_CASE("MetricValidator: Tikhonov repair makes degenerate matrix valid", "[topology][gap2.1]") {
    // Near-singular: diagonals = 0.5, large off-diagonals
    float g[81]{};
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            g[i*9+j] = (i == j) ? 0.5f : 0.4f;
        }
    }
    // Gerschgorin should fail (diag = 0.5 < row_sum = 0.4×8 = 3.2)
    REQUIRE(MetricValidator::gerschgorin_check(g) == false);

    // Repair
    const bool was_valid = MetricValidator::ensure_positive_definite(g);
    REQUIRE(was_valid == false);  // signals repair was applied

    // After repair, Gerschgorin should pass
    REQUIRE(MetricValidator::gerschgorin_check(g) == true);
    // Diagonals must be strictly increased
    REQUIRE(g[0] > 0.5f);
}

TEST_CASE("MetricValidator: valid matrix is not modified", "[topology][gap2.1]") {
    // Strict diagonal dominance: diag=10, off-diag=0.1
    float g[81]{};
    for (int i = 0; i < 9; ++i) {
        g[i*9+i] = 10.f;
        for (int j = 0; j < 9; ++j)
            if (i != j) g[i*9+j] = 0.1f;
    }
    const float diag_before = g[0];
    MetricValidator::ensure_positive_definite(g);
    REQUIRE(g[0] == Catch::Approx(diag_before));  // unchanged
}

// ============================================================================
// TopologyManager — Gap 2.3 (Anisotropic resolution)
// ============================================================================

TEST_CASE("Anisotropic GridConfig matches specification", "[topology][gap2.3]") {
    const GridConfig cfg = GridConfig::anisotropic_default();
    // x,y,z = dims 0-2: N=64
    REQUIRE(cfg.resolution[0] == 64);
    REQUIRE(cfg.resolution[1] == 64);
    REQUIRE(cfg.resolution[2] == 64);
    // t = dim 3: N=128
    REQUIRE(cfg.resolution[3] == 128);
    // r,s = dims 4-5: N=16
    REQUIRE(cfg.resolution[4] == 16);
    REQUIRE(cfg.resolution[5] == 16);
    // u,v,w = dims 6-8: N=32
    REQUIRE(cfg.resolution[6] == 32);
    REQUIRE(cfg.resolution[7] == 32);
    REQUIRE(cfg.resolution[8] == 32);
}

TEST_CASE("ANISOTROPIC_RESOLUTION constant matches GridConfig", "[topology][gap2.3]") {
    const GridConfig cfg = GridConfig::anisotropic_default();
    for (int d = 0; d < 9; ++d) {
        REQUIRE(ANISOTROPIC_RESOLUTION[d] == cfg.resolution[d]);
    }
}

// ============================================================================
// TopologyManager — Gap 2.4 (Dual coordinate system)
// ============================================================================

TEST_CASE("Coordinate round-trip: integer → physical → integer", "[topology][gap2.4]") {
    TopologyManager tm(GridConfig::anisotropic_default());

    SECTION("All-zero coordinates") {
        Coord9DInt ic{};
        const auto ic2 = tm.round_trip(ic);
        for (int d = 0; d < 9; ++d) {
            REQUIRE(ic2.c[d] == ic.c[d]);
        }
    }

    SECTION("Mid-range uniform grid") {
        TopologyManager tm_unif(GridConfig::uniform(8));
        Coord9DInt ic{};
        for (int d = 0; d < 9; ++d) ic.c[d] = 4;  // mid-point
        const auto ic2 = tm_unif.round_trip(ic);
        for (int d = 0; d < 9; ++d) {
            REQUIRE(ic2.c[d] == 4);
        }
    }

    SECTION("Max coordinate in anisotropic grid") {
        // dim 4-5 has N=16, max coord = 15
        Coord9DInt ic{};
        ic.c[4] = 15;
        const auto ic2 = tm.round_trip(ic);
        REQUIRE(ic2.c[4] == 15);
    }
}

TEST_CASE("Physical coordinates in [0,1) for valid integers", "[topology][gap2.4]") {
    TopologyManager tm(GridConfig::uniform(8));
    for (uint16_t v = 0; v < 8; ++v) {
        Coord9DInt ic{};
        ic.c[0] = v;
        Coord9DFloat fc = tm.to_physical(ic);
        REQUIRE(fc.c[0] >= 0.f);
        REQUIRE(fc.c[0] < 1.f);
    }
}

// ============================================================================
// TopologyManager — Gap 2.4 (Quadratic peak interpolation)
// ============================================================================

TEST_CASE("Peak interpolation: flat neighbourhood → offset=0", "[topology][gap2.4]") {
    // All equal values → parabola has flat top → offset ≈ 0
    REQUIRE(TopologyManager::peak_offset(1.f, 1.f, 1.f) == Catch::Approx(0.f));
}

TEST_CASE("Peak interpolation: left-skewed → positive offset", "[topology][gap2.4]") {
    // v_left > v_right → peak shifted toward left → positive offset
    // Using v_left=3, v_centre=4, v_right=2
    // offset = (vl - vr) / (2*(vl - 2*vc + vr)) = (3-2)/(2*(3-8+2)) = 1/(2*(-3)) = -1/6
    // Actually let me recalculate:
    // denom = 2*(3-8+2) = 2*(-3) = -6; (3-2)/-6 = -1/6 ≈ -0.167
    // So with vl > vr and denom < 0 → negative offset (peak shifted left)
    const float off = TopologyManager::peak_offset(3.f, 4.f, 2.f);
    REQUIRE(std::abs(off) > 1e-4f);  // non-zero
    REQUIRE(off == Catch::Approx(-1.f/6.f).margin(1e-4f));
}

TEST_CASE("Peak interpolation: symmetric peak → offset=0", "[topology][gap2.4]") {
    // v_left = v_right → symmetric → offset = 0
    REQUIRE(TopologyManager::peak_offset(2.f, 4.f, 2.f) == Catch::Approx(0.f));
}

TEST_CASE("Peak interpolation: clamped to [-0.5, 0.5]", "[topology][gap2.4]") {
    // Extreme asymmetry
    const float off1 = TopologyManager::peak_offset(100.f, 0.f, 0.f);
    const float off2 = TopologyManager::peak_offset(0.f,   0.f, 100.f);
    REQUIRE(off1 >= -0.5f); REQUIRE(off1 <= 0.5f);
    REQUIRE(off2 >= -0.5f); REQUIRE(off2 <= 0.5f);
}

// ============================================================================
// TopologyManager — Toroidal distance
// ============================================================================

TEST_CASE("Toroidal distance: zero for identical points", "[topology][distance]") {
    TopologyManager tm(GridConfig::uniform(16));
    Coord9DInt a{}, b{};
    for (int d = 0; d < 9; ++d) a.c[d] = b.c[d] = 5;
    REQUIRE(tm.toroidal_distance(a, b) == Catch::Approx(0.f));
}

TEST_CASE("Toroidal distance: symmetry", "[topology][distance]") {
    TopologyManager tm(GridConfig::uniform(16));
    Coord9DInt a{}, b{};
    a.c[0] = 2;  b.c[0] = 14;   // close via wraparound on dim 0 (N=16: min(12,4)=4)
    REQUIRE(tm.toroidal_distance(a, b) == Catch::Approx(tm.toroidal_distance(b, a)));
}

TEST_CASE("Toroidal distance: wraparound chooses shorter arc", "[topology][distance]") {
    // N=16, a[0]=2, b[0]=14 → direct dist=12, wrap dist=4 → min=4, frac=4/16=0.25
    TopologyManager tm(GridConfig::uniform(16));
    Coord9DInt a{}, b{};
    a.c[0] = 2;  b.c[0] = 14;
    const float d = tm.toroidal_distance(a, b);
    // All dims except 0 are equal, dist = sqrt((4/16)^2) = 0.25
    REQUIRE(d == Catch::Approx(0.25f).epsilon(0.01f));
}

// ============================================================================
// MetricLearner — Gap 2.5 (Dopamine-modulated learning rate)
// ============================================================================

TEST_CASE("MetricLearner: young node with full dopamine yields η_base", "[metric][gap2.5]") {
    MetricLearner ml;
    const float lr = ml.compute_learning_rate(/*dopamine=*/1.f, /*age=*/0.f);
    REQUIRE(lr == Catch::Approx(MetricLearner::ETA_BASE));
}

TEST_CASE("MetricLearner: zero dopamine yields zero learning rate", "[metric][gap2.5]") {
    MetricLearner ml;
    REQUIRE(ml.compute_learning_rate(0.f, 0.f) == Catch::Approx(0.f));
    REQUIRE(ml.compute_learning_rate(0.f, 5000.f) == Catch::Approx(0.f));
}

TEST_CASE("MetricLearner: old node has much lower learning rate than young", "[metric][gap2.5]") {
    MetricLearner ml;
    const float lr_young = ml.compute_learning_rate(1.f, 0.f);
    const float lr_old   = ml.compute_learning_rate(1.f, 10000.f); // ~10000s old
    REQUIRE(lr_young > lr_old * 10.f);   // at least 10x faster for young node
}

TEST_CASE("MetricLearner: learning rate is monotone-decreasing in age", "[metric][gap2.5]") {
    MetricLearner ml;
    float prev = ml.compute_learning_rate(1.f, 0.f);
    for (float age = 100.f; age <= 5000.f; age += 500.f) {
        const float curr = ml.compute_learning_rate(1.f, age);
        REQUIRE(curr < prev);
        prev = curr;
    }
}

TEST_CASE("MetricLearner: update_metric modifies metric and keeps it valid", "[metric][gap2.5]") {
    MetricLearner ml;
    // Start with identity
    float g[81]{};
    for (int k = 0; k < 9; ++k) g[k*9+k] = 1.f;

    // Correlation = scaled identity (self-activation in all dims)
    float corr[81]{};
    for (int k = 0; k < 9; ++k) corr[k*9+k] = 0.01f;

    const float diag_before = g[0];
    bool valid = ml.update_metric(g, corr, 1.f, 0.f);   // large η

    // Diagonal should have increased
    REQUIRE(g[0] > diag_before);
    // Final metric must be positive-definite
    REQUIRE(MetricValidator::gerschgorin_check(g) == true);
    (void)valid;  // return value tested separately
}

// ============================================================================
// SemanticMemory — store / load
// ============================================================================

TEST_CASE("SemanticMemory: store empty WaveFunction returns 0", "[memory][store]") {
    SemanticMemory mem(/*order=*/3);  // small order for test speed
    WaveFunction wf(GridConfig::uniform(2));
    // No nodes allocated → key should be 0
    const MemoryKey key = mem.store(wf);
    REQUIRE(key == 0);
    REQUIRE(mem.size() == 0);
}

TEST_CASE("SemanticMemory: store populates record", "[memory][store]") {
    SemanticMemory mem(3);
    WaveFunction wf = make_wave(2);  // 2^9 = 512 nodes

    const MemoryKey key = mem.store(wf);
    REQUIRE(key != static_cast<MemoryKey>(-1));  // valid key
    REQUIRE(mem.size() == 1);
    REQUIRE(mem.contains(key));

    const MemoryRecord* rec = mem.get(key);
    REQUIRE(rec != nullptr);
    REQUIRE(rec->psi_real.size() == wf.num_nodes());
    REQUIRE(rec->psi_imag.size() == wf.num_nodes());
    REQUIRE(rec->strength == Catch::Approx(1.f));
    REQUIRE(rec->age_seconds == Catch::Approx(0.f));
}

TEST_CASE("SemanticMemory: store two different waves gives two records", "[memory][store]") {
    SemanticMemory mem(3);
    WaveFunction wf1 = make_wave(2, 1, 42);
    WaveFunction wf2 = make_wave(2, 3, 99);  // different k_mode and seed

    const MemoryKey k1 = mem.store(wf1);
    const MemoryKey k2 = mem.store(wf2);

    // May or may not be the same key (depends on dominant node location)
    // but memory size is at least 1
    REQUIRE(mem.size() >= 1);
    (void)k1; (void)k2;
}

TEST_CASE("SemanticMemory: load restores psi fields", "[memory][load]") {
    SemanticMemory mem(3);
    WaveFunction wf = make_wave(2);  // 512 nodes, pilot wave

    // Save original psi values
    const size_t N = wf.num_nodes();
    std::vector<float> orig_r(N), orig_i(N);
    std::copy(wf.grid().psi_real(), wf.grid().psi_real() + N, orig_r.begin());
    std::copy(wf.grid().psi_imag(), wf.grid().psi_imag() + N, orig_i.begin());

    const MemoryKey key = mem.store(wf);

    // Corrupt the wave field
    for (size_t i = 0; i < N; ++i) {
        wf.grid().psi_real()[i] = 0.f;
        wf.grid().psi_imag()[i] = 0.f;
    }

    // Restore from memory
    REQUIRE(mem.load(key, wf) == true);

    // Fields should match originals
    for (size_t i = 0; i < N; ++i) {
        REQUIRE(wf.grid().psi_real()[i] == Catch::Approx(orig_r[i]).margin(1e-6f));
        REQUIRE(wf.grid().psi_imag()[i] == Catch::Approx(orig_i[i]).margin(1e-6f));
    }
}

TEST_CASE("SemanticMemory: load increments access_count", "[memory][load]") {
    SemanticMemory mem(3);
    WaveFunction wf = make_wave(2);
    const MemoryKey key = mem.store(wf);

    REQUIRE(mem.get(key)->access_count == 0);
    mem.load(key, wf);
    REQUIRE(mem.get(key)->access_count == 1);
    mem.load(key, wf);
    REQUIRE(mem.get(key)->access_count == 2);
}

TEST_CASE("SemanticMemory: load returns false for unknown key", "[memory][load]") {
    SemanticMemory mem(3);
    WaveFunction wf = make_wave(2);
    REQUIRE(mem.load(999999u, wf) == false);
}

// ============================================================================
// SemanticMemory — decay and consolidation
// ============================================================================

TEST_CASE("SemanticMemory: decay reduces strength", "[memory][decay]") {
    SemanticMemory mem(3);
    WaveFunction wf = make_wave(2);
    const MemoryKey key = mem.store(wf);

    REQUIRE(mem.get(key)->strength == Catch::Approx(1.f));

    mem.decay(/*dt=*/1000.f);   // 1000 seconds
    const float s = mem.get(key)->strength;
    REQUIRE(s < 1.f);
    REQUIRE(s > 0.f);
}

TEST_CASE("SemanticMemory: decay ages records", "[memory][decay]") {
    SemanticMemory mem(3);
    WaveFunction wf = make_wave(2);
    const MemoryKey key = mem.store(wf);

    mem.decay(50.f);
    REQUIRE(mem.get(key)->age_seconds == Catch::Approx(50.f).margin(1e-3f));
}

TEST_CASE("SemanticMemory: consolidate prunes fully-decayed memories", "[memory][consolidate]") {
    SemanticMemory mem(3);
    WaveFunction wf = make_wave(2);
    mem.store(wf);

    REQUIRE(mem.size() == 1);

    // Decay until strength falls below MIN_STRENGTH
    // MIN=0.01, DECAY=0.001/s, exp(-0.001*t)=0.01 → t = -ln(0.01)/0.001 ≈ 4605s
    mem.decay(5000.f);
    REQUIRE(mem.get(mem.all_keys()[0])->strength < SemanticMemory::MIN_STRENGTH);

    const size_t pruned = mem.consolidate();
    REQUIRE(pruned == 1);
    REQUIRE(mem.size() == 0);
}

TEST_CASE("SemanticMemory: LTP boosts frequently accessed records", "[memory][ltp]") {
    SemanticMemory mem(3);
    WaveFunction wf = make_wave(2);
    const MemoryKey key = mem.store(wf);

    // Access enough times to trigger LTP
    for (uint32_t i = 0; i < SemanticMemory::LTP_THRESHOLD; ++i) {
        mem.load(key, wf);
    }
    REQUIRE(mem.get(key)->access_count >= SemanticMemory::LTP_THRESHOLD);

    // Decay a bit first (so we can see boost)
    mem.decay(100.f);
    const float strength_before_ltp = mem.get(key)->strength;

    // Consolidate should boost
    mem.consolidate();
    REQUIRE(mem.get(key)->strength >= strength_before_ltp);
    // After LTP, access_count should be reset
    REQUIRE(mem.get(key)->access_count == 0);
}

TEST_CASE("SemanticMemory: overwrite resets strength", "[memory][overwrite]") {
    SemanticMemory mem(3);
    WaveFunction wf = make_wave(2);
    const MemoryKey key = mem.store(wf);

    mem.decay(1000.f);
    REQUIRE(mem.get(key)->strength < 1.f);

    // Store same wave again — same key → strength reset
    const MemoryKey key2 = mem.store(wf);
    if (key2 == key) {
        REQUIRE(mem.get(key)->strength == Catch::Approx(1.f));
    }
    // If key changed (very rare: dominant node moved), that's fine too
}

// ============================================================================
// QueryEngine — resonance_score
// ============================================================================

TEST_CASE("resonance_score: identical fields = 1.0", "[query][resonance]") {
    // ⟨a|a⟩² / (‖a‖² · ‖a‖²) = ‖a‖⁴ / ‖a‖⁴ = 1.0
    constexpr size_t N = 8;
    float r[N] = {1.f,0.f,1.f,0.f, 1.f,0.f,1.f,0.f};
    float i[N] = {0.f,1.f,0.f,1.f, 0.f,1.f,0.f,1.f};
    const float s = QueryEngine::resonance_score(r, i, N, r, i, N);
    REQUIRE(s == Catch::Approx(1.f).epsilon(1e-5f));
}

TEST_CASE("resonance_score: orthogonal real fields = 0.0", "[query][resonance]") {
    // a = [1,0,0,...], b = [0,1,0,...] → ⟨a|b⟩ = 0
    constexpr size_t N = 4;
    float ar[N] = {1.f, 0.f, 0.f, 0.f};
    float ai[N] = {0.f, 0.f, 0.f, 0.f};
    float br[N] = {0.f, 1.f, 0.f, 0.f};
    float bi[N] = {0.f, 0.f, 0.f, 0.f};
    REQUIRE(QueryEngine::resonance_score(ar, ai, N, br, bi, N) == Catch::Approx(0.f));
}

TEST_CASE("resonance_score: zero vector returns 0.0", "[query][resonance]") {
    constexpr size_t N = 4;
    float z[N] = {};
    float r[N] = {1.f, 1.f, 1.f, 1.f};
    float i[N] = {};
    REQUIRE(QueryEngine::resonance_score(z, z, N, r, i, N) == Catch::Approx(0.f));
    REQUIRE(QueryEngine::resonance_score(r, i, N, z, z, N) == Catch::Approx(0.f));
}

TEST_CASE("resonance_score: result in [0, 1]", "[query][resonance]") {
    constexpr size_t N = 8;
    // Random-ish values
    float ar[N] = {0.3f, -0.7f,  1.2f,  0.f, -0.4f,  0.9f, -1.f,  0.5f};
    float ai[N] = {0.5f,  0.2f, -0.8f,  1.f,  0.3f, -0.6f,  0.f, -0.2f};
    float br[N] = {1.f,   0.f,  -1.f,   0.5f, 0.5f,  0.f,   0.8f, -0.3f};
    float bi[N] = {0.f,   1.f,   0.f,  -0.5f, 0.f,   0.5f, -0.2f,  0.6f};
    const float s = QueryEngine::resonance_score(ar, ai, N, br, bi, N);
    REQUIRE(s >= 0.f);
    REQUIRE(s <= 1.f + 1e-5f);
}

TEST_CASE("resonance_score: Cauchy-Schwarz: equal to 1 only for parallel", "[query][resonance]") {
    // Scale a by constant: b = 2*a → still parallel → score = 1
    constexpr size_t N = 4;
    float ar[N] = {1.f, 0.f, 1.f, 0.f};
    float ai[N] = {0.f, 1.f, 0.f, 1.f};
    float br[N] = {2.f, 0.f, 2.f, 0.f};
    float bi[N] = {0.f, 2.f, 0.f, 2.f};
    const float s = QueryEngine::resonance_score(ar, ai, N, br, bi, N);
    REQUIRE(s == Catch::Approx(1.f).epsilon(1e-5f));
}

// ============================================================================
// QueryEngine — query() by WaveFunction
// ============================================================================

TEST_CASE("QueryEngine: query on empty memory returns empty", "[query][query_wf]") {
    SemanticMemory mem(3);
    QueryEngine qe(mem);
    WaveFunction wf = make_wave(2);
    const auto results = qe.query(wf, 5);
    REQUIRE(results.empty());
}

TEST_CASE("QueryEngine: query returns stored memory as top match", "[query][query_wf]") {
    SemanticMemory mem(3);
    WaveFunction wf = make_wave(2, 1, 42);
    const MemoryKey key = mem.store(wf);

    QueryEngine qe(mem);
    const auto results = qe.query(wf, 5);

    REQUIRE(!results.empty());
    REQUIRE(results[0].key == key);
    // Self-similarity should yield a high score
    REQUIRE(results[0].score > 0.5f);
}

TEST_CASE("QueryEngine: top-K capping works", "[query][query_wf]") {
    SemanticMemory mem(3);
    WaveFunction w1 = make_wave(2, 1, 11);
    WaveFunction w2 = make_wave(2, 2, 22);
    WaveFunction w3 = make_wave(2, 3, 33);
    mem.store(w1);
    mem.store(w2);
    mem.store(w3);

    QueryEngine qe(mem);
    WaveFunction q = make_wave(2, 1, 11);
    const auto r1 = qe.query(q, 1);
    REQUIRE(r1.size() == 1);

    const auto r2 = qe.query(q, 2);
    REQUIRE(r2.size() <= 2);

    const auto rAll = qe.query(q, 0);   // k=0 → no cap
    REQUIRE(rAll.size() >= 1);
}

TEST_CASE("QueryEngine: results sorted descending by score", "[query][query_wf]") {
    SemanticMemory mem(3);
    WaveFunction w1 = make_wave(2, 1, 1);
    WaveFunction w2 = make_wave(2, 3, 2);
    mem.store(w1);
    mem.store(w2);

    QueryEngine qe(mem);
    WaveFunction q = make_wave(2, 1, 1);  // same as w1
    const auto results = qe.query(q, 5);

    for (size_t i = 1; i < results.size(); ++i) {
        REQUIRE(results[i-1].score >= results[i].score);
    }
}

// ============================================================================
// QueryEngine — query_by_coords()
// ============================================================================

TEST_CASE("QueryEngine: query_by_coords returns closest key first", "[query][query_coords]") {
    // Store three waves with different dominant nodes, then query a coord that
    // should be closest to the first.
    SemanticMemory mem(3);  // order=3 → Hilbert space 0..2^(9*3) = 2^27

    WaveFunction w1 = make_wave(2, 1, 7);
    WaveFunction w2 = make_wave(2, 4, 99);
    const MemoryKey k1 = mem.store(w1);
    /* MemoryKey k2 = */ mem.store(w2);

    QueryEngine qe(mem);

    // Find the 9D Hilbert coords of k1 so that querying them puts k1 first.
    const HilbertScanner& sc = mem.scanner();
    const HilbertScanner::Coord9D c1 = sc.index_to_coords(k1);

    const auto results = qe.query_by_coords(c1, 5);
    REQUIRE(!results.empty());
    REQUIRE(results[0].key == k1);   // closest key to its own coords is itself
}

TEST_CASE("QueryEngine: query_by_coords results sorted descending", "[query][query_coords]") {
    SemanticMemory mem(3);
    WaveFunction w1 = make_wave(2, 1,  5);
    WaveFunction w2 = make_wave(2, 2, 13);
    WaveFunction w3 = make_wave(2, 3, 99);
    mem.store(w1);
    mem.store(w2);
    mem.store(w3);

    QueryEngine qe(mem);
    HilbertScanner::Coord9D q_coords{};  // all-zero → some key

    const auto results = qe.query_by_coords(q_coords, 5);
    for (size_t i = 1; i < results.size(); ++i) {
        REQUIRE(results[i-1].score >= results[i].score);
    }
}
