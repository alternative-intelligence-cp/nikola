// ============================================================
// v0.1.5 — Spatial Index Test Suite
// tests/unit/spatial_index_test.cpp
//
// Validates O(log N) spatial lookup for 9D toroidal manifold:
//   §1  Build and find
//   §2  Morton-key neighbors
//   §3  Grid neighbors (face-adjacent, toroidal)
//   §4  Empty/edge cases
//   §5  Production grid size (3^9 = 19,683 nodes)
//   §6  Benchmark: lookup latency
// ============================================================

#include <catch2/catch_test_macros.hpp>

#include <nikola/spatial/spatial_index.hpp>
#include <nikola/spatial/morton_encoder.hpp>

#include <chrono>
#include <cmath>
#include <random>
#include <vector>

using namespace nikola::spatial;

// Helper: generate a full grid of coords [0, side)^9
static std::vector<Coord9D> make_grid(uint32_t side) {
    uint32_t total = 1;
    for (int d = 0; d < MORTON_DIMS; ++d) total *= side;

    std::vector<Coord9D> coords(total);
    for (uint32_t i = 0; i < total; ++i) {
        uint32_t tmp = i;
        for (int d = 0; d < MORTON_DIMS; ++d) {
            coords[i][d] = tmp % side;
            tmp /= side;
        }
    }
    return coords;
}

// ═══════════════════════════════════════════════════════════════════════════
// §1  Build and find
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("§1-1 SpatialIndex: build and find — small grid", "[spatial][index]") {
    auto coords = make_grid(3);  // 3^9 = 19,683
    SpatialIndex idx;
    idx.build(coords);
    REQUIRE(idx.size() == 19683);

    // Every coord should be findable
    for (const auto& c : coords) {
        auto key = morton_encode(c);
        auto* entry = idx.find(key);
        REQUIRE(entry != nullptr);
        REQUIRE(entry->coord == c);
    }
}

TEST_CASE("§1-2 SpatialIndex: find — missing key returns nullptr", "[spatial][index]") {
    // Index a 2^9 grid but query with coords outside it
    auto coords = make_grid(2);  // 2^9 = 512
    SpatialIndex idx;
    idx.build(coords);

    Coord9D outside{};
    outside[0] = 5;  // Not in [0,2) grid
    auto* entry = idx.find(morton_encode(outside));
    REQUIRE(entry == nullptr);
}

// ═══════════════════════════════════════════════════════════════════════════
// §2  Morton-key neighbors
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("§2-1 SpatialIndex: morton_neighbors returns correct count",
          "[spatial][index]") {
    auto coords = make_grid(3);
    SpatialIndex idx;
    idx.build(coords);

    Coord9D center{};
    center.fill(1);  // Center of 3^9 grid
    auto neighbors = idx.morton_neighbors(center, 10);
    REQUIRE(neighbors.size() == 10);
}

TEST_CASE("§2-2 SpatialIndex: morton_neighbors excludes query point",
          "[spatial][index]") {
    auto coords = make_grid(3);
    SpatialIndex idx;
    idx.build(coords);

    Coord9D center{};
    center.fill(1);
    MortonKey center_key = morton_encode(center);
    auto neighbors = idx.morton_neighbors(center, 18);

    for (const auto& n : neighbors) {
        REQUIRE(n.key != center_key);
    }
}

TEST_CASE("§2-3 SpatialIndex: morton_neighbors — request more than available",
          "[spatial][index]") {
    auto coords = make_grid(2);  // 512 points
    SpatialIndex idx;
    idx.build(coords);

    Coord9D q{};
    auto neighbors = idx.morton_neighbors(q, 1000);
    // Should get at most 511 (all except self)
    REQUIRE(neighbors.size() <= 511);
    REQUIRE(neighbors.size() >= 100);  // Should get many
}

// ═══════════════════════════════════════════════════════════════════════════
// §3  Grid neighbors (face-adjacent, toroidal)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("§3-1 SpatialIndex: grid_neighbors — center of 3^9",
          "[spatial][index]") {
    auto coords = make_grid(3);
    SpatialIndex idx;
    idx.build(coords);

    Coord9D center{};
    center.fill(1);
    auto neighbors = idx.grid_neighbors(center, 3);

    // 9 dims × 2 directions = 18 face-adjacent neighbors, all in grid
    REQUIRE(neighbors.size() == 18);

    // Each neighbor should differ from center in exactly one dimension by ±1
    for (const auto& n : neighbors) {
        int diffs = 0;
        for (int d = 0; d < MORTON_DIMS; ++d) {
            if (n.coord[d] != center[d]) diffs++;
        }
        REQUIRE(diffs == 1);
    }
}

TEST_CASE("§3-2 SpatialIndex: grid_neighbors — toroidal wrapping",
          "[spatial][index]") {
    auto coords = make_grid(3);
    SpatialIndex idx;
    idx.build(coords);

    // Corner point (0,0,...,0) — neighbors should wrap to (2,0,...,0) etc.
    Coord9D corner{};
    auto neighbors = idx.grid_neighbors(corner, 3);
    REQUIRE(neighbors.size() == 18);

    // Check that dim 0 has neighbors at coord[0]=1 and coord[0]=2 (wrapped)
    bool found_plus = false, found_wrap = false;
    for (const auto& n : neighbors) {
        if (n.coord[0] == 1) found_plus = true;
        if (n.coord[0] == 2) found_wrap = true;
    }
    REQUIRE(found_plus);
    REQUIRE(found_wrap);  // toroidal wrap: (0-1) mod 3 = 2
}

TEST_CASE("§3-3 SpatialIndex: grid_neighbors — all neighbors are in the index",
          "[spatial][index]") {
    auto coords = make_grid(3);
    SpatialIndex idx;
    idx.build(coords);

    // Check 100 random points
    std::mt19937 rng(7);
    std::uniform_int_distribution<uint32_t> dist(0, 2);
    for (int i = 0; i < 100; ++i) {
        Coord9D q{};
        for (auto& v : q) v = dist(rng);
        auto neighbors = idx.grid_neighbors(q, 3);
        REQUIRE(neighbors.size() == 18);
        for (const auto& n : neighbors) {
            REQUIRE(idx.find(n.key) != nullptr);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §4  Edge cases
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("§4-1 SpatialIndex: empty index", "[spatial][index]") {
    SpatialIndex idx;
    REQUIRE(idx.empty());
    REQUIRE(idx.size() == 0);

    Coord9D q{};
    REQUIRE(idx.find(morton_encode(q)) == nullptr);
    REQUIRE(idx.morton_neighbors(q, 5).empty());
    REQUIRE(idx.grid_neighbors(q, 3).empty());
}

TEST_CASE("§4-2 SpatialIndex: single-point index", "[spatial][index]") {
    Coord9D point{};
    point.fill(5);
    SpatialIndex idx;
    idx.build({point});
    REQUIRE(idx.size() == 1);
    REQUIRE(idx.find(morton_encode(point)) != nullptr);

    // Morton neighbors of self should be empty (only self in index)
    auto mn = idx.morton_neighbors(point, 5);
    REQUIRE(mn.empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// §5  Production grid (3^9 = 19,683 nodes)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("§5-1 SpatialIndex: 19,683-node grid — build + 1000 lookups",
          "[spatial][index]") {
    auto coords = make_grid(3);
    SpatialIndex idx;
    idx.build(coords);
    REQUIRE(idx.size() == 19683);

    // 1000 random lookups should all succeed
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, 2);
    for (int i = 0; i < 1000; ++i) {
        Coord9D q{};
        for (auto& v : q) v = dist(rng);
        REQUIRE(idx.find(morton_encode(q)) != nullptr);
    }
}

TEST_CASE("§5-2 SpatialIndex: grid_neighbors O(log N) — production grid",
          "[spatial][index]") {
    auto coords = make_grid(3);
    SpatialIndex idx;
    idx.build(coords);

    // Verify all 19,683 points have exactly 18 face-adjacent neighbors
    for (const auto& c : coords) {
        auto neighbors = idx.grid_neighbors(c, 3);
        REQUIRE(neighbors.size() == 18);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §6  Benchmark: lookup latency
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("§6-1 SpatialIndex: find() latency — 19,683 nodes",
          "[spatial][index][!benchmark]") {
    auto coords = make_grid(3);
    SpatialIndex idx;
    idx.build(coords);

    // Pre-encode keys
    std::vector<MortonKey> keys(coords.size());
    for (size_t i = 0; i < coords.size(); ++i) {
        keys[i] = morton_encode(coords[i]);
    }

    constexpr int ITERS = 100'000;
    std::mt19937 rng(0);
    std::uniform_int_distribution<size_t> pick(0, keys.size() - 1);

    auto t0 = std::chrono::steady_clock::now();
    volatile const SpatialIndex::Entry* sink = nullptr;
    for (int i = 0; i < ITERS; ++i) {
        sink = idx.find(keys[pick(rng)]);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / ITERS;

    INFO("find() latency: " << ns << " ns/op (19,683 entries, log2=" 
         << std::log2(19683.0) << ")");
    // Binary search over 19,683 entries should be < 1 µs
    REQUIRE(ns < 1000.0);
}

TEST_CASE("§6-2 SpatialIndex: grid_neighbors() latency — 19,683 nodes",
          "[spatial][index][!benchmark]") {
    auto coords = make_grid(3);
    SpatialIndex idx;
    idx.build(coords);

    constexpr int ITERS = 10'000;
    std::mt19937 rng(1);
    std::uniform_int_distribution<size_t> pick(0, coords.size() - 1);

    auto t0 = std::chrono::steady_clock::now();
    volatile size_t sink = 0;
    for (int i = 0; i < ITERS; ++i) {
        auto n = idx.grid_neighbors(coords[pick(rng)], 3);
        sink = n.size();
    }
    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / ITERS;

    INFO("grid_neighbors() latency: " << ns << " ns/op (18 lookups × log2(19683))");
    // 18 binary searches should be < 50 µs
    REQUIRE(ns < 50000.0);
}
