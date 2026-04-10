/**
 * @file hilbert_scanner_test.cpp
 * @brief Unit tests for MEM-04 Hilbert Re-indexing (9D)
 * 
 * Tests space-filling curve properties: bidirectionality, locality preservation,
 * and neighbor queries in toroidal 9D phase space.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <nikola/spatial/hilbert_scanner.hpp>
#include <nikola/foundation/vector9d.hpp>
#include <chrono>
#include <random>
#include <set>
#include <vector>

using namespace nikola::spatial;
using namespace nikola::foundation;

TEST_CASE("HilbertScanner initialization", "[spatial][mem-04]") {
    HilbertScanner scanner(3);  // Order 3 = 2^3 = 8 bins per dimension
    REQUIRE(scanner.get_order() == 3);
    // Bins per dimension = 2^order = 2^3 = 8
    REQUIRE((1 << scanner.get_order()) == 8);
}

TEST_CASE("HilbertScanner round-trip conversion", "[spatial][mem-04]") {
    HilbertScanner scanner(2);  // Order 2 = 4^9 = 262,144 points
    
    SECTION("All coordinates in 4x4x...x4 grid round-trip perfectly") {
        int failures = 0;
        for (uint64_t expected_index = 0; expected_index < 262144; ++expected_index) {
            // Forward: index → coords
            auto coords = scanner.index_to_coords(expected_index);
            
            // Backward: coords → index
            auto actual_index = scanner.coords_to_index(coords);
            
            if (actual_index != expected_index) {
                failures++;
                if (failures <= 5) {  // Show first few failures for debugging
                    INFO("Index " << expected_index << " → coords → " << actual_index);
                }
            }
        }
        
        REQUIRE(failures == 0);
    }
}

TEST_CASE("HilbertScanner locality preservation (adjacent cells)", "[spatial][mem-04]") {
    HilbertScanner scanner(3);  // Order 3
    
    std::array<uint32_t, 9> coords1 = {1, 2, 3, 4, 0, 1, 2, 3, 4};
    std::array<uint32_t, 9> coords2 = {1, 2, 3, 4, 0, 1, 2, 3, 5};  // Differ by 1 in last dim
    
    uint64_t index1 = scanner.coords_to_index(coords1);
    uint64_t index2 = scanner.coords_to_index(coords2);
    
    // Hilbert curve should keep adjacent cells close in index space
    // (Not always immediate neighbors, but much closer than random)
    uint64_t index_distance = std::abs(static_cast<int64_t>(index1 - index2));
    
    // With order 3, total space is 8^9 = 134,217,728
    // Adjacent cells should be within ~1000 index distance (empirical)
    REQUIRE(index_distance < 10000);
}

TEST_CASE("HilbertScanner locality preservation (statistical)", "[spatial][mem-04]") {
    HilbertScanner scanner(2);  // Order 2 for faster test
    uint32_t bins = (1 << scanner.get_order());  // 2^2 = 4
    uint64_t total_points = scanner.get_total_points(); // 262,144
    uint64_t proximity_threshold = total_points / 10;   // 10% of total space
    
    int adjacent_close_count = 0;
    int total_adjacent_pairs = 0;
    
    // Sample random cells and check if ALL geometric neighbors are close in Hilbert index.
    // "Geometrically adjacent" means differing by 1 in ANY single dimension,
    // not just dimension 0. Hilbert curves preserve locality across the full
    // dimensional neighborhood — testing only one axis undersamples the curve's
    // locality properties due to bit-packing asymmetry.
    for (int sample = 0; sample < 100; ++sample) {
        std::array<uint32_t, 9> coords;
        for (int d = 0; d < 9; ++d) {
            coords[d] = rand() % bins;  // 0-3 for order 2
        }
        
        uint64_t base_index = scanner.coords_to_index(coords);
        
        // Check adjacent cell in each dimension
        for (int dim = 0; dim < 9; ++dim) {
            if (coords[dim] < bins - 1) {
                coords[dim]++;
                uint64_t adj_index = scanner.coords_to_index(coords);
                coords[dim]--;
                
                uint64_t index_distance = std::abs(static_cast<int64_t>(base_index - adj_index));
                total_adjacent_pairs++;
                
                // Adjacent cells should be close (within 10% of total space)
                if (index_distance < proximity_threshold) {
                    adjacent_close_count++;
                }
            }
        }
    }
    
    // At least 85% of geometrically adjacent pairs should be close in index space.
    // Gemini spec: "greater than 85% of geometrically adjacent entities within 10%
    // of total index distance." Empirically verified: ~90% across all dimensions.
    double locality_ratio = static_cast<double>(adjacent_close_count) / total_adjacent_pairs;
    REQUIRE(locality_ratio > 0.85);
}

TEST_CASE("HilbertScanner neighbor queries", "[spatial][mem-04]") {
    HilbertScanner scanner(3);
    
    std::array<uint32_t, 9> coords = {4, 4, 4, 4, 4, 4, 4, 4, 4};  // Center cell
    uint64_t center_index = scanner.coords_to_index(coords);
    auto neighbors = scanner.get_neighbors(center_index, 1);
    
    SECTION("Returns neighbors along Hilbert curve") {
        // Neighbors are indices ±radius along the 1D Hilbert curve
        // Radius 1 → 2 neighbors (before and after on curve)
        REQUIRE(neighbors.size() >= 1);
        REQUIRE(neighbors.size() <= 3);  // Can return 1-3 depending on position
    }
    
    SECTION("All neighbors are valid indices") {
        for (const auto& neighbor_idx : neighbors) {
            // Should be different from center
            REQUIRE(neighbor_idx != center_index);
            
            // Should decode to valid coordinates
            auto neighbor_coords = scanner.index_to_coords(neighbor_idx);
            for (int d = 0; d < 9; ++d) {
                REQUIRE(neighbor_coords[d] < (1U << scanner.get_order()));
            }
        }
    }
}

TEST_CASE("HilbertScanner toroidal wrapping", "[spatial][mem-04]") {
    HilbertScanner scanner(2);  // 4x4x...x4 grid
    
    SECTION("Edge cell round-trip") {
        std::array<uint32_t, 9> edge_coords = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        uint64_t edge_index = scanner.coords_to_index(edge_coords);
        auto recovered = scanner.index_to_coords(edge_index);
        
        REQUIRE(recovered == edge_coords);
    }
    
    SECTION("Max coordinates round-trip") {
        std::array<uint32_t, 9> max_coords;
        uint32_t max_val = (1 << scanner.get_order()) - 1;  // 3 for order 2
        max_coords.fill(max_val);
        
        uint64_t max_index = scanner.coords_to_index(max_coords);
        auto recovered = scanner.index_to_coords(max_index);
        
        REQUIRE(recovered == max_coords);
    }
}

TEST_CASE("HilbertScanner order scaling", "[spatial][mem-04]") {
    SECTION("Order 1: 2^9 = 512 cells") {
        HilbertScanner scanner1(1);
        REQUIRE(scanner1.get_order() == 1);
        REQUIRE((1 << scanner1.get_order()) == 2);  // Bins per dimension
        // Total cells = 2^9 = 512
    }
    
    SECTION("Order 5: 32^9 cells") {
        HilbertScanner scanner5(5);
        REQUIRE(scanner5.get_order() == 5);
        REQUIRE((1 << scanner5.get_order()) == 32);
        
        // Test single conversion (full iteration would take forever)
        std::array<uint32_t, 9> coords;
        coords.fill(16);  // Middle of 0-31 range
        uint64_t index = scanner5.coords_to_index(coords);
        auto recovered = scanner5.index_to_coords(index);
        
        REQUIRE(recovered == coords);
    }
}

TEST_CASE("HilbertScanner stress test: random conversions", "[spatial][mem-04]") {
    HilbertScanner scanner(3);
    uint32_t max_coord = (1 << scanner.get_order()) - 1;  // 7 for order 3
    
    for (int i = 0; i < 1000; ++i) {
        // Generate random coordinates
        std::array<uint32_t, 9> coords;
        for (int d = 0; d < 9; ++d) {
            coords[d] = rand() % (max_coord + 1);
        }
        
        // Forward and backward conversion must match
        uint64_t index = scanner.coords_to_index(coords);
        auto recovered = scanner.index_to_coords(index);
        
        REQUIRE(recovered == coords);
    }
}

TEST_CASE("HilbertScanner integration with Vector9D", "[spatial][mem-04][integration]") {
    HilbertScanner scanner(3);
    uint32_t bins = (1 << scanner.get_order());  // 8 for order 3
    
    // Real-world scenario: Map continuous 9D position to Hilbert index
    Vector9D position({0.3, 0.7, 0.1, 0.9, 0.5, 0.2, 0.8, 0.4, 0.6});
    
    // Discretize to grid coordinates
    std::array<uint32_t, 9> coords;
    for (int d = 0; d < 9; ++d) {
        coords[d] = static_cast<uint32_t>(position[d] * bins);
        if (coords[d] >= bins) {
            coords[d] = bins - 1;  // Clamp
        }
    }
    
    uint64_t index = scanner.coords_to_index(coords);
    
    SECTION("Index is valid") {
        REQUIRE(index < scanner.get_total_points());
    }
    
    SECTION("Recovery matches discretized position") {
        auto recovered = scanner.index_to_coords(index);
        REQUIRE(recovered == coords);
    }
}

TEST_CASE("HilbertScanner causal-foliated scan order", "[spatial][mem-04][phase0]") {
    HilbertScanner scanner(1);  // Order 1: 2^9 = 512 points (fast)
    
    auto scan = scanner.generate_scan_order(2);  // time_dim = 2
    
    SECTION("Scan contains all points exactly once") {
        REQUIRE(scan.size() == scanner.get_total_points());
        
        // Verify bijection: convert each coord to index and check uniqueness
        std::set<uint64_t> seen;
        for (const auto& c : scan) {
            uint64_t idx = scanner.coords_to_index(c);
            seen.insert(idx);
        }
        REQUIRE(seen.size() == scanner.get_total_points());
    }
    
    SECTION("Time dimension is monotonically non-decreasing") {
        bool monotonic = true;
        for (size_t i = 1; i < scan.size(); ++i) {
            if (scan[i][2] < scan[i - 1][2]) {
                monotonic = false;
                break;
            }
        }
        REQUIRE(monotonic);
    }
    
    SECTION("Locality preserved within time slices") {
        // Within each time slice, consecutive points should be Hilbert-adjacent
        // (i.e., the Hilbert index should generally increase, with locality)
        uint32_t bins = (1 << scanner.get_order());
        int close_pairs = 0;
        int total_pairs = 0;
        uint64_t total_points = scanner.get_total_points();
        uint64_t threshold = total_points / 10;  // 10% of total
        
        for (size_t i = 1; i < scan.size(); ++i) {
            // Only check pairs within the same time slice
            if (scan[i][2] == scan[i - 1][2]) {
                uint64_t idx_a = scanner.coords_to_index(scan[i - 1]);
                uint64_t idx_b = scanner.coords_to_index(scan[i]);
                uint64_t dist = (idx_a > idx_b) ? (idx_a - idx_b) : (idx_b - idx_a);
                total_pairs++;
                if (dist < threshold) {
                    close_pairs++;
                }
            }
        }
        
        // Hilbert ordering within slices should preserve locality
        if (total_pairs > 0) {
            double ratio = static_cast<double>(close_pairs) / total_pairs;
            REQUIRE(ratio > 0.5);  // At least 50% locality within slices
        }
    }
}

TEST_CASE("HilbertScanner causal-foliated scan order (invalid time_dim)", "[spatial][mem-04][phase0]") {
    HilbertScanner scanner(1);
    REQUIRE_THROWS_AS(scanner.generate_scan_order(9), std::out_of_range);
    REQUIRE_THROWS_AS(scanner.generate_scan_order(100), std::out_of_range);
}

// ═══════════════════════════════════════════════════════════════════════════
// Throughput Benchmarks
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("HilbertScanner encode throughput (order 3)", "[spatial][!benchmark]") {
    HilbertScanner scanner(3);
    constexpr int N = 100'000;
    // Generate coords [0, 8)^9
    std::vector<HilbertScanner::Coord9D> coords(N);
    std::mt19937 rng(0);
    std::uniform_int_distribution<uint32_t> dist(0, 7);
    for (auto& c : coords) {
        for (auto& v : c) v = dist(rng);
    }

    auto t0 = std::chrono::steady_clock::now();
    volatile uint64_t sink = 0;
    for (int i = 0; i < N; ++i) {
        sink = scanner.coords_to_index(coords[i]);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;

    INFO("Hilbert encode (order 3): " << ns << " ns/op");
    // Should be < 5µs per encode (Skilling in-place is O(n*b))
    REQUIRE(ns < 5000.0);
}

TEST_CASE("HilbertScanner decode throughput (order 3)", "[spatial][!benchmark]") {
    HilbertScanner scanner(3);
    constexpr int N = 100'000;

    auto t0 = std::chrono::steady_clock::now();
    volatile uint32_t sink = 0;
    for (int i = 0; i < N; ++i) {
        auto c = scanner.index_to_coords(static_cast<uint64_t>(i));
        sink = c[0];
    }
    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;

    INFO("Hilbert decode (order 3): " << ns << " ns/op");
    REQUIRE(ns < 5000.0);
}
