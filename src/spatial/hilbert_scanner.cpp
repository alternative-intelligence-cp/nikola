/**
 * @file src/spatial/hilbert_scanner.cpp  
 * @brief Exact Variable-Precision 9D Hilbert Curve Implementation.
 *
 * Implements John Skilling's algorithmic transformation for space-filling curves
 * to guarantee optimal spatial locality preservation for quantum simulation models.
 * Resolves MEM-04 coordinate permutation anomalies through strict interleaved transposition.
 *
 * Based on Gemini Deep Research: "Hilbert Curve Precision Implementation Specification"  
 * Reference: "Programming the Hilbert Curve" by John Skilling (2004)
 * License: MIT / Public Domain (Skilling adaptation)
 */

#include "nikola/spatial/hilbert_scanner.hpp"
#include <stdexcept>
#include <algorithm>

namespace nikola::spatial {

HilbertScanner::HilbertScanner(uint32_t order)
    : order_(order) {
    
    if (order > 7) {
        throw std::invalid_argument("Order exceeds 64-bit index capacity for 9D space.");
    }
    if (order == 0) {
        throw std::invalid_argument("Order must be at least 1.");
    }
}

uint64_t HilbertScanner::get_total_points() const noexcept {
    return 1ULL << (DIMENSIONS * order_);
}

uint64_t HilbertScanner::coords_to_index(const Coord9D& coords) const {
    // Validate coordinates
    const uint32_t max_coord = (1U << order_) - 1;
    for (size_t i = 0; i < DIMENSIONS; ++i) {
        if (coords[i] > max_coord) {
            throw std::out_of_range("Coordinate exceeds grid resolution");
        }
    }

    Coord9D X = coords;
    
    // Apply Skilling's axes_to_transpose transformation
    if (order_ > 0) {
        uint32_t M = 1U << (order_ - 1);  // MSB mask
        uint32_t P, Q, t;

        // Phase 1: Inverse Undo
        for (Q = M; Q > 1; Q >>= 1) {
            P = Q - 1;
            for (size_t i = 0; i < DIMENSIONS; ++i) {
                if (X[i] & Q) {
                    X[0] ^= P;  // Invert condition
                } else {
                    t = (X[0] ^ X[i]) & P;  // Exchange condition
                    X[0] ^= t;
                    X[i] ^= t;
                }
            }
        }

        // Phase 2: Gray Encode
        for (size_t i = 1; i < DIMENSIONS; ++i) {
            X[i] ^= X[i - 1];
        }

        t = 0;
        for (Q = M; Q > 1; Q >>= 1) {
            if (X[DIMENSIONS-1] & Q) {
                t ^= (Q - 1);
            }
        }

        for (size_t i = 0; i < DIMENSIONS; ++i) {
            X[i] ^= t;
        }
    }
    
    // Pack transposed coordinates into 64-bit index
    // Pack strictly from highest precision bit down, traversing spatial dimensions 0 to N-1
    uint64_t H = 0;
    for (int b = static_cast<int>(order_) - 1; b >= 0; --b) {
        for (size_t i = 0; i < DIMENSIONS; ++i) {
            uint64_t bit = (X[i] >> b) & 1U;
            H = (H << 1) | bit;
        }
    }
    
    return H;
}

HilbertScanner::Coord9D HilbertScanner::index_to_coords(uint64_t index) const {
    if (index >= get_total_points()) {
        throw std::out_of_range("Index exceeds Hilbert curve range");
    }
    
    // Unpack index to transposed coordinates
    // Extract strictly from the lowest scalar bit up, assigning to dimensions N-1 down to 0
    Coord9D X = {0};
    uint64_t H = index;

    for (int b = 0; b < static_cast<int>(order_); ++b) {
        for (int i = static_cast<int>(DIMENSIONS) - 1; i >= 0; --i) {
            X[i] |= static_cast<uint32_t>(H & 1ULL) << b;
            H >>= 1;
        }
    }
    
    // Apply Skilling's transpose_to_axes transformation
    if (order_ > 0) {
        // Phase 1: Gray Decode
        uint32_t t = X[DIMENSIONS-1] >> 1;
        
        // Critical Fix: i > 0 prevents out of bounds X[-1]
        for (size_t i = DIMENSIONS - 1; i > 0; --i) {
            X[i] ^= X[i - 1];
        }
        X[0] ^= t;

        // Phase 2: Undo Excess Work utilizing dynamically calculated boundaries
        uint32_t N_limit = 2U << (order_ - 1);
        uint32_t P, Q;

        for (Q = 2; Q != N_limit; Q <<= 1) {
            P = Q - 1;
            for (int i = static_cast<int>(DIMENSIONS) - 1; i >= 0; --i) {
                if (X[i] & Q) {
                    X[0] ^= P;  // Invert condition
                } else {
                    t = (X[0] ^ X[i]) & P;  // Exchange condition
                    X[0] ^= t;
                    X[i] ^= t;
                }
            }
        }
    }
    
    return X;
}

std::vector<uint64_t> HilbertScanner::get_neighbors(uint64_t center_index, uint32_t radius) const {
    std::vector<uint64_t> neighbors;
    neighbors.reserve(radius * 2);
    
    int64_t min_idx = std::max(static_cast<int64_t>(0), static_cast<int64_t>(center_index) - radius);
    int64_t max_idx = std::min(static_cast<int64_t>(get_total_points() - 1), static_cast<int64_t>(center_index) + radius);

    for (int64_t idx = min_idx; idx <= max_idx; ++idx) {
        if (static_cast<uint64_t>(idx) != center_index) {
            neighbors.push_back(static_cast<uint64_t>(idx));
        }
    }
    return neighbors;
}

void HilbertScanner::rotate_left(Coord9D&, uint32_t) noexcept {
    // Not needed for Skilling's algorithm
}

std::vector<HilbertScanner::Coord9D> HilbertScanner::generate_scan_order(size_t time_dim) const {
    if (time_dim >= DIMENSIONS) {
        throw std::out_of_range("time_dim must be < 9");
    }

    const uint32_t bins = 1U << order_;
    
    // Build an 8D Hilbert scanner for the spatial subspace (one order lower
    // would change resolution, so we use the same order and manually pack/unpack
    // the 8 non-time dimensions into an auxiliary Coord9D with dim[time_dim]=0).
    //
    // For each time slice t:
    //   1. Enumerate all 8D spatial points via Hilbert curve (locality within slice)
    //   2. Reconstruct full 9D coord by inserting t at time_dim position
    //
    // This gives us time-monotonic ordering with spatial locality per slice.

    // Total points in the 8D spatial subspace per time slice
    const uint64_t spatial_points = 1ULL << (8 * order_);
    // Total scan size
    const uint64_t total = static_cast<uint64_t>(bins) * spatial_points;
    
    std::vector<Coord9D> scan_order;
    scan_order.reserve(total);

    // We iterate the 8D subspace by iterating all Hilbert indices [0, 2^(9*order))
    // and grouping by the time coordinate. Instead, we directly iterate time slices
    // and within each slice, iterate spatial coords in a cache-friendly order.
    //
    // Since we don't have a separate 8D Hilbert implementation, we use the existing
    // 9D scanner: for each time slice t, we scan the full 9D curve and pick only
    // the points where coord[time_dim] == t. This preserves Hilbert locality within
    // each slice but is O(bins * total_9d_points) which is too expensive.
    //
    // Better approach: iterate all possible 8D coordinates in Hilbert order by
    // mapping them through a synthetic 9D index with dim[time_dim] fixed.
    // We enumerate all (bins^8) points per time slice by looping through 
    // 8D multi-index space in standard order, then Hilbert-rank each.
    // Then sort within each slice by Hilbert rank.
    
    // Most practical approach for correctness: iterate all points, sort by
    // (time_coord, hilbert_index). This is O(N log N) where N = bins^9.
    // For order ≤ 2 this is fine (262K points). For higher orders, this
    // function would need the 8D sub-scanner optimization.
    
    // Collect all points with their time coordinate and Hilbert index
    struct ScanEntry {
        uint32_t time_val;
        uint64_t hilbert_idx;
        Coord9D coords;
    };
    
    std::vector<ScanEntry> entries;
    entries.reserve(total);
    
    // Iterate all 9D Hilbert indices in curve order
    const uint64_t total_points = get_total_points();
    for (uint64_t h = 0; h < total_points; ++h) {
        Coord9D c = index_to_coords(h);
        entries.push_back({c[time_dim], h, c});
    }
    
    // Stable sort by time coordinate — preserves Hilbert order within each time slice
    std::stable_sort(entries.begin(), entries.end(),
        [](const ScanEntry& a, const ScanEntry& b) {
            return a.time_val < b.time_val;
        });
    
    // Extract coordinates in causal-foliated order
    for (const auto& e : entries) {
        scan_order.push_back(e.coords);
    }
    
    return scan_order;
}

} // namespace nikola::spatial
