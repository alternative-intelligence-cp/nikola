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

} // namespace nikola::spatial
