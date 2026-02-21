/**
 * @file src/spatial/hilbert_scanner.cpp
 * @brief Production 9D Hilbert space-filling curve with variable precision.
 *
 * Implements John Skilling's algorithm adapted for variable precision (1-7 bits).
 * Based on "Programming the Hilbert Curve" by John Skilling (2004).
 */

#include "nikola/spatial/hilbert_scanner.hpp"
#include <stdexcept>

namespace nikola::spatial {

HilbertScanner::HilbertScanner(uint32_t order)
    : order_(order) {
    
    if (order == 0 || order > 7) {
        throw std::invalid_argument(
            "Hilbert order must be in range [1, 7]"
        );
    }
}

uint64_t HilbertScanner::get_total_points() const noexcept {
    return 1ULL << (DIMENSIONS * order_);
}

uint64_t HilbertScanner::coords_to_index(const Coord9D& coords) const {
    // Validate
    const uint32_t max_coord = (1U << order_) - 1;
    for (size_t i = 0; i < DIMENSIONS; ++i) {
        if (coords[i] > max_coord) {
            throw std::out_of_range("Coordinate exceeds grid resolution");
        }
    }
    
    // Copy for in-place transformation
    Coord9D X = coords;
    
    // Skilling's PositionToIndex algorithm
    // Phase 1: Reverse transforms (convert position to transposed Gray code)
    {
        uint32_t cur_bit = 1U << (order_ - 1);  // Start at MSB
        uint32_t low_bits;
        
        do {
            low_bits = cur_bit - 1;
            
            for (size_t n = 0; n < DIMENSIONS; ++n) {
                if (X[n] & cur_bit) {
                    X[0] ^= low_bits;  // Flip low bits
                } else {
                    uint32_t t = (X[n] ^ X[0]) & low_bits;
                    X[n] ^= t;  // Swap low bits
                    X[0] ^= t;
                }
            }
            
            cur_bit >>= 1;
        } while (low_bits > 1);
    }
    
    // Phase 2: Remove Gray code from transposed vector
    {
        // XOR chain
        for (size_t n = 1; n < DIMENSIONS; ++n) {
            X[n] ^= X[n-1];
        }
        
        // Accumulate correction factor
        uint32_t cur_bit = 1U << (order_ - 1);
        uint32_t t = 0;
        
        do {
            if (X[DIMENSIONS-1] & cur_bit) {
                t ^= (cur_bit - 1);  // KEY: cur_bit - 1, not cur_bit
            }
            cur_bit >>= 1;
        } while (cur_bit > 1);
        
        // Apply correction to ALL dimensions
        for (auto& v : X) {
            v ^= t;
        }
    }
    
    // Pack to 64-bit index: dimension-by-dimension (all bits of each dimension)
    uint64_t index = 0;
    for (size_t i = 0; i < DIMENSIONS; ++i) {
        for (int b = 0; b < static_cast<int>(order_); ++b) {
            index <<= 1;
            index |= (X[i] >> b) & 1;
        }
    }
    
    return index;
}

HilbertScanner::Coord9D HilbertScanner::index_to_coords(uint64_t index) const {
    if (index >= get_total_points()) {
        throw std::out_of_range("Index exceeds Hilbert curve range");
    }
    
    // Unpack index: dimension-by-dimension (reverse ofpacking)
    Coord9D X{};
    uint64_t temp = index;
    
    for (int i = DIMENSIONS - 1; i >= 0; --i) {
        for (int b = order_ - 1; b >= 0; --b) {
            X[i] |= (temp & 1) << b;
            temp >>= 1;
        }
    }
    
    // Skilling's IndexToPosition algorithm
    // Phase 1: Gray decode transposed vector
    {
        uint32_t tmp = X[DIMENSIONS-1] >> 1;
        
        // XOR chain
        for (size_t n = DIMENSIONS - 1; n > 0; --n) {
            X[n] ^= X[n-1];
        }
        
        X[0] ^= tmp;
    }
    
    // Phase 2: Forward transforms (convert Gray code to position)
    {
        uint32_t cur_bit = 2;  // Start at bit 1
        uint32_t low_bits;
        uint32_t max_bit = 1U << order_;  // Stop before overflow
        
        while (cur_bit < max_bit) {
            low_bits = cur_bit - 1;
            
            size_t n = DIMENSIONS;
            do {
                --n;
                if (X[n] & cur_bit) {
                    X[0] ^= low_bits;  // Flip low bits
                } else {
                    uint32_t t = (X[n] ^ X[0]) & low_bits;
                    X[n] ^= t;  // Swap low bits
                    X[0] ^= t;
                }
            } while (n > 0);
            
            cur_bit <<= 1;
        }
    }
    
    return X;
}

std::vector<uint64_t> HilbertScanner::get_neighbors(uint64_t index, uint32_t radius) const {
    std::vector<uint64_t> neighbors;
    neighbors.reserve(2 * radius);
    
    int64_t signed_index = static_cast<int64_t>(index);
    int64_t total = static_cast<int64_t>(get_total_points());
    
    for (uint32_t r = 1; r <= radius; ++r) {
        int64_t prev = signed_index - r;
        int64_t next = signed_index + r;
        
        if (prev >= 0) {
            neighbors.push_back(static_cast<uint64_t>(prev));
        }
        if (next < total) {
            neighbors.push_back(static_cast<uint64_t>(next));
        }
    }
    
    return neighbors;
}

void HilbertScanner::rotate_left(Coord9D&, uint32_t) noexcept {}

} // namespace nikola::spatial
