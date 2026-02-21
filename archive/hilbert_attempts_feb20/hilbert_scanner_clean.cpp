/**
 * @file src/spatial/hilbert_scanner.cpp
 * @brief Production 9D Hilbert space-filling curve using spectral3d reference.
 *
 * Uses spectral3d's proven implementation with precision limiting.
 */

#include "nikola/spatial/hilbert_scanner.hpp"
#include "nikola/spatial/hilbert_reference.hpp"
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

uint32_t HilbertScanner::get_order() const noexcept {
    return order_;
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
    
    // Use spectral3d's reference implementation
    std::array<uint8_t, DIMENSIONS> pos{};
    for (size_t i = 0; i < DIMENSIONS; ++i) {
        pos[i] = static_cast<uint8_t>(coords[i]);
    }
    
    auto idx = hilbert::v1::PositionToIndex(pos);
    
    // Pack only our precision bits
    uint64_t result = 0;
    for (size_t i = 0; i < DIMENSIONS; ++i) {
        for (size_t b = 0; b < order_; ++b) {
            result <<=  1;
            result |= (idx[i] >> b) & 1;
        }
    }
    
    return result;
}

HilbertScanner::Coord9D HilbertScanner::index_to_coords(uint64_t index) const {
    if (index >= get_total_points()) {
        throw std::out_of_range("Index exceeds Hilbert curve range");
    }
    
    // Unpack to spectral3d format
    std::array<uint8_t, DIMENSIONS> idx{};
    uint64_t temp = index;
    
    for (int i = DIMENSIONS - 1; i >= 0; --i) {
        for (int b = order_ - 1; b >= 0; --b) {
            idx[i] |= (temp & 1) << b;
            temp >>= 1;
        }
    }
    
    // Use spectral3d's reference implementation
    auto pos = hilbert::v1::IndexToPosition(idx);
    
    Coord9D coords{};
    const uint32_t mask = (1U << order_) -  1;
    for (size_t i = 0; i < DIMENSIONS; ++i) {
        coords[i] = pos[i] & mask;
    }
    
    return coords;
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
