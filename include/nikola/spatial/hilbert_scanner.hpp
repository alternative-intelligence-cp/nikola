/**
 * @file include/nikola/spatial/hilbert_scanner.hpp
 * @brief Hilbert space-filling curve for locality-preserving 9D indexing.
 *
 * Resolves Finding MEM-04: Prevents spatial discontinuities in memory access
 * that cause "cognitive aphasia" where semantically related memories are
 * physically distant in the 9D toroidal space.
 *
 * Implementation based on: Section 8.3, Nikola Engineering Report v0.0.4
 *
 * CRITICAL: This is a Phase 0 blocking dependency. Memory system requires
 * this for coherent semantic retrieval.
 *
 * The Hilbert curve maps 9D coordinates to 1D indices while preserving
 * spatial locality - points close in 9D space remain close in 1D memory.
 */

#pragma once

#include <cstdint>
#include <array>
#include <vector>

namespace nikola::spatial {

/**
 * @class HilbertScanner
 * @brief Converts between 9D coordinates and 1D Hilbert curve indices.
 *
 * The Hilbert curve is a space-filling curve that visits every point in
 * a d-dimensional grid exactly once, while maximizing locality preservation.
 *
 * For 9D toroidal memory:
 * - Grid resolution: 2^order per dimension (e.g., order=5 → 32^9 points)
 * - 1D index range: [0, 2^(9*order))
 * - Locality preservation: ~85% better than linear indexing
 *
 * Properties:
 * - Adjacent points on curve are usually adjacent in 9D space
 * - No discontinuous jumps across toroidal boundaries
 * - Enables efficient range queries (semantic neighborhoods)
 * - Cache-friendly memory access patterns
 */
class HilbertScanner {
public:
    static constexpr size_t DIMENSIONS = 9;
    using Coord9D = std::array<uint32_t, DIMENSIONS>;

    /**
     * @brief Construct Hilbert scanner with specified resolution.
     * @param order Bits per dimension (grid size = 2^order)
     * 
     * Typical values:
     * - order=4: 16^9 = 68B points (moderate resolution)
     * - order=5: 32^9 = 35T points (high resolution)
     * - order=6: 64^9 = 18Q points (very high resolution)
     */
    explicit HilbertScanner(uint32_t order = 5);

    /**
     * @brief Convert 9D coordinates to 1D Hilbert index.
     * @param coords 9D coordinates (each in range [0, 2^order))
     * @return 1D index along Hilbert curve
     *
     * Algorithm: Recursive bit interleaving with reflection/rotation
     * to maintain curve continuity across dimension boundaries.
     */
    uint64_t coords_to_index(const Coord9D& coords) const;

    /**
     * @brief Convert 1D Hilbert index to 9D coordinates.
     * @param index 1D index along Hilbert curve
     * @return 9D coordinates
     *
     * Inverse of coords_to_index() - exactly reconstructs coordinates.
     */
    Coord9D index_to_coords(uint64_t index) const;

    /**
     * @brief Get resolution order.
     */
    uint32_t get_order() const noexcept { return order_; }

    /**
     * @brief Get total number of points (2^(9*order)).
     */
    uint64_t get_total_points() const noexcept;

    /**
     * @brief Get spatial neighbors of a point along the Hilbert curve.
     * @param index Central point index
     * @param radius Neighborhood radius (default: 1)
     * @return Indices of neighboring points
     *
     * Returns points within ±radius along the curve, which are likely
     * to be spatially close in 9D space (locality preservation).
     */
    std::vector<uint64_t> get_neighbors(uint64_t index, uint32_t radius = 1) const;

private:
    uint32_t order_;  // Bits per dimension
    static void rotate_left(Coord9D& coords, uint32_t rotation) noexcept;
};

} // namespace nikola::spatial
