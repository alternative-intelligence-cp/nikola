/**
 * @file include/nikola/spatial/spatial_index.hpp
 * @brief O(log N) spatial lookup for the T⁹ toroidal manifold.
 *
 * Uses Morton (Z-order) keys to map 9D grid points to a sorted 1D key space.
 * Nearby points in 9D map to nearby Morton keys (with Z-order clustering),
 * enabling efficient neighbor queries via binary search.
 *
 * Two query modes:
 *   1. morton_neighbors() — finds the k nearest Morton-key neighbors of a
 *      point by scanning outward from its key's position in the sorted index.
 *      O(log N + k) per query.  Good for cache-friendly sequential access.
 *
 *   2. grid_neighbors() — enumerates the ±1 face-adjacent neighbors in each
 *      of the 9 dimensions (up to 18 neighbors), with toroidal wrapping,
 *      and looks each one up in the index via binary search.
 *      O(18 * log N) per query.  Exact spatial neighbors.
 *
 * The index is designed for static grids (built once, queried many times).
 * For the production 3^9 = 19,683 node grid, building is instantaneous and
 * each query takes microseconds.
 */
#pragma once

#include <nikola/spatial/morton_encoder.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace nikola::spatial {

/**
 * @brief A sorted-key spatial index over 9D Morton-encoded coordinates.
 *
 * Build with a set of Coord9D points; query for neighbors.
 */
class SpatialIndex {
public:
    /// Entry: Morton key + original coordinate.
    struct Entry {
        MortonKey key;
        Coord9D   coord;

        bool operator<(const Entry& o) const noexcept { return key < o.key; }
    };

    SpatialIndex() = default;

    /**
     * @brief Build the index from a set of coordinates.
     *
     * Encodes each coordinate to a Morton key, sorts by key.
     * O(N log N) construction.
     *
     * @param coords  Vector of 9D coordinates to index.
     */
    void build(const std::vector<Coord9D>& coords) {
        entries_.clear();
        entries_.reserve(coords.size());
        for (const auto& c : coords) {
            entries_.push_back({morton_encode(c), c});
        }
        std::sort(entries_.begin(), entries_.end());
    }

    /// Number of indexed points.
    [[nodiscard]] size_t size() const noexcept { return entries_.size(); }

    /// Whether the index is empty.
    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }

    /**
     * @brief Look up a coordinate by its Morton key.
     *
     * O(log N) via binary search.
     *
     * @param key  Morton key to find.
     * @return Pointer to the entry if found, nullptr otherwise.
     */
    [[nodiscard]] const Entry* find(MortonKey key) const noexcept {
        auto it = std::lower_bound(
            entries_.begin(), entries_.end(), key,
            [](const Entry& e, MortonKey k) { return e.key < k; });
        if (it != entries_.end() && it->key == key) return &(*it);
        return nullptr;
    }

    /**
     * @brief Find the k nearest Morton-key neighbors of a point.
     *
     * Locates the point's key in the sorted index, then returns the
     * k closest entries by Morton key distance (scanning left/right).
     * O(log N + k).
     *
     * @param coord  Query point.
     * @param k      Number of neighbors to return.
     * @return Vector of up to k neighboring entries (excluding the query
     *         point itself if it's in the index).
     */
    [[nodiscard]] std::vector<Entry>
    morton_neighbors(const Coord9D& coord, size_t k) const {
        if (entries_.empty()) return {};
        MortonKey qk = morton_encode(coord);

        // Find insertion point
        auto it = std::lower_bound(
            entries_.begin(), entries_.end(), qk,
            [](const Entry& e, MortonKey key) { return e.key < key; });
        auto pos = static_cast<size_t>(it - entries_.begin());

        std::vector<Entry> result;
        result.reserve(k);

        size_t left = (pos > 0) ? pos - 1 : 0;
        size_t right = pos;
        bool left_valid = (pos > 0);

        // Skip self if present
        if (right < entries_.size() && entries_[right].key == qk) {
            right++;
        }

        while (result.size() < k) {
            bool have_left = left_valid && left < entries_.size();
            bool have_right = right < entries_.size();
            if (!have_left && !have_right) break;

            if (have_left && have_right) {
                MortonKey dl = qk - entries_[left].key;
                MortonKey dr = entries_[right].key - qk;
                if (dl <= dr) {
                    result.push_back(entries_[left]);
                    if (left == 0) left_valid = false;
                    else left--;
                } else {
                    result.push_back(entries_[right]);
                    right++;
                }
            } else if (have_left) {
                result.push_back(entries_[left]);
                if (left == 0) left_valid = false;
                else left--;
            } else {
                result.push_back(entries_[right]);
                right++;
            }
        }
        return result;
    }

    /**
     * @brief Find exact grid neighbors (face-adjacent in each dimension).
     *
     * For each of the 9 dimensions, computes the ±1 neighbor with toroidal
     * wrapping, encodes to Morton key, and looks it up in the index.
     * Returns only neighbors that exist in the index.
     * O(18 * log N).
     *
     * @param coord       Query point.
     * @param grid_size   Number of nodes per dimension (for toroidal wrap).
     * @return Vector of face-adjacent entries found in the index.
     */
    [[nodiscard]] std::vector<Entry>
    grid_neighbors(const Coord9D& coord, uint32_t grid_size) const {
        std::vector<Entry> result;
        result.reserve(2 * MORTON_DIMS);

        for (int d = 0; d < MORTON_DIMS; ++d) {
            // +1 neighbor (with toroidal wrap)
            Coord9D plus = coord;
            plus[d] = (coord[d] + 1) % grid_size;
            if (auto* e = find(morton_encode(plus))) {
                result.push_back(*e);
            }

            // -1 neighbor (with toroidal wrap)
            Coord9D minus = coord;
            minus[d] = (coord[d] + grid_size - 1) % grid_size;
            if (auto* e = find(morton_encode(minus))) {
                result.push_back(*e);
            }
        }
        return result;
    }

    /// Access the sorted entries (for iteration, serialization, etc.).
    [[nodiscard]] const std::vector<Entry>& entries() const noexcept {
        return entries_;
    }

private:
    std::vector<Entry> entries_;
};

} // namespace nikola::spatial
