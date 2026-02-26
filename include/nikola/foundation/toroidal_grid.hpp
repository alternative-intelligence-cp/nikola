/**
 * @file toroidal_grid.hpp
 * @brief Sparse 9D toroidal grid with Structure-of-Arrays memory layout.
 *
 * The TorusGrid manages the discrete sample points of the UFIE manifold.
 * Nodes are stored in SoA (Structure of Arrays) format for SIMD-friendliness.
 * Addressing uses 128-bit Morton codes (9 dims × 14 bits = 126 bits total,
 * fits in __uint128_t) as specified in the implementation guide.
 *
 * Toroidal boundary conditions are enforced via modular coordinate arithmetic.
 * Neighbor lookups return VACUUM_NODE (size_t max) for unallocated cells,
 * which the propagator handles with a PML ghost value.
 *
 * Reference: nikola engineering report, Phase 1 (Gap 2.1, IMP-03)
 */
#pragma once

#include <nikola/foundation/complex_field.hpp>
#include <array>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <limits>
#if __cplusplus >= 202002L
#  include <span>
#endif
#include <cstring>
#include <random>

namespace nikola::foundation {

// ============================================================================
// Constants
// ============================================================================

/// Sentinel value for unallocated neighbour slots.
inline constexpr size_t VACUUM_NODE = std::numeric_limits<size_t>::max();

/// Number of spatial dimensions.
inline constexpr int TORUS_DIMS = 9;

/// Bits per dimension in a 128-bit Morton code (9 × 14 = 126 ≤ 128).
inline constexpr int MORTON_BITS_PER_DIM = 14;

/// Maximum addressable resolution per dimension (2^14 = 16384).
inline constexpr int MORTON_MAX_RESOLUTION = 1 << MORTON_BITS_PER_DIM;

// ============================================================================
// Grid configuration
// ============================================================================

/**
 * @brief Per-dimension resolution and spacing for the 9D manifold.
 *
 * Default resolutions from the engineering report (anisotropic):
 *   dims 0-2 (x,y,z spatial)       : 64
 *   dim  3   (t time)               : 128
 *   dims 4-5 (r resonance, s state) : 16
 *   dims 6-8 (u,v,w quantum)        : 32
 */
struct GridConfig {
    static constexpr int DIMS = TORUS_DIMS;

    std::array<int, DIMS> resolution;  ///< N_d for each dimension
    std::array<float, DIMS> spacing;   ///< h_d (physical grid spacing)

    /// Default anisotropic resolution from spec.
    static GridConfig anisotropic_default() {
        GridConfig cfg;
        cfg.resolution = {64, 64, 64, 128, 16, 16, 32, 32, 32};
        cfg.spacing.fill(1.0f);
        return cfg;
    }

    /// Uniform cubic grid for testing (e.g., 3^9 = 19683 nodes).
    static GridConfig uniform(int n_per_dim) {
        GridConfig cfg;
        cfg.resolution.fill(n_per_dim);
        cfg.spacing.fill(1.0f);
        return cfg;
    }

    /// Total number of nodes if the grid were dense.
    size_t total_nodes() const noexcept {
        size_t total = 1;
        for (int d = 0; d < DIMS; ++d) total *= resolution[d];
        return total;
    }
};

// ============================================================================
// 128-bit Morton code encoding / decoding
// ============================================================================

using Morton128 = __uint128_t;

/**
 * @brief Encode 9D integer coordinates into a 128-bit Morton code.
 *
 * Each coordinate is clamped to [0, MORTON_MAX_RESOLUTION).  Bits are
 * interleaved: bit k of dimension d occupies position (d + k*9) of the code.
 *
 * @param coords  Integer coordinates in each dimension.
 * @return        128-bit Morton key.
 */
[[nodiscard]]
inline Morton128 encode_morton(const std::array<int, TORUS_DIMS>& coords) noexcept {
    Morton128 code = 0;
    for (int bit = 0; bit < MORTON_BITS_PER_DIM; ++bit) {
        for (int dim = 0; dim < TORUS_DIMS; ++dim) {
            const Morton128 b = static_cast<Morton128>((coords[dim] >> bit) & 1);
            code |= b << (bit * TORUS_DIMS + dim);
        }
    }
    return code;
}

/**
 * @brief Decode a 128-bit Morton code back to 9D integer coordinates.
 */
[[nodiscard]]
inline std::array<int, TORUS_DIMS> decode_morton(Morton128 code) noexcept {
    std::array<int, TORUS_DIMS> coords{};
    for (int bit = 0; bit < MORTON_BITS_PER_DIM; ++bit) {
        for (int dim = 0; dim < TORUS_DIMS; ++dim) {
            const int b = static_cast<int>((code >> (bit * TORUS_DIMS + dim)) & 1);
            coords[dim] |= b << bit;
        }
    }
    return coords;
}

// Hash functor for Morton128 keys in unordered_map.
struct Morton128Hash {
    size_t operator()(Morton128 key) const noexcept {
        // Fold 128-bit key to 64-bit via XOR of high/low halves.
        const uint64_t lo = static_cast<uint64_t>(key);
        const uint64_t hi = static_cast<uint64_t>(key >> 64);
        // FNV-inspired mixing
        return lo ^ (hi * 0x9e3779b97f4a7c15ULL);
    }
};

// ============================================================================
// Per-node state  (struct for construction helpers; physics kernel uses SoA)
// ============================================================================

/**
 * @brief Complete wave state at a single grid node.
 *
 * Used when adding nodes or reading/writing individual states.
 * The propagator operates on the flat SoA arrays directly.
 */
struct TorusNode {
    Complex psi{0.f, 0.f};      ///< Wavefunction Ψ
    Complex vel{0.f, 0.f};      ///< Velocity ∂Ψ/∂t
    float   resonance{0.5f};    ///< r field  (memory plasticity, 0=erase 1=freeze)
    float   state_field{0.f};   ///< s field  (refractive index modifier)
};

// ============================================================================
// Neighbour direction descriptor
// ============================================================================

/**
 * @brief Identifies one of the 18 axis-aligned neighbours in 9D.
 *
 * For dimension d, neighbour index 2d   = +e_d  (positive step)
 *                                2d+1  = -e_d  (negative step)
 */
struct NeighborDir {
    int dim;  ///< Which dimension (0-8)
    int dir;  ///< +1 or -1
};

/// All 18 neighbour directions (2 per dimension).
inline std::array<NeighborDir, 18> all_neighbor_dirs() noexcept {
    std::array<NeighborDir, 18> dirs;
    for (int d = 0; d < 9; ++d) {
        dirs[2*d]     = {d, +1};
        dirs[2*d + 1] = {d, -1};
    }
    return dirs;
}

// ============================================================================
// TorusGrid  — sparse 9D grid with SoA physics data
// ============================================================================

/**
 * @brief Sparse 9D toroidal grid.
 *
 * Nodes are individually allocated (sparse addressing).  A Morton-keyed hash
 * map provides O(1) average-case coordinate-to-index lookup.  The physics
 * data (ψ, v, r, s) is stored in flat SoA vectors for SIMD efficiency.
 *
 * Toroidal boundary conditions wrap coordinates modulo N_d.
 * Unallocated neighbours return VACUUM_NODE.
 */
class TorusGrid {
public:
    // ------------------------------------------------------------------ ctor

    explicit TorusGrid(GridConfig config = GridConfig::uniform(3))
        : config_(std::move(config))
    {}

    // ------------------------------------------------------------------ node management

    /**
     * @brief Allocate a node at the given (toroidal-wrapped) coordinates.
     *
     * If a node already exists at these coordinates, returns its index.
     * Otherwise creates a new node with the provided state.
     *
     * @param coords  Integer coordinates (will be wrapped mod N_d).
     * @param state   Initial physical state.
     * @return        Node index into the SoA arrays.
     */
    size_t add_node(const std::array<int, TORUS_DIMS>& coords,
                    const TorusNode& state = {})
    {
        auto wcoords = wrap_coords(coords);
        const Morton128 key = encode_morton(wcoords);

        auto it = coord_to_index_.find(key);
        if (it != coord_to_index_.end()) {
            return it->second;  // Already exists
        }

        const size_t idx = soa_psi_real_.size();
        coord_to_index_.emplace(key, idx);
        index_to_coords_.push_back(wcoords);

        soa_psi_real_  .push_back(state.psi.real());
        soa_psi_imag_  .push_back(state.psi.imag());
        soa_vel_real_  .push_back(state.vel.real());
        soa_vel_imag_  .push_back(state.vel.imag());
        soa_resonance_ .push_back(state.resonance);
        soa_state_     .push_back(state.state_field);

        adj_valid_ = false;   // Invalidate cached adjacency
        return idx;
    }

    /**
     * @brief Find the index of a node at (unwrapped) coordinates.
     * @return node index, or VACUUM_NODE if not allocated.
     */
    [[nodiscard]]
    size_t find_node(const std::array<int, TORUS_DIMS>& coords) const noexcept {
        auto wcoords = wrap_coords(coords);
        const Morton128 key = encode_morton(wcoords);
        auto it = coord_to_index_.find(key);
        return (it != coord_to_index_.end()) ? it->second : VACUUM_NODE;
    }

    /**
     * @brief Get all 18 axis-aligned neighbour indices for node i.
     *
     * Array layout: index 2d = neighbour in +e_d,  2d+1 = neighbour in -e_d.
     * Unallocated neighbours are VACUUM_NODE.
     *
     * @param node_idx  Source node index.
     */
    [[nodiscard]]
    std::array<size_t, 18> get_neighbors(size_t node_idx) const {
        assert(node_idx < num_active_nodes());
        const auto& coords = index_to_coords_[node_idx];
        std::array<size_t, 18> result;
        for (int d = 0; d < 9; ++d) {
            for (int s : {+1, -1}) {
                auto nc = coords;
                nc[d] = wrap_coord(nc[d] + s, d);
                result[2*d + (s < 0 ? 1 : 0)] = find_node(nc);
            }
        }
        return result;
    }

    // ------------------------------------------------------------------ dense slab allocation

    /**
     * @brief Populate a dense hypercubic sub-grid of side length n_per_dim.
     *
     * Allocates n_per_dim^9 nodes centred at the origin of the grid.
     * States are set to default (zero psi, thermal vel sampled if sigma > 0).
     *
     * @param n_per_dim  Number of nodes per dimension.
     * @param rng        RNG for thermal init (pass nullptr to skip).
     * @param sigma      Thermal noise standard deviation.
     */
    void fill_dense_cube(int n_per_dim, std::mt19937* rng = nullptr, float sigma = 0.f) {
        const int half = n_per_dim / 2;
        std::array<int, TORUS_DIMS> c{};
        fill_recursive(c, 0, n_per_dim, -half, rng, sigma);
    }

    // ------------------------------------------------------------------ node access

    /// Number of currently allocated nodes.
    [[nodiscard]] size_t num_active_nodes() const noexcept {
        return soa_psi_real_.size();
    }

    /// Read the complete state of a node.
    [[nodiscard]]
    TorusNode get_node(size_t idx) const {
        assert(idx < num_active_nodes());
        return {
            {soa_psi_real_[idx], soa_psi_imag_[idx]},
            {soa_vel_real_[idx], soa_vel_imag_[idx]},
            soa_resonance_[idx],
            soa_state_[idx]
        };
    }

    /// Write the complete state of a node.
    void set_node(size_t idx, const TorusNode& n) {
        assert(idx < num_active_nodes());
        soa_psi_real_[idx]  = n.psi.real();
        soa_psi_imag_[idx]  = n.psi.imag();
        soa_vel_real_[idx]  = n.vel.real();
        soa_vel_imag_[idx]  = n.vel.imag();
        soa_resonance_[idx] = n.resonance;
        soa_state_[idx]     = n.state_field;
    }

    /// Coordinates of node at index.
    [[nodiscard]]
    const std::array<int, TORUS_DIMS>& coords_of(size_t idx) const {
        assert(idx < num_active_nodes());
        return index_to_coords_[idx];
    }

    // ------------------------------------------------------------------ precomputed adjacency  (fast physics loop)

    /**
     * @brief Precompute all 18 neighbours for every allocated node.
     *
     * Builds a flat array of size N×18 containing neighbour indices (or
     * VACUUM_NODE).  Must be called:
     *   - After all nodes have been allocated (add_node / fill_dense_cube).
     *   - Again if new nodes are added after the fact.
     *
     * Once built, use get_neighbors_fast(i) to avoid hash-map lookups in
     * the hot physics loop.
     */
    void precompute_adjacency() {
        const size_t N = soa_psi_real_.size();
        adj_.resize(N * 18);
        for (size_t i = 0; i < N; ++i) {
            auto nbrs = get_neighbors(i);   // uses hash-map lookup
            for (int n = 0; n < 18; ++n) {
                adj_[i * 18 + n] = nbrs[n];
            }
        }
        adj_valid_ = true;
    }

    /// Whether precomputed adjacency is available.
    bool adjacency_valid() const noexcept { return adj_valid_; }

    /**
     * @brief Fast neighbour access (requires adjacency precomputed).
     * @return Pointer to 18-element array of neighbour indices.
     */
    [[nodiscard]]
    const size_t* get_neighbors_fast(size_t i) const noexcept {
        assert(adj_valid_ && i < adj_.size() / 18);
        return &adj_[i * 18];
    }

    /**
     * @brief Total number of elements in the precomputed adjacency table.
     *
     * Equal to num_active_nodes() * 18.  Valid only after precompute_adjacency().
     * Compatible with C++17 and nvcc.
     */
    [[nodiscard]]
    size_t adjacency_table_size() const noexcept { return adj_.size(); }

    /**
     * @brief Raw pointer to the precomputed adjacency table data.
     *
     * Layout: adj[i * 18 + n] is the n-th neighbour of node i.
     * VACUUM_NODE entries use std::numeric_limits<size_t>::max().
     * Valid only after precompute_adjacency().  Compatible with C++17 and nvcc.
     */
    [[nodiscard]]
    const size_t* adjacency_table() const noexcept { return adj_.data(); }

    // ------------------------------------------------------------------ SoA data access (for propagator)

    float*       psi_real()       noexcept { return soa_psi_real_.data(); }
    float*       psi_imag()       noexcept { return soa_psi_imag_.data(); }
    float*       vel_real()       noexcept { return soa_vel_real_.data(); }
    float*       vel_imag()       noexcept { return soa_vel_imag_.data(); }
    float*       resonance()      noexcept { return soa_resonance_.data(); }
    float*       state_field()    noexcept { return soa_state_.data(); }

    const float* psi_real()  const noexcept { return soa_psi_real_.data(); }
    const float* psi_imag()  const noexcept { return soa_psi_imag_.data(); }
    const float* vel_real()  const noexcept { return soa_vel_real_.data(); }
    const float* vel_imag()  const noexcept { return soa_vel_imag_.data(); }
    const float* resonance() const noexcept { return soa_resonance_.data(); }
    const float* state_field()const noexcept { return soa_state_.data(); }

    /// Const grid configuration.
    const GridConfig& config() const noexcept { return config_; }

    /**
     * @brief Nodes-per-dimension for a uniform grid (resolution[0]).
     *
     * Assumes all 9 dimensions have the same resolution.  Valid after any
     * call to seed_manifold(n, ...) or GridConfig::uniform(n).
     */
    [[nodiscard]] int grid_n() const noexcept { return config_.resolution[0]; }

    /**
     * @brief Add a small complex perturbation to the ψ-field at node @p idx.
     *
     * Used by HolographicInjector to apply chord-wave energy to the grid.
     * The perturbation is NOT clamped here — callers must ensure it is safe.
     *
     * @param idx  Node index (must be < num_active_nodes()).
     * @param dr   Real increment to add to psi_real[idx].
     * @param di   Imaginary increment to add to psi_imag[idx].
     */
    void perturb_wavefunction(size_t idx, float dr, float di) noexcept {
        const size_t N = num_active_nodes();
        if (idx >= N) return;  // bounds-safe: silently ignore out-of-range
        soa_psi_real_[idx] += dr;
        soa_psi_imag_[idx] += di;
    }

    // ------------------------------------------------------------------ coordinate utilities

    /**
     * @brief Wrap a single coordinate component into [0, N_d).
     */
    [[nodiscard]]
    int wrap_coord(int coord, int dim) const noexcept {
        const int N = config_.resolution[dim];
        coord %= N;
        if (coord < 0) coord += N;
        return coord;
    }

    /**
     * @brief Wrap all 9 components of a coordinate array.
     */
    [[nodiscard]]
    std::array<int, TORUS_DIMS> wrap_coords(
            const std::array<int, TORUS_DIMS>& c) const noexcept
    {
        std::array<int, TORUS_DIMS> out;
        for (int d = 0; d < TORUS_DIMS; ++d)
            out[d] = wrap_coord(c[d], d);
        return out;
    }

    // ------------------------------------------------------------------ grid spacing

    /**
     * @brief Physical grid spacing in dimension d.
     */
    [[nodiscard]]
    float spacing(int dim) const noexcept {
        return config_.spacing[dim];
    }

private:
    // ------------------------------------------------------------------ private helpers

    /// Recursive helper for fill_dense_cube.
    void fill_recursive(std::array<int,TORUS_DIMS>& c, int depth,
                        int n_per_dim, int offset,
                        std::mt19937* rng, float sigma)
    {
        if (depth == TORUS_DIMS) {
            TorusNode node{};
            node.resonance   = 0.5f;
            node.state_field = 0.f;
            if (rng && sigma > 0.f) {
                node.vel = sample_thermal(sigma, *rng);
            }
            add_node(c, node);
            return;
        }
        for (int i = 0; i < n_per_dim; ++i) {
            c[depth] = i + offset;
            fill_recursive(c, depth + 1, n_per_dim, offset, rng, sigma);
        }
    }

    // ------------------------------------------------------------------ data

    GridConfig config_;

    // SoA physics arrays (indexed by node index)
    std::vector<float> soa_psi_real_;
    std::vector<float> soa_psi_imag_;
    std::vector<float> soa_vel_real_;
    std::vector<float> soa_vel_imag_;
    std::vector<float> soa_resonance_;
    std::vector<float> soa_state_;

    // Sparse addressing
    std::unordered_map<Morton128, size_t, Morton128Hash> coord_to_index_;
    std::vector<std::array<int, TORUS_DIMS>> index_to_coords_;

    // Precomputed adjacency (built by precompute_adjacency())
    std::vector<size_t> adj_;
    bool adj_valid_{false};
};

} // namespace nikola::foundation
