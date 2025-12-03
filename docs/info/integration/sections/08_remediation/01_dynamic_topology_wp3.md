# WORK PACKAGE 3: DYNAMIC TOPOLOGY

## WP3.1 Overview

**Purpose:** Remediate critical defects in toroidal topology and dynamic grid expansion.

**Status:** All CRITICAL defects FIXED, specifications complete

## WP3.2 Defect: GEO-TOPO-01 - Missing Toroidal Wrapping Logic

**Defect ID:** GEO-TOPO-01
**Severity:** CRITICAL
**Status:** ✓ FIXED

### Impact

Grid boundaries acted as walls instead of wrapping; wave propagation failed at edges, violating toroidal topology axiom.

### Root Cause

`Coord9D` lacked modular arithmetic for periodic boundaries.

### Resolution

Added `wrap()` method implementing modular arithmetic for all 9 dimensions.

**Implementation:**

```cpp
// File: include/nikola/types/coord9d.hpp

struct Coord9D {
    std::array<int32_t, 9> coords;

    // Wrap coordinates to enforce toroidal topology
    void wrap(const std::array<int32_t, 9>& dimensions) {
        for (size_t i = 0; i < 9; ++i) {
            if (coords[i] < 0) {
                coords[i] = (coords[i] % dimensions[i] + dimensions[i]) % dimensions[i];
            } else {
                coords[i] = coords[i] % dimensions[i];
            }
        }
    }

    // Calculate geodesic distance on torus
    double distance_to(const Coord9D& other, const std::array<int32_t, 9>& dimensions) const {
        double sum = 0.0;
        for (size_t i = 0; i < 9; ++i) {
            int32_t diff = std::abs(coords[i] - other.coords[i]);
            int32_t wrapped_diff = std::min(diff, dimensions[i] - diff);
            sum += wrapped_diff * wrapped_diff;
        }
        return std::sqrt(sum);
    }
};
```

### Verification

- ✓ Unit tests confirm correct geodesic distance calculation
- ✓ Wave propagation no longer fails at grid boundaries
- ✓ Topological invariants maintained

### Location

Section 4.2.1 - [include/nikola/types/coord9d.hpp](include/nikola/types/coord9d.hpp) (Line 67-75)

## WP3.3 Defect: MEM-INIT-01 - Ambiguous TorusNode Flag Initialization

**Defect ID:** MEM-INIT-01
**Severity:** HIGH
**Status:** ✓ FIXED

### Impact

"Ghost charges" from recycled memory contaminated new nodes, causing spurious resonances and non-deterministic physics behavior.

### Root Cause

18-byte padding array left uninitialized in `TorusNode` constructor.

### Resolution

Explicit `memset` of padding to zero in constructor.

**Implementation:**

```cpp
// File: include/nikola/types/torus_node.hpp

struct TorusNode {
    std::complex<double> wavefunction;
    std::array<float, 45> metric_tensor;
    float resonance_r;
    float state_s;
    uint8_t padding[18];  // For alignment

    // Constructor with zero initialization
    TorusNode()
        : wavefunction(0.0, 0.0),
          resonance_r(0.5f),
          state_s(1.0f) {

        // Initialize metric to identity
        std::fill(metric_tensor.begin(), metric_tensor.end(), 0.0f);
        for (int i = 0; i < 9; ++i) {
            int idx = i * 9 - (i * (i + 1)) / 2 + i;  // Diagonal index
            metric_tensor[idx] = 1.0f;
        }

        // CRITICAL: Zero the padding to prevent ghost charges
        std::memset(padding, 0, sizeof(padding));
    }
};
```

### Verification

- ✓ Valgrind memcheck reports zero uninitialized bytes
- ✓ Physics behavior is now deterministic
- ✓ No spurious resonances from recycled memory

### Location

Section 4.3.1 - [include/nikola/types/torus_node.hpp](include/nikola/types/torus_node.hpp) (Line 45-62)

## WP3.4 Enhancement: Sparse Hyper-Voxel Octree (SHVO)

**Status:** IMPLEMENTED

### Purpose

Support the requirement "grow the torus as needed" efficiently without static multi-dimensional arrays.

### Architecture

The 9D space is virtualized. Only "active" regions (voxels) where the wavefunction energy $|\Psi|^2 > \epsilon$ consume memory.

**Key Features:**

1. **Coordinate Hashing:** Z-order curve (Morton code) maps 9D coordinates to single 64-bit integer
2. **Expansion (Neurogenesis):** Probes adjacent coordinates when saturation reached
3. **Memory Pool:** Pre-allocated slab prevents heap fragmentation

**Implementation:**

```cpp
// File: include/nikola/physics/shvo_grid.hpp

namespace nikola::physics {

class SparseHyperVoxelGrid {
private:
    // Spatial Hash Map: 64-bit Morton Code -> Node Pointer
    std::unordered_map<uint64_t, TorusNode*> active_voxels;

    // Memory Pool for fast allocation/deallocation
    std::vector<TorusNode> node_pool;
    std::vector<size_t> free_indices;

    // Saturation threshold for neurogenesis
    const float NEUROGENESIS_THRESHOLD = 4.0f;

public:
    SparseHyperVoxelGrid(size_t initial_capacity);

    // Convert 9D coords to Morton code
    uint64_t hash_coordinates(const Coord9D& pos) const;

    // Access or create node (Neurogenesis trigger)
    TorusNode* get_or_create(const Coord9D& pos);

    // Check saturation and trigger local expansion
    void check_neurogenesis(const Coord9D& center_pos);

    // Prune low-energy nodes (Neuro-necrosis)
    void prune_vacuum_nodes(float energy_threshold);
};

} // namespace nikola::physics
```

### Benefits

- **$O(1)$ spatial neurogenesis** without full tensor reallocation
- **Memory efficiency:** Only active regions consume RAM
- **Dynamic growth:** Torus can expand without system pause

### Location

Section 3.6 - [include/nikola/physics/shvo_grid.hpp](include/nikola/physics/shvo_grid.hpp)

---

**Cross-References:**
- See Section 3 for Toroidal Geometry fundamentals
- See Section 4 for Wave Propagation on wrapped manifold
- See Section 19 for Persistence of dynamic topology
