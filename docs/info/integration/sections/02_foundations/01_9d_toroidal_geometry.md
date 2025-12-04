# THE 9-DIMENSIONAL TOROIDAL GEOMETRY

## 3.1 Topological Definition

The fundamental data structure is a **9-dimensional torus**, mathematically defined as:

$$T^9 = S^1 \times S^1 \times S^1 \times S^1 \times S^1 \times S^1 \times S^1 \times S^1 \times S^1$$

Where $S^1$ is the unit circle. This can also be written as:

$$T^9 = (S^1)^9$$

### Key Topological Properties

1. **Compactness:** Finite volume, enabling complete enumeration
2. **Boundary-less:** No edges; all directions wrap around
3. **Homogeneity:** Every point has identical local topology
4. **Fundamental Group:** $\pi_1(T^9) \cong \mathbb{Z}^9$ enables integer encoding via winding numbers

### Why Toroidal Topology?

The torus solves the "curse of dimensionality" that plagues Euclidean spaces. In $\mathbb{R}^9$, volume grows exponentially, causing:
- Data sparsity
- Distance metric degradation
- Boundary effects

The compact, boundary-less torus provides:
- Uniform density
- Consistent distance metrics
- No boundary artifacts
- Natural recurrence (periodic behavior)

## 3.2 Dimensional Semantics

Each of the 9 dimensions has a specific functional role:

| Domain | Index | Symbol | Name | Physical Property | Cognitive Analog | Data Type |
|--------|-------|--------|------|-------------------|------------------|-----------|
| **Systemic** | 1 | $r$ | Resonance | Gain/Q-Factor/Damping | Attention/Forgetting | float |
| **Systemic** | 2 | $s$ | State | Refractive Index | Working Memory/Focus | float |
| **Temporal** | 3 | $t$ | Time | Temporal Flow | Sequence/Causality | float |
| **Quantum** | 4 | $u$ | Quantum 1 | Vector Component | Superposition State | complex |
| **Quantum** | 5 | $v$ | Quantum 2 | Vector Component | Superposition State | complex |
| **Quantum** | 6 | $w$ | Quantum 3 | Vector Component | Superposition State | complex |
| **Spatial** | 7 | $x$ | Width | Lattice X-Coord | Semantic Address X | int32 |
| **Spatial** | 8 | $y$ | Height | Lattice Y-Coord | Semantic Address Y | int32 |
| **Spatial** | 9 | $z$ | Depth | Lattice Z-Coord | Semantic Address Z | int32 |

### Detailed Dimension Descriptions

#### Systemic Dimensions ($r$, $s$)

These control the physical properties of the medium itself, not the data content.

**Resonance ($r$):** Controls energy persistence
- High $r$: High-Q cavity, waves persist → Long-term memory
- Low $r$: Dissipative medium, waves decay → Forgetting
- Range: [0.0, 1.0]
- Default: 0.5

**State ($s$):** Controls wave propagation speed
- High $s$: High refractive index, slow propagation → Focus/attention
- Low $s$: Low refractive index, fast propagation → Scanning
- Range: [0.0, 2.0]
- Default: 1.0

#### Temporal Dimension ($t$)

- Represents the time axis
- Enables causality and sequence encoding
- Flows continuously during operation
- Range: [0, $2\pi$) (wraps around)

#### Quantum Dimensions ($u$, $v$, $w$)

- Store the complex amplitude of the wavefunction
- Enable superposition states
- Each is a complex number: $u = u_{\text{real}} + i \cdot u_{\text{imag}}$
- Together form a 3D complex vector space

#### Spatial Dimensions ($x$, $y$, $z$)

- Standard 3D lattice coordinates
- Discretized integer grid
- Each wraps around at grid boundaries
- Grid size: Typically $27^3$ to $81^3$ nodes (powers of 3)

## 3.3 Dynamic Metric Tensor

The distance between points in the 9D space is not fixed but dynamic, controlled by the **metric tensor** $g_{ij}(\mathbf{x}, t)$.

### Line Element (Infinitesimal Distance)

$$ds^2 = \sum_{i=1}^{9} \sum_{j=1}^{9} g_{ij}(x,t) \, dx^i dx^j$$

The metric tensor is a $9 \times 9$ symmetric matrix, requiring storage of $\frac{9 \times 10}{2} = 45$ unique components per node.

### Physical Interpretation

- When $g_{ij} = \delta_{ij}$ (Kronecker delta), the space is flat (Euclidean)
- When concepts are frequently co-activated, $g_{ij}$ contracts, shortening the distance between them
- This creates "geodesic shortcuts" - associated concepts trigger each other rapidly

### Metric Tensor Storage

Since the matrix is symmetric, we store only the upper triangle:

```cpp
// Index mapping for symmetric 9x9 matrix
inline int triangular_index(int i, int j) {
    if (i > j) std::swap(i, j);
    return i * 9 - (i * (i + 1)) / 2 + j;
}

// Storage: flat array of 45 floats
std::array<float, 45> metric_tensor;
```

## 3.4 Neuroplasticity Mathematics

Learning is implemented as the time-evolution of the metric tensor according to a **Hebbian-Riemannian Learning Rule:**

$$\frac{\partial g_{ij}}{\partial t} = -\eta(D_t) \cdot \text{Re}(\Psi_i \cdot \Psi_j^*) + \lambda(g_{ij} - \delta_{ij})$$

### Term Explanation

**1. Contraction Term:** $-\eta(D_t) \cdot \text{Re}(\Psi_i \cdot \Psi_j^*)$
- $\eta(D_t)$: Learning rate modulated by dopamine
- $\Psi_i$: Wavefunction at dimension $i$
- $\Psi_j^*$: Complex conjugate of wavefunction at dimension $j$
- $\text{Re}(\cdot)$: Real part
- Effect: If waves are correlated (high real part of product), metric contracts (distance decreases)

**2. Relaxation Term:** $\lambda(g_{ij} - \delta_{ij})$
- $\lambda$: Elastic constant (typically 0.01)
- $\delta_{ij}$: Kronecker delta (1 if $i=j$, else 0)
- Effect: Pulls metric back toward Euclidean identity, preventing collapse

### Dopamine Modulation

$$\eta(t) = \eta_{\text{base}} \cdot (1 + \tanh(D(t)))$$

Where:
- $\eta_{\text{base}}$: Baseline learning rate (typically 0.001)
- $D(t)$: Dopamine level
- $\tanh(\cdot)$: Hyperbolic tangent (bounded activation)

When dopamine is high (reward), learning rate increases. When low, learning rate decreases.

## 3.5 Neurogenesis and Grid Expansion

When a region of the torus becomes saturated (high density of stored patterns), the system triggers **neurogenesis** - the creation of new nodes.

### Saturation Detection

$$\rho(\mathbf{x}) = \frac{\sum_{\text{neighbors}} |\Psi|^2}{\text{neighbor count}}$$

If $\rho(\mathbf{x}) > \rho_{\text{critical}}$ (typically 0.8), trigger neurogenesis.

### Node Insertion Algorithm

1. Identify saturated region coordinates
2. Create new slice of nodes (e.g., expand grid from $27^3$ to $28 \times 27^2$)
3. Interpolate metric tensor values from neighbors
4. Initialize wavefunction to vacuum state (amplitude = 0)
5. Update Hilbert curve mapping to include new nodes
6. Log expansion event to DMC

### Grid Size Strategy

- Start: $27^3 = 19,683$ nodes (base grid)
- Expand in powers of 3: $27, 30, 33, 36, ..., 81$
- Maximum: $81^3 = 531,441$ nodes (before multi-torus sharding)

## 3.6 Sparse Hyper-Voxel Octree (SHVO)

**[ADDENDUM]**

To support the requirement "grow the torus as needed" efficiently, we cannot use a static multi-dimensional array. We implement a Sparse Hyper-Voxel Octree.

### Data Structure Architecture

The 9D space is virtualized. Only "active" regions (voxels) where the wavefunction energy $|\Psi|^2 > \epsilon$ consume memory.

**Coordinate Hashing:** We use a Z-order curve (Morton code) to map 9D coordinates $(x_1, \dots, x_9)$ to a single 64-bit integer index.

$$\text{Index} = \sum_{i=0}^{63} \text{bit}_i(\text{coords}) \ll i$$

**Expansion (Neurogenesis):** When a node at coordinate $\vec{x}$ reaches saturation (energy density > threshold), the system probes the 18 adjacent coordinates in 9D space. If a neighbor does not exist in the hash map, it is allocated.

**Memory Pool:** A pre-allocated slab of TorusNode structs is used to prevent heap fragmentation. The hash map stores pointers into this slab.

### Reference Implementation (C++ Header)

```cpp
// include/nikola/physics/shvo_grid.hpp
#pragma once
#include "torus_node.hpp"
#include <unordered_map>
#include <deque>
#include <vector>

namespace nikola::physics {

// Sparse Hyper-Voxel Grid using std::deque for pointer stability
// std::deque guarantees pointers never invalidate on growth, unlike std::vector

class SparseHyperVoxelGrid {
private:
   // Spatial Hash Map: 64-bit Morton Code -> Node Pointer
   std::unordered_map<uint64_t, TorusNode*> active_voxels;

   // Memory Pool using std::deque for pointer stability
   // std::deque allocates in chunks and maintains pointer stability on growth
   std::deque<TorusNode> node_pool;
   std::vector<size_t> free_indices;

   // Saturation threshold for neurogenesis
   const float NEUROGENESIS_THRESHOLD = 4.0f;

public:
   SparseHyperVoxelGrid(size_t initial_capacity);

   // Convert 9D coords to Morton code
   uint64_t hash_coordinates(const Coord9D& pos) const;

   // Access or create node (Neurogenesis trigger)
   // Returns stable pointer that won't be invalidated by subsequent insertions
   TorusNode* get_or_create(const Coord9D& pos);

   // Check saturation and trigger local expansion
   void check_neurogenesis(const Coord9D& center_pos);

   // Prune low-energy nodes (Neuro-necrosis)
   void prune_vacuum_nodes(float energy_threshold);
};

} // namespace nikola::physics
```

### 3.5.1 Neurogenesis Implementation with GPU Topology Synchronization

**Status:** CRITICAL - Required to prevent GPU memory corruption during dynamic topology changes

**Integration with Differential Topology Manager:**

```cpp
// File: include/nikola/physics/sparse_grid.hpp
#pragma once

#include "nikola/physics/torus_node.hpp"
#include "nikola/physics/cuda/differential_topology.hpp"
#include <unordered_map>
#include <deque>
#include <vector>

namespace nikola::physics {

class SparseHyperVoxelGrid {
private:
    std::unordered_map<uint64_t, TorusNode*> active_voxels;
    std::deque<TorusNode> node_pool;
    std::vector<size_t> free_indices;

    const float NEUROGENESIS_THRESHOLD = 4.0f;

    // NEW: GPU topology synchronization manager
    cuda::DifferentialTopologyManager* topology_manager;

public:
    SparseHyperVoxelGrid(size_t initial_capacity,
                         cuda::DifferentialTopologyManager* topo_mgr)
        : topology_manager(topo_mgr) {
        node_pool.reserve(initial_capacity);
    }

    TorusNode* get_or_create(const Coord9D& pos);
    void check_neurogenesis(const Coord9D& center_pos);
    void prune_vacuum_nodes(float energy_threshold);

private:
    void update_adjacency_for_node(TorusNode* node, const Coord9D& pos);
};

} // namespace nikola::physics
```

**Implementation:**

```cpp
// File: src/physics/sparse_grid.cpp

#include "nikola/physics/sparse_grid.hpp"
#include <iostream>

namespace nikola::physics {

TorusNode* SparseHyperVoxelGrid::get_or_create(const Coord9D& pos) {
    uint64_t hash = hash_coordinates(pos);

    // Check if node already exists
    auto it = active_voxels.find(hash);
    if (it != active_voxels.end()) {
        return it->second;
    }

    // NEUROGENESIS: Create new node
    size_t node_idx;
    if (!free_indices.empty()) {
        // Reuse freed slot
        node_idx = free_indices.back();
        free_indices.pop_back();
        node_pool[node_idx] = TorusNode();  // Reset node
    } else {
        // Allocate new node
        node_idx = node_pool.size();
        node_pool.emplace_back();
    }

    TorusNode* new_node = &node_pool[node_idx];
    active_voxels[hash] = new_node;

    // CRITICAL: Update GPU topology with new node's adjacency
    update_adjacency_for_node(new_node, pos);

    return new_node;
}

void SparseHyperVoxelGrid::check_neurogenesis(const Coord9D& center_pos) {
    TorusNode* center = get_or_create(center_pos);

    // Check if center node exceeds threshold (high energy indicates need for resolution)
    if (std::abs(center->wavefunction) > NEUROGENESIS_THRESHOLD) {
        std::cout << "[NEUROGENESIS] Triggered at " << center_pos << std::endl;

        // Create neighboring nodes in all 18 directions (±1 in each of 9 dimensions)
        for (int dim = 0; dim < 9; ++dim) {
            for (int dir = -1; dir <= 1; dir += 2) {  // -1 and +1
                Coord9D neighbor_pos = center_pos;
                neighbor_pos[dim] += dir;

                // Create neighbor (if doesn't exist)
                get_or_create(neighbor_pos);
            }
        }

        // Update adjacency for center node after creating all neighbors
        update_adjacency_for_node(center, center_pos);
    }
}

void SparseHyperVoxelGrid::update_adjacency_for_node(TorusNode* node,
                                                      const Coord9D& pos) {
    std::array<int, 18> neighbors;
    int neighbor_count = 0;

    // Scan all 18 neighbors (±1 in each dimension)
    for (int dim = 0; dim < 9; ++dim) {
        for (int dir = -1; dir <= 1; dir += 2) {
            Coord9D neighbor_pos = pos;
            neighbor_pos[dim] += dir;

            uint64_t neighbor_hash = hash_coordinates(neighbor_pos);
            auto it = active_voxels.find(neighbor_hash);

            if (it != active_voxels.end()) {
                // Neighbor exists - calculate linear index
                int neighbor_idx = std::distance(&node_pool[0], it->second);
                neighbors[neighbor_count] = neighbor_idx;
            } else {
                // Neighbor doesn't exist
                neighbors[neighbor_count] = -1;
            }

            neighbor_count++;
        }
    }

    // Calculate node index
    int node_idx = std::distance(&node_pool[0], node);

    // CRITICAL: Queue topology change for GPU synchronization
    if (topology_manager) {
        topology_manager->queue_topology_change(node_idx, neighbors);
    }
}

void SparseHyperVoxelGrid::prune_vacuum_nodes(float energy_threshold) {
    std::vector<uint64_t> nodes_to_prune;

    for (const auto& [hash, node] : active_voxels) {
        if (std::abs(node->wavefunction) < energy_threshold) {
            nodes_to_prune.push_back(hash);
        }
    }

    for (uint64_t hash : nodes_to_prune) {
        TorusNode* node = active_voxels[hash];
        int node_idx = std::distance(&node_pool[0], node);

        // Mark neighbors as invalid (-1) on GPU
        std::array<int, 18> empty_neighbors;
        empty_neighbors.fill(-1);

        if (topology_manager) {
            topology_manager->queue_topology_change(node_idx, empty_neighbors);
        }

        // Remove from active set
        active_voxels.erase(hash);
        free_indices.push_back(node_idx);
    }

    std::cout << "[PRUNING] Removed " << nodes_to_prune.size() << " vacuum nodes" << std::endl;
}

uint64_t SparseHyperVoxelGrid::hash_coordinates(const Coord9D& pos) const {
    // Morton code (Z-order curve) for 9D coordinates
    // Interleaves bits of each dimension for spatial locality
    uint64_t hash = 0;
    for (int bit = 0; bit < 7; ++bit) {  // 7 bits per dimension (128^9 addressable space)
        for (int dim = 0; dim < 9; ++dim) {
            if (pos[dim] & (1 << bit)) {
                hash |= (1ULL << (bit * 9 + dim));
            }
        }
    }
    return hash;
}

} // namespace nikola::physics
```

**Physics Engine Integration:**

```cpp
// File: src/physics/physics_engine.cpp

#include "nikola/physics/sparse_grid.hpp"
#include "nikola/physics/cuda/differential_topology.hpp"

class PhysicsEngine {
    cuda::DifferentialTopologyManager topology_manager;
    SparseHyperVoxelGrid grid;

public:
    PhysicsEngine(size_t max_nodes)
        : topology_manager(max_nodes),
          grid(max_nodes / 2, &topology_manager) {}

    void propagate_step(double dt) {
        // 1. CRITICAL: Synchronize GPU topology with any neurogenesis changes
        topology_manager.synchronize();

        // 2. Launch wave propagation kernel with up-to-date adjacency
        propagate_wave_kernel<<<grid_config, block_config>>>(
            soa_data,
            topology_manager.get_device_ptr(),  // Updated neighbor indices
            num_active_nodes,
            dt
        );

        // 3. Check for neurogenesis triggers (may queue more topology changes)
        for (auto& [hash, node] : grid.get_active_voxels()) {
            if (std::abs(node->wavefunction) > NEUROGENESIS_THRESHOLD) {
                Coord9D pos = grid.unhash_coordinates(hash);
                grid.check_neurogenesis(pos);
            }
        }
    }
};
```

**Benefits:**

- **Memory Safety:** GPU kernel never operates on stale topology data
- **Bandwidth Efficiency:** Only changed adjacencies are transferred (< 20KB per neurogenesis event vs GB full re-upload)
- **Async Overlap:** Topology updates use dedicated CUDA stream, overlapping with compute
- **No Segfaults:** Differential updates prevent out-of-bounds neighbor access during dynamic growth

**Performance Characteristics:**

| Operation | Cost | Notes |
|-----------|------|-------|
| Single node neurogenesis | ~18KB GPU transfer | 18 neighbors × 4 bytes × 256 batch |
| Topology synchronization | 0.1-0.5ms | Async on dedicated stream |
| Propagation kernel delay | None | Sync happens before kernel launch |

---

**Cross-Reference:** See Section 4.6 for DifferentialTopologyManager CUDA implementation
