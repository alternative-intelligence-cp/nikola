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
#include <vector>

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

---

**Cross-Reference:** See Section 8.1 (Work Package 3) for GPU synchronization of dynamic topology
