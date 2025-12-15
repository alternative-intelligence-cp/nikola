# Bug Sweep 002 - 9D Geometry Integration Notes

**Date:** December 12, 2025  
**Status:** Ready for Integration  
**Target Document:** `02_foundations/01_9d_toroidal_geometry.md`

## Executive Summary

Bug sweep 002 provides complete mathematical specifications that resolve 4 major gaps in the geometry implementation:

1. **Gap #1:** Dimension mapping (semantic → continuous → discrete) ✅
2. **Gap #2:** Coordinate transformation algorithms ✅
3. **Gap #3:** Spatial matrices (metric tensor operations) ✅
4. **Gap #4:** Traversal logic (Causal-Foliated Hilbert Scan) ✅

## Critical New Content to Integrate

### 1. Coord9D Structure **[CRITICAL]**

**Location to integrate:** Section 3.2 or new subsection after dimensional definitions

**New content:**
- Complete `Coord9D` bitfield struct with variable bit-widths:
  - Systemic (r, s): 4 bits each (16 levels)
  - Temporal (t): 14 bits (16,384 timesteps)
  - Quantum (u, v, w): 8 bits each (256 levels)
  - Spatial (x, y, z): 14 bits each (16,384 grid points)
- Total: 88 bits (fits in uint128_t with 40 bits spare for metadata)
- Helper methods for normalized float conversion

**Why critical:** Currently no specification for how coordinates are discretized and stored. This is foundational for all memory addressing.

### 2. Toroidal Wrapping Logic **[CRITICAL]**

**Location to integrate:** Section 3.2 or 3.3 (boundary conditions)

**New content:**
```cpp
inline int wrap(int k, int N) {
   int r = k % N;
   return r < 0 ? r + N : r;
}
```

**Mathematical foundation:**
```
x^μ_new = (x^μ + δ) mod N_μ
```

**Why critical:** C++ modulo operator returns negative values for negative operands. Incorrect wrapping causes segfaults at boundaries instead of proper toroidal topology.

### 3. Projective Locality Mapper (SEM-01) **[HIGH PRIORITY]**

**Location to integrate:** New section 3.4 or after metric tensor section

**New content:**
- Complete algorithm for mapping embeddings (768D) → 9D torus
- Johnson-Lindenstrauss projection with Gaussian matrix P (9×768)
- Quantile normalization using error function (erf) for uniform distribution
- Full C++ implementation with SIMD-optimized dot product

**Mathematical foundation:**
```
y = P × v  (Project 768D → 9D)
y' = y / (σ√2)  (Normalize)
u = 0.5(1 + erf(y'))  (Map to uniform [0,1])
x = floor(u × N)  (Quantize to grid)
```

**Why critical:** Resolves SEM-01 audit gap. Standard hashing destroys semantic locality. This preserves it so related concepts cluster spatially and can interfere constructively.

### 4. Metric Tensor Operations **[CRITICAL]**

**Location to integrate:** Section 3.3 (expand existing metric tensor content)

**New content:**

#### Hebbian-Riemannian Update Rule:
```
∂g_ij/∂t = -η·Re(Ψ_i Ψ_j*) + λ(g_ij - δ_ij)
```
- Contraction term: pulls correlated dimensions closer
- Relaxation term: prevents singularities, implements forgetting

#### Storage Optimization:
- Symmetric matrix: store upper triangle only (45 floats vs 81)
- Indexing formula: `index(row,col) = 9·row - row·(row+1)/2 + col`

#### Lazy Cholesky Inversion:
- Cache inverse g^ij and determinant √|g|
- Dirty flag system to minimize recomputation
- Tikhonov regularization fallback: `g'_ij = g_ij + ε·δ_ij` when SPD fails

**Why critical:** Current spec mentions metric tensor but lacks concrete implementation details for deformation, storage, and inversion.

### 5. Morton Coding (128-bit) **[HIGH PRIORITY]**

**Location to integrate:** New section 3.5 "Spatial Hashing"

**New content:**
- Complete 128-bit Morton key specification
- Bit interleaving algorithm with BMI2 intrinsics
- Split encoding (low 64 bits + high 64 bits)
- Position formula: `pos = 9·b + d` for bit b of dimension d

**Structure:**
```cpp
struct MortonKey128 {
   uint64_t lo;  // Bits 0-6 of all 9 dimensions
   uint64_t hi;  // Bits 7-13 of all 9 dimensions
   
   bool operator==(const MortonKey128& other) const {
       return lo == other.lo && hi == other.hi;
   }
};
```

**Why critical:** Enables O(1) sparse storage in hash maps. Standard 64-bit Morton codes insufficient for 9D space with 14-bit resolution.

### 6. Causal-Foliated Hilbert Scanning **[CRITICAL - NEW ALGORITHM]**

**Location to integrate:** New section 3.6 "Traversal Algorithms"

**New content:**

**Problem:** Mamba-9D requires temporally-ordered linear sequence. Standard Morton/Hilbert curves violate causality.

**Solution:** Two-stage sorting:
```
n_a <_scan n_b ⟺ (t_a < t_b) ∨ (t_a = t_b ∧ H_8(s_a) < H_8(s_b))
```

**Algorithm:**
1. Primary sort: Time coordinate t
2. Secondary sort: 8D Hilbert index of (r, s, u, v, w, x, y, z)

**Complexity:** O(N_active log N_active)

**Why Hilbert over Morton:** Hilbert curve preserves spatial continuity. Morton codes have discontinuities at bit-carry boundaries (e.g., 011→100 causes spatial jump).

**Why critical:** This is entirely NEW algorithm not mentioned in existing docs. Mandatory for Mamba-9D integration. Resolves temporal causality in high-dimensional traversal.

### 7. Neurogenesis Algorithm **[HIGH PRIORITY]**

**Location to integrate:** Section 3.7 or within SHVO section

**New content:**

**Dynamic expansion protocol:**
1. Saturation check: `|Ψ|² > ε_genesis`
2. Neighbor probe: Calculate 128-bit Morton keys for 18 von Neumann neighbors
3. Existence check: Query sparse_map
4. Allocation with geometric continuity:
   - Metric interpolation (Log-Euclidean): `g_new = exp(log(g_parent))`
   - Prevents infinite curvature discontinuities
5. Topology sync: Update GPU neighbor list

**Why critical:** Resolves GEO-01 audit gap. Simple identity metric initialization creates discontinuities. Log-Euclidean interpolation ensures smooth geometric growth.

### 8. Emitter Geometric Placement **[MEDIUM PRIORITY]**

**Location to integrate:** Section 4.1 (emitter array specs)

**New content:**

**Spatial positioning:** 8th roots of unity in x-y plane
```
θ_k = 2πk/8  for k=1..8
(x_k, y_k) = (R·cos(θ_k), R·sin(θ_k))
```

**Mathematical justification:** Maximally-spaced placement minimizes destructive interference at sources.

**Synchronizer placement:** Center position with frequency:
```
f_sync = π·(1/φ)·√2·(32/27) ≈ 3.2 Hz
```

**Why critical:** Current spec defines frequencies but not geometric positions. Placement affects interference patterns.

## Mathematical Foundations Enhanced

### Dimensional Anisotropy

The bug sweep emphasizes that T^9 is NOT isotropic (unlike standard hypercubes):
- Each dimension has distinct physical semantics
- Different resolutions (4-14 bits)
- Different update rates
- Different coupling strengths

### Thermodynamic Justification

Enhanced explanation of why toroidal topology:
1. **Energy conservation:** Waves wrap around, no boundary dissipation
2. **Homogeneity:** No center/edge bias, uniform processing physics
3. **Compactness:** Finite phase space enables ergodic theory application

### Laplace-Beltrami on Curved Manifold

Complete derivation:
```
ΔΨ = (1/√|g|) ∂_i(√|g| g^ij ∂_j Ψ)
```

Requires:
- Inverse metric tensor g^ij
- Determinant √|g|
- Lazy recomputation strategy

## Integration Strategy

### Phase 1: Critical Structures (This Session)
1. ✅ Create integration notes document
2. ⏳ Add Coord9D struct specification
3. ⏳ Add toroidal wrapping functions
4. ⏳ Add MortonKey128 struct

### Phase 2: Transformation Algorithms (Next Session)
1. Add Projective Locality Mapper (SEM-01 resolution)
2. Add Hebbian-Riemannian plasticity equations
3. Add Lazy Cholesky inversion algorithm

### Phase 3: Traversal & Dynamics (Next Session)
1. Add Causal-Foliated Hilbert Scanning algorithm
2. Add Neurogenesis protocol with Log-Euclidean interpolation
3. Add Emitter geometric placement specifications

## Notes for Randy

**Document Status:**
- Target document `01_9d_toroidal_geometry.md` is 2,473 lines
- Has good topological foundations
- Missing concrete implementation algorithms
- Bug sweep fills critical implementation gaps

**Key Gaps Resolved:**
- ✅ Coord9D bitfield structure (completely new)
- ✅ Projective Locality Mapper (SEM-01 audit resolution)
- ✅ Causal-Foliated Hilbert Scan (entirely new algorithm)
- ✅ Neurogenesis with geometric continuity (GEO-01 resolution)
- ✅ 128-bit Morton coding for 9D sparse storage

**Critical Dependencies:**
- Coord9D → All memory addressing
- MortonKey128 → Sparse hash map (SHVO)
- Causal-Foliated Hilbert → Mamba-9D integration
- Projective Locality Mapper → Semantic clustering

**Estimated Integration Time:**
- Full integration: 2-3 hours
- Critical structures only: 45-60 minutes
- Documentation phase (current): Complete ✅

## Next Steps

1. Review integration notes
2. Decide: full integration vs. critical-only
3. Move to bug_sweep_003 (Nonary Encoding)

---

**Cross-references:**
- Bug sweep source: `gemini/responses/bug_sweep_002_9d_geometry.txt`
- Target document: `02_foundations/01_9d_toroidal_geometry.md`
- Backup created: Ready for creation before modifications
- Related audits: SEM-01, GEO-01, PHY-MEM-01
