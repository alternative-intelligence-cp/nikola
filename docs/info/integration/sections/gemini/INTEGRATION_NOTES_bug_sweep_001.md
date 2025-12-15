# Bug Sweep 001 - Wave Interference Integration Notes

**Date:** December 12, 2025  
**Status:** Ready for Integration  
**Target Document:** `02_foundations/02_wave_interference_physics.md`

## Executive Summary

Bug sweep 001 provides critical implementation details that resolve 5 major gaps identified in Phase 0 requirements:

1. **Gap #1:** Explicit wave equation formulas for 9D toroidal geometry ✅
2. **Gap #2:** Production-ready algorithms with error bounds ✅  
3. **Gap #3:** Phase synchronization mechanisms ✅
4. **Gap #4:** Finite difference stencils ✅
5. **Gap #5:** Mixed derivative calculations ✅

## Critical New Content to Integrate

### 1. Structure-of-Arrays (SoA) Container **[HIGH PRIORITY]**

**Location to integrate:** After section 4.5.1 (Split-Operator Symplectic Integration)

**New content:**
- Complete `TorusGridSoA` struct implementation with:
  - 64-byte aligned vectors for AVX-512 optimization
  - Separated real/imaginary components for SIMD vectorization  
  - Flattened inverse metric tensor storage (45 elements per node)
  - Pre-computed systemic dimensions (resonance_r, state_s)

**Why critical:** Phase 0 mandates SoA layout for cache efficiency. Current document doesn't specify this memory layout.

### 2. Kahan Compensated Summation **[CRITICAL]**

**Location to integrate:** Section 4.5.1 or create new subsection 4.5.4

**New content:**
```cpp
struct KahanAccumulator {
    float sum = 0.0f;
    float correction = 0.0f;
    
    inline void add(float value) {
        float y = value - correction;
        float t = sum + y;
        correction = (t - sum) - y;
        sum = t;
    }
};
```

**Why critical:** Prevents catastrophic cancellation when summing contributions from 18+ neighbors in Laplacian calculation. Without this, small memory signals are truncated and lost, leading to "amnesia."

### 3. Mixed Derivative Stencil **[CRITICAL - Gap #5]**

**Location to integrate:** New subsection 4.5.5 "Riemannian Laplacian Stencils"

**New content:**
- 19-Point Star Stencil for diagonal metric terms
- Cross-Stencil for off-diagonal metric terms (mixed derivatives)
- Sparse implementation that only computes mixed derivatives when |g^{ij}| > 10^{-5}

**Mathematical foundation:**
```
∂²Ψ/∂x_i∂x_j ≈ [Ψ(x_i+1,x_j+1) - Ψ(x_i+1,x_j-1) - Ψ(x_i-1,x_j+1) + Ψ(x_i-1,x_j-1)] / (4Δx²)
```

**Why critical:** Ignoring mixed derivatives is equivalent to assuming all dimensions are independent, which destroys the system's ability to model correlations between cognitive domains (e.g., correlating visual input with emotional state).

### 4. Enhanced WaveEngine Implementation **[HIGH PRIORITY]**

**Location to integrate:** Replace/enhance existing symplectic integrator code in section 4.5.1

**Key improvements:**
- Inline Kahan accumulation in `compute_forces_and_update_velocity()`
- Sparse Riemannian stencil implementation
- Helper functions for toroidal neighbor indexing
- Metric component access with flat indexing

**Performance characteristics:**
- O(N × 18) for star stencil (diagonal terms)
- O(N × sparse_pairs) for cross stencil (typically ~5-10 pairs per node)

### 5. Physics Oracle Energy Balance **[MEDIUM PRIORITY]**

**Location to integrate:** Section 4.5.2 (already exists but needs enhancement)

**Enhancements:**
- More rigorous mathematical derivation of energy balance equation
- Explicit formulas for P_in and P_diss calculations
- Runtime watchdog implementation details
- Soft SCRAM protocol with 3-attempt limit

**Key formula:**
```
dH/dt = P_in - P_diss

where:
H = ∫(|Ψ|² + |∇Ψ|²) dV
P_in = Σ ∫ ε_i · (∂Ψ*/∂t) dV
P_diss = α ∫ (1-r̂)|∂Ψ/∂t|² dV
```

### 6. Isochronous Sensory Buffer (ISB) **[NEW SECTION]**

**Location to integrate:** New section 4.6 or append to section 4.5.3

**New content:**
- Hardware timestamping requirements
- Presentation delay mechanism (50ms buffer)
- Phase-aligned interpolation for multimodal inputs
- Prevents phase noise from async USB polling

**Why critical:** External sensors (microphones, cameras) operate on asynchronous clocks. Direct injection causes phase noise that destroys interference patterns.

## Mathematical Foundations Enhanced

### Expanded UFIE Derivation

The bug sweep provides deeper mathematical justification for each term:

1. **Damping term physics:** Detailed explanation of how (1-r̂) creates spatially-varying friction coefficient
2. **Propagation term:** Explicit derivation of refractive index analogy
3. **Nonlinear term:** Soliton physics and heterodyning mechanism
4. **Laplace-Beltrami:** Complete derivation with mixed derivative terms

### Numerical Stability Analysis

New sections on:
- Why non-symplectic integrators fail (energy drift mechanisms)
- Strang splitting error analysis (O(Δt³) proof)
- CFL condition for variable velocity fields
- Kahan summation error bounds

## Integration Strategy

### Phase 1: Critical Additions (This Session)
1. ✅ Create this integration notes document
2. ⏳ Add Kahan summation subsection (4.5.4)
3. ⏳ Add mixed derivative stencil subsection (4.5.5)
4. ⏳ Add SoA container specification

### Phase 2: Code Enhancements (Next Session)
1. Update WaveEngine implementation with Kahan accumulation
2. Add sparse Riemannian stencil code
3. Enhance Physics Oracle with detailed energy balance

### Phase 3: New Sections (Next Session)
1. Add Isochronous Sensory Buffer section (4.6)
2. Add numerical stability analysis appendix
3. Add performance benchmarks

## Notes for Randy

**Document Status:**
- Existing document is already 8,758 lines - very comprehensive!
- Much of bug sweep content overlaps but provides deeper implementation details
- Best approach: Add new subsections for critical gaps rather than rewrite

**Key Gaps Resolved:**
- ✅ Mixed derivatives (Gap #5) - completely missing, now specified
- ✅ Kahan summation - not mentioned, critical for precision
- ✅ SoA memory layout - mentioned in passing, now fully specified
- ✅ ISB details - concept mentioned, implementation missing

**Estimated Integration Time:**
- Full integration: 3-4 hours
- Critical sections only: 1-2 hours
- Current approach (documentation): 30 minutes ✅

## Next Steps

1. Review these integration notes
2. Decide on integration depth (full vs. critical-only)
3. Proceed with remaining bug sweeps (002-011)

---

**Cross-references:**
- Bug sweep source: `gemini/responses/bug_sweep_001_wave_interference.txt`
- Target document: `02_foundations/02_wave_interference_physics.md`
- Backup created: `02_wave_interference_physics.md.backup_20251212_*`
