# Gemini Deep Research Task: Hilbert Curve Dimensionality Mismatch

## Problem Statement

**Location**: Section 3.2.1-3.2.2 (Causal-Foliated Hilbert Scanning for Mamba-9D)

**Issue Discovered**: Type mismatch between the causal foliation scanner's 8D coordinate extraction and the HilbertMapper API contract.

### Specific Details

1. **In Section 3.2.1**, the `HilbertMapper::encode()` function signature is:
   ```cpp
   static uint64_t encode(const std::array<uint32_t, 9>& coords, int bits)
   ```

2. **In Section 3.2.2**, the `CausalFoliationScanner` extracts **8D spatial coordinates** (excluding time dimension):
   ```cpp
   Coord8D space = {
       grid.coords_r[i], grid.coords_s[i],
       grid.coords_u[i], grid.coords_v[i], grid.coords_w[i],
       grid.coords_x[i], grid.coords_y[i], grid.coords_z[i]
   };
   uint64_t h = compute_hilbert_8d_bmi2(space);
   ```

3. **Type Definition Conflict**:
   - `Coord8D` is defined as `using Coord8D = std::array<uint32_t, 8>;`
   - But `encode()` expects `std::array<uint32_t, 9>`
   - The private method `compute_hilbert_8d_bmi2()` exists but is never connected to the public API

## Research Objectives

### Primary Question
**What is the correct API design for supporting both 9D full-manifold Hilbert encoding AND 8D causal-foliated spatial-only encoding?**

### Sub-Questions to Investigate

1. **API Design Patterns**:
   - Should there be two separate public methods: `encode_9d()` and `encode_8d()`?
   - Should there be one variadic method: `encode<size_t N>(const std::array<uint32_t, N>& coords)`?
   - Should the causal scanner use a different class entirely (`SpatialHilbertMapper`)?

2. **Performance Implications**:
   - Does BMI2 PDEP instruction performance differ between 8-way and 9-way interleaving?
   - Are there cache-line alignment benefits to 8D vs 9D?
   - What is the impact on Morton code density (64 bits vs 72 bits)?

3. **Correctness Verification**:
   - Does excluding the time dimension from Hilbert encoding preserve spatial locality correctly?
   - Are there edge cases where temporal ordering conflicts with spatial Hilbert order?
   - How do you handle the transition from temporal to spatial sorting (tie-breaking)?

4. **Implementation Strategy**:
   - Should the time dimension be included in Hilbert encoding but given lowest-order bits?
   - Should there be explicit masking of the time dimension in the causal scanner?
   - What happens if time resolution changes (14-bit → 16-bit)?

## Required Deliverables

1. **Production-Ready API Specification** (C++23):
   - Complete class definition for `HilbertMapper` supporting both use cases
   - Clear documentation of which method to use when
   - Template specialization if using generic approach
   - Unit test cases for both 8D and 9D encoding

2. **Performance Analysis**:
   - Theoretical FLOPS comparison for 8D vs 9D BMI2 encoding
   - Memory bandwidth analysis (64-bit vs 72-bit indices)
   - Cache hit rate prediction for Morton ordering

3. **Correctness Proof**:
   - Mathematical proof that 8D Hilbert + temporal sort = total order
   - Edge case analysis (what if all nodes same timestamp?)
   - Stability guarantees for the combined sort

4. **Integration Patch**:
   - Exact code changes needed in Section 3.2.1 and 3.2.2
   - Migration path from current broken API
   - Backward compatibility considerations

## Context Files to Reference

- Section 2.1: Coord9D structure definition and bit-packing scheme
- Section 3.2.1: HilbertMapper implementation with BMI2 intrinsics
- Section 3.2.2: CausalFoliationScanner implementation
- Section 8.1: Phase 0 critical remediations (check for dependencies)

## Success Criteria

- [ ] API compiles with zero warnings on GCC 13+ and Clang 16+
- [ ] Both 8D and 9D encoding have explicit test coverage
- [ ] Performance is within 5% of theoretical BMI2 maximum
- [ ] Causal ordering is mathematically proven correct
- [ ] No ABI breaks from current specification

## Additional Considerations

- AVX-512 vectorization: Can we encode 8 coordinates in parallel using `_mm512_*` intrinsics?
- GPU compatibility: Does CUDA have equivalent to BMI2 PDEP? (Answer: No, need software fallback)
- Endianness: Are the bit masks portable to big-endian architectures? (ARM64 compatibility)

## Output Format

Please provide:
1. **Executive Summary** (1 page): Recommended solution with rationale
2. **Technical Specification** (5-10 pages): Complete API with code examples
3. **Proof of Correctness** (2-3 pages): Mathematical verification
4. **Test Suite** (code): Unit tests covering all edge cases
5. **Integration Instructions** (1 page): Step-by-step patch application

---

**Priority**: P1 - CRITICAL (Blocks Mamba-9D integration)
**Estimated Research Time**: 4-6 hours
**Dependencies**: None
