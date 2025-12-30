# Gemini Deep Research Task: Metric Tensor Precision and Quantization Strategy

## Problem Statement

**Location**: Section 2.1 (Foundations) + Section 3.1 (Wave Interference Processor)

**Issue Discovered**: The metric tensor `g_ij` is stored as `float` (FP32) but used in `double` precision (FP64) calculations, creating **repeated round-trip conversions** that may accumulate numerical error.

### Specific Details

1. **Storage Format** (Section 2.1):
   ```cpp
   struct TorusGridSoA {
       std::vector<float> metric_tensor;  // 45 floats per node (symmetric 9×9)
       // ...
   };
   ```

2. **Usage in Physics Kernel** (Section 3.1):
   ```cpp
   void propagate_wave_kernel(/* ... */) {
       // ... 
       std::complex<double> laplacian = compute_laplacian(coord);  // Uses double!
       // ...
   }
   
   std::complex<double> compute_laplacian(Coord9D coord) {
       // Requires metric inverse: g^ij
       // Requires determinant: sqrt(|g|)
       // Requires Christoffel symbols: Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
       
       // WHERE does float → double conversion happen?
   }
   ```

3. **Precision Escalation Questions**:
   - Is the metric tensor upcast to `double` **per node** during Laplacian computation?
   - Is it upcast **once per timestep** and cached?
   - Is the inverse `g^ij` stored separately in double precision?
   - What about Cholesky decomposition for geodesic calculation?

4. **Numerical Stability Concerns**:
   - **Cholesky Decomposition**: Requires SPD matrix, sensitive to rounding errors
   - **Matrix Inversion**: Condition number can amplify FP32 errors
   - **Christoffel Symbols**: Involve derivatives of metric → amplifies noise
   - **Geodesic Integration**: Accumulated error over many timesteps

## Research Objectives

### Primary Question
**What is the optimal precision strategy for the metric tensor across storage, computation, and GPU transfer?**

### Sub-Questions to Investigate

1. **Mixed Precision Analysis**:
   ```
   Strategy A: Store FP32, Compute FP32
   - Memory: 45 floats × 4 bytes = 180 bytes/node
   - Precision: ~7 decimal digits
   - Risk: Matrix inversion unstable for ill-conditioned metrics
   
   Strategy B: Store FP32, Compute FP64 (Current Implied)
   - Memory: 180 bytes/node (storage)
   - Precision: Computation accurate, but repeated conversions costly
   - Risk: Conversion overhead in tight loop
   
   Strategy C: Store FP64, Compute FP64
   - Memory: 45 doubles × 8 bytes = 360 bytes/node (2× larger!)
   - Precision: Maximum accuracy
   - Risk: Cache misses, memory bandwidth limited
   
   Strategy D: Store FP32, Cache FP64 Inverse
   - Memory: 180 bytes (g_ij) + 360 bytes (g^ij cached) = 540 bytes/node
   - Precision: High for computation, compact storage
   - Risk: Cache invalidation logic complex
   ```

2. **Condition Number Analysis**:
   - The metric tensor must be SPD (Section 9.1: Gershgorin initialization)
   - What is the typical condition number: κ(g) = λ_max / λ_min?
   - Does FP32 precision suffice for matrices with κ < 10³?
   - Do we need iterative refinement for inversion?

3. **GPU Precision**:
   - CUDA FP32 throughput: ~82 TFLOPS (RTX 4090)
   - CUDA FP64 throughput: ~1.3 TFLOPS (RTX 4090) → 63× slower!
   - Can we use **Tensor Cores** for FP16 with FP32 accumulation?
   - Does cuBLAS `cublasGemmEx` mixed precision help?

4. **Quantization Schemes**:
   - Can we compress metric to **block floating-point** (BFP)?
   - Can we use **posit** arithmetic (variable precision)?
   - Can we quantize to INT8 for storage, dequantize for compute?

## Required Deliverables

1. **Precision Budget**:
   For each operation, specify input/output precision:
   ```
   Operation                    | Input Precision | Compute Precision | Output Precision
   -----------------------------|-----------------|-------------------|-----------------
   Metric Storage               | -               | -                 | FP32
   Metric Inverse (Cholesky)    | FP32            | FP64              | FP64 (cached)
   Laplace-Beltrami Operator    | FP64 (from cache)| FP64             | FP64 (complex)
   Christoffel Symbols          | FP64            | FP64              | FP64 (9×9×9 tensor)
   Geodesic Integration (RK4)   | FP64            | FP64              | FP64 (trajectory)
   Metric Update (Plasticity)   | FP64 (computation)| FP64           | FP32 (write back)
   ```

2. **Error Propagation Analysis**:
   - Given FP32 storage with ε_machine ≈ 1.2 × 10⁻⁷
   - After Cholesky: ε_cholesky ≈ κ(g) × ε_machine
   - After Laplacian (27 neighbors × 9 dimensions): ε_laplacian ≈ 243 × κ × ε
   - After 1000 timesteps: How much drift?

3. **Performance Benchmarks**:
   Measure actual GPU kernel performance:
   ```cpp
   // Benchmark: Laplacian computation on 10^5 nodes
   
   Test 1: All FP32
   - Time: ??? ms
   - Accuracy: ??? (compare to reference)
   
   Test 2: FP32 storage, FP64 compute (current)
   - Time: ??? ms (conversion overhead)
   - Accuracy: ???
   
   Test 3: FP64 storage, FP64 compute
   - Time: ??? ms (memory bandwidth limited?)
   - Accuracy: ??? (ground truth)
   
   Test 4: FP16 Tensor Core compute
   - Time: ??? ms (should be fastest)
   - Accuracy: ??? (may be insufficient)
   ```

4. **Implementation Specification**:
   Complete code showing:
   - Where conversions happen (explicit casts)
   - Cache invalidation logic (if storing inverse)
   - CUDA kernel variants for each precision strategy
   - Performance vs accuracy tradeoff table

## Proposed Solutions to Evaluate

### Solution 1: Lazy FP64 Conversion with Caching
```cpp
class MetricTensorCache {
private:
    std::vector<float> storage_;         // FP32 storage
    std::vector<double> inverse_cache_;  // FP64 inverse (9×9 per node)
    std::vector<bool> cache_valid_;      // Dirty flag
    
public:
    Eigen::Matrix<double, 9, 9> get_inverse(size_t node_idx) {
        if (!cache_valid_[node_idx]) {
            // 1. Load FP32 metric
            auto g_fp32 = load_metric_fp32(node_idx);
            
            // 2. Convert to FP64
            Eigen::Matrix<double, 9, 9> g_fp64 = g_fp32.cast<double>();
            
            // 3. Invert with Cholesky (requires FP64 for stability)
            Eigen::LLT<Eigen::Matrix<double, 9, 9>> cholesky(g_fp64);
            auto g_inv = cholesky.solve(Eigen::Matrix<double, 9, 9>::Identity());
            
            // 4. Cache result
            inverse_cache_[node_idx] = g_inv;
            cache_valid_[node_idx] = true;
        }
        return inverse_cache_[node_idx];
    }
    
    void invalidate_cache(size_t node_idx) {
        cache_valid_[node_idx] = false;
    }
};
```

**Pro**: High accuracy, FP32 storage
**Con**: Cache memory overhead, invalidation complexity

### Solution 2: Pure FP32 with Iterative Refinement
```cpp
Eigen::Matrix<float, 9, 9> invert_metric_refined(const Eigen::Matrix<float, 9, 9>& g) {
    // Initial solve in FP32
    auto g_inv_approx = g.inverse();
    
    // Iterative refinement to reduce residual
    for (int iter = 0; iter < 3; ++iter) {
        auto residual = Eigen::Matrix<float, 9, 9>::Identity() - g * g_inv_approx;
        g_inv_approx += g_inv_approx * residual;
    }
    
    return g_inv_approx;
}
```

**Pro**: No mixed precision, simpler
**Con**: Still limited to FP32 accuracy (may be insufficient)

### Solution 3: Quantized Storage with FP64 Compute
```cpp
struct QuantizedMetric {
    int8_t quantized[45];   // -128 to +127
    float scale;            // Dequantization factor
    float bias;             // Offset
    
    Eigen::Matrix<double, 9, 9> dequantize_fp64() const {
        Eigen::Matrix<double, 9, 9> g;
        for (int i = 0; i < 45; ++i) {
            g(i / 9, i % 9) = static_cast<double>(quantized[i]) * scale + bias;
        }
        return g;
    }
};
// Storage: 45 bytes + 8 bytes = 53 bytes/node (vs 180 bytes FP32)
```

**Pro**: 3.4× memory savings
**Con**: Quantization error, complex calibration

## Research Questions

1. **Mixed Precision Deep Learning**:
   - How does PyTorch/TensorFlow handle FP16→FP32 mixed precision training?
   - Does NVIDIA's Automatic Mixed Precision (AMP) apply here?
   - What are the best practices for gradient scaling?

2. **Numerical Linear Algebra**:
   - Does LAPACK have recommended precision for Cholesky decomposition?
   - What condition number threshold requires FP64 vs FP32?
   - Can we use QR decomposition instead of Cholesky for better stability?

3. **Physics Simulation Standards**:
   - How does GROMACS (molecular dynamics) handle precision?
   - How does OpenFOAM (CFD) balance memory vs accuracy?
   - What does the SYCL/DPC++ standard recommend for heterogeneous computing?

## Success Criteria

- [ ] Numerical error < 1% over 10⁶ timesteps
- [ ] GPU kernel meets <1ms target
- [ ] Memory footprint reasonable (< 1 GB for 10⁵ nodes)
- [ ] No catastrophic cancellation in Laplacian
- [ ] Cholesky decomposition succeeds 100% of time

## Output Format

Please provide:
1. **Error Analysis** (2-3 pages): Theoretical error bounds
2. **Benchmark Results** (1 page): GPU timing data
3. **Recommended Strategy** (1 page): Which precision scheme to use
4. **Code Implementation** (C++/CUDA): Complete kernel variants
5. **Migration Path** (1 page): How to transition from FP32 storage

## Additional Context

This affects:
- Section 2.1: Coord9D and metric tensor storage
- Section 3.1.7: SoA memory layout
- Section 3.2.3: Mamba parameter extraction (uses metric)
- Section 7.3: Visual cymatics (geometric rendering needs metric)
- Section 8.1: Phase 0 (numerical stability validation)

Performance is critical: The Laplacian is computed **10⁵ nodes × 1000 Hz = 10⁸ times per second**.

---

**Priority**: P2 - HIGH (Affects core physics stability)
**Estimated Research Time**: 5-7 hours
**Dependencies**: TASK_COORD9D_PORTABILITY (bit depth affects quantization resolution)
