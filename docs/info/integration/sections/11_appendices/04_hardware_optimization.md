# APPENDIX D: HARDWARE OPTIMIZATION GUIDELINES

## D.1 AVX-512 Vectorization

**Status:** RECOMMENDED - Significant performance improvement on supported hardware

### D.1.1 Compiler Flags

**Full AVX-512 Feature Set:**

```bash
-mavx512f      # Foundation (required)
-mavx512cd     # Conflict detection
-mavx512bw     # Byte and word
-mavx512dq     # Doubleword and quadword
-mavx512vl     # Vector length extensions
```

**CMake Configuration:**

```cmake
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)

if(COMPILER_SUPPORTS_AVX512)
    message(STATUS "AVX-512 support detected")
    add_compile_options(
        -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl
    )
    add_definitions(-DUSE_AVX512)
else()
    message(WARNING "AVX-512 not supported, falling back to AVX2")
    add_compile_options(-mavx2 -mfma)
    add_definitions(-DUSE_AVX2)
endif()
```

### D.1.2 Critical Loops to Vectorize

**1. DDS Emitter Phase Accumulator:**

```cpp
#ifdef USE_AVX512
void EmitterArray::tick_avx512(double* outputs) {
    __m512i phases_vec = _mm512_loadu_epi64(phases.data());
    __m512i tuning_vec = _mm512_loadu_epi64(tuning_words.data());

    // Add tuning words to phases (8 at once)
    phases_vec = _mm512_add_epi64(phases_vec, tuning_vec);

    // Store back
    _mm512_storeu_epi64(phases.data(), phases_vec);

    // Lookup sine values (8 parallel lookups)
    for (int i = 0; i < 8; ++i) {
        uint32_t lut_index = phases[i] >> 18;  // Top 14 bits
        outputs[i] = sine_lut[lut_index];
    }
}
#endif
```

**Expected Speedup:** 6-8x over scalar code

**2. Wave Propagation Step:**

```cpp
#ifdef USE_AVX512
void propagate_batch_avx512(std::complex<float>* wavefunctions,
                            const float* metric_tensors,
                            int batch_size) {
    for (int i = 0; i < batch_size; i += 8) {
        // Load 8 complex numbers (16 floats)
        __m512 real_vec = _mm512_loadu_ps(&wavefunctions[i]);
        __m512 imag_vec = _mm512_loadu_ps(&wavefunctions[i + 8]);

        // Perform wave computation on 8 nodes simultaneously
        // ... (wave propagation math)

        // Store results
        _mm512_storeu_ps(&wavefunctions[i], real_vec);
        _mm512_storeu_ps(&wavefunctions[i + 8], imag_vec);
    }
}
#endif
```

**Expected Speedup:** 4-6x over scalar code

**3. Metric Tensor Multiplication:**

```cpp
#ifdef USE_AVX512
void matmul_9x9_avx512(const float* A, const float* B, float* C) {
    // 9x9 matrix multiplication using AVX-512
    // Process 8 elements at a time

    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; j += 8) {
            __m512 sum = _mm512_setzero_ps();

            for (int k = 0; k < 9; ++k) {
                __m512 a_vec = _mm512_set1_ps(A[i * 9 + k]);
                __m512 b_vec = _mm512_loadu_ps(&B[k * 9 + j]);
                sum = _mm512_fmadd_ps(a_vec, b_vec, sum);
            }

            _mm512_storeu_ps(&C[i * 9 + j], sum);
        }
    }
}
#endif
```

**Expected Speedup:** 10-12x over scalar code

---

## D.2 CUDA Acceleration

**Status:** OPTIONAL - Recommended for large grids (81³+)

### D.2.1 Key Kernels to Implement

**1. Wave Propagation Kernel:**

```cuda
// File: src/physics/kernels/wave_propagate.cu

__global__ void wave_propagate_kernel(
    cuFloatComplex* wavefunctions,
    const float* metric_tensors,
    const float* resonances,
    const float* states,
    int num_nodes,
    float dt,
    float c0,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_nodes) return;

    // Load current state
    cuFloatComplex psi = wavefunctions[idx];
    float r = resonances[idx];
    float s = states[idx];

    // Compute damping and velocity
    float damping = alpha * (1.0f - r);
    float velocity = c0 / (1.0f + s);

    // Neighbor summation (18 neighbors in 9D)
    cuFloatComplex laplacian = make_cuFloatComplex(0.0f, 0.0f);

    for (int i = 0; i < 18; ++i) {
        int neighbor_idx = get_neighbor_index(idx, i);
        if (neighbor_idx >= 0) {
            laplacian = cuCaddf(laplacian, wavefunctions[neighbor_idx]);
        }
    }

    laplacian = cuCsubf(laplacian, cuCmulf(psi, make_cuFloatComplex(18.0f, 0.0f)));

    // Update wavefunction
    cuFloatComplex delta = cuCmulf(laplacian, make_cuFloatComplex(velocity * velocity * dt, 0.0f));
    delta = cuCsubf(delta, cuCmulf(psi, make_cuFloatComplex(damping * dt, 0.0f)));

    wavefunctions[idx] = cuCaddf(psi, delta);
}
```

**Launch Configuration:**

```cpp
int block_size = 256;
int num_blocks = (num_nodes + block_size - 1) / block_size;

wave_propagate_kernel<<<num_blocks, block_size>>>(
    d_wavefunctions,
    d_metric_tensors,
    d_resonances,
    d_states,
    num_nodes,
    dt, c0, alpha
);

cudaDeviceSynchronize();
```

**Expected Performance:**
- 81³ grid (531K nodes): 1-2ms per step
- 162³ grid (4.25M nodes): 10-15ms per step

**2. FFT Kernel (Spectral Firewall):**

```cuda
#include <cufft.h>

void spectral_analysis_cuda(const cuFloatComplex* signal,
                            float* spectrum,
                            int signal_length) {
    cufftHandle plan;
    cufftPlan1d(&plan, signal_length, CUFFT_C2C, 1);

    cuFloatComplex* d_signal;
    cudaMalloc(&d_signal, signal_length * sizeof(cuFloatComplex));

    cudaMemcpy(d_signal, signal, signal_length * sizeof(cuFloatComplex),
               cudaMemcpyHostToDevice);

    // Execute FFT
    cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);

    // Compute magnitudes on GPU
    compute_magnitudes_kernel<<<blocks, threads>>>(d_signal, spectrum, signal_length);

    cudaMemcpy(spectrum, /* device result */, signal_length * sizeof(float),
               cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_signal);
}
```

**3. Attention Computation (Transformer):**

```cuda
__global__ void wave_attention_kernel(
    const cuFloatComplex* queries,
    const cuFloatComplex* keys,
    const cuFloatComplex* values,
    cuFloatComplex* outputs,
    int seq_len,
    int d_model
) {
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (q_idx >= seq_len) return;

    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);

    for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
        // Wave correlation: Q · conj(K)
        cuFloatComplex score = cuCmulf(queries[q_idx], cuConjf(keys[k_idx]));

        // Weighted value
        sum = cuCaddf(sum, cuCmulf(score, values[k_idx]));
    }

    outputs[q_idx] = sum;
}
```

### D.2.2 CUDA Build Configuration

**CMakeLists.txt:**

```cmake
if(ENABLE_CUDA)
    enable_language(CUDA)

    find_package(CUDAToolkit REQUIRED)

    cuda_add_library(nikola_cuda STATIC
        src/physics/kernels/wave_propagate.cu
        src/reasoning/kernels/attention.cu
    )

    target_link_libraries(nikola_cuda
        PUBLIC
            CUDA::cudart
            CUDA::cufft
    )

    set_target_properties(nikola_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "80;86;89"  # Ampere, Ada, Hopper
    )
endif()
```

---

## D.3 Memory Layout Optimization

### D.3.1 Cache-Friendly Access Patterns

**✓ GOOD: Sequential Hilbert Order**

```cpp
// Sort nodes by Hilbert index for cache efficiency
std::vector<std::pair<uint64_t, TorusNode*>> indexed_nodes;

for (auto& [coord, node] : grid) {
    uint64_t hilbert_idx = HilbertMapper::encode(coord, 10);
    indexed_nodes.push_back({hilbert_idx, &node});
}

std::sort(indexed_nodes.begin(), indexed_nodes.end());

// Now iterate in cache-friendly order
for (auto& [idx, node_ptr] : indexed_nodes) {
    process_node(node_ptr);
}
```

**✗ BAD: Random Hash Map Iteration**

```cpp
// Poor cache locality - random memory access
for (auto& [coord, node] : grid) {
    process_node(&node);
}
```

**Performance Impact:** 3-5x speedup from improved cache hits

### D.3.2 Structure Alignment

```cpp
// 256-byte alignment for cache line optimization
struct alignas(256) TorusNode {
    std::complex<double> wavefunction;  // 16 bytes
    std::array<float, 45> metric_tensor;  // 180 bytes
    float resonance_r;  // 4 bytes
    float state_s;  // 4 bytes
    float padding[12];  // Pad to 256 bytes

    TorusNode() {
        std::memset(this, 0, sizeof(TorusNode));  // Zero padding
    }
};

static_assert(sizeof(TorusNode) == 256, "TorusNode must be 256 bytes");
```

**Benefit:** Exactly 4 cache lines (64 bytes × 4), no false sharing

### D.3.3 Structure-of-Arrays (SoA) for SIMD

**✓ GOOD: SoA Layout**

```cpp
struct TorusGridSoA {
    std::vector<float> wavefunction_real;      // Contiguous
    std::vector<float> wavefunction_imag;      // Contiguous
    std::vector<float> resonances;             // Contiguous
    std::vector<float> states;                 // Contiguous
    std::vector<std::array<float, 45>> metrics; // Contiguous

    // Vectorizable operations
    void update_resonances(float delta) {
        #pragma omp simd
        for (size_t i = 0; i < resonances.size(); ++i) {
            resonances[i] += delta;
        }
    }
};
```

**✗ BAD: Array-of-Structures (AoS)**

```cpp
struct TorusNode {
    float wavefunction_real;
    float wavefunction_imag;
    float resonance;
    float state;
};

std::vector<TorusNode> nodes;  // Interleaved - poor SIMD
```

---

## D.4 Recommended Hardware

### D.4.1 Minimum Configuration

**For Development and Testing:**

| Component | Specification | Notes |
|-----------|---------------|-------|
| **CPU** | Intel Xeon Gold 6248<br>or AMD EPYC 7452 | 20 cores, AVX-512 support |
| **RAM** | 64GB DDR4-3200 ECC | Minimum for 81³ grid |
| **GPU** | NVIDIA RTX 4060 Ti (16GB) | CUDA Compute 8.9 |
| **Storage** | 1TB NVMe SSD (PCIe 4.0) | For DMC checkpoints |
| **Network** | 1 Gbps Ethernet | For external API calls |

**Estimated Cost:** ~$5,000 USD

**Expected Performance:**
- 27³ grid: <1ms per physics step
- 81³ grid: 8-10ms per physics step (GPU)
- Training: ~50 samples/sec

### D.4.2 Recommended Configuration

**For Production Deployment:**

| Component | Specification | Notes |
|-----------|---------------|-------|
| **CPU** | Intel Xeon Platinum 8380<br>or AMD EPYC 9554 | 40 cores, AVX-512, high frequency |
| **RAM** | 256GB DDR5-4800 ECC | Large grid support (162³) |
| **GPU** | NVIDIA RTX 4090 (24GB)<br>or A100 (40GB/80GB) | High throughput, Tensor Cores |
| **Storage** | 4TB NVMe SSD (PCIe 5.0)<br>RAID 1 for redundancy | Fast checkpointing, persistence |
| **Network** | 10 Gbps Ethernet | Low-latency API access |

**Estimated Cost:** ~$15,000-25,000 USD

**Expected Performance:**
- 27³ grid: <0.3ms per physics step
- 81³ grid: 2-3ms per physics step (GPU)
- 162³ grid: 15-20ms per physics step (GPU)
- Training: 200+ samples/sec

### D.4.3 Cloud Deployment Options

**AWS EC2 Instances:**

| Instance Type | vCPUs | RAM | GPU | Use Case | Cost/Hour |
|--------------|-------|-----|-----|----------|-----------|
| **c7i.8xlarge** | 32 | 64GB | None | CPU-only (AVX-512) | ~$1.50 |
| **g5.4xlarge** | 16 | 64GB | A10G (24GB) | GPU acceleration | ~$1.60 |
| **p4d.24xlarge** | 96 | 1.1TB | 8× A100 | Large-scale training | ~$32.77 |

**Google Cloud Compute Engine:**

| Instance Type | vCPUs | RAM | GPU | Use Case | Cost/Hour |
|--------------|-------|-----|-----|----------|-----------|
| **c2-standard-30** | 30 | 120GB | None | CPU-only | ~$1.50 |
| **a2-highgpu-1g** | 12 | 85GB | A100 (40GB) | GPU acceleration | ~$3.67 |

---

## D.5 CPU-Specific Optimizations

### D.5.1 Intel Xeon (Skylake-SP and newer)

**Optimal Flags:**

```bash
-march=skylake-avx512 \
-mtune=skylake-avx512 \
-mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl \
-mprefer-vector-width=512
```

**Microarchitecture Features:**
- AVX-512 with 2 FMA units
- 512-bit vector registers (ZMM0-ZMM31)
- Hardware prefetching

### D.5.2 AMD EPYC (Zen 4 and newer)

**Optimal Flags:**

```bash
-march=znver4 \
-mtune=znver4 \
-mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl \
-mprefer-vector-width=256  # AMD optimized for 256-bit
```

**Note:** AMD Zen 4 has AVX-512, but performance may favor 256-bit vectors for some workloads.

### D.5.3 ARM (Apple Silicon, Graviton)

**Fallback to NEON:**

```bash
-march=armv8.2-a+fp16+simd
```

**Note:** No AVX-512 on ARM. Use NEON intrinsics for vectorization.

---

## D.6 Power and Thermal Considerations

### D.6.1 CPU Power Management

```bash
# Set performance governor (disable CPU frequency scaling)
sudo cpupower frequency-set -g performance

# Verify
cpupower frequency-info
```

**Impact:** Reduces jitter, improves consistency

### D.6.2 GPU Power Limits

```bash
# Set NVIDIA GPU to maximum power
sudo nvidia-smi -pl 350  # Watts (adjust for your GPU)

# Disable ECC (trade reliability for performance)
sudo nvidia-smi -e 0

# Set persistence mode
sudo nvidia-smi -pm 1
```

### D.6.3 Thermal Throttling Prevention

**Monitoring:**

```bash
# Watch CPU temperatures
watch -n 1 sensors

# Watch GPU temperatures
watch -n 1 nvidia-smi
```

**Recommended Cooling:**
- CPU: High-performance air cooler or 280mm+ AIO liquid cooler
- GPU: Case with good airflow (3+ intake fans)
- Ambient: Data center or air-conditioned room (<25°C)

---

**Cross-References:**
- See Section 9.4 for CMake build configuration
- See Appendix C for performance benchmarks
- See Appendix E for troubleshooting hardware issues
- See official documentation:
  - Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
  - CUDA Programming Guide: https://docs.nvidia.com/cuda/

