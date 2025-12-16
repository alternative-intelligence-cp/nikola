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

    // Extract LUT indices (top 14 bits of each 64-bit phase)
    __m256i indices_32 = _mm512_cvtepi64_epi32(
        _mm512_srli_epi64(phases_vec, 18)  // Shift right by 18 bits
    );

    // AVX-512 Gather: Load 8 sine values from LUT in parallel
    __m512d sine_values = _mm512_i32gather_pd(
        indices_32,                     // Indices (32-bit)
        sine_lut,                       // Base pointer
        8                               // Scale factor (8 bytes per double)
    );

    // Store results
    _mm512_storeu_pd(outputs, sine_values);
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

**Sequential Hilbert Order for Cache Efficiency:**

```cpp
// Sort nodes by Hilbert index for optimal cache line utilization
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

**Random Hash Map Iteration (Poor Cache Locality):**

```cpp
// Random memory access pattern
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

**SoA Layout for Vectorization:**

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

**Array-of-Structures (AoS) - Poor SIMD Performance:**

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

---

## GAP-043: Performance Tuning Cookbook

**SOURCE**: Gemini Deep Research Round 2, Batch 41-44
**INTEGRATION DATE**: December 16, 2025
**GAP ID**: GAP-043 (TASK-043)
**PRIORITY**: CRITICAL
**STATUS**: FABRICATION-READY SPECIFICATION

### Optimization Philosophy: The "Phase 0" Mandate

Performance tuning of Nikola v0.0.4 is governed by "Phase 0" mandates. Unlike typical AI optimization focusing on Matrix Multiplication (MatMul) FLOPS, Nikola architecture is **Memory-Bound**. Bottleneck is moving 9D grid state between VRAM and Compute Units. Therefore, all tuning focuses on **Data Locality**, **Cache Efficiency**, and **Bandwidth Saturation**.

### Knob-Tuning Guide

Operators control system's cognitive dynamics via specific parameters:

| Knob | Parameter Name | Default | Range | Impact | Tuning Advice |
|------|----------------|---------|-------|--------|---------------|
| **Learning Rate** | `hebbian_rate` ($\eta$) | 0.01 | 0.001 - 0.1 | Controls speed of metric tensor warping | Reduce if system exhibits "Manic" switching (instability). Increase if "Boredom" is high or learning is stagnant |
| **ATP Cost** | `metabolic_cost_plasticity` | 1.5 | 1.0 - 5.0 | Cost to write to long-term memory | Increase to force system to prioritize only high-resonance memories (better filtering). Decrease to allow rapid, broad learning |
| **Consolidation** | `nap_interval_trigger` | 15% | 5% - 30% | ATP threshold to trigger Nap | Higher % = more frequent, shorter naps (better for stability). Lower % = longer wake periods (better for complex tasks) |
| **Time Step** | `physics_dt` | 1ms | 0.1ms - 5ms | Physics integration resolution | **WARNING**: Must satisfy $\Delta t < 1/(\beta \cdot \|Psi\|_{max})$ for stability. Reduce if energy diverges |
| **Grid Size** | `block_size` | 19683 | $3^9$ powers | Number of nodes per block | Fixed at compile time. Changing requires recompilation. Must align with $3^9$ for efficient Torus mapping |
| **Dither Noise** | `dither_amplitude` | 1e-4 | 1e-5 - 1e-3 | Amplitude of injected noise | Increase to prevent "Resonance Lock-in" (obsessive thoughts). Decrease if Signal-to-Noise ratio drops below 20dB |

### Diagnostic Flowcharts

#### Scenario A: System Latency is High (> 100ms response)

1. **Check Physics Loop**: Is `tick_time` > 1ms?
   * **Yes**: Memory Bottleneck. Run `perf stat`. Check L1/L2 Cache Miss Rate.
     - If Miss Rate > 10%: Verify Structure-of-Arrays (SoA) alignment (`alignas(64)`). Verify Hilbert Curve indexing effectively clusters nodes.
     - If Miss Rate < 10%: Check AVX-512 usage. Are intrinsics being generated? Recompile with `-march=native`.
   * **No**: Proceed to 2.

2. **Check Message Queue**: Is ZMQ High-Water Mark (HWM) reached?
   * **Yes**: Backpressure. Cognitive layer (Mamba-9D) too slow for physics engine. Increase `control_plane_timeout` or throttle physics via sleep.
   * **No**: Proceed to 3.

3. **Check Garbage Collection**: Is `shm_unlink` lagging?
   * **Yes**: OS Overhead. Reduce shared memory segment size or frequency of frame exports.

#### Scenario B: Energy Divergence (Hallucinations/Crashes)

1. **Check Hamiltonians**: Is Energy Drift > 0.01%?
   * **Yes**: Integration Failure.
     - **Immediate Action**: Reduce `physics_dt` by 50%.
     - **Root Cause Check**: Verify Symplectic Integrator uses Split-Operator method, not Verlet. Verify Kahan Summation active for Laplacian accumulation.
   * **No**: Proceed to 2.

2. **Check Neurochemistry**: Is Dopamine pinned at 1.0 or 0.0?
   * **Yes**: Gating Failure. Check `AtomicDopamine` implementation for race conditions. Verify beta sensitivity parameter.

### Benchmark Suite and Baseline Expectations

Run `twi-ctl benchmark` to validate system health against these baselines.

**Baseline Expectations** (Hardware: Single NVIDIA RTX 4090 / Intel Xeon w/ AVX-512):

| Metric | Benchmark Test | Baseline Target | Failure Threshold |
|--------|----------------|-----------------|-------------------|
| **Physics Latency** | `BM_WavePropagation_81^3` | 7.8 ms / step | > 12 ms |
| **Small Grid Latency** | `BM_WavePropagation_27^3` | 0.48 ms / step | > 1 ms (Critical P0 requirement) |
| **Memory Bandwidth** | SoA Efficiency Test | 100% utilization | < 80% (Indicates AoS regression) |
| **Cache Hit Rate** | L1/L2 Cache Profiling | ~95% | < 85% |
| **Precision** | Laplacian Accuracy (Kahan) | Error $\sim 10^{-7}$ | $> 10^{-5}$ (Indicates Kahan failure) |
| **Energy Drift** | 24-hour Stability Test | < 0.01% | > 0.05% |

### Hardware-Specific Profiles

#### Profile 1: CPU-Only (Dev/Debug)

* **Target**: Intel Core i9 / Xeon / AMD Ryzen 9 (AVX-512 Support MANDATORY)
* **Rationale**: Uses vector units to simulate parallel wave propagation
* **Settings**:
  - `ENABLE_CUDA = OFF`
  - `OMP_NUM_THREADS = <physical_cores>`
  - `physics_dt = 5ms` (Slower simulation time, physics runs at 200Hz instead of 1kHz)
* **Optimization**: Relies entirely on AVX-512 vectorization of SoA layout. Requires `alignas(64)` strict enforcement

#### Profile 2: Single GPU (Consumer High-End - RTX 4090)

* **Target**: NVIDIA RTX 4090 (24GB VRAM)
* **Rationale**: Excellent FP32 performance, decent memory bandwidth
* **Settings**:
  - `ENABLE_CUDA = ON`
  - `CUDA_BLOCK_SIZE = 256`
  - `precision = FP32` (FP64 too slow on consumer cards; use Kahan Summation for precision)
* **Optimization**: Uses "Coalesced Memory Access" patterns in CUDA kernels. Grid size limited to ~14M active nodes due to 24GB VRAM limit

#### Profile 3: Multi-GPU Cluster (Datacenter - A100/H100)

* **Target**: 4x or 8x NVIDIA A100 (80GB) with NVLink
* **Rationale**: Massive VRAM allows for "Neurogenesis" without OOM crashes
* **Settings**:
  - `ENABLE_CUDA = ON`
  - `precision = FP64` (Optional, for higher fidelity/research)
  - `distributed_sharding = ENABLED` (Morton-code based partitioning)
* **Optimization**: Requires MPI/NCCL integration for halo exchange. Uses NVLink for high-bandwidth transfer of boundary regions. Can scale to >100M active nodes

### Implementation Status

- **Status**: SPECIFICATION COMPLETE
- **Optimization Focus**: Memory-bound (not compute-bound), data locality, cache efficiency, bandwidth saturation
- **Tuning Parameters**: Learning rate, ATP cost, consolidation trigger, physics timestep, dither noise
- **Diagnostic Flowcharts**: Latency diagnosis (cache miss, ZMQ backpressure), energy divergence (timestep, Kahan summation)
- **Benchmark Baselines**: Physics latency (0.48ms - 7.8ms), cache hit (95%), energy drift (<0.01%)
- **Hardware Profiles**: CPU-only (AVX-512, 200Hz), Single GPU (RTX 4090, 14M nodes), Multi-GPU (A100, >100M nodes)

### Cross-References

- [Structure-of-Arrays (SoA) Layout](../04_infrastructure/06_database_persistence.md)
- [AVX-512 Vectorization](../02_foundations/03_balanced_nonary_logic.md)
- [Hilbert Curve Indexing](../04_infrastructure/06_database_persistence.md)
- [Symplectic Integrator](../02_foundations/02_wave_interference_physics.md)
- [Kahan Summation](../02_foundations/02_wave_interference_physics.md)
- [Metabolic Controller (ATP)](../05_autonomous_systems/01_computational_neurochemistry.md)
- [Nap System](../06_persistence/04_nap_system.md)
- [ZeroMQ Backpressure](../04_infrastructure/01_zeromq_spine.md)
- [Metric Tensor Warping](../02_foundations/01_9d_toroidal_geometry.md)

---

## GAP-046: High-Frequency CUDA Kernel Optimization Strategies

**SOURCE**: Gemini Deep Research Round 2, Batch 45-47
**INTEGRATION DATE**: December 16, 2025
**GAP ID**: GAP-046 (TASK-046)
**PRIORITY**: CRITICAL
**STATUS**: FABRICATION-READY SPECIFICATION

### The Launch Overhead Bottleneck

Nikola Physics Engine is governed by Unified Field Interference Equation (UFIE), requiring symplectic integration step every 1ms (1000 Hz) to maintain energy conservation ($|dH/dt| < 0.01\%$). This requirement imposes **hard real-time constraint** on GPU compute pipeline fundamentally at odds with batch-oriented design of modern CUDA drivers.

In standard CUDA execution model, host (CPU) enqueues kernel launch command to device (GPU) driver. This involves traversing PCIe bus, driver validation, and insertion into GPU's hardware work queue.

**Overhead Breakdown**:
- **Driver Overhead**: Typically 5-20 μs per launch
- **PCIe Latency**: 2-5 μs for command transmission
- **Kernel Execution**: For sparse grid update, potentially 50-100 μs

Symplectic Split-Operator method requires decomposing Hamiltonian evolution into sequential operators: Kinetic → Potential → Nonlinear → Damping. This results in 5-6 separate kernel launches per timestep.

$$\text{Total Overhead} \approx 6 \text{ kernels} \times 15 \mu s = 90 \mu s$$

This consumes nearly **10% of 1000 μs budget** purely on metadata management. When combined with memory transfers for audio/visual pipeline and synchronization barriers, "Temporal Decoherence" threshold (500 μs) is easily breached, leading to numerical instability and "cognitive seizures."

### Strategy A: CUDA Graphs for Deterministic Execution

To eliminate CPU-side launch overhead, we implement **CUDA Graphs**. This feature allows definition of dependency graph of kernels and memory operations once, and then execution of entire graph with single CPU launch call.

#### Graph Capture and Replay Architecture

Instead of `cudaLaunchKernel_A → cudaLaunchKernel_B → cudaLaunchKernel_C`, we capture this sequence into `cudaGraphExec_t`. GPU driver uploads entire work definition to Command Processor (CP) on GPU.

**Implementation Specification**:

```cpp
// include/nikola/physics/cuda_graph_manager.hpp

class PhysicsGraph {
   cudaGraph_t graph;
   cudaGraphExec_t instance;
   cudaStream_t stream;
   bool captured = false;

public:
   void capture_sequence(std::function<void()> kernel_sequence) {
       cudaStreamCreate(&stream);
       // Begin capture in Global mode to catch all stream activities
       cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

       // Execute the lambda containing the 5 symplectic substeps
       kernel_sequence();

       cudaStreamEndCapture(stream, &graph);
       // Instantiate the executable graph (upload to GPU)
       cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
       captured = true;
   }

   void launch() {
       if (!captured) throw std::runtime_error("Graph not captured");
       // Single launch call triggers the entire 5-kernel sequence
       cudaGraphLaunch(instance, stream);
   }
};
```

**Application to UFIE**:

Symplectic integrator is encapsulated in `kernel_sequence` lambda. Graph captures dependencies between Kinetic (`wave_kinetic_kernel`) and Potential (`wave_potential_kernel`) steps.

- **Result**: Launch overhead reduces from $6 \times 15 \mu s$ to $1 \times 5 \mu s$
- **Benefit**: Deterministic execution time. GPU scheduler handles transitions between kernels without CPU intervention, minimizing jitter caused by OS interrupts on host

**Dynamic Topology Challenge**:

Nikola grid supports Neurogenesis (dynamic addition of nodes). CUDA Graphs are static; grid dimensions and memory pointers are baked into instantiated graph.

- **Update Protocol**: When `active_node_count` changes, DifferentialTopologyManager must trigger `graph_update_required` flag
- **Re-instantiation**: Graph must be re-captured or updated using `cudaGraphExecUpdate`. This is expensive operation (~200 μs). Therefore, Neurogenesis events are batched and processed only during specific "Plasticity Windows" to avoid stalling physics loop

### Strategy B: Persistent Kernels (The Mega-Kernel)

For ultra-low latency scenarios where even 5 μs is too costly (e.g., high-frequency audio resonance at 44.1 kHz), we utilize **Persistent Kernel** pattern. This eliminates launch overhead entirely by keeping kernel running indefinitely on GPU.

#### Producer-Consumer Mechanism via Zero-Copy Memory

This approach turns GPU into autonomous agent that polls for work:

1. **Launch**: Kernel launched at system boot with infinite loop: `while(system_running) {...}`
2. **Communication**: CPU writes input data (e.g., new audio samples) to Zero-Copy Memory (pinned host memory mapped to device address space via `cudaHostAllocMapped`)
3. **Signaling**: CPU sets atomic flag `doorbell` in mapped memory
4. **Reaction**: GPU threads, spinning on doorbell address, detect change, execute physics step, and write completion flag

**Implementation Specification**:

```cpp
// src/physics/kernels/persistent_loop.cu

struct ControlBlock {
   volatile uint32_t host_seq;   // CPU increments to trigger tick
   volatile uint32_t device_seq; // GPU increments when done
   volatile bool running;
};

__global__ void persistent_physics_loop(
   TorusGridSoA grid,
   ControlBlock* ctrl,
   float dt
) {
   // Shared memory cache to reduce traffic to system memory (PCIe)
   __shared__ uint32_t cached_seq;

   // Only thread 0 in the block monitors the doorbell
   if (threadIdx.x == 0) {
       cached_seq = ctrl->device_seq;
   }
   __syncthreads();

   while (ctrl->running) {
       // Spin-wait loop
       if (threadIdx.x == 0) {
           // Wait for host_seq to advance beyond what we last processed
           while (ctrl->host_seq == cached_seq && ctrl->running) {
               // Optimization: nanosleep to reduce power/heat on empty spins
               // Requires Compute Capability 7.0+
               __nanosleep(100);
           }
           cached_seq = ctrl->host_seq;
       }
       __syncthreads(); // All threads wake up to process the new tick

       if (!ctrl->running) break;

       // --- EXECUTE PHYSICS STEP ---
       // Critical: All threads in the grid must participate.
       // We use Cooperative Groups for global synchronization if needed.
       process_symplectic_step_device(grid, dt);
       // ----------------------------

       // Signal completion
       __syncthreads();
       if (threadIdx.x == 0) {
           ctrl->device_seq = cached_seq;
           __threadfence_system(); // Ensure write is visible to CPU across PCIe
       }
   }
}
```

#### Cooperative Groups and Occupancy

Standard kernel cannot synchronize across different Thread Blocks. If physics simulation requires global data dependencies (e.g., global FFT for spectral analysis), persistent kernel will deadlock if one block waits for another that hasn't been scheduled.

- **Solution**: Mandate use of Cooperative Groups (`cooperative_groups::this_grid().sync()`)
- **Launch Requirement**: Kernel must be launched via `cudaLaunchCooperativeKernel`
- **Occupancy Constraint**: Total number of blocks must fit on GPU's Streaming Multiprocessors (SMs) simultaneously. For NVIDIA H100 with 132 SMs, if kernel uses 256 threads/block, we can launch roughly $132 \times 8 = 1056$ blocks resident. This sets hard limit on grid size supported by this mode. If grid exceeds residency, must fall back to CUDA Graphs

### Integration of Visual and Audio Pipelines

1000 Hz physics loop must interface with 60 Hz video and 44.1 kHz audio. This creates multi-rate signal processing problem.

#### Audio-Visual Ring Buffers

To bridge 44.1 kHz audio stream (22 μs period) with 1000 Hz physics tick (1000 μs period), we utilize WaveformSHM zero-copy shared memory architecture.

**Mechanism**:

1. **Audio Ingestion**: Dedicated thread captures PCM audio and writes it to ring buffer in shared memory
2. **Spectral Injection**: Audio thread performs FFT on incoming window. Resulting frequency bins are mapped directly to Resonance ($r$) dimension of 9D grid. Physics Engine reads this spectral map once per millisecond. This preserves harmonic content without requiring physics engine to run at 44.1 kHz

**Visual Pipeline**:

Visual data (60 Hz) is static for ~16 physics ticks. To prevent "step function" artifacts which cause high-frequency ripple in UFIE, inputs are temporally interpolated (faded) between frames over 16ms window. This smoothing is applied via simple linear interpolation kernel fused into Persistent Kernel's input reading stage.

### Implementation Status

- **Status**: SPECIFICATION COMPLETE
- **CUDA Graphs**: 80% launch overhead reduction ($6 \times 15\mu s \to 1 \times 5\mu s$), deterministic execution
- **Graph Capture**: PhysicsGraph class with capture_sequence() and launch() methods
- **Dynamic Topology**: Neurogenesis batching during Plasticity Windows to avoid 200μs re-instantiation stalls
- **Persistent Kernels**: Zero-copy memory doorbell pattern, __nanosleep() for power efficiency
- **Cooperative Groups**: cudaLaunchCooperativeKernel for global synchronization, occupancy limit ~1056 blocks (H100)
- **Audio Integration**: Spectral injection via FFT to Resonance dimension, preserves harmonics at 1kHz physics rate
- **Visual Integration**: 16ms temporal interpolation to prevent step-function ripple

### Cross-References

- [Symplectic Integration](../02_foundations/02_wave_interference_physics.md)
- [Unified Field Interference Equation (UFIE)](../02_foundations/02_wave_interference_physics.md)
- [Temporal Decoherence](../04_infrastructure/02_orchestrator_router.md)
- [Neurogenesis](../02_foundations/01_9d_toroidal_geometry.md)
- [DifferentialTopologyManager](../02_foundations/01_9d_toroidal_geometry.md)
- [WaveformSHM](../04_infrastructure/06_database_persistence.md)
- [9D Resonance Dimension](../02_foundations/01_9d_toroidal_geometry.md)

---

