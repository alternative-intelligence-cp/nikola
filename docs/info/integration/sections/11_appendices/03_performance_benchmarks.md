# APPENDIX C: PERFORMANCE BENCHMARKS AND TARGETS

## C.1 Target Performance Metrics

**Status:** CRITICAL - System must meet these benchmarks for production readiness

### C.1.1 Core Performance Targets

| Metric | Target | Critical? | Measurement Method |
|--------|--------|-----------|-------------------|
| Physics step time | <1ms | YES | Single propagation cycle (sparse 27³ grid) |
| Wave propagation (27³) | <0.5ms | YES | 19,683 nodes, 100 cycles |
| Wave propagation (81³) | <5ms | NO | 531,441 nodes, 100 cycles |
| Memory retrieval (resonance) | <10ms | YES | Query → peak detection |
| Query end-to-end latency | <100ms | NO | CLI → response (cache hit) |
| Neuroplastic update | <1ms | YES | Single metric tensor update |
| Hilbert encoding | <0.1ms | YES | 9D coord → 1D index |
| Nap duration | <5s | NO | Full DMC checkpoint save |
| GGUF export | <60s | NO | Complete state → .gguf file |
| ZeroMQ message latency | <0.5ms | YES | IPC socket round-trip |
| Emitter DDS tick | <0.01ms | YES | 8 emitters, single tick |

### C.1.2 Scaling Behavior

Expected performance with increasing grid size:

| Grid Size | Total Nodes | Active Nodes (sparse) | Step Time | Memory Usage | Energy/Step |
|-----------|-------------|----------------------|-----------|--------------|-------------|
| 27³ | 19,683 | ~2,000 | 0.5ms | 5MB | 0.8ms |
| 54³ | 157,464 | ~15,000 | 3ms | 40MB | 5ms |
| 81³ | 531,441 | ~50,000 | 8ms | 135MB | 15ms |
| 162³ | 4,251,528 | ~400,000 | 60ms | 1GB | 120ms |

**Sparse Grid Assumption:** Only 10% of nodes are active (non-zero amplitude)

### C.1.3 Throughput Targets

| Operation | Target Throughput | Notes |
|-----------|------------------|-------|
| Query processing | 10 queries/sec | End-to-end with external tools |
| Cache-hit queries | 100 queries/sec | Memory retrieval only |
| Waveform injections | 1000 injections/sec | Physics engine ingestion rate |
| Training samples | 100 samples/sec | Mamba/Transformer combined |
| File ingestion | 10 files/sec | Text files, ~10KB each |
| Neurogenesis events | 1 event/sec | Grid expansion rate limit |

---

## C.2 Benchmark Suite

### C.2.1 Physics Engine Benchmarks

**File:** `tests/benchmarks/bench_propagation.cpp`

```cpp
#include <benchmark/benchmark.h>
#include "nikola/physics/torus_manifold.hpp"

static void BM_WavePropagation_27x27x27(benchmark::State& state) {
    TorusManifold torus({27, 27, 27, 9, 9, 9, 27, 27, 9});

    // Inject initial wave
    torus.inject_wave({13, 13, 13, 4, 4, 4, 13, 13, 4},
                     std::complex<double>(1.0, 0.0));

    for (auto _ : state) {
        torus.propagate(0.01);  // Single step
    }

    state.SetItemsProcessed(state.iterations() * torus.active_node_count());
}
BENCHMARK(BM_WavePropagation_27x27x27);

static void BM_WavePropagation_81x81x81(benchmark::State& state) {
    TorusManifold torus({81, 81, 81, 27, 27, 27, 81, 81, 9});

    torus.inject_wave({40, 40, 40, 13, 13, 13, 40, 40, 4},
                     std::complex<double>(1.0, 0.0));

    for (auto _ : state) {
        torus.propagate(0.01);
    }

    state.SetItemsProcessed(state.iterations() * torus.active_node_count());
}
BENCHMARK(BM_WavePropagation_81x81x81);

BENCHMARK_MAIN();
```

**Expected Output:**

```
--------------------------------------------------------------
Benchmark                              Time             CPU
--------------------------------------------------------------
BM_WavePropagation_27x27x27       482 us          481 us
BM_WavePropagation_81x81x81      7.8 ms          7.8 ms
```

### C.2.2 Hilbert Curve Benchmarks

**File:** `tests/benchmarks/bench_hilbert.cpp`

```cpp
static void BM_HilbertEncode(benchmark::State& state) {
    Coord9D coord{40, 40, 40, 13, 13, 13, 40, 40, 4};

    for (auto _ : state) {
        uint64_t index = HilbertMapper::encode(coord, 10);
        benchmark::DoNotOptimize(index);
    }
}
BENCHMARK(BM_HilbertEncode);

static void BM_HilbertDecode(benchmark::State& state) {
    uint64_t index = 123456789012345ULL;

    for (auto _ : state) {
        Coord9D coord = HilbertMapper::decode(index, 10);
        benchmark::DoNotOptimize(coord);
    }
}
BENCHMARK(BM_HilbertDecode);
```

**Expected Output:**

```
--------------------------------------------------------------
Benchmark                              Time             CPU
--------------------------------------------------------------
BM_HilbertEncode                   85 ns           85 ns
BM_HilbertDecode                   92 ns           92 ns
```

### C.2.3 Memory Operations Benchmarks

```cpp
static void BM_ResonancePeakDetection(benchmark::State& state) {
    TorusManifold torus({27, 27, 27, 9, 9, 9, 27, 27, 9});

    // Inject test pattern
    torus.inject_wave({13, 13, 13, 4, 4, 4, 13, 13, 4},
                     std::complex<double>(1.0, 0.0));
    torus.propagate_n_steps(100);

    for (auto _ : state) {
        auto peak = torus.find_resonance_peak();
        benchmark::DoNotOptimize(peak);
    }
}
BENCHMARK(BM_ResonancePeakDetection);
```

**Expected Output:**

```
BM_ResonancePeakDetection           8.5 ms          8.5 ms
```

### C.2.4 Serialization Benchmarks

```cpp
static void BM_ProtobufSerialize(benchmark::State& state) {
    NeuralSpike spike;
    spike.set_request_id("550e8400-e29b-41d4-a716-446655440000");
    spike.set_timestamp(1701234567890);
    spike.set_sender(ComponentID::ORCHESTRATOR);
    spike.set_recipient(ComponentID::CLI_CONTROLLER);
    spike.set_text_data("What is the golden ratio?");

    for (auto _ : state) {
        std::string serialized;
        spike.SerializeToString(&serialized);
        benchmark::DoNotOptimize(serialized);
    }
}
BENCHMARK(BM_ProtobufSerialize);

static void BM_ProtobufDeserialize(benchmark::State& state) {
    NeuralSpike spike;
    spike.set_request_id("test");
    spike.set_text_data("Test data");

    std::string serialized;
    spike.SerializeToString(&serialized);

    for (auto _ : state) {
        NeuralSpike deserialized;
        deserialized.ParseFromString(serialized);
        benchmark::DoNotOptimize(deserialized);
    }
}
BENCHMARK(BM_ProtobufDeserialize);
```

**Expected Output:**

```
BM_ProtobufSerialize                120 ns          120 ns
BM_ProtobufDeserialize              150 ns          150 ns
```

---

## C.3 Profiling Tools and Commands

### C.3.1 CPU Profiling with perf

```bash
# Record performance data
sudo perf record -g ./build/tests/benchmarks/bench_propagation

# Analyze results
sudo perf report

# Hotspot visualization
sudo perf report --stdio | head -50
```

**Expected Hotspots:**
1. `TorusManifold::propagate()` - 60-70% CPU time
2. `EmitterArray::tick()` - 10-15%
3. `std::complex<double>::operator*` - 5-10%

### C.3.2 Memory Profiling with Valgrind

```bash
# Track heap allocations
valgrind --tool=massif --massif-out-file=massif.out \
    ./build/bin/twi-ctl query "test"

# Visualize memory usage
ms_print massif.out

# Check for leaks
valgrind --leak-check=full --show-leak-kinds=all \
    ./build/bin/twi-ctl status
```

**Expected Memory Profile:**
- Peak heap: 135MB (81³ grid)
- Total allocations: ~500K
- Leaked bytes: 0 (no leaks)

### C.3.3 GPU Profiling with nvprof

```bash
# Profile CUDA kernels
nvprof ./build/bin/twi-ctl query "test"

# Detailed metrics
nvprof --metrics achieved_occupancy,gld_efficiency \
    ./build/tests/unit/test_wave_cuda
```

**Expected CUDA Metrics:**
- Kernel: `wave_propagate_kernel`
- Occupancy: >75%
- Global load efficiency: >85%
- Execution time: <2ms (81³ grid)

### C.3.4 Cache Analysis with perf

```bash
# Cache miss rates
perf stat -e cache-references,cache-misses \
    ./build/tests/benchmarks/bench_propagation

# Output:
# 12,456,789 cache-references
#    234,567 cache-misses              # 1.88% of all cache refs
```

**Target Cache Miss Rate:** <3%

---

## C.4 Optimization Checklist

### C.4.1 Compiler Optimizations

**CMakeLists.txt Flags:**

```cmake
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Optional aggressive optimizations
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
    -ffast-math \
    -funroll-loops \
    -finline-functions \
    -flto")  # Link-Time Optimization
```

**AVX-512 Specific:**

```cmake
if(COMPILER_SUPPORTS_AVX512)
    add_compile_options(-mavx512f -mavx512cd -mavx512bw -mavx512dq)
    add_definitions(-DUSE_AVX512)
endif()
```

### C.4.2 Critical Loop Optimizations

**Wave Propagation Loop:**

```cpp
// ✓ GOOD: Cache-friendly Hilbert order traversal
for (auto [hilbert_idx, node_ptr] : sorted_nodes) {
    propagate_node(node_ptr);
}

// ✗ BAD: Random memory access
for (auto& [coord, node] : grid) {
    propagate_node(&node);  // Poor cache locality
}
```

**Vectorization:**

```cpp
// ✓ GOOD: Vectorizable loop (8 emitters at once with AVX-512)
#pragma omp simd
for (int i = 0; i < 8; ++i) {
    phases[i] += tuning_words[i];
    outputs[i] = sine_lut[phases[i] >> 18];  // Top 14 bits
}

// ✗ BAD: Not vectorizable (function calls in loop)
for (int i = 0; i < 8; ++i) {
    outputs[i] = std::sin(2 * M_PI * phases[i] / (1ULL << 32));
}
```

### C.4.3 Memory Layout

**Structure-of-Arrays (SoA) for SIMD:**

```cpp
// ✓ GOOD: SoA layout (vectorizable)
struct TorusGrid {
    std::vector<std::complex<float>> wavefunctions;  // Contiguous
    std::vector<float> resonances;                   // Contiguous
    std::vector<float> states;                       // Contiguous
};

// ✗ BAD: Array-of-Structures (AoS) - poor SIMD
struct TorusNode {
    std::complex<float> wavefunction;
    float resonance;
    float state;
};
std::vector<TorusNode> nodes;  // Interleaved data
```

---

## C.5 Performance Regression Testing

### C.5.1 Automated Benchmark CI

**GitHub Actions Workflow:**

```yaml
name: Performance Benchmarks

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Build benchmarks
        run: |
          cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON .
          make bench_propagation bench_hilbert

      - name: Run benchmarks
        run: |
          ./build/tests/benchmarks/bench_propagation --benchmark_format=json \
            > benchmark_results.json

      - name: Check for regressions
        run: |
          python3 scripts/check_performance_regression.py \
            --baseline=benchmarks/baseline.json \
            --current=benchmark_results.json \
            --threshold=10  # 10% regression tolerance
```

### C.5.2 Baseline Results

**File:** `benchmarks/baseline.json`

```json
{
  "context": {
    "date": "2024-12-01",
    "host_name": "benchmark-server",
    "executable": "./bench_propagation",
    "num_cpus": 64,
    "cpu_scaling_enabled": false
  },
  "benchmarks": [
    {
      "name": "BM_WavePropagation_27x27x27",
      "real_time": 481.2,
      "cpu_time": 481.0,
      "time_unit": "us",
      "items_per_second": 4152834
    },
    {
      "name": "BM_WavePropagation_81x81x81",
      "real_time": 7812.5,
      "cpu_time": 7810.3,
      "time_unit": "us",
      "items_per_second": 68042
    }
  ]
}
```

---

## C.6 Production Performance Monitoring

### C.6.1 Metrics to Track

```cpp
struct PerformanceMetrics {
    double avg_physics_step_ms;
    double avg_query_latency_ms;
    double avg_resonance_detection_ms;
    int64_t queries_per_second;
    int64_t active_node_count;
    double memory_usage_mb;
    double gpu_utilization_percent;
};
```

### C.6.2 CLI Performance Query

```bash
# Get detailed performance metrics
twi-ctl metrics --json

# Output:
{
  "physics": {
    "avg_step_ms": 0.48,
    "peak_step_ms": 1.2,
    "steps_per_second": 2083
  },
  "query": {
    "avg_latency_ms": 87,
    "p50_latency_ms": 45,
    "p95_latency_ms": 180,
    "p99_latency_ms": 320
  },
  "memory": {
    "active_nodes": 2187,
    "total_memory_mb": 42,
    "gpu_memory_mb": 128
  }
}
```

---

**Cross-References:**
- See Section 4 for Physics Engine implementation
- See Section 9.4 for build system configuration
- See Appendix D for hardware optimization guidelines
- See Appendix E for troubleshooting slow performance

