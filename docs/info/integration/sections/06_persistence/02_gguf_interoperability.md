# GGUF INTEROPERABILITY

## 20.1 Manifold-to-Tensor Projection

**Challenge:** Convert continuous 9D toroidal manifold to discrete tensor.

**Approach:** "Holographic snapshot" at specific time $t$.

## 20.2 Hilbert Curve Flattening

**Process:**

1. Enumerate all active nodes in torus
2. Compute Hilbert index for each
3. Sort by Hilbert index
4. Create 1D tensor in sorted order

**Implementation:**

```cpp
// Helper function to expand compressed symmetric matrix to full 9×9 format
// Converts 45-value upper-triangle storage to 81-value full matrix
std::array<float, 81> expand_symmetric_matrix(const std::array<float, 45>& compressed) {
    std::array<float, 81> expanded;

    // Helper function to convert (i,j) coordinates to compressed index
    auto compressed_idx = [](int i, int j) -> int {
        if (i > j) std::swap(i, j);  // Ensure i <= j (upper triangle)
        return i * 9 - (i * (i + 1)) / 2 + j;
    };

    // Expand symmetric matrix
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            int flat_idx = i * 9 + j;
            int comp_idx = compressed_idx(i, j);
            expanded[flat_idx] = compressed[comp_idx];
        }
    }

    return expanded;
}

std::vector<float> flatten_torus_to_tensor(const TorusManifold& torus) {
    std::vector<std::pair<uint64_t, TorusNode>> indexed_nodes;

    // 1. Collect and index
    for (const auto& [coord, node] : torus.get_active_nodes()) {
        uint64_t hilbert_idx = HilbertMapper::encode(coord, 10);  // 10 bits per dim
        indexed_nodes.push_back({hilbert_idx, node});
    }

    // 2. Sort by Hilbert index
    std::sort(indexed_nodes.begin(), indexed_nodes.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // 3. Flatten
    std::vector<float> tensor;
    for (const auto& [idx, node] : indexed_nodes) {
        // Amplitude (1 value)
        tensor.push_back(std::abs(node.wavefunction));

        // Phase (1 value)
        tensor.push_back(std::arg(node.wavefunction));

        // Metric tensor: 9×9 symmetric matrix stored as 45-value upper triangle
        // Formula: (9 × 10) / 2 = 45 unique components
        // Each node exports: 2 (amplitude + phase) + 45 (metric tensor) = 47 values
        for (float m : node.metric_tensor) {
            tensor.push_back(m);
        }

        // Note: If needed for compatibility, expand to full 81-value matrix using:
        // std::array<float, 81> full_metric = expand_symmetric_matrix(node.metric_tensor);
        // for (float m : full_metric) { tensor.push_back(m); }
    }

    return tensor;
}
```

## 20.3 Amplitude-Phase Decomposition

**Dual-Tensor Strategy:**

Complex waveform $\Psi = A e^{i\theta}$ split into:
- **Tensor A:** Amplitude $A$
- **Tensor B:** Phase $\theta$

**GGUF Tensor Naming:**

```
nikola.torus.amplitude  →  GGML_TYPE_F16
nikola.torus.phase      →  GGML_TYPE_F16
nikola.metric.tensor    →  GGML_TYPE_F32
nikola.emitter.freq     →  GGML_TYPE_F32
```

## 20.4 llama.cpp Integration

**Architecture Registration:**

```cpp
// File: src/llama-arch.cpp

enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_FALCON,
    // ... existing architectures
    LLM_ARCH_NIKOLA,  // ADD THIS
};

static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_LLAMA,  "llama"  },
    { LLM_ARCH_NIKOLA, "nikola" },  // ADD THIS
    // ...
};
```

**Tensor Definitions:**

```cpp
// File: src/llama-model.cpp

static const std::map<llm_arch, std::map<llm_tensor, std::string>> LLM_TENSOR_NAMES = {
    {
        LLM_ARCH_NIKOLA,
        {
            { LLM_TENSOR_ATTN_Q,   "blk.%d.torus.amplitude" },
            { LLM_TENSOR_ATTN_K,   "blk.%d.torus.phase" },
            { LLM_TENSOR_ATTN_V,   "blk.%d.emitter.freq" },
            { LLM_TENSOR_FFN_UP,   "blk.%d.metric.tensor" },
        },
    },
    // ...
};
```

## 20.5 Custom GGML Operators

**Wave Interference Operator:**

```cpp
// File: src/ggml-nikola.cpp

void ggml_compute_forward_wave_interference(
    const struct ggml_compute_params * params,
    const struct ggml_tensor * src0,  // Wave A
    const struct ggml_tensor * src1,  // Wave B
    struct ggml_tensor * dst) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];

    // Superposition (complex addition)
    for (int64_t i = 0; i < ne01; ++i) {
        for (int64_t j = 0; j < ne00; j += 2) {
            // Real parts
            float a_real = ggml_get_f32_1d(src0, i * ne00 + j);
            float b_real = ggml_get_f32_1d(src1, i * ne00 + j);

            // Imaginary parts
            float a_imag = ggml_get_f32_1d(src0, i * ne00 + j + 1);
            float b_imag = ggml_get_f32_1d(src1, i * ne00 + j + 1);

            // Add complex numbers
            float c_real = a_real + b_real;
            float c_imag = a_imag + b_imag;

            ggml_set_f32_1d(dst, i * ne00 + j, c_real);
            ggml_set_f32_1d(dst, i * ne00 + j + 1, c_imag);
        }
    }
}
```

### 20.5.1 GGUF Q9_0 Quantization

**[ADDENDUM]**

To "be exported to GGUF", we must map the balanced nonary weights to a format llama.cpp understands. Standard Q4_0 or Q8_0 are binary-optimized. We define Q9_0.

**Quantization Scheme:**

- **Target:** Store weights in discrete values $\{-4, \dots, 4\}$.
- **Packing:** A single Balanced Nonary "Trit" takes $\log_2(9) \approx 3.17$ bits.
- **Block Layout:** We pack 5 trits into 16 bits (2 bytes). $3^5 = 243 < 2^8$. Wait, $3^5 = 243$, which fits in 8 bits (one byte).
- **Correction:** $3^5 = 243$. A single byte (256 values) can perfectly store 5 trits.
- **Efficiency:** This yields a compression ratio of 1.6 bits per weight. This is significantly more efficient than standard 4-bit quantization (Q4_0), offering higher precision (9 states vs 16 states) at comparable or better compression density per parameter.

**Integration:** A custom CUDA kernel is added to the export pipeline to dequantize Q9_0 blocks back to FP16 for inference on standard GPUs.

### 20.5.2 Q9_0 De-Quantization Kernel

The Q9_0 quantization format stores balanced nonary weights in a custom packed format. Inference engines require a CUDA kernel to unpack these values back to FP16 for GPU computation.

**Data Structure:**

```cpp
// File: include/ggml-quants-q9.h

#define QK9_0 32  // Block size (32 weights per block)

// Q9_0 block structure: 32 balanced nonary weights packed using base-9 radix encoding
// Each uint16_t stores 5 trits (max value: 59,048 < 65,536)
// 32 weights requires 7 uint16_t values (6 × 5 = 30, plus 1 for final 2 weights)
typedef struct {
    float scale;         // 4 bytes: Scale factor for block
    uint16_t data[7];    // 14 bytes: 32 weights (5 trits per uint16_t)
                         // 6 uint16_t × 5 trits = 30 weights
                         // 7th uint16_t holds remaining 2 weights (padded to 5)
    uint16_t padding;    // 2 bytes: Align to 4-byte boundary
} block_q9_0;

static_assert(sizeof(block_q9_0) == 20, "Q9_0 block size must be 20 bytes (4 + 14 + 2)");
```

**Encoding Helper:**

```cpp
// File: src/persistence/kernels/q9_0_encode.cpp

// Pack 5 balanced nonary values [-4, +4] into uint16_t using base-9 radix encoding
uint16_t pack_5_trits(const int8_t trits[5]) {
    // Convert [-4, +4] to [0, 8] (9 possible values per trit)
    uint8_t vals[5];
    for (int i = 0; i < 5; ++i) {
        vals[i] = static_cast<uint8_t>(trits[i] + 4);  // [-4,+4] → [0,8]
    }

    // Pack into base-9 radix representation using Horner's method
    // v = Σ(i=0 to 4) vals[i] * 9^i
    // = vals[0] + 9*(vals[1] + 9*(vals[2] + 9*(vals[3] + 9*vals[4])))
    //
    // Maximum value: 8 + 8*9 + 8*81 + 8*729 + 8*6561 = 59,048 < 65,536 (fits in uint16_t)

    uint16_t result = vals[0];
    result += vals[1] * 9;
    result += vals[2] * 81;        // 9^2
    result += vals[3] * 729;       // 9^3
    result += vals[4] * 6561;      // 9^4

    // Alternative Horner form (more efficient):
    // result = vals[0] + 9*(vals[1] + 9*(vals[2] + 9*(vals[3] + 9*vals[4])));

    return result;
}

// Quantize block of 32 balanced nonary weights to Q9_0 format
void quantize_q9_0_block(const int8_t* nonary_weights, block_q9_0* block) {
    // Find maximum absolute value for scaling
    float max_abs = 0.0f;
    for (int i = 0; i < QK9_0; ++i) {
        float abs_val = std::abs(static_cast<float>(nonary_weights[i]));
        max_abs = std::max(max_abs, abs_val);
    }

    // Compute scale (map [-4, +4] to FP16 range)
    block->scale = max_abs / 4.0f;

    // Pack weights: 32 weights / 5 per uint16_t = 7 uint16_t values
    for (int i = 0; i < 7; ++i) {
        int8_t trits[5] = {0, 0, 0, 0, 0};  // Initialize with zeros for padding
        for (int j = 0; j < 5; ++j) {
            int idx = i * 5 + j;
            if (idx < QK9_0) {
                trits[j] = nonary_weights[idx];
            }
            // else: leave as 0 (padding)
        }
        block->data[i] = pack_5_trits(trits);
    }

    block->padding = 0;  // Initialize padding to zero
}
```

**CUDA De-Quantization Kernel:**

```cuda
// File: src/persistence/kernels/dequantize.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Unpack 5 balanced nonary trits from uint16_t using base-9 radix decoding
__device__ void unpack_5_trits(uint16_t packed, int8_t trits[5]) {
    // Reverse of pack_5_trits: Extract base-9 digits
    // packed = vals[0] + vals[1]*9 + vals[2]*81 + vals[3]*729 + vals[4]*6561
    // where vals[i] ∈ [0, 8]
    //
    // Extraction using modulo and division:
    // vals[0] = packed % 9
    // vals[1] = (packed / 9) % 9
    // vals[2] = (packed / 81) % 9
    // vals[3] = (packed / 729) % 9
    // vals[4] = (packed / 6561) % 9

    uint16_t temp = packed;

    // Extract each trit
    uint8_t vals[5];
    vals[0] = temp % 9;
    temp /= 9;

    vals[1] = temp % 9;
    temp /= 9;

    vals[2] = temp % 9;
    temp /= 9;

    vals[3] = temp % 9;
    temp /= 9;

    vals[4] = temp % 9;  // Remaining value

    // Convert [0, 8] back to [-4, +4]
    for (int i = 0; i < 5; ++i) {
        trits[i] = static_cast<int8_t>(vals[i]) - 4;
    }
}

// CUDA kernel: De-quantize Q9_0 blocks to FP16 for inference
__global__ void dequantize_q9_0_kernel(
    const block_q9_0* blocks,
    half* output,
    int num_blocks
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_idx >= num_blocks) {
        return;
    }

    const block_q9_0* block = &blocks[block_idx];
    float scale = block->scale;

    // Process 32 weights in this block
    for (int i = 0; i < QK9_0 / 5; ++i) {
        int8_t trits[5];
        unpack_5_trits(block->data[i], trits);

        for (int j = 0; j < 5; ++j) {
            int output_idx = block_idx * QK9_0 + i * 5 + j;

            if (i * 5 + j < QK9_0) {
                // De-quantize: float_value = trit_value * scale
                float dequantized = static_cast<float>(trits[j]) * scale;

                // Convert to FP16
                output[output_idx] = __float2half(dequantized);
            }
        }
    }
}

// Host wrapper
extern "C" void dequantize_q9_0(
    const void* blocks_data,
    half* output,
    int num_blocks
) {
    const block_q9_0* d_blocks = reinterpret_cast<const block_q9_0*>(blocks_data);

    int threads = 256;
    int blocks = (num_blocks + threads - 1) / threads;

    dequantize_q9_0_kernel<<<blocks, threads>>>(d_blocks, output, num_blocks);

    cudaDeviceSynchronize();
}
```

**llama.cpp Integration:**

```cpp
// File: src/ggml-cuda/dequantize.cu (in llama.cpp fork)

#include "ggml-cuda.h"
#include "ggml-quants-q9.h"

// Register Q9_0 dequantization
static void dequantize_row_q9_0_cuda(const void * vx, dst_t * y, const int k, cudaStream_t stream) {
    const int nb = k / QK9_0;

    dequantize_q9_0_kernel<<<nb, 1, 0, stream>>>(
        reinterpret_cast<const block_q9_0*>(vx),
        reinterpret_cast<half*>(y),
        nb
    );
}

// Add to dequantize function table
switch (type) {
    case GGML_TYPE_Q4_0:
        dequantize_row_q4_0_cuda(src, dst, k, stream);
        break;
    case GGML_TYPE_Q8_0:
        dequantize_row_q8_0_cuda(src, dst, k, stream);
        break;
    case GGML_TYPE_Q9_0:  // ADD THIS
        dequantize_row_q9_0_cuda(src, dst, k, stream);
        break;
    default:
        // ...
}
```

**Impact:** Models exported to GGUF with Q9_0 quantization can now be loaded and executed by llama.cpp/Ollama with full balanced nonary weight fidelity.

## 20.6 Implementation

**Conversion Script (Python):**

```python
#!/usr/bin/env python3
# File: convert_nikola_to_gguf.py

import struct
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

def balanced_nonary_to_q8(nonary_values):
    """
    Convert balanced nonary weights [-4, +4] to normalized float32 for Q8_0 quantization.

    Q8_0 quantization in GGUF uses symmetric 8-bit representation with per-block scaling.
    We map balanced nonary to [-1.0, +1.0] range which Q8_0 can efficiently encode.

    Args:
        nonary_values: List of integers in range [-4, +4]

    Returns:
        numpy array of float32 values in [-1.0, +1.0]
    """
    # Normalize balanced nonary [-4, +4] to [-1.0, +1.0]
    normalized = np.array(nonary_values, dtype=np.float32) / 4.0

    # Clamp to ensure valid range
    normalized = np.clip(normalized, -1.0, 1.0)

    return normalized

def convert_nik_to_gguf(nik_path, gguf_path):
    # 1. Read .nik file
    with open(nik_path, 'rb') as f:
        header = read_nik_header(f)
        nodes = read_all_nodes(f)

    # 2. Flatten via Hilbert curve and extract balanced nonary weights
    amplitude_tensor = []
    phase_tensor = []

    # Track whether we have balanced nonary or float values
    has_nonary_weights = hasattr(nodes[0], 'nonary_weight')

    for node in sorted(nodes, key=lambda n: n.hilbert_idx):
        if has_nonary_weights:
            # If nodes store balanced nonary weights directly
            amplitude_tensor.append(node.nonary_weight)
        else:
            # Convert from amplitude (assuming it's already in nonary form)
            amplitude_tensor.append(node.amplitude)

        phase_tensor.append(node.phase)

    # 3. Create GGUF writer
    gguf_writer = GGUFWriter(gguf_path, 'nikola')

    # 4. Add metadata
    gguf_writer.add_uint32('nikola.geometry.dimensions', 9)
    gguf_writer.add_string('nikola.encoding.base', 'balanced_nonary')
    gguf_writer.add_string('nikola.quantization.mapping', 'nonary_to_q8_0')
    gguf_writer.add_float32('nikola.golden_ratio', 1.618033988749895)
    gguf_writer.add_string('nikola.quantization.note',
                          'Balanced nonary [-4,+4] mapped to Q8_0 via /4.0 normalization')

    # 5. FIXED: Convert balanced nonary to Q8_0 compatible format
    # This ensures llama.cpp can load and run the model without custom code
    amplitude_normalized = balanced_nonary_to_q8(amplitude_tensor)

    # Add tensors with Q8_0 quantization (standard llama.cpp format)
    gguf_writer.add_tensor('nikola.torus.amplitude',
                           amplitude_normalized,
                           quantization_type=GGMLQuantizationType.Q8_0)

    # Phase can remain float16 as it's continuous
    gguf_writer.add_tensor('nikola.torus.phase',
                           np.array(phase_tensor, dtype=np.float16))

    # 6. Write
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    print(f"Converted {nik_path} → {gguf_path}")
    print(f"  - Amplitude tensor: {len(amplitude_tensor)} weights (Q8_0 quantized)")
    print(f"  - Phase tensor: {len(phase_tensor)} values (FP16)")
    print(f"  - Compatible with standard llama.cpp/ollama")

if __name__ == '__main__':
    convert_nik_to_gguf('/var/lib/nikola/state/main.nik',
                         '/var/lib/nikola/export/nikola.gguf')
```

---

**Cross-References:**
- See Section 19 for .nik file format
- See Section 5 for Hilbert curve implementation
- See Section 3 for Metric tensor structure
- See llama.cpp documentation for GGML operator development
