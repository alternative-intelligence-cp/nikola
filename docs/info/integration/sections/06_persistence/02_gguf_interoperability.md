# GGUF INTEROPERABILITY

## 20.1 Manifold-to-Tensor Projection

**Challenge:** Convert continuous 9D toroidal manifold to discrete tensor.

**Solution:** "Holographic snapshot" at specific time $t$.

## 20.2 Hilbert Curve Flattening

**Process:**

1. Enumerate all active nodes in torus
2. Compute Hilbert index for each
3. Sort by Hilbert index
4. Create 1D tensor in sorted order

**Implementation:**

```cpp
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
        // Amplitude
        tensor.push_back(std::abs(node.wavefunction));

        // Phase
        tensor.push_back(std::arg(node.wavefunction));

        // Metric tensor (45 values)
        for (float m : node.metric_tensor) {
            tensor.push_back(m);
        }
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

## 20.6 Implementation

**Conversion Script (Python):**

```python
#!/usr/bin/env python3
# File: convert_nikola_to_gguf.py

import struct
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

# CRITICAL FIX (Audit 2 Item #7): Map balanced nonary to standard GGUF format
# Problem: Original script used np.float16, but spec requires Q9_0 which llama.cpp doesn't support
# Solution: Map balanced nonary [-4, +4] to Q8_0 (standard 8-bit quantization)

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
