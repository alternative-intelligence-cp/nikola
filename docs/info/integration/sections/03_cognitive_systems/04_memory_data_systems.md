# MEMORY AND DATA SYSTEMS

## 9.1 Nonary Embedder

The **Custom Nonary Embedder** converts text to waveforms.

### Pipeline

1. **Tokenization:** Byte-Pair Encoding (BPE)
2. **Vectorization:** Lightweight transformer (e.g., distilBERT-tiny)
3. **Quantization:** Map to balanced nonary
4. **Holographic Encoding:** Create interference pattern

### Implementation

**PRODUCTION: TinyTransformer with ONNX Runtime**

The encoder uses a distilled BERT-Tiny model (4-layer, 128-dim) loaded via ONNX Runtime C++ API for efficient inference.

```cpp
// File: include/nikola/reasoning/tiny_transformer.hpp
#pragma once

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

namespace nikola::reasoning {

class TinyTransformer {
private:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info;
    Ort::AllocatorWithDefaultOptions allocator;

    // Model metadata
    std::vector<const char*> input_names{"input_ids", "attention_mask"};
    std::vector<const char*> output_names{"last_hidden_state"};

    // Model dimensions (BERT-Tiny: 4 layers, 128 hidden, 2 attn heads, 512 seq len)
    static constexpr int64_t HIDDEN_DIM = 128;
    static constexpr int64_t MAX_SEQ_LEN = 512;

public:
    TinyTransformer(const std::string& model_path)
        : memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

        // Initialize ONNX Runtime environment
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NikolaTinyTransformer");

        // Configure session options for CPU inference
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);  // Parallel execution within ops
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Load ONNX model
        session = std::make_unique<Ort::Session>(*env, model_path.c_str(), session_options);

        std::cout << "[TinyTransformer] Loaded ONNX model from " << model_path << std::endl;
        std::cout << "[TinyTransformer] Architecture: BERT-Tiny (4L/128H/2A)" << std::endl;
    }

    // Forward pass: tokens → 128-dim embeddings
    std::vector<float> forward(const std::vector<int64_t>& token_ids) {
        // Prepare input tensors
        size_t seq_len = std::min(token_ids.size(), static_cast<size_t>(MAX_SEQ_LEN));

        // Input IDs tensor [batch_size=1, seq_len]
        std::vector<int64_t> input_ids(seq_len);
        std::copy(token_ids.begin(), token_ids.begin() + seq_len, input_ids.begin());

        // Attention mask tensor [batch_size=1, seq_len] (all 1s for valid tokens)
        std::vector<int64_t> attention_mask(seq_len, 1);

        // Create input tensors
        std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(seq_len)};

        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids.data(), input_ids.size(),
            input_shape.data(), input_shape.size()
        );

        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, attention_mask.data(), attention_mask.size(),
            input_shape.data(), input_shape.size()
        );

        // Run inference
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_ids_tensor));
        input_tensors.push_back(std::move(attention_mask_tensor));

        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), input_tensors.size(),
            output_names.data(), output_names.size()
        );

        // Extract output: [batch_size=1, seq_len, hidden_dim=128]
        // Use [CLS] token embedding (first token) as sentence representation
        float* output_data = output_tensors[0].GetTensorMutableData<float>();

        // Copy [CLS] embedding (first HIDDEN_DIM floats)
        std::vector<float> cls_embedding(output_data, output_data + HIDDEN_DIM);

        return cls_embedding;
    }
};

} // namespace nikola::reasoning
```

**NonaryEmbedder with TinyTransformer Integration:**

```cpp
class NonaryEmbedder {
    BPETokenizer tokenizer;
    nikola::reasoning::TinyTransformer encoder;

public:
    NonaryEmbedder(const std::string& tokenizer_path, const std::string& model_path)
        : tokenizer(tokenizer_path),
          encoder(model_path) {
        std::cout << "[NonaryEmbedder] Initialized with ONNX TinyTransformer" << std::endl;
    }

    std::vector<Nit> embed(const std::string& text) {
        // 1. Tokenize text to BPE token IDs
        auto tokens = tokenizer.encode(text);

        // 2. Vectorize using TinyTransformer (128-dim embedding)
        auto vector = encoder.forward(tokens);

        // 3. Quantize to balanced nonary (128 floats → 128 Nits)
        std::vector<Nit> nonary_vector;
        nonary_vector.reserve(vector.size());

        for (float val : vector) {
            nonary_vector.push_back(quantize_to_nit(val));
        }

        return nonary_vector;
    }

private:
    Nit quantize_to_nit(float val) {
        // Normalize with tanh to [-1, 1]
        float normalized = std::tanh(val);

        // Scale to [-4, 4] for balanced nonary
        int quantized = static_cast<int>(std::round(normalized * 4.0));

        return static_cast<Nit>(std::clamp(quantized, -4, 4));
    }
};
```

### Holographic Multiplexing

Chunk vector into groups of 9, each creating a "chord" across emitters:

```cpp
std::complex<double> create_chord(const std::array<Nit, 9>& chunk,
                                   const EmitterArray& emitters,
                                   double time) {
    std::complex<double> sum = 0.0;

    for (int i = 0; i < 9; ++i) {
        double amplitude = static_cast<double>(chunk[i]);
        double freq = emitters.get_frequency(i);
        double phase = emitters.get_phase(i);

        sum += amplitude * std::exp(std::complex<double>(0, freq * time + phase));
    }

    return sum;
}
```

## 9.2 High-Performance Database

**Technology:** LMDB (Lightning Memory-Mapped Database)

### Why LMDB?

- Zero-copy reads
- Memory-mapped for speed
- ACID transactions
- Compact storage

### Schema

- **Key:** Hilbert index (uint64_t)
- **Value:** Serialized TorusNode (Protocol Buffer)

### Protocol Buffer Definition

```protobuf
syntax = "proto3";

message TorusNodeProto {
    double wavefunction_real = 1;
    double wavefunction_imag = 2;
    repeated float metric_tensor = 3;  // 45 elements
    repeated float ssm_state = 4;      // 8 elements
    int32 nonary_value = 5;
    float resonance_r = 6;
    float state_s = 7;
}
```

### Database Operations

```cpp
class TorusDatabase {
    lmdb::env env;
    lmdb::dbi dbi;

public:
    TorusDatabase(const std::string& path) {
        env = lmdb::env::create();
        env.set_mapsize(100UL * 1024UL * 1024UL * 1024UL);  // 100GB
        env.open(path.c_str());

        auto txn = lmdb::txn::begin(env);
        dbi = lmdb::dbi::open(txn, nullptr);
        txn.commit();
    }

    void store_node(uint64_t hilbert_idx, const TorusNode& node) {
        // Serialize to protobuf
        TorusNodeProto proto = serialize(node);
        std::string data;
        proto.SerializeToString(&data);

        // Write to LMDB
        auto txn = lmdb::txn::begin(env);
        lmdb::dbi_put(txn, dbi,
                      lmdb::val(&hilbert_idx, sizeof(hilbert_idx)),
                      lmdb::val(data));
        txn.commit();
    }

    std::optional<TorusNode> load_node(uint64_t hilbert_idx) {
        auto txn = lmdb::txn::begin(env, nullptr, MDB_RDONLY);
        lmdb::val key(&hilbert_idx, sizeof(hilbert_idx));
        lmdb::val data;

        if (!lmdb::dbi_get(txn, dbi, key, data)) {
            return std::nullopt;  // Not found
        }

        // Deserialize
        TorusNodeProto proto;
        proto.ParseFromArray(data.data(), data.size());
        return deserialize(proto);
    }
};
```

## 9.3 Search-Retrieve-Store Loop

### Algorithm

```
1. Query arrives (text)
2. Embed query → nonary waveform
3. Compute injection coordinates (hash-based or learned)
4. Inject waveform into torus
5. Run wave propagation (multiple cycles)
6. Monitor for resonance peaks (high amplitude regions)
7. IF resonance > threshold:
       Retrieve data at peak location
       Return to user
   ELSE:
       Dispatch to external tools (Tavily/Firecrawl/Gemini)
8. External tool returns data
9. Embed returned data → waveform
10. Store in torus at new coordinates
11. Trigger neuroplastic reinforcement (increase metric in that region)
12. Return data to user
```

### Implementation

```cpp
class Orchestrator {
    TorusManifold torus;
    NonaryEmbedder embedder;
    TorusDatabase db;
    ExternalToolManager tools;

public:
    std::string process_query(const std::string& query) {
        // 1. Embed
        auto waveform = embedder.embed(query);

        // 2. Inject
        Coord9D inject_pos = compute_injection_point(query);
        torus.inject_wave(inject_pos, waveform_to_complex(waveform));

        // 3. Propagate
        for (int i = 0; i < 100; ++i) {
            torus.propagate(0.01);  // dt = 0.01
        }

        // 4. Check resonance
        auto peak = torus.find_resonance_peak();

        if (peak.amplitude > RESONANCE_THRESHOLD) {
            // 5. Retrieve
            auto data = torus.retrieve_at(peak.location);
            return decode_to_text(data);
        } else {
            // 6. Fetch external
            auto external_data = tools.fetch(query);

            // 7. Store
            auto new_waveform = embedder.embed(external_data);
            torus.inject_wave(compute_storage_point(external_data),
                              waveform_to_complex(new_waveform));

            // 8. Reinforce
            torus.reinforce_region(compute_storage_point(external_data));

            return external_data;
        }
    }
};
```

## 9.4 External Tool Integration

As specified in the core requirements, the system must check if it has necessary data and initiate searches if not found.

### Supported Tools

1. **Tavily Search:** Web search API
2. **Firecrawl:** Web scraping with JavaScript rendering
3. **Gemini CLI:** Direct LLM queries for reasoning
4. **Custom HTTP Client:** Postman-like interface for APIs

### Tool Selection Strategy

```cpp
class ExternalToolManager {
public:
    std::string fetch(const std::string& query) {
        // Analyze query to pick best tool
        if (is_factual_query(query)) {
            return tavily_search(query);
        } else if (is_web_content(query)) {
            return firecrawl_scrape(query);
        } else if (is_reasoning_task(query)) {
            return gemini_query(query);
        } else {
            return http_request(query);
        }
    }

private:
    bool is_factual_query(const std::string& query) {
        // Heuristics: Contains question words, specific entities
        return query.find("what") != std::string::npos ||
               query.find("when") != std::string::npos ||
               query.find("who") != std::string::npos;
    }
};
```

### Data Flow

```
User Query
    ↓
[Nonary Embedder]
    ↓
[Torus Injection]
    ↓
[Wave Propagation] → [Resonance Detection]
    ↓                         ↓
[Found?] ←──────────────────┘
    │
    ├─ Yes → [Retrieve] → Return to User
    │
    └─ No → [External Tools] → [Re-embed] → [Store] → Return to User
```

---

**Cross-References:**
- See Section 5.2 for Balanced Nonary encoding
- See Section 7.1 for Hilbert curve indexing
- See Section 10 for ZeroMQ Spine integration
- See Section 4.3 (External Tool Agents) for detailed tool specifications
- See Appendix C for Protocol Buffer schemas
