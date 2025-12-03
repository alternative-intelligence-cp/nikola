# MEMORY AND DATA SYSTEMS

## 9.1 Nonary Embedder

The **Custom Nonary Embedder** converts text to waveforms.

### Pipeline

1. **Tokenization:** Byte-Pair Encoding (BPE)
2. **Vectorization:** Lightweight transformer (e.g., distilBERT-tiny)
3. **Quantization:** Map to balanced nonary
4. **Holographic Encoding:** Create interference pattern

### Implementation

```cpp
class NonaryEmbedder {
    BPETokenizer tokenizer;
    TinyTransformer encoder;  // Distilled model

public:
    std::vector<Nit> embed(const std::string& text) {
        // 1. Tokenize
        auto tokens = tokenizer.encode(text);

        // 2. Vectorize (get 768-dim float vector)
        auto vector = encoder.forward(tokens);

        // 3. Quantize to balanced nonary
        std::vector<Nit> nonary_vector;
        for (float val : vector) {
            nonary_vector.push_back(quantize_to_nit(val));
        }

        return nonary_vector;
    }

private:
    Nit quantize_to_nit(float val) {
        // Normalize with tanh to [-1, 1]
        float normalized = std::tanh(val);

        // Scale to [-4, 4]
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
