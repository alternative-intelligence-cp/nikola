# ORCHESTRATOR AND SMART ROUTER

## 11.1 Cognitive Switchboard

The **Orchestrator** is the central nervous system hub. It:

1. Receives queries from CLI
2. Coordinates between physics engine, memory, and reasoning
3. Selects external tools when needed
4. Routes messages via ZeroMQ spine

## 11.2 Query Processing

### State Machine

```
IDLE → EMBEDDING → INJECTION → PROPAGATION → RESONANCE_CHECK
     ↓                                            ↓
     ↓ (if no resonance)                         ↓ (if resonance)
     ↓                                            ↓
TOOL_DISPATCH → TOOL_WAIT → STORAGE → REINFORCEMENT → IDLE
     ↓                                            ↓
     └───────────────────────────────────────────┘
                      RESPONSE
```

## 11.3 Tool Selection Logic

### Decision Tree

```cpp
ExternalTool select_tool(const std::string& query) {
    // Pattern matching for tool selection

    // Factual lookup (URLs, entities)
    if (is_factual_query(query)) {
        return ExternalTool::TAVILY;
    }

    // Deep content extraction from specific URL
    if (contains_url(query)) {
        return ExternalTool::FIRECRAWL;
    }

    // Translation, summarization, understanding
    if (is_semantic_task(query)) {
        return ExternalTool::GEMINI;
    }

    // Raw API/HTTP request
    if (is_api_request(query)) {
        return ExternalTool::HTTP_CLIENT;
    }

    // Default: Try Tavily first
    return ExternalTool::TAVILY;
}

bool is_factual_query(const std::string& query) {
    // Simple heuristics (can be ML-based later)
    std::vector<std::string> factual_patterns = {
        "what is", "where is", "who is", "when did", "how many"
    };

    for (const auto& pattern : factual_patterns) {
        if (query.find(pattern) != std::string::npos) {
            return true;
        }
    }

    return false;
}
```

## 11.4 Implementation

### Orchestrator Class

```cpp
class Orchestrator {
    ComponentClient spine_client;
    TorusManifold torus;
    NonaryEmbedder embedder;
    EmitterArray emitters;
    ExternalToolManager tool_manager;

    enum class State {
        IDLE, EMBEDDING, INJECTION, PROPAGATION,
        RESONANCE_CHECK, TOOL_DISPATCH, TOOL_WAIT,
        STORAGE, REINFORCEMENT, RESPONSE
    };

    State current_state = State::IDLE;

public:
    Orchestrator()
        : spine_client(ComponentID::ORCHESTRATOR, load_broker_public_key()) {
    }

    std::string process_query(const std::string& query) {
        current_state = State::EMBEDDING;

        // 1. Embed
        auto waveform = embedder.embed(query);

        // 2. Inject
        current_state = State::INJECTION;
        Coord9D pos = compute_injection_point(query);
        torus.inject_wave(pos, waveform_to_complex(waveform));

        // 3. Propagate
        current_state = State::PROPAGATION;
        run_propagation_cycles(100);

        // 4. Check resonance
        current_state = State::RESONANCE_CHECK;
        auto peak = torus.find_resonance_peak();

        if (peak.amplitude > RESONANCE_THRESHOLD) {
            // Data found in memory
            current_state = State::RESPONSE;
            auto data = torus.retrieve_at(peak.location);
            current_state = State::IDLE;
            return decode_to_text(data);
        } else {
            // Need external tool
            current_state = State::TOOL_DISPATCH;
            ExternalTool tool = select_tool(query);

            auto tool_response = dispatch_tool(tool, query);

            // Store response
            current_state = State::STORAGE;
            store_in_torus(tool_response);

            // Reinforce
            current_state = State::REINFORCEMENT;
            reinforce_pathway(query, tool_response);

            current_state = State::IDLE;
            return tool_response;
        }
    }

private:
    void run_propagation_cycles(int count) {
        for (int i = 0; i < count; ++i) {
            // Tick emitters
            std::array<double, 9> emitter_outputs;
            emitters.tick(emitter_outputs.data());

            // Inject emitter signals
            for (int e = 0; e < 8; ++e) {
                torus.apply_emitter(e, emitter_outputs[e]);
            }

            // Update wave physics
            torus.propagate(0.01);  // dt = 10ms
        }
    }
};
```

---

**Cross-References:**
- See Section 10 for ZeroMQ Spine integration
- See Section 12 for External Tool Agents implementation
- See Section 9 for Memory Search-Retrieve-Store Loop
- See Section 6 for Wave Interference Processor
