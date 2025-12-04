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

### 11.4.1 Reference Synchronous Implementation (DO NOT USE IN PRODUCTION)

**WARNING:** The following synchronous implementation is provided for reference and testing only. It blocks the main thread during physics propagation and tool dispatch, preventing concurrent query processing. **Production systems must use the AsyncOrchestrator (Section 11.4.2).**

```cpp
class SynchronousOrchestrator {  // Reference implementation only
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

### 11.4.2 Production Asynchronous Implementation (MANDATORY)

**CRITICAL FIX (Audit 6 Item #4):** The orchestrator MUST be asynchronous to prevent blocking the entire system during physics propagation (100ms) and tool dispatch (potentially seconds). The synchronous implementation causes:
- Complete system freeze during query processing
- Inability to handle concurrent requests
- Blocking of API, CLI, and ZMQ message handling
- Failure to process incoming sensor data (audio, video)

**Production-Grade Implementation:**

```cpp
#include <boost/asio.hpp>
#include <future>
#include <thread>

class AsyncOrchestrator {
    boost::asio::io_context io_context;
    boost::asio::thread_pool thread_pool{4};

public:
    // Non-blocking query processing using futures
    std::future<std::string> process_query_async(const std::string& query) {
        return std::async(std::launch::async, [this, query]() {
            // Embed
            auto waveform = embedder.embed(query);

            // Inject
            Coord9D pos = compute_injection_point(query);
            torus.inject_wave(pos, waveform_to_complex(waveform));

            // Propagate asynchronously without blocking
            auto propagation_future = std::async(std::launch::async, [this]() {
                run_propagation_cycles(100);
            });

            // While propagating, can handle other requests
            propagation_future.wait();

            // Check resonance
            auto peak = torus.find_resonance_peak();

            if (peak.amplitude > RESONANCE_THRESHOLD) {
                auto data = torus.retrieve_at(peak.location);
                return decode_to_text(data);
            } else {
                // Async tool dispatch
                ExternalTool tool = select_tool(query);
                auto tool_response_future = dispatch_tool_async(tool, query);
                auto tool_response = tool_response_future.get();

                store_in_torus(tool_response);
                reinforce_pathway(query, tool_response);

                return tool_response;
            }
        });
    }

    // Background physics loop runs continuously
    void start_physics_loop() {
        std::thread([this]() {
            while (running) {
                std::array<double, 9> emitter_outputs;
                emitters.tick(emitter_outputs.data());

                for (int e = 0; e < 8; ++e) {
                    torus.apply_emitter(e, emitter_outputs[e]);
                }

                torus.propagate(0.001);  // 1ms timestep
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }).detach();
    }
};
```

This architecture allows the system to "think" (physics propagation) while simultaneously waiting for external I/O (tool responses), preventing the cognitive loop from blocking.

### 11.4.3 Deployment Recommendation

**Production systems MUST:**
1. Use `AsyncOrchestrator` (Section 11.4.2) as the primary orchestrator
2. Run `start_physics_loop()` at system startup to enable continuous background wave propagation
3. Use `process_query_async()` for all query processing, returning futures immediately
4. Configure thread pool size based on available CPU cores (default: 4 threads)

**Testing/Development systems MAY:**
1. Use `SynchronousOrchestrator` (Section 11.4.1) for single-threaded debugging
2. Accept the performance penalty for simplified call stack analysis

**NEVER:**
1. Deploy `SynchronousOrchestrator` to production environments
2. Mix synchronous and asynchronous orchestrators in the same process

---

**Cross-References:**
- See Section 10 for ZeroMQ Spine integration
- See Section 12 for External Tool Agents implementation
- See Section 9 for Memory Search-Retrieve-Store Loop
- See Section 6 for Wave Interference Processor
