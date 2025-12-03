# APPENDIX B: PROTOCOL BUFFER REFERENCE

## B.1 Complete Protocol Buffer Schema

**File:** `proto/neural_spike.proto`

**Status:** MANDATORY - This is the canonical message format specification

### B.1.1 Full Schema Definition

```protobuf
syntax = "proto3";

package nikola;

// ============================================================================
// COMPONENT IDENTIFICATION
// ============================================================================

enum ComponentID {
    ORCHESTRATOR = 0;
    PHYSICS_ENGINE = 1;
    MEMORY_SYSTEM = 2;
    REASONING_ENGINE = 3;
    TAVILY_AGENT = 4;
    FIRECRAWL_AGENT = 5;
    GEMINI_AGENT = 6;
    HTTP_CLIENT = 7;
    EXECUTOR_KVM = 8;
    NEUROCHEMISTRY = 9;
    TRAINER_MAMBA = 10;
    TRAINER_TRANSFORMER = 11;
    INGESTION = 12;
    PERSISTENCE = 13;
    SECURITY = 14;
    CLI_CONTROLLER = 15;
}

// ============================================================================
// DATA PAYLOADS
// ============================================================================

// Complex waveform representation (for physics engine)
message Waveform {
    repeated double real_parts = 1;  // Real components of complex wavefunction
    repeated double imag_parts = 2;  // Imaginary components
    int32 length = 3;                // Number of samples
    double sampling_rate = 4;        // Hz (for audio processing)
}

// Sandboxed command execution request
message CommandRequest {
    string task_id = 1;                      // Unique task identifier (UUID)
    string command = 2;                      // Command to execute (e.g., "gcc")
    repeated string args = 3;                // Command arguments
    map<string, string> env = 4;             // Environment variables
    repeated string permissions = 5;         // Requested permissions
    int32 timeout_ms = 6;                    // Execution timeout (milliseconds)
    bool capture_stdout = 7;                 // Capture standard output
    bool capture_stderr = 8;                 // Capture standard error
    string working_directory = 9;            // Working directory (default: /tmp)
}

// Command execution response
message CommandResponse {
    string task_id = 1;                      // Matches request task_id
    int32 exit_code = 2;                     // Process exit code
    string stdout = 3;                       // Standard output (if captured)
    string stderr = 4;                       // Standard error (if captured)
    int64 time_started = 5;                  // Unix timestamp (milliseconds)
    int64 time_ended = 6;                    // Unix timestamp (milliseconds)
    bool timeout_occurred = 7;               // True if timeout triggered
    map<string, int64> usage = 8;            // Resource usage (cpu_ms, mem_kb, etc.)
}

// Neurogenesis event (grid expansion notification)
message NeurogenesisEvent {
    repeated int32 coordinates = 1;          // 9D coordinates (flattened array)
    int32 new_node_count = 2;                // Number of new nodes created
    double trigger_threshold = 3;            // Saturation threshold that triggered event
    int64 timestamp = 4;                     // Unix timestamp (milliseconds)
    string reason = 5;                       // Human-readable reason for expansion
}

// Physics metadata (attached to responses)
message PhysicsMetadata {
    double resonance = 1;                    // Peak resonance amplitude [0.0, 1.0]
    repeated int32 peak_location = 2;        // 9D coordinates of resonance peak
    double energy = 3;                       // Total energy in toroidal system
    int32 active_node_count = 4;             // Number of active nodes in grid
    double interference_strength = 5;        // Magnitude of wave superposition
    int32 propagation_cycles = 6;            // Number of cycles executed
}

// Response metadata (performance tracking)
message ResponseMetadata {
    int64 latency_ms = 1;                    // Processing time (milliseconds)
    int32 propagation_cycles = 2;            // Number of wave propagation cycles
    bool cache_hit = 3;                      // True if retrieved from memory
    string source = 4;                       // "memory" | "tavily" | "firecrawl" | etc.
    string model_version = 5;                // System version that generated response
}

// Rich payload with confidence and citations
message Payload {
    string text = 1;                         // Text content
    double confidence = 2;                   // Confidence score [0.0, 1.0]
    repeated string citations = 3;           // Source URLs or references
    bytes binary_data = 4;                   // Binary data (images, audio, etc.)
    string mime_type = 5;                    // MIME type of binary_data
}

// Neurochemical state (autonomous system)
message NeurochemicalState {
    double dopamine = 1;                     // Reward signal [0.0, 1.0]
    double serotonin = 2;                    // Mood/stability [0.0, 1.0]
    double norepinephrine = 3;               // Alertness [0.0, 1.0]
    double boredom = 4;                      // Entropy-based boredom [0.0, 1.0]
    double curiosity = 5;                    // Curiosity trigger [0.0, 1.0]
    int64 timestamp = 6;                     // When state was measured
}

// Training metrics (autonomous trainers)
message TrainingMetrics {
    int64 epoch = 1;                         // Current training epoch
    double loss = 2;                         // Training loss
    double accuracy = 3;                     // Validation accuracy
    double learning_rate = 4;                // Current learning rate
    int64 samples_processed = 5;             // Total samples seen
    int64 training_time_ms = 6;              // Time spent training (milliseconds)
    string trainer_id = 7;                   // "mamba" | "transformer"
}

// System status report
message StatusReport {
    double dopamine = 1;                     // Current dopamine level
    double boredom = 2;                      // Current boredom level
    int64 active_nodes = 3;                  // Number of active grid nodes
    int64 uptime_seconds = 4;                // System uptime
    map<string, double> metrics = 5;         // Additional metrics (key-value pairs)
    string system_state = 6;                 // "idle" | "processing" | "training" | "nap"
}

// ============================================================================
// MAIN MESSAGE TYPE
// ============================================================================

message NeuralSpike {
    // Header (always present)
    string request_id = 1;                   // UUID for request tracking
    int64 timestamp = 2;                     // Unix timestamp (milliseconds)
    ComponentID sender = 3;                  // Source component
    ComponentID recipient = 4;               // Destination component

    // Optional metadata
    PhysicsMetadata physics = 10;            // Physics engine state
    ResponseMetadata meta = 11;              // Response performance data
    NeurochemicalState neurochemistry = 12;  // Autonomous system state
    TrainingMetrics training = 13;           // Training progress

    // Payload (exactly one of the following)
    oneof payload {
        Waveform data_wave = 5;              // Complex wavefunction data
        CommandRequest command_req = 6;      // Sandboxed execution request
        CommandResponse command_resp = 7;    // Execution result
        NeurogenesisEvent neurogenesis = 8;  // Grid expansion notification
        string text_data = 9;                // Plain text (queries, responses)
        Payload rich_payload = 14;           // Rich text with metadata
        StatusReport status = 15;            // System status
    }
}
```

---

## B.2 Message Usage Patterns

### B.2.1 Query-Response Pattern

**Client Query:**

```protobuf
NeuralSpike {
    request_id: "550e8400-e29b-41d4-a716-446655440000"
    timestamp: 1701234567890
    sender: CLI_CONTROLLER
    recipient: ORCHESTRATOR
    text_data: "What is the golden ratio?"
}
```

**Server Response:**

```protobuf
NeuralSpike {
    request_id: "550e8400-e29b-41d4-a716-446655440000"
    timestamp: 1701234567998
    sender: ORCHESTRATOR
    recipient: CLI_CONTROLLER

    rich_payload: {
        text: "The golden ratio is approximately 1.618033988749895..."
        confidence: 0.92
        citations: ["https://en.wikipedia.org/wiki/Golden_ratio"]
    }

    physics: {
        resonance: 0.87
        peak_location: [12, 34, 56, 15, 22, 8, 45, 67, 3]
        active_node_count: 2187
    }

    meta: {
        latency_ms: 108
        propagation_cycles: 100
        cache_hit: true
        source: "memory"
    }
}
```

### B.2.2 Command Execution Pattern

**Execution Request:**

```protobuf
NeuralSpike {
    request_id: "abc123..."
    sender: ORCHESTRATOR
    recipient: EXECUTOR_KVM

    command_req: {
        task_id: "task-001"
        command: "python3"
        args: ["script.py", "--input", "data.txt"]
        env: {"PYTHONPATH": "/opt/libs"}
        permissions: ["filesystem:read", "filesystem:write:/tmp"]
        timeout_ms: 30000
        capture_stdout: true
        capture_stderr: true
        working_directory: "/tmp/workspace"
    }
}
```

**Execution Response:**

```protobuf
NeuralSpike {
    request_id: "abc123..."
    sender: EXECUTOR_KVM
    recipient: ORCHESTRATOR

    command_resp: {
        task_id: "task-001"
        exit_code: 0
        stdout: "Processing complete\nResults: 42\n"
        stderr: ""
        time_started: 1701234567890
        time_ended: 1701234569120
        timeout_occurred: false
        usage: {
            "cpu_ms": 1250
            "mem_kb": 8192
            "io_kb": 512
        }
    }
}
```

### B.2.3 Waveform Injection Pattern

**Waveform Data:**

```protobuf
NeuralSpike {
    sender: REASONING_ENGINE
    recipient: PHYSICS_ENGINE

    data_wave: {
        real_parts: [0.5, 0.3, -0.2, 0.8, ...]
        imag_parts: [0.1, -0.4, 0.6, 0.0, ...]
        length: 1024
        sampling_rate: 44100.0
    }
}
```

### B.2.4 Neurogenesis Notification Pattern

**Grid Expansion Event:**

```protobuf
NeuralSpike {
    sender: PHYSICS_ENGINE
    recipient: MEMORY_SYSTEM

    neurogenesis: {
        coordinates: [40, 40, 40, 13, 13, 13, 40, 40, 4]
        new_node_count: 27
        trigger_threshold: 0.95
        timestamp: 1701234567890
        reason: "Saturation detected in r-dimension"
    }
}
```

### B.2.5 Status Query Pattern

**Status Request:**

```protobuf
NeuralSpike {
    sender: CLI_CONTROLLER
    recipient: ORCHESTRATOR
    text_data: "status"
}
```

**Status Response:**

```protobuf
NeuralSpike {
    sender: ORCHESTRATOR
    recipient: CLI_CONTROLLER

    status: {
        dopamine: 0.65
        boredom: 0.12
        active_nodes: 2187
        uptime_seconds: 86400
        metrics: {
            "energy": 0.73
            "last_nap_hours_ago": 2.5
            "training_progress": 0.89
        }
        system_state: "idle"
    }
}
```

---

## B.3 Compilation and Integration

### B.3.1 CMake Integration

**proto/CMakeLists.txt:**

```cmake
find_package(Protobuf REQUIRED)

# Generate C++ sources from .proto file
protobuf_generate_cpp(
    PROTO_SRCS
    PROTO_HDRS
    neural_spike.proto
)

# Create static library
add_library(nikola_proto STATIC
    ${PROTO_SRCS}
    ${PROTO_HDRS}
)

target_link_libraries(nikola_proto
    PUBLIC
        protobuf::libprotobuf
)

target_include_directories(nikola_proto
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}  # For generated headers
)

# Install headers
install(FILES ${PROTO_HDRS}
        DESTINATION include/nikola/proto)
```

### B.3.2 Command-Line Compilation

```bash
# Generate C++ code
protoc --cpp_out=./src/generated proto/neural_spike.proto

# Generates:
# - src/generated/neural_spike.pb.h
# - src/generated/neural_spike.pb.cc

# Compile generated code
g++ -c src/generated/neural_spike.pb.cc \
    -o build/neural_spike.pb.o \
    $(pkg-config --cflags protobuf)

# Link with your application
g++ my_app.cpp build/neural_spike.pb.o \
    -o my_app \
    $(pkg-config --libs protobuf)
```

### B.3.3 Usage in C++ Code

**Include and Namespace:**

```cpp
#include "neural_spike.pb.h"

using nikola::NeuralSpike;
using nikola::ComponentID;
using nikola::Waveform;
```

**Creating Messages:**

```cpp
NeuralSpike create_query(const std::string& text) {
    NeuralSpike spike;

    // Set header
    spike.set_request_id(generate_uuid());
    spike.set_timestamp(current_timestamp_ms());
    spike.set_sender(ComponentID::CLI_CONTROLLER);
    spike.set_recipient(ComponentID::ORCHESTRATOR);

    // Set payload
    spike.set_text_data(text);

    return spike;
}
```

**Serialization:**

```cpp
// Serialize to string
std::string serialized;
if (!spike.SerializeToString(&serialized)) {
    throw std::runtime_error("Serialization failed");
}

// Send via ZeroMQ
socket.send(zmq::buffer(serialized), zmq::send_flags::none);
```

**Deserialization:**

```cpp
// Receive from ZeroMQ
zmq::message_t message;
socket.recv(message);

// Deserialize
NeuralSpike received_spike;
if (!received_spike.ParseFromArray(message.data(), message.size())) {
    throw std::runtime_error("Deserialization failed");
}

// Access fields
std::cout << "Request ID: " << received_spike.request_id() << std::endl;
std::cout << "Sender: " << received_spike.sender() << std::endl;

// Check payload type
if (received_spike.has_text_data()) {
    std::cout << "Text: " << received_spike.text_data() << std::endl;
} else if (received_spike.has_command_req()) {
    auto cmd = received_spike.command_req();
    std::cout << "Command: " << cmd.command() << std::endl;
}
```

---

## B.4 Field Numbering and Versioning

### B.4.1 Reserved Field Numbers

**NEVER reuse these field numbers:**

```protobuf
message NeuralSpike {
    reserved 16, 17, 18, 19, 20;
    reserved "old_field_name", "deprecated_field";
}
```

### B.4.2 Backward Compatibility Rules

1. **Adding Fields:** Always safe (old clients ignore new fields)
2. **Removing Fields:** Mark as `reserved` instead of deleting
3. **Changing Field Types:** NEVER change types (breaks compatibility)
4. **Renaming Fields:** Safe (field names not serialized, only numbers)

**Safe Evolution Example:**

```protobuf
// Version 0.0.3
message Payload {
    string text = 1;
    double confidence = 2;
}

// Version 0.0.4 (backward compatible)
message Payload {
    string text = 1;
    double confidence = 2;
    repeated string citations = 3;  // NEW field (safe to add)
    bytes binary_data = 4;          // NEW field (safe to add)
    string mime_type = 5;           // NEW field (safe to add)
}
```

### B.4.3 Version Detection

**Recommended Practice:**

```cpp
// Check if new field exists
if (payload.citations_size() > 0) {
    // Version 0.0.4+ feature
    for (const auto& citation : payload.citations()) {
        process_citation(citation);
    }
} else {
    // Fallback for 0.0.3 compatibility
    std::cout << "No citations available" << std::endl;
}
```

---

## B.5 Performance Considerations

### B.5.1 Message Size Optimization

**Avoid Large Repeated Fields:**

```cpp
// ✓ GOOD: Send data in chunks
for (int i = 0; i < data.size(); i += CHUNK_SIZE) {
    NeuralSpike chunk;
    auto* wave = chunk.mutable_data_wave();
    for (int j = i; j < i + CHUNK_SIZE && j < data.size(); ++j) {
        wave->add_real_parts(data[j].real());
        wave->add_imag_parts(data[j].imag());
    }
    send_spike(chunk);
}

// ✗ BAD: Send all data at once (millions of elements)
NeuralSpike spike;
auto* wave = spike.mutable_data_wave();
for (const auto& sample : all_data) {  // Could be huge!
    wave->add_real_parts(sample.real());
    wave->add_imag_parts(sample.imag());
}
```

### B.5.2 Serialization Performance

**Pre-allocate String Buffers:**

```cpp
std::string serialized;
serialized.reserve(spike.ByteSizeLong());  // Pre-allocate
spike.SerializeToString(&serialized);
```

**Use Arena Allocation for Repeated Messages:**

```cpp
#include <google/protobuf/arena.h>

google::protobuf::Arena arena;
NeuralSpike* spike = google::protobuf::Arena::CreateMessage<NeuralSpike>(&arena);

// Messages allocated on arena (faster, no fragmentation)
// Arena automatically frees memory when destroyed
```

---

**Cross-References:**
- See Section 10.1 for ZeroMQ Spine usage
- See Section 10.2 for Data Format Specifications
- See Appendix C for Virtio-Serial JSON protocol
- See official Protocol Buffers documentation: https://protobuf.dev/

