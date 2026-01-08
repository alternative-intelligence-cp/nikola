# REMOTE COGNITIVE INTERFACE SPECIFICATION (RCIS)

## 23.1 Protocol Overview

The Remote Cognitive Interface Specification (RCIS) defines the message protocol for external clients to interact with the Nikola Model. RCIS operates over ZeroMQ sockets with Protocol Buffer serialization and CurveZMQ security.

### Design Principles

1. **Asynchronous:** Non-blocking request/response pattern
2. **Secure:** CurveZMQ encryption with public key authentication
3. **Extensible:** Protocol Buffer schema evolution support
4. **Stateless:** Each request is self-contained
5. **Idempotent:** Retry-safe operations

## 23.2 Protocol Buffer Schema

### Core Message Structure

```protobuf
syntax = "proto3";

package nikola.rcis;

// Request envelope
message RCISRequest {
    string request_id = 1;      // UUID for tracking
    int64 timestamp = 2;        // Unix epoch milliseconds
    string auth_token = 3;      // Authentication token (optional with CurveZMQ)

    oneof payload {
        QueryRequest query = 10;
        IngestRequest ingest = 11;
        RetrieveRequest retrieve = 12;
        CommandRequest command = 13;
        MetricsRequest metrics = 14;
    }
}

// Response envelope
message RCISResponse {
    string request_id = 1;      // Matches request
    int64 timestamp = 2;
    int32 status_code = 3;      // HTTP-style codes
    string status_message = 4;

    oneof payload {
        QueryResponse query_response = 10;
        IngestResponse ingest_response = 11;
        RetrieveResponse retrieve_response = 12;
        CommandResponse command_response = 13;
        MetricsResponse metrics_response = 14;
    }
}
```

### Query Operations

```protobuf
message QueryRequest {
    string query_text = 1;
    repeated string context_tags = 2;       // Optional context
    float resonance_threshold = 3;          // Min resonance for results
    int32 max_propagation_steps = 4;        // Max physics cycles
    bool use_external_tools = 5;            // Allow web search
}

message QueryResponse {
    string response_text = 1;
    float resonance_score = 2;              // Peak resonance achieved
    repeated uint32 location_9d = 3;        // 9D coordinates of resonance
    int32 propagation_steps_taken = 4;
    repeated string sources = 5;            // External tool citations
    bool used_external_tool = 6;
    string tool_name = 7;
}
```

### Ingest Operations

```protobuf
message IngestRequest {
    string content = 1;
    string content_type = 2;                // "text", "audio", "image"
    map<string, string> metadata = 3;       // Arbitrary key-value tags
    repeated uint32 target_location = 4;    // Optional 9D injection point
}

message IngestResponse {
    bool success = 1;
    repeated uint32 stored_location = 2;    // Actual 9D coordinates
    float resonance_strength = 3;           // Initial resonance
    string checkpoint_id = 4;               // Snapshot ID after ingest
}
```

### Retrieve Operations

```protobuf
message RetrieveRequest {
    repeated uint32 location_9d = 1;        // Explicit 9D coordinates
    float radius = 2;                       // Neighborhood radius
}

message RetrieveResponse {
    Waveform wavefunction = 1;              // Complex amplitude
    float resonance_r = 2;
    float state_s = 3;
    repeated float metric_tensor = 4;       // 45-element upper triangle
}

message Waveform {
    repeated double real_parts = 1;
    repeated double imag_parts = 2;
}
```

### Command Operations

```protobuf
message CommandRequest {
    enum CommandType {
        NAP = 0;                // Trigger consolidation
        WAKE = 1;               // Resume operation
        CHECKPOINT = 2;         // Force snapshot
        EXPORT_GGUF = 3;        // Export to GGUF
        TRAIN = 4;              // Manual training trigger
    }

    CommandType command = 1;
    map<string, string> parameters = 2;
}

message CommandResponse {
    bool success = 1;
    string result_message = 2;
    bytes result_data = 3;                  // Binary payload (e.g., GGUF file)
}
```

### Metrics Operations

```protobuf
message MetricsRequest {
    bool include_physics = 1;
    bool include_memory = 2;
    bool include_neurochemistry = 3;
}

message MetricsResponse {
    PhysicsMetrics physics = 1;
    MemoryMetrics memory = 2;
    NeurochemistryMetrics neuro = 3;
}

message PhysicsMetrics {
    double avg_step_ms = 1;
    int64 total_propagations = 2;
    int32 active_nodes = 3;
    double energy_total = 4;
}

message MemoryMetrics {
    int64 total_bytes = 1;
    int32 checkpoint_count = 2;
    string latest_checkpoint_id = 3;
    double lsm_compaction_ratio = 4;
}

message NeurochemistryMetrics {
    double dopamine_level = 1;
    double serotonin_level = 2;
    double norepinephrine_level = 3;
    double boredom_score = 4;
}
```

## 23.3 ZeroMQ Socket Configuration

### Client-Side Connection

```cpp
#include <zmq.hpp>
#include "neural_spike.pb.h"

class RCISClient {
    zmq::context_t ctx;
    zmq::socket_t socket;
    std::string public_key;
    std::string secret_key;

public:
    RCISClient(const std::string& server_endpoint,
               const std::string& server_public_key)
        : ctx(1), socket(ctx, ZMQ_DEALER) {

        // Generate client keypair
        char pub[41], sec[41];
        zmq_curve_keypair(pub, sec);
        public_key = std::string(pub);
        secret_key = std::string(sec);

        // Configure CurveZMQ client
        socket.set(zmq::sockopt::curve_secretkey, secret_key);
        socket.set(zmq::sockopt::curve_publickey, public_key);
        socket.set(zmq::sockopt::curve_serverkey, server_public_key);

        // Connect
        socket.connect(server_endpoint);
    }

    RCISResponse send_request(const RCISRequest& request) {
        // Serialize request
        std::string serialized;
        request.SerializeToString(&serialized);

        // Send
        socket.send(zmq::buffer(serialized), zmq::send_flags::none);

        // Receive response
        zmq::message_t reply;
        socket.recv(reply, zmq::recv_flags::none);

        // Deserialize
        RCISResponse response;
        response.ParseFromArray(reply.data(), reply.size());

        return response;
    }
};
```

### Server-Side Endpoint

```cpp
class RCISServer {
    zmq::context_t ctx;
    zmq::socket_t socket;
    TorusManifold& torus;
    Orchestrator& orchestrator;

public:
    RCISServer(TorusManifold& t, Orchestrator& o)
        : ctx(1), socket(ctx, ZMQ_ROUTER), torus(t), orchestrator(o) {

        // Bind to endpoint
        socket.bind("tcp://0.0.0.0:9001");
    }

    void run() {
        while (true) {
            // Receive request
            zmq::message_t identity, request_msg;
            socket.recv(identity, zmq::recv_flags::none);
            socket.recv(request_msg, zmq::recv_flags::none);

            RCISRequest request;
            request.ParseFromArray(request_msg.data(), request_msg.size());

            // Dispatch
            RCISResponse response = handle_request(request);

            // Serialize and send
            std::string serialized;
            response.SerializeToString(&serialized);

            socket.send(identity, zmq::send_flags::sndmore);
            socket.send(zmq::buffer(serialized), zmq::send_flags::none);
        }
    }

private:
    RCISResponse handle_request(const RCISRequest& request) {
        RCISResponse response;
        response.set_request_id(request.request_id());
        response.set_timestamp(std::time(nullptr) * 1000);

        if (request.has_query()) {
            handle_query(request.query(), response);
        } else if (request.has_ingest()) {
            handle_ingest(request.ingest(), response);
        } else if (request.has_retrieve()) {
            handle_retrieve(request.retrieve(), response);
        } else if (request.has_command()) {
            handle_command(request.command(), response);
        } else if (request.has_metrics()) {
            handle_metrics(request.metrics(), response);
        }

        return response;
    }

    void handle_query(const QueryRequest& query, RCISResponse& response) {
        auto result = orchestrator.process_query(query.query_text());

        auto* query_resp = response.mutable_query_response();
        query_resp->set_response_text(result.text);
        query_resp->set_resonance_score(result.resonance);
        query_resp->set_used_external_tool(result.used_tool);

        response.set_status_code(200);
        response.set_status_message("OK");
    }

    void handle_ingest(const IngestRequest& ingest, RCISResponse& response) {
        auto waveform = embedder.embed(ingest.content());
        auto location = torus.inject_wave(waveform);

        auto* ingest_resp = response.mutable_ingest_response();
        ingest_resp->set_success(true);
        for (uint32_t coord : location) {
            ingest_resp->add_stored_location(coord);
        }

        response.set_status_code(201);
        response.set_status_message("Created");
    }

    void handle_retrieve(const RetrieveRequest& retrieve, RCISResponse& response) {
        Coord9D location;
        for (int i = 0; i < 9; ++i) {
            location.coords[i] = retrieve.location_9d(i);
        }

        auto node = torus.get_node_at(location);

        auto* retrieve_resp = response.mutable_retrieve_response();
        retrieve_resp->set_resonance_r(node.resonance_r);
        retrieve_resp->set_state_s(node.state_s);

        auto* wf = retrieve_resp->mutable_wavefunction();
        wf->add_real_parts(node.wavefunction.real());
        wf->add_imag_parts(node.wavefunction.imag());

        response.set_status_code(200);
        response.set_status_message("OK");
    }

    void handle_command(const CommandRequest& command, RCISResponse& response) {
        bool success = false;

        switch (command.command()) {
            case CommandRequest::NAP:
                torus.trigger_consolidation();
                success = true;
                break;
            case CommandRequest::CHECKPOINT:
                persistence.save_checkpoint();
                success = true;
                break;
            case CommandRequest::EXPORT_GGUF:
                export_to_gguf();
                success = true;
                break;
        }

        auto* cmd_resp = response.mutable_command_response();
        cmd_resp->set_success(success);

        response.set_status_code(success ? 200 : 500);
        response.set_status_message(success ? "OK" : "Failed");
    }

    void handle_metrics(const MetricsRequest& metrics, RCISResponse& response) {
        auto* metrics_resp = response.mutable_metrics_response();

        if (metrics.include_physics()) {
            auto* phys = metrics_resp->mutable_physics();
            phys->set_avg_step_ms(torus.get_avg_step_time());
            phys->set_active_nodes(torus.get_active_count());
        }

        response.set_status_code(200);
        response.set_status_message("OK");
    }
};
```

## 23.4 Status Codes

RCIS uses HTTP-style status codes:

| Code | Meaning | Usage |
|------|---------|-------|
| 200 | OK | Successful operation |
| 201 | Created | Resource created (ingest) |
| 400 | Bad Request | Invalid request format |
| 401 | Unauthorized | Authentication failed |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side failure |
| 503 | Service Unavailable | System overloaded |

## 23.5 Error Handling

```protobuf
message ErrorDetails {
    string error_code = 1;          // Machine-readable code
    string error_message = 2;       // Human-readable message
    repeated string stack_trace = 3; // Debug info (dev mode only)
}
```

Example error response:

```cpp
RCISResponse error_response;
error_response.set_status_code(400);
error_response.set_status_message("Invalid query format");

auto* error = error_response.mutable_error_details();
error->set_error_code("INVALID_QUERY");
error->set_error_message("Query text exceeds 10000 character limit");
```

## 23.6 Rate Limiting

RCIS implements token bucket rate limiting:

- **Burst:** 100 requests
- **Refill Rate:** 10 requests/second
- **429 Response:** Includes `Retry-After` header in metadata

```cpp
class RateLimiter {
    int tokens = 100;
    const int max_tokens = 100;
    const int refill_rate = 10;  // per second
    std::chrono::steady_clock::time_point last_refill;

public:
    bool allow_request() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - last_refill
        ).count();

        tokens = std::min(max_tokens, tokens + elapsed * refill_rate);
        last_refill = now;

        if (tokens > 0) {
            --tokens;
            return true;
        }

        return false;
    }
};
```

---

**Cross-References:**
- See Section 10 for ZeroMQ Spine architecture
- See Section 25 for CLI Controller implementation
- See Appendix C for complete Protocol Buffer schemas
