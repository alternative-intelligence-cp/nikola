# ZEROMQ SPINE ARCHITECTURE

## 10.1 Protocol Definition

**Pattern:** ROUTER-DEALER (asynchronous message broker)

### Topology

```
┌──────────────────────────────────────────────┐
│           ZeroMQ Spine Broker                │
│                                              │
│  Frontend (ROUTER) ←→ Backend (DEALER)       │
└──┬────────────────────────────────────────┬──┘
   │                                        │
   ▼ (Internal Components)                  ▼ (External Agents)
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ Physics │  │ Memory  │  │Reasoning│  │ Tavily  │  │Executor │
│ Engine  │  │ System  │  │ Engine  │  │ Agent   │  │  KVM    │
└─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘
```

## 10.2 Message Types

### Protocol Buffer Definition

```protobuf
syntax = "proto3";

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
}

message Waveform {
    repeated double real_parts = 1;
    repeated double imag_parts = 2;
}

message CommandRequest {
    string task_id = 1;
    string command = 2;
    repeated string args = 3;
    map<string, string> env = 4;
    repeated string permissions = 5;
    int32 timeout_ms = 6;
}

message CommandResponse {
    string task_id = 1;
    int32 exit_code = 2;
    string stdout = 3;
    string stderr = 4;
    int64 time_started = 5;
    int64 time_ended = 6;
}

message NeurogenesisEvent {
    repeated uint32 coordinates = 1;  // 9D coord
    int32 new_node_count = 2;
}

message NeuralSpike {
    string request_id = 1;
    int64 timestamp = 2;
    ComponentID sender = 3;
    ComponentID recipient = 4;

    oneof payload {
        Waveform data_wave = 5;
        CommandRequest command_req = 6;
        CommandResponse command_resp = 7;
        NeurogenesisEvent neurogenesis = 8;
        string text_data = 9;
    }
}
```

## 10.3 Security: CurveZMQ Ironhouse

### Architecture

- Each component has a Curve25519 keypair (public/private)
- Orchestrator acts as ZAP (ZeroMQ Authentication Protocol) authority
- Whitelist of authorized public keys
- Deny-by-default: Unknown keys rejected immediately

### Key Generation

```cpp
#include <zmq.hpp>
#include <sodium.h>

class CurveKeyPair {
public:
    std::array<uint8_t, 32> public_key;
    std::array<uint8_t, 32> secret_key;

    CurveKeyPair() {
        crypto_box_keypair(public_key.data(), secret_key.data());
    }

    std::string public_key_z85() const {
        char z85[41];
        zmq_z85_encode(z85, public_key.data(), 32);
        return std::string(z85);
    }
};
```

### ZAP Handler (Whitelist)

```cpp
class ZAPHandler {
    std::unordered_set<std::string> whitelist;
    zmq::context_t& ctx;
    zmq::socket_t zap_socket;

public:
    ZAPHandler(zmq::context_t& context)
        : ctx(context), zap_socket(ctx, ZMQ_REP) {
        zap_socket.bind("inproc://zeromq.zap.01");
    }

    void add_authorized_key(const std::string& public_key_z85) {
        whitelist.insert(public_key_z85);
    }

    void run() {
        while (true) {
            zmq::message_t version, request_id, domain, address, identity, mechanism, client_key;

            zap_socket.recv(version);
            zap_socket.recv(request_id);
            zap_socket.recv(domain);
            zap_socket.recv(address);
            zap_socket.recv(identity);
            zap_socket.recv(mechanism);
            zap_socket.recv(client_key);

            std::string client_key_str(static_cast<char*>(client_key.data()), client_key.size());

            // Check whitelist
            bool authorized = whitelist.count(client_key_str) > 0;

            // Send response
            zap_socket.send(zmq::str_buffer("1.0"), zmq::send_flags::sndmore);
            zap_socket.send(request_id, zmq::send_flags::sndmore);
            zap_socket.send(zmq::str_buffer(authorized ? "200" : "400"), zmq::send_flags::sndmore);
            zap_socket.send(zmq::str_buffer(authorized ? "OK" : "Unauthorized"), zmq::send_flags::sndmore);
            zap_socket.send(zmq::str_buffer(""), zmq::send_flags::sndmore);
            zap_socket.send(zmq::str_buffer(""));
        }
    }
};
```

## 10.4 Implementation

### Spine Broker

```cpp
class SpineBroker {
    zmq::context_t ctx;
    zmq::socket_t frontend;   // ROUTER for internal components
    zmq::socket_t backend;    // DEALER for external agents
    zmq::socket_t monitor;    // PUB for logging
    CurveKeyPair broker_keys;
    ZAPHandler zap_handler;

public:
    SpineBroker()
        : ctx(1),
          frontend(ctx, ZMQ_ROUTER),
          backend(ctx, ZMQ_DEALER),
          monitor(ctx, ZMQ_PUB),
          zap_handler(ctx) {

        // Configure security
        configure_curve_server(frontend, broker_keys);
        configure_curve_server(backend, broker_keys);

        // Bind sockets
        frontend.bind("ipc:///tmp/nikola/spine_frontend.ipc");
        backend.bind("ipc:///tmp/nikola/spine_backend.ipc");
        monitor.bind("inproc://logger");
    }

    void run() {
        // Start ZAP handler in separate thread
        std::thread zap_thread([this]() { zap_handler.run(); });
        zap_thread.detach();

        // Run proxy
        zmq::proxy(frontend, backend, monitor);
    }
};
```

### Component Connection

```cpp
class ComponentClient {
    zmq::context_t ctx;
    zmq::socket_t socket;
    CurveKeyPair my_keys;
    ComponentID my_id;

public:
    ComponentClient(ComponentID id, const std::string& broker_public_key)
        : ctx(1), socket(ctx, ZMQ_DEALER), my_id(id) {

        // Configure security
        configure_curve_client(socket, my_keys, broker_public_key);

        // Set identity
        std::string identity = "component_" + std::to_string(static_cast<int>(id));
        socket.set(zmq::sockopt::routing_id, identity);

        // Connect
        socket.connect("ipc:///tmp/nikola/spine_frontend.ipc");
    }

    void send_spike(const NeuralSpike& spike) {
        // Serialize protobuf
        std::string data;
        spike.SerializeToString(&data);

        // Send
        socket.send(zmq::buffer(data), zmq::send_flags::none);
    }

    std::optional<NeuralSpike> recv_spike(int timeout_ms = -1) {
        zmq::pollitem_t items[] = {{socket, 0, ZMQ_POLLIN, 0}};
        zmq::poll(items, 1, std::chrono::milliseconds(timeout_ms));

        if (items[0].revents & ZMQ_POLLIN) {
            zmq::message_t msg;
            socket.recv(msg);

            NeuralSpike spike;
            spike.ParseFromArray(msg.data(), msg.size());
            return spike;
        }

        return std::nullopt;
    }
};
```

## 10.5 Shadow Spine Protocol

**Status:** MANDATORY - Required for safe deployment

### Purpose

Test candidate systems in parallel with production without user disruption.

### Architecture

```
User Query
    ↓
┌─────────┐
│ Splitter│ (ZMQ Proxy)
└─┬───┬───┘
  │   │
  ↓   ↓
┌──────────┐  ┌────────────┐
│Prod Sys  │  │Candidate   │
└──────────┘  └────────────┘
  │            │
  │            ↓ (To Architect for analysis)
  │
  ↓ (To User)
```

### Voting Mechanism

If Candidate response has:
- Higher resonance
- Lower latency
- Equal or higher confidence

Then: Vote for promotion.

After 100 consecutive votes, promote Candidate to Production.

### Implementation

```cpp
// File: include/nikola/spine/shadow_spine.hpp
#pragma once

#include "nikola/spine/broker.hpp"

namespace nikola::spine {

class ShadowSpine {
    SpineBroker production_broker;
    SpineBroker candidate_broker;

    int votes_for_candidate = 0;
    const int PROMOTION_THRESHOLD = 100;

public:
    void route_query(const NeuralSpike& query);

    void compare_responses(const NeuralSpike& prod_response,
                          const NeuralSpike& cand_response);

    void promote_candidate_if_ready();
};

} // namespace nikola::spine
```

**Feasibility Rank:** MEDIUM (requires careful orchestration)

---

**Cross-References:**
- See Section 11 for Orchestrator implementation
- See Section 12 for External Tool Agents
- See Section 8.4 (Work Package 4) for Shadow Spine detailed implementation
- See Appendix C for complete Protocol Buffer schemas
