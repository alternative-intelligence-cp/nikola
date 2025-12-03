# COMMUNICATION PROTOCOLS

## 10.1 ZeroMQ Spine Architecture

**Status:** MANDATORY - Core infrastructure for all inter-component communication

### 10.1.1 Protocol Definition

**Pattern:** ROUTER-DEALER (asynchronous message broker)

**Topology:**

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

### 10.1.2 Component Identification

**Registered Components:**

| Component ID | Name | Role | Connection Type |
|-------------|------|------|-----------------|
| 0 | ORCHESTRATOR | Central coordinator | Frontend (ROUTER) |
| 1 | PHYSICS_ENGINE | Toroidal wave simulation | Frontend |
| 2 | MEMORY_SYSTEM | LMDB persistence | Frontend |
| 3 | REASONING_ENGINE | Transformer/Mamba | Frontend |
| 4 | TAVILY_AGENT | Web search | Backend (DEALER) |
| 5 | FIRECRAWL_AGENT | Web scraping | Backend |
| 6 | GEMINI_AGENT | Translation/semantic | Backend |
| 7 | HTTP_CLIENT | Custom API calls | Backend |
| 8 | EXECUTOR_KVM | Sandboxed execution | Backend |
| 9 | NEUROCHEMISTRY | ENGS system | Frontend |
| 10 | TRAINER_MAMBA | Autonomous Mamba training | Frontend |
| 11 | TRAINER_TRANSFORMER | Autonomous Transformer training | Frontend |

### 10.1.3 Spine Broker Implementation

**Header Declaration:**

```cpp
// File: include/nikola/spine/broker.hpp
#pragma once

#include <zmq.hpp>
#include <thread>
#include <sodium.h>

namespace nikola::spine {

class SpineBroker {
    zmq::context_t ctx;
    zmq::socket_t frontend;   // ROUTER for internal components
    zmq::socket_t backend;    // DEALER for external agents
    zmq::socket_t monitor;    // PUB for logging

    struct CurveKeyPair {
        std::array<uint8_t, 32> public_key;
        std::array<uint8_t, 32> secret_key;
    };

    CurveKeyPair broker_keys;
    class ZAPHandler;
    std::unique_ptr<ZAPHandler> zap_handler;

public:
    SpineBroker();

    void run();
    void shutdown();

    std::string get_public_key_z85() const;
};

} // namespace nikola::spine
```

**Implementation:**

```cpp
// File: src/spine/broker.cpp

SpineBroker::SpineBroker()
    : ctx(1),
      frontend(ctx, ZMQ_ROUTER),
      backend(ctx, ZMQ_DEALER),
      monitor(ctx, ZMQ_PUB) {

    // Generate broker keypair
    crypto_box_keypair(broker_keys.public_key.data(),
                      broker_keys.secret_key.data());

    // Configure security
    frontend.set(zmq::sockopt::curve_server, 1);
    frontend.set(zmq::sockopt::curve_secretkey,
                broker_keys.secret_key.data(), 32);
    frontend.set(zmq::sockopt::curve_publickey,
                broker_keys.public_key.data(), 32);

    backend.set(zmq::sockopt::curve_server, 1);
    backend.set(zmq::sockopt::curve_secretkey,
               broker_keys.secret_key.data(), 32);
    backend.set(zmq::sockopt::curve_publickey,
               broker_keys.public_key.data(), 32);

    // Bind sockets
    frontend.bind("ipc:///tmp/nikola/spine_frontend.ipc");
    backend.bind("ipc:///tmp/nikola/spine_backend.ipc");
    monitor.bind("inproc://logger");

    // Create ZAP handler
    zap_handler = std::make_unique<ZAPHandler>(ctx);
}

void SpineBroker::run() {
    // Start ZAP authentication handler in separate thread
    std::thread zap_thread([this]() {
        zap_handler->run();
    });
    zap_thread.detach();

    // Run proxy (blocks until shutdown)
    zmq::proxy(frontend, backend, monitor);
}
```

### 10.1.4 Component Client

**Client Interface:**

```cpp
// File: include/nikola/spine/component_client.hpp
#pragma once

#include "neural_spike.pb.h"
#include <zmq.hpp>
#include <optional>

namespace nikola::spine {

class ComponentClient {
    zmq::context_t ctx;
    zmq::socket_t socket;

    struct CurveKeyPair {
        std::array<uint8_t, 32> public_key;
        std::array<uint8_t, 32> secret_key;
    };

    CurveKeyPair my_keys;
    ComponentID my_id;

public:
    ComponentClient(ComponentID id, const std::string& broker_public_key);

    void send_spike(const NeuralSpike& spike);
    std::optional<NeuralSpike> recv_spike(int timeout_ms = -1);

    ComponentID get_id() const { return my_id; }
};

} // namespace nikola::spine
```

**Implementation:**

```cpp
// File: src/spine/component_client.cpp

ComponentClient::ComponentClient(ComponentID id,
                                 const std::string& broker_public_key)
    : ctx(1), socket(ctx, ZMQ_DEALER), my_id(id) {

    // Generate keypair
    crypto_box_keypair(my_keys.public_key.data(),
                      my_keys.secret_key.data());

    // Configure CurveZMQ client
    socket.set(zmq::sockopt::curve_secretkey, my_keys.secret_key.data(), 32);
    socket.set(zmq::sockopt::curve_publickey, my_keys.public_key.data(), 32);

    // Set server public key
    std::array<uint8_t, 32> server_key;
    zmq_z85_decode(server_key.data(), broker_public_key.c_str());
    socket.set(zmq::sockopt::curve_serverkey, server_key.data(), 32);

    // Set identity
    std::string identity = "component_" + std::to_string(static_cast<int>(id));
    socket.set(zmq::sockopt::routing_id, identity);

    // Connect
    socket.connect("ipc:///tmp/nikola/spine_frontend.ipc");
}

void ComponentClient::send_spike(const NeuralSpike& spike) {
    // Serialize protobuf
    std::string data;
    spike.SerializeToString(&data);

    // Send
    socket.send(zmq::buffer(data), zmq::send_flags::none);
}

std::optional<NeuralSpike> ComponentClient::recv_spike(int timeout_ms) {
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
```

---

## 10.2 Security: CurveZMQ Ironhouse Pattern

**Status:** MANDATORY - Required for production deployment

### 10.2.1 Cryptography

**Algorithm:** Curve25519 Elliptic Curve Diffie-Hellman

**Library:** libsodium (NaCl-compatible)

**Key Properties:**
- Public key: 32 bytes (encoded as 40-character Z85 string)
- Secret key: 32 bytes (NEVER transmitted)
- Encryption: ChaCha20-Poly1305 AEAD

### 10.2.2 Key Generation

```cpp
// File: include/nikola/security/curve_keypair.hpp
#pragma once

#include <sodium.h>
#include <zmq.hpp>
#include <array>
#include <string>

namespace nikola::security {

class CurveKeyPair {
public:
    std::array<uint8_t, 32> public_key;
    std::array<uint8_t, 32> secret_key;

    CurveKeyPair() {
        if (sodium_init() == -1) {
            throw std::runtime_error("libsodium initialization failed");
        }
        crypto_box_keypair(public_key.data(), secret_key.data());
    }

    std::string public_key_z85() const {
        char z85[41];
        zmq_z85_encode(z85, public_key.data(), 32);
        return std::string(z85);
    }

    static std::array<uint8_t, 32> decode_z85(const std::string& z85_str) {
        std::array<uint8_t, 32> decoded;
        zmq_z85_decode(decoded.data(), z85_str.c_str());
        return decoded;
    }
};

} // namespace nikola::security
```

### 10.2.3 ZAP Authentication Handler

**ZeroMQ Authentication Protocol (ZAP):**

```cpp
// File: include/nikola/spine/zap_handler.hpp
#pragma once

#include <zmq.hpp>
#include <unordered_set>
#include <string>

namespace nikola::spine {

class ZAPHandler {
    std::unordered_set<std::string> whitelist;
    zmq::context_t& ctx;
    zmq::socket_t zap_socket;
    bool running = false;

public:
    explicit ZAPHandler(zmq::context_t& context);

    void add_authorized_key(const std::string& public_key_z85);
    void remove_authorized_key(const std::string& public_key_z85);

    void run();
    void shutdown();
};

} // namespace nikola::spine
```

**Implementation:**

```cpp
// File: src/spine/zap_handler.cpp

ZAPHandler::ZAPHandler(zmq::context_t& context)
    : ctx(context), zap_socket(ctx, ZMQ_REP) {
    zap_socket.bind("inproc://zeromq.zap.01");
}

void ZAPHandler::add_authorized_key(const std::string& public_key_z85) {
    whitelist.insert(public_key_z85);
}

void ZAPHandler::run() {
    running = true;

    while (running) {
        zmq::message_t version, request_id, domain, address,
                      identity, mechanism, client_key;

        // Receive ZAP request (7 frames)
        zap_socket.recv(version);
        zap_socket.recv(request_id);
        zap_socket.recv(domain);
        zap_socket.recv(address);
        zap_socket.recv(identity);
        zap_socket.recv(mechanism);
        zap_socket.recv(client_key);

        // Extract client public key
        std::string client_key_str(
            static_cast<char*>(client_key.data()),
            client_key.size()
        );

        // Check whitelist
        bool authorized = whitelist.count(client_key_str) > 0;

        // Send ZAP response (6 frames)
        zap_socket.send(zmq::str_buffer("1.0"), zmq::send_flags::sndmore);
        zap_socket.send(request_id, zmq::send_flags::sndmore);
        zap_socket.send(
            zmq::str_buffer(authorized ? "200" : "400"),
            zmq::send_flags::sndmore
        );
        zap_socket.send(
            zmq::str_buffer(authorized ? "OK" : "Unauthorized"),
            zmq::send_flags::sndmore
        );
        zap_socket.send(zmq::str_buffer(""), zmq::send_flags::sndmore);
        zap_socket.send(zmq::str_buffer(""));
    }
}

void ZAPHandler::shutdown() {
    running = false;
}
```

### 10.2.4 Security Policy

**Ironhouse Pattern:**

1. **Deny-by-Default:** All connections rejected unless public key is whitelisted
2. **Key Distribution:** Public keys exchanged out-of-band (configuration files)
3. **No Anonymous Access:** Every component must have a valid keypair
4. **Encryption:** All messages encrypted end-to-end with ChaCha20-Poly1305

**Key Storage:**

```bash
# Example key configuration
/etc/nikola/keys/
├── broker_public.key        # Broker public key (Z85 format)
├── broker_secret.key        # Broker secret key (Z85, chmod 600)
├── orchestrator.key         # Orchestrator keypair
├── physics_engine.key
├── memory_system.key
└── whitelist.txt            # Authorized public keys (one per line)
```

---

## 10.3 Shadow Spine Protocol

**Status:** MANDATORY - Required for safe autonomous evolution

**Work Package:** WP4 (Safety and Self-Improvement)

### 10.3.1 Purpose

Test **candidate systems** in parallel with **production** without disrupting user experience. Enables safe A/B testing of self-improved code.

### 10.3.2 Architecture

```
User Query
    │
┌───┴───────┐
│ Splitter  │ (ZMQ Proxy)
└───┬───┬───┘
    │   │
    ▼   ▼
┌────────┐  ┌────────────┐
│Prod Sys│  │ Candidate  │
└────┬───┘  └─────┬──────┘
     │            │
     │            ▼ (To Architect for analysis)
     │
     ▼ (To User - ALWAYS production response)
```

### 10.3.3 Voting Mechanism

**Promotion Criteria:**

Candidate response must have **ALL** of:
1. Higher resonance score (better pattern match)
2. Lower latency (faster response)
3. Equal or higher confidence

**Vote Counter:** Track consecutive successful comparisons

**Promotion Threshold:** 100 consecutive votes → Promote to production

### 10.3.4 Implementation

**Header:**

```cpp
// File: include/nikola/spine/shadow_spine.hpp
#pragma once

#include "nikola/spine/broker.hpp"
#include "neural_spike.pb.h"
#include <atomic>

namespace nikola::spine {

class ShadowSpine {
    SpineBroker production_broker;
    SpineBroker candidate_broker;

    std::atomic<int> votes_for_candidate{0};
    const int PROMOTION_THRESHOLD = 100;

    struct ResponsePair {
        NeuralSpike production;
        NeuralSpike candidate;
        std::chrono::steady_clock::time_point timestamp;
    };

    std::map<std::string, ResponsePair> pending_comparisons;

public:
    ShadowSpine();

    void route_query(const NeuralSpike& query);
    void compare_responses(const std::string& request_id);
    void promote_candidate_if_ready();

    int get_vote_count() const { return votes_for_candidate.load(); }
};

} // namespace nikola::spine
```

**Implementation:**

```cpp
// File: src/spine/shadow_spine.cpp

#include "nikola/spine/shadow_spine.hpp"
#include <iostream>

void ShadowSpine::route_query(const NeuralSpike& query) {
    // Send to BOTH systems
    production_broker.forward_spike(query);
    candidate_broker.forward_spike(query);

    // Record timestamp
    pending_comparisons[query.request_id()] = {
        .timestamp = std::chrono::steady_clock::now()
    };
}

void ShadowSpine::compare_responses(const std::string& request_id) {
    auto& pair = pending_comparisons.at(request_id);

    const auto& prod = pair.production;
    const auto& cand = pair.candidate;

    // Extract metrics
    bool higher_resonance = cand.physics().resonance() > prod.physics().resonance();
    bool lower_latency = cand.meta().latency_ms() < prod.meta().latency_ms();
    bool equal_confidence = cand.payload().confidence() >= prod.payload().confidence();

    if (higher_resonance && lower_latency && equal_confidence) {
        // Vote for candidate
        int current_votes = ++votes_for_candidate;

        std::cout << "[Shadow Spine] Vote for candidate: "
                  << current_votes << "/" << PROMOTION_THRESHOLD << std::endl;

        if (current_votes >= PROMOTION_THRESHOLD) {
            promote_candidate_if_ready();
        }
    } else {
        // Reset vote counter (must be CONSECUTIVE wins)
        votes_for_candidate = 0;
    }

    // Clean up
    pending_comparisons.erase(request_id);
}

void ShadowSpine::promote_candidate_if_ready() {
    std::cout << "[Shadow Spine] PROMOTING CANDIDATE TO PRODUCTION" << std::endl;

    // 1. Backup current production
    // 2. Swap candidate → production
    // 3. Reset vote counter
    // 4. Notify Architect component

    // TODO: Implement safe promotion with rollback capability

    votes_for_candidate = 0;
}
```

### 10.3.5 Integration with CSVP

**Cross-Reference:** See [Section 8.4: Safety Evolution (WP4)](../08_remediation/04_safety_evolution_wp4.md)

Before promoting candidate:
1. Run Code Safety Verification Protocol (CSVP)
2. Verify physics invariants
3. Check security regression tests
4. Validate performance benchmarks

**Promotion Flow:**

```
Candidate reaches 100 votes
    ↓
Trigger CSVP verification
    ↓
[PASS] → Promote
[FAIL] → Reject, log analysis
```

### 10.3.6 Monitoring

**Metrics to Track:**

```cpp
struct ShadowSpineMetrics {
    int total_queries_routed;
    int candidate_wins;
    int production_wins;
    int ties;
    int current_vote_streak;
    int promotions_count;
    double avg_resonance_delta;
    double avg_latency_delta;
};
```

**Logging:**

```cpp
void log_comparison(const ResponsePair& pair) {
    nlohmann::json log_entry = {
        {"request_id", pair.production.request_id()},
        {"production", {
            {"resonance", pair.production.physics().resonance()},
            {"latency_ms", pair.production.meta().latency_ms()},
            {"confidence", pair.production.payload().confidence()}
        }},
        {"candidate", {
            {"resonance", pair.candidate.physics().resonance()},
            {"latency_ms", pair.candidate.meta().latency_ms()},
            {"confidence", pair.candidate.payload().confidence()}
        }},
        {"winner", determine_winner(pair)}
    };

    // Write to analysis log
    std::ofstream log_file("/var/log/nikola/shadow_spine.jsonl", std::ios::app);
    log_file << log_entry.dump() << std::endl;
}
```

---

## 10.4 Communication Patterns

### 10.4.1 Request-Reply Pattern

**Use Case:** Query processing, tool dispatch

```cpp
// Client sends request
NeuralSpike request;
request.set_request_id(generate_uuid());
request.set_sender(ComponentID::ORCHESTRATOR);
request.set_recipient(ComponentID::TAVILY_AGENT);
request.set_text_data("What is the golden ratio?");

client.send_spike(request);

// Wait for reply
auto reply = client.recv_spike(5000);  // 5 second timeout
if (reply) {
    std::cout << reply->text_data() << std::endl;
}
```

### 10.4.2 Publish-Subscribe Pattern

**Use Case:** Neurogenesis events, dopamine updates

```cpp
// Publisher (Physics Engine)
NeuralSpike event;
event.set_sender(ComponentID::PHYSICS_ENGINE);
event.set_recipient(ComponentID::ORCHESTRATOR);  // Broadcast

auto* neurogenesis = event.mutable_neurogenesis();
neurogenesis->add_coordinates(81);  // 9D coords flattened
neurogenesis->set_new_node_count(27);

physics_client.send_spike(event);

// Subscriber (Memory System)
auto event_msg = memory_client.recv_spike();
if (event_msg && event_msg->has_neurogenesis()) {
    handle_neurogenesis(event_msg->neurogenesis());
}
```

### 10.4.3 Pipeline Pattern

**Use Case:** Multi-stage processing (embed → inject → propagate → retrieve)

```cpp
// Stage 1: Orchestrator → Embedder
spike1.set_recipient(ComponentID::REASONING_ENGINE);
client.send_spike(spike1);

// Stage 2: Embedder → Physics Engine
auto embedded = client.recv_spike();
embedded->set_recipient(ComponentID::PHYSICS_ENGINE);
client.send_spike(*embedded);

// Stage 3: Physics Engine → Memory System
auto propagated = client.recv_spike();
propagated->set_recipient(ComponentID::MEMORY_SYSTEM);
client.send_spike(*propagated);

// Final: Memory System → Orchestrator
auto result = client.recv_spike();
```

---

## 10.5 Error Handling and Reliability

### 10.5.1 Timeout Policy

```cpp
const int TIMEOUT_MS_SHORT = 1000;      // Quick operations
const int TIMEOUT_MS_MEDIUM = 5000;     // External API calls
const int TIMEOUT_MS_LONG = 30000;      // Training, large propagations

auto response = client.recv_spike(TIMEOUT_MS_MEDIUM);
if (!response) {
    // Timeout occurred
    handle_timeout(original_request);
}
```

### 10.5.2 Retry Logic

```cpp
template<typename Func>
std::optional<NeuralSpike> retry_with_backoff(Func operation, int max_retries = 3) {
    int backoff_ms = 100;

    for (int attempt = 0; attempt < max_retries; ++attempt) {
        auto result = operation();
        if (result) return result;

        std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
        backoff_ms *= 2;  // Exponential backoff
    }

    return std::nullopt;
}
```

### 10.5.3 Circuit Breaker

```cpp
class CircuitBreaker {
    int failure_count = 0;
    const int FAILURE_THRESHOLD = 5;
    std::chrono::steady_clock::time_point last_failure;

public:
    bool should_allow_request() {
        if (failure_count >= FAILURE_THRESHOLD) {
            auto elapsed = std::chrono::steady_clock::now() - last_failure;
            if (elapsed < std::chrono::seconds(60)) {
                return false;  // Circuit open
            } else {
                failure_count = 0;  // Reset after cooldown
            }
        }
        return true;
    }

    void record_failure() {
        ++failure_count;
        last_failure = std::chrono::steady_clock::now();
    }

    void record_success() {
        failure_count = 0;
    }
};
```

---

**Cross-References:**
- See Section 10.2 for Protocol Buffer message definitions
- See Section 8.4 for CSVP integration details
- See Section 9.4 for build system configuration
- See Appendix B for complete protobuf schemas

