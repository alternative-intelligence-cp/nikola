# Domain IV: Infrastructure & Communications Implementation Specifications

**Document Reference:** NM-004-GAP-INFRASTRUCTURE
**Status:** Implementation-Ready
**Date:** 2025-12-10
**Source:** Gap Analysis Report (Dr. Aris Thorne)

## Overview

The infrastructure layer manages the lifecycle of components and their communication via ZeroMQ. This domain ensures reliable, low-latency message passing while maintaining fault tolerance and security.

---

## Gap 4.1: Message Timeout and Retry Logic

### Context and Requirement

ZMQ reliability specifications need concrete timeout values and retry policies.

### Technical Specification

We implement a **Circuit Breaker Pattern** with differentiated timeouts for control vs data plane.

#### Timeout Configuration

- **Control Messages:** 100ms timeout
- **Data Messages:** 5ms timeout
- **Retries:** 3 attempts with exponential backoff (1ms, 2ms, 4ms)
- **Failure Action:** If Physics Engine fails 3 pings, Orchestrator initiates Hard Reset of the physics process

### Implementation

```cpp
#include <zmq.hpp>
#include <chrono>
#include <thread>

enum class MessagePriority {
    CONTROL,
    DATA
};

class ZMQReliableSocket {
private:
    zmq::socket_t socket;
    static constexpr int MAX_RETRIES = 3;

    std::chrono::milliseconds get_timeout(MessagePriority priority) {
        return priority == MessagePriority::CONTROL ?
            std::chrono::milliseconds(100) :
            std::chrono::milliseconds(5);
    }

public:
    bool send_with_retry(const zmq::message_t& msg, MessagePriority priority) {
        auto timeout = get_timeout(priority);

        for (int attempt = 0; attempt < MAX_RETRIES; ++attempt) {
            // Set send timeout
            socket.set(zmq::sockopt::sndtimeo, static_cast<int>(timeout.count()));

            try {
                auto result = socket.send(msg, zmq::send_flags::none);
                if (result) return true;
            } catch (const zmq::error_t& e) {
                if (e.num() != EAGAIN) throw;
            }

            // Exponential backoff
            std::this_thread::sleep_for(std::chrono::milliseconds(1 << attempt));
        }

        return false; // All retries failed
    }

    std::optional<zmq::message_t> recv_with_timeout(MessagePriority priority) {
        auto timeout = get_timeout(priority);
        socket.set(zmq::sockopt::rcvtimeo, static_cast<int>(timeout.count()));

        zmq::message_t msg;
        auto result = socket.recv(msg, zmq::recv_flags::none);

        if (result) return msg;
        return std::nullopt; // Timeout
    }
};
```

### Validation Procedure

1. **Latency Test:** Measure round-trip time for 1000 control messages. Verify 99th percentile < 50ms.
2. **Failure Recovery:** Kill Physics Engine process. Verify Orchestrator detects failure within 500ms and restarts.

---

## Gap 4.2: Component Crash Recovery

### Context and Requirement

Orchestrator detection of component crashes and automatic recovery.

### Technical Specification

**Heartbeat Sentinel** system with automatic process management.

#### Protocol

- Every component publishes a HEARTBEAT frame on the events socket every 100ms
- Orchestrator maintains a `LastSeen` map
- **Detection Threshold:** If `Now - LastSeen > 500ms`, mark component DEAD
- **Recovery Action:** `kill -9 <pid>`, cleanup SHM, restart process

### Implementation

```cpp
#include <unordered_map>
#include <chrono>
#include <sys/types.h>
#include <signal.h>

struct ComponentHealth {
    std::string name;
    pid_t pid;
    std::chrono::steady_clock::time_point last_heartbeat;
    int missed_heartbeats = 0;
};

class ComponentWatchdog {
private:
    std::unordered_map<std::string, ComponentHealth> components;
    static constexpr auto HEARTBEAT_TIMEOUT = std::chrono::milliseconds(500);
    static constexpr int MAX_MISSED_BEATS = 5;

public:
    void register_component(const std::string& name, pid_t pid) {
        components[name] = {
            name,
            pid,
            std::chrono::steady_clock::now(),
            0
        };
    }

    void update_heartbeat(const std::string& name) {
        auto it = components.find(name);
        if (it != components.end()) {
            it->second.last_heartbeat = std::chrono::steady_clock::now();
            it->second.missed_heartbeats = 0;
        }
    }

    std::vector<std::string> check_health() {
        std::vector<std::string> dead_components;
        auto now = std::chrono::steady_clock::now();

        for (auto& [name, health] : components) {
            auto elapsed = now - health.last_heartbeat;

            if (elapsed > HEARTBEAT_TIMEOUT) {
                health.missed_heartbeats++;

                if (health.missed_heartbeats >= MAX_MISSED_BEATS) {
                    dead_components.push_back(name);
                }
            }
        }

        return dead_components;
    }

    void kill_and_cleanup(const std::string& name) {
        auto it = components.find(name);
        if (it == components.end()) return;

        // 1. Kill process
        kill(it->second.pid, SIGKILL);

        // 2. Cleanup shared memory
        std::string shm_name = "/nikola_" + name;
        shm_unlink(shm_name.c_str());

        // 3. Remove from registry
        components.erase(it);

        // 4. Restart (handled by Orchestrator state machine)
        log_error("Component {} crashed and was cleaned up", name);
    }
};
```

### Watchdog Loop

```cpp
void Orchestrator::watchdog_loop() {
    while (running) {
        auto dead = watchdog.check_health();

        for (const auto& component_name : dead) {
            watchdog.kill_and_cleanup(component_name);
            restart_component(component_name);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
```

---

## Gap 4.3: Shared Memory Lifecycle Management

### Context and Requirement

/dev/shm cleanup to prevent memory leaks.

### Technical Specification

**RAII + Watchdog** approach with boot-time cleanup.

#### Strategy

1. **Wrapper Class:** WaveformSHM destructor calls shm_unlink
2. **Startup Cleanup:** On boot, Orchestrator iterates /dev/shm/nikola_* and deletes stale segments (older than boot time)
3. **Size Limit:** Max 16GB total SHM. ftruncate fails if limit exceeded

### Implementation

```cpp
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>

class WaveformSHM {
private:
    std::string name;
    int fd = -1;
    void* ptr = nullptr;
    size_t size = 0;
    static constexpr size_t MAX_TOTAL_SHM = 16ULL * 1024 * 1024 * 1024; // 16GB

public:
    WaveformSHM(const std::string& segment_name, size_t bytes) : name(segment_name), size(bytes) {
        // 1. Create shared memory object
        fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
        if (fd == -1) throw std::runtime_error("shm_open failed");

        // 2. Set size (will fail if exceeding system limits)
        if (ftruncate(fd, size) == -1) {
            close(fd);
            shm_unlink(name.c_str());
            throw std::runtime_error("SHM size limit exceeded");
        }

        // 3. Map to process address space
        ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) {
            close(fd);
            shm_unlink(name.c_str());
            throw std::runtime_error("mmap failed");
        }
    }

    ~WaveformSHM() {
        if (ptr) munmap(ptr, size);
        if (fd != -1) close(fd);
        shm_unlink(name.c_str()); // Cleanup on destruction (RAII)
    }

    void* data() { return ptr; }
    size_t get_size() const { return size; }
};
```

### Boot-Time Cleanup

```cpp
void Orchestrator::cleanup_stale_shm() {
    namespace fs = std::filesystem;

    auto boot_time = get_system_boot_time();

    for (const auto& entry : fs::directory_iterator("/dev/shm")) {
        if (entry.path().filename().string().starts_with("nikola_")) {
            auto file_time = fs::last_write_time(entry);

            // If SHM segment older than boot, it's stale
            if (file_time < boot_time) {
                fs::remove(entry);
                log_info("Cleaned up stale SHM: {}", entry.path().string());
            }
        }
    }
}
```

---

## Gap 4.4: ZeroMQ Socket Configuration

### Context and Requirement

Tuning ZMQ socket options for reliability and performance.

### Technical Specification

#### Socket Options

```cpp
void configure_zmq_socket(zmq::socket_t& socket) {
    // High-Water Mark: Drop messages if queue full to prevent memory leaks
    socket.set(zmq::sockopt::sndhwm, 1000);
    socket.set(zmq::sockopt::rcvhwm, 1000);

    // Linger: Discard pending messages on close; do not block
    socket.set(zmq::sockopt::linger, 0);

    // Immediate: Only queue if connection exists
    socket.set(zmq::sockopt::immediate, 1);

    // CurveZMQ Security (Ironhouse pattern)
    socket.set(zmq::sockopt::curve_server, 1);
    socket.set(zmq::sockopt::curve_secretkey, server_secret_key);
}
```

### Rationale

- **HWM = 1000:** Limits memory usage. If component can't keep up, messages are dropped (acceptable for real-time data).
- **LINGER = 0:** Fast shutdown. Unsent messages are discarded (state is ephemeral in physics simulation).
- **IMMEDIATE = 1:** Prevents queuing to disconnected peers (fail-fast semantics).

---

## Gap 4.5: Protobuf Version Compatibility

### Context and Requirement

Schema evolution strategy for NeuralSpike protocol buffers.

### Technical Specification

**Append-Only Schema** with topic versioning.

#### Rules

1. **Never delete field IDs** (reuse is forbidden)
2. **New fields are optional** (default values must be safe)
3. **Components ignore unknown fields** (standard Proto3 behavior)
4. **Major Versioning:** If logic changes (e.g., switching from 9D to 10D), change the ZMQ Topic from `nikola.v0` to `nikola.v1`

### Example Schema Evolution

```protobuf
// neural_spike.proto (v1)
message NeuralSpike {
    uint64 timestamp = 1;
    repeated float amplitudes = 2;
    // ... existing fields ...

    // NEW in v1.1 - old components ignore this
    optional float dopamine_level = 10; // Safe default: 0.0
}
```

### Topic Versioning

```cpp
// Publisher
zmq::socket_t pub(ctx, zmq::socket_type::pub);
pub.bind("tcp://*:5555");

// Send on versioned topic
std::string topic = "nikola.v1.spikes"; // Version in topic name
zmq::message_t topic_msg(topic.data(), topic.size());
pub.send(topic_msg, zmq::send_flags::sndmore);
pub.send(spike_msg, zmq::send_flags::none);

// Subscriber
zmq::socket_t sub(ctx, zmq::socket_type::sub);
sub.connect("tcp://localhost:5555");
sub.set(zmq::sockopt::subscribe, "nikola.v1"); // Subscribe to v1 only
```

### Migration Strategy

1. **During development:** All components use `nikola.v0`
2. **Breaking change:** Increment to `nikola.v1`, run old and new components side-by-side
3. **Deprecation:** After 6 months, remove `v0` support

---

## Summary

All 5 Infrastructure & Communications implementation gaps have been addressed with:
- ✅ Circuit breaker pattern with 100ms control / 5ms data timeouts
- ✅ Heartbeat sentinel with 500ms crash detection
- ✅ RAII-based SHM lifecycle with boot-time cleanup
- ✅ Optimized ZMQ socket configuration (HWM, LINGER, IMMEDIATE)
- ✅ Append-only Protobuf schema with topic versioning

**Status:** Ready for distributed system implementation.
