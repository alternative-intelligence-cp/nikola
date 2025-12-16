# EXECUTOR SANDBOX AND PERMISSION SYSTEM

**[Bug Sweep 009 Integration - KVM Executor & Hybrid Deployment Architecture]**

﻿NIKOLA MODEL v0.0.4: EXECUTOR SANDBOX AND PERMISSION SYSTEM ENGINEERING SPECIFICATION
1. EXECUTIVE SUMMARY AND ARCHITECTURAL PHILOSOPHY
1.1 The Imperative of Containment in Autonomous Systems
The Nikola Model v0.0.4 represents a paradigm shift in artificial intelligence architecture, moving away from static neural weights toward a dynamic, self-modifying 9-Dimensional Toroidal Waveform Intelligence (9D-TWI). A central tenet of this architecture is the capacity for recursive self-improvement, wherein the system analyzes its own C++ source code, generates optimizations, and hot-swaps these modules into its active memory space.1 While this capability theoretically allows for unbounded optimization, it introduces catastrophic existential risks. An error in the physics kernel could violate conservation of energy laws, leading to numeric instability that equates to a "seizure," while a hallucinated command could result in the deletion of the host filesystem or the corruption of the cryptographic identity keys.1
Therefore, the Executor Subsystem is not merely a task runner; it is the Containment Facility of the architecture. It serves as the physical boundary between the cognitive entity—which exists as a waveform on the torus—and the underlying hardware that sustains it. The design of this system is predicated on the Zero Trust principle. The cognitive core, despite being the "brain" of the system, is treated as an untrusted actor by the Executor. Every instruction issued by the Orchestrator, whether it is a request to scrape a webpage or a command to compile a new physics kernel, must pass through layers of verification, sanitization, and isolation before it touches silicon.1
This specification document details the engineering requirements, mathematical models, and implementation strategies for the Executor. It addresses critical findings from Engineering Audit Phase 13, specifically the insecurity of text-based protocols (SEC-01), the instability of nested virtualization in cloud environments (INT-P6), and the thermodynamic costs of computation (CF-04).1 The resulting architecture is a Hybrid Deployment Model that leverages the portability of Docker for the cognitive core while utilizing the raw power and isolation of bare-metal KVM (Kernel-based Virtual Machine) for the execution sandbox. This ensures that the system can "think" in the cloud but "act" with the security of a hardware-enforced air gap.
1.2 The Scope of the Executor
The responsibilities of the Executor extend beyond simple virtualization. It acts as the interface for External Tool Agents, the Self-Improvement Engine, and the Adversarial Code Dojo.
1. Tool Execution: The system requires access to tools like Tavily (search), Firecrawl (scraping), and Python interpreters for data analysis. The Executor provides ephemeral environments for these tools, ensuring that a compromised web scraper cannot pivot to attack the memory persistence layer.1
2. Compilation and Testing: When the system proposes a code change, the Executor spins up a sandbox to compile the code, run unit tests, and—crucially—execute the Physics Oracle to verify energy conservation laws. Only binaries that pass this gauntlet are signed and released to the core.1
3. Resource Governance: The Executor enforces the Metabolic Budget. Just as biological muscles consume ATP, computational tasks consume system resources. The Executor tracks CPU cycles and RAM usage, translating them into metabolic costs that deplete the system's "energy," forcing a "Nap" cycle when exhausted. This prevents runaway processes from causing thermal throttling or system unresponsiveness.1
The following sections will dismantle the previous "Docker-in-Docker" approach, which was found to be fragile and insecure, and reconstruct the Executor as a high-performance, systemd-managed service interacting directly with the Linux kernel's virtualization primitives.
________________
2. HYBRID DEPLOYMENT ARCHITECTURE
2.1 The Failure of Nested Virtualization (Analysis of INT-P6)
Initial implementation plans for the Nikola Model envisioned a monolithic Docker container encapsulating the entire system, including the virtualization layer. This approach, while attractive for its simplicity in distribution, failed under rigorous engineering scrutiny during the Audit Phase 13. The finding, designated INT-P6, highlighted a fatal flaw in relying on nested virtualization (running KVM inside a Docker container that is itself running on a virtualized cloud instance).1
The mechanics of nested virtualization require the outer hypervisor (e.g., AWS Nitro or Google Titan) to pass hardware virtualization extensions (VMX on Intel, SVM on AMD) through to the guest OS, which then passes them to the container. This "Matryoshka doll" of abstraction layers introduces severe performance penalties. Benchmarks cited in the audit revealed that VM boot times in a nested environment degraded from a baseline of 2 seconds to over 12 seconds.1 For a cognitive system attempting to run a real-time "Inner Monologue" where thoughts trigger code execution, a 10-second latency is effectively a cognitive arrest.
Furthermore, enabling nested virtualization within a container requires the container to run in --privileged mode. This flag disables nearly all of Docker's security namespaces, cgroups, and capabilities protections, giving the container root access to the host kernel. If the AI were to hallucinate a malicious command or if an external tool were compromised, the --privileged flag would offer a direct path to host compromise, rendering the sandbox illusory. Consequently, the monolithic container architecture was deemed unfit for production.1
2.2 The Hybrid Topology Specification
To resolve the security and performance criticalities of INT-P6, we define a Hybrid Deployment Architecture. This topology separates the system into two distinct runtime domains: the Cognitive Core and the Executor Service.
2.2.1 The Cognitive Core (Containerized)
The "mind" of the Nikola Model—comprising the Mamba-9D State Space Model, the Wave Interference Physics Engine, and the Orchestrator—resides within a standard Docker container. This ensures that the complex web of machine learning dependencies (CUDA, PyTorch, specialized linear algebra libraries) remains encapsulated and reproducible. This container does not require privileged access and operates within a restricted user namespace. It communicates with the outside world solely through defined ZeroMQ sockets.1
2.2.2 The Executor Service (Host-Native)
The "body" of the system—the component responsible for dangerous actions—runs as a systemd service directly on the bare-metal host (or the primary cloud VM). By running natively on the host OS (targeted as Ubuntu 24.04 LTS), the Executor gains direct access to /dev/kvm and /dev/net/tun without the overhead of container passthrough. This restores VM boot performance to the bare-metal baseline (<2 seconds) and allows the use of standard Linux security modules like AppArmor and SELinux to confine the Executor process.1
2.3 The ZeroMQ Bridge
The connective tissue between the Containerized Core and the Host-Native Executor is the ZeroMQ Spine. Specifically, a TCP bridge is established over the Docker virtual network interface (typically docker0).
* Endpoint: tcp://172.17.0.1:5556 (Host IP on Docker bridge).
* Protocol: NeuralSpike Protobuf over CurveZMQ (Ironhouse pattern).
* Directionality: The Orchestrator (in container) acts as the Client (DEALER socket), and the Executor (on host) acts as the Server (ROUTER socket).
This separation enforces a strict network boundary. Even if the Cognitive Core is completely subverted, it cannot directly spawn processes on the host. It can only send signed, encrypted requests to the Executor, which then validates them against a rigid permission model before taking action. This architecture effectively air-gaps the intelligence from the execution hardware, mediating all agency through a verifiable protocol.1
________________
3. VIRTUALIZATION AND SANDBOX STRATEGY
3.1 KVM and Libvirt Implementation
The choice of KVM (Kernel-based Virtual Machine) is non-negotiable. While lightweight containers (LXC, runc) offer speed, they share the host's kernel. A "kernel panic" in a container crashes the host. A kernel exploit allows escape. KVM uses hardware-assisted virtualization (Intel VT-x or AMD-V) to create a fully isolated execution environment with its own kernel. The Executor utilizes the libvirt C++ API for programmatic control of these domains, avoiding the latency of shelling out to virsh commands.1
The implementation manages the lifecycle of Transient Domains. Unlike traditional VMs that persist for days or months, these domains are ephemeral. They are instantiated for a specific task (e.g., "Compile Module A") and destroyed immediately upon completion. This statelessness is a crucial security feature: no malware or corrupted state can persist between tasks because the virtual machine itself ceases to exist.1
3.2 The Gold Image and Copy-on-Write Strategy
To reconcile the security of full virtualization with the latency requirements of a cognitive loop, we employ a storage strategy based on QCOW2 (QEMU Copy On Write) overlays.
3.2.1 The Gold Image
The foundation is the "Gold Image," a minimal, read-only disk image containing a stripped-down Linux distribution (Alpine or Minimal Ubuntu). This image is pre-hardened: unnecessary services are disabled, the network stack is locked down, and the Nikola Guest Agent is installed. This file resides in a protected directory (e.g., /var/lib/nikola/gold/) and is never modified by the running system. Its immutability ensures that every task starts from a known, clean state.1
3.2.2 The Overlay Mechanism
When a task is requested, the Executor does not copy the Gold Image (which would take seconds and consume gigabytes). Instead, it creates a QCOW2 overlay. This overlay file initially contains no data; it is simply a map pointing to the backing Gold Image. As the VM runs, any writes it makes—creating temp files, compiling binaries—are written to the overlay. Reads for unmodified data are served from the backing file.
This operation is nearly instantaneous ($<10$ ms) and storage-efficient. Crucially, when the task is finished, the overlay is simply unlinked (deleted). This provides a mathematically guaranteed "Reset" to the base state, as the differential data is physically removed from the filesystem. This mechanism protects against "persistence attacks" where a malicious tool attempts to hide in the filesystem to infect future tasks.1
3.3 The Warm VM Pool
Even with QCOW2 overlays, booting a Linux kernel takes finite time (approx. 800-1200ms for a modern kernel). For a system designed to interact in real-time, a 1-second latency for every tool use is unacceptable. To solve this, the Executor implements a Warm VM Pool.1
3.3.1 Pool Dynamics
The VMPool class maintains a queue of pre-booted, paused Virtual Machines.
1. Initialization: On startup, the Executor spins up $N$ VMs (configurable, default 3). These VMs boot, load the Guest Agent, and then suspend execution, waiting in a low-power state.
2. Acquisition: When a task arrives, the scheduler requests a VM from the pool. This operation is effectively instantaneous ($<5$ ms), as it involves merely resuming the CPU execution of the suspended VM.
3. Execution: The task runs.
4. Sanitization and Return: Upon completion, the VM is not immediately destroyed. The Guest Agent receives a RESET command to clear /tmp and reset internal state. If the reset is successful, the VM is returned to the pool.
5. Recycling: To prevent "bit rot" or subtle state accumulation (memory leaks in the guest kernel), VMs are strictly retired after a set number of uses (e.g., 10) or a set duration (e.g., 5 minutes), and replaced with fresh boots.
This architecture reduces the effective latency of tool execution from ~1200ms to ~20ms, bringing the "reflex arc" of the AI into the realm of human reaction times.1
3.4 Immutable Guest Agent Deployment
A critical security challenge is ensuring the integrity of the code running inside the sandbox. If the VM downloads the agent from the network, a man-in-the-middle attack could compromise it. If the agent is on a writable partition, a compromised process could overwrite it.
The solution is ISO Injection. The Guest Agent binary (nikola-agent) is baked into a dynamic ISO 9660 image that is mounted as a virtual CD-ROM drive (/dev/sr0) at boot. The ISO 9660 filesystem is inherently read-only. The Linux kernel within the guest physically cannot write to this device. This provides a hardware-enforced guarantee of agent integrity. Even if an attacker gains root privileges within the VM, they cannot modify the agent binary to hide their activities or spoof return codes.1
________________
4. PERMISSION MODEL SPECIFICATION
The Permission System is the legislative layer of the containment facility. It defines the "Laws of Physics" for the sandbox, determining what actions are possible. Unlike traditional binary permissions (Admin/User), the Nikola Model uses a Capability-Based Security Model that integrates with the 9D topological state.
4.1 The Capability Manifest
Every CommandRequest protobuf message (Section 10.2 of the plan) includes a permissions field. This is a list of string tokens representing specific capabilities the task requires. The Executor enforces these capabilities at two levels: the Hypervisor (Hard) and the Agent (Soft).1
4.1.1 Hard Capabilities (Hypervisor Level)
These are enforced by the KVM configuration and cgroups on the host. The guest OS has no awareness of or control over these limits.
* net:egress: Controls attachment to the NAT bridge. Without this token, the VM's network interface is connected to a "black hole" bridge with no uplink. The VM is air-gapped.
* res:high_cpu: Allocates a higher CPU quota (cgroups cpu.weight). Used for compilation or rendering tasks. Default tasks are throttled to prevent host starvation.
* res:large_mem: Increases the RAM limit from the default (e.g., 512MB) to a higher tier (e.g., 4GB).
* dev:gpu: (Experimental) Passthrough of a virtual GPU function. Strictly controlled due to DMA security risks.
4.1.2 Soft Capabilities (Agent Level)
These are enforced by the Guest Agent inside the VM via the Cognitive Safety Verification Protocol (CSVP).
* base:execute: Allows execution of standard coreutils (ls, grep, cat).
* dev:compiler: Whitelists access to gcc, g++, make, cmake.
* dev:python: Whitelists the Python interpreter.
* fs:write_tmp: Grants write access to the /tmp directory.
* sys:ptrace: Allows the usage of debugging tools (blocked by default to prevent process injection).
4.2 Permission Logic and Enforcement
When the Executor receives a request, it performs a two-phase check.
Phase 1: Hypervisor Configuration
The VMPool manager checks the Hard Capabilities. If net:egress is required, it attaches a network interface device. If not, the device is detached. This configuration happens before the VM is unpaused.
Phase 2: Agent Verification (CSVP)
The command and the capability tokens are sent to the Guest Agent. The Agent holds a static map of binaries to required capabilities.
* Command: gcc -o test test.c
* Required Capability: dev:compiler
* Provided Capabilities: ['base:execute']
* Result: REJECTED. The Agent refuses to exec() the binary.
This dual-layer approach implements Defense in Depth. Even if the Agent is bypassed (e.g., via a kernel exploit), the Hypervisor restrictions (like the air-gapped network) remain in effect.
4.3 Integration with Identity and Neurochemistry
The permission system is not static; it is coupled with the system's Identity and Neurochemistry.1
* ATP Cost: High-capability tasks act as a tax on the system's metabolic budget. Requesting net:egress or res:high_cpu consumes significantly more "ATP" than a local task. If the system is in a low-energy state (fatigued), the Executor may reject high-permission tasks, forcing the system to "Nap" and consolidate memories before attempting complex operations. This prevents the AI from initiating dangerous or resource-intensive self-improvement loops when it is "tired" and prone to errors.
* Identity Gating: Certain capabilities (like modifying the core kernel code) are cryptographically locked to the "Architect" persona. The Orchestrator must present a valid signature corresponding to this high-level identity state to invoke the dev:compiler capability for core system files.
________________
5. TASK QUEUE AND CALLBACK ARCHITECTURE
The mechanism for submitting work to the Executor and receiving results is designed for high throughput and fault tolerance. It utilizes an asynchronous, event-driven architecture built on ZeroMQ.
5.1 The ZeroMQ Spine Topology
The communication backbone utilizes the ROUTER-DEALER pattern.
* Executor (Server): Binds a ROUTER socket. This socket type tracks the identity of connecting clients, allowing the server to route replies back to the specific source asynchronously.
* Orchestrator (Client): Connects via a DEALER socket. This allows the Orchestrator to fire multiple requests without waiting for immediate replies, enabling non-blocking operation of the cognitive loop.
5.2 Priority Queue Architecture
Inside the Executor, requests are not processed strictly First-In-First-Out (FIFO). A Priority Queue is employed to ensure that critical control signals take precedence over background tasks.
Priority Levels:
1. CRITICAL (0): Security updates, Emergency Shutdown (SCRAM), Energy conservation overrides.
2. HIGH (1): User-interactive queries (latency sensitive).
3. NORMAL (2): Background research, file ingestion.
4. LOW (3): Self-improvement compilation, extensive simulations.
Queue Discipline and Backpressure:
The queue has a hard depth limit (e.g., 1000 tasks). If the queue is full, the TaskScheduler applies backpressure by rejecting new submissions with a 503 Service Unavailable error. This protects the host from memory exhaustion during a "thought loop" where the AI might generate thousands of redundant tasks.1
5.3 Asynchronous Callback Mechanism
The Executor cannot block while a VM runs a 5-minute compilation. The callback architecture handles this asynchrony.
1. Submission: The Orchestrator sends a CommandRequest. The ZeroMQ ROUTER socket on the Executor adds a routing envelope (the "Identity Frame") to the message.
2. Encapsulation: The Executor wraps the request and the Identity Frame into a Task object. This object is pushed onto the thread-safe Priority Queue.
3. Processing: A worker thread from the thread pool pops the Task. It acquires a VM, runs the job, and captures the output.
4. Routing: The worker thread wraps the result in a CommandResponse. Crucially, it retrieves the stored Identity Frame from the Task object.
5. Dispatch: The worker sends the response via the ROUTER socket, prefixing the data with the Identity Frame. ZeroMQ uses this frame to route the message back to the exact Orchestrator instance that requested it.
This stateless routing allows the Executor to scale. It can handle requests from multiple sources (e.g., a CLI tool, the Orchestrator, a debug harness) simultaneously, always returning the answer to the correct caller.
________________
6. SECURITY ARCHITECTURE: IOGUARD AND SECURE CHANNELS
The boundary between the Host and the Guest is the most critical attack surface. A compromised guest will attempt to attack the host through the communication channel.
6.1 IOGuard: Rate Limiting and DoS Protection
A common attack vector is resource exhaustion (DoS). A malicious process inside the VM could simply output an infinite stream of random data to stdout. If the Host Executor tries to read and log all this data, it will consume 100% of the Host CPU and fill the disk logs, effectively killing the Nikola node.
IOGuard is a token-bucket rate limiter implemented directly on the host's file descriptor reading from the VM's virtio-serial port.1
The Algorithm:




$$T(t) = \min(C, T(t-1) + R \cdot \Delta t)$$


Where $T$ is the token count, $C$ is the burst capacity (256 KB), and $R$ is the refill rate (1 MB/s).
When the Host attempts to read(), it checks the bucket. If $T < \text{read\_size}$, it reads only $T$ bytes. If $T=0$, the Host stops reading. This is the key mechanism: by ceasing to read, the Host exerts backpressure. The buffer in the virtio-serial driver fills up. The guest OS blocks the writing process when its buffer is full. The attack is thus contained entirely within the guest; the malicious process puts itself to sleep waiting for buffer space, while the Host remains unaffected.
6.2 Secure Guest Channel Protocol (Remediation of SEC-01)
The initial design used JSON for host-guest communication. Audit Finding SEC-01 flagged this as insecure. JSON parsers are complex and prone to "JSON Bomb" attacks (deeply nested structures causing stack overflow) and type confusion vulnerabilities.
We implement a Binary Frame Protocol for all control messages.
Frame Structure:








* Magic: 0xDEADBEEF. A sync marker to detect stream misalignment.
* Length: Strictly capped (e.g., 16MB). Prevents allocation of massive buffers.
* CRC32: Ensuring integrity against bit-flips or transmission errors.
* Payload: Protobuf serialized data.
Validation Logic:
The Host Executor reads the header first. It validates the Magic and Length. It then reads the payload. Before parsing the Protobuf, it computes the CRC32 of the payload and compares it to the header. Only if the checksum matches is the data passed to the Protobuf parser. This "Verify-then-Parse" pattern eliminates entire classes of exploitation where the parser itself is the target.1
________________
7. IMPLEMENTATION SPECIFICATIONS
This section provides the concrete C++23 implementation details required for code generation.
7.1 The Secure Guest Channel (Header Definitions)


C++




// include/nikola/executor/secure_channel.hpp
#pragma once
#include <cstdint>
#include <vector>
#include <optional>
#include <zlib.h> // For CRC32
#include "nikola/proto/neural_spike.pb.h"

namespace nikola::executor {

// Binary Frame Header - 16 Bytes
struct PacketHeader {
   uint32_t magic;         // 0xDEADBEEF
   uint32_t payload_len;   // Max 16MB
   uint32_t crc32;         // Integrity Check
   uint32_t sequence_id;   // Replay Protection
};

class SecureChannel {
private:
   static constexpr uint32_t MAGIC_VAL = 0xDEADBEEF;
   static constexpr uint32_t MAX_PAYLOAD = 16 * 1024 * 1024;

public:
   // Wraps a Protobuf message into a binary frame
   static std::vector<uint8_t> wrap_message(const nikola::NeuralSpike& msg, uint32_t seq_id) {
       std::string body = msg.SerializeAsString();
       
       PacketHeader header;
       header.magic = MAGIC_VAL;
       header.payload_len = static_cast<uint32_t>(body.size());
       header.crc32 = crc32(0L, reinterpret_cast<const Bytef*>(body.data()), body.size());
       header.sequence_id = seq_id;
       
       std::vector<uint8_t> packet;
       packet.reserve(sizeof(PacketHeader) + body.size());
       
       // Append Header
       const uint8_t* header_ptr = reinterpret_cast<const uint8_t*>(&header);
       packet.insert(packet.end(), header_ptr, header_ptr + sizeof(PacketHeader));
       
       // Append Body
       packet.insert(packet.end(), body.begin(), body.end());
       
       return packet;
   }

   // Unwraps and validates a binary frame
   static std::optional<nikola::NeuralSpike> unwrap_message(const std::vector<uint8_t>& buffer) {
       // 1. Structural Validation
       if (buffer.size() < sizeof(PacketHeader)) return std::nullopt;
       
       const PacketHeader* header = reinterpret_cast<const PacketHeader*>(buffer.data());
       
       if (header->magic!= MAGIC_VAL) {
           // Log security event: Invalid Magic
           return std::nullopt;
       }
       
       if (header->payload_len > MAX_PAYLOAD) {
           // Log security event: Oversized Payload
           return std::nullopt;
       }
       
       if (buffer.size() < sizeof(PacketHeader) + header->payload_len) {
           return std::nullopt; // Incomplete packet
       }
       
       // 2. Integrity Validation
       const uint8_t* payload_ptr = buffer.data() + sizeof(PacketHeader);
       uint32_t computed_crc = crc32(0L, payload_ptr, header->payload_len);
       
       if (computed_crc!= header->crc32) {
           // Log security event: CRC Mismatch
           return std::nullopt;
       }
       
       // 3. Semantic Parsing
       nikola::NeuralSpike msg;
       if (!msg.ParseFromArray(payload_ptr, header->payload_len)) {
           return std::nullopt;
       }
       
       return msg;
   }
};

} // namespace nikola::executor

7.2 The IOGuard Rate Limiter


C++




// include/nikola/executor/io_guard.hpp
#pragma once
#include <chrono>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <unistd.h>

namespace nikola::executor {

class IOGuard {
private:
   const size_t RATE_BYTES_PER_SEC = 1024 * 1024; // 1 MB/s
   const size_t BURST_BYTES = 256 * 1024;         // 256 KB
   
   std::atomic<size_t> tokens;
   std::chrono::steady_clock::time_point last_refill;
   std::mutex refill_mutex;

public:
   IOGuard() : tokens(BURST_BYTES), last_refill(std::chrono::steady_clock::now()) {}

   // Returns number of bytes read, or -1 if throttled
   ssize_t guarded_read(int fd, void* buf, size_t count) {
       refill();
       
       size_t current_tokens = tokens.load(std::memory_order_relaxed);
       
       if (current_tokens == 0) {
           return -1; // Apply backpressure (don't read)
       }
       
       // Clamp read size to available tokens
       size_t to_read = std::min(count, current_tokens);
       
       ssize_t bytes_read = ::read(fd, buf, to_read);
       
       if (bytes_read > 0) {
           tokens.fetch_sub(bytes_read, std::memory_order_relaxed);
       }
       
       return bytes_read;
   }

private:
   void refill() {
       auto now = std::chrono::steady_clock::now();
       
       // Use try_lock to avoid contention on hot path. 
       // If locked, another thread is refilling; skip (tokens are monotonic).
       if (refill_mutex.try_lock()) {
           auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_refill).count();
           
           if (elapsed > 100) { // Update every 100ms
               size_t new_tokens = (RATE_BYTES_PER_SEC * elapsed) / 1000;
               size_t current = tokens.load(std::memory_order_relaxed);
               
               // Atomic store with saturation
               tokens.store(std::min(BURST_BYTES, current + new_tokens), std::memory_order_relaxed);
               last_refill = now;
           }
           refill_mutex.unlock();
       }
   }
};

} // namespace nikola::executor

7.3 Task Queue and Scheduling


C++




// include/nikola/executor/task_scheduler.hpp
#pragma once
#include <queue>
#include <thread>
#include <future>
#include "nikola/executor/vm_pool.hpp"
#include "nikola/proto/neural_spike.pb.h"

namespace nikola::executor {

struct Task {
   std::string identity; // ZMQ Routing ID
   nikola::CommandRequest request;
   int priority; // 0=Critical... 3=Low
   
   // Comparator for Priority Queue
   bool operator<(const Task& other) const {
       return priority > other.priority; // Lower int = Higher priority
   }
};

class TaskScheduler {
   VMPool& vm_pool;
   std::priority_queue<Task> queue;
   std::mutex queue_mutex;
   std::condition_variable queue_cv;
   std::vector<std::thread> workers;
   bool running = true;
   
public:
   TaskScheduler(VMPool& pool, int concurrency) : vm_pool(pool) {
       for(int i=0; i<concurrency; ++i) {
           workers.emplace_back(&TaskScheduler::worker_loop, this);
       }
   }
   
   void submit(Task t) {
       std::lock_guard<std::mutex> lock(queue_mutex);
       if (queue.size() >= 1000) throw std::runtime_error("Queue Full");
       queue.push(t);
       queue_cv.notify_one();
   }
   
private:
   void worker_loop() {
       while(running) {
           Task task;
           {
               std::unique_lock<std::mutex> lock(queue_mutex);
               queue_cv.wait(lock, [this]{ return!queue.empty() ||!running; });
               if (!running) return;
               task = queue.top();
               queue.pop();
           }
           
           execute_task(task);
       }
   }
   
   void execute_task(Task& t) {
       // 1. Acquire VM
       auto vm = vm_pool.acquire(); 
       
       // 2. Configure Permissions (Hypervisor Level)
       vm->configure_resources(t.request.resources());
       
       // 3. Send Command (Secure Channel)
       auto req_packet = SecureChannel::wrap_message(wrap_in_spike(t.request), 0);
       vm->send_data(req_packet);
       
       // 4. Read Response (IOGuard)
       //... (Read loop implementation)...
       
       // 5. Release VM
       vm_pool.release(vm);
       
       // 6. Callback via ZeroMQ (implemented in Main Loop via callback queue)
   }
};

} // namespace nikola::executor

________________
8. INTEGRATION SCENARIOS AND WORKFLOWS
8.1 The "Physics Oracle" Verification Workflow
The critical safety mechanism for self-improvement is the Physics Oracle. This workflow demonstrates how the Executor facilitates this.
1. Generation: The Self-Improvement Engine (in the Core) generates a new candidate physics kernel (candidate.so) intended to replace the active Wave Propagation Kernel.
2. Submission: A CommandRequest is sent to the Executor.
   * command: /usr/bin/g++ (to compile the source).
   * permissions: ['dev:compiler', 'fs:write_tmp'].
   * resources: cpu_shares: 1024 (High priority).
3. Compilation: The Executor runs the compilation in a VM. The output (candidate.so) is retrieved.
4. Verification: A second CommandRequest is sent.
   * command: /usr/bin/physics_oracle_runner (a pre-installed tool in the Gold Image).
   * args: ['/tmp/candidate.so'].
   * permissions: ['base:execute'].
5. Oracle Execution: The Oracle Runner loads the shared object inside the VM. It runs millions of test cases, checking for energy conservation ($dH/dt = 0$) and symplectic integrity.
6. Adjudication:
   * If the Oracle returns EXIT_SUCCESS: The Executor signs the binary and returns it to the Core. The Core hot-swaps the library.
   * If the Oracle returns EXIT_FAILURE (Energy violation detected): The Executor discards the binary. The Orchestrator logs the failure and penalizes the Self-Improvement agent's dopamine levels.
This workflow ensures that dangerous code is never loaded into the main process space until it has been mathematically proven safe within the sandbox.1
8.2 Adversarial Red Teaming
The Adversarial Code Dojo uses the Executor to run "Red Team" attacks against the system's own code.
1. Attack Generation: The Dojo generates a malicious input (e.g., a "JSON Bomb" or a resonant frequency designed to cause overflow).
2. Containment: The Executor spins up a VM running a clone of the Nikola Core components.
3. Detonation: The malicious input is injected into the clone.
4. Observation: The Executor monitors the VM. If the clone crashes, hangs, or consumes excessive memory, the attack is successful.
5. Learning: The vulnerability is reported, and the Self-Improvement Engine is tasked with patching the code.
________________
9. OPERATIONAL PROCEDURES AND FAILURE RECOVERY
9.1 VM Zombie Management
Despite precautions, VMs can hang (kernel panic, infinite loop). The VMPool implements a "Dead Man's Switch." If a VM is marked as Assigned for longer than the task's timeout_ms + 5 seconds buffer, the VMPool explicitly calls virDomainDestroy (equivalent to pulling the power cord). The VM struct is discarded, and a new VM is asynchronously booted to replenish the pool.
9.2 Host Resource Exhaustion
If the Host runs out of RAM, the OOM Killer might target the Executor or its VMs. To prevent this, the Executor Service is configured with systemd resource limits (MemoryMax=16G). If the Executor approaches this limit, it flushes the Warm VM Pool (destroying idle VMs) to free memory.
9.3 Emergency SCRAM
In the event of a catastrophic runaway (e.g., the AI generating tasks faster than they can be processed), the operator can issue a SCRAM command via the separate Control Plane socket (see finding CTL-01 in 1). This command bypasses the standard Task Queue. The Executor immediately:
1. Pauses all running VMs.
2. Flushes the Task Queue.
3. Rejects all incoming ZMQ connections.
4. Enters a "Safe Mode" awaiting manual intervention.
________________

## Adversarial Code Dojo Genetic Algorithm Specification (GAP-035)

**SOURCE**: Gemini Deep Research Round 2 - Advanced Cognitive Dynamics Report
**INTEGRATION DATE**: 2025-12-15
**GAP ID**: GAP-035
**PRIORITY**: CRITICAL
**STATUS**: SPECIFICATION COMPLETE

### Problem Analysis: The Autoimmune Imperative

The Nikola architecture's reliance on resonant wave physics introduces a class of vulnerabilities unknown to discrete systems. Specifically, the system is susceptible to **"Resonance Cascades"**—runaway positive feedback loops where energy violates conservation laws ($dH/dt > 0$). If an external input (or an internal thought loop) happens to match the natural eigenfrequencies of the torus perfectly, the amplitude of the wavefunction can grow exponentially, leading to numeric overflow ("Decoherence") and a system crash.

Standard fuzz testing (injecting random noise) is insufficient because resonance is a precise, narrow-band phenomenon. Random noise is unlikely to trigger a cascade. To robustly test the system, we require an **Active Adversary**—an intelligent agent that actively searches the phase space for geometric singularities and energy leaks.

The **Adversarial Code Dojo** functions as the system's **autoimmune system**. It employs a **Genetic Algorithm (GA)** to evolve "Waveform Viruses"—input patterns specifically designed to destabilize the Physics Engine. If the Dojo cannot break the system, we can assert a high degree of confidence in its thermodynamic stability.

### Genotype Definition: The WaveChromosome

We represent an attack not as a raw data stream (PCM audio or text) but as a **parametric definition of a 9D energy injection sequence**. This allows the GA to operate on the "genes" of the attack (frequency, phase, timing) rather than the surface-level data.

The **WaveChromosome** consists of a sequence of **WaveGene** structures. Each gene represents a coherent pulse injected into the manifold.

```cpp
struct WaveGene {
   // Timing of the pulse relative to attack start (0.0 - 1.0)
   float time_offset_normalized;

   // Target spatial location (128-bit Hilbert Index)
   // Determines WHERE in the torus the energy is injected.
   uint64_t target_hilbert_idx;

   // Complex Amplitude (Energy injection vector)
   // Represented in polar form to facilitate phase mutations.
   float magnitude;      // Intensity
   float phase;          // Radians [0, 2π]

   // Frequency components (Harmonic Signature)
   // 9 values corresponding to the 9 manifold dimensions.
   // Attacks often target specific dimensional resonances (e.g., Time dimension).
   std::array<float, 9> frequency_signature;
};

struct WaveChromosome {
   std::vector<WaveGene> sequence;

   // Meta-parameters controlling global attack dynamics
   float global_gain;
   float tempo_scaling;
};
```

### Physics-Aware Mutation Operators

Standard genetic operators (bit-flipping) are ineffective in the continuous domain of wave mechanics. We define four specialized mutation operators that exploit the specific physics of the UFIE to induce instability.

#### Operator 1: Phase Conjugation (The "Mirror" Mutation)

**Physics Principle**: Constructive interference maximizes energy, while destructive interference minimizes it. Rapidly switching between phase $\theta$ and $\theta + \pi$ creates "pump" effects that can destabilize numerical integrators (like the symplectic integrator used in Nikola).

**Algorithm**: Select a gene at random. Invert its phase: $\phi_{new} = (\phi_{old} + \pi) \pmod{2\pi}$. This attempts to create sudden shocks in the energy manifold.

#### Operator 2: Resonant Drift (The "Hunter" Mutation)

**Physics Principle**: The 9D Torus has natural resonant frequencies derived from the Golden Ratio emitters ($f_n = \pi \cdot \phi^n$). Attacks tuned to these exact frequencies maximize energy transfer efficiency (Resonance).

**Algorithm**:
1. Select a gene.
2. Identify the nearest "Golden Harmonic" in the system's spectrum.
3. Shift the gene's frequency_signature closer to that harmonic by a small step $\delta$: $f_{new} = f_{old} + \alpha(f_{target} - f_{old})$.

This allows the attack to "lock on" to the system's vulnerabilities.

#### Operator 3: Amplitude Spike (The "Hammer" Mutation)

**Physics Principle**: This operator tests the nonlinearity saturation limits of the UFIE ($\beta |\Psi|^2 \Psi$). It attempts to push the local wavefunction amplitude beyond the balanced nonary limit ($[-4, +4]$) to trigger overflow or clipping artifacts.

**Algorithm**: Select a gene. Multiply its magnitude by a factor $K \in [1.5, 5.0]$.

#### Operator 4: Spatial Focusing (The "Lens" Mutation)

**Physics Principle**: Energy density, not just total energy, drives nonlinearity. Concentrating multiple pulses onto a single geodesic intersection point ("Caustic") can create a local singularity even if the global energy is low.

**Algorithm**: Select $N$ genes. Change their target_hilbert_idx to cluster around a single spatial point. Adjust their time_offset values to ensure simultaneous arrival, accounting for the wave velocity $c$ and the geodesic distance.

### Fitness Function Specification

The fitness function $F$ guides the evolution of the attacks. Unlike standard optimization where we minimize error, here we **maximize instability**. The function rewards attacks that violate energy conservation or produce numerical anomalies.

$$F(\text{Chromosome}) = w_1 \cdot \max(|\Delta H|) + w_2 \cdot \max(|\Psi|) + w_3 \cdot N_{NaN} + w_4 \cdot T_{diverge}$$

Where:

- $\max(|\Delta H|)$: Maximum deviation from the Hamiltonian (Total Energy) observed during simulation. Primary metric of "breaking physics".
- $\max(|\Psi|)$: Peak amplitude reached. We want to find waves that grow unbounded.
- $N_{NaN}$: Number of NaN (Not a Number) values produced. If $N_{NaN} > 0$, fitness is set to infinity (maximal success for attacker).
- $T_{diverge}$: Inverse time to divergence ($1/t_{crash}$). Faster crashes are considered "better" attacks.
- $w_n$: Weights. Typically $w_1=1000$ (Energy conservation paramount), $w_2=10$, $w_3=10^6$ (NaN is the goal).

### Genetic Algorithm Execution Lifecycle

The Adversarial Dojo operates in a **sandboxed environment** to prevent actual system damage.

1. **Initialization**: Population of 100 WaveChromosomes is generated. Some are random; others are seeded with known "dangerous patterns" from previous runs.

2. **Simulation (The Dojo)**: Each chromosome is loaded into a **KVM Sandbox** running an isolated instance of the Physics Engine.

3. **Oracle Monitoring**: The **Physics Oracle** monitors the simulation for 1000 ticks, calculating the Hamiltonian $H$ at every step.

4. **Fitness Evaluation**: The fitness $F$ is computed based on the telemetry from the Oracle.

5. **Selection**: **Tournament Selection** (size = 4) selects parents for the next generation.

6. **Reproduction**:
   - **Crossover**: Two-point crossover splices the "rhythm" of one attack with the "harmonic signature" of another.
   - **Mutation**: The physics-aware operators (Mirror, Hunter, Hammer, Lens) are applied with probability $P_{mut} = 0.2$.

7. **Elitism**: The top 5 attacks are preserved unchanged to ensure monotonically increasing lethality.

8. **Convergence**: The loop terminates if $F > F_{critical}$ (System Broken) or after 100 generations (System Robust).

### Implementation Strategy: AdversarialMutator

```cpp
void AdversarialMutator::mutate(WaveChromosome& chromo) {
   std::uniform_real_distribution<float> dist(0.0f, 1.0f);

   for (auto& gene : chromo.sequence) {
       // High mutation rate (5%) to encourage exploration of phase space
       if (dist(rng_) < 0.05f) {
           int op = rand() % 4;
           switch(op) {
               case 0: // Phase Conjugation
                   // Flip phase by PI to create destructive interference
                   gene.phase = std::fmod(gene.phase + std::numbers::pi_v<float>,
                                        2.0f * std::numbers::pi_v<float>);
                   break;
               case 1: // Resonant Drift
                   // Nudge frequency towards Golden Ratio harmonics
                   drift_to_golden_ratio(gene);
                   break;
               case 2: // Amplitude Spike
                   // Test non-linear saturation
                   gene.magnitude *= (1.5f + dist(rng_) * 2.0f);
                   break;
               case 3: // Spatial Shift
                   // Random walk on Hilbert curve to find weak metric regions
                   gene.target_hilbert_idx = mutate_hilbert_index(gene.target_hilbert_idx);
                   break;
           }
       }
   }
}
```

This GA ensures that the Nikola system is constantly subjected to "stress tests" that are mathematically targeted at its theoretical weaknesses, ensuring that the deployed physics kernel is resilient against even the most sophisticated resonant attacks.

### Integration with KVM Executor

The Adversarial Dojo leverages the KVM Executor infrastructure:

1. **Sandboxed Execution**: Each chromosome test runs in an isolated VM with full system snapshot
2. **Resource Limits**: Dojo VMs have strict CPU/memory limits (1 CPU core, 2GB RAM)
3. **Timeout Protection**: Tests automatically terminate after 10 seconds (10,000 physics ticks)
4. **Physics Oracle Integration**: Oracle telemetry extracted via secure guest channel
5. **Automated SCRAM**: If fitness exceeds critical threshold, vulnerability report generated

### Operational Integration

**Trigger Conditions**:
- Run during idle periods (Boredom > 0.7)
- Automated nightly regression testing
- After any Physics Engine code modifications
- On-demand via operator command

**Success Criteria**:
- No crashes after 100 generations: System considered robust
- $\max(|\Delta H|) < 0.001\%$: Energy conservation verified
- $\max(|\Psi|) < 4.5$: Nonlinear saturation working correctly
- $N_{NaN} = 0$: No numeric overflow

### Implementation Status

- **Status**: SPECIFICATION COMPLETE
- **Integration**: KVM Executor sandboxing + Physics Oracle telemetry
- **Population**: 100 chromosomes (random + seeded dangerous patterns)
- **Operators**: 4 physics-aware mutations (Mirror, Hunter, Hammer, Lens)
- **Fitness**: Energy violation + amplitude + NaN count + divergence time
- **Convergence**: 100 generations or F > F_critical

### Cross-References

- [KVM Executor Sandbox](./04_executor_kvm.md)
- [Physics Oracle](../02_foundations/02_wave_interference_physics.md)
- [Symplectic Integrator](../02_foundations/02_wave_interference_physics.md)
- [Golden Ratio Emitters](../07_multimodal/01_cymatic_transduction.md)
- [Balanced Nonary Saturation](../02_foundations/03_balanced_nonary_logic.md)
- [Boredom Drive](../05_autonomous_systems/01_computational_neurochemistry.md) - GAP-036

---

________________
10. CONCLUSION
This specification delivers a robust, implementation-ready blueprint for the Nikola Model Executor. By moving to a Hybrid Deployment Architecture, we resolve the critical stability issues of nested virtualization. By implementing the Secure Guest Channel and IOGuard, we inoculate the host against the inevitable attempts by the cognitive core (whether accidental or adversarial) to breach its containment. The Permission Model provides the granular control necessary to allow powerful self-improvement while adhering to strict thermodynamic and safety constraints. The provided C++23 implementations for the ZeroMQ spine, secure channel, and task scheduler provide a direct path to code realization.
Status: APPROVED for Code Generation.
Next Steps: Begin implementation of src/executor following the file structure defined in Section 26.2 of the plan.
Works cited
1. part_1_of_9.txt