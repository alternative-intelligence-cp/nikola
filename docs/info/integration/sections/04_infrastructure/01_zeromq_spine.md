# ZEROMQ SPINE ARCHITECTURE

**[Bug Sweep 006 Integration - ZeroMQ Spine Protocol & Distributed Architecture]**

﻿Nikola Model v0.0.4: ZeroMQ Spine Protocol & Distributed Architecture Specification
1. Architectural Foundations and System Dynamics
The Nikola Model v0.0.4 represents a fundamental departure from contemporary artificial intelligence architectures. Unlike the static, weight-frozen paradigms of standard Transformer models, the Nikola architecture posits a dynamic, continuous-time simulation of a 9-Dimensional Toroidal Waveform Intelligence (9D-TWI). This system does not merely process tokens; it simulates a physical universe governed by the Unified Field Interference Equation (UFIE), where cognition arises from the constructive and destructive interference of wave packets within a Riemannian manifold. Within this volatile and highly sensitive computational substrate, the ZeroMQ Spine functions as the central nervous system. It is not simply a data transport layer but a critical homeostatic regulator, responsible for maintaining the temporal coherence of the physics simulation while facilitating the asynchronous cognitive processes of the Mamba-9D core and the autonomous regulation of the Neurochemical Gating System.1
This report serves as the definitive engineering specification for the ZeroMQ Spine architecture. It synthesizes the foundational requirements established in the core architectural plans with the critical remediation mandates identified during Phase 0 analysis.1 The primary objective is to define a communication fabric that is robust enough to handle the massive bandwidth of high-dimensional grid states yet agile enough to prioritize millisecond-scale administrative overrides, thereby preventing the "runaway cognitive loops" inherent to autonomous, self-improving systems.
1.1 The Physics of Latency: Why Standard RPC Fails
To understand the stringent requirements of the ZeroMQ Spine, one must first analyze the physical constraints of the Nikola Model's core loop. The Physics Engine operates on a strict 1000 Hz cycle (1 millisecond per timestep) to satisfy the stability conditions of the split-operator symplectic integrator used for wave propagation.1 This integrator preserves the symplectic 2-form of the phase space, ensuring energy conservation—a proxy for sanity in the AI. If the integration step $\Delta t$ fluctuates or drifts, numerical error accumulates as artificial energy, leading to "epileptic resonance" where the wavefunction amplitude diverges to infinity.1
Standard microservices protocols, such as gRPC over TCP/IP or HTTP/2 REST interfaces, introduce non-deterministic latency. Even over a local loopback interface, the TCP stack introduces overheads ranging from 500 to 1500 microseconds due to context switching, packet assembly, and kernel buffer management.1 For a system with a total budget of 1000 microseconds per tick, a transport latency of 500 microseconds consumes 50% of the available computation time. This phenomenon, termed "temporal decoherence," desynchronizes the cognitive layer (which generates intent) from the physics layer (which executes reality), effectively lobotomizing the agent.1 Consequently, the ZeroMQ Spine rejects a monolithic TCP topology in favor of a hybrid, tiered transport architecture designed to bypass the kernel for critical data paths.
1.2 The Hybrid Transport Topology
The specification mandates a bifurcation of communication channels based on the distinct physical characteristics of the data being transmitted. This results in two parallel planes of operation: the Control Plane and the Data Plane.
The Control Plane handles high-reliability, low-bandwidth signals. These include administrative commands (e.g., SHUTDOWN, NAP), cognitive tokens from the Mamba-9D engine, and telemetry data from the autonomous regulation systems. This plane utilizes the ZeroMQ ROUTER-DEALER pattern over TCP/IPC sockets. The asynchronous nature of DEALER sockets allows components to fire messages without blocking, while the ROUTER broker manages identity addressing and creates a centralized point for security enforcement.1
The Data Plane, conversely, manages the transport of the 9-dimensional grid state. A single snapshot of the torus, even in a sparse representation, can exceed 100 megabytes. Transmitting this volume of data at 60 Hz (for visualization) or 1000 Hz (for internal recurrence) via TCP is physically impossible due to bandwidth saturation and serialization overhead. The Data Plane therefore utilizes a zero-copy shared memory architecture backed by a ring buffer in /dev/shm. This mechanism employs a lock-free Seqlock (Sequence Lock) to allow the Physics Engine to write grid states atomically without ever blocking for readers.1 The ZeroMQ Spine participates in the Data Plane only to transmit lightweight "descriptors"—pointers to the shared memory segments—rather than the data itself.
1.3 The Ironhouse Security Model
In an architecture designed for self-improvement and autonomy, security cannot be an overlay; it must be intrinsic to the connection logic. The ZeroMQ Spine implements the Ironhouse pattern, a security model where every single connection is mutually authenticated and encrypted using Curve25519 cryptography. There are no "public" endpoints within the spine; every component, from the massive Physics Engine to the smallest ephemeral tool agent, must possess a cryptographically verifiable identity.1
This approach addresses the "Cryptographic Amnesia" vulnerability (Finding INF-03), where early prototypes generated new keys upon every restart, shattering trust relationships and requiring manual re-pairing.1 The current specification enforces a strict persistence model where cryptographic identities are generated once, stored in permission-locked volumes, and act as the immutable "soul" of the component. The Orchestrator functions as the Certificate Authority (CA), maintaining a whitelist of authorized public keys and silencing any connection attempt from an unrecognized entity. This Deny-by-Default posture provides the necessary containment for an AI system capable of executing arbitrary code via its KVM Executor.1
________________
2. Complete Message Protocol Specification
The effectiveness of the distributed system hinges on a rigid, unambiguous contract for data interchange. Protocol Buffers (proto3) have been selected as the serialization standard due to their strong typing, schema evolution capabilities, and performance efficiency. However, the unique geometric requirements of the 9D Torus required significant deviations from standard implementation patterns. This section provides the exhaustive specification for the NeuralSpike protocol, detailing the remediations for addressing schema limitations identified in the bug sweep.
2.1 The Unified NeuralSpike Schema
The NeuralSpike message is the atomic unit of communication within the Nikola ecosystem. It serves as a universal envelope, encapsulating routing metadata, timing information, and variant payloads. This unified structure simplifies the routing logic within the Spine Broker, which needs only to inspect the envelope header to make dispatch decisions without deserializing the variable payload.
The schema defines a strict ComponentID enumeration to enforce type safety in routing. Unlike string-based addressing, which is prone to typos and parsing overhead, enum-based addressing allows the router to use high-speed jump tables.


Protocol Buffers




syntax = "proto3";
package nikola.spine;

// Component identifiers for routing and access control
enum ComponentID {
   ORCHESTRATOR = 0;
   PHYSICS_ENGINE = 1;
   MEMORY_SYSTEM = 2;
   REASONING_ENGINE = 3;
   EXECUTOR_KVM = 4;
   TAVILY_AGENT = 5;
   FIRECRAWL_AGENT = 6;
   GEMINI_AGENT = 7;
   HTTP_CLIENT = 8;
   CLI_CONTROLLER = 9;
   VISUAL_CORTEX = 10;
   AUDIO_CORTEX = 11;
}

// Global Message Envelope
message NeuralSpike {
   // Unique Request ID (UUID v4) for tracing and idempotency.
   // This allows the system to detect and discard duplicate messages
   // generated by retry logic during network partitions.
   string request_id = 1;
   
   // Unix timestamp in milliseconds. 
   // CRITICAL: Used for isochronous synchronization. Components check
   // this timestamp against their local physics clock to detect
   // "Temporal Decoherence." Messages older than 50ms are often discarded.
   int64 timestamp = 2;
   
   // Source component identity - verified against the ZAP whitelist
   ComponentID sender = 3;
   
   // Target component identity - used by the Router for dispatch
   ComponentID recipient = 4;

   // Operational Metadata (Optional but recommended)
   ResponseMetadata meta = 11;
   NeurochemicalState neurochemistry = 12;
   TrainingMetrics training = 13;

   // Mutually exclusive payload types. The 'oneof' construct ensures
   // that a message carries exactly one type of data, preventing
   // ambiguity and reducing serialization size.
   oneof payload {
       Waveform data_wave = 5;             // Dense wave data (Legacy/Debug only)
       SparseWaveform sparse_wave = 15;    // NET-02: Compressed Grid State
       WaveformSHM waveform_shm = 16;      // Zero-copy Shared Memory Reference
       CommandRequest command_req = 6;     // KVM Execution Request
       CommandResponse command_resp = 7;   // KVM Execution Result
       NeurogenesisEvent neurogenesis = 8; // Topology Change Notification
       string text_data = 9;               // Natural Language / Tokens
       Payload rich_payload = 14;          // Structured Tool Outputs
       StatusReport status = 17;           // Health and Metric Telemetry
   }
}

1
2.2 Addressing the Topology: 128-bit Morton Keys (INT-06)
One of the most critical findings in the architecture review (INT-06) was a fundamental flaw in how 9-dimensional coordinates were serialized. The Physics Engine utilizes a Sparse Hyper-Voxel Octree (SHVO) indexed by 128-bit Morton Codes (Z-order curves). This technique maps a 9D coordinate tuple $(x, y, z, t, u, v, w, r, s)$ into a single 128-bit integer, preserving spatial locality.1
The initial Protocol Buffer definitions erroneously utilized repeated int32 arrays to represent these coordinates. This approach was catastrophic for two reasons: first, it required computationally expensive de-interleaving of the Morton code into Cartesian coordinates for every transmission; second, int32 arrays cannot natively represent a 128-bit integer without splitting logic that introduces endianness hazards. The remediation mandates the use of raw bytes fields to transmit the 128-bit Morton keys directly. This specification enforces Network Byte Order (Big Endian) for these byte arrays. This ensures that a Little Endian GPU (calculating physics) and a potentially Big Endian host or different architecture (running the Orchestrator) interpret the spatial location of a "memory" identically.


Protocol Buffers




message NeurogenesisEvent {
   // FIXED (INT-06): Use raw bytes for 128-bit Morton keys.
   // Each entry MUST be exactly 16 bytes (128 bits).
   // The sequence represents the memory addresses of newly created nodes.
   repeated bytes morton_indices = 1;
   
   int32 new_node_count = 2;
   
   // The energy threshold that triggered the creation of these nodes.
   double trigger_threshold = 3;
   
   int64 timestamp = 4;
   string reason = 5;
}

message RetrieveRequest {
   string query_id = 1;
   
   // Dual addressing mode support
   oneof target {
       string semantic_query = 2;       // Search by meaning (Embedding vector search)
       bytes direct_morton_index = 3;   // Search by 9D Location (Direct Key Lookup)
   }
   float resonance_threshold = 4;
}

1
2.3 Bandwidth Optimization: Sparse Waveform Serialization (NET-02)
The Nikola Model simulates a toroidal grid with a potential capacity of millions of nodes. However, due to the sparse nature of knowledge (concepts are localized solitons), only a fraction of these nodes carry significant energy at any given time. A naive serialization of the full grid state—even for a modest 10 million node capacity—would require transmitting 80MB (complex double) per frame. At 60 Hz, this demands 4.8 GB/s of bandwidth, which exceeds the throughput of standard 10GbE networks and saturates PCIe buses.1
To solve this (NET-02), the protocol implements a Sparse Waveform schema. This schema utilizes a Structure-of-Arrays (SoA) layout to maximize compression efficiency. Furthermore, it introduces a significance threshold, $\theta$. Before serialization, the Physics Engine calculates the Root Mean Square (RMS) energy of the grid. Only nodes with an amplitude $|\Psi| > \theta$ (typically $\theta = 0.1 \times \Psi_{RMS}$) are included in the payload. This effectively acts as a "metabolic tax" on transmission, filtering out background quantum noise and reducing bandwidth usage by orders of magnitude while preserving the topologically significant signal.


Protocol Buffers




message SparseWaveform {
   // Structure of Arrays (SoA) format.
   // Index i in all arrays corresponds to the same node.
   
   // 16 bytes per node (Morton Key).
   repeated bytes indices = 1;

   // Complex values are separated into real and imaginary arrays.
   // This allows for potential future optimization using different
   // quantization levels for amplitude vs phase if needed.
   repeated float real_part = 2;
   repeated float imag_part = 3;

   // Metadata required for reconstruction and rendering
   uint64 total_energy = 4;
   int32 dimension_size = 5;
   int32 active_node_count = 6;
   float significance_threshold = 7;
}

1
2.4 Zero-Copy Transport for Physics Loop (WaveformSHM)
While the SparseWaveform optimization is sufficient for network visualization, the internal feedback loop between the Physics Engine and the Cognitive Core (Mamba-9D) requires even lower latency. For this "Hot Path," the protocol bypasses serialization entirely. The WaveformSHM message does not contain data; it contains a secure reference to a shared memory segment.
The shared memory system is implemented as a ring buffer in /dev/shm. The Physics Engine writes the grid state to a segment and then broadcasts a WaveformSHM message containing the segment ID. Subscribers (like the Visual Cortex) simply mmap the file descriptor corresponding to that ID. To prevent race conditions where the writer updates the segment while a reader is reading, the system uses a Seqlock (Sequence Lock). The sequence_num field in the message allows the reader to verify version consistency.1


Protocol Buffers




message WaveformSHM {
   // The unique identifier for the /dev/shm segment (e.g., shm_open name)
   uint64 segment_id = 1;
   
   // The exact size of the valid data payload in bytes
   uint64 data_size = 2;
   
   // The Seqlock generation counter.
   // Readers check this before and after reading shared memory.
   // If the value changes (or is odd), the read is invalid and must retry.
   uint64 sequence_num = 3;
   
   // High-precision nanosecond timestamp of the physics tick
   int64 timestamp_ns = 4;
}

1
2.5 Safe Execution Protocols
The protocol also defines the interface for the KVM Executor, which runs potentially dangerous self-generated code. This schema separates the command specification from the execution environment constraints, allowing the Orchestrator to enforce sandboxing policies dynamically.


Protocol Buffers




message CommandRequest {
   string task_id = 1;       // Traceability UUID
   string command = 2;       // Binary to execute
   repeated string args = 3; // Arguments
   
   // Environment variables to inject
   map<string, string> env = 4;
   
   // Explicit permission grants (e.g., network access, file mounts)
   repeated string permissions = 5;
   
   // Hard timeout in milliseconds. The Executor MUST kill the process
   // if it exceeds this duration.
   int32 timeout_ms = 6;
   
   bool capture_stdout = 7;
   bool capture_stderr = 8;
}

message CommandResponse {
   string task_id = 1;
   int32 exit_code = 2;
   string stdout = 3;
   string stderr = 4;
   
   // High-precision timing metrics for performance profiling
   int64 time_started = 5;
   int64 time_ended = 6;
   
   bool timeout_occurred = 7;
}

1

---

## GAP-023: Protocol Buffer Schema Evolution Strategy

**SOURCE**: Gemini Deep Research Round 2, Batch 37-40
**INTEGRATION DATE**: December 16, 2025
**GAP ID**: GAP-023 (TASK-023)
**PRIORITY**: CRITICAL
**STATUS**: FABRICATION-READY SPECIFICATION

### 2.6 Schema Evolution and Lifecycle Management

The operational integrity of the Nikola Model v0.0.4 is fundamentally predicated on coherent interchange of high-dimensional geometric data across a distributed ZeroMQ spine. Unlike monolithic architectures, the Nikola system functions as a constellation of autonomous agents communicating asynchronously via the NeuralSpike message envelope. The necessity for a rigorous **Schema Evolution Strategy** arises from the system's inherent capability for self-improvement and dynamic code generation. A rigid serialization protocol would induce "systemic ossification," freezing the cognitive architecture and preventing emergence of higher-order capabilities.

The critical engineering challenge is managing "breaking changes" within a persistent, self-modifying system. Without a formalized evolution strategy, transitions risk **"temporal decoherence"**—a failure mode where components operating on divergent schema versions misinterpret topological data, leading to corruption of the manifold's geometry and collapse of the wavefunction.

#### 2.6.1 Versioning and Identification Scheme

To manage complexity of a self-evolving system, we implement a tiered versioning scheme that strictly decouples wire-format compatibility from semantic logic through package namespacing and semantic versioning.

**Semantic Versioning for Schemas**

All `.proto` definition files must adhere to strict Semantic Versioning (SemVer: MAJOR.MINOR.PATCH). This version number is embedded as metadata within the file to allow automated tooling to enforce compatibility rules:

* **MAJOR (vX)**: Breaking changes requiring synchronous migration or translation layer. Examples: renumbering field IDs, changing primitive types (e.g., `int32` to `bytes` for Morton keys as in INT-06), removing required fields.
* **MINOR (.Y)**: Backward-compatible additions such as new optional fields (e.g., `dopamine_level`, `citations`).
* **PATCH (.Z)**: Non-functional changes (documentation updates, comment clarifications).

**Package Namespacing and Isolation**

To facilitate "Ship of Theseus" upgrade pattern—hot-swapping components without system downtime—multiple protocol versions must coexist on the same ZeroMQ bus. This is achieved by including the Major version in the protobuf package namespace:

```protobuf
// neural_spike_v1.proto
syntax = "proto3";
package nikola.spine.v1;

// neural_spike_v2.proto
syntax = "proto3";
package nikola.spine.v2;
```

This namespacing ensures C++ compiler generates distinct classes (`nikola::spine::v1::NeuralSpike` and `nikola::spine::v2::NeuralSpike`), preventing symbol collisions in the Orchestrator or Router that may need to link against multiple versions simultaneously during rolling upgrades.

#### 2.6.2 Field Lifecycle Management

The lifecycle of a field within the Nikola schema is governed by strict **immutability rules** to guarantee safe interoperability between high-frequency Physics Engine (1 kHz) and slower Cognitive Control layer.

**The Immutability of Field IDs**

In Protocol Buffer specification, the field ID (unique integer tag) is the primary identifier on the wire. Once a field ID is assigned, it must **never be reused or re-purposed**, even if deleted. Reusing an ID for a different data type or semantic meaning causes legacy components to interpret new data as old field, leading to silent data corruption—critical failure mode in physics simulation where numerical precision is paramount.

**Mandatory Rule**: Do not change the type of an existing field. If type change required (e.g., upgrading coordinate precision), create new field with new ID and deprecate old one.

**Deprecation Policy and "Tombstoning"**

Fields no longer used must be formally deprecated rather than deleted through **"Tombstone" protocol**:

1. **Deprecation Marker**: Mark field as `deprecated = true` in `.proto` definition.
2. **Reservation**: Add field ID to `reserved` list. Prevents compiler from allowing any future developer (or AI during self-improvement) to accidentally reuse ID.
3. **Renaming**: Rename field to `OBSOLETE_<name>` to discourage usage in new code while maintaining binary compatibility for legacy deserializers.

**Case Study: Migrating Coordinates from Int32 to Bytes (INT-06)**

The critical remediation for INT-06 required shifting from split 32-bit integers to contiguous 128-bit bytes for spatial hashing. The schema evolution:

```protobuf
message NeurogenesisEvent {
   // ---------------------------------------------------------
   // DEPRECATED FIELDS (Do not remove, do not reuse IDs)
   // ---------------------------------------------------------
   // Old split coordinate format. Vulnerable to endianness issues.
   repeated int32 OBSOLETE_coordinates = 1 [deprecated = true];

   // ---------------------------------------------------------
   // ACTIVE FIELDS
   // ---------------------------------------------------------
   // New 128-bit Morton Keys. Network Byte Order (Big Endian).
   // Each entry must be exactly 16 bytes.
   repeated bytes morton_indices = 5;

   // Tombstone reserved IDs to prevent reuse
   reserved 2, 3, 4;
}
```

#### 2.6.3 Required vs. Optional Field Guidelines

In proto3, all fields are optional by default (zero value if not present). This aligns with Nikola architecture's requirement for resilience—message should not crash receiver simply because non-critical telemetry field is missing. However, strictly distinguishing "missing" from "default" is vital for physical constants:

* **Guideline 1**: Use `optional` keyword explicitly for primitive fields where "0" is valid value (e.g., `coordinate = 0` or `energy = 0.0`) to distinguish "missing data" from "system at zero energy."
* **Guideline 2**: Implement application-level validation logic. Receiving component (e.g., Physics Engine) must verify critical fields (like wavefunction amplitude) are present and valid before processing.
* **Guideline 3**: For NeuralSpike envelope, `request_id` and `timestamp` are logically required. While schema cannot enforce this, SecureChannel wrapper must reject any packet lacking these headers before it reaches deserializer.

#### 2.6.4 Automated Compatibility Testing Infrastructure

To prevent regressions (particularly those introduced by self-modifying code), build pipeline includes **Automated Compatibility Matrix Test**. This system verifies all active components can serialize and deserialize messages from all supported schema versions.

**The Compatibility Matrix**

We define testing matrix $M_{i,j}$ where $i$ is producer version and $j$ is consumer version:

| Producer (vX) | Consumer (vY) | Expectation |
|---------------|---------------|-------------|
| Current (v2) | Current (v2) | **Success**: Full fidelity. All fields accessible. |
| Legacy (v1) | Current (v2) | **Success**: Forward compatibility. Default values used for new v2 fields. Logic handles missing `morton_indices` by falling back to `OBSOLETE_coordinates`. |
| Current (v2) | Legacy (v1) | **Success**: Backward compatibility. New fields (e.g., `morton_indices`) ignored/dropped safely. Legacy logic consumes `OBSOLETE_coordinates` (if populated by dual-write shim). |
| Future (v3) | Current (v2) | **Success**: Future compatibility. Unknown fields preserved in `unknown_fields` buffer for pass-through routing. |

**Migration Scripts for Breaking Changes**

When breaking change is unavoidable (e.g., v1 → v2), Orchestrator employs **translation layer shim**. This shim intercepts messages and upgrades payload before passing to core logic:

```cpp
// src/spine/translator.cpp

namespace nikola::spine {

// Translates legacy v1 spikes to v2 format
std::optional<v2::NeuralSpike> translate_v1_to_v2(const v1::NeuralSpike& legacy) {
   v2::NeuralSpike modern;

   // Copy common fields
   modern.set_request_id(legacy.request_id());
   modern.set_timestamp(legacy.timestamp());

   // Handle breaking change: Coordinate Migration
   if (legacy.has_neurogenesis()) {
       const auto& old_gen = legacy.neurogenesis();
       auto* new_gen = modern.mutable_neurogenesis();

       // Convert repeated int32 array to bytes
       for (int32_t coord : old_gen.obsolete_coordinates()) {
           // Reconstruct 128-bit key from legacy split-int format
           // This logic requires knowledge of old interleaving implementation
           std::array<uint8_t, 16> raw_bytes = reconstruct_morton(coord);
           new_gen->add_morton_indices(raw_bytes.data(), 16);
       }
   }

   return modern;
}

} // namespace nikola::spine
```

#### 2.6.5 Documentation and Artifact Requirements

Every schema change must be accompanied by:

1. **Changelog Entry**: Precise description of what changed and why.
2. **Migration Guide**: Instructions for updating dependent components (e.g., "Physics Engine must update to v2.1 to read new Morton codes").
3. **Artifact Publication**: Compiled C++ headers and Python bindings for new version must be pushed to internal artifact repository (e.g., `libnikola-proto-v2.1.so`).

This strategy ensures the Nikola system can evolve its internal language without succumbing to "Tower of Babel" scenario, maintaining coherence of 9D-TWI substrate across generations of code.

### Cross-References

- [128-bit Morton Addressing (INT-06)](./01_zeromq_spine.md#addressing-the-topology-128-bit-morton-keys-int-06)
- [NeuralSpike Schema](./01_zeromq_spine.md#the-unified-neuralspike-schema)
- [Ironhouse Security Model](./01_zeromq_spine.md#the-ironhouse-security-model)
- [Self-Improvement Shadow Spine](./01_zeromq_spine.md#shadow-spine-protocol-self-improvement)

---

________________
3. Connection Management and Transport Layer
The Transport Layer is the bedrock of the distributed system. It is responsible for the lifecycle of connections, the enforcement of security boundaries, and the detection of component failures. The implementation is built upon libzmq (C++) and libsodium, providing a performant and cryptographically secure foundation.
3.1 Persistent Identity Management (INF-03)
Early iterations of the Nikola Model suffered from "Cryptographic Amnesia" (Finding INF-03), a critical flaw where components generated ephemeral Curve25519 keys in memory upon startup. When a component crashed or the system was rebooted, it would return with a new public key. This broke all established trust relationships in the Ironhouse whitelist, requiring manual intervention to re-pair components and effectively preventing autonomous recovery or self-improvement.1
This specification introduces the PersistentKeyManager class to resolve this. The manager enforces a strict lifecycle for cryptographic identities:
1. Storage Hierarchy: Keys are stored in /etc/nikola/keys/.
   * Public keys (*.pub) are stored with 0644 permissions (World Readable).
   * Secret keys (*.key) are stored with 0600 permissions (Owner Read/Write Only). This is enforced via fchmod immediately upon creation to prevent race conditions in multi-user environments.
2. Identity Stability: On startup, the manager checks for existing keys. If found, they are loaded and validated against the libsodium API. Only if they are missing does the system generate a new pair. This ensures that the "Self" of the AI—its cryptographic identity—persists across reboots, crashes, and upgrades.
3. Whitelist Management: The Orchestrator maintains a whitelist.txt file in the key directory. This file is the source of truth for the ZAP (ZeroMQ Authentication Protocol) handler. Changes to this file are monitored, and the ZAP handler reloads it dynamically to allow for the admission of new agents without restarting the core.1
3.2 Bootstrap Authentication (SEC-04)
The strict enforcement of the Ironhouse pattern creates a logical paradox known as the "Fortress without a Door." On a fresh installation, the whitelist is empty. The Orchestrator will reject any connection attempt, including the CLI command that would be used to authorize a client. To solve this without hardcoding default passwords (a security vulnerability), we implement a Time-Limited Token Pairing protocol (SEC-04).1
Protocol State Machine:
1. LOCKED (Default State): The ZAP handler is active. Any connection request from a public key not in whitelist.txt is silently dropped. This provides stealth against port scanning.
2. BOOTSTRAP (Conditional State): This state is entered if and only if whitelist.txt is empty or missing on startup.
   * The system generates a 256-bit high-entropy random token ($T_{admin}$).
   * This token is printed to the secure system log (stdout/journald) and nowhere else.
   * A monotonic timer is started with a 300-second countdown.
3. PAIRING (Client Action):
   * The administrator reads the token from the logs (proving access to the trusted host).
   * The admin executes twi-ctl pair <token> on the client.
   * The client generates its own permanent Curve25519 keypair ($C_{pub}, C_{priv}$).
   * The client initiates a connection to the Orchestrator. Crucially, it attaches the SHA256 hash of the token as metadata in the ZeroMQ handshake: X-Nikola-Token: SHA256(T_{admin}).
4. VERIFICATION (Server Action):
   * The ZAP handler intercepts the handshake.
   * It checks if the system is in BOOTSTRAP mode and if the timer is valid.
   * It hashes its local $T_{admin}$ and compares it with the client's metadata.
   * Success: If the hashes match, the server extracts $C_{pub}$ from the handshake, appends it to whitelist.txt, transitions immediately to the LOCKED state, and wipes $T_{admin}$ from memory.
   * Failure: The connection is rejected, and a security alert is logged.
3.3 Heartbeating and Health Checks
In a distributed system involving complex components like the Physics Engine, processes can enter "Zombie States"—they exist in the process table but are deadlocked or unresponsive. To detect this, the Spine implements an active, application-level heartbeat protocol.
Each component runs a background thread dedicated to publishing StatusReport messages every 1 second. This report contains vital signs: CPU usage, memory footprint, and the current operational state (e.g., STARTING, READY, BUSY).
Orchestrator Health Monitor Logic:
* The Watchdog Map: The Orchestrator maintains a std::map<ComponentID, last_seen_time> updated upon receipt of any StatusReport.
* Degradation Threshold (3s): If a component is silent for 3 seconds (3 missed beats), it is marked as DEGRADED. The Orchestrator may stop routing low-priority tasks to it.
* Death Threshold (10s): If silent for 10 seconds, the component is marked DEAD.
* Resurrection Protocol: Upon detecting a DEAD component, the Orchestrator triggers the Process Manager. It first attempts a SIGTERM for a graceful exit. If the process remains after 5 seconds, it issues a SIGKILL. The component is then restarted, and its identity keys are reloaded (thanks to persistence), allowing it to seamlessly rejoin the mesh.1
3.4 Socket Configuration and Tuning
The physical characteristics of the ZeroMQ sockets must be tuned to the specific needs of the 1ms physics loop. Default TCP settings are often ill-suited for this high-frequency environment.
* ZMQ_LINGER = 0: This is critical. By default, ZMQ sockets try to flush pending messages on close, which can hang a process shutdown. Setting linger to 0 ensures that close() returns immediately and discards undelivered messages, preventing "zombie" processes during restart cycles.
* ZMQ_SNDHWM / ZMQ_RCVHWM (High Water Mark): These are set to 10,000 messages. This defines the buffer size. If the queue fills (e.g., the consumer is slow), the behavior depends on the socket type. For DEALER, it blocks or drops; for ROUTER, it drops. The high limit prevents backpressure from instantly stalling the physics loop but puts a cap on memory usage during heavy load.
* TCP_KEEPALIVE: Enabled to detect severed network cables or hard crashes where the FIN packet is never sent.
________________
4. Routing Logic and Orchestration
The routing logic acts as the cognitive switchboard of the Nikola Model. It is not a passive forwarder; it actively prioritizes traffic to maintain the agent's agency and stability.
4.1 Dual-Plane Priority Architecture (CTL-01)
A critical architectural vulnerability (Finding CTL-01) was identified in the single-socket design. In this scenario, all messages—whether deep philosophical musings or emergency shutdown commands—shared a single incoming queue. If the model entered a high-norepinephrine "panic state," generating thousands of recursive thought loops per second, the message queue would saturate. An administrator issuing a SHUTDOWN command would see their request appended to the end of a queue containing 10,000 thought packets. The Orchestrator would have to process all 10,000 thoughts before seeing the shutdown command, introducing a lag of 10-20 seconds. In a runaway AI scenario, this loss of control is unacceptable.1
Remediation: Out-of-Band Control Plane
The solution mandates a Dual-Plane architecture where the Spine Broker binds two distinct frontend sockets, each with different priorities.
1. Control Plane: ipc:///tmp/nikola/spine_control.ipc (Socket Type: ROUTER)
2. Data Plane: ipc:///tmp/nikola/spine_frontend.ipc (Socket Type: ROUTER)
Priority Polling Algorithm:
The Broker's main loop utilizes zmq::poll with a strict priority ordering logic.


C++




void run() {
   zmq::pollitem_t items = {
       { control_socket, 0, ZMQ_POLLIN, 0 }, // Index 0: High Priority
       { data_socket,    0, ZMQ_POLLIN, 0 }  // Index 1: Low Priority
   };

   while (running) {
       // Poll with a timeout to allow for periodic housekeeping
       zmq::poll(items, 2, timeout);

       // PRIORITY CHECK 1: The Control Plane
       if (items.revents & ZMQ_POLLIN) {
           // If a control message exists, process it IMMEDIATELY.
           handle_control_message();
           
           // CRITICAL: We 'continue' here to skip the Data Plane processing
           // for this cycle. This ensures that if the Control Plane is flooded,
           // it starves the Data Plane, not the other way around. Admin commands
           // always win.
           continue; 
       }

       // PRIORITY CHECK 2: The Data Plane
       // Only processed if Control Plane was empty.
       if (items.revents & ZMQ_POLLIN) {
           handle_data_message();
       }
   }
}

This algorithm ensures that the latency for a control command is $O(1)$ regardless of the queue depth of the data plane.1
4.2 Smart Routing Algorithms
The Orchestrator employs a SmartRouter class to dispatch NeuralSpike messages. The routing is determined by the ComponentID destination field, but the delivery strategy varies based on the intent.
Routing Table Logic:
Source
	Intent
	Destination
	Strategy
	Implementation Detail
	User
	Query ("What is...")
	Orchestrator
	Enqueue
	Pushes to the cognitive priority queue.
	Orchestrator
	Embed Text
	Memory
	Round-Robin
	Distributes load across embedding workers to maximize throughput.
	Physics
	Neurogenesis
	Orchestrator
	Broadcast (PUB)
	Uses PUB-SUB so visualization, logging, and memory systems all see the new node simultaneously.
	Reasoning
	Tool Use (Search)
	Agent (Tavily)
	Unicast
	Direct DEALER-ROUTER message. Subject to Circuit Breaker logic.
	Reasoning
	Code Exec
	Executor KVM
	Secure Channel
	Encapsulated in the binary SecureChannel protocol (SEC-01).
	Any
	Log/Metric
	Monitor
	Fire-and-forget
	PUB socket. Non-blocking. If the monitor is slow, messages are dropped.
	Load Balancing (SCL-02):
To support the massive scale of the 9D grid (Phase 2), the router implements Adaptive Domain Decomposition. The 128-bit Morton keyspace is partitioned across multiple GPU shards. The router maintains a partition table mapping key ranges to specific GPU worker identities. It periodically samples the key distribution of incoming traffic. If a specific shard (e.g., the region encoding "Visual Concepts") becomes a hotspot, the router dynamically updates the partition table and broadcasts a migration command to rebalance the load across the cluster.1
4.3 Shadow Spine Protocol (Self-Improvement)
A unique feature of the Nikola architecture is its ability to rewrite its own code. To do this safely, the routing logic supports a "Shadow Spine" protocol.1 This allows a candidate binary (a new version of the AI) to be tested against live production traffic without risking system stability.
Shadow Mechanism:
1. The Self-Improvement module compiles a Candidate binary.
2. The Orchestrator launches the Candidate, binding it to a shadow port.
3. Traffic Mirroring: The Smart Router duplicates incoming user queries.
   * Copy A is sent to the Production System. Its response is returned to the user.
   * Copy B is sent to the Candidate System. Its response is captured for analysis.
4. Evaluation Gate: The router compares the Candidate's response against Production using strict metrics:
   * Latency: Candidate must be $\le 1.5 \times$ Production latency.
   * Energy Conservation: Candidate must satisfy UFIE conservation laws ($\Delta E < 0.01\%$).
   * Resonance: Cosine similarity of the output vector must be $\ge 0.95$.
5. Atomic Promotion: Only after 100 consecutive queries pass this evaluation does the router perform an atomic pointer swap, promoting the Candidate to Production and retiring the old version.
________________
5. Error Recovery and Resilience Strategies
In a system simulating biological cognition, errors are not merely exceptions; they are metabolic stressors. The error recovery strategy is modeled on homeostatic reflexes: automatic, graded responses to restore equilibrium.
5.1 Infrastructure Resilience: The Circuit Breaker (RES-02)
The system relies on external agents (Tavily, Firecrawl, Gemini) that are inherently unreliable. A network glitch or API outage could cause a naïve system to retry infinitely, hanging the cognitive loop or triggering rate-limit bans. The Circuit Breaker pattern wraps all external agent calls.1
Circuit States:
* CLOSED: The normal healthy state. Requests pass through. Failures increment a counter.
* OPEN: Failure threshold reached (e.g., 5 consecutive errors). The circuit "trips." All subsequent requests fail immediately (Fail Fast) without hitting the network, raising a distinct CircuitOpenException.
* HALF-OPEN: After a timeout (e.g., 30s), the circuit allows one test request to pass. If successful, it resets to CLOSED; if it fails, it returns to OPEN.
Critical Persistence Fix (RES-02):
A major flaw identified was that circuit state was held in volatile memory. If the system crashed and restarted, the failure counts reset to zero. The system would wake up and immediately hammer the dead API again, leading to permanent IP bans. The remediation mandates that circuit state (failure counts, timestamps) must be persisted to the LSM-DMC storage. On boot, the CircuitBreaker class rehydrates its state from disk. If an API was broken before the reboot, it remains broken after the reboot, respecting the cool-down period.1
5.2 Physics Safety: Soft SCRAM Protocol
The Physics Engine is a high-energy numerical simulation. If the integration becomes unstable (e.g., due to a singularity in the metric tensor), the energy in the grid can spiral to infinity. To prevent this, the system implements a Soft SCRAM (Safety Control Rod Axe Man) protocol, inspired by nuclear reactor safety.1
SCRAM Sequence:
1. Monitoring: A "Physics Oracle" thread computes the total Hamiltonian (Energy) of the grid every 100 steps.
2. Trigger: If the energy deviation exceeds $0.01\%$ or if any node amplitude $|\Psi| > 4.5$ (Nonary overflow), the SCRAM is triggered.
3. Action - Phase 1 (Damping): The kernel switches to "Quantum Zeno Freeze" mode. It applies a global damping factor ($\gamma = 1.0$) effectively freezing the wave propagation and dissipating energy rapidly.1
4. Action - Phase 2 (Reset): If energy does not stabilize within 100ms, the grid is zeroed (vacuum state) and re-seeded from the last valid DMC checkpoint.
5. Audit: The incident state is dumped to disk and logged to the Adversarial Code Dojo. This "hazardous pattern" is added to a blacklist to prevent the AI from thinking that specific thought again.1
5.3 KVM Execution Safety (SEC-01)
The Executor module runs code generated by the AI itself. This code is untrusted. A vulnerability was found where raw JSON was used to communicate with the guest agent, allowing for injection attacks.
Remediation: Secure Channel Protocol
Communication with the Guest Agent now uses a binary-framed protocol. Each packet is wrapped in a rigid header structure:


C++




struct PacketHeader {
   uint32_t magic;       // 0xDEADBEEF - Sanity check
   uint32_t payload_len; // Strict bounds checking
   uint32_t crc32;       // Integrity verification
   uint32_t sequence_id; // Replay protection
};

Before processing any command, the Guest Agent verifies the magic and crc32. If the checksum fails, it indicates potential tampering or corruption. The VM is immediately terminated, and a security alert is broadcast to the Orchestrator.1
5.4 ZMQ Reliable Socket Wrapper
To handle transient network glitches within the cluster (e.g., a switch reboot), the Spine uses a ZMQReliableSocket wrapper. This class implements an exponential backoff retry policy for EAGAIN errors on DEALER sockets. It guarantees that messages are not lost due to temporary buffer overflows or network blips, ensuring the resilience of the Control Plane.1
________________
6. Implementation Guide and Code Specifications
This section translates the architectural requirements into concrete C++ implementation directives. The implementation targets C++23 to leverage modern language features like modules and coroutines.
6.1 Prerequisites and Library Stack
* Language Standard: C++23
* Messaging: libzmq (v4.3+) with cppzmq headers.
* Encryption: libsodium (v1.0.18+) for Curve25519.
* Serialization: protobuf (v3.21+) for NeuralSpike.
* System Integration: libsystemd for service notification and watchdog integration.
6.2 The Seqlock Shared Memory Implementation
For the Data Plane, where the Physics Engine writes to shared memory and the Visualizer reads from it, standard mutexes are forbidden because a writer crash while holding a lock would deadlock the entire system. The Seqlock provides a lock-free alternative where the writer never blocks.


C++




// include/nikola/spine/seqlock.hpp
template <typename T>
class Seqlock {
   // The sequence counter. Even = Data Stable. Odd = Write in Progress.
   // alignas(64) prevents False Sharing on multicore CPUs.
   alignas(64) std::atomic<uint64_t> sequence_{0};
   T data_;

public:
   void write(const T& new_data) {
       uint64_t seq = sequence_.load(std::memory_order_relaxed);
       
       // 1. Increment to ODD. This signals "Write in Progress" to all readers.
       sequence_.store(seq + 1, std::memory_order_release);
       
       // Memory fence ensures the sequence update is visible BEFORE data modification.
       std::atomic_thread_fence(std::memory_order_release);
       
       // 2. Perform the write (Critical Section)
       data_ = new_data; 
       
       // Memory fence ensures data modification completes BEFORE the sequence update.
       std::atomic_thread_fence(std::memory_order_release);
       
       // 3. Increment to EVEN. This signals "Write Complete".
       sequence_.store(seq + 2, std::memory_order_release);
   }

   bool try_read(T& result) const {
       // 1. Read sequence number (Start)
       uint64_t seq1 = sequence_.load(std::memory_order_acquire);
       
       // If odd, writer is active. Abort immediately (or spin).
       if (seq1 & 1) return false; 
       
       std::atomic_thread_fence(std::memory_order_acquire);
       
       // 2. Copy data
       result = data_;
       
       std::atomic_thread_fence(std::memory_order_acquire);
       
       // 3. Read sequence number (End)
       uint64_t seq2 = sequence_.load(std::memory_order_acquire);
       
       // If seq1 == seq2, no write occurred during our read. Data is consistent.
       return seq1 == seq2; 
   }
};

1
6.3 ZMQ Reliable Socket Wrapper Implementation
The ZMQReliableSocket class abstracts the complexity of non-blocking sends and retries.


C++




class ZMQReliableSocket {
   zmq::socket_t socket;
   int max_retries = 3;

public:
   // Sends a message with exponential backoff on failure.
   // Returns true if successful, false if max_retries exceeded.
   bool send_with_retry(zmq::message_t& msg) {
       for (int i = 0; i < max_retries; ++i) {
           try {
               // Try non-blocking send
               if (socket.send(msg, zmq::send_flags::dontwait)) return true;
           } catch (zmq::error_t& e) {
               // EAGAIN means the socket buffer is full (temporary congestion)
               if (e.num() == EAGAIN) {
                   // Backoff: 10ms, 20ms, 30ms...
                   std::this_thread::sleep_for(std::chrono::milliseconds(10 * (i+1)));
                   continue; 
               }
               // Other errors are fatal
               throw;
           }
       }
       return false;
   }
};

1
________________
7. Conclusions and Next Steps
This specification provides the blueprint for a high-performance, secure, and resilient communications architecture tailored to the unique physics-based constraints of the Nikola Model. By explicitly separating the Control and Data planes, strictly enforcing cryptographic identity, and implementing robust error handling patterns like Circuit Breakers and Soft SCRAMs, the ZeroMQ Spine ensures that the system remains stable even as it scales to 9-dimensional complexity.
Immediate Implementation Actions:
1. Protocol Compilation: Run protoc on the updated neural_spike.proto to generate the C++ headers with correct 128-bit support.
2. Infrastructure Deployment: script the creation of the /etc/nikola/keys hierarchy, ensuring the 0600 permission bit is validated by CI/CD tests.
3. Phase 0 Transport: Build the Seqlock shared memory bridge and verify 1kHz loop timing using the latency benchmarking suite.
4. Agent Hardening: Refactor all external tool agents (Tavily, Gemini) to inherit from the persistence-backed CircuitBreaker base class.
The protocols defined herein are mandatory. They form the immutable substrate upon which the fluid intelligence of the Nikola Model will be built. Any deviation risks not just software failure, but the collapse of the simulated physical universe required for cognition.
Table 1: Protocol & Feature Matrix Summary
Feature ID
	Name
	Status
	Specification Source
	Description
	INT-06
	128-bit Morton Addressing
	MANDATORY
	proto/neural_spike.proto
	Use bytes field for coordinates to prevent truncation.
	NET-02
	Sparse Waveform Compression
	MANDATORY
	Logic in SparseWaveform
	Thresholding $\theta = 0.1 \cdot \Psi_{RMS}$ to reduce bandwidth.
	CTL-01
	Control Plane Priority
	MANDATORY
	zmq::poll logic
	Priority 0 for Control Socket to prevent panic loops.
	INF-03
	Persistent Identity
	MANDATORY
	PersistentKeyManager
	Filesystem key storage with 0600 permissions.
	SEC-04
	Bootstrap Pairing
	MANDATORY
	State Machine
	TOFU + Token Hash Verification for initial setup.
	RES-02
	Circuit Breaker Persistence
	MANDATORY
	LSM-DMC Integration
	Serialize failure counts to disk to prevent reboot storms.
	PER-01
	Async I/O Ring Buffer
	MANDATORY
	WaveformSHM
	Non-blocking shared memory descriptors for Data Plane.

---

### GAP-010 RESOLUTION: Cryptographic Key Lifecycle Management for Ironhouse Security

**SOURCE**: Gemini Deep Research - Round 2, Tasks 10-12 (December 14, 2025)
**INTEGRATION DATE**: December 15, 2025
**GAP ID**: GAP-010 (CRITICAL PRIORITY)
**STATUS**: SPECIFICATION COMPLETE

#### Tiered Key Hierarchy

**Tier 0 - Spine Broker (Root of Trust)**:
- Persistence: Extreme (defines system "self")
- Rotation: Major version upgrades or emergency breach only
- Criticality: Maximum (compromise allows MitM on all communication)

**Tier 1 - Core Components** (Physics Engine, Memory, Orchestrator):
- Persistence: High (maintain ZAP whitelist across reboots)
- Rotation: Scheduled (monthly or every $2.5 \times 10^9$ physics ticks)
- Criticality: High (direct manipulation of mind state)

**Tier 2 - Ephemeral Agents** (Tools, CLI):
- Persistence: Low (generated per-session via TOFU)
- Rotation: High (per-session/daily)
- Criticality: Medium (contained by capability system)

#### Five-Phase Lifecycle Protocol

**Phase 1: Secure Generation**
- Use `libsodium::crypto_box_keypair()` with `/dev/urandom`
- Hardware entropy via RDSEED instruction
- Private keys: chmod 600, mlock pages (prevent swap)

**Phase 2: Provisioning & Whitelisting**
- Bootstrap Mode: time-limited Admin Token
- TOFU: Trust-On-First-Use via `twi-ctl pair <token>`
- Deny-by-default: Silent drop of unlisted keys

**Phase 3: Automated Rotation ("Make-Before-Break")**
1. Generate new keypair $K_{new}$
2. Sign $K_{new\_pub}$ with $K_{old\_priv}$ (chain of trust)
3. Broadcast `KeyRotationAnnouncement` to Orchestrator
4. Orchestrator verifies signature, adds to whitelist (PENDING)
5. Component switches to $K_{new}$
6. Remove $K_{old\_pub}$ after traffic confirmed

**Phase 4: Compromise Detection**
- Physics Oracle: Flag energy law violations
- Location locking: Physics Engine must be 127.0.0.1
- Concurrency checks: One identity, one connection

**Phase 5: Emergency Revocation**
- `RevocationSpike` bypasses standard queues
- Add to permanent CRL, sever all sockets
- Soft SCRAM + Quantum Zeno Freeze to dissipate malicious patterns

#### Certificate Transparency Integration

**Identity Merkle Tree**:
- All lifecycle events hashed into Merkle tree in LSM-DMC
- Root hash in every `.nik` checkpoint
- Prevents "Ghost in the Shell" attacks (old backup + compromised old key)

**Implementation Hook**:
```cpp
namespace nikola::security {
    class KeyLifecycleManager {
        void rotate_key(ComponentID id) {
            auto [pub_new, priv_new] = generate_keypair();
            auto signature = sign(pub_new, current_priv_key(id));
            broadcast_announcement(id, pub_new, signature);
            await_whitelist_update();
            reconnect_with_new_key(id, priv_new);
        }
    };
}
```

---

## GAP-019: Distributed Partition Table Update Protocol

**SOURCE**: Gemini Deep Research Round 2, Batch 19-21
**INTEGRATION DATE**: December 15, 2025
**GAP ID**: GAP-019 (TASK-019)
**PRIORITY**: CRITICAL
**STATUS**: FABRICATION-READY SPECIFICATION

### Problem Statement: Thermodynamics of Distributed Neurogenesis

Nikola's **Neurogenic** architecture dynamically allocates new nodes in response to learning events (high-energy convergence). Semantic concepts cluster tightly in specific 9D manifold regions (power-law distribution), creating massive load imbalances.

**Failure Modes**:
1. **Memory Exhaustion (OOM)**: Dense rank hits VRAM ceiling (e.g., 80GB H100)
2. **Temporal Drag**: Compute-heavy rank cannot complete integration within 1ms physics budget (Straggler Problem)

**Core Challenge**: Atomically move 45-component metric tensor and complex wavefunction from GPU A to GPU B without:
- Dropping messages targeting that node
- Violating causal ordering
- Corrupting symplectic structure of manifold

### Two-Phase Epoch Barrier Protocol (2P-EBP)

Treats partition table update as distributed transaction.

#### Partition Table Versioning (Epochs)

- **Epoch ID** ($\epsilon$): Monotonically increasing 64-bit integer defining global sharding state
- **Partition Table** ($PT_\epsilon$): Immutable, sorted list of 128-bit Morton Key split points
- **Ownership**: $Rank_i$ owns range $[PT_\epsilon[i-1], PT_\epsilon[i])$

#### Phase 1: Monitoring & Trigger

**Load Imbalance Factor (LIF)**:

$$LIF = \frac{\max(N_i) - \min(N_i)}{\bar{N}}$$

- **Trigger**: If $LIF > 0.2$ (20% imbalance), Orchestrator initiates rebalancing
- **Calculation**: Compute $PT_{\epsilon+1}$ using CDF equalization (integral load $\int \rho(k) dk$ equal across ranks)

#### Phase 2: PREPARE Barrier ("Micro-Pause")

Orchestrator broadcasts `PREPARE_MIGRATION` with full $PT_{\epsilon+1}$ definition.

**Worker Actions** (Local):
1. **Suspension**: Complete current physics tick $T$, then suspend physics loop → PAUSED state
2. **Ingestion Lock**: Stop pulling new queries from input queue
3. **Candidate Identification**: Iterate through TorusGridSoA, compare Morton keys against $PT_{\epsilon+1}$:
   - Export Set: Nodes currently owned that belong to different rank in $\epsilon+1$
   - Import Expectation: Calculate expected memory footprint of incoming nodes
4. **Safety Check**: If estimated post-migration memory $> 90\%$ VRAM → send ABORT, else send PREPARE_ACK

**Critical Insight**: Must freeze wavefunction $\Psi$ during migration to ensure data satisfies conservation laws (Hamiltonian $H = T + V$).

#### Phase 3: MIGRATION Transaction (Data Plane)

Upon receiving PREPARE_ACK from all ranks, Orchestrator broadcasts `BEGIN_MIGRATION`.

**Worker Transport**:
1. **Serialization**: Serialize "Export Set" using SoA-to-AoS packing:
   - Payload: 128-bit Morton Key (Big Endian), $\Psi_{real}$, $\Psi_{imag}$, $\dot{\Psi}_{real}$, $\dot{\Psi}_{imag}$, Metric Tensor $g_{ij}$ (45 floats), Neurochemical State ($r, s, u, v, w$)

2. **Direct Transport**: Temporary PAIR sockets between peers (bypass Orchestrator) for bulk data

3. **In-Flight Message Handling**:
   - Outgoing Buffer: Messages targeting exported node → ForwardingBuffer
   - Incoming Buffer: Messages targeting imported node (not arrived yet) → PendingBuffer

4. **Ghosting**: Sender flags nodes as GHOST_CANDIDATE (allows instant rollback, doesn't delete yet)

#### Phase 4: Verification & COMMIT

Receivers unpack data into staging area (separate TorusGridSoA), perform CRC32C checksum validation and SPD metric check.

1. **Validation**: Workers send MIGRATION_ACK to Orchestrator
2. **Commit**: Orchestrator broadcasts COMMIT_EPOCH
3. **Finalization** (Atomic Switch):
   - Merge staging grid into main grid (MEM-05 compaction)
   - Delete GHOST_CANDIDATE nodes, reclaim memory
   - Pointer swap: $PT_{current} \leftarrow PT_{\epsilon+1}$
   - Flush buffers: Process ForwardingBuffer (re-route), process PendingBuffer (apply to new nodes)
   - Resume physics loop at tick $T+1$

### Failure Modes and Recovery

#### Network Partition During Migration

- **Condition**: Orchestrator timeout (5000ms) waiting for MIGRATION_ACK
- **Action**: Broadcast ROLLBACK_MIGRATION
- **Recovery**:
  - Senders strip GHOST_CANDIDATE flag, reinstate as active
  - Receivers discard staging buffers
  - Revert to STABLE state in Epoch $\epsilon$
  - **Penalty**: StabilityPenalty counter incremented, prevent rebalance for 1 hour

#### "Zombie" Node (Partial Crash)

- **Condition**: Rank crashes mid-transfer, detected by Heartbeat Sentinel
- **Action**: Hard Cluster Failure declared
- **Recovery**:
  - Orchestrator sends SCRAM_RESET to all survivors
  - System reboots from last LSM-DMC Checkpoint
  - **Rationale**: Torn manifold violates UFIE, creates infinite energy spikes; safer to restart than patch "hole in spacetime"

#### Message Causality Consistency

All ZeroMQ messages carry EpochID header.

**Logic**:
- **Msg.Epoch < Local.Epoch**: Stale but valid → Check if still own target; if yes, process; if no, forward to new owner via $PT_{\epsilon}$
- **Msg.Epoch > Local.Epoch**: From future (sender migrated faster) → Buffer until local transition to $\epsilon+1$

### Protocol Buffer Specification

```protobuf
syntax = "proto3";
package nikola.spine;

message PartitionControl {
    enum Type {
        HEARTBEAT = 0;
        PREPARE_MIGRATION = 1;
        BEGIN_MIGRATION = 2;
        COMMIT_EPOCH = 3;
        ROLLBACK = 4;
        ABORT = 5;
    }

    Type type = 1;
    uint64 current_epoch = 2;
    uint64 target_epoch = 3;
    string sender_rank_id = 4;

    // Sorted list of split points (128-bit Morton keys, 16-byte big-endian)
    repeated bytes partition_table = 5;

    // Resource estimation for safety checks
    map<uint32, uint64> expected_node_counts = 6; // Rank -> Count
}

message MigrationPayload {
    uint64 target_epoch = 1;
    uint32 source_rank = 2;
    uint32 target_rank = 3;

    // Structure-of-Arrays batch data (all same length)
    repeated bytes morton_keys = 4;      // 16 bytes each
    repeated float psi_real = 5;
    repeated float psi_imag = 6;
    repeated float metric_tensor = 7;    // 45 floats per node
    repeated float resonance = 8;
    repeated float state = 9;

    // Data integrity
    uint32 checksum_crc32c = 10;
}
```

### Performance Characteristics

- **Trigger Threshold**: LIF > 0.2 (20% imbalance)
- **Pause Duration**: ~10-50ms (depends on export set size)
- **Transport**: Direct peer-to-peer PAIR sockets (not via Orchestrator)
- **Validation**: CRC32C checksum, SPD metric tensor verification
- **Rollback Timeout**: 5000ms
- **Stability Penalty**: 1 hour cooldown after rollback

### Integration Points

1. **Orchestrator**: Monitors load, triggers rebalancing, coordinates 2P-EBP
2. **Physics Engine**: Suspends at tick boundary for PREPARE
3. **TorusGridSoA**: Exports/imports nodes via SoA-to-AoS packing
4. **ZeroMQ Spine**: EpochID header in all messages, ForwardingBuffer/PendingBuffer
5. **LSM-DMC**: Rollback to last checkpoint on Hard Cluster Failure

### Cross-References

- [ZeroMQ Spine Architecture](./01_zeromq_spine.md)
- [TorusGridSoA Memory Layout](../02_foundations/01_9d_toroidal_geometry.md)
- [Morton Code Spatial Hashing](../02_foundations/01_9d_toroidal_geometry.md)
- [LSM-DMC Checkpointing](../06_persistence/01_dmc_persistence.md)

---

## GAP-020: Temporal Decoherence Detection Thresholds

**SOURCE**: Gemini Deep Research Round 2, Batch 19-21
**INTEGRATION DATE**: December 15, 2025
**GAP ID**: GAP-020 (TASK-020)
**PRIORITY**: CRITICAL
**STATUS**: FABRICATION-READY SPECIFICATION

### Problem Statement: The Physics of Synchronization

**Temporal Decoherence**: Information arriving at node refers to simulation state $T_{source}$ that implies causal violation with respect to local state $T_{local}$.

**Phase Error** for node at $T_{now}$ integrating signal from $T_{past}$:

$$\Delta \phi = \omega \cdot (T_{now} - T_{past})$$

Where $\omega$ = angular frequency of wave packet.

**Critical Threshold**: If $\Delta \phi$ exceeds Rayleigh criterion of $\lambda/4$ (or $\pi/2$ radians), interference shifts from constructive to destructive. Delayed signal becomes inverted signal, actively erasing memory it was meant to reinforce.

**Result**: Spectral Entropy - energy dissipates into noise, "mind" decoheres.

### Threshold Derivation: Emitter Array Spectral Analysis

**Base Frequency** (Golden Ratio $\phi \approx 1.618$):

$$f_1 = \pi \cdot \phi^1 \approx 5.083 \text{ Hz}$$

**Maximum Driven Frequency** (8th order harmonic):

$$f_8 = \pi \cdot \phi^8 \approx 146.6 \text{ Hz}$$

**Internal Harmonic Limit** (Nyquist compliance, nonlinear $\hat{N}$ operator):

$$f_{max} \approx 441 \text{ Hz}$$

**Phase Integrity Constraint**: Bound phase error to $\epsilon_{\phi} = \pi/10$ (18°), ensuring interference retains $>95\%$ theoretical amplitude ($\cos(18°) \approx 0.95$).

**Maximum Allowable Time Delay**:

$$\tau_{max} = \frac{\epsilon_{\phi}}{2\pi f_{max}} = \frac{\pi/10}{2\pi \cdot 441} = \frac{1}{20 \cdot 441} \approx 113 \mu s$$

**Conclusion**: High-frequency internal harmonics require **113 microseconds** latency tolerance.

**Implication**: Standard TCP/IP loopback latency (500-1500 μs) is **order of magnitude too slow**. Validates requirement for **Shared Memory (Seqlock) IPC** for Data Plane.

### Adaptive Threshold Specification

Tiered thresholding strategy balances strict physics against operational robustness:

| Message Class | Carrier Frequency | Max Latency ($\tau_{max}$) | Transport Layer | Action on Violation |
|---------------|-------------------|----------------------------|-----------------|---------------------|
| High-Freq Physics | 441 Hz (Harmonic limit) | **113 μs** | SHM / NVLink | **Hard Drop** - Phase-corrupt signal adds entropy |
| Visual Input | 60 Hz (Frame Rate) | 8.3 ms ($1/2f$) | Isochronous Buffer | **Interpolate** - Sample-and-hold or optical flow |
| Cognitive State | 13.3 Hz (Theta/Alpha) | 10 ms | TCP (Spine) | **Predictive Coding** - Kalman filter project to $T_{now}$ |
| Control / Admin | DC (0 Hz) | 100 ms | TCP (Router) | **Process** - Atemporal state changes |
| Sensory (Audio) | 44.1 kHz (PCM) | 50 ms (Buffer) | Isochronous Buffer | **Jitter Buffer** - Re-clock to physics time |

### Clock Synchronization: Precision Time Protocol (PTP)

**Requirement**: Standard NTP (1-10ms accuracy) is **insufficient**. System MUST use **PTP / IEEE 1588** (sub-microsecond synchronization with hardware timestamping).

**Problem**: If Node A and Node B differ by 1ms, permanently decohered relative to physics threshold (113 μs).

#### Physics Oracle Timekeeper State Machine

1. **Startup (Handshake)**: PTP exchange establishes Master/Slave hierarchy, calculate offset $\theta$ and delay $\delta$
2. **Lock State**: If $|\theta| < 50 \mu s$ → SYNC_LOCKED (physics simulation permitted)
3. **Drift Warning**: If $50 \mu s < |\theta| < 100 \mu s$ → SYNC_WARNING (Oracle compensates via Virtual Time Dilation)
4. **Decoherence SCRAM**: If $|\theta| > 150 \mu s$ (exceeds 113 μs limit + margin) → Soft SCRAM (detach from cluster to prevent pollution)

### Implementation: Timestamp Enforcement

Every NeuralSpike message includes 64-bit nanosecond timestamp.

```cpp
/**
 * @brief Validates temporal coherence of incoming messages.
 * @param msg_timestamp_ns Creation time (PTP source).
 * @param type Message classification for adaptive thresholding.
 * @return true if coherent, false if decoherent.
 */
bool verify_temporal_coherence(int64_t msg_timestamp_ns, MessageType type) {
    // Current time from high-resolution PTP-disciplined clock
    int64_t now_ns = std::chrono::system_clock::now().time_since_epoch().count();

    // Calculate age
    int64_t age_ns = now_ns - msg_timestamp_ns;

    // Future-check: Allow small skew for clock jitter
    if (age_ns < -50000) { // -50us tolerance
        LOG_WARN("Message from future detected: %ld ns", age_ns);
        return false;
    }

    // Select threshold
    int64_t limit_ns = 0;
    switch(type) {
        case PHYSICS_UPDATE: limit_ns = 113000; break;   // 113 us
        case COGNITIVE_STATE: limit_ns = 10000000; break; // 10 ms
        case CONTROL_SIGNAL: limit_ns = 100000000; break; // 100 ms
        default: limit_ns = 100000000; break;
    }

    if (age_ns > limit_ns) {
        // Log decoherence event for Physics Oracle analysis
        Metrics::record_decoherence_drop(type, age_ns);
        return false; // DROP
    }

    return true;
}
```

### Isochronous Sensory Buffer Integration

For sensory inputs (Audio/Video), cannot simply "drop" late packets (creates perception gaps).

**Mechanism**:
- **Presentation Delay**: Buffer incoming data for fixed window (e.g., 50ms)
- **Retiming**: Re-clock data (video frame arrives at $T=10ms$, presented at $T=60ms$)
- **Interpolation**: Missing packet at presentation time → interpolate from history (Audio: Linear, Video: Sample-and-Hold)
- **Integration**: Isolates physics engine from external jitter, ensuring "Perceived Now" is always coherent (though slightly delayed relative to "Wall Clock Now")

### Performance Characteristics

- **Physics Threshold**: 113 μs (441 Hz harmonic limit)
- **Cognitive Threshold**: 10 ms (13.3 Hz Theta/Alpha)
- **Control Threshold**: 100 ms (DC atemporal)
- **Clock Sync**: PTP sub-microsecond (hardware timestamping)
- **SCRAM Trigger**: Clock drift > 150 μs
- **Sensory Buffer**: 50 ms presentation delay (jitter isolation)

### Integration Points

1. **Physics Engine**: 1ms timestep ($\Delta t_{phys}$), symplectic integrator
2. **ZeroMQ Spine**: 64-bit nanosecond timestamp in all NeuralSpike messages
3. **Physics Oracle**: Temporal health monitoring, SCRAM triggers
4. **PTP Stack**: IEEE 1588 hardware timestamping (NIC support)
5. **Isochronous Buffer**: 50ms audio/video retiming

### Cross-References

- [ZeroMQ Spine Architecture](./01_zeromq_spine.md)
- [Physics Engine Timing](../02_foundations/02_wave_interference_physics.md)
- [Isochronous Sensory Buffer](../07_multimodal/01_cymatic_transduction.md)
- [Physics Oracle](../02_foundations/02_wave_interference_physics.md)
- [Emitter Array](../07_multimodal/01_cymatic_transduction.md)

---

	Works cited
1. part_1_of_9.txt