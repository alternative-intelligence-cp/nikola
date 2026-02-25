#pragma once
// nikola/infrastructure/docker_compose_orchestration.hpp
//
// GAP-026: Docker Compose Service Orchestration
// Source: Gemini Deep Research Round 2, Batch 25-27
// Spec: docs/info/integration/sections/04_infrastructure/02_orchestrator_router.md §GAP-026
//
// Encodes the 4-layer Nikola service dependency graph, healthcheck parameters,
// startup/shutdown timing constraints, resource limits, and the memory-safe
// shutdown sequence state machine. Header-only, no external dependencies.

#include <cstdint>

namespace nikola::infrastructure {

// ─── Service Topology ────────────────────────────────────────────────────────

/// Four-layer service dependency hierarchy.
/// Services in higher layers depend on services in lower layers being healthy.
enum class ServiceLayer : uint8_t {
    CORE       = 0,  ///< nikola-spine  — ZeroMQ Broker, no dependencies
    PHYSICS    = 1,  ///< nikola-physics — GPU engine, depends on CORE healthy
    COGNITION  = 2,  ///< nikola-orchestrator + nikola-memory, depend on PHYSICS
    TOOLS      = 3,  ///< nikola-executor + nikola-web, depend on COGNITION
};

/// Named services within the cluster.
enum class NikolaService : uint8_t {
    SPINE        = 0,  ///< Layer 0: ZeroMQ Ironhouse broker
    PHYSICS      = 1,  ///< Layer 1: GPU-accelerated wave physics
    MEMORY       = 2,  ///< Layer 2: LSM-DMC persistence
    ORCHESTRATOR = 3,  ///< Layer 2: Cognitive logic + external agents
    EXECUTOR     = 4,  ///< Layer 3: KVM security sandbox
    WEB          = 5,  ///< Layer 3: User interface
};

/// Runtime state of a service during lifecycle management.
enum class ServiceState : uint8_t {
    PENDING   = 0,  ///< Waiting for dependency readiness
    STARTING  = 1,  ///< Process launched, healthcheck pending
    HEALTHY   = 2,  ///< Healthcheck passed, accepting connections
    DEGRADED  = 3,  ///< Running but under-capacity or retrying
    HALTED    = 4,  ///< Ordered shutdown in progress
    CRASHED   = 5,  ///< Unexpected exit, restart eligible
};

// ─── Shutdown Sequence State Machine ─────────────────────────────────────────

/// Phases of the memory-safe graceful shutdown sequence.
/// Must progress sequentially; only MANIFEST_WRITTEN → TERMINATED is safe for
/// the memory subsystem to force-kill.
///
/// Sequence:
///   RUNNING → SIGTERM_RECEIVED → HALT_BROADCAST → PHYSICS_TICK_COMPLETE
///   → MEMORY_LOCK_ACQUIRED → MEMTABLE_FLUSHED → WAL_FSYNCED
///   → MANIFEST_WRITTEN → TERMINATED
enum class ShutdownPhase : uint8_t {
    RUNNING               = 0,  ///< Normal operation
    SIGTERM_RECEIVED      = 1,  ///< SIGTERM delivered to orchestrator
    HALT_BROADCAST        = 2,  ///< SYSTEM_HALT published on ZeroMQ Control Plane
    PHYSICS_TICK_COMPLETE = 3,  ///< Final 1ms tick done; Ψ serialized to /dev/shm
    MEMORY_LOCK_ACQUIRED  = 4,  ///< Global Write Lock held; no new writes accepted
    MEMTABLE_FLUSHED      = 5,  ///< In-memory MemTable written to SSTable (Level 0)
    WAL_FSYNCED           = 6,  ///< Write-Ahead Log fsync'd to disk
    MANIFEST_WRITTEN      = 7,  ///< MANIFEST updated with new Merkle Root hash
    TERMINATED            = 8,  ///< Process exit confirmed
};

// ─── Healthcheck Constants (nikola-spine ZeroMQ handshake) ───────────────────

/// Poll interval between healthcheck attempts (seconds).
inline constexpr int HEALTHCHECK_INTERVAL_S  = 5;

/// Timeout for a single healthcheck attempt (seconds).
inline constexpr int HEALTHCHECK_TIMEOUT_S   = 2;

/// Maximum healthcheck retries before container is marked unhealthy.
inline constexpr int HEALTHCHECK_RETRIES     = 5;

/// Maximum total wait for spine to become healthy:
/// HEALTHCHECK_INTERVAL_S × HEALTHCHECK_RETRIES = 25 s.
inline constexpr int HEALTHCHECK_MAX_WAIT_S  = HEALTHCHECK_INTERVAL_S * HEALTHCHECK_RETRIES;

// ─── Startup Timing Targets ───────────────────────────────────────────────────

/// nikola-spine bind time lower bound (seconds).
inline constexpr int SPINE_STARTUP_MIN_S     = 1;

/// nikola-spine bind time upper bound (seconds).
inline constexpr int SPINE_STARTUP_MAX_S     = 2;

/// nikola-physics GPU init + ZeroMQ connect lower bound (seconds).
inline constexpr int PHYSICS_STARTUP_MIN_S   = 3;

/// nikola-physics GPU init + ZeroMQ connect upper bound (seconds).
inline constexpr int PHYSICS_STARTUP_MAX_S   = 5;

/// Maximum time from `docker compose up` to full cluster ready (seconds).
inline constexpr int FULL_CLUSTER_STARTUP_S  = 30;

// ─── Shutdown Timing ──────────────────────────────────────────────────────────

/// Docker `stop_grace_period` — time before SIGKILL is sent (seconds).
/// Must exceed worst-case MemTable flush + fsync time.
inline constexpr int STOP_GRACE_PERIOD_S     = 60;

/// Minimum graceful shutdown time observed in practice (seconds).
inline constexpr int GRACEFUL_SHUTDOWN_MIN_S = 10;

/// Maximum graceful shutdown time before abort risk (seconds).
inline constexpr int GRACEFUL_SHUTDOWN_MAX_S = STOP_GRACE_PERIOD_S;

/// Number of distinct phases in the shutdown state machine (excluding RUNNING).
inline constexpr int SHUTDOWN_SEQUENCE_STEPS = 8;

// ─── Resource Limits ─────────────────────────────────────────────────────────

/// Stack size ulimit for nikola-physics (bytes).
/// 64 MB required for deep Hilbert-curve recursion in Mamba-9D.
inline constexpr long  STACK_ULIMIT_BYTES    = 67'108'864L;  // 64 MB

/// Sentinel for unlimited memory-lock (passed to setrlimit RLIMIT_MEMLOCK).
/// Prevents GPU memory pages from being swapped, protecting 1ms tick budget.
inline constexpr int   MEMLOCK_UNLIMITED     = -1;

/// CPU limit for nikola-spine container (fractional cores).
inline constexpr float SPINE_CPU_LIMIT       = 2.0f;

/// Memory limit for nikola-spine container (GB).
inline constexpr int   SPINE_MEMORY_LIMIT_GB = 4;

/// Number of NVIDIA GPU devices reserved for nikola-physics.
inline constexpr int   GPU_DEVICE_COUNT      = 1;

/// OpenMP thread count for AVX-512 physics sections.
inline constexpr int   OMP_NUM_THREADS       = 16;

// ─── Topology Queries ────────────────────────────────────────────────────────

/// Layer a service belongs to.
[[nodiscard]] constexpr ServiceLayer service_layer(NikolaService svc) noexcept {
    switch (svc) {
        case NikolaService::SPINE:        return ServiceLayer::CORE;
        case NikolaService::PHYSICS:      return ServiceLayer::PHYSICS;
        case NikolaService::MEMORY:       return ServiceLayer::COGNITION;
        case NikolaService::ORCHESTRATOR: return ServiceLayer::COGNITION;
        case NikolaService::EXECUTOR:     return ServiceLayer::TOOLS;
        case NikolaService::WEB:          return ServiceLayer::TOOLS;
    }
    return ServiceLayer::TOOLS;
}

/// True if the service's dependency condition is `service_healthy` (ZMQ
/// healthcheck), rather than plain `service_started`.
/// Only the spine uses the full ZMQ handshake healthcheck.
[[nodiscard]] constexpr bool requires_healthy_dependency(NikolaService svc) noexcept {
    return svc == NikolaService::SPINE;
}

/// True if the service requires NVIDIA GPU runtime (`nvidia` docker runtime).
[[nodiscard]] constexpr bool requires_gpu(NikolaService svc) noexcept {
    return svc == NikolaService::PHYSICS;
}

/// True if the service needs `privileged: true` (KVM/QEMU hardware access).
[[nodiscard]] constexpr bool requires_privileged(NikolaService svc) noexcept {
    return svc == NikolaService::EXECUTOR;
}

/// True if the service holds durable persistent state that must be flushed
/// before SIGKILL (triggers the 60 s grace period requirement).
[[nodiscard]] constexpr bool holds_persistent_state(NikolaService svc) noexcept {
    return svc == NikolaService::MEMORY;
}

/// True if the service participates in the CurveZMQ Ironhouse key exchange.
[[nodiscard]] constexpr bool uses_curvezmq(NikolaService svc) noexcept {
    // All services except web mount the keys volume.
    return svc != NikolaService::WEB;
}

/// Startup deadline for a given service (seconds); exceeding this indicates
/// a configuration or infrastructure fault.
[[nodiscard]] constexpr int startup_deadline_s(NikolaService svc) noexcept {
    switch (svc) {
        case NikolaService::SPINE:   return SPINE_STARTUP_MAX_S;
        case NikolaService::PHYSICS: return PHYSICS_STARTUP_MAX_S;
        default:                     return FULL_CLUSTER_STARTUP_S;
    }
}

/// True if the observed startup time is within spec for the given service.
[[nodiscard]] constexpr bool startup_within_budget(NikolaService svc, int elapsed_s) noexcept {
    return elapsed_s <= startup_deadline_s(svc);
}

/// True if the healthcheck wait has not yet exceeded the retry ceiling (25 s).
[[nodiscard]] constexpr bool healthcheck_within_budget(int elapsed_s) noexcept {
    return elapsed_s <= HEALTHCHECK_MAX_WAIT_S;
}

// ─── Shutdown Sequence Predicates ────────────────────────────────────────────

/// True when the shutdown sequence has reached terminal state.
[[nodiscard]] constexpr bool shutdown_complete(ShutdownPhase phase) noexcept {
    return phase == ShutdownPhase::TERMINATED;
}

/// True when the memory subsystem has safely persisted all state and the
/// process may be force-killed without data loss.
[[nodiscard]] constexpr bool memory_safe_to_kill(ShutdownPhase phase) noexcept {
    return static_cast<uint8_t>(phase) >=
           static_cast<uint8_t>(ShutdownPhase::MANIFEST_WRITTEN);
}

/// True when the WAL has been fsynced (data durable even if process is killed).
[[nodiscard]] constexpr bool wal_is_durable(ShutdownPhase phase) noexcept {
    return static_cast<uint8_t>(phase) >=
           static_cast<uint8_t>(ShutdownPhase::WAL_FSYNCED);
}

/// True when physics tick is complete and wavefunction is in a valid state.
[[nodiscard]] constexpr bool physics_state_serialized(ShutdownPhase phase) noexcept {
    return static_cast<uint8_t>(phase) >=
           static_cast<uint8_t>(ShutdownPhase::PHYSICS_TICK_COMPLETE);
}

/// True when elapsed time is within the stop_grace_period (60 s) window.
[[nodiscard]] constexpr bool shutdown_within_grace(int elapsed_s) noexcept {
    return elapsed_s <= STOP_GRACE_PERIOD_S;
}

/// True when elapsed time has exceeded the grace period — SIGKILL imminent.
[[nodiscard]] constexpr bool shutdown_grace_expired(int elapsed_s) noexcept {
    return elapsed_s > STOP_GRACE_PERIOD_S;
}

// ─── Resource Predicates ─────────────────────────────────────────────────────

/// True if the stack allocation is at least the required 64 MB.
[[nodiscard]] constexpr bool stack_is_sufficient(long bytes) noexcept {
    return bytes >= STACK_ULIMIT_BYTES;
}

/// True if the OMP thread count is at spec (16 for AVX-512 sections).
[[nodiscard]] constexpr bool omp_threads_at_spec(int count) noexcept {
    return count == OMP_NUM_THREADS;
}

/// True if the CPU limit meets spine's minimum requirement.
[[nodiscard]] constexpr bool spine_cpu_is_sufficient(float cpus) noexcept {
    return cpus >= SPINE_CPU_LIMIT;
}

// ─── Label Functions ─────────────────────────────────────────────────────────

[[nodiscard]] constexpr const char* service_layer_name(ServiceLayer layer) noexcept {
    switch (layer) {
        case ServiceLayer::CORE:      return "CORE";
        case ServiceLayer::PHYSICS:   return "PHYSICS";
        case ServiceLayer::COGNITION: return "COGNITION";
        case ServiceLayer::TOOLS:     return "TOOLS";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr const char* service_name(NikolaService svc) noexcept {
    switch (svc) {
        case NikolaService::SPINE:        return "nikola-spine";
        case NikolaService::PHYSICS:      return "nikola-physics";
        case NikolaService::MEMORY:       return "nikola-memory";
        case NikolaService::ORCHESTRATOR: return "nikola-orchestrator";
        case NikolaService::EXECUTOR:     return "nikola-executor";
        case NikolaService::WEB:          return "nikola-web";
    }
    return "unknown";
}

[[nodiscard]] constexpr const char* service_state_name(ServiceState state) noexcept {
    switch (state) {
        case ServiceState::PENDING:  return "PENDING";
        case ServiceState::STARTING: return "STARTING";
        case ServiceState::HEALTHY:  return "HEALTHY";
        case ServiceState::DEGRADED: return "DEGRADED";
        case ServiceState::HALTED:   return "HALTED";
        case ServiceState::CRASHED:  return "CRASHED";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr const char* shutdown_phase_name(ShutdownPhase phase) noexcept {
    switch (phase) {
        case ShutdownPhase::RUNNING:               return "RUNNING";
        case ShutdownPhase::SIGTERM_RECEIVED:      return "SIGTERM_RECEIVED";
        case ShutdownPhase::HALT_BROADCAST:        return "HALT_BROADCAST";
        case ShutdownPhase::PHYSICS_TICK_COMPLETE: return "PHYSICS_TICK_COMPLETE";
        case ShutdownPhase::MEMORY_LOCK_ACQUIRED:  return "MEMORY_LOCK_ACQUIRED";
        case ShutdownPhase::MEMTABLE_FLUSHED:      return "MEMTABLE_FLUSHED";
        case ShutdownPhase::WAL_FSYNCED:           return "WAL_FSYNCED";
        case ShutdownPhase::MANIFEST_WRITTEN:      return "MANIFEST_WRITTEN";
        case ShutdownPhase::TERMINATED:            return "TERMINATED";
    }
    return "UNKNOWN";
}

} // namespace nikola::infrastructure
