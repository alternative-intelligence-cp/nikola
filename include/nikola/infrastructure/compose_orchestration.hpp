// ============================================================
// include/nikola/infrastructure/compose_orchestration.hpp
// Phase 71 — GAP-026  Docker Compose Service Orchestration
// ============================================================
// Models the 4-layer distributed service topology for Nikola:
//   Layer 0 (Core)      : nikola-spine   — ZeroMQ CurveZMQ broker
//   Layer 1 (Physics)   : nikola-physics — GPU-accelerated engine
//   Layer 2 (Cognition) : nikola-orchestrator + nikola-memory
//   Layer 3 (Tools)     : nikola-executor + nikola-web
// ============================================================
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace nikola::infrastructure {

// ────────────────────────────────────────────────────────────────────────────
// §1  Topology counts
// ────────────────────────────────────────────────────────────────────────────

/// Number of hierarchical layers in the dependency graph (0–3).
constexpr uint8_t  LAYER_COUNT    = 4u;

/// Total number of named services in the Compose stack.
constexpr uint8_t  SERVICE_COUNT  = 6u;

// ────────────────────────────────────────────────────────────────────────────
// §2  Startup timing (milliseconds)
// ────────────────────────────────────────────────────────────────────────────

/// Maximum expected startup latency for nikola-spine (ZeroMQ bind).
constexpr uint64_t SPINE_STARTUP_MAX_MS    = 2'000u;

/// Maximum expected startup latency for nikola-physics (GPU init + connect).
constexpr uint64_t PHYSICS_STARTUP_MAX_MS  = 5'000u;

/// Full cluster must be ready within this window from `docker compose up`.
constexpr uint64_t CLUSTER_STARTUP_MAX_MS  = 30'000u;

// ────────────────────────────────────────────────────────────────────────────
// §3  Healthcheck parameters (nikola-spine)
// ────────────────────────────────────────────────────────────────────────────

/// Interval between healthcheck attempts (seconds converted to ms).
constexpr uint64_t HEALTHCHECK_INTERVAL_MS  = 5'000u;

/// Per-attempt timeout for the ZeroMQ handshake probe.
constexpr uint64_t HEALTHCHECK_TIMEOUT_MS   = 2'000u;

/// Maximum number of failed retries before container marked unhealthy.
constexpr uint8_t  HEALTHCHECK_MAX_RETRIES  = 5u;

// ────────────────────────────────────────────────────────────────────────────
// §4  Shutdown timing
// ────────────────────────────────────────────────────────────────────────────

/// Docker default stop_grace_period (seconds → ms) before SIGKILL.
constexpr uint64_t DEFAULT_STOP_GRACE_MS  = 10'000u;

/// nikola-memory stop_grace_period: allows LSM MemTable + WAL flush.
constexpr uint64_t STOP_GRACE_PERIOD_MS   = 60'000u;

/// Force-kill budget tied to Docker default (services without custom grace).
constexpr uint64_t FORCE_KILL_BUDGET_MS   = 10'000u;

/// Number of ordered steps in the nikola-memory graceful shutdown sequence.
constexpr uint8_t  MEMORY_SHUTDOWN_STEPS  = 4u;
// Step 1: Acquire Global Write Lock
// Step 2: Flush MemTable → Level-0 SSTable on disk
// Step 3: fsync WAL to disk
// Step 4: Write MANIFEST (update Merkle Root hash)

// ────────────────────────────────────────────────────────────────────────────
// §5  Physics-engine resource parameters
// ────────────────────────────────────────────────────────────────────────────

/// OMP thread count for AVX-512 parallelism in phonon sections.
constexpr uint32_t PHYSICS_OMP_NUM_THREADS = 16u;

/// Stack size (bytes) for deep Hilbert-curve recursion in physics engine — 64 MB.
constexpr uint64_t PHYSICS_STACK_BYTES     = 67'108'864u;  // 64 × 1024 × 1024

/// memlock ulimit value meaning "unlimited" (prevents GPU memory from being swapped).
constexpr int      PHYSICS_MEMLOCK_UNLIMITED = -1;

// ────────────────────────────────────────────────────────────────────────────
// §6  Spine resource limits
// ────────────────────────────────────────────────────────────────────────────

/// CPU cores limit for nikola-spine.
constexpr float    SPINE_CPU_CORES_LIMIT  = 2.0f;

/// Memory limit for nikola-spine in GiB.
constexpr uint32_t SPINE_MEMORY_LIMIT_GIB = 4u;

// ────────────────────────────────────────────────────────────────────────────
// §7  Enumerations
// ────────────────────────────────────────────────────────────────────────────

/// Services in the Nikola Compose stack.
enum class Service : uint8_t {
    SPINE        = 0,  ///< nikola-spine
    PHYSICS      = 1,  ///< nikola-physics
    MEMORY       = 2,  ///< nikola-memory
    ORCHESTRATOR = 3,  ///< nikola-orchestrator
    EXECUTOR     = 4,  ///< nikola-executor
    WEB          = 5,  ///< nikola-web
};

/// Dependency health condition used in depends_on.
enum class DependencyCondition : uint8_t {
    NONE            = 0,  ///< Layer 0 (no upstream deps)
    SERVICE_HEALTHY = 1,  ///< depends_on with condition: service_healthy
    SERVICE_STARTED = 2,  ///< depends_on with condition: service_started
};

/// Ordered steps in the nikola-memory graceful shutdown protocol.
enum class ShutdownStep : uint8_t {
    ACQUIRE_WRITE_LOCK  = 0,  ///< Prevent new writes
    FLUSH_MEMTABLE      = 1,  ///< Flush RAM MemTable → Level-0 SSTable
    FSYNC_WAL           = 2,  ///< Sync Write-Ahead Log to disk
    WRITE_MANIFEST      = 3,  ///< Record Merkle Root in MANIFEST file
};

// ────────────────────────────────────────────────────────────────────────────
// §8  Service topology queries
// ────────────────────────────────────────────────────────────────────────────

/// Return the dependency-graph layer for a service (0 = no deps, 3 = last).
[[nodiscard]] constexpr uint8_t service_layer(Service s) noexcept {
    switch (s) {
        case Service::SPINE:        return 0u;
        case Service::PHYSICS:      return 1u;
        case Service::MEMORY:       return 2u;
        case Service::ORCHESTRATOR: return 2u;
        case Service::EXECUTOR:     return 3u;
        case Service::WEB:          return 3u;
    }
    return 255u; // unreachable
}

/// Return startup order index (lower starts first; services in the same layer share equal order).
[[nodiscard]] constexpr uint8_t service_start_order(Service s) noexcept {
    return service_layer(s);
}

/// True when a service directly depends on nikola-spine.
[[nodiscard]] constexpr bool depends_on_spine(Service s) noexcept {
    return s == Service::PHYSICS || s == Service::MEMORY;
}

/// True when a service directly depends on nikola-orchestrator.
[[nodiscard]] constexpr bool depends_on_orchestrator(Service s) noexcept {
    return s == Service::EXECUTOR || s == Service::WEB;
}

/// True when a service depends on *any* upstream service.
[[nodiscard]] constexpr bool has_upstream_dependency(Service s) noexcept {
    return s != Service::SPINE;
}

/// Return the dependency_condition applied to a service's upstream.
/// MEMORY and ORCHESTRATOR depend on SPINE with service_healthy;
/// all Layer-2/3 services depend on their Layer-1/2 peers with service_started.
[[nodiscard]] constexpr DependencyCondition dependency_condition(Service s) noexcept {
    switch (s) {
        case Service::SPINE:         return DependencyCondition::NONE;
        case Service::PHYSICS:       return DependencyCondition::SERVICE_HEALTHY; // waits for spine healthcheck
        case Service::MEMORY:        return DependencyCondition::SERVICE_HEALTHY; // waits for spine healthcheck
        case Service::ORCHESTRATOR:  return DependencyCondition::SERVICE_STARTED;
        case Service::EXECUTOR:      return DependencyCondition::SERVICE_STARTED;
        case Service::WEB:           return DependencyCondition::SERVICE_STARTED;
    }
    return DependencyCondition::NONE; // unreachable
}

// ────────────────────────────────────────────────────────────────────────────
// §9  Resource requirement predicates
// ────────────────────────────────────────────────────────────────────────────

/// Only physics requires the `runtime: nvidia` Docker runtime.
[[nodiscard]] constexpr bool requires_gpu_runtime(Service s) noexcept {
    return s == Service::PHYSICS;
}

/// Only executor requires `privileged: true` for KVM/QEMU access.
[[nodiscard]] constexpr bool requires_privileged_mode(Service s) noexcept {
    return s == Service::EXECUTOR;
}

/// Only physics requires `ulimits: memlock: -1` to pin GPU memory.
[[nodiscard]] constexpr bool requires_memlock_unlimited(Service s) noexcept {
    return s == Service::PHYSICS;
}

/// Only physics uses /dev/shm for the Seqlock Ring Buffer (zero-copy IPC).
[[nodiscard]] constexpr bool mounts_dev_shm(Service s) noexcept {
    return s == Service::PHYSICS;
}

// ────────────────────────────────────────────────────────────────────────────
// §10  Volume sharing predicates
// ────────────────────────────────────────────────────────────────────────────

/// All services except executor and web mount /etc/nikola/keys (CurveZMQ keys).
[[nodiscard]] constexpr bool mounts_curve_keys(Service s) noexcept {
    return s != Service::EXECUTOR && s != Service::WEB;
}

/// All services except executor and web share /tmp/nikola/ipc (ZMQ IPC sockets).
[[nodiscard]] constexpr bool mounts_ipc_socket_dir(Service s) noexcept {
    return s != Service::EXECUTOR && s != Service::WEB;
}

/// Two services share the IPC socket volume when both mount it.
[[nodiscard]] constexpr bool services_share_ipc(Service a, Service b) noexcept {
    return mounts_ipc_socket_dir(a) && mounts_ipc_socket_dir(b);
}

// ────────────────────────────────────────────────────────────────────────────
// §11  Healthcheck analysis
// ────────────────────────────────────────────────────────────────────────────

/// Maximum total wait for spine to become healthy: interval × retries.
[[nodiscard]] constexpr uint64_t max_spine_healthcheck_wait_ms() noexcept {
    return HEALTHCHECK_INTERVAL_MS * HEALTHCHECK_MAX_RETRIES; // 25,000 ms
}

/// True when the given number of retries is still within the allowed budget.
[[nodiscard]] constexpr bool is_within_healthcheck_retries(uint8_t retries_used) noexcept {
    return retries_used <= HEALTHCHECK_MAX_RETRIES;
}

/// True when the elapsed time since healthcheck start is within the budget.
[[nodiscard]] constexpr bool is_healthcheck_within_budget(uint64_t elapsed_ms) noexcept {
    return elapsed_ms <= max_spine_healthcheck_wait_ms();
}

// ────────────────────────────────────────────────────────────────────────────
// §12  Startup budget checks
// ────────────────────────────────────────────────────────────────────────────

/// True when elapsed_ms is within the spine startup window.
[[nodiscard]] constexpr bool is_spine_startup_within_budget(uint64_t elapsed_ms) noexcept {
    return elapsed_ms <= SPINE_STARTUP_MAX_MS;
}

/// True when elapsed_ms is within the physics startup window.
[[nodiscard]] constexpr bool is_physics_startup_within_budget(uint64_t elapsed_ms) noexcept {
    return elapsed_ms <= PHYSICS_STARTUP_MAX_MS;
}

/// True when the whole cluster is up within the 30-second spec budget.
[[nodiscard]] constexpr bool is_cluster_startup_within_budget(uint64_t elapsed_ms) noexcept {
    return elapsed_ms <= CLUSTER_STARTUP_MAX_MS;
}

// ────────────────────────────────────────────────────────────────────────────
// §13  Shutdown analysis
// ────────────────────────────────────────────────────────────────────────────

/// True when elapsed_ms is still within the graceful (non-force) shutdown window.
[[nodiscard]] constexpr bool is_within_graceful_shutdown_period(uint64_t elapsed_ms) noexcept {
    return elapsed_ms <= STOP_GRACE_PERIOD_MS;
}

/// Force-kill required once the stop_grace_period is exceeded.
[[nodiscard]] constexpr bool is_force_kill_required(uint64_t elapsed_ms) noexcept {
    return elapsed_ms > STOP_GRACE_PERIOD_MS;
}

/// True when a service uses SIGTERM as its stop_signal (all except default).
/// Only nikola-memory is explicitly annotated; others inherit Docker default (SIGTERM).
[[nodiscard]] constexpr bool uses_sigterm_stop(Service /*s*/) noexcept {
    return true; // SIGTERM is Docker's default and explicitly set for memory
}

/// True when the given shutdown step index is valid.
[[nodiscard]] constexpr bool is_valid_shutdown_step(uint8_t step_index) noexcept {
    return step_index < MEMORY_SHUTDOWN_STEPS;
}

/// True when a shutdown step precedes another (lower index = earlier in sequence).
[[nodiscard]] constexpr bool shutdown_step_precedes(ShutdownStep a, ShutdownStep b) noexcept {
    return static_cast<uint8_t>(a) < static_cast<uint8_t>(b);
}

// ────────────────────────────────────────────────────────────────────────────
// §14  Physics resource validation
// ────────────────────────────────────────────────────────────────────────────

/// True when a thread count is at or below the OMP spec limit.
[[nodiscard]] constexpr bool is_omp_thread_count_valid(uint32_t threads) noexcept {
    return threads > 0u && threads <= PHYSICS_OMP_NUM_THREADS;
}

/// True when a stack size (bytes) meets the spec minimum for deep Hilbert recursion.
[[nodiscard]] constexpr bool is_stack_size_sufficient(uint64_t stack_bytes) noexcept {
    return stack_bytes >= PHYSICS_STACK_BYTES;
}

// ────────────────────────────────────────────────────────────────────────────
// §15  Diagnostic names
// ────────────────────────────────────────────────────────────────────────────

[[nodiscard]] constexpr std::string_view service_name(Service s) noexcept {
    switch (s) {
        case Service::SPINE:        return "nikola-spine";
        case Service::PHYSICS:      return "nikola-physics";
        case Service::MEMORY:       return "nikola-memory";
        case Service::ORCHESTRATOR: return "nikola-orchestrator";
        case Service::EXECUTOR:     return "nikola-executor";
        case Service::WEB:          return "nikola-web";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view dependency_condition_name(DependencyCondition c) noexcept {
    switch (c) {
        case DependencyCondition::NONE:             return "none";
        case DependencyCondition::SERVICE_HEALTHY:  return "service_healthy";
        case DependencyCondition::SERVICE_STARTED:  return "service_started";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view shutdown_step_name(ShutdownStep step) noexcept {
    switch (step) {
        case ShutdownStep::ACQUIRE_WRITE_LOCK: return "ACQUIRE_WRITE_LOCK";
        case ShutdownStep::FLUSH_MEMTABLE:     return "FLUSH_MEMTABLE";
        case ShutdownStep::FSYNC_WAL:          return "FSYNC_WAL";
        case ShutdownStep::WRITE_MANIFEST:     return "WRITE_MANIFEST";
    }
    return "unknown";
}

} // namespace nikola::infrastructure
