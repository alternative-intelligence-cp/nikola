// SPDX-License-Identifier: MIT
// GAP-026: Docker Compose Service Orchestration Policy
// Phase 93 — nikola::system
//
// Machine-checkable encoding of the 4-layer service dependency hierarchy,
// healthcheck parameters, resource limits and shutdown policies defined in
// the Nikola Docker Compose specification.
//
// The actual docker-compose.yml is in the repository root; this header
// makes the key numerical constants available to integration tests and
// readiness-probe code that must agree with the compose file.
//
// Source: 02_orchestrator_router.md §"Docker Compose Service Orchestration"

#pragma once

#include <cstdint>
#include <string_view>

namespace nikola::system {

// ─── Service layer taxonomy ───────────────────────────────────────────────────

/// The 4-layer startup hierarchy for the Nikola distributed system.
enum class ServiceLayer : uint8_t {
    LAYER_0_CORE        = 0,  ///< nikola-spine   (ZeroMQ broker — no deps)
    LAYER_1_PHYSICS     = 1,  ///< nikola-physics (GPU engine — depends on spine)
    LAYER_2_COGNITION   = 2,  ///< nikola-orchestrator + nikola-memory
    LAYER_3_TOOLS       = 3   ///< nikola-executor + nikola-web
};

/// Named services that the orchestration policy applies to.
enum class ServiceId : uint8_t {
    SPINE         = 0,
    PHYSICS       = 1,
    MEMORY        = 2,
    ORCHESTRATOR  = 3,
    EXECUTOR      = 4,
    WEB           = 5
};

// ─── Healthcheck parameters (nikola-spine) ───────────────────────────────────

/// Interval between spine healthcheck probes (seconds).
inline constexpr uint32_t SPINE_HEALTHCHECK_INTERVAL_S  = 5;

/// Timeout for one healthcheck probe (seconds).
inline constexpr uint32_t SPINE_HEALTHCHECK_TIMEOUT_S   = 2;

/// Number of consecutive failures before the container is marked unhealthy.
inline constexpr uint32_t SPINE_HEALTHCHECK_RETRIES     = 5;

/// Maximum time (seconds) it may take to reach "healthy" from first start.
/// start_period = retries × (interval + timeout) rounded up conservatively.
inline constexpr uint32_t SPINE_HEALTHCHECK_START_WINDOW_S  =
    SPINE_HEALTHCHECK_RETRIES * (SPINE_HEALTHCHECK_INTERVAL_S + SPINE_HEALTHCHECK_TIMEOUT_S); // 35 s

// ─── Resource limits ──────────────────────────────────────────────────────────

/// Maximum memory for the spine container (bytes, 4 GiB).
inline constexpr uint64_t SPINE_MEMORY_LIMIT_BYTES       = 4ULL * 1024 * 1024 * 1024;

/// CPU count allocated to the spine container.
inline constexpr double   SPINE_CPU_LIMIT                = 2.0;

/// OpenMP thread count for AVX-512 sections inside nikola-physics.
inline constexpr uint32_t PHYSICS_OMP_NUM_THREADS        = 16;

/// Stack size for the physics container (bytes, 64 MiB) — needed for
/// deep Mamba-9D recursion up to D_hard = 12 levels.
inline constexpr uint64_t PHYSICS_STACK_SIZE_BYTES       = 64ULL * 1024 * 1024;

// ─── Shutdown policy ─────────────────────────────────────────────────────────

/// Signal sent to nikola-memory on `docker compose down` to trigger graceful
/// LSM WAL flush before termination.
inline constexpr std::string_view MEMORY_STOP_SIGNAL     = "SIGTERM";

/// Grace period allowed for nikola-memory to complete its WAL flush (seconds).
inline constexpr uint32_t MEMORY_STOP_GRACE_PERIOD_S     = 60;

// ─── Security model ───────────────────────────────────────────────────────────

/// Security model used for ZeroMQ inter-container communication.
inline constexpr std::string_view ZMQ_SECURITY_MODEL     = "CurveZMQ";  ///< Ironhouse

/// Environment variable that enables CurveZMQ server mode in the spine.
inline constexpr std::string_view ZMQ_CURVE_SERVER_ENV   = "ZMQ_CURVE_SERVER";

// ─── IPC path ────────────────────────────────────────────────────────────────

/// Host path mounted into all containers for IPC sockets.
inline constexpr std::string_view IPC_SOCKET_HOST_PATH   = "/tmp/nikola/ipc";

/// Host path mounted read-only for CurveZMQ key material.
inline constexpr std::string_view CURVZMQ_KEYS_HOST_PATH = "/etc/nikola/keys";

// ─── Container name constants ─────────────────────────────────────────────────

inline constexpr std::string_view CONTAINER_SPINE        = "nikola-spine";
inline constexpr std::string_view CONTAINER_PHYSICS      = "nikola-physics";
inline constexpr std::string_view CONTAINER_MEMORY       = "nikola-memory";
inline constexpr std::string_view CONTAINER_ORCHESTRATOR = "nikola-orchestrator";
inline constexpr std::string_view CONTAINER_EXECUTOR     = "nikola-executor";
inline constexpr std::string_view CONTAINER_WEB          = "nikola-web";

// ─── Policy predicates ───────────────────────────────────────────────────────

/// True when the spine has passed all retries and may be considered healthy.
[[nodiscard]] constexpr bool spine_healthy(uint32_t consecutive_passes) noexcept {
    return consecutive_passes >= SPINE_HEALTHCHECK_RETRIES;
}

/// True when memory shutdown is within the allowed grace period.
[[nodiscard]] constexpr bool memory_within_grace_period(uint32_t elapsed_s) noexcept {
    return elapsed_s <= MEMORY_STOP_GRACE_PERIOD_S;
}

// ─── Layer / service label helpers ───────────────────────────────────────────

[[nodiscard]] constexpr std::string_view service_layer_label(ServiceLayer l) noexcept {
    switch (l) {
        case ServiceLayer::LAYER_0_CORE:      return "Layer 0 (Core)";
        case ServiceLayer::LAYER_1_PHYSICS:   return "Layer 1 (Physics)";
        case ServiceLayer::LAYER_2_COGNITION: return "Layer 2 (Cognition & Memory)";
        case ServiceLayer::LAYER_3_TOOLS:     return "Layer 3 (Tools & Interface)";
        default:                              return "Unknown";
    }
}

[[nodiscard]] constexpr std::string_view service_id_label(ServiceId s) noexcept {
    switch (s) {
        case ServiceId::SPINE:        return "nikola-spine";
        case ServiceId::PHYSICS:      return "nikola-physics";
        case ServiceId::MEMORY:       return "nikola-memory";
        case ServiceId::ORCHESTRATOR: return "nikola-orchestrator";
        case ServiceId::EXECUTOR:     return "nikola-executor";
        case ServiceId::WEB:          return "nikola-web";
        default:                      return "unknown";
    }
}

/// The service layer a given service belongs to.
[[nodiscard]] constexpr ServiceLayer layer_of(ServiceId s) noexcept {
    switch (s) {
        case ServiceId::SPINE:        return ServiceLayer::LAYER_0_CORE;
        case ServiceId::PHYSICS:      return ServiceLayer::LAYER_1_PHYSICS;
        case ServiceId::MEMORY:
        case ServiceId::ORCHESTRATOR: return ServiceLayer::LAYER_2_COGNITION;
        default:                      return ServiceLayer::LAYER_3_TOOLS;
    }
}

} // namespace nikola::system
