// ============================================================
// tests/unit/phase71_compose_orchestration_test.cpp
// Phase 71 — GAP-026  Docker Compose Service Orchestration
// ============================================================
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <array>
#include <cstdint>

#include "nikola/infrastructure/compose_orchestration.hpp"

using namespace nikola::infrastructure;
using Catch::Approx;

// ────────────────────────────────────────────────────────────────────────────
// §1  Topology count constants
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("LAYER_COUNT is 4 (layers 0-3)", "[constants]") {
    CHECK(LAYER_COUNT == 4u);
}

TEST_CASE("SERVICE_COUNT is 6", "[constants]") {
    CHECK(SERVICE_COUNT == 6u);
}

// ────────────────────────────────────────────────────────────────────────────
// §2  Startup timing constants
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("SPINE_STARTUP_MAX_MS is 2 seconds", "[constants]") {
    CHECK(SPINE_STARTUP_MAX_MS == 2'000u);
}

TEST_CASE("PHYSICS_STARTUP_MAX_MS is 5 seconds", "[constants]") {
    CHECK(PHYSICS_STARTUP_MAX_MS == 5'000u);
}

TEST_CASE("CLUSTER_STARTUP_MAX_MS is 30 seconds", "[constants]") {
    CHECK(CLUSTER_STARTUP_MAX_MS == 30'000u);
}

TEST_CASE("Physics startup window is wider than spine startup window", "[constants]") {
    CHECK(SPINE_STARTUP_MAX_MS < PHYSICS_STARTUP_MAX_MS);
}

TEST_CASE("Cluster startup budget exceeds individual service budgets", "[constants]") {
    CHECK(CLUSTER_STARTUP_MAX_MS > SPINE_STARTUP_MAX_MS);
    CHECK(CLUSTER_STARTUP_MAX_MS > PHYSICS_STARTUP_MAX_MS);
}

// ────────────────────────────────────────────────────────────────────────────
// §3  Healthcheck constants
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("HEALTHCHECK_INTERVAL_MS is 5 seconds", "[constants]") {
    CHECK(HEALTHCHECK_INTERVAL_MS == 5'000u);
}

TEST_CASE("HEALTHCHECK_TIMEOUT_MS is 2 seconds", "[constants]") {
    CHECK(HEALTHCHECK_TIMEOUT_MS == 2'000u);
}

TEST_CASE("HEALTHCHECK_MAX_RETRIES is 5", "[constants]") {
    CHECK(HEALTHCHECK_MAX_RETRIES == 5u);
}

TEST_CASE("Healthcheck interval is longer than per-attempt timeout", "[constants]") {
    CHECK(HEALTHCHECK_TIMEOUT_MS < HEALTHCHECK_INTERVAL_MS);
}

// ────────────────────────────────────────────────────────────────────────────
// §4  Shutdown timing constants
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("DEFAULT_STOP_GRACE_MS is 10 seconds", "[constants]") {
    CHECK(DEFAULT_STOP_GRACE_MS == 10'000u);
}

TEST_CASE("STOP_GRACE_PERIOD_MS is 60 seconds (memory service)", "[constants]") {
    CHECK(STOP_GRACE_PERIOD_MS == 60'000u);
}

TEST_CASE("FORCE_KILL_BUDGET_MS matches Docker default", "[constants]") {
    CHECK(FORCE_KILL_BUDGET_MS == DEFAULT_STOP_GRACE_MS);
}

TEST_CASE("Memory grace period is 6x the default", "[constants]") {
    CHECK(STOP_GRACE_PERIOD_MS == 6u * DEFAULT_STOP_GRACE_MS);
}

TEST_CASE("MEMORY_SHUTDOWN_STEPS is 4", "[constants]") {
    CHECK(MEMORY_SHUTDOWN_STEPS == 4u);
}

// ────────────────────────────────────────────────────────────────────────────
// §5  Physics resource constants
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("PHYSICS_OMP_NUM_THREADS is 16", "[constants]") {
    CHECK(PHYSICS_OMP_NUM_THREADS == 16u);
}

TEST_CASE("PHYSICS_STACK_BYTES is exactly 64 MiB", "[constants]") {
    CHECK(PHYSICS_STACK_BYTES == 64u * 1024u * 1024u);
    CHECK(PHYSICS_STACK_BYTES == 67'108'864u);
}

TEST_CASE("PHYSICS_MEMLOCK_UNLIMITED is -1", "[constants]") {
    CHECK(PHYSICS_MEMLOCK_UNLIMITED == -1);
}

// ────────────────────────────────────────────────────────────────────────────
// §6  Spine resource constants
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("SPINE_CPU_CORES_LIMIT is 2.0", "[constants]") {
    CHECK(SPINE_CPU_CORES_LIMIT == Approx(2.0f).epsilon(1e-6f));
}

TEST_CASE("SPINE_MEMORY_LIMIT_GIB is 4", "[constants]") {
    CHECK(SPINE_MEMORY_LIMIT_GIB == 4u);
}

// ────────────────────────────────────────────────────────────────────────────
// §7  service_layer — 4-layer hierarchy
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("service_layer: Layer 0 — spine has no dependencies", "[topology]") {
    CHECK(service_layer(Service::SPINE) == 0u);
}

TEST_CASE("service_layer: Layer 1 — physics depends on spine", "[topology]") {
    CHECK(service_layer(Service::PHYSICS) == 1u);
}

TEST_CASE("service_layer: Layer 2 — orchestrator and memory", "[topology]") {
    CHECK(service_layer(Service::MEMORY)       == 2u);
    CHECK(service_layer(Service::ORCHESTRATOR) == 2u);
}

TEST_CASE("service_layer: Layer 3 — executor and web are last", "[topology]") {
    CHECK(service_layer(Service::EXECUTOR) == 3u);
    CHECK(service_layer(Service::WEB)      == 3u);
}

TEST_CASE("Layers are within [0, LAYER_COUNT)", "[topology]") {
    constexpr std::array services = {
        Service::SPINE, Service::PHYSICS, Service::MEMORY,
        Service::ORCHESTRATOR, Service::EXECUTOR, Service::WEB
    };
    for (auto s : services) {
        CHECK(service_layer(s) < LAYER_COUNT);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// §8  service_start_order
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("service_start_order matches service_layer", "[topology]") {
    constexpr std::array services = {
        Service::SPINE, Service::PHYSICS, Service::MEMORY,
        Service::ORCHESTRATOR, Service::EXECUTOR, Service::WEB
    };
    for (auto s : services) {
        CHECK(service_start_order(s) == service_layer(s));
    }
}

TEST_CASE("Spine starts before physics, which starts before layer-2 services", "[topology]") {
    CHECK(service_start_order(Service::SPINE)   < service_start_order(Service::PHYSICS));
    CHECK(service_start_order(Service::PHYSICS) < service_start_order(Service::MEMORY));
    CHECK(service_start_order(Service::PHYSICS) < service_start_order(Service::ORCHESTRATOR));
}

// ────────────────────────────────────────────────────────────────────────────
// §9  depends_on_spine
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("depends_on_spine: only physics and memory depend directly on spine", "[topology]") {
    CHECK(depends_on_spine(Service::PHYSICS)      == true);
    CHECK(depends_on_spine(Service::MEMORY)       == true);
    CHECK(depends_on_spine(Service::SPINE)        == false);
    CHECK(depends_on_spine(Service::ORCHESTRATOR) == false);
    CHECK(depends_on_spine(Service::EXECUTOR)     == false);
    CHECK(depends_on_spine(Service::WEB)          == false);
}

// ────────────────────────────────────────────────────────────────────────────
// §10  depends_on_orchestrator
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("depends_on_orchestrator: only executor and web depend on orchestrator", "[topology]") {
    CHECK(depends_on_orchestrator(Service::EXECUTOR)     == true);
    CHECK(depends_on_orchestrator(Service::WEB)          == true);
    CHECK(depends_on_orchestrator(Service::ORCHESTRATOR) == false);
    CHECK(depends_on_orchestrator(Service::SPINE)        == false);
    CHECK(depends_on_orchestrator(Service::PHYSICS)      == false);
    CHECK(depends_on_orchestrator(Service::MEMORY)       == false);
}

// ────────────────────────────────────────────────────────────────────────────
// §11  has_upstream_dependency
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("has_upstream_dependency: only spine has no upstream", "[topology]") {
    CHECK(has_upstream_dependency(Service::SPINE)        == false);
    CHECK(has_upstream_dependency(Service::PHYSICS)      == true);
    CHECK(has_upstream_dependency(Service::MEMORY)       == true);
    CHECK(has_upstream_dependency(Service::ORCHESTRATOR) == true);
    CHECK(has_upstream_dependency(Service::EXECUTOR)     == true);
    CHECK(has_upstream_dependency(Service::WEB)          == true);
}

// ────────────────────────────────────────────────────────────────────────────
// §12  dependency_condition
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("dependency_condition: spine — NONE (no upstream)", "[topology]") {
    CHECK(dependency_condition(Service::SPINE) == DependencyCondition::NONE);
}

TEST_CASE("dependency_condition: physics and memory wait for spine to be HEALTHY", "[topology]") {
    CHECK(dependency_condition(Service::PHYSICS) == DependencyCondition::SERVICE_HEALTHY);
    CHECK(dependency_condition(Service::MEMORY)  == DependencyCondition::SERVICE_HEALTHY);
}

TEST_CASE("dependency_condition: layer 2+ use SERVICE_STARTED", "[topology]") {
    CHECK(dependency_condition(Service::ORCHESTRATOR) == DependencyCondition::SERVICE_STARTED);
    CHECK(dependency_condition(Service::EXECUTOR)     == DependencyCondition::SERVICE_STARTED);
    CHECK(dependency_condition(Service::WEB)          == DependencyCondition::SERVICE_STARTED);
}

// ────────────────────────────────────────────────────────────────────────────
// §13  Resource requirements
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("requires_gpu_runtime: only physics needs nvidia runtime", "[resources]") {
    CHECK(requires_gpu_runtime(Service::PHYSICS)      == true);
    CHECK(requires_gpu_runtime(Service::SPINE)        == false);
    CHECK(requires_gpu_runtime(Service::MEMORY)       == false);
    CHECK(requires_gpu_runtime(Service::ORCHESTRATOR) == false);
    CHECK(requires_gpu_runtime(Service::EXECUTOR)     == false);
    CHECK(requires_gpu_runtime(Service::WEB)          == false);
}

TEST_CASE("requires_privileged_mode: only executor needs privileged for KVM", "[resources]") {
    CHECK(requires_privileged_mode(Service::EXECUTOR)     == true);
    CHECK(requires_privileged_mode(Service::PHYSICS)      == false);
    CHECK(requires_privileged_mode(Service::SPINE)        == false);
    CHECK(requires_privileged_mode(Service::ORCHESTRATOR) == false);
    CHECK(requires_privileged_mode(Service::MEMORY)       == false);
    CHECK(requires_privileged_mode(Service::WEB)          == false);
}

TEST_CASE("requires_memlock_unlimited: only physics locks GPU pages", "[resources]") {
    CHECK(requires_memlock_unlimited(Service::PHYSICS)      == true);
    CHECK(requires_memlock_unlimited(Service::SPINE)        == false);
    CHECK(requires_memlock_unlimited(Service::MEMORY)       == false);
    CHECK(requires_memlock_unlimited(Service::ORCHESTRATOR) == false);
    CHECK(requires_memlock_unlimited(Service::EXECUTOR)     == false);
    CHECK(requires_memlock_unlimited(Service::WEB)          == false);
}

TEST_CASE("mounts_dev_shm: only physics uses Seqlock Ring Buffer", "[resources]") {
    CHECK(mounts_dev_shm(Service::PHYSICS)      == true);
    CHECK(mounts_dev_shm(Service::SPINE)        == false);
    CHECK(mounts_dev_shm(Service::MEMORY)       == false);
    CHECK(mounts_dev_shm(Service::ORCHESTRATOR) == false);
    CHECK(mounts_dev_shm(Service::EXECUTOR)     == false);
    CHECK(mounts_dev_shm(Service::WEB)          == false);
}

// ────────────────────────────────────────────────────────────────────────────
// §14  Volume sharing
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("mounts_curve_keys: all except executor and web mount /etc/nikola/keys", "[volumes]") {
    CHECK(mounts_curve_keys(Service::SPINE)        == true);
    CHECK(mounts_curve_keys(Service::PHYSICS)      == true);
    CHECK(mounts_curve_keys(Service::MEMORY)       == true);
    CHECK(mounts_curve_keys(Service::ORCHESTRATOR) == true);
    CHECK(mounts_curve_keys(Service::EXECUTOR)     == false);
    CHECK(mounts_curve_keys(Service::WEB)          == false);
}

TEST_CASE("mounts_ipc_socket_dir: same set as curve keys", "[volumes]") {
    CHECK(mounts_ipc_socket_dir(Service::SPINE)        == true);
    CHECK(mounts_ipc_socket_dir(Service::PHYSICS)      == true);
    CHECK(mounts_ipc_socket_dir(Service::MEMORY)       == true);
    CHECK(mounts_ipc_socket_dir(Service::ORCHESTRATOR) == true);
    CHECK(mounts_ipc_socket_dir(Service::EXECUTOR)     == false);
    CHECK(mounts_ipc_socket_dir(Service::WEB)          == false);
}

TEST_CASE("services_share_ipc: spine-physics, spine-memory, physics-orchestrator all share IPC", "[volumes]") {
    CHECK(services_share_ipc(Service::SPINE,   Service::PHYSICS)      == true);
    CHECK(services_share_ipc(Service::SPINE,   Service::MEMORY)       == true);
    CHECK(services_share_ipc(Service::PHYSICS, Service::ORCHESTRATOR) == true);
}

TEST_CASE("services_share_ipc: executor and web do NOT share IPC with anyone", "[volumes]") {
    CHECK(services_share_ipc(Service::EXECUTOR,     Service::SPINE)   == false);
    CHECK(services_share_ipc(Service::WEB,          Service::PHYSICS) == false);
    CHECK(services_share_ipc(Service::ORCHESTRATOR, Service::WEB)     == false);
}

TEST_CASE("services_share_ipc is symmetric", "[volumes]") {
    CHECK(services_share_ipc(Service::SPINE, Service::PHYSICS) ==
          services_share_ipc(Service::PHYSICS, Service::SPINE));
    CHECK(services_share_ipc(Service::EXECUTOR, Service::MEMORY) ==
          services_share_ipc(Service::MEMORY, Service::EXECUTOR));
}

// ────────────────────────────────────────────────────────────────────────────
// §15  Healthcheck analysis
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("max_spine_healthcheck_wait_ms = 5000 * 5 = 25,000 ms", "[healthcheck]") {
    CHECK(max_spine_healthcheck_wait_ms() == 25'000u);
}

TEST_CASE("is_within_healthcheck_retries: at boundary passes", "[healthcheck]") {
    CHECK(is_within_healthcheck_retries(0u) == true);
    CHECK(is_within_healthcheck_retries(5u) == true);   // exactly 5 passes
    CHECK(is_within_healthcheck_retries(6u) == false);
}

TEST_CASE("is_healthcheck_within_budget: at boundary", "[healthcheck]") {
    CHECK(is_healthcheck_within_budget(0u)        == true);
    CHECK(is_healthcheck_within_budget(25'000u)   == true);   // exactly at budget
    CHECK(is_healthcheck_within_budget(25'001u)   == false);
}

// ────────────────────────────────────────────────────────────────────────────
// §16  Startup budget checks
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("is_spine_startup_within_budget: boundary at 2000 ms", "[startup]") {
    CHECK(is_spine_startup_within_budget(0u)      == true);
    CHECK(is_spine_startup_within_budget(2'000u)  == true);
    CHECK(is_spine_startup_within_budget(2'001u)  == false);
}

TEST_CASE("is_physics_startup_within_budget: boundary at 5000 ms", "[startup]") {
    CHECK(is_physics_startup_within_budget(0u)     == true);
    CHECK(is_physics_startup_within_budget(5'000u) == true);
    CHECK(is_physics_startup_within_budget(5'001u) == false);
}

TEST_CASE("is_cluster_startup_within_budget: boundary at 30,000 ms", "[startup]") {
    CHECK(is_cluster_startup_within_budget(0u)       == true);
    CHECK(is_cluster_startup_within_budget(30'000u)  == true);
    CHECK(is_cluster_startup_within_budget(30'001u)  == false);
}

// ────────────────────────────────────────────────────────────────────────────
// §17  Shutdown analysis
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("is_within_graceful_shutdown_period: exactly at 60 s still ok", "[shutdown]") {
    CHECK(is_within_graceful_shutdown_period(0u)       == true);
    CHECK(is_within_graceful_shutdown_period(60'000u)  == true);
    CHECK(is_within_graceful_shutdown_period(60'001u)  == false);
}

TEST_CASE("is_force_kill_required: triggered strictly after 60 s", "[shutdown]") {
    CHECK(is_force_kill_required(60'000u) == false);   // still within grace
    CHECK(is_force_kill_required(60'001u) == true);
    CHECK(is_force_kill_required(120'000u) == true);
}

TEST_CASE("is_force_kill_required and is_within_graceful are exclusive at boundary+1", "[shutdown]") {
    // At any elapsed_ms, exactly one is true
    for (uint64_t t : {0u, 30'000u, 60'000u, 60'001u, 120'000u}) {
        const bool in_grace = is_within_graceful_shutdown_period(t);
        const bool force    = is_force_kill_required(t);
        CHECK(in_grace != force);
    }
}

TEST_CASE("is_valid_shutdown_step: 0-3 valid, 4+ invalid", "[shutdown]") {
    CHECK(is_valid_shutdown_step(0u) == true);
    CHECK(is_valid_shutdown_step(3u) == true);
    CHECK(is_valid_shutdown_step(4u) == false);
}

TEST_CASE("shutdown_step_precedes: ACQUIRE_WRITE_LOCK before all others", "[shutdown]") {
    CHECK(shutdown_step_precedes(ShutdownStep::ACQUIRE_WRITE_LOCK, ShutdownStep::FLUSH_MEMTABLE) == true);
    CHECK(shutdown_step_precedes(ShutdownStep::ACQUIRE_WRITE_LOCK, ShutdownStep::FSYNC_WAL)      == true);
    CHECK(shutdown_step_precedes(ShutdownStep::ACQUIRE_WRITE_LOCK, ShutdownStep::WRITE_MANIFEST) == true);
}

TEST_CASE("shutdown_step_precedes: WRITE_MANIFEST is last", "[shutdown]") {
    CHECK(shutdown_step_precedes(ShutdownStep::FLUSH_MEMTABLE,    ShutdownStep::WRITE_MANIFEST) == true);
    CHECK(shutdown_step_precedes(ShutdownStep::FSYNC_WAL,         ShutdownStep::WRITE_MANIFEST) == true);
    CHECK(shutdown_step_precedes(ShutdownStep::WRITE_MANIFEST,    ShutdownStep::FLUSH_MEMTABLE) == false);
}

TEST_CASE("shutdown_step_precedes: irreflexive (no step precedes itself)", "[shutdown]") {
    CHECK(shutdown_step_precedes(ShutdownStep::ACQUIRE_WRITE_LOCK, ShutdownStep::ACQUIRE_WRITE_LOCK) == false);
    CHECK(shutdown_step_precedes(ShutdownStep::WRITE_MANIFEST,     ShutdownStep::WRITE_MANIFEST)     == false);
}

// ────────────────────────────────────────────────────────────────────────────
// §18  Physics resource validation
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("is_omp_thread_count_valid: 1..16 valid, 0 and 17+ invalid", "[physics_resources]") {
    CHECK(is_omp_thread_count_valid(1u)  == true);
    CHECK(is_omp_thread_count_valid(16u) == true);   // exactly max
    CHECK(is_omp_thread_count_valid(17u) == false);
    CHECK(is_omp_thread_count_valid(0u)  == false);
}

TEST_CASE("is_stack_size_sufficient: below 64 MiB fails, at or above passes", "[physics_resources]") {
    CHECK(is_stack_size_sufficient(PHYSICS_STACK_BYTES)     == true);
    CHECK(is_stack_size_sufficient(PHYSICS_STACK_BYTES + 1) == true);
    CHECK(is_stack_size_sufficient(PHYSICS_STACK_BYTES - 1) == false);
    CHECK(is_stack_size_sufficient(0u)                      == false);
}

// ────────────────────────────────────────────────────────────────────────────
// §19  Diagnostic names
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("service_name: correct container names", "[diagnostics]") {
    CHECK(service_name(Service::SPINE)        == "nikola-spine");
    CHECK(service_name(Service::PHYSICS)      == "nikola-physics");
    CHECK(service_name(Service::MEMORY)       == "nikola-memory");
    CHECK(service_name(Service::ORCHESTRATOR) == "nikola-orchestrator");
    CHECK(service_name(Service::EXECUTOR)     == "nikola-executor");
    CHECK(service_name(Service::WEB)          == "nikola-web");
}

TEST_CASE("dependency_condition_name: Compose YAML values", "[diagnostics]") {
    CHECK(dependency_condition_name(DependencyCondition::NONE)            == "none");
    CHECK(dependency_condition_name(DependencyCondition::SERVICE_HEALTHY) == "service_healthy");
    CHECK(dependency_condition_name(DependencyCondition::SERVICE_STARTED) == "service_started");
}

TEST_CASE("shutdown_step_name: all 4 steps named", "[diagnostics]") {
    CHECK(shutdown_step_name(ShutdownStep::ACQUIRE_WRITE_LOCK) == "ACQUIRE_WRITE_LOCK");
    CHECK(shutdown_step_name(ShutdownStep::FLUSH_MEMTABLE)     == "FLUSH_MEMTABLE");
    CHECK(shutdown_step_name(ShutdownStep::FSYNC_WAL)          == "FSYNC_WAL");
    CHECK(shutdown_step_name(ShutdownStep::WRITE_MANIFEST)     == "WRITE_MANIFEST");
}

TEST_CASE("Diagnostic names are non-empty", "[diagnostics]") {
    for (auto s : {Service::SPINE, Service::PHYSICS, Service::MEMORY,
                   Service::ORCHESTRATOR, Service::EXECUTOR, Service::WEB}) {
        CHECK_FALSE(service_name(s).empty());
    }
    CHECK_FALSE(dependency_condition_name(DependencyCondition::SERVICE_HEALTHY).empty());
    CHECK_FALSE(shutdown_step_name(ShutdownStep::WRITE_MANIFEST).empty());
}

// ────────────────────────────────────────────────────────────────────────────
// §20  Invariants
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("Invariant: layer 0 has no dependencies", "[invariants]") {
    CHECK(has_upstream_dependency(Service::SPINE)        == false);
    CHECK(depends_on_spine(Service::SPINE)               == false);
    CHECK(depends_on_orchestrator(Service::SPINE)        == false);
    CHECK(dependency_condition(Service::SPINE)           == DependencyCondition::NONE);
}

TEST_CASE("Invariant: healthcheck total budget = interval × retries", "[invariants]") {
    CHECK(max_spine_healthcheck_wait_ms() ==
          static_cast<uint64_t>(HEALTHCHECK_INTERVAL_MS) * HEALTHCHECK_MAX_RETRIES);
}

TEST_CASE("Invariant: 64 MiB stack = 67,108,864 bytes", "[invariants]") {
    CHECK(PHYSICS_STACK_BYTES == 67'108'864u);
}

TEST_CASE("Invariant: STOP_GRACE_PERIOD_MS > DEFAULT_STOP_GRACE_MS", "[invariants]") {
    CHECK(STOP_GRACE_PERIOD_MS > DEFAULT_STOP_GRACE_MS);
}

TEST_CASE("Invariant: only one service may require GPU runtime", "[invariants]") {
    uint32_t gpu_count = 0u;
    for (auto s : {Service::SPINE, Service::PHYSICS, Service::MEMORY,
                   Service::ORCHESTRATOR, Service::EXECUTOR, Service::WEB}) {
        if (requires_gpu_runtime(s)) ++gpu_count;
    }
    CHECK(gpu_count == 1u);
}

TEST_CASE("Invariant: only one service may require privileged mode", "[invariants]") {
    uint32_t priv_count = 0u;
    for (auto s : {Service::SPINE, Service::PHYSICS, Service::MEMORY,
                   Service::ORCHESTRATOR, Service::EXECUTOR, Service::WEB}) {
        if (requires_privileged_mode(s)) ++priv_count;
    }
    CHECK(priv_count == 1u);
}

TEST_CASE("Invariant: service_layer is consistent with dependency ordering", "[invariants]") {
    // Every spine-dependent service must be in a higher layer
    CHECK(service_layer(Service::PHYSICS) > service_layer(Service::SPINE));
    CHECK(service_layer(Service::MEMORY)  > service_layer(Service::SPINE));
    // Every orchestrator-dependent service must be in a higher layer
    CHECK(service_layer(Service::EXECUTOR) > service_layer(Service::ORCHESTRATOR));
    CHECK(service_layer(Service::WEB)      > service_layer(Service::ORCHESTRATOR));
}

TEST_CASE("Invariant: all 6 service names are unique", "[invariants]") {
    constexpr std::array services = {
        Service::SPINE, Service::PHYSICS, Service::MEMORY,
        Service::ORCHESTRATOR, Service::EXECUTOR, Service::WEB
    };
    for (std::size_t i = 0; i < services.size(); ++i) {
        for (std::size_t j = i + 1; j < services.size(); ++j) {
            CHECK(service_name(services[i]) != service_name(services[j]));
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// §21  Integration scenarios
// ────────────────────────────────────────────────────────────────────────────

TEST_CASE("Integration: normal startup — spine → physics → cognition → tools", "[integration]") {
    // Spine starts with no dependencies
    CHECK(has_upstream_dependency(Service::SPINE) == false);
    CHECK(is_spine_startup_within_budget(1'500u)  == true);

    // Spine healthcheck completes at attempt 2 (10 seconds elapsed)
    CHECK(is_within_healthcheck_retries(2u)       == true);
    CHECK(is_healthcheck_within_budget(10'000u)   == true);

    // Physics starts after healthcheck
    CHECK(dependency_condition(Service::PHYSICS)  == DependencyCondition::SERVICE_HEALTHY);
    CHECK(is_physics_startup_within_budget(4'000u) == true);

    // Orchestrator and Memory start after physics and spine (service_started)
    CHECK(dependency_condition(Service::ORCHESTRATOR) == DependencyCondition::SERVICE_STARTED);

    // Cluster is ready in 25 seconds
    CHECK(is_cluster_startup_within_budget(25'000u) == true);
}

TEST_CASE("Integration: healthcheck failure race — max retries exceeded", "[integration]") {
    // Attempt 6 exceeds max
    CHECK(is_within_healthcheck_retries(6u) == false);
    // Time budget: 30,001 ms past healthcheck budget
    CHECK(is_healthcheck_within_budget(26'000u) == false);
}

TEST_CASE("Integration: graceful shutdown of nikola-memory", "[integration]") {
    // Memory uses SIGTERM and has extended grace period
    CHECK(uses_sigterm_stop(Service::MEMORY) == true);

    // All 4 steps are valid
    for (uint8_t i = 0; i < MEMORY_SHUTDOWN_STEPS; ++i) {
        CHECK(is_valid_shutdown_step(i) == true);
    }
    CHECK(is_valid_shutdown_step(MEMORY_SHUTDOWN_STEPS) == false);

    // Ordered correctly
    CHECK(shutdown_step_precedes(ShutdownStep::ACQUIRE_WRITE_LOCK, ShutdownStep::FLUSH_MEMTABLE) == true);
    CHECK(shutdown_step_precedes(ShutdownStep::FLUSH_MEMTABLE,     ShutdownStep::FSYNC_WAL)      == true);
    CHECK(shutdown_step_precedes(ShutdownStep::FSYNC_WAL,          ShutdownStep::WRITE_MANIFEST) == true);

    // 59 seconds — still within grace; 61 → force-kill
    CHECK(is_within_graceful_shutdown_period(59'000u) == true);
    CHECK(is_force_kill_required(61'000u)             == true);
}

TEST_CASE("Integration: IPC sharing topology mirrors ZeroMQ bus", "[integration]") {
    // Spine is the hub: it shares IPC with physics, memory, and orchestrator
    CHECK(services_share_ipc(Service::SPINE, Service::PHYSICS)      == true);
    CHECK(services_share_ipc(Service::SPINE, Service::MEMORY)       == true);
    CHECK(services_share_ipc(Service::SPINE, Service::ORCHESTRATOR) == true);
    // Executor (KVM sandbox) and web are NOT on the IPC bus
    CHECK(services_share_ipc(Service::SPINE, Service::EXECUTOR)     == false);
    CHECK(services_share_ipc(Service::SPINE, Service::WEB)          == false);
}

TEST_CASE("Integration: physics cannot start until spine is healthy (prevents Cryptographic Amnesia)", "[integration]") {
    // Physics uses SERVICE_HEALTHY (not SERVICE_STARTED), ensuring CurveZMQ key exchange succeeds
    CHECK(dependency_condition(Service::PHYSICS) == DependencyCondition::SERVICE_HEALTHY);
    // Its CurveZMQ keys are mounted
    CHECK(mounts_curve_keys(Service::PHYSICS) == true);
    // And it connects via the IPC socket
    CHECK(mounts_ipc_socket_dir(Service::PHYSICS) == true);
}
