// phase78_docker_compose_orchestration_test.cpp
//
// GAP-026: Docker Compose Service Orchestration
// Spec: docs/info/integration/sections/04_infrastructure/02_orchestrator_router.md §GAP-026
//
// Tests: 4-layer topology, healthcheck parameters, startup/shutdown timing,
// resource limits, service capability flags, shutdown state machine predicates.

#include <catch2/catch_test_macros.hpp>
#include <nikola/infrastructure/docker_compose_orchestration.hpp>

using namespace nikola::infrastructure;

// ─── §1 ServiceLayer Enum Values ─────────────────────────────────────────────

TEST_CASE("ServiceLayer CORE is layer 0", "[layer][enum]") {
    REQUIRE(static_cast<uint8_t>(ServiceLayer::CORE) == 0);
}

TEST_CASE("ServiceLayer PHYSICS is layer 1", "[layer][enum]") {
    REQUIRE(static_cast<uint8_t>(ServiceLayer::PHYSICS) == 1);
}

TEST_CASE("ServiceLayer COGNITION is layer 2", "[layer][enum]") {
    REQUIRE(static_cast<uint8_t>(ServiceLayer::COGNITION) == 2);
}

TEST_CASE("ServiceLayer TOOLS is layer 3", "[layer][enum]") {
    REQUIRE(static_cast<uint8_t>(ServiceLayer::TOOLS) == 3);
}

TEST_CASE("Four distinct service layers", "[layer][enum]") {
    REQUIRE(ServiceLayer::CORE      != ServiceLayer::PHYSICS);
    REQUIRE(ServiceLayer::PHYSICS   != ServiceLayer::COGNITION);
    REQUIRE(ServiceLayer::COGNITION != ServiceLayer::TOOLS);
}

// ─── §2 NikolaService Enum Values ────────────────────────────────────────────

TEST_CASE("NikolaService SPINE is 0", "[service][enum]") {
    REQUIRE(static_cast<uint8_t>(NikolaService::SPINE) == 0);
}

TEST_CASE("NikolaService PHYSICS is 1", "[service][enum]") {
    REQUIRE(static_cast<uint8_t>(NikolaService::PHYSICS) == 1);
}

TEST_CASE("NikolaService MEMORY is 2", "[service][enum]") {
    REQUIRE(static_cast<uint8_t>(NikolaService::MEMORY) == 2);
}

TEST_CASE("NikolaService ORCHESTRATOR is 3", "[service][enum]") {
    REQUIRE(static_cast<uint8_t>(NikolaService::ORCHESTRATOR) == 3);
}

TEST_CASE("NikolaService EXECUTOR is 4", "[service][enum]") {
    REQUIRE(static_cast<uint8_t>(NikolaService::EXECUTOR) == 4);
}

TEST_CASE("NikolaService WEB is 5", "[service][enum]") {
    REQUIRE(static_cast<uint8_t>(NikolaService::WEB) == 5);
}

// ─── §3 ShutdownPhase Enum Values and Ordering ───────────────────────────────

TEST_CASE("ShutdownPhase RUNNING is 0", "[shutdown][enum]") {
    REQUIRE(static_cast<uint8_t>(ShutdownPhase::RUNNING) == 0);
}

TEST_CASE("ShutdownPhase SIGTERM_RECEIVED is 1", "[shutdown][enum]") {
    REQUIRE(static_cast<uint8_t>(ShutdownPhase::SIGTERM_RECEIVED) == 1);
}

TEST_CASE("ShutdownPhase HALT_BROADCAST is 2", "[shutdown][enum]") {
    REQUIRE(static_cast<uint8_t>(ShutdownPhase::HALT_BROADCAST) == 2);
}

TEST_CASE("ShutdownPhase PHYSICS_TICK_COMPLETE is 3", "[shutdown][enum]") {
    REQUIRE(static_cast<uint8_t>(ShutdownPhase::PHYSICS_TICK_COMPLETE) == 3);
}

TEST_CASE("ShutdownPhase MEMORY_LOCK_ACQUIRED is 4", "[shutdown][enum]") {
    REQUIRE(static_cast<uint8_t>(ShutdownPhase::MEMORY_LOCK_ACQUIRED) == 4);
}

TEST_CASE("ShutdownPhase MEMTABLE_FLUSHED is 5", "[shutdown][enum]") {
    REQUIRE(static_cast<uint8_t>(ShutdownPhase::MEMTABLE_FLUSHED) == 5);
}

TEST_CASE("ShutdownPhase WAL_FSYNCED is 6", "[shutdown][enum]") {
    REQUIRE(static_cast<uint8_t>(ShutdownPhase::WAL_FSYNCED) == 6);
}

TEST_CASE("ShutdownPhase MANIFEST_WRITTEN is 7", "[shutdown][enum]") {
    REQUIRE(static_cast<uint8_t>(ShutdownPhase::MANIFEST_WRITTEN) == 7);
}

TEST_CASE("ShutdownPhase TERMINATED is 8", "[shutdown][enum]") {
    REQUIRE(static_cast<uint8_t>(ShutdownPhase::TERMINATED) == 8);
}

TEST_CASE("Shutdown phases are strictly ordered", "[shutdown][enum]") {
    REQUIRE(ShutdownPhase::SIGTERM_RECEIVED      > ShutdownPhase::RUNNING);
    REQUIRE(ShutdownPhase::HALT_BROADCAST        > ShutdownPhase::SIGTERM_RECEIVED);
    REQUIRE(ShutdownPhase::PHYSICS_TICK_COMPLETE > ShutdownPhase::HALT_BROADCAST);
    REQUIRE(ShutdownPhase::MEMORY_LOCK_ACQUIRED  > ShutdownPhase::PHYSICS_TICK_COMPLETE);
    REQUIRE(ShutdownPhase::MEMTABLE_FLUSHED      > ShutdownPhase::MEMORY_LOCK_ACQUIRED);
    REQUIRE(ShutdownPhase::WAL_FSYNCED           > ShutdownPhase::MEMTABLE_FLUSHED);
    REQUIRE(ShutdownPhase::MANIFEST_WRITTEN      > ShutdownPhase::WAL_FSYNCED);
    REQUIRE(ShutdownPhase::TERMINATED            > ShutdownPhase::MANIFEST_WRITTEN);
}

// ─── §4 Healthcheck Constants ─────────────────────────────────────────────────

TEST_CASE("Healthcheck interval is 5 seconds", "[healthcheck][constants]") {
    REQUIRE(HEALTHCHECK_INTERVAL_S == 5);
}

TEST_CASE("Healthcheck timeout is 2 seconds", "[healthcheck][constants]") {
    REQUIRE(HEALTHCHECK_TIMEOUT_S == 2);
}

TEST_CASE("Healthcheck retries is 5", "[healthcheck][constants]") {
    REQUIRE(HEALTHCHECK_RETRIES == 5);
}

TEST_CASE("Healthcheck max wait is interval times retries (25s)", "[healthcheck][constants]") {
    REQUIRE(HEALTHCHECK_MAX_WAIT_S == 25);
    REQUIRE(HEALTHCHECK_MAX_WAIT_S == HEALTHCHECK_INTERVAL_S * HEALTHCHECK_RETRIES);
}

// ─── §5 Startup Timing Constants ──────────────────────────────────────────────

TEST_CASE("Spine startup min is 1s", "[startup][constants]") {
    REQUIRE(SPINE_STARTUP_MIN_S == 1);
}

TEST_CASE("Spine startup max is 2s", "[startup][constants]") {
    REQUIRE(SPINE_STARTUP_MAX_S == 2);
}

TEST_CASE("Physics startup min is 3s", "[startup][constants]") {
    REQUIRE(PHYSICS_STARTUP_MIN_S == 3);
}

TEST_CASE("Physics startup max is 5s", "[startup][constants]") {
    REQUIRE(PHYSICS_STARTUP_MAX_S == 5);
}

TEST_CASE("Full cluster startup budget is 30s", "[startup][constants]") {
    REQUIRE(FULL_CLUSTER_STARTUP_S == 30);
}

TEST_CASE("Spine starts before physics", "[startup][constants]") {
    REQUIRE(SPINE_STARTUP_MAX_S < PHYSICS_STARTUP_MIN_S);
}

// ─── §6 Shutdown Timing Constants ─────────────────────────────────────────────

TEST_CASE("Stop grace period is 60 seconds", "[shutdown][constants]") {
    REQUIRE(STOP_GRACE_PERIOD_S == 60);
}

TEST_CASE("Graceful shutdown min is 10s", "[shutdown][constants]") {
    REQUIRE(GRACEFUL_SHUTDOWN_MIN_S == 10);
}

TEST_CASE("Graceful shutdown max equals grace period", "[shutdown][constants]") {
    REQUIRE(GRACEFUL_SHUTDOWN_MAX_S == STOP_GRACE_PERIOD_S);
}

TEST_CASE("Shutdown sequence steps is 8", "[shutdown][constants]") {
    REQUIRE(SHUTDOWN_SEQUENCE_STEPS == 8);
}

// ─── §7 Resource Limit Constants ──────────────────────────────────────────────

TEST_CASE("Stack ulimit is 64 MB (67108864 bytes)", "[resource][constants]") {
    REQUIRE(STACK_ULIMIT_BYTES == 67'108'864L);
    REQUIRE(STACK_ULIMIT_BYTES == 64L * 1024L * 1024L);
}

TEST_CASE("Memlock unlimited sentinel is -1", "[resource][constants]") {
    REQUIRE(MEMLOCK_UNLIMITED == -1);
}

TEST_CASE("Spine CPU limit is 2.0 cores", "[resource][constants]") {
    REQUIRE(SPINE_CPU_LIMIT == 2.0f);
}

TEST_CASE("Spine memory limit is 4 GB", "[resource][constants]") {
    REQUIRE(SPINE_MEMORY_LIMIT_GB == 4);
}

TEST_CASE("GPU device count is 1", "[resource][constants]") {
    REQUIRE(GPU_DEVICE_COUNT == 1);
}

TEST_CASE("OMP thread count is 16 for AVX-512 sections", "[resource][constants]") {
    REQUIRE(OMP_NUM_THREADS == 16);
}

// ─── §8 service_layer() Mapping ───────────────────────────────────────────────

TEST_CASE("SPINE is in CORE layer", "[service_layer]") {
    REQUIRE(service_layer(NikolaService::SPINE) == ServiceLayer::CORE);
}

TEST_CASE("PHYSICS is in PHYSICS layer", "[service_layer]") {
    REQUIRE(service_layer(NikolaService::PHYSICS) == ServiceLayer::PHYSICS);
}

TEST_CASE("MEMORY is in COGNITION layer", "[service_layer]") {
    REQUIRE(service_layer(NikolaService::MEMORY) == ServiceLayer::COGNITION);
}

TEST_CASE("ORCHESTRATOR is in COGNITION layer", "[service_layer]") {
    REQUIRE(service_layer(NikolaService::ORCHESTRATOR) == ServiceLayer::COGNITION);
}

TEST_CASE("EXECUTOR is in TOOLS layer", "[service_layer]") {
    REQUIRE(service_layer(NikolaService::EXECUTOR) == ServiceLayer::TOOLS);
}

TEST_CASE("WEB is in TOOLS layer", "[service_layer]") {
    REQUIRE(service_layer(NikolaService::WEB) == ServiceLayer::TOOLS);
}

TEST_CASE("MEMORY and ORCHESTRATOR share COGNITION layer", "[service_layer]") {
    REQUIRE(service_layer(NikolaService::MEMORY) == service_layer(NikolaService::ORCHESTRATOR));
}

TEST_CASE("EXECUTOR and WEB share TOOLS layer", "[service_layer]") {
    REQUIRE(service_layer(NikolaService::EXECUTOR) == service_layer(NikolaService::WEB));
}

// ─── §9 requires_healthy_dependency() ────────────────────────────────────────

TEST_CASE("SPINE requires healthy dependency condition", "[requires_healthy]") {
    REQUIRE(requires_healthy_dependency(NikolaService::SPINE));
}

TEST_CASE("PHYSICS does not require healthy dependency (uses service_started)", "[requires_healthy]") {
    REQUIRE_FALSE(requires_healthy_dependency(NikolaService::PHYSICS));
}

TEST_CASE("MEMORY does not require healthy dependency", "[requires_healthy]") {
    REQUIRE_FALSE(requires_healthy_dependency(NikolaService::MEMORY));
}

TEST_CASE("ORCHESTRATOR does not require healthy dependency", "[requires_healthy]") {
    REQUIRE_FALSE(requires_healthy_dependency(NikolaService::ORCHESTRATOR));
}

TEST_CASE("EXECUTOR does not require healthy dependency", "[requires_healthy]") {
    REQUIRE_FALSE(requires_healthy_dependency(NikolaService::EXECUTOR));
}

TEST_CASE("WEB does not require healthy dependency", "[requires_healthy]") {
    REQUIRE_FALSE(requires_healthy_dependency(NikolaService::WEB));
}

// ─── §10 requires_gpu() ───────────────────────────────────────────────────────

TEST_CASE("PHYSICS requires GPU runtime", "[requires_gpu]") {
    REQUIRE(requires_gpu(NikolaService::PHYSICS));
}

TEST_CASE("SPINE does not require GPU", "[requires_gpu]") {
    REQUIRE_FALSE(requires_gpu(NikolaService::SPINE));
}

TEST_CASE("MEMORY does not require GPU", "[requires_gpu]") {
    REQUIRE_FALSE(requires_gpu(NikolaService::MEMORY));
}

TEST_CASE("ORCHESTRATOR does not require GPU", "[requires_gpu]") {
    REQUIRE_FALSE(requires_gpu(NikolaService::ORCHESTRATOR));
}

TEST_CASE("EXECUTOR does not require GPU", "[requires_gpu]") {
    REQUIRE_FALSE(requires_gpu(NikolaService::EXECUTOR));
}

TEST_CASE("WEB does not require GPU", "[requires_gpu]") {
    REQUIRE_FALSE(requires_gpu(NikolaService::WEB));
}

// ─── §11 requires_privileged() ────────────────────────────────────────────────

TEST_CASE("EXECUTOR requires privileged mode for KVM access", "[privileged]") {
    REQUIRE(requires_privileged(NikolaService::EXECUTOR));
}

TEST_CASE("SPINE does not require privileged mode", "[privileged]") {
    REQUIRE_FALSE(requires_privileged(NikolaService::SPINE));
}

TEST_CASE("PHYSICS does not require privileged mode", "[privileged]") {
    REQUIRE_FALSE(requires_privileged(NikolaService::PHYSICS));
}

TEST_CASE("MEMORY does not require privileged mode", "[privileged]") {
    REQUIRE_FALSE(requires_privileged(NikolaService::MEMORY));
}

TEST_CASE("ORCHESTRATOR does not require privileged mode", "[privileged]") {
    REQUIRE_FALSE(requires_privileged(NikolaService::ORCHESTRATOR));
}

TEST_CASE("WEB does not require privileged mode", "[privileged]") {
    REQUIRE_FALSE(requires_privileged(NikolaService::WEB));
}

// ─── §12 holds_persistent_state() ────────────────────────────────────────────

TEST_CASE("MEMORY holds persistent state (LSM-DMC)", "[persistent]") {
    REQUIRE(holds_persistent_state(NikolaService::MEMORY));
}

TEST_CASE("SPINE does not hold persistent state", "[persistent]") {
    REQUIRE_FALSE(holds_persistent_state(NikolaService::SPINE));
}

TEST_CASE("PHYSICS does not hold persistent state", "[persistent]") {
    REQUIRE_FALSE(holds_persistent_state(NikolaService::PHYSICS));
}

TEST_CASE("ORCHESTRATOR does not hold persistent state", "[persistent]") {
    REQUIRE_FALSE(holds_persistent_state(NikolaService::ORCHESTRATOR));
}

TEST_CASE("EXECUTOR does not hold persistent state", "[persistent]") {
    REQUIRE_FALSE(holds_persistent_state(NikolaService::EXECUTOR));
}

TEST_CASE("WEB does not hold persistent state", "[persistent]") {
    REQUIRE_FALSE(holds_persistent_state(NikolaService::WEB));
}

// ─── §13 uses_curvezmq() ──────────────────────────────────────────────────────

TEST_CASE("SPINE uses CurveZMQ Ironhouse", "[curvezmq]") {
    REQUIRE(uses_curvezmq(NikolaService::SPINE));
}

TEST_CASE("PHYSICS uses CurveZMQ", "[curvezmq]") {
    REQUIRE(uses_curvezmq(NikolaService::PHYSICS));
}

TEST_CASE("MEMORY uses CurveZMQ", "[curvezmq]") {
    REQUIRE(uses_curvezmq(NikolaService::MEMORY));
}

TEST_CASE("ORCHESTRATOR uses CurveZMQ", "[curvezmq]") {
    REQUIRE(uses_curvezmq(NikolaService::ORCHESTRATOR));
}

TEST_CASE("EXECUTOR uses CurveZMQ", "[curvezmq]") {
    REQUIRE(uses_curvezmq(NikolaService::EXECUTOR));
}

TEST_CASE("WEB does not use CurveZMQ", "[curvezmq]") {
    REQUIRE_FALSE(uses_curvezmq(NikolaService::WEB));
}

// ─── §14 startup_deadline_s() ─────────────────────────────────────────────────

TEST_CASE("Spine deadline is SPINE_STARTUP_MAX_S (2s)", "[deadline]") {
    REQUIRE(startup_deadline_s(NikolaService::SPINE) == SPINE_STARTUP_MAX_S);
    REQUIRE(startup_deadline_s(NikolaService::SPINE) == 2);
}

TEST_CASE("Physics deadline is PHYSICS_STARTUP_MAX_S (5s)", "[deadline]") {
    REQUIRE(startup_deadline_s(NikolaService::PHYSICS) == PHYSICS_STARTUP_MAX_S);
    REQUIRE(startup_deadline_s(NikolaService::PHYSICS) == 5);
}

TEST_CASE("MEMORY deadline falls through to full cluster budget (30s)", "[deadline]") {
    REQUIRE(startup_deadline_s(NikolaService::MEMORY) == FULL_CLUSTER_STARTUP_S);
}

TEST_CASE("ORCHESTRATOR deadline is full cluster budget (30s)", "[deadline]") {
    REQUIRE(startup_deadline_s(NikolaService::ORCHESTRATOR) == FULL_CLUSTER_STARTUP_S);
}

TEST_CASE("EXECUTOR deadline is full cluster budget (30s)", "[deadline]") {
    REQUIRE(startup_deadline_s(NikolaService::EXECUTOR) == FULL_CLUSTER_STARTUP_S);
}

TEST_CASE("Spine deadline is tighter than physics deadline", "[deadline]") {
    REQUIRE(startup_deadline_s(NikolaService::SPINE) < startup_deadline_s(NikolaService::PHYSICS));
}

// ─── §15 startup_within_budget() ──────────────────────────────────────────────

TEST_CASE("Spine at 1s is within budget", "[startup_budget]") {
    REQUIRE(startup_within_budget(NikolaService::SPINE, 1));
}

TEST_CASE("Spine at 2s is exactly at budget limit", "[startup_budget]") {
    REQUIRE(startup_within_budget(NikolaService::SPINE, 2));
}

TEST_CASE("Spine at 3s exceeds budget", "[startup_budget]") {
    REQUIRE_FALSE(startup_within_budget(NikolaService::SPINE, 3));
}

TEST_CASE("Physics at 5s is exactly at budget limit", "[startup_budget]") {
    REQUIRE(startup_within_budget(NikolaService::PHYSICS, 5));
}

TEST_CASE("Physics at 6s exceeds budget", "[startup_budget]") {
    REQUIRE_FALSE(startup_within_budget(NikolaService::PHYSICS, 6));
}

TEST_CASE("Memory at 30s is within budget (full cluster window)", "[startup_budget]") {
    REQUIRE(startup_within_budget(NikolaService::MEMORY, 30));
}

TEST_CASE("Memory at 31s exceeds budget", "[startup_budget]") {
    REQUIRE_FALSE(startup_within_budget(NikolaService::MEMORY, 31));
}

// ─── §16 healthcheck_within_budget() ─────────────────────────────────────────

TEST_CASE("0 seconds elapsed is within healthcheck budget", "[healthcheck_budget]") {
    REQUIRE(healthcheck_within_budget(0));
}

TEST_CASE("25 seconds elapsed is exactly at healthcheck budget (5×5)", "[healthcheck_budget]") {
    REQUIRE(healthcheck_within_budget(25));
}

TEST_CASE("26 seconds elapsed exceeds healthcheck budget", "[healthcheck_budget]") {
    REQUIRE_FALSE(healthcheck_within_budget(26));
}

TEST_CASE("10 seconds elapsed is within healthcheck budget", "[healthcheck_budget]") {
    REQUIRE(healthcheck_within_budget(10));
}

// ─── §17 shutdown_complete() ──────────────────────────────────────────────────

TEST_CASE("TERMINATED phase is shutdown complete", "[shutdown_complete]") {
    REQUIRE(shutdown_complete(ShutdownPhase::TERMINATED));
}

TEST_CASE("MANIFEST_WRITTEN is not yet complete", "[shutdown_complete]") {
    REQUIRE_FALSE(shutdown_complete(ShutdownPhase::MANIFEST_WRITTEN));
}

TEST_CASE("RUNNING is not shutdown complete", "[shutdown_complete]") {
    REQUIRE_FALSE(shutdown_complete(ShutdownPhase::RUNNING));
}

TEST_CASE("HALT_BROADCAST is not shutdown complete", "[shutdown_complete]") {
    REQUIRE_FALSE(shutdown_complete(ShutdownPhase::HALT_BROADCAST));
}

// ─── §18 memory_safe_to_kill() ────────────────────────────────────────────────

TEST_CASE("MANIFEST_WRITTEN: memory safe to kill", "[memory_safe]") {
    REQUIRE(memory_safe_to_kill(ShutdownPhase::MANIFEST_WRITTEN));
}

TEST_CASE("TERMINATED: memory safe to kill", "[memory_safe]") {
    REQUIRE(memory_safe_to_kill(ShutdownPhase::TERMINATED));
}

TEST_CASE("WAL_FSYNCED: not yet safe to kill (MANIFEST not written)", "[memory_safe]") {
    REQUIRE_FALSE(memory_safe_to_kill(ShutdownPhase::WAL_FSYNCED));
}

TEST_CASE("MEMTABLE_FLUSHED: not safe to kill", "[memory_safe]") {
    REQUIRE_FALSE(memory_safe_to_kill(ShutdownPhase::MEMTABLE_FLUSHED));
}

TEST_CASE("RUNNING: not safe to kill", "[memory_safe]") {
    REQUIRE_FALSE(memory_safe_to_kill(ShutdownPhase::RUNNING));
}

TEST_CASE("PHYSICS_TICK_COMPLETE: memory not yet safe to kill", "[memory_safe]") {
    REQUIRE_FALSE(memory_safe_to_kill(ShutdownPhase::PHYSICS_TICK_COMPLETE));
}

// ─── §19 wal_is_durable() ────────────────────────────────────────────────────

TEST_CASE("WAL_FSYNCED: WAL is durable", "[wal_durable]") {
    REQUIRE(wal_is_durable(ShutdownPhase::WAL_FSYNCED));
}

TEST_CASE("MANIFEST_WRITTEN: WAL is durable", "[wal_durable]") {
    REQUIRE(wal_is_durable(ShutdownPhase::MANIFEST_WRITTEN));
}

TEST_CASE("TERMINATED: WAL is durable", "[wal_durable]") {
    REQUIRE(wal_is_durable(ShutdownPhase::TERMINATED));
}

TEST_CASE("MEMTABLE_FLUSHED: WAL not yet durable", "[wal_durable]") {
    REQUIRE_FALSE(wal_is_durable(ShutdownPhase::MEMTABLE_FLUSHED));
}

TEST_CASE("RUNNING: WAL not durable", "[wal_durable]") {
    REQUIRE_FALSE(wal_is_durable(ShutdownPhase::RUNNING));
}

// ─── §20 physics_state_serialized() ──────────────────────────────────────────

TEST_CASE("PHYSICS_TICK_COMPLETE: physics state serialized", "[physics_state]") {
    REQUIRE(physics_state_serialized(ShutdownPhase::PHYSICS_TICK_COMPLETE));
}

TEST_CASE("MEMORY_LOCK_ACQUIRED: physics state serialized (later phase)", "[physics_state]") {
    REQUIRE(physics_state_serialized(ShutdownPhase::MEMORY_LOCK_ACQUIRED));
}

TEST_CASE("TERMINATED: physics state serialized", "[physics_state]") {
    REQUIRE(physics_state_serialized(ShutdownPhase::TERMINATED));
}

TEST_CASE("HALT_BROADCAST: physics tick not yet complete", "[physics_state]") {
    REQUIRE_FALSE(physics_state_serialized(ShutdownPhase::HALT_BROADCAST));
}

TEST_CASE("RUNNING: physics state not serialized", "[physics_state]") {
    REQUIRE_FALSE(physics_state_serialized(ShutdownPhase::RUNNING));
}

// ─── §21 shutdown_within_grace() and shutdown_grace_expired() ────────────────

TEST_CASE("0 seconds is within grace period", "[grace_period]") {
    REQUIRE(shutdown_within_grace(0));
}

TEST_CASE("60 seconds is exactly at grace period boundary", "[grace_period]") {
    REQUIRE(shutdown_within_grace(60));
}

TEST_CASE("61 seconds exceeds grace period", "[grace_period]") {
    REQUIRE_FALSE(shutdown_within_grace(61));
}

TEST_CASE("Grace expired at 61 seconds", "[grace_period]") {
    REQUIRE(shutdown_grace_expired(61));
}

TEST_CASE("Grace not expired at 60 seconds", "[grace_period]") {
    REQUIRE_FALSE(shutdown_grace_expired(60));
}

// ─── §22 Resource Predicates ──────────────────────────────────────────────────

TEST_CASE("Stack at exactly 64 MB is sufficient", "[stack_sufficient]") {
    REQUIRE(stack_is_sufficient(64L * 1024L * 1024L));
}

TEST_CASE("Stack at STACK_ULIMIT_BYTES is sufficient", "[stack_sufficient]") {
    REQUIRE(stack_is_sufficient(STACK_ULIMIT_BYTES));
}

TEST_CASE("Stack above 64 MB is sufficient", "[stack_sufficient]") {
    REQUIRE(stack_is_sufficient(128L * 1024L * 1024L));
}

TEST_CASE("Stack below 64 MB is insufficient", "[stack_sufficient]") {
    REQUIRE_FALSE(stack_is_sufficient(32L * 1024L * 1024L));
}

TEST_CASE("OMP threads at 16 is at spec", "[omp_threads]") {
    REQUIRE(omp_threads_at_spec(16));
}

TEST_CASE("OMP threads at 15 is not at spec", "[omp_threads]") {
    REQUIRE_FALSE(omp_threads_at_spec(15));
}

TEST_CASE("OMP threads at 32 is not at spec", "[omp_threads]") {
    REQUIRE_FALSE(omp_threads_at_spec(32));
}

TEST_CASE("Spine CPU 2.0 is sufficient", "[spine_cpu]") {
    REQUIRE(spine_cpu_is_sufficient(2.0f));
}

TEST_CASE("Spine CPU 4.0 is sufficient (exceeds minimum)", "[spine_cpu]") {
    REQUIRE(spine_cpu_is_sufficient(4.0f));
}

TEST_CASE("Spine CPU 1.9 is insufficient", "[spine_cpu]") {
    REQUIRE_FALSE(spine_cpu_is_sufficient(1.9f));
}

// ─── §23 Label Functions ──────────────────────────────────────────────────────

TEST_CASE("service_layer_name CORE", "[labels]") {
    REQUIRE(std::string(service_layer_name(ServiceLayer::CORE)) == "CORE");
}

TEST_CASE("service_layer_name PHYSICS", "[labels]") {
    REQUIRE(std::string(service_layer_name(ServiceLayer::PHYSICS)) == "PHYSICS");
}

TEST_CASE("service_layer_name COGNITION", "[labels]") {
    REQUIRE(std::string(service_layer_name(ServiceLayer::COGNITION)) == "COGNITION");
}

TEST_CASE("service_layer_name TOOLS", "[labels]") {
    REQUIRE(std::string(service_layer_name(ServiceLayer::TOOLS)) == "TOOLS");
}

TEST_CASE("service_name SPINE", "[labels]") {
    REQUIRE(std::string(service_name(NikolaService::SPINE)) == "nikola-spine");
}

TEST_CASE("service_name PHYSICS", "[labels]") {
    REQUIRE(std::string(service_name(NikolaService::PHYSICS)) == "nikola-physics");
}

TEST_CASE("service_name MEMORY", "[labels]") {
    REQUIRE(std::string(service_name(NikolaService::MEMORY)) == "nikola-memory");
}

TEST_CASE("service_name ORCHESTRATOR", "[labels]") {
    REQUIRE(std::string(service_name(NikolaService::ORCHESTRATOR)) == "nikola-orchestrator");
}

TEST_CASE("service_name EXECUTOR", "[labels]") {
    REQUIRE(std::string(service_name(NikolaService::EXECUTOR)) == "nikola-executor");
}

TEST_CASE("service_name WEB", "[labels]") {
    REQUIRE(std::string(service_name(NikolaService::WEB)) == "nikola-web");
}

TEST_CASE("service_state_name HEALTHY", "[labels]") {
    REQUIRE(std::string(service_state_name(ServiceState::HEALTHY)) == "HEALTHY");
}

TEST_CASE("service_state_name CRASHED", "[labels]") {
    REQUIRE(std::string(service_state_name(ServiceState::CRASHED)) == "CRASHED");
}

TEST_CASE("service_state_name PENDING", "[labels]") {
    REQUIRE(std::string(service_state_name(ServiceState::PENDING)) == "PENDING");
}

TEST_CASE("shutdown_phase_name RUNNING", "[labels]") {
    REQUIRE(std::string(shutdown_phase_name(ShutdownPhase::RUNNING)) == "RUNNING");
}

TEST_CASE("shutdown_phase_name HALT_BROADCAST", "[labels]") {
    REQUIRE(std::string(shutdown_phase_name(ShutdownPhase::HALT_BROADCAST)) == "HALT_BROADCAST");
}

TEST_CASE("shutdown_phase_name WAL_FSYNCED", "[labels]") {
    REQUIRE(std::string(shutdown_phase_name(ShutdownPhase::WAL_FSYNCED)) == "WAL_FSYNCED");
}

TEST_CASE("shutdown_phase_name MANIFEST_WRITTEN", "[labels]") {
    REQUIRE(std::string(shutdown_phase_name(ShutdownPhase::MANIFEST_WRITTEN)) == "MANIFEST_WRITTEN");
}

TEST_CASE("shutdown_phase_name TERMINATED", "[labels]") {
    REQUIRE(std::string(shutdown_phase_name(ShutdownPhase::TERMINATED)) == "TERMINATED");
}

// ─── §24 Integration Scenarios ────────────────────────────────────────────────

TEST_CASE("Ironhouse pattern: only spine has healthy dependency condition", "[integration]") {
    // All services that depend transitively on spine, but only spine itself
    // uses service_healthy instead of service_started for dependency checks
    int healthy_dep_count = 0;
    for (auto svc : {NikolaService::SPINE, NikolaService::PHYSICS, NikolaService::MEMORY,
                     NikolaService::ORCHESTRATOR, NikolaService::EXECUTOR, NikolaService::WEB}) {
        if (requires_healthy_dependency(svc)) ++healthy_dep_count;
    }
    REQUIRE(healthy_dep_count == 1);
}

TEST_CASE("Only one service requires GPU runtime", "[integration]") {
    int gpu_count = 0;
    for (auto svc : {NikolaService::SPINE, NikolaService::PHYSICS, NikolaService::MEMORY,
                     NikolaService::ORCHESTRATOR, NikolaService::EXECUTOR, NikolaService::WEB}) {
        if (requires_gpu(svc)) ++gpu_count;
    }
    REQUIRE(gpu_count == GPU_DEVICE_COUNT);
}

TEST_CASE("Only one service requires privileged mode", "[integration]") {
    int priv_count = 0;
    for (auto svc : {NikolaService::SPINE, NikolaService::PHYSICS, NikolaService::MEMORY,
                     NikolaService::ORCHESTRATOR, NikolaService::EXECUTOR, NikolaService::WEB}) {
        if (requires_privileged(svc)) ++priv_count;
    }
    REQUIRE(priv_count == 1);
}

TEST_CASE("Only one service holds persistent state", "[integration]") {
    int pers_count = 0;
    for (auto svc : {NikolaService::SPINE, NikolaService::PHYSICS, NikolaService::MEMORY,
                     NikolaService::ORCHESTRATOR, NikolaService::EXECUTOR, NikolaService::WEB}) {
        if (holds_persistent_state(svc)) ++pers_count;
    }
    REQUIRE(pers_count == 1);
}

TEST_CASE("All services except WEB use CurveZMQ", "[integration]") {
    int zmq_count = 0;
    for (auto svc : {NikolaService::SPINE, NikolaService::PHYSICS, NikolaService::MEMORY,
                     NikolaService::ORCHESTRATOR, NikolaService::EXECUTOR, NikolaService::WEB}) {
        if (uses_curvezmq(svc)) ++zmq_count;
    }
    REQUIRE(zmq_count == 5);
}

TEST_CASE("Shutdown Scenario: physics safe but memory not at PHYSICS_TICK_COMPLETE", "[integration]") {
    auto phase = ShutdownPhase::PHYSICS_TICK_COMPLETE;
    REQUIRE(physics_state_serialized(phase));
    REQUIRE_FALSE(wal_is_durable(phase));
    REQUIRE_FALSE(memory_safe_to_kill(phase));
    REQUIRE_FALSE(shutdown_complete(phase));
}

TEST_CASE("Shutdown Scenario: WAL durable but memory not safe at WAL_FSYNCED", "[integration]") {
    auto phase = ShutdownPhase::WAL_FSYNCED;
    REQUIRE(wal_is_durable(phase));
    REQUIRE_FALSE(memory_safe_to_kill(phase));  // MANIFEST not yet written
    REQUIRE_FALSE(shutdown_complete(phase));
}

TEST_CASE("Shutdown Scenario: fully safe after MANIFEST_WRITTEN", "[integration]") {
    auto phase = ShutdownPhase::MANIFEST_WRITTEN;
    REQUIRE(physics_state_serialized(phase));
    REQUIRE(wal_is_durable(phase));
    REQUIRE(memory_safe_to_kill(phase));
    REQUIRE_FALSE(shutdown_complete(phase));  // Process hasn't exited yet
}

TEST_CASE("Shutdown Scenario: complete at TERMINATED, all invariants hold", "[integration]") {
    auto phase = ShutdownPhase::TERMINATED;
    REQUIRE(physics_state_serialized(phase));
    REQUIRE(wal_is_durable(phase));
    REQUIRE(memory_safe_to_kill(phase));
    REQUIRE(shutdown_complete(phase));
}

TEST_CASE("Startup Scenario: valid 2s spine + 5s physics + under 30s cluster", "[integration]") {
    REQUIRE(startup_within_budget(NikolaService::SPINE,   2));
    REQUIRE(startup_within_budget(NikolaService::PHYSICS, 5));
    REQUIRE(startup_within_budget(NikolaService::MEMORY,  28));
    // Total observed < 30s
    REQUIRE(25 < FULL_CLUSTER_STARTUP_S);
}

TEST_CASE("Healthcheck budget covers full 5-retry window exactly", "[integration]") {
    REQUIRE(healthcheck_within_budget(HEALTHCHECK_MAX_WAIT_S));
    REQUIRE_FALSE(healthcheck_within_budget(HEALTHCHECK_MAX_WAIT_S + 1));
}

TEST_CASE("Resource profile: default stack and OMP threads at spec", "[integration]") {
    REQUIRE(stack_is_sufficient(STACK_ULIMIT_BYTES));
    REQUIRE(omp_threads_at_spec(OMP_NUM_THREADS));
    REQUIRE(spine_cpu_is_sufficient(SPINE_CPU_LIMIT));
}
