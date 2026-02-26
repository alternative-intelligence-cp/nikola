// =============================================================================
// tests/unit/phase93_service_orchestration_test.cpp
// Phase 93 — GAP-026: Docker Compose Service Orchestration Policy
//
// Tests for nikola::system::service_orchestration.hpp
// Spec: docs/info/integration/sections/04_infrastructure/02_orchestrator_router.md
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "nikola/system/service_orchestration.hpp"

using namespace nikola::system;

// ---------------------------------------------------------------------------
// § Enums
// ---------------------------------------------------------------------------

TEST_CASE("ServiceLayer enum values are distinct and ordered", "[enums][phase93]") {
    CHECK(static_cast<int>(ServiceLayer::LAYER_0_CORE)      == 0);
    CHECK(static_cast<int>(ServiceLayer::LAYER_1_PHYSICS)   == 1);
    CHECK(static_cast<int>(ServiceLayer::LAYER_2_COGNITION) == 2);
    CHECK(static_cast<int>(ServiceLayer::LAYER_3_TOOLS)     == 3);
}

TEST_CASE("ServiceId enum values are distinct", "[enums][phase93]") {
    CHECK(static_cast<int>(ServiceId::SPINE)        == 0);
    CHECK(static_cast<int>(ServiceId::PHYSICS)      == 1);
    CHECK(static_cast<int>(ServiceId::MEMORY)       == 2);
    CHECK(static_cast<int>(ServiceId::ORCHESTRATOR) == 3);
    CHECK(static_cast<int>(ServiceId::EXECUTOR)     == 4);
    CHECK(static_cast<int>(ServiceId::WEB)          == 5);
}

// ---------------------------------------------------------------------------
// § Healthcheck constants (nikola-spine)
// ---------------------------------------------------------------------------

TEST_CASE("SPINE_HEALTHCHECK_INTERVAL_S is 5", "[constants][phase93]") {
    CHECK(SPINE_HEALTHCHECK_INTERVAL_S == 5u);
}

TEST_CASE("SPINE_HEALTHCHECK_TIMEOUT_S is 2", "[constants][phase93]") {
    CHECK(SPINE_HEALTHCHECK_TIMEOUT_S == 2u);
}

TEST_CASE("SPINE_HEALTHCHECK_RETRIES is 5", "[constants][phase93]") {
    CHECK(SPINE_HEALTHCHECK_RETRIES == 5u);
}

TEST_CASE("SPINE_HEALTHCHECK_START_WINDOW_S equals retries*(interval+timeout)", "[constants][phase93]") {
    uint32_t expected = SPINE_HEALTHCHECK_RETRIES *
        (SPINE_HEALTHCHECK_INTERVAL_S + SPINE_HEALTHCHECK_TIMEOUT_S);
    CHECK(SPINE_HEALTHCHECK_START_WINDOW_S == expected);  // 35 s
}

// ---------------------------------------------------------------------------
// § Resource limits
// ---------------------------------------------------------------------------

TEST_CASE("SPINE_MEMORY_LIMIT_BYTES is 4 GiB", "[constants][phase93]") {
    CHECK(SPINE_MEMORY_LIMIT_BYTES == 4ULL * 1024 * 1024 * 1024);
}

TEST_CASE("PHYSICS_OMP_NUM_THREADS is 16", "[constants][phase93]") {
    CHECK(PHYSICS_OMP_NUM_THREADS == 16u);
}

TEST_CASE("PHYSICS_STACK_SIZE_BYTES is 64 MiB", "[constants][phase93]") {
    CHECK(PHYSICS_STACK_SIZE_BYTES == 64ULL * 1024 * 1024);
}

// ---------------------------------------------------------------------------
// § Shutdown policy
// ---------------------------------------------------------------------------

TEST_CASE("MEMORY_STOP_SIGNAL is SIGTERM", "[constants][phase93]") {
    CHECK(MEMORY_STOP_SIGNAL == "SIGTERM");
}

TEST_CASE("MEMORY_STOP_GRACE_PERIOD_S is 60", "[constants][phase93]") {
    CHECK(MEMORY_STOP_GRACE_PERIOD_S == 60u);
}

// ---------------------------------------------------------------------------
// § Security / IPC paths
// ---------------------------------------------------------------------------

TEST_CASE("ZMQ_SECURITY_MODEL is CurveZMQ", "[constants][phase93]") {
    CHECK(ZMQ_SECURITY_MODEL == "CurveZMQ");
}

TEST_CASE("IPC_SOCKET_HOST_PATH matches spec", "[constants][phase93]") {
    CHECK(IPC_SOCKET_HOST_PATH == "/tmp/nikola/ipc");
}

TEST_CASE("CURVZMQ_KEYS_HOST_PATH matches spec", "[constants][phase93]") {
    CHECK(CURVZMQ_KEYS_HOST_PATH == "/etc/nikola/keys");
}

// ---------------------------------------------------------------------------
// § Container name constants
// ---------------------------------------------------------------------------

TEST_CASE("Container names match specification", "[constants][phase93]") {
    CHECK(CONTAINER_SPINE        == "nikola-spine");
    CHECK(CONTAINER_PHYSICS      == "nikola-physics");
    CHECK(CONTAINER_MEMORY       == "nikola-memory");
    CHECK(CONTAINER_ORCHESTRATOR == "nikola-orchestrator");
    CHECK(CONTAINER_EXECUTOR     == "nikola-executor");
    CHECK(CONTAINER_WEB          == "nikola-web");
}

// ---------------------------------------------------------------------------
// § spine_healthy
// ---------------------------------------------------------------------------

TEST_CASE("spine_healthy requires SPINE_HEALTHCHECK_RETRIES consecutive passes", "[functions][phase93]") {
    CHECK(spine_healthy(4) == false);
    CHECK(spine_healthy(5) == true);
    CHECK(spine_healthy(9) == true);
}

// ---------------------------------------------------------------------------
// § memory_within_grace_period
// ---------------------------------------------------------------------------

TEST_CASE("memory_within_grace_period allows <= 60 seconds", "[functions][phase93]") {
    CHECK(memory_within_grace_period(0)  == true);
    CHECK(memory_within_grace_period(60) == true);
    CHECK(memory_within_grace_period(61) == false);
}

// ---------------------------------------------------------------------------
// § layer_of
// ---------------------------------------------------------------------------

TEST_CASE("layer_of maps services to correct layers", "[functions][phase93]") {
    CHECK(layer_of(ServiceId::SPINE)        == ServiceLayer::LAYER_0_CORE);
    CHECK(layer_of(ServiceId::PHYSICS)      == ServiceLayer::LAYER_1_PHYSICS);
    CHECK(layer_of(ServiceId::MEMORY)       == ServiceLayer::LAYER_2_COGNITION);
    CHECK(layer_of(ServiceId::ORCHESTRATOR) == ServiceLayer::LAYER_2_COGNITION);
    CHECK(layer_of(ServiceId::EXECUTOR)     == ServiceLayer::LAYER_3_TOOLS);
    CHECK(layer_of(ServiceId::WEB)          == ServiceLayer::LAYER_3_TOOLS);
}

// ---------------------------------------------------------------------------
// § Label helpers
// ---------------------------------------------------------------------------

TEST_CASE("service_layer_label returns correct strings", "[labels][phase93]") {
    CHECK(service_layer_label(ServiceLayer::LAYER_0_CORE)      == "Layer 0 (Core)");
    CHECK(service_layer_label(ServiceLayer::LAYER_1_PHYSICS)   == "Layer 1 (Physics)");
    CHECK(service_layer_label(ServiceLayer::LAYER_2_COGNITION) == "Layer 2 (Cognition & Memory)");
    CHECK(service_layer_label(ServiceLayer::LAYER_3_TOOLS)     == "Layer 3 (Tools & Interface)");
}

TEST_CASE("service_id_label returns correct container names", "[labels][phase93]") {
    CHECK(service_id_label(ServiceId::SPINE)        == "nikola-spine");
    CHECK(service_id_label(ServiceId::PHYSICS)      == "nikola-physics");
    CHECK(service_id_label(ServiceId::MEMORY)       == "nikola-memory");
    CHECK(service_id_label(ServiceId::ORCHESTRATOR) == "nikola-orchestrator");
    CHECK(service_id_label(ServiceId::EXECUTOR)     == "nikola-executor");
    CHECK(service_id_label(ServiceId::WEB)          == "nikola-web");
}
