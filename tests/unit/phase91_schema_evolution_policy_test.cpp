// =============================================================================
// tests/unit/phase91_schema_evolution_policy_test.cpp
// Phase 91 — GAP-023: Protocol Buffer Schema Evolution Strategy
//
// Tests for nikola::system::schema_evolution_policy.hpp
// Spec: docs/info/integration/sections/04_infrastructure/01_zeromq_spine.md
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "nikola/system/schema_evolution_policy.hpp"

using namespace nikola::system;

// ---------------------------------------------------------------------------
// § Enum smoke-tests
// ---------------------------------------------------------------------------

TEST_CASE("SchemaChangeClass enum values are distinct", "[enums][phase91]") {
    CHECK(static_cast<int>(SchemaChangeClass::BREAKING)        == 0);
    CHECK(static_cast<int>(SchemaChangeClass::BACKWARD_COMPAT) == 1);
    CHECK(static_cast<int>(SchemaChangeClass::NON_FUNCTIONAL)  == 2);
}

TEST_CASE("FieldLifecycle enum values are distinct", "[enums][phase91]") {
    CHECK(static_cast<int>(FieldLifecycle::ACTIVE)     == 0);
    CHECK(static_cast<int>(FieldLifecycle::DEPRECATED) == 1);
    CHECK(static_cast<int>(FieldLifecycle::TOMBSTONED) == 2);
}

TEST_CASE("CompatibilityResult enum values are distinct", "[enums][phase91]") {
    CHECK(static_cast<int>(CompatibilityResult::SUCCESS_FULL)     == 0);
    CHECK(static_cast<int>(CompatibilityResult::SUCCESS_FORWARD)  == 1);
    CHECK(static_cast<int>(CompatibilityResult::SUCCESS_BACKWARD) == 2);
    CHECK(static_cast<int>(CompatibilityResult::FAILURE_MAJOR)    == 3);
}

// ---------------------------------------------------------------------------
// § Version constants
// ---------------------------------------------------------------------------

TEST_CASE("SCHEMA_VERSION_MAJOR is 2", "[constants][phase91]") {
    CHECK(SCHEMA_VERSION_MAJOR == 2u);
}

TEST_CASE("SCHEMA_PACKAGE_ROOT is nikola.spine", "[constants][phase91]") {
    CHECK(SCHEMA_PACKAGE_ROOT == "nikola.spine");
}

// ---------------------------------------------------------------------------
// § Morton key migration constants (INT-06)
// ---------------------------------------------------------------------------

TEST_CASE("MORTON_KEY_BYTES_V2 is 16", "[constants][phase91]") {
    CHECK(MORTON_KEY_BYTES_V2 == 16u);
}

TEST_CASE("OBSOLETE_COORDINATES_FIELD_TAG is 1", "[constants][phase91]") {
    CHECK(OBSOLETE_COORDINATES_FIELD_TAG == 1u);
}

TEST_CASE("MORTON_INDICES_FIELD_TAG is 5", "[constants][phase91]") {
    CHECK(MORTON_INDICES_FIELD_TAG == 5u);
}

// ---------------------------------------------------------------------------
// § requires_major_bump
// ---------------------------------------------------------------------------

TEST_CASE("requires_major_bump is true only for BREAKING changes", "[functions][phase91]") {
    CHECK(requires_major_bump(SchemaChangeClass::BREAKING)        == true);
    CHECK(requires_major_bump(SchemaChangeClass::BACKWARD_COMPAT) == false);
    CHECK(requires_major_bump(SchemaChangeClass::NON_FUNCTIONAL)  == false);
}

// ---------------------------------------------------------------------------
// § compatible_without_shim
// ---------------------------------------------------------------------------

TEST_CASE("compatible_without_shim is true only for equal MAJOR versions", "[functions][phase91]") {
    CHECK(compatible_without_shim(2, 2) == true);
    CHECK(compatible_without_shim(1, 2) == false);
    CHECK(compatible_without_shim(2, 1) == false);
}

// ---------------------------------------------------------------------------
// § must_reserve_tombstoned_id
// ---------------------------------------------------------------------------

TEST_CASE("must_reserve_tombstoned_id always returns true", "[functions][phase91]") {
    CHECK(must_reserve_tombstoned_id(1)    == true);
    CHECK(must_reserve_tombstoned_id(999)  == true);
}

// ---------------------------------------------------------------------------
// § compatibility matrix
// ---------------------------------------------------------------------------

TEST_CASE("compatibility matrix — same version yields SUCCESS_FULL", "[functions][phase91]") {
    CHECK(compatibility(2, 2) == CompatibilityResult::SUCCESS_FULL);
    CHECK(compatibility(1, 1) == CompatibilityResult::SUCCESS_FULL);
}

TEST_CASE("compatibility matrix — legacy producer→new consumer is FORWARD", "[functions][phase91]") {
    // producer older than consumer
    CHECK(compatibility(1, 2) == CompatibilityResult::SUCCESS_FORWARD);
}

TEST_CASE("compatibility matrix — newer producer→older consumer (1 major) is BACKWARD", "[functions][phase91]") {
    CHECK(compatibility(2, 1) == CompatibilityResult::SUCCESS_BACKWARD);
}

TEST_CASE("compatibility matrix — large MAJOR gap is FAILURE", "[functions][phase91]") {
    CHECK(compatibility(3, 1) == CompatibilityResult::FAILURE_MAJOR);
    CHECK(compatibility(5, 1) == CompatibilityResult::FAILURE_MAJOR);
}

// ---------------------------------------------------------------------------
// § Label helpers
// ---------------------------------------------------------------------------

TEST_CASE("change_class_label returns non-empty recognized strings", "[labels][phase91]") {
    CHECK(!change_class_label(SchemaChangeClass::BREAKING).empty());
    CHECK(!change_class_label(SchemaChangeClass::BACKWARD_COMPAT).empty());
    CHECK(!change_class_label(SchemaChangeClass::NON_FUNCTIONAL).empty());
}

TEST_CASE("lifecycle_label returns correct strings", "[labels][phase91]") {
    CHECK(lifecycle_label(FieldLifecycle::ACTIVE)     == "active");
    CHECK(lifecycle_label(FieldLifecycle::DEPRECATED) == "deprecated");
    CHECK(lifecycle_label(FieldLifecycle::TOMBSTONED) == "tombstoned");
}

TEST_CASE("compat_result_label returns correct strings", "[labels][phase91]") {
    CHECK(compat_result_label(CompatibilityResult::SUCCESS_FULL)     == "SUCCESS_FULL");
    CHECK(compat_result_label(CompatibilityResult::SUCCESS_FORWARD)  == "SUCCESS_FORWARD");
    CHECK(compat_result_label(CompatibilityResult::SUCCESS_BACKWARD) == "SUCCESS_BACKWARD");
    CHECK(compat_result_label(CompatibilityResult::FAILURE_MAJOR)    == "FAILURE_MAJOR");
}
