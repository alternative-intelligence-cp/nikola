// =============================================================================
// tests/unit/phase87_coordinate_semantics_test.cpp
// Phase 87 — GAP-041: 9D Coordinate Dimension Semantics Glossary
//
// Tests for nikola::math::coordinate_semantics.hpp
// Spec: docs/info/integration/sections/02_foundations/01_9d_toroidal_geometry.md
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <complex>

#include "nikola/math/coordinate_semantics.hpp"

using namespace nikola::math;
using Catch::Approx;

// ---------------------------------------------------------------------------
// § CoordinateDomain enum
// ---------------------------------------------------------------------------

TEST_CASE("CoordinateDomain enum values are distinct", "[enums][phase87]") {
    CHECK(static_cast<int>(CoordinateDomain::SYSTEMIC) == 0);
    CHECK(static_cast<int>(CoordinateDomain::TEMPORAL) == 1);
    CHECK(static_cast<int>(CoordinateDomain::QUANTUM)  == 2);
    CHECK(static_cast<int>(CoordinateDomain::SPATIAL)  == 3);
}

TEST_CASE("domain_label returns correct strings", "[enums][phase87]") {
    CHECK(domain_label(CoordinateDomain::SYSTEMIC) == "systemic");
    CHECK(domain_label(CoordinateDomain::TEMPORAL) == "temporal");
    CHECK(domain_label(CoordinateDomain::QUANTUM)  == "quantum");
    CHECK(domain_label(CoordinateDomain::SPATIAL)  == "spatial");
}

// ---------------------------------------------------------------------------
// § Dimension enum + domain_of
// ---------------------------------------------------------------------------

TEST_CASE("Dimension enum has 9 values starting from 0", "[enums][phase87]") {
    CHECK(static_cast<int>(Dimension::R) == 0);
    CHECK(static_cast<int>(Dimension::S) == 1);
    CHECK(static_cast<int>(Dimension::T) == 2);
    CHECK(static_cast<int>(Dimension::U) == 3);
    CHECK(static_cast<int>(Dimension::V) == 4);
    CHECK(static_cast<int>(Dimension::W) == 5);
    CHECK(static_cast<int>(Dimension::X) == 6);
    CHECK(static_cast<int>(Dimension::Y) == 7);
    CHECK(static_cast<int>(Dimension::Z) == 8);
}

TEST_CASE("domain_of maps dimensions to correct domains", "[functions][phase87]") {
    CHECK(domain_of(Dimension::R) == CoordinateDomain::SYSTEMIC);
    CHECK(domain_of(Dimension::S) == CoordinateDomain::SYSTEMIC);
    CHECK(domain_of(Dimension::T) == CoordinateDomain::TEMPORAL);
    CHECK(domain_of(Dimension::U) == CoordinateDomain::QUANTUM);
    CHECK(domain_of(Dimension::V) == CoordinateDomain::QUANTUM);
    CHECK(domain_of(Dimension::W) == CoordinateDomain::QUANTUM);
    CHECK(domain_of(Dimension::X) == CoordinateDomain::SPATIAL);
    CHECK(domain_of(Dimension::Y) == CoordinateDomain::SPATIAL);
    CHECK(domain_of(Dimension::Z) == CoordinateDomain::SPATIAL);
}

TEST_CASE("is_systemic / is_temporal / is_quantum / is_spatial helpers", "[functions][phase87]") {
    CHECK(is_systemic(Dimension::R) == true);
    CHECK(is_systemic(Dimension::T) == false);
    CHECK(is_temporal(Dimension::T) == true);
    CHECK(is_quantum (Dimension::U) == true);
    CHECK(is_quantum (Dimension::X) == false);
    CHECK(is_spatial (Dimension::Z) == true);
    CHECK(is_spatial (Dimension::R) == false);
}

// ---------------------------------------------------------------------------
// § Resonance (r) constants and predicates
// ---------------------------------------------------------------------------

TEST_CASE("Resonance range constants are [0, 1]", "[constants][phase87]") {
    CHECK(RESONANCE_MIN     == Approx(0.0f));
    CHECK(RESONANCE_MAX     == Approx(1.0f));
    CHECK(RESONANCE_HIGH_Q  == Approx(0.8f));
    CHECK(RESONANCE_LOW_Q   == Approx(0.2f));
    CHECK(RESONANCE_DEFAULT == Approx(0.5f));
}

TEST_CASE("is_high_q_resonance triggers above 0.8", "[functions][phase87]") {
    CHECK(is_high_q_resonance(0.81f) == true);
    CHECK(is_high_q_resonance(0.80f) == false);
    CHECK(is_high_q_resonance(1.00f) == true);
}

TEST_CASE("is_transient triggers below 0.2", "[functions][phase87]") {
    CHECK(is_transient(0.19f) == true);
    CHECK(is_transient(0.20f) == false);
}

TEST_CASE("is_valid_resonance accepts [0, 1]", "[functions][phase87]") {
    CHECK(is_valid_resonance(0.0f)  == true);
    CHECK(is_valid_resonance(1.0f)  == true);
    CHECK(is_valid_resonance(-0.1f) == false);
    CHECK(is_valid_resonance(1.01f) == false);
}

// ---------------------------------------------------------------------------
// § State (s) constants and predicates
// ---------------------------------------------------------------------------

TEST_CASE("State range constants are [0, 2]", "[constants][phase87]") {
    CHECK(STATE_MIN           == Approx(0.0f));
    CHECK(STATE_MAX           == Approx(2.0f));
    CHECK(STATE_VACUUM        == Approx(0.0f));
    CHECK(STATE_BASELINE      == Approx(1.0f));
    CHECK(STATE_DEEP_FOCUS_THR == Approx(1.5f));
}

TEST_CASE("is_deep_focus triggers above 1.5", "[functions][phase87]") {
    CHECK(is_deep_focus(1.51f) == true);
    CHECK(is_deep_focus(1.50f) == false);
}

TEST_CASE("is_near_vacuum triggers below 0.1", "[functions][phase87]") {
    CHECK(is_near_vacuum(0.09f) == true);
    CHECK(is_near_vacuum(0.10f) == false);
}

TEST_CASE("effective_speed equals 1 at baseline state", "[functions][phase87]") {
    CHECK(effective_speed(STATE_BASELINE) == Approx(1.0f));
}

TEST_CASE("effective_speed decreases as s increases beyond baseline", "[functions][phase87]") {
    CHECK(effective_speed(1.5f) < effective_speed(STATE_BASELINE));
}

// ---------------------------------------------------------------------------
// § Spatial (x, y, z) constants
// ---------------------------------------------------------------------------

TEST_CASE("SPATIAL_BITS is 14 and SPATIAL_MAX is 16383", "[constants][phase87]") {
    CHECK(SPATIAL_BITS == 14);
    CHECK(SPATIAL_MAX  == ( (1 << 14) - 1 ));
    CHECK(SPATIAL_MAX  == 16383);
    CHECK(SPATIAL_MIN  == 0);
}

TEST_CASE("is_valid_spatial accepts [0, 16383]", "[functions][phase87]") {
    CHECK(is_valid_spatial(0)     == true);
    CHECK(is_valid_spatial(16383) == true);
    CHECK(is_valid_spatial(-1)    == false);
    CHECK(is_valid_spatial(16384) == false);
}

// ---------------------------------------------------------------------------
// § Quantum amplitude predicates
// ---------------------------------------------------------------------------

TEST_CASE("is_decohered returns true for near-zero amplitude", "[functions][phase87]") {
    // |0|² = 0 < 1e-6  → decohered
    CHECK(is_decohered(std::complex<float>{0.0f, 0.0f}) == true);
    // |1e-3|² = 1e-6  exactly at threshold → not strictly < threshold → not decohered
    // Use a clearly above-threshold amplitude: |0.01|² = 1e-4 >> 1e-6
    CHECK(is_decohered(std::complex<float>{0.01f, 0.0f}) == false);
}

// ---------------------------------------------------------------------------
// § dimension_label
// ---------------------------------------------------------------------------

TEST_CASE("dimension_label returns non-empty string for all 9 dimensions", "[functions][phase87]") {
    for (int i = 0; i < 9; ++i) {
        auto label = dimension_label(static_cast<Dimension>(i));
        CHECK(!label.empty());
    }
}
