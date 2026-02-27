/**
 * @file phase132_membrane_test.cpp
 * @brief Phase 132 — SocialMembrane unit tests
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/social/membrane.hpp>

using namespace nikola::social;
using Catch::Approx;

// ---------------------------------------------------------------------------
// Static helper
// ---------------------------------------------------------------------------

TEST_CASE("SocialMembrane::compute_permeability — trust=1 dissonance=0",
          "[Phase132][static]") {
    // trust/(dissonance+eps) = 1/(0+eps) → clamped to 1.0
    REQUIRE(SocialMembrane::compute_permeability(1.0, 0.0) == Approx(1.0));
}

TEST_CASE("SocialMembrane::compute_permeability — trust=0 → 0",
          "[Phase132][static]") {
    REQUIRE(SocialMembrane::compute_permeability(0.0, 0.5) == Approx(0.0));
}

TEST_CASE("SocialMembrane::compute_permeability — typical values",
          "[Phase132][static]") {
    // trust=0.5, dissonance=0.5 → 0.5/0.5 = 1.0 (clamped to 1.0)
    REQUIRE(SocialMembrane::compute_permeability(0.5, 0.5) == Approx(1.0));

    // trust=0.2, dissonance=0.8 → 0.2/0.8 = 0.25
    REQUIRE(SocialMembrane::compute_permeability(0.2, 0.8) == Approx(0.25).margin(0.01));
}

// ---------------------------------------------------------------------------
// Default state
// ---------------------------------------------------------------------------

TEST_CASE("SocialMembrane — default state", "[Phase132][init]") {
    SocialMembrane m;
    REQUIRE(m.get_trust()      == Approx(0.5));
    REQUIRE(m.get_dissonance() == Approx(0.5));
    // permeability = 0.5/(0.5+eps) ≈ 1.0, but default is 0.1 from ctor
    // (ctor sets permeability_=0.1 before any recalc)
    REQUIRE(m.get_permeability() == Approx(0.1));
    REQUIRE(m.interaction_count() == 0);
}

// ---------------------------------------------------------------------------
// filter_incoming
// ---------------------------------------------------------------------------

TEST_CASE("SocialMembrane::filter_incoming — permeability=0 → self_wave",
          "[Phase132][filter]") {
    SocialMembrane m;
    m.set_permeability(0.0);
    const std::complex<double> fw{2.0, 3.0};
    const std::complex<double> sw{1.0, 0.5};
    const auto result = m.filter_incoming(fw, sw);
    REQUIRE(result.real() == Approx(sw.real()));
    REQUIRE(result.imag() == Approx(sw.imag()));
}

TEST_CASE("SocialMembrane::filter_incoming — permeability=1 → friend_wave",
          "[Phase132][filter]") {
    SocialMembrane m;
    m.set_permeability(1.0);
    const std::complex<double> fw{2.0, 3.0};
    const std::complex<double> sw{1.0, 0.5};
    const auto result = m.filter_incoming(fw, sw);
    REQUIRE(result.real() == Approx(fw.real()));
    REQUIRE(result.imag() == Approx(fw.imag()));
}

TEST_CASE("SocialMembrane::filter_incoming — permeability=0.5 → midpoint",
          "[Phase132][filter]") {
    SocialMembrane m;
    m.set_permeability(0.5);
    const std::complex<double> fw{2.0, 0.0};
    const std::complex<double> sw{0.0, 0.0};
    const auto result = m.filter_incoming(fw, sw);
    REQUIRE(result.real() == Approx(1.0));
    REQUIRE(result.imag() == Approx(0.0));
}

// ---------------------------------------------------------------------------
// update_trust
// ---------------------------------------------------------------------------

TEST_CASE("SocialMembrane::update_trust — positive increases trust",
          "[Phase132][trust]") {
    SocialMembrane m;
    const double before = m.get_trust();
    m.update_trust(true);
    REQUIRE(m.get_trust() > before);
    REQUIRE(m.positive_interaction_count() == 1);
}

TEST_CASE("SocialMembrane::update_trust — negative decreases trust",
          "[Phase132][trust]") {
    SocialMembrane m;
    const double before = m.get_trust();
    m.update_trust(false);
    REQUIRE(m.get_trust() < before);
    REQUIRE(m.negative_interaction_count() == 1);
}

TEST_CASE("SocialMembrane::update_trust — trust clamped at [0,1]",
          "[Phase132][trust]") {
    SocialMembrane m;
    m.set_trust(1.0);
    for (int i = 0; i < 20; ++i) m.update_trust(true);
    REQUIRE(m.get_trust() <= 1.0);

    m.set_trust(0.0);
    for (int i = 0; i < 20; ++i) m.update_trust(false);
    REQUIRE(m.get_trust() >= 0.0);
}

TEST_CASE("SocialMembrane::update_trust — recalculates permeability",
          "[Phase132][trust]") {
    SocialMembrane m;
    m.set_trust(0.1);
    m.set_dissonance(0.9);
    const double p_before = m.get_permeability();

    m.update_trust(true);  // raises trust → should raise permeability
    REQUIRE(m.get_permeability() >= p_before);
}

// ---------------------------------------------------------------------------
// update_dissonance
// ---------------------------------------------------------------------------

TEST_CASE("SocialMembrane::update_dissonance — positive delta increases",
          "[Phase132][dissonance]") {
    SocialMembrane m;
    m.set_dissonance(0.3);
    m.update_dissonance(0.2);
    REQUIRE(m.get_dissonance() == Approx(0.5));
}

TEST_CASE("SocialMembrane::update_dissonance — clamped at [0,1]",
          "[Phase132][dissonance]") {
    SocialMembrane m;
    m.set_dissonance(0.9);
    m.update_dissonance(5.0);
    REQUIRE(m.get_dissonance() <= 1.0);

    m.set_dissonance(0.1);
    m.update_dissonance(-5.0);
    REQUIRE(m.get_dissonance() >= 0.0);
}

// ---------------------------------------------------------------------------
// set_trust / set_dissonance / set_permeability
// ---------------------------------------------------------------------------

TEST_CASE("SocialMembrane::set_trust — updates and recomputes permeability",
          "[Phase132][setters]") {
    SocialMembrane m;
    m.set_trust(1.0);
    REQUIRE(m.get_trust() == Approx(1.0));
    // recalculated permeability should be max
    REQUIRE(m.get_permeability() == Approx(1.0));
}

TEST_CASE("SocialMembrane::set_dissonance — updates and recomputes permeability",
          "[Phase132][setters]") {
    SocialMembrane m;
    m.set_trust(0.5);
    m.set_dissonance(0.0);
    // permeability → clamped at 1.0
    REQUIRE(m.get_permeability() == Approx(1.0));
}

TEST_CASE("SocialMembrane::set_permeability — clamps to [0,1]",
          "[Phase132][setters]") {
    SocialMembrane m;
    m.set_permeability(2.0);
    REQUIRE(m.get_permeability() <= 1.0);
    m.set_permeability(-1.0);
    REQUIRE(m.get_permeability() >= 0.0);
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

TEST_CASE("SocialMembrane::reset — restores defaults", "[Phase132][reset]") {
    SocialMembrane m;
    m.update_trust(true);
    m.update_trust(true);
    m.update_dissonance(0.3);
    m.reset();
    REQUIRE(m.get_trust()        == Approx(0.5));
    REQUIRE(m.get_dissonance()   == Approx(0.5));
    REQUIRE(m.get_permeability() == Approx(0.1));
    REQUIRE(m.interaction_count() == 0);
}

// ---------------------------------------------------------------------------
// stats
// ---------------------------------------------------------------------------

TEST_CASE("SocialMembrane::stats — reflects state", "[Phase132][stats]") {
    SocialMembrane m;
    m.update_trust(true);
    m.update_trust(false);
    const auto s = m.stats();
    REQUIRE(s.positive_count == 1);
    REQUIRE(s.negative_count == 1);
}
