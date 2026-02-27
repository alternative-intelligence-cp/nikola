/**
 * @file phase129_homeostasis_test.cpp
 * @brief Phase 129 — HomeostasisMonitor unit tests
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/security/homeostasis.hpp>
#include <nikola/autonomy/decision_loop.hpp>

using namespace nikola::security;
using Catch::Approx;

// Helper: build a NikolaState from key fields
static nikola::autonomy::NikolaState make_state(float dopamine, float atp,
                                                  float boredom, float entropy) {
    nikola::autonomy::NikolaState s;
    s.dopamine = dopamine;
    s.atp      = atp;
    s.boredom  = boredom;
    s.entropy  = entropy;
    return s;
}

// "Healthy" steady-state baseline
static nikola::autonomy::NikolaState healthy_state() {
    return make_state(0.5f, 0.7f, 0.2f, 0.3f);
}

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

TEST_CASE("HomeostasisMonitor::compute_energy — formula check",
          "[Phase129][static]") {
    // energy = 0.3*dopamine + 0.4*atp + 0.3*(1-boredom)
    // = 0.3*0.5 + 0.4*0.7 + 0.3*0.8 = 0.15 + 0.28 + 0.24 = 0.67
    const auto st = make_state(0.5f, 0.7f, 0.2f, 0.0f);
    REQUIRE(HomeostasisMonitor::compute_energy(st) == Approx(0.67).margin(0.001));
}

TEST_CASE("HomeostasisMonitor::compute_energy — clamped at [0,1]",
          "[Phase129][static]") {
    const auto hi = make_state(2.0f, 2.0f, -1.0f, 0.0f);
    REQUIRE(HomeostasisMonitor::compute_energy(hi) == Approx(1.0));

    const auto lo = make_state(-1.0f, -1.0f, 2.0f, 0.0f);
    REQUIRE(HomeostasisMonitor::compute_energy(lo) == Approx(0.0));
}

TEST_CASE("HomeostasisMonitor::compute_entropy — just returns clamped entropy field",
          "[Phase129][static]") {
    const auto st = make_state(0.0f, 0.0f, 0.0f, 0.6f);
    REQUIRE(HomeostasisMonitor::compute_entropy(st) == Approx(0.6));
}

TEST_CASE("HomeostasisMonitor::compute_severity — proportional to delta",
          "[Phase129][static]") {
    // delta=0.5, tol=1.0 → severity=0.5
    REQUIRE(HomeostasisMonitor::compute_severity(0.5, 1.0) == Approx(0.5));
    // delta=2.0, tol=1.0 → clamped at 1.0
    REQUIRE(HomeostasisMonitor::compute_severity(2.0, 1.0) == Approx(1.0));
    // delta=-0.3, tol=0.6 → 0.5
    REQUIRE(HomeostasisMonitor::compute_severity(-0.3, 0.6) == Approx(0.5));
}

// ---------------------------------------------------------------------------
// set_baseline / has_baseline
// ---------------------------------------------------------------------------

TEST_CASE("HomeostasisMonitor — no baseline initially", "[Phase129][baseline]") {
    HomeostasisMonitor hm;
    REQUIRE(hm.has_baseline() == false);
}

TEST_CASE("HomeostasisMonitor::set_baseline — sets baseline flag",
          "[Phase129][baseline]") {
    HomeostasisMonitor hm;
    hm.set_baseline(healthy_state());
    REQUIRE(hm.has_baseline() == true);
}

TEST_CASE("HomeostasisMonitor::stats — reflects baseline after set",
          "[Phase129][baseline]") {
    HomeostasisMonitor hm;
    hm.set_baseline(healthy_state());
    const auto s = hm.stats();
    REQUIRE(s.has_baseline    == true);
    REQUIRE(s.baseline_energy > 0.0);
}

// ---------------------------------------------------------------------------
// check — no anomaly
// ---------------------------------------------------------------------------

TEST_CASE("HomeostasisMonitor::check — identical state passes",
          "[Phase129][check]") {
    HomeostasisMonitor hm;
    const auto ref = healthy_state();
    hm.set_baseline(ref);

    REQUIRE(hm.check(ref) == true);
    REQUIRE(hm.anomaly_count() == 0);
}

TEST_CASE("HomeostasisMonitor::check — no baseline always passes",
          "[Phase129][check]") {
    HomeostasisMonitor hm;
    REQUIRE(hm.check(healthy_state()) == true);
    REQUIRE(hm.anomaly_history().empty() == true);
}

TEST_CASE("HomeostasisMonitor::check — small drift within tolerance passes",
          "[Phase129][check]") {
    HomeostasisMonitor hm;
    hm.set_baseline(healthy_state());
    hm.set_energy_tolerance(0.20);

    // Perturb slightly (< 0.20 energy delta)
    const auto nearby = make_state(0.52f, 0.72f, 0.18f, 0.31f);
    REQUIRE(hm.check(nearby) == true);
}

// ---------------------------------------------------------------------------
// check — energy spike anomaly
// ---------------------------------------------------------------------------

TEST_CASE("HomeostasisMonitor::check — energy spike detected",
          "[Phase129][energy]") {
    HomeostasisMonitor hm;
    hm.set_baseline(make_state(0.3f, 0.3f, 0.5f, 0.3f)); // energy ≈ 0.30
    hm.set_energy_tolerance(0.05);

    // Large dopamine+atp spike → energy jumps
    const auto spiked = make_state(1.0f, 1.0f, 0.0f, 0.3f);  // energy = 1.0
    const bool ok = hm.check(spiked);

    REQUIRE(ok == false);
    REQUIRE(hm.anomaly_count() > 0);

    const auto& rec = hm.anomaly_history().back();
    REQUIRE(rec.type == AnomalyType::ENERGY_SPIKE);
    REQUIRE(rec.delta > 0.0);
    REQUIRE(rec.severity > 0.0);
}

TEST_CASE("HomeostasisMonitor::check — energy drop detected",
          "[Phase129][energy]") {
    HomeostasisMonitor hm;
    hm.set_baseline(make_state(1.0f, 1.0f, 0.0f, 0.3f));   // high energy
    hm.set_energy_tolerance(0.05);

    const auto dropped = make_state(0.0f, 0.0f, 1.0f, 0.3f); // low energy
    const bool ok = hm.check(dropped);

    REQUIRE(ok == false);
    const auto& rec = hm.anomaly_history().back();
    REQUIRE(rec.type == AnomalyType::ENERGY_DROP);
    REQUIRE(rec.delta < 0.0);
}

// ---------------------------------------------------------------------------
// check — entropy anomaly
// ---------------------------------------------------------------------------

TEST_CASE("HomeostasisMonitor::check — entropy spike detected",
          "[Phase129][entropy]") {
    HomeostasisMonitor hm;
    hm.set_baseline(make_state(0.5f, 0.5f, 0.3f, 0.1f));  // low entropy
    hm.set_entropy_tolerance(0.10);

    const auto chaotic = make_state(0.5f, 0.5f, 0.3f, 0.9f);  // entropy=0.9
    const bool ok = hm.check(chaotic);

    REQUIRE(ok == false);

    // Find entropy spike record
    bool found = false;
    for (const auto& r : hm.anomaly_history()) {
        if (r.type == AnomalyType::ENTROPY_SPIKE) found = true;
    }
    REQUIRE(found == true);
}

TEST_CASE("HomeostasisMonitor::check — entropy drop detected",
          "[Phase129][entropy]") {
    HomeostasisMonitor hm;
    hm.set_baseline(make_state(0.5f, 0.5f, 0.3f, 0.9f));  // high entropy
    hm.set_entropy_tolerance(0.10);

    const auto stable = make_state(0.5f, 0.5f, 0.3f, 0.1f);
    REQUIRE(hm.check(stable) == false);

    const auto& rec = hm.anomaly_history().back();
    REQUIRE(rec.type == AnomalyType::ENTROPY_DROP);
}

// ---------------------------------------------------------------------------
// verify_integrity
// ---------------------------------------------------------------------------

TEST_CASE("HomeostasisMonitor::verify_integrity — no side-effects",
          "[Phase129][verify]") {
    HomeostasisMonitor hm;
    hm.set_baseline(healthy_state());

    const bool ok = hm.verify_integrity(healthy_state());
    REQUIRE(ok == true);
    REQUIRE(hm.check_count()   == 0);
    REQUIRE(hm.anomaly_count() == 0);
}

TEST_CASE("HomeostasisMonitor::verify_integrity — detects bad state",
          "[Phase129][verify]") {
    HomeostasisMonitor hm;
    hm.set_baseline(make_state(0.5f, 0.5f, 0.3f, 0.2f));
    hm.set_energy_tolerance(0.05);

    const auto spiked = make_state(1.0f, 1.0f, 0.0f, 0.2f);
    REQUIRE(hm.verify_integrity(spiked) == false);
    // No history side-effects
    REQUIRE(hm.anomaly_count() == 0);
}

// ---------------------------------------------------------------------------
// Lockdown
// ---------------------------------------------------------------------------

TEST_CASE("HomeostasisMonitor — not locked down initially",
          "[Phase129][lockdown]") {
    HomeostasisMonitor hm;
    REQUIRE(hm.is_locked_down() == false);
}

TEST_CASE("HomeostasisMonitor::trigger_lockdown / release_lockdown",
          "[Phase129][lockdown]") {
    HomeostasisMonitor hm;
    hm.trigger_lockdown();
    REQUIRE(hm.is_locked_down() == true);

    hm.release_lockdown();
    REQUIRE(hm.is_locked_down() == false);
}

TEST_CASE("HomeostasisMonitor — auto-lockdown on severe anomaly",
          "[Phase129][lockdown]") {
    HomeostasisMonitor hm;
    hm.set_baseline(make_state(0.5f, 0.5f, 0.3f, 0.2f));
    hm.set_energy_tolerance(0.01);     // ultra-tight
    hm.set_lockdown_threshold(0.5);    // low threshold → easy to trigger

    const auto extreme = make_state(1.0f, 1.0f, 0.0f, 0.2f);
    hm.check(extreme);

    REQUIRE(hm.is_locked_down() == true);
}

// ---------------------------------------------------------------------------
// Anomaly callback
// ---------------------------------------------------------------------------

TEST_CASE("HomeostasisMonitor::on_anomaly — fires on anomaly",
          "[Phase129][callback]") {
    HomeostasisMonitor hm;
    hm.set_baseline(make_state(0.5f, 0.5f, 0.3f, 0.2f));
    hm.set_energy_tolerance(0.01);

    bool fired = false;
    AnomalyType fired_type = AnomalyType::ENERGY_SPIKE;

    hm.on_anomaly([&](const AnomalyRecord& r) {
        fired      = true;
        fired_type = r.type;
    });

    hm.check(make_state(1.0f, 1.0f, 0.0f, 0.2f));
    REQUIRE(fired == true);
    REQUIRE(fired_type == AnomalyType::ENERGY_SPIKE);
}

TEST_CASE("HomeostasisMonitor::on_anomaly — not fired when no anomaly",
          "[Phase129][callback]") {
    HomeostasisMonitor hm;
    const auto ref = healthy_state();
    hm.set_baseline(ref);

    bool fired = false;
    hm.on_anomaly([&](const AnomalyRecord&) { fired = true; });

    hm.check(ref);
    REQUIRE(fired == false);
}

// ---------------------------------------------------------------------------
// History FIFO cap
// ---------------------------------------------------------------------------

TEST_CASE("HomeostasisMonitor — anomaly history respects FIFO cap",
          "[Phase129][history]") {
    HomeostasisMonitor hm;
    hm.set_baseline(make_state(0.5f, 0.5f, 0.3f, 0.2f));
    hm.set_energy_tolerance(0.001);  // trigger anomaly every check

    const auto bad = make_state(1.0f, 1.0f, 0.0f, 0.2f);

    for (size_t i = 0; i <= HSK_MAX_HISTORY + 10; ++i) {
        hm.check(bad);
    }
    REQUIRE(hm.anomaly_history().size() <= HSK_MAX_HISTORY);
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

TEST_CASE("HomeostasisMonitor::stats — empty state", "[Phase129][stats]") {
    HomeostasisMonitor hm;
    const auto s = hm.stats();
    REQUIRE(s.has_baseline      == false);
    REQUIRE(s.total_checks      == 0);
    REQUIRE(s.total_anomalies   == 0);
    REQUIRE(s.locked_down       == false);
    REQUIRE(s.monitoring_active == false);
}

TEST_CASE("HomeostasisMonitor::stats — reflects check history",
          "[Phase129][stats]") {
    HomeostasisMonitor hm;
    hm.set_baseline(healthy_state());
    hm.check(healthy_state());
    hm.check(healthy_state());

    const auto s = hm.stats();
    REQUIRE(s.total_checks    == 2);
    REQUIRE(s.total_anomalies == 0);
}
