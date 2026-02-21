/**
 * @file tests/unit/phase15_hamiltonian_value_test.cpp
 * @brief NIK-005 — Hamiltonian Value Function test suite (Catch2 v3).
 *
 * Verifies the core theorem from Section 3 of TASK_HAMILTONIAN_VALUE_FUNCTION:
 *
 *   Stroboscopic Value Collapse:
 *     - Σ|Ψ|² oscillates at 2ω for a standing wave   → spurious TD errors
 *     - H (Hamiltonian) is INVARIANT for stable waves → TD error δ ≈ 0
 *
 * Tests cover:
 *   NIK-005-A  basic compute_spans returns finite, positive result
 *   NIK-005-B  standing-wave invariance: H stable, |Ψ|² oscillates
 *   NIK-005-C  γ coefficients scale contributions independently
 *   NIK-005-D  H_max clamping (epileptic resonance safety valve)
 *   NIK-005-E  stability_penalty — zero below threshold, proportional above
 *   NIK-005-F  td_error helper — δ=0 for stable conservative state
 *   NIK-005-G  NaN/Inf input → clamp to 0, no UB
 *   NIK-005-H  AutonomyEngine::tick_physics() integrates correctly
 *
 * No live physics required — all tests use synthetic wavefunction buffers.
 */

#define NIKOLA_AUTONOMY_ENGINE_IMPL

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/autonomy/hamiltonian_value.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <vector>

using namespace nikola::autonomy;
using Catch::Matchers::WithinAbs;
using Catch::Approx;

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers — synthetic standing-wave snapshots
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build a "standing wave" snapshot at phase ωt.
 * Ψ(x,t)  = A cos(kx) cos(ωt)    → psi_r[i] = A*cos(kx_i)*cos(ωt)
 * ∂_t Ψ   = -Aω cos(kx) sin(ωt) → vel_r[i] = -A*ω*cos(kx_i)*sin(ωt)
 * (imaginary parts zero for a real wave)
 */
static void make_standing_wave(
        std::vector<float>& psi_r, std::vector<float>& psi_i,
        std::vector<float>& vel_r, std::vector<float>& vel_i,
        int N, float A, float k, float omega, float t)
{
    psi_r.resize(N); psi_i.resize(N, 0.0f);
    vel_r.resize(N); vel_i.resize(N, 0.0f);
    for (int i = 0; i < N; ++i) {
        const float x = static_cast<float>(i) / static_cast<float>(N);
        psi_r[i] =  A * std::cos(k * x) * std::cos(omega * t);
        vel_r[i] = -A * omega * std::cos(k * x) * std::sin(omega * t);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-005-A  basic compute_spans
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-005-A — compute_spans returns finite positive value",
          "[hamiltonian_value][nik005]")
{
    constexpr int N = 64;
    std::vector<float> psi_r(N, 1.0f), psi_i(N, 0.0f);
    std::vector<float> vel_r(N, 0.5f), vel_i(N, 0.0f);

    HamiltonianValue hv;
    const float H = hv.compute_spans(psi_r, psi_i, vel_r, vel_i, 0.0f);

    CHECK(H > 0.0f);
    CHECK(std::isfinite(H));
}

TEST_CASE("NIK-005-A — empty spans return 0", "[hamiltonian_value][nik005]")
{
    HamiltonianValue hv;
    std::vector<float> empty;
    CHECK(hv.compute_spans(empty, empty, empty, empty, 0.0f) == Approx(0.0f));
}

TEST_CASE("NIK-005-A — zero field returns 0", "[hamiltonian_value][nik005]")
{
    constexpr int N = 32;
    std::vector<float> zero(N, 0.0f);
    HamiltonianValue hv;
    CHECK(hv.compute_spans(zero, zero, zero, zero, 0.0f) == Approx(0.0f));
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-005-B  standing-wave invariance (the core theorem)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-005-B — Hamiltonian is invariant for a standing wave",
          "[hamiltonian_value][nik005][core_theorem]")
{
    // Standing wave: T + V = const regardless of phase ωt
    constexpr int   N     = 256;
    constexpr float A     = 1.0f;
    constexpr float k     = 2.0f * std::numbers::pi_v<float>;   // one cycle
    constexpr float omega = 1.0f;
    constexpr int   steps = 32;

    HamiltonianValue hv;
    // Keep default gamma_k=1, gamma_p=1 so H = T + V = conserved invariant.
    // (gamma_p=0 would give pure kinetic, which is zero at t=0 for a standing wave.)

    std::vector<float> psi_r, psi_i, vel_r, vel_i;
    std::vector<float> H_vals(steps);

    for (int s = 0; s < steps; ++s) {
        const float t = static_cast<float>(s) / static_cast<float>(steps)
                        * 2.0f * std::numbers::pi_v<float> / omega;
        make_standing_wave(psi_r, psi_i, vel_r, vel_i, N, A, k, omega, t);
        H_vals[s] = hv.compute_spans(psi_r, psi_i, vel_r, vel_i, 0.0f);
    }

    // All Hamiltonian values should be within ±2% of the first
    const float H0 = H_vals[0];
    REQUIRE(H0 > 0.0f);
    for (int s = 1; s < steps; ++s) {
        CHECK_THAT(H_vals[s], WithinAbs(H0, H0 * 0.02f));
    }
}

TEST_CASE("NIK-005-B — potential-only DOES oscillate (confirms the bug)",
          "[hamiltonian_value][nik005][core_theorem]")
{
    // Demonstrate that pure Σ|Ψ|² varies with wave phase
    constexpr int   N     = 256;
    constexpr float A     = 1.5f;
    constexpr float k     = 2.0f * std::numbers::pi_v<float>;
    constexpr float omega = 1.0f;

    // At t=0: full potential (max), velocity=0
    std::vector<float> psi_r0, psi_i0, vel_r0, vel_i0;
    make_standing_wave(psi_r0, psi_i0, vel_r0, vel_i0, N, A, k, omega, 0.0f);

    // At t=π/(2ω): zero potential, full kinetic
    std::vector<float> psi_r1, psi_i1, vel_r1, vel_i1;
    const float t_quarter = std::numbers::pi_v<float> / (2.0f * omega);
    make_standing_wave(psi_r1, psi_i1, vel_r1, vel_i1, N, A, k, omega, t_quarter);

    // Σ|Ψ|² at t=0 (should be large)
    float psi_sq_0 = 0.0f;
    for (int i = 0; i < N; ++i) psi_sq_0 += psi_r0[i]*psi_r0[i];

    // Σ|Ψ|² at t=π/2ω (should be ≈ 0)
    float psi_sq_quarter = 0.0f;
    for (int i = 0; i < N; ++i) psi_sq_quarter += psi_r1[i]*psi_r1[i];

    // Confirm the stroboscopic oscillation
    CHECK(psi_sq_0 > 0.0f);
    CHECK(psi_sq_quarter < psi_sq_0 * 0.05f);   // < 5% of max = near zero

    // Now verify Hamiltonian value (kinetic+potential) is stable by contrast
    HamiltonianValue hv;
    const float H0 = hv.compute_spans(psi_r0, psi_i0, vel_r0, vel_i0, 0.0f);
    const float H1 = hv.compute_spans(psi_r1, psi_i1, vel_r1, vel_i1, 0.0f);

    // H should be approximately equal at both phases
    CHECK_THAT(H1, WithinAbs(H0, H0 * 0.02f));
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-005-C  γ coefficient scaling
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-005-C — gamma_k=0 removes kinetic contribution",
          "[hamiltonian_value][nik005][gamma]")
{
    constexpr int N = 32;
    std::vector<float> psi_r(N, 0.0f), psi_i(N, 0.0f);   // no psi
    std::vector<float> vel_r(N, 2.0f), vel_i(N, 0.0f);   // only kinetic

    HamiltonianValue hv;
    hv.gamma_p  = 0.0f;
    hv.gamma_nl = 0.0f;

    hv.gamma_k = 1.0f;
    const float H_with_kinetic = hv.compute_spans(psi_r, psi_i, vel_r, vel_i, 0.0f);
    CHECK(H_with_kinetic > 0.0f);

    hv.gamma_k = 0.0f;
    const float H_no_kinetic = hv.compute_spans(psi_r, psi_i, vel_r, vel_i, 0.0f);
    CHECK(H_no_kinetic == Approx(0.0f));
}

TEST_CASE("NIK-005-C — gamma_p=0 removes potential contribution",
          "[hamiltonian_value][nik005][gamma]")
{
    constexpr int N = 16;
    std::vector<float> psi_r(N, 3.0f), psi_i(N, 0.0f);   // only potential
    std::vector<float> vel_r(N, 0.0f), vel_i(N, 0.0f);

    HamiltonianValue hv;
    hv.gamma_k  = 0.0f;
    hv.gamma_nl = 0.0f;

    hv.gamma_p = 1.0f;
    const float H_with_p = hv.compute_spans(psi_r, psi_i, vel_r, vel_i, 0.0f);
    CHECK(H_with_p > 0.0f);

    hv.gamma_p = 0.0f;
    const float H_no_p = hv.compute_spans(psi_r, psi_i, vel_r, vel_i, 0.0f);
    CHECK(H_no_p == Approx(0.0f));
}

TEST_CASE("NIK-005-C — doubling gamma_k doubles kinetic contribution",
          "[hamiltonian_value][nik005][gamma]")
{
    constexpr int N = 16;
    std::vector<float> psi_r(N, 0.0f), psi_i(N, 0.0f);   // zero psi
    std::vector<float> vel_r(N, 1.0f), vel_i(N, 0.0f);   // unit velocity

    HamiltonianValue hv;
    hv.gamma_p  = 0.0f;
    hv.gamma_nl = 0.0f;

    hv.gamma_k = 1.0f;
    const float H1 = hv.compute_spans(psi_r, psi_i, vel_r, vel_i, 0.0f);

    hv.gamma_k = 2.0f;
    const float H2 = hv.compute_spans(psi_r, psi_i, vel_r, vel_i, 0.0f);

    CHECK(H2 == Approx(2.0f * H1).epsilon(0.001f));
}

TEST_CASE("NIK-005-C — nonlinear term scales with beta",
          "[hamiltonian_value][nik005][gamma]")
{
    constexpr int N = 8;
    std::vector<float> psi_r(N, 1.0f), psi_i(N, 0.0f);
    std::vector<float> vel_r(N, 0.0f), vel_i(N, 0.0f);

    HamiltonianValue hv;
    hv.gamma_k = 0.0f;
    hv.gamma_p = 0.0f;
    hv.gamma_nl = 1.0f;

    const float H_beta1 = hv.compute_spans(psi_r, psi_i, vel_r, vel_i, 1.0f);
    const float H_beta2 = hv.compute_spans(psi_r, psi_i, vel_r, vel_i, 2.0f);

    // NL = γ_NL * (beta/2) * |Ψ|⁴ — doubling beta doubles contribution
    CHECK(H_beta2 == Approx(2.0f * H_beta1).epsilon(0.001f));
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-005-D  H_max clamping
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-005-D — H_max clamps large values", "[hamiltonian_value][nik005][hmax]")
{
    constexpr int N = 64;
    // Large amplitude → large H
    std::vector<float> psi_r(N, 1000.0f), psi_i(N, 0.0f);
    std::vector<float> vel_r(N, 1000.0f), vel_i(N, 0.0f);

    HamiltonianValue hv;
    hv.h_max = 100.0f;   // set a low cap

    const float H = hv.compute_spans(psi_r, psi_i, vel_r, vel_i, 0.0f);
    CHECK(H <= 100.0f);
}

TEST_CASE("NIK-005-D — small H is not clamped", "[hamiltonian_value][nik005][hmax]")
{
    constexpr int N = 4;
    std::vector<float> psi_r(N, 0.1f), psi_i(N, 0.0f);
    std::vector<float> vel_r(N, 0.1f), vel_i(N, 0.0f);

    HamiltonianValue hv;
    hv.h_max = 1e6f;

    const float H = hv.compute_spans(psi_r, psi_i, vel_r, vel_i, 0.0f);
    CHECK(H < hv.h_max);
    CHECK(H > 0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-005-E  stability_penalty
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-005-E — no penalty below H_max", "[hamiltonian_value][nik005][penalty]")
{
    CHECK(HamiltonianValue::stability_penalty(50.0f, 100.0f) == Approx(0.0f));
    CHECK(HamiltonianValue::stability_penalty(100.0f, 100.0f) == Approx(0.0f));
}

TEST_CASE("NIK-005-E — penalty proportional to excess above H_max",
          "[hamiltonian_value][nik005][penalty]")
{
    constexpr float H_max = 100.0f;

    CHECK(HamiltonianValue::stability_penalty(110.0f, H_max, 1.0f) == Approx(10.0f));
    CHECK(HamiltonianValue::stability_penalty(200.0f, H_max, 1.0f) == Approx(100.0f));
    CHECK(HamiltonianValue::stability_penalty(110.0f, H_max, 2.0f) == Approx(20.0f));
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-005-F  td_error helper
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-005-F — TD error ≈ 0 for stable state (core theorem)",
          "[hamiltonian_value][nik005][td_error]")
{
    // Stable state: H_prev = H_curr, R = 0, γ = 1
    constexpr float H = 42.7f;
    const float delta = HamiltonianValue::td_error(H, H, 0.0f, 1.0f);
    CHECK(delta == Approx(0.0f).margin(1e-5f));
}

TEST_CASE("NIK-005-F — TD error positive for energy gain",
          "[hamiltonian_value][nik005][td_error]")
{
    const float delta = HamiltonianValue::td_error(10.0f, 20.0f, 0.0f, 1.0f);
    CHECK(delta > 0.0f);
}

TEST_CASE("NIK-005-F — TD error negative for energy loss",
          "[hamiltonian_value][nik005][td_error]")
{
    const float delta = HamiltonianValue::td_error(20.0f, 10.0f, 0.0f, 1.0f);
    CHECK(delta < 0.0f);
}

TEST_CASE("NIK-005-F — TD error includes reward signal",
          "[hamiltonian_value][nik005][td_error]")
{
    // Same energy, positive reward → positive TD error
    const float delta = HamiltonianValue::td_error(5.0f, 5.0f, 1.0f, 1.0f);
    CHECK(delta > 0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-005-G  NaN/Inf robustness
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-005-G — NaN input is handled gracefully",
          "[hamiltonian_value][nik005][robustness]")
{
    constexpr int N = 4;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> psi_r = {nan, 1.0f, 1.0f, 1.0f};
    std::vector<float> psi_i(N, 0.0f);
    std::vector<float> vel_r(N, 1.0f), vel_i(N, 0.0f);

    HamiltonianValue hv;
    const float H = hv.compute_spans(psi_r, psi_i, vel_r, vel_i, 0.0f);
    // Must return finite (clamped) result, not NaN
    CHECK(std::isfinite(H));
}

TEST_CASE("NIK-005-G — Inf input is clamped to 0 or h_max",
          "[hamiltonian_value][nik005][robustness]")
{
    constexpr int N = 4;
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<float> psi_r(N, inf), psi_i(N, 0.0f);
    std::vector<float> vel_r(N, 0.0f), vel_i(N, 0.0f);

    HamiltonianValue hv;
    hv.h_max = 1000.0f;
    const float H = hv.compute_spans(psi_r, psi_i, vel_r, vel_i, 0.0f);
    CHECK(std::isfinite(H));
    CHECK(H <= hv.h_max);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-005-H  AutonomyEngine::tick_physics() integration
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-005-H — tick_physics() advances engine without error",
          "[hamiltonian_value][nik005][integration]")
{
    AutonomyEngine engine;

    constexpr int N = 64;
    std::vector<float> psi_r(N, 0.5f), psi_i(N, 0.1f);
    std::vector<float> vel_r(N, 0.2f), vel_i(N, 0.0f);

    CHECK_NOTHROW(engine.tick_physics(0.016f, psi_r, psi_i, vel_r, vel_i,
                                       0.0f, Reward::POSITIVE, 0.0f));

    // Dopamine should have moved from baseline after a positive reward
    CHECK(engine.dopamine() > 0.0f);
}

TEST_CASE("NIK-005-H — tick_physics() dopamine is stable for standing wave",
          "[hamiltonian_value][nik005][integration]")
{
    // For a standing wave, tick_physics() dopamine should NOT oscillate wildly
    // (in contrast to tick() which uses stroboscopic |Ψ|²)
    AutonomyEngine engine;
    AutonomyConfig cfg;
    cfg.enable_boredom = false;
    AutonomyEngine eng2(cfg);

    constexpr int   N     = 128;
    constexpr float A     = 1.0f;
    constexpr float k     = 2.0f * std::numbers::pi_v<float>;
    constexpr float omega = 1.0f;
    constexpr int   steps = 16;

    std::vector<float> psi_r, psi_i, vel_r, vel_i;
    std::vector<float> dopa_vals(steps);

    for (int s = 0; s < steps; ++s) {
        const float t = static_cast<float>(s) * 0.1f;
        make_standing_wave(psi_r, psi_i, vel_r, vel_i, N, A, k, omega, t);
        eng2.tick_physics(0.016f, psi_r, psi_i, vel_r, vel_i,
                           0.0f, Reward::NEUTRAL, static_cast<float>(s) * 0.016f);
        dopa_vals[s] = eng2.dopamine();
    }

    // After the initial transient (first 2 steps), dopamine should be stable
    float dmin = *std::min_element(dopa_vals.begin() + 2, dopa_vals.end());
    float dmax = *std::max_element(dopa_vals.begin() + 2, dopa_vals.end());
    CHECK((dmax - dmin) < 0.15f);   // ≤0.15 range — not wildly oscillating
}

TEST_CASE("NIK-005-H — hamiltonian_value() accessor allows config",
          "[hamiltonian_value][nik005][integration]")
{
    AutonomyEngine engine;
    engine.hamiltonian_value().gamma_k  = 2.0f;
    engine.hamiltonian_value().gamma_p  = 0.5f;
    engine.hamiltonian_value().h_max    = 500.0f;

    CHECK(engine.hamiltonian_value().gamma_k  == Approx(2.0f));
    CHECK(engine.hamiltonian_value().gamma_p  == Approx(0.5f));
    CHECK(engine.hamiltonian_value().h_max    == Approx(500.0f));
}
