/**
 * @file src/nikola_state_dump.cpp
 * @brief Nikola state snapshot — JSON output for AI bridge
 *
 * Boots the 9D torus, runs N_STEPS of physics + autonomy, then emits a
 * single JSON object to stdout describing the live system state.
 *
 * Used by tools/nikola_hello.py to inject Nikola's actual physics state
 * into the Aria specialist's context for the first AI↔Nikola dialogue.
 *
 * Output format (single JSON line):
 * {
 *   "version":   "0.0.4",
 *   "grid_n":    3,
 *   "n_nodes":   19683,
 *   "steps":     100,
 *   "dt":        0.001,
 *   "c0":        1.0,
 *   "beta":      1.0,
 *   "h0":        68890.123,
 *   "h_final":   68891.456,
 *   "drift_pct": 0.00193,
 *   "elapsed_ms":14.7,
 *   "dopamine":  0.7231,
 *   "atp":       0.9812,
 *   "boredom":   0.0331,
 *   "psi_sample": [
 *     {"re": 0.312, "im": -0.041, "vr": 0.001, "vi": 0.000},
 *     ...  (5 nodes)
 *   ]
 * }
 */

#include <nikola/physics/wave_function.hpp>
#include <nikola/physics/hamiltonian.hpp>
#include <nikola/physics/propagator.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/autonomy/hamiltonian_value.hpp>
#include <nikola/security/bootstrap_manager.hpp>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <string>

// ── Physics constants (identical to first_light) ────────────────────────────
static constexpr int   GRID_N   = 3;
static constexpr float C0       = 1.0f;
static constexpr float BETA     = 1.0f;
static constexpr float DT       = 0.001f;
static constexpr int   N_STEPS  = 100;   // short run — just enough for stable state

// ── Minimal JSON helpers ─────────────────────────────────────────────────────

static std::string jf(double v, int prec = 6) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << v;
    return o.str();
}

int main() {
    using namespace nikola;

    // ── Bootstrap ────────────────────────────────────────────────────────────
    security::BootstrapManager bootstrap;
    bootstrap.set_silent(true);   // JSON goes to stdout; suppress bootstrap box
    bootstrap.get_token();

    // ── Wave function + manifold seeding ─────────────────────────────────────
    physics::WaveFunction wf;
    wf.seed_manifold(GRID_N, /*pilot_dim=*/0, /*k_mode=*/1,
                     /*amplitude=*/1.0f, /*seed=*/2026u);
    const std::size_t N_NODES = wf.grid().num_active_nodes();

    // ── Hamiltonian baseline ──────────────────────────────────────────────────
    physics::Hamiltonian ham;
    ham.set_c0(C0).set_beta(BETA);
    ham.verify_initial_conditions(wf);
    const double H0 = ham.compute(wf);

    // ── Autonomy engine ───────────────────────────────────────────────────────
    autonomy::AutonomyConfig acfg;
    acfg.initial_atp        = 1.0f;
    acfg.entropy_sample_dt  = 0.1f;
    acfg.enable_boredom     = true;
    acfg.enable_dream_weave = false;
    autonomy::AutonomyEngine engine(acfg);
    engine.hamiltonian_value().gamma_k  = 1.0f;
    engine.hamiltonian_value().gamma_p  = 1.0f;
    engine.hamiltonian_value().gamma_nl = 1.0f;
    engine.hamiltonian_value().h_max    = static_cast<float>(H0 * 100.0);

    // ── Propagator ───────────────────────────────────────────────────────────
    physics::Propagator propagator;
    propagator.set_beta(BETA).set_c0(C0);

    // ── Main loop ────────────────────────────────────────────────────────────
    auto t_start = std::chrono::steady_clock::now();

    for (int step = 1; step <= N_STEPS; ++step) {
        propagator.step(wf, DT);

        const float wall_t = static_cast<float>(step) * DT;
        const std::size_t N = wf.grid().num_active_nodes();
        std::span<const float> psi_r(wf.grid().psi_real(), N);
        std::span<const float> psi_i(wf.grid().psi_imag(), N);
        std::span<const float> vel_r(wf.grid().vel_real(), N);
        std::span<const float> vel_i(wf.grid().vel_imag(), N);
        engine.tick_physics(DT, psi_r, psi_i, vel_r, vel_i,
                            BETA, autonomy::Reward::NEUTRAL, wall_t);
    }

    auto t_end = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // ── Final measurements ────────────────────────────────────────────────────
    const double H_final     = ham.compute(wf);
    const double drift_pct   = (H0 > 0.0) ? (H_final - H0) / H0 * 100.0 : 0.0;
    const auto   snap        = engine.snapshot();

    // ── Sample 5 nodes from the wavefunction ─────────────────────────────────
    const std::size_t SAMPLE = (N_NODES >= 5) ? 5 : N_NODES;
    const float* pr = wf.grid().psi_real();
    const float* pi = wf.grid().psi_imag();
    const float* vr = wf.grid().vel_real();
    const float* vi = wf.grid().vel_imag();

    // ── Emit JSON ─────────────────────────────────────────────────────────────
    std::cout << "{"
              << "\"version\":\"0.0.4\","
              << "\"grid_n\":"    << GRID_N    << ","
              << "\"n_nodes\":"   << N_NODES   << ","
              << "\"steps\":"     << N_STEPS   << ","
              << "\"dt\":"        << DT        << ","
              << "\"c0\":"        << C0        << ","
              << "\"beta\":"      << BETA      << ","
              << "\"h0\":"        << jf(H0)    << ","
              << "\"h_final\":"   << jf(H_final) << ","
              << "\"drift_pct\":" << jf(drift_pct, 6) << ","
              << "\"elapsed_ms\":" << jf(elapsed_ms, 2) << ","
              << "\"dopamine\":"  << jf(snap.dopamine, 4) << ","
              << "\"atp\":"       << jf(snap.atp,      4) << ","
              << "\"boredom\":"   << jf(snap.boredom,  4) << ","
              << "\"psi_sample\":[";

    for (std::size_t i = 0; i < SAMPLE; ++i) {
        if (i > 0) std::cout << ",";
        std::cout << "{"
                  << "\"re\":"  << jf(pr[i], 5) << ","
                  << "\"im\":"  << jf(pi[i], 5) << ","
                  << "\"vr\":"  << jf(vr[i], 5) << ","
                  << "\"vi\":"  << jf(vi[i], 5)
                  << "}";
    }

    std::cout << "]}" << std::endl;
    return 0;
}
