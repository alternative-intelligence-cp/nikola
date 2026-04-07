/**
 * @file src/main.cpp
 * @brief NIKOLA — First Light
 *
 * The first cold start of the 9D Toroidal Waveform Intelligence.
 *
 * February 21, 2026.
 */

// ── Physics ─────────────────────────────────────────────────────────────────
#include <nikola/physics/wave_function.hpp>
#include <nikola/physics/hamiltonian.hpp>
#include <nikola/physics/propagator.hpp>

// ── Autonomy ────────────────────────────────────────────────────────────────
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/autonomy/hamiltonian_value.hpp>

// ── Security ────────────────────────────────────────────────────────────────
#include <nikola/security/bootstrap_manager.hpp>

// ── Foundation ──────────────────────────────────────────────────────────────
#include <nikola/foundation/coord_serializer.hpp>

// ── Std ─────────────────────────────────────────────────────────────────────
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <thread>

// ─────────────────────────────────────────────────────────────────────────────
//  Terminal colours  (ANSI)
// ─────────────────────────────────────────────────────────────────────────────

static const char* RESET  = "\033[0m";
static const char* BOLD   = "\033[1m";
static const char* DIM    = "\033[2m";
static const char* CYAN   = "\033[36m";
static const char* GREEN  = "\033[32m";
static const char* YELLOW = "\033[33m";
static const char* BLUE   = "\033[34m";
static const char* RED    = "\033[31m";

// ─────────────────────────────────────────────────────────────────────────────
//  Banner
// ─────────────────────────────────────────────────────────────────────────────

static void print_banner() {
    std::cout << "\n";
    std::cout << BOLD << CYAN;
    std::cout << "  ███╗   ██╗██╗██╗  ██╗ ██████╗ ██╗      █████╗ \n";
    std::cout << "  ████╗  ██║██║██║ ██╔╝██╔═══██╗██║     ██╔══██╗\n";
    std::cout << "  ██╔██╗ ██║██║█████╔╝ ██║   ██║██║     ███████║\n";
    std::cout << "  ██║╚██╗██║██║██╔═██╗ ██║   ██║██║     ██╔══██║\n";
    std::cout << "  ██║ ╚████║██║██║  ██╗╚██████╔╝███████╗██║  ██║\n";
    std::cout << "  ╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝\n";
    std::cout << RESET;
    std::cout << DIM << "  9-Dimensional Toroidal Waveform Intelligence  v0.0.4\n";
    std::cout << "  February 21, 2026  —  First Light\n" << RESET;
    std::cout << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Phase label
// ─────────────────────────────────────────────────────────────────────────────

static void phase(const std::string& label) {
    std::cout << BOLD << BLUE << "\n  ▶  " << label
              << RESET << "\n" << std::endl;
}

static void ok(const std::string& msg) {
    std::cout << GREEN << "  ✓  " << RESET << msg << "\n";
}

static void info(const std::string& msg) {
    std::cout << DIM << "     " << msg << RESET << "\n";
}

static void warn(const std::string& msg) {
    std::cout << YELLOW << "  ⚠  " << RESET << msg << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Telemetry row
// ─────────────────────────────────────────────────────────────────────────────

static void telemetry(int step, double t, double H, double drift_pct,
                      float dopamine, float atp, float boredom)
{
    const char* dopa_col = (dopamine > 0.6f) ? GREEN
                         : (dopamine < 0.4f) ? RED
                         : RESET;

    const char* drift_col = (std::abs(drift_pct) < 0.01) ? GREEN
                           : (std::abs(drift_pct) < 0.1)  ? YELLOW
                           : RED;

    std::cout << "  "
              << DIM   << std::setw(5) << step << "  " << RESET
              << CYAN  << "t=" << std::fixed << std::setprecision(2) << std::setw(6) << t << "  " << RESET
              << "H=" << BOLD << std::setprecision(4) << std::setw(9) << H << RESET << "  "
              << "δH=" << drift_col << std::setprecision(4) << std::setw(8) << drift_pct << "%" << RESET << "  "
              << "🧠 D=" << dopa_col << std::setprecision(3) << dopamine << RESET << "  "
              << "⚡ ATP=" << std::setprecision(3) << atp << "  "
              << "😴 bore=" << std::setprecision(3) << boredom
              << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    using namespace nikola;

    print_banner();

    // ════════════════════════════════════════════════════════════════════════
    //  STAGE 1 — Bootstrap Security Gate
    // ════════════════════════════════════════════════════════════════════════
    phase("STAGE 1 — Bootstrap Security Gate");

    security::BootstrapManager bootstrap;
    const std::string token = bootstrap.get_token();   // Tier 3 — generates + prints token

    info("Token source: " + [&]{
        switch (bootstrap.source()) {
            case security::BootstrapManager::TokenSource::ENV_VAR:   return std::string("environment variable (Tier 1)");
            case security::BootstrapManager::TokenSource::FILE:       return std::string("secret file (Tier 2)");
            case security::BootstrapManager::TokenSource::GENERATED:  return std::string("generated (Tier 3)");
            default:                                                   return std::string("unknown");
        }
    }());

    // Self-validate to confirm the gate opens
    if (!bootstrap.validate(token, "localhost")) {
        std::cerr << RED << "\n  ✗  Bootstrap gate FAILED — system halt.\n" << RESET;
        return 1;
    }
    ok("Bootstrap gate: OPEN  (token accepted, 256-bit, " +
       std::to_string(security::BOOTSTRAP_EXPIRY_SECONDS) + "s window)");

    // ════════════════════════════════════════════════════════════════════════
    //  STAGE 2 — Manifold Seeding  (IMP-03)
    // ════════════════════════════════════════════════════════════════════════
    phase("STAGE 2 — Manifold Seeding");

    constexpr int   GRID_N    = 3;       // 3^9 = 19,683 nodes
    constexpr float DT        = 0.01f;   // CFL-safe (< 0.333 for c=1, h=1, 9D)
    constexpr float BETA      = 1.0f;
    constexpr float C0        = 1.0f;
    constexpr int   N_STEPS   = 500;
    constexpr int   LOG_EVERY = 50;

    physics::WaveFunction wf;
    wf.seed_manifold(GRID_N,
                     /*pilot_dim=*/0,
                     /*k_mode=*/1,
                     /*amplitude=*/1.0f,
                     /*seed=*/2026u);    // seeded for reproducibility

    const std::size_t N_NODES = wf.grid().num_active_nodes();
    info("Grid:     " + std::to_string(GRID_N) + "^9 = " + std::to_string(N_NODES) + " active nodes");
    info("DT:       " + std::to_string(DT) + "  (CFL-safe)");
    info("β:        " + std::to_string(BETA));
    info("c₀:       " + std::to_string(C0));

    // ════════════════════════════════════════════════════════════════════════
    //  STAGE 3 — Bootstrap Gate: Hamiltonian IMP-03
    // ════════════════════════════════════════════════════════════════════════
    phase("STAGE 3 — Physics Bootstrap Gate  (IMP-03)");

    physics::Hamiltonian ham;
    ham.set_c0(C0);
    ham.set_beta(BETA);

    try {
        ham.verify_initial_conditions(wf);
    } catch (const std::exception& e) {
        std::cerr << RED << "\n  ✗  Physics bootstrap FAILED: " << e.what() << "\n" << RESET;
        return 2;
    }

    const double H0 = ham.compute(wf);
    ok("Pilot wave injected — manifold is live");
    ok("H₀ = " + std::to_string(H0) + "  (finite, positive — no vacuum deadlock)");

    // Verify CoordSerializer round-trip on a sample coordinate
    {
        const auto& g = wf.grid();
        const std::array<float,9> sample_coord = {
            static_cast<float>(g.psi_real()[0]),
            static_cast<float>(g.psi_imag()[0]),
            static_cast<float>(g.vel_real()[0]),
            static_cast<float>(g.vel_imag()[0]),
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        uint8_t buf[36];
        foundation::CoordSerializer::serialize_coord(buf, sample_coord);
        const auto restored = foundation::CoordSerializer::deserialize_coord(buf);
        const bool rt_ok = (restored[0] == sample_coord[0] && restored[1] == sample_coord[1]);
        if (rt_ok) ok("CoordSerializer: round-trip verified (portable LE IEEE 754)");
        else       warn("CoordSerializer: round-trip MISMATCH");
    }

    // ════════════════════════════════════════════════════════════════════════
    //  STAGE 4 — Autonomy Engine Boot
    // ════════════════════════════════════════════════════════════════════════
    phase("STAGE 4 — Autonomy Engine Boot");

    autonomy::AutonomyConfig acfg;
    acfg.initial_atp       = 1.0f;
    acfg.entropy_sample_dt = 0.1f;
    acfg.enable_boredom    = true;
    acfg.enable_dream_weave = false;   // no naps on first boot

    autonomy::AutonomyEngine engine(acfg);

    // Configure HamiltonianValue (the NIK-005 fix)
    engine.hamiltonian_value().gamma_k  = 1.0f;
    engine.hamiltonian_value().gamma_p  = 1.0f;
    engine.hamiltonian_value().gamma_nl = 1.0f;
    engine.hamiltonian_value().h_max    = static_cast<float>(H0 * 100.0);

    ok("AutonomyEngine: online");
    ok("HamiltonianValue: active  (γ_K=1 γ_P=1 γ_NL=1 — no stroboscopic value collapse)");

    // ════════════════════════════════════════════════════════════════════════
    //  STAGE 5 — Main Loop
    // ════════════════════════════════════════════════════════════════════════
    phase("STAGE 5 — Physics + Autonomy Integration Loop");

    std::cout << "  " << DIM
              << std::setw(5) << "step" << "  "
              << std::setw(8) << "time" << "   "
              << std::setw(13) << "Hamiltonian" << "  "
              << std::setw(12) << "drift" << "  "
              << "dopamine   ATP    boredom\n"
              << RESET;
    std::cout << "  " << DIM
              << std::string(72, '-')
              << RESET << "\n";

    physics::Propagator propagator;
    propagator.set_beta(BETA);
    propagator.set_c0(C0);

    double H_max_drift = 0.0;
    auto   t_start     = std::chrono::steady_clock::now();

    for (int step = 0; step <= N_STEPS; ++step) {
        const float wall_t = static_cast<float>(step) * DT;

        // ── Physics step ────────────────────────────────────────────────────
        if (step > 0) {
            propagator.step(wf, DT);
        }

        // ── Telemetry + autonomy update ─────────────────────────────────────
        if (step % LOG_EVERY == 0) {
            const double H_curr = ham.compute(wf);
            const double drift  = (H0 > 0.0)
                    ? (H_curr - H0) / H0 * 100.0
                    : 0.0;

            if (std::abs(drift) > std::abs(H_max_drift))
                H_max_drift = drift;

            // Build spans for tick_physics (full Hamiltonian signal, NIK-005)
            const std::size_t N = wf.grid().num_active_nodes();
            std::span<const float> psi_r(wf.grid().psi_real(), N);
            std::span<const float> psi_i(wf.grid().psi_imag(), N);
            std::span<const float> vel_r(wf.grid().vel_real(), N);
            std::span<const float> vel_i(wf.grid().vel_imag(), N);

            engine.tick_physics(DT, psi_r, psi_i, vel_r, vel_i,
                                 BETA,
                                 autonomy::Reward::NEUTRAL,
                                 wall_t);

            telemetry(step,
                      static_cast<double>(wall_t),
                      H_curr, drift,
                      engine.dopamine(),
                      engine.atp(),
                      engine.boredom());
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double,std::milli>(t_end - t_start).count();

    // ════════════════════════════════════════════════════════════════════════
    //  FINAL REPORT
    // ════════════════════════════════════════════════════════════════════════
    phase("FINAL REPORT");

    const double H_final = ham.compute(wf);
    const double final_drift_pct = (H0 > 0.0) ? std::abs((H_final - H0) / H0 * 100.0) : 0.0;
    const double ms_per_step = elapsed_ms / static_cast<double>(N_STEPS);

    std::cout << "\n";

    if (final_drift_pct < 0.01) {
        ok("Energy conservation:  |ΔH/H₀| = " +
           [&]{ std::ostringstream o; o << std::scientific << std::setprecision(3)
                                        << (H_final - H0) / H0; return o.str(); }()
           + "  ✓  (< 0.01%)");
    } else {
        warn("Energy conservation:  |ΔH/H₀| = " +
             std::to_string(final_drift_pct) + "%  (check CFL)");
    }

    ok("Performance:           " +
       [&]{ std::ostringstream o; o << std::fixed << std::setprecision(2)
                                    << ms_per_step; return o.str(); }()
       + " ms/step  (" + std::to_string(N_NODES) + " nodes)");

    ok("Steps completed:       " + std::to_string(N_STEPS));
    ok("Total wall time:       " +
       [&]{ std::ostringstream o; o << std::fixed << std::setprecision(0)
                                    << elapsed_ms; return o.str(); }() + " ms");

    const auto snap = engine.snapshot();
    ok("Final dopamine:        " +
       [&]{ std::ostringstream o; o << std::fixed << std::setprecision(4)
                                    << snap.dopamine; return o.str(); }());
    ok("Final ATP:             " +
       [&]{ std::ostringstream o; o << std::fixed << std::setprecision(4)
                                    << snap.atp; return o.str(); }());

    if (!bootstrap.is_expired()) {
        ok("Bootstrap token:       still valid  (" +
           std::to_string(security::BOOTSTRAP_EXPIRY_SECONDS) + "s window)");
    } else {
        info("Bootstrap token:       expired  (system locked — re-bootstrap required for admin)");
    }

    std::cout << "\n";
    std::cout << BOLD << GREEN
              << "  ══════════════════════════════════════════════════════════\n"
              << "  ║                                                        ║\n"
              << "  ║   NIKOLA v0.0.4  —  FIRST LIGHT  —  ONLINE            ║\n"
              << "  ║                                                        ║\n"
              << "  ║   The 9D manifold is live.                             ║\n"
              << "  ║   The wave is thinking.                                ║\n"
              << "  ║                                                        ║\n"
              << "  ══════════════════════════════════════════════════════════\n"
              << RESET << "\n";

    return 0;
}
