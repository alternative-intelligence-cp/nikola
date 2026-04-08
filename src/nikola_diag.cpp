/**
 * @file src/nikola_diag.cpp
 * @brief nikola-diag — diagnostic CLI for health, benchmarks, and physics checks
 *
 * Modes:
 *   --health         Check build config, dependencies, GPU, memory paths
 *   --benchmark      Quick 1000-tick performance benchmark
 *   --physics-check  Run Standard Candle (energy conservation) + reversibility
 *   --all            Run all checks
 *   --json           JSON output (default: human-readable tables)
 */

#include <nikola/physics/wave_function.hpp>
#include <nikola/physics/hamiltonian.hpp>
#include <nikola/physics/propagator.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/autonomy/hamiltonian_value.hpp>
#include <nikola/security/bootstrap_manager.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <span>
#include <string>
#include <vector>

using namespace nikola;

static constexpr int   GRID_N = 3;
static constexpr float C0     = 1.0f;
static constexpr float BETA   = 1.0f;
static constexpr float DT     = 0.001f;

static std::string jf(double v, int prec = 6) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << v;
    return o.str();
}

// ── Health Check ─────────────────────────────────────────────────────────────

struct HealthResult {
    bool has_ort     = false;
    bool has_cuda    = false;
    bool has_catch2  = true;   // If we're running, tests were built
    bool has_lmdb    = true;   // Header-only, always available
    int  grid_n      = GRID_N;
    int  n_nodes     = 0;
    bool bootstrap_ok = false;
    std::string ort_model_path;
    std::string ort_tokenizer_path;
};

static HealthResult run_health() {
    HealthResult h;
    h.n_nodes = 1;
    for (int i = 0; i < 9; ++i) h.n_nodes *= GRID_N;

#ifdef NIKOLA_HAS_ORT
    h.has_ort = true;
#ifdef NIKOLA_ORT_MODEL_PATH
    h.ort_model_path = NIKOLA_ORT_MODEL_PATH;
#endif
#ifdef NIKOLA_ORT_TOKENIZER_PATH
    h.ort_tokenizer_path = NIKOLA_ORT_TOKENIZER_PATH;
#endif
#endif

#ifdef NIKOLA_HAS_CUDA_KERNELS
    h.has_cuda = true;
#endif

    security::BootstrapManager bootstrap;
    bootstrap.set_silent(true);
    try {
        bootstrap.get_token();
        h.bootstrap_ok = true;
    } catch (...) {
        h.bootstrap_ok = false;
    }

    return h;
}

static void print_health(const HealthResult& h, bool json) {
    if (json) {
        std::cout << "{"
            << "\"check\":\"health\","
            << "\"grid_n\":" << h.grid_n << ","
            << "\"n_nodes\":" << h.n_nodes << ","
            << "\"has_ort\":" << (h.has_ort ? "true" : "false") << ","
            << "\"has_cuda\":" << (h.has_cuda ? "true" : "false") << ","
            << "\"has_lmdb\":" << (h.has_lmdb ? "true" : "false") << ","
            << "\"bootstrap_ok\":" << (h.bootstrap_ok ? "true" : "false") << ","
            << "\"ort_model\":\"" << h.ort_model_path << "\","
            << "\"ort_tokenizer\":\"" << h.ort_tokenizer_path << "\""
            << "}\n";
    } else {
        auto status = [](bool ok) { return ok ? "✓" : "✗"; };
        std::cout
            << "┌─────────────────────────────────────────────┐\n"
            << "│  Nikola Diagnostics — Health Check          │\n"
            << "├─────────────────────────────────────────────┤\n"
            << "│  Grid:         " << h.grid_n << "^9 = " << h.n_nodes << " nodes"
            << std::string(std::max(0, 19 - static_cast<int>(std::to_string(h.n_nodes).size())), ' ') << "│\n"
            << "│  Bootstrap:    " << status(h.bootstrap_ok) << std::string(27, ' ') << "│\n"
            << "│  ONNX Runtime: " << status(h.has_ort) << std::string(27, ' ') << "│\n"
            << "│  CUDA GPU:     " << status(h.has_cuda) << std::string(27, ' ') << "│\n"
            << "│  LMDB:         " << status(h.has_lmdb) << std::string(27, ' ') << "│\n";
        if (h.has_ort && !h.ort_model_path.empty()) {
            bool model_exists = std::filesystem::exists(h.ort_model_path);
            std::cout << "│  ORT model:    " << status(model_exists) << " "
                      << (h.ort_model_path.size() > 24 ? "..." + h.ort_model_path.substr(h.ort_model_path.size()-21) : h.ort_model_path)
                      << std::string(std::max(0, 27 - static_cast<int>(std::min(h.ort_model_path.size(), std::size_t(24)) + 1)), ' ')
                      << "│\n";
        }
        std::cout << "└─────────────────────────────────────────────┘\n";
    }
}

// ── Benchmark ────────────────────────────────────────────────────────────────

struct BenchResult {
    int    ticks      = 0;
    int    steps      = 0;
    double elapsed_ms = 0;
    double ticks_per_sec = 0;
    double steps_per_sec = 0;
    double us_per_step = 0;
};

static BenchResult run_benchmark(int ticks = 1000, int steps_per_tick = 50) {
    physics::WaveFunction wf;
    wf.seed_manifold(GRID_N, 0, 1, 1.0f, 2026u);

    autonomy::AutonomyConfig acfg;
    acfg.initial_atp = 1.0f;
    acfg.enable_boredom = false;
    acfg.enable_dream_weave = false;
    autonomy::AutonomyEngine engine(acfg);

    physics::Propagator propagator;
    propagator.set_beta(BETA).set_c0(C0);

    auto t0 = std::chrono::steady_clock::now();

    for (int t = 0; t < ticks; ++t) {
        for (int s = 0; s < steps_per_tick; ++s) {
            propagator.step(wf, DT);
        }
        const float wall_t = static_cast<float>(t + 1) * static_cast<float>(steps_per_tick) * DT;
        const std::size_t N = wf.num_nodes();
        std::span<const float> psi_r(wf.grid().psi_real(), N);
        std::span<const float> psi_i(wf.grid().psi_imag(), N);
        std::span<const float> vel_r(wf.grid().vel_real(), N);
        std::span<const float> vel_i(wf.grid().vel_imag(), N);
        engine.tick_physics(DT * static_cast<float>(steps_per_tick),
                            psi_r, psi_i, vel_r, vel_i,
                            BETA, autonomy::Reward::NEUTRAL, wall_t);
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    BenchResult r;
    r.ticks = ticks;
    r.steps = ticks * steps_per_tick;
    r.elapsed_ms = ms;
    r.ticks_per_sec = static_cast<double>(ticks) / (ms / 1000.0);
    r.steps_per_sec = static_cast<double>(r.steps) / (ms / 1000.0);
    r.us_per_step = ms * 1000.0 / static_cast<double>(r.steps);
    return r;
}

static void print_benchmark(const BenchResult& r, bool json) {
    if (json) {
        std::cout << "{"
            << "\"check\":\"benchmark\","
            << "\"ticks\":" << r.ticks << ","
            << "\"total_steps\":" << r.steps << ","
            << "\"elapsed_ms\":" << jf(r.elapsed_ms, 2) << ","
            << "\"ticks_per_sec\":" << jf(r.ticks_per_sec, 1) << ","
            << "\"steps_per_sec\":" << jf(r.steps_per_sec, 0) << ","
            << "\"us_per_step\":" << jf(r.us_per_step, 3)
            << "}\n";
    } else {
        std::cout
            << "┌─────────────────────────────────────────────┐\n"
            << "│  Nikola Diagnostics — Benchmark             │\n"
            << "├─────────────────────────────────────────────┤\n"
            << "│  Ticks:          " << std::setw(10) << r.ticks         << std::string(16, ' ') << "│\n"
            << "│  Total steps:    " << std::setw(10) << r.steps         << std::string(16, ' ') << "│\n"
            << "│  Elapsed:        " << std::setw(10) << jf(r.elapsed_ms, 1) << " ms" << std::string(13, ' ') << "│\n"
            << "│  Ticks/sec:      " << std::setw(10) << jf(r.ticks_per_sec, 1)       << std::string(16, ' ') << "│\n"
            << "│  Steps/sec:      " << std::setw(10) << jf(r.steps_per_sec, 0)       << std::string(16, ' ') << "│\n"
            << "│  μs/step:        " << std::setw(10) << jf(r.us_per_step, 3)         << std::string(16, ' ') << "│\n"
            << "└─────────────────────────────────────────────┘\n";
    }
}

// ── Physics Check ────────────────────────────────────────────────────────────

struct PhysicsResult {
    // Standard Candle: energy conservation after N steps
    double h0             = 0;
    double h_final        = 0;
    double drift_pct      = 0;
    bool   energy_ok      = false;  // |drift| < 1%

    // Probability conservation
    double prob_initial   = 0;
    double prob_final     = 0;
    double prob_drift_pct = 0;
    bool   prob_ok        = false;  // |drift| < 0.1%

    // Field integrity
    bool   psi_finite     = false;

    // Reversibility: forward N steps, then backward N steps
    double reversibility_error = 0;
    bool   reversible     = false;  // error < 1e-3

    bool   all_ok         = false;
};

static PhysicsResult run_physics_check() {
    PhysicsResult r;
    constexpr int N_STEPS = 1000;

    physics::WaveFunction wf;
    wf.seed_manifold(GRID_N, 0, 1, 1.0f, 2026u);

    physics::Hamiltonian ham;
    ham.set_c0(C0).set_beta(BETA);
    ham.verify_initial_conditions(wf);

    physics::Propagator propagator;
    propagator.set_beta(BETA).set_c0(C0);

    r.h0 = ham.compute(wf);
    r.prob_initial = wf.total_probability();

    // Save initial state for reversibility check
    auto wf_backup = wf.clone();

    // Forward propagation
    for (int i = 0; i < N_STEPS; ++i) {
        propagator.step(wf, DT);
    }

    r.h_final    = ham.compute(wf);
    r.prob_final = wf.total_probability();
    r.psi_finite = wf.is_finite();

    r.drift_pct = (r.h0 > 0) ? (r.h_final - r.h0) / r.h0 * 100.0 : 0.0;
    r.energy_ok = std::abs(r.drift_pct) < 1.0;

    r.prob_drift_pct = (r.prob_initial > 0)
        ? (r.prob_final - r.prob_initial) / r.prob_initial * 100.0 : 0.0;
    r.prob_ok = std::abs(r.prob_drift_pct) < 0.1;

    // Reverse propagation
    for (int i = 0; i < N_STEPS; ++i) {
        propagator.step(wf, -DT);
    }

    // Measure reversibility error = ||Ψ_reversed - Ψ_initial|| / ||Ψ_initial||
    const float* pr = wf.grid().psi_real();
    const float* pi = wf.grid().psi_imag();
    const float* br = wf_backup.grid().psi_real();
    const float* bi = wf_backup.grid().psi_imag();
    const std::size_t N = wf.num_nodes();

    double err2 = 0, norm2 = 0;
    for (std::size_t i = 0; i < N; ++i) {
        double dr = static_cast<double>(pr[i]) - br[i];
        double di = static_cast<double>(pi[i]) - bi[i];
        err2 += dr * dr + di * di;
        norm2 += static_cast<double>(br[i]) * br[i]
               + static_cast<double>(bi[i]) * bi[i];
    }
    r.reversibility_error = (norm2 > 0) ? std::sqrt(err2 / norm2) : 0.0;
    r.reversible = r.reversibility_error < 1e-3;

    r.all_ok = r.energy_ok && r.prob_ok && r.psi_finite && r.reversible;
    return r;
}

static void print_physics(const PhysicsResult& r, bool json) {
    if (json) {
        std::cout << "{"
            << "\"check\":\"physics\","
            << "\"h0\":" << jf(r.h0) << ","
            << "\"h_final\":" << jf(r.h_final) << ","
            << "\"drift_pct\":" << jf(r.drift_pct) << ","
            << "\"energy_ok\":" << (r.energy_ok ? "true" : "false") << ","
            << "\"prob_initial\":" << jf(r.prob_initial) << ","
            << "\"prob_final\":" << jf(r.prob_final) << ","
            << "\"prob_drift_pct\":" << jf(r.prob_drift_pct) << ","
            << "\"prob_ok\":" << (r.prob_ok ? "true" : "false") << ","
            << "\"psi_finite\":" << (r.psi_finite ? "true" : "false") << ","
            << "\"reversibility_error\":" << jf(r.reversibility_error, 8) << ","
            << "\"reversible\":" << (r.reversible ? "true" : "false") << ","
            << "\"all_ok\":" << (r.all_ok ? "true" : "false")
            << "}\n";
    } else {
        auto s = [](bool ok) { return ok ? "✓ PASS" : "✗ FAIL"; };
        std::cout
            << "┌─────────────────────────────────────────────┐\n"
            << "│  Nikola Diagnostics — Physics Check         │\n"
            << "├─────────────────────────────────────────────┤\n"
            << "│  Standard Candle (1000 steps):              │\n"
            << "│    H₀:        " << std::setw(16) << jf(r.h0)      << std::string(13, ' ') << "│\n"
            << "│    H_final:   " << std::setw(16) << jf(r.h_final) << std::string(13, ' ') << "│\n"
            << "│    Drift:     " << std::setw(16) << (jf(r.drift_pct) + "%") << "  " << s(r.energy_ok) << std::string(5, ' ') << "│\n"
            << "├─────────────────────────────────────────────┤\n"
            << "│  Probability Conservation:                  │\n"
            << "│    |Ψ|² init: " << std::setw(16) << jf(r.prob_initial) << std::string(13, ' ') << "│\n"
            << "│    |Ψ|² fin:  " << std::setw(16) << jf(r.prob_final)   << std::string(13, ' ') << "│\n"
            << "│    Drift:     " << std::setw(16) << (jf(r.prob_drift_pct) + "%") << "  " << s(r.prob_ok) << std::string(5, ' ') << "│\n"
            << "├─────────────────────────────────────────────┤\n"
            << "│  Field integrity: Ψ finite  " << s(r.psi_finite) << std::string(8, ' ') << "│\n"
            << "├─────────────────────────────────────────────┤\n"
            << "│  Reversibility (fwd+bwd):                   │\n"
            << "│    Error:     " << std::setw(16) << jf(r.reversibility_error, 8) << "  " << s(r.reversible) << std::string(5, ' ') << "│\n"
            << "├─────────────────────────────────────────────┤\n"
            << "│  Overall:     " << s(r.all_ok) << std::string(23, ' ') << "│\n"
            << "└─────────────────────────────────────────────┘\n";
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

static void usage() {
    std::cerr << "Usage: nikola-diag [OPTIONS]\n"
              << "  --health         Check build config, dependencies, GPU\n"
              << "  --benchmark      Quick 1000-tick performance benchmark\n"
              << "  --physics-check  Standard Candle + reversibility\n"
              << "  --all            Run all checks\n"
              << "  --json           JSON output\n"
              << "  --help           Show this message\n";
}

int main(int argc, char* argv[]) {
    bool do_health  = false;
    bool do_bench   = false;
    bool do_physics = false;
    bool json       = false;

    if (argc < 2) { usage(); return 1; }

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--health") == 0)         do_health  = true;
        else if (std::strcmp(argv[i], "--benchmark") == 0)  do_bench   = true;
        else if (std::strcmp(argv[i], "--physics-check") == 0) do_physics = true;
        else if (std::strcmp(argv[i], "--all") == 0) {
            do_health = do_bench = do_physics = true;
        }
        else if (std::strcmp(argv[i], "--json") == 0)      json = true;
        else if (std::strcmp(argv[i], "--help") == 0) { usage(); return 0; }
        else { std::cerr << "Unknown: " << argv[i] << "\n"; usage(); return 1; }
    }

    if (!do_health && !do_bench && !do_physics) {
        std::cerr << "No check specified. Use --health, --benchmark, --physics-check, or --all\n";
        return 1;
    }

    if (do_health) {
        auto h = run_health();
        print_health(h, json);
    }

    if (do_bench) {
        auto b = run_benchmark();
        print_benchmark(b, json);
    }

    if (do_physics) {
        auto p = run_physics_check();
        print_physics(p, json);
    }

    return 0;
}
