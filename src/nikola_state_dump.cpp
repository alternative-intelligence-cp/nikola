/**
 * @file src/nikola_state_dump.cpp
 * @brief nikola-state-dump — comprehensive system state inspector
 *
 * Boots CognitiveTorus + full autonomy, runs N ticks, then dumps:
 *   - Physics: Hamiltonian, drift, total probability, kinetic energy
 *   - Ψ field: max amplitude, mean curvature, 2D projected heatmap
 *   - Metabolic: ATP, dopamine, serotonin, norepinephrine, boredom, entropy
 *   - Memory: LMDB record count (if --memory-lmdb given)
 *   - Psi sample: top-k hottest nodes
 *
 * Modes:
 *   --json         JSON output (default: human-readable table)
 *   --memory-lmdb  Path to LMDB memory database
 *   --ticks N      Number of ticks to run (default: 100)
 *   --steps N      Physics steps per tick (default: 50)
 *   --help         Show usage
 */

#include <nikola/physics/wave_function.hpp>
#include <nikola/physics/hamiltonian.hpp>
#include <nikola/physics/propagator.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/autonomy/hamiltonian_value.hpp>
#include <nikola/cognitive/semantic_memory.hpp>
#include <nikola/cognitive/lmdb_memory_store.hpp>
#include <nikola/security/bootstrap_manager.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <vector>

// ── Physics constants ───────────────────────────────────────────────────────
static constexpr int   GRID_N = 3;
static constexpr float C0     = 1.0f;
static constexpr float BETA   = 1.0f;
static constexpr float DT     = 0.001f;

// ── Helpers ─────────────────────────────────────────────────────────────────

static std::string jf(double v, int prec = 6) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << v;
    return o.str();
}

static std::string escape_json(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

// ── Ψ heatmap: project 3^9 nodes onto 27×27 grid (dims 0,1 vs dims 2,3) ──

static void print_psi_heatmap(const nikola::physics::WaveFunction& wf,
                              std::ostream& out) {
    const std::size_t N = wf.num_nodes();
    const float* pr = wf.grid().psi_real();
    const float* pi = wf.grid().psi_imag();

    // Project onto first 4 dims: row = dim0*3+dim1 (0..8), col = dim2*3+dim3 (0..8)
    // Each cell accumulates |ψ|²
    constexpr int PROJ = 9;  // 3^2
    std::vector<double> heat(PROJ * PROJ, 0.0);

    for (std::size_t n = 0; n < N; ++n) {
        double amp2 = static_cast<double>(pr[n]) * pr[n]
                    + static_cast<double>(pi[n]) * pi[n];
        // Decompose mixed-radix index: digit[d] = (n / 3^d) % 3
        std::size_t tmp = n;
        int d[9];
        for (int k = 0; k < 9; ++k) { d[k] = static_cast<int>(tmp % 3); tmp /= 3; }
        int row = d[0] * 3 + d[1];
        int col = d[2] * 3 + d[3];
        heat[static_cast<std::size_t>(row * PROJ + col)] += amp2;
    }

    // Find max for normalization
    double mx = 0.0;
    for (double v : heat) mx = std::max(mx, v);
    if (mx < 1e-30) mx = 1.0;

    // Render with block characters
    const char* blocks[] = {" ", "░", "▒", "▓", "█"};
    out << "  Ψ field (dims 0,1 × 2,3 projection, |ψ|² heatmap):\n";
    for (int r = 0; r < PROJ; ++r) {
        out << "    ";
        for (int c = 0; c < PROJ; ++c) {
            double norm = heat[static_cast<std::size_t>(r * PROJ + c)] / mx;
            int level = std::min(4, static_cast<int>(norm * 5.0));
            out << blocks[level] << blocks[level];
        }
        out << "\n";
    }
}

struct Config {
    int ticks         = 100;
    int steps         = 50;
    bool json_mode    = false;
    std::string lmdb_path;
};

static void usage() {
    std::cerr << "Usage: nikola-state-dump [OPTIONS]\n"
              << "  --json              JSON output (default: human-readable)\n"
              << "  --memory-lmdb PATH  LMDB memory database to inspect\n"
              << "  --ticks N           Ticks to run (default: 100)\n"
              << "  --steps N           Physics steps per tick (default: 50)\n"
              << "  --help              Show this message\n";
}

int main(int argc, char* argv[]) {
    using namespace nikola;

    Config cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0)
            cfg.json_mode = true;
        else if (std::strcmp(argv[i], "--memory-lmdb") == 0 && i + 1 < argc)
            cfg.lmdb_path = argv[++i];
        else if (std::strcmp(argv[i], "--ticks") == 0 && i + 1 < argc)
            cfg.ticks = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
            cfg.steps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--help") == 0) {
            usage(); return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            usage(); return 1;
        }
    }

    // ── Bootstrap ────────────────────────────────────────────────────────────
    security::BootstrapManager bootstrap;
    bootstrap.set_silent(true);
    bootstrap.get_token();

    // ── Wave function ────────────────────────────────────────────────────────
    physics::WaveFunction wf;
    wf.seed_manifold(GRID_N, 0, 1, 1.0f, 2026u);
    const std::size_t N_NODES = wf.num_nodes();

    physics::Hamiltonian ham;
    ham.set_c0(C0).set_beta(BETA);
    ham.verify_initial_conditions(wf);
    const double H0 = ham.compute(wf);

    // ── Autonomy ─────────────────────────────────────────────────────────────
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

    physics::Propagator propagator;
    propagator.set_beta(BETA).set_c0(C0);

    // ── Run physics ──────────────────────────────────────────────────────────
    auto t_start = std::chrono::steady_clock::now();

    const int total_steps = cfg.ticks * cfg.steps;
    for (int step = 1; step <= total_steps; ++step) {
        propagator.step(wf, DT);

        // Tick autonomy once per cfg.steps physics steps
        if (step % cfg.steps == 0) {
            const float wall_t = static_cast<float>(step) * DT;
            const std::size_t N = wf.num_nodes();
            std::span<const float> psi_r(wf.grid().psi_real(), N);
            std::span<const float> psi_i(wf.grid().psi_imag(), N);
            std::span<const float> vel_r(wf.grid().vel_real(), N);
            std::span<const float> vel_i(wf.grid().vel_imag(), N);
            engine.tick_physics(DT * static_cast<float>(cfg.steps),
                                psi_r, psi_i, vel_r, vel_i,
                                BETA, autonomy::Reward::NEUTRAL, wall_t);
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // ── Measurements ─────────────────────────────────────────────────────────
    const double H_final   = ham.compute(wf);
    const double drift_pct = (H0 > 0.0) ? (H_final - H0) / H0 * 100.0 : 0.0;
    const double total_p   = wf.total_probability();
    const double kinetic   = wf.total_kinetic_energy();
    const float  max_amp   = wf.max_amplitude();
    const double curvature = wf.mean_curvature();
    const auto   snap      = engine.snapshot();
    const float  serotonin = engine.serotonin();
    const float  norepinephrine = engine.norepinephrine();

    // ── Memory stats (optional) ──────────────────────────────────────────────
    std::size_t memory_count = 0;
    float avg_strength = 0.0f;
    if (!cfg.lmdb_path.empty()) {
        cognitive::SemanticMemory mem(2);
        cognitive::LmdbMemoryStore store(cfg.lmdb_path);
        memory_count = store.load_all(mem);
        if (memory_count > 0) {
            double total_str = 0.0;
            for (auto key : mem.all_keys()) {
                const auto* rec = mem.get(key);
                if (rec) total_str += rec->strength;
            }
            avg_strength = static_cast<float>(total_str / static_cast<double>(memory_count));
        }
    }

    // ── Top-5 hottest nodes ──────────────────────────────────────────────────
    const float* pr = wf.grid().psi_real();
    const float* pi = wf.grid().psi_imag();
    const float* vr = wf.grid().vel_real();
    const float* vi = wf.grid().vel_imag();

    struct HotNode { std::size_t idx; double amp2; };
    std::vector<HotNode> nodes(N_NODES);
    for (std::size_t i = 0; i < N_NODES; ++i) {
        nodes[i] = {i, static_cast<double>(pr[i])*pr[i]
                     + static_cast<double>(pi[i])*pi[i]};
    }
    std::partial_sort(nodes.begin(), nodes.begin() + std::min<std::size_t>(5, N_NODES),
                      nodes.end(), [](const HotNode& a, const HotNode& b) {
                          return a.amp2 > b.amp2;
                      });
    const std::size_t SAMPLE = std::min<std::size_t>(5, N_NODES);

    // ── Output ───────────────────────────────────────────────────────────────
    if (cfg.json_mode) {
        std::cout << "{"
            << "\"version\":\"0.0.12\","
            << "\"grid_n\":" << GRID_N << ","
            << "\"n_nodes\":" << N_NODES << ","
            << "\"ticks\":" << cfg.ticks << ","
            << "\"steps_per_tick\":" << cfg.steps << ","
            << "\"dt\":" << DT << ","
            << "\"physics\":{"
                << "\"h0\":" << jf(H0) << ","
                << "\"h_final\":" << jf(H_final) << ","
                << "\"drift_pct\":" << jf(drift_pct) << ","
                << "\"total_probability\":" << jf(total_p) << ","
                << "\"kinetic_energy\":" << jf(kinetic) << ","
                << "\"max_amplitude\":" << jf(max_amp) << ","
                << "\"mean_curvature\":" << jf(curvature)
            << "},"
            << "\"metabolic\":{"
                << "\"atp\":" << jf(snap.atp, 4) << ","
                << "\"dopamine\":" << jf(snap.dopamine, 4) << ","
                << "\"serotonin\":" << jf(serotonin, 4) << ","
                << "\"norepinephrine\":" << jf(norepinephrine, 4) << ","
                << "\"boredom\":" << jf(snap.boredom, 4) << ","
                << "\"entropy\":" << jf(snap.entropy, 4) << ","
                << "\"state\":\"" << static_cast<int>(snap.state) << "\","
                << "\"nap_count\":" << snap.nap_count
            << "},"
            << "\"memory\":{"
                << "\"lmdb_path\":\"" << escape_json(cfg.lmdb_path) << "\","
                << "\"record_count\":" << memory_count << ","
                << "\"avg_strength\":" << jf(avg_strength, 4)
            << "},"
            << "\"elapsed_ms\":" << jf(elapsed_ms, 2) << ","
            << "\"hot_nodes\":[";

        for (std::size_t i = 0; i < SAMPLE; ++i) {
            auto idx = nodes[i].idx;
            if (i > 0) std::cout << ",";
            std::cout << "{"
                << "\"index\":" << idx << ","
                << "\"amp2\":" << jf(nodes[i].amp2, 8) << ","
                << "\"re\":" << jf(pr[idx], 6) << ","
                << "\"im\":" << jf(pi[idx], 6) << ","
                << "\"vr\":" << jf(vr[idx], 6) << ","
                << "\"vi\":" << jf(vi[idx], 6)
                << "}";
        }
        std::cout << "]}" << std::endl;

    } else {
        // ── Human-readable table output ──────────────────────────────────────
        auto& o = std::cout;
        o << "╔══════════════════════════════════════════════════════════╗\n"
          << "║          Nikola State Dump — v0.0.12                    ║\n"
          << "╠══════════════════════════════════════════════════════════╣\n";

        o << "║  Grid:  " << GRID_N << "^9 = " << N_NODES << " nodes"
          << std::string(39 - std::to_string(N_NODES).size(), ' ') << "║\n"
          << "║  Ticks: " << cfg.ticks << "  Steps/tick: " << cfg.steps
          << "  dt: " << DT << std::string(22, ' ') << "║\n"
          << "║  Elapsed: " << jf(elapsed_ms, 1) << " ms"
          << std::string(std::max(0, 43 - static_cast<int>(jf(elapsed_ms,1).size())), ' ')
          << "║\n";

        o << "╠══════════════════════════════════════════════════════════╣\n"
          << "║  PHYSICS                                                ║\n"
          << "╠──────────────────────────────────────────────────────────╣\n"
          << "║  H₀:          " << std::setw(16) << jf(H0)        << std::string(26, ' ') << "║\n"
          << "║  H_final:     " << std::setw(16) << jf(H_final)   << std::string(26, ' ') << "║\n"
          << "║  Drift:       " << std::setw(16) << jf(drift_pct) << " %"
          << std::string(24, ' ') << "║\n"
          << "║  Total |Ψ|²:  " << std::setw(16) << jf(total_p)   << std::string(26, ' ') << "║\n"
          << "║  Kinetic E:   " << std::setw(16) << jf(kinetic)   << std::string(26, ' ') << "║\n"
          << "║  Max |Ψ|:     " << std::setw(16) << jf(max_amp)   << std::string(26, ' ') << "║\n"
          << "║  Curvature:   " << std::setw(16) << jf(curvature) << std::string(26, ' ') << "║\n";

        o << "╠══════════════════════════════════════════════════════════╣\n"
          << "║  METABOLIC                                              ║\n"
          << "╠──────────────────────────────────────────────────────────╣\n"
          << "║  ATP:              " << std::setw(8) << jf(snap.atp, 4)              << std::string(30, ' ') << "║\n"
          << "║  Dopamine:         " << std::setw(8) << jf(snap.dopamine, 4)         << std::string(30, ' ') << "║\n"
          << "║  Serotonin:        " << std::setw(8) << jf(serotonin, 4)             << std::string(30, ' ') << "║\n"
          << "║  Norepinephrine:   " << std::setw(8) << jf(norepinephrine, 4)        << std::string(30, ' ') << "║\n"
          << "║  Boredom:          " << std::setw(8) << jf(snap.boredom, 4)          << std::string(30, ' ') << "║\n"
          << "║  Entropy:          " << std::setw(8) << jf(snap.entropy, 4)          << std::string(30, ' ') << "║\n"
          << "║  Nap count:        " << std::setw(8) << snap.nap_count               << std::string(30, ' ') << "║\n";

        if (!cfg.lmdb_path.empty()) {
            o << "╠══════════════════════════════════════════════════════════╣\n"
              << "║  MEMORY (LMDB)                                         ║\n"
              << "╠──────────────────────────────────────────────────────────╣\n"
              << "║  Records:          " << std::setw(8) << memory_count     << std::string(30, ' ') << "║\n"
              << "║  Avg strength:     " << std::setw(8) << jf(avg_strength, 4) << std::string(30, ' ') << "║\n";
        }

        o << "╠══════════════════════════════════════════════════════════╣\n"
          << "║  Ψ FIELD                                                ║\n"
          << "╠──────────────────────────────────────────────────────────╣\n";
        print_psi_heatmap(wf, o);

        o << "╠──────────────────────────────────────────────────────────╣\n"
          << "║  HOT NODES (top-5 by |Ψ|²)                             ║\n"
          << "╠──────────────────────────────────────────────────────────╣\n";
        for (std::size_t i = 0; i < SAMPLE; ++i) {
            auto idx = nodes[i].idx;
            o << "║  [" << std::setw(5) << idx << "] |Ψ|²="
              << std::setw(12) << jf(nodes[i].amp2, 8)
              << "  ψ=(" << jf(pr[idx], 4) << "," << jf(pi[idx], 4) << ")"
              << std::string(5, ' ') << "║\n";
        }

        o << "╚══════════════════════════════════════════════════════════╝\n";
    }

    return 0;
}
