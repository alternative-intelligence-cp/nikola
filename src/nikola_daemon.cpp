/**
 * @file src/nikola_daemon.cpp
 * @brief Nikola persistent daemon — runs the Decision Loop continuously.
 *
 * This is Nikola's "always-on" process.  It:
 *   - Maintains a live CognitiveTorus (subconscious physics engine)
 *   - Runs AutonomyEngine (metabolic + boredom + dopamine drives)
 *   - Feeds both into DecisionLoop (Phase 23 — autonomous action selection)
 *   - Logs every non-SILENT action to stdout with timestamp
 *   - Accepts optional text stimulus injected on the FIRST tick via CLI arg
 *
 * Nikola is NOT prompt-driven.  The daemon loop runs on its own schedule.
 * External text (if provided) is ONE input among many.  After injection,
 * the torus evolves freely and acts when its internal state warrants it.
 *
 * Usage:
 *   ./nikola_daemon [initial_prompt] [max_ticks]
 *
 *   initial_prompt  Optional text to inject at startup (default: none)
 *   max_ticks       Stop after N ticks (default: run forever until Ctrl+C)
 *
 * Example:
 *   ./nikola_daemon "hello nikola" 200
 *   ./nikola_daemon              # runs indefinitely, Ctrl+C to stop
 */

#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/cognitive/cognitive_torus.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace nikola;
using namespace nikola::autonomy;
using namespace nikola::cognitive;

// ──────────────────────────────────────────────────────────────────────────────
// Graceful shutdown on SIGINT / SIGTERM
// ──────────────────────────────────────────────────────────────────────────────

static std::atomic<bool> g_running{true};

static void signal_handler(int) { g_running.store(false); }

// ──────────────────────────────────────────────────────────────────────────────
// Timestamp helper
// ──────────────────────────────────────────────────────────────────────────────

static std::string timestamp()
{
    using namespace std::chrono;
    const auto now  = system_clock::now();
    const auto t    = system_clock::to_time_t(now);
    const auto ms   = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    std::ostringstream oss;
    oss << buf << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ── CLI args ──────────────────────────────────────────────────────────────
    const std::string initial_prompt = (argc >= 2) ? argv[1] : "";
    const int64_t max_ticks = (argc >= 3) ? std::atoll(argv[2]) : -1;  // -1 = infinite

    // ── Component setup ───────────────────────────────────────────────────────
    std::cout << "[NIKOLA] Initialising CognitiveTorus (3^9 = 19,683 nodes)...\n";
    // amplitude=0.01f: small enough that nonlinear term stays stable over long
    // runs while still producing measurable resonance patterns.
#ifdef NIKOLA_HAS_ORT
    CognitiveTorus torus(3, NIKOLA_ORT_TOKENIZER_PATH, NIKOLA_ORT_MODEL_PATH,
                         /*pilot_dim=*/3, /*amplitude=*/0.01f);
#else
    CognitiveTorus torus(3, /*pilot_dim=*/3, /*amplitude=*/0.01f);
#endif

    std::cout << "[NIKOLA] Initialising AutonomyEngine...\n";
    AutonomyConfig eng_cfg;
    eng_cfg.enable_dream_weave = false;  // disable dream weave in daemon (Phase 24+)
    AutonomyEngine engine(eng_cfg);

    // ── DecisionLoop config ───────────────────────────────────────────────────
    DecisionLoopConfig dl_cfg;
    dl_cfg.steps_per_tick      = 50;
    dl_cfg.action_threshold    = 0.05f;
    dl_cfg.min_emit_interval_s = 5.0f;
    dl_cfg.vocabulary = {
        "hello", "curious", "wonder", "explore", "think",
        "memory", "wave", "resonance", "energy", "time",
        "nikola", "system", "pattern", "flow", "understand",
        "silence", "question", "unknown", "signal", "field",
    };

    // ORT paths — enables full-fidelity mode:
    //   · ResonanceDecoder vocabulary registered via real BERT-Tiny embeddings
    //   · ThoughtComposer selects templates via Transformer cosine-similarity
    //     rather than scalar heuristics
#ifdef NIKOLA_HAS_ORT
    dl_cfg.tokenizer_json_path    = std::string(NIKOLA_ORT_TOKENIZER_PATH) + "/tokenizer.json";
    dl_cfg.transformer_model_path = NIKOLA_ORT_MODEL_PATH;
    std::cout << "[NIKOLA] ORT mode: full-fidelity embeddings + Transformer thought composition\n";
#else
    std::cout << "[NIKOLA] No-ORT mode: synthetic vocabulary waves + heuristic thought composition\n";
#endif

    DecisionLoop loop(torus, engine, dl_cfg);

    // ── Callbacks ─────────────────────────────────────────────────────────────

    // Non-SILENT action → print to stdout
    loop.on_action = [](const DecisionResult& r) {
        std::cout << "[" << timestamp() << "] "
                  << action_name(r.type) << " (score=" << r.score << ")"
                  << (r.payload.empty() ? "" : ": " + r.payload)
                  << "\n";
        std::cout.flush();
    };

    // Every tick → print compact state line at reduced rate (every 10 ticks)
    uint64_t last_log_tick = 0;
    loop.on_tick = [&last_log_tick](const NikolaState& s) {
        // Only print every 10 ticks to avoid flooding
        // (accessed via closure capture, not thread-safe if multithreaded)
        static uint64_t tick_n = 0;
        ++tick_n;
        if (tick_n - last_log_tick >= 10) {
            last_log_tick = tick_n;
            std::cout << "[TICK " << std::setw(6) << tick_n << "] "
                      << "t=" << std::fixed << std::setprecision(2) << s.time
                      << "  E=" << s.torus_energy
                      << "  dopa=" << s.dopamine
                      << "  atp=" << s.atp
                      << "  boredom=" << s.boredom
                      << "  H=" << std::setprecision(2) << s.entropy
                      << "\n";
            std::cout.flush();
        }
    };

    // ── Optional initial stimulus ─────────────────────────────────────────────
    if (!initial_prompt.empty()) {
        std::cout << "[NIKOLA] Injecting stimulus: \"" << initial_prompt << "\"\n";
        loop.inject_stimulus(initial_prompt);
    } else {
        std::cout << "[NIKOLA] No initial stimulus. Running freely.\n";
    }

    std::cout << "[NIKOLA] Daemon running"
              << (max_ticks > 0 ? " (max " + std::to_string(max_ticks) + " ticks)" : "")
              << " — Ctrl+C to stop.\n\n";

    // ── Main loop ─────────────────────────────────────────────────────────────
    int64_t tick_n = 0;
    while (g_running.load()) {
        loop.tick();
        ++tick_n;
        if (max_ticks > 0 && tick_n >= max_ticks) break;
    }

    const NikolaState& final = loop.last_state();
    std::cout << "\n[NIKOLA] Shutting down after " << loop.tick_count() << " ticks.\n"
              << "  Final: t=" << final.time
              << "  atp=" << final.atp
              << "  dopamine=" << final.dopamine
              << "  last_action=" << action_name(final.last_action) << "\n";

    return 0;
}
