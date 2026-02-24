/**
 * @file src/nikola_node_main.cpp
 * @brief Phase 30 — NikolaNode binary: Nikola as a distributed ZMQ network node.
 *
 * Nikola binds two ZMQ sockets and becomes a first-class network citizen:
 *
 *   PUB  (default tcp://0.0.0.0:5560)  publishes actions + state telemetry
 *   PULL (default tcp://0.0.0.0:5561)  receives text stimuli from clients
 *
 * The cognitive core (CognitiveTorus + AutonomyEngine + DecisionLoop) is
 * unchanged.  ZMQ is purely an I/O wrapper around the existing tick loop.
 *
 * Usage:
 *   ./nikola_node [initial_stimulus] [max_ticks] [pub_endpoint] [pull_endpoint]
 *
 *   ./nikola_node "hello nikola" 500
 *   ./nikola_node "hello nikola" 500
 *   ./nikola_node "" -1 tcp://0.0.0.0:5560 tcp://0.0.0.0:5561
 *
 * Quick client snippet (Python):
 *   import zmq, time
 *   ctx = zmq.Context()
 *   sub = ctx.socket(zmq.SUB); sub.connect("tcp://localhost:5560")
 *   sub.setsockopt_string(zmq.SUBSCRIBE, "nikola.v1")
 *   push = ctx.socket(zmq.PUSH); push.connect("tcp://localhost:5561")
 *   time.sleep(0.3)   # slow-joiner delay
 *   push.send_string("what is consciousness")
 *   while True:
 *       topic, msg = sub.recv_multipart()
 *       print(topic.decode(), msg.decode())
 */

#include <nikola/autonomy/nikola_node.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

using namespace nikola::autonomy;

static std::atomic<bool> g_running{true};
static void signal_handler(int) { g_running.store(false); }

static std::string timestamp()
{
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto t   = system_clock::to_time_t(now);
    const auto ms  = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    std::ostringstream oss;
    oss << buf << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

int main(int argc, char* argv[])
{
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    const std::string initial_stimulus = (argc >= 2) ? argv[1] : "";
    const int64_t     max_ticks        = (argc >= 3) ? std::atoll(argv[2]) : -1;
    const std::string pub_endpoint     = (argc >= 4) ? argv[3] : "tcp://*:5560";
    const std::string pull_endpoint    = (argc >= 5) ? argv[4] : "tcp://*:5561";

    std::cout << "[NIKOLA-NODE] Binding sockets\n"
              << "  PUB  (actions + state) → " << pub_endpoint  << "\n"
              << "  PULL (stimuli)         ← " << pull_endpoint << "\n\n";

    // ── Build node ────────────────────────────────────────────────────────────
    NikolaNodeConfig cfg;
    cfg.pub_endpoint           = pub_endpoint;
    cfg.pull_endpoint          = pull_endpoint;
    cfg.state_publish_interval = 10;

    cfg.decision_config.steps_per_tick      = 50;
    cfg.decision_config.action_threshold    = 0.05f;
    cfg.decision_config.min_emit_interval_s = 5.0f;
    cfg.decision_config.vocabulary = {
        "hello", "curious", "wonder", "explore", "think",
        "memory", "wave", "resonance", "energy", "time",
        "nikola", "system", "pattern", "flow", "understand",
        "silence", "question", "unknown", "signal", "field",
    };

#ifdef NIKOLA_HAS_ORT
    cfg.decision_config.tokenizer_json_path    =
        std::string(NIKOLA_ORT_TOKENIZER_PATH) + "/tokenizer.json";
    cfg.decision_config.transformer_model_path = NIKOLA_ORT_MODEL_PATH;
    std::cout << "[NIKOLA-NODE] ORT mode: full-fidelity BERT embeddings\n";
#else
    std::cout << "[NIKOLA-NODE] Synthetic mode: heuristic vocabulary waves\n";
#endif

    NikolaNode node(cfg);

    // ── Console logging (mirrors nikola_daemon.cpp) ───────────────────────────
    node.on_action = [](const DecisionResult& r) {
        std::cout << "[" << timestamp() << "] "
                  << action_name(r.type) << " (score=" << r.score << ")"
                  << (r.payload.empty() ? "" : ": " + r.payload)
                  << "\n";
        std::cout.flush();
    };

    node.on_tick = [](const NikolaState& s, uint64_t tick) {
        if (tick % 10 == 0) {
            std::cout << "[TICK " << std::setw(6) << tick << "] "
                      << "t=" << std::fixed << std::setprecision(2) << s.time
                      << "  E="      << s.torus_energy
                      << "  dopa="   << s.dopamine
                      << "  atp="    << s.atp
                      << "  boredom=" << s.boredom
                      << "  H="      << s.entropy
                      << "\n";
            std::cout.flush();
        }
    };

    // ── Optional initial stimulus ─────────────────────────────────────────────
    if (!initial_stimulus.empty()) {
        std::cout << "[NIKOLA-NODE] Injecting stimulus: \""
                  << initial_stimulus << "\"\n";
        node.inject_stimulus(initial_stimulus);
    }

    std::cout << "[NIKOLA-NODE] Running"
              << (max_ticks > 0 ? " (max " + std::to_string(max_ticks) + " ticks)" : "")
              << " — Ctrl+C or send SIGTERM to stop.\n\n";

    // ── Run in a thread so SIGINT can interrupt us ────────────────────────────
    std::atomic<bool> done{false};
    std::thread worker([&]() {
        node.run(max_ticks);
        done.store(true);
    });

    while (!done.load() && g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (!done.load()) {
        node.stop();
    }
    worker.join();

    const NikolaState& fs = node.last_state();
    std::cout << "\n[NIKOLA-NODE] Shutdown after " << node.tick_count() << " ticks.\n"
              << "  Final: t=" << fs.time
              << "  atp="      << fs.atp
              << "  dopamine=" << fs.dopamine
              << "  last_action=" << action_name(fs.last_action) << "\n";
    return 0;
}
