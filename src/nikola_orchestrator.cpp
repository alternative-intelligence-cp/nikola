/**
 * @file src/nikola_orchestrator.cpp
 * @brief nikola-orchestrator — ZMQ Spine + Component Lifecycle Service.
 *
 * The orchestrator is the root process of the Nikola distributed system.
 * It owns the ZMQ spine, monitors component heartbeats, manages circuit
 * breakers, and triggers crash recovery.
 *
 * Other Nikola processes (nikola-daemon, nikola-run, tools) connect to
 * the orchestrator's PUB/SUB endpoints for control and data flow.
 *
 * USAGE
 * ─────
 *   nikola-orchestrator                           # defaults
 *   nikola-orchestrator --events tcp://*:6000      # custom PUB endpoint
 *   nikola-orchestrator --data tcp://*:6002        # custom data endpoint
 *   nikola-orchestrator --io-threads 2             # more IO threads
 *   nikola-orchestrator --no-shm-cleanup           # skip stale SHM cleanup
 *   nikola-orchestrator --help
 */

#ifndef NIKOLA_ORCHESTRATOR_IMPL
#define NIKOLA_ORCHESTRATOR_IMPL
#endif

#include <nikola/infrastructure/orchestrator.hpp>
#include <nikola/infrastructure/spine.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

using namespace nikola::infrastructure;

// ─────────────────────────────────────────────────────────────────────────────
// ANSI helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace {
    const char* RST   = "\033[0m";
    const char* BOLD  = "\033[1m";
    const char* DIM   = "\033[2m";
    const char* CYAN  = "\033[36m";
    const char* GREEN = "\033[32m";
    const char* RED   = "\033[31m";
}

// ─────────────────────────────────────────────────────────────────────────────
// Shutdown signal
// ─────────────────────────────────────────────────────────────────────────────
static std::atomic<bool> g_shutdown_requested{false};

static void signal_handler(int) {
    g_shutdown_requested.store(true, std::memory_order_release);
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────
struct OrchestratorCli {
    std::string events_endpoint  = "tcp://*:5555";
    std::string control_endpoint = "tcp://*:5556";
    std::string data_endpoint    = "tcp://*:5557";
    int         io_threads       = 1;
    bool        cleanup_stale_shm = true;
    bool        help              = false;
};

static void print_usage() {
    std::cerr
        << "Usage: nikola-orchestrator [OPTIONS]\n\n"
        << "Options:\n"
        << "  --events <endpoint>   PUB endpoint for events/control  [tcp://*:5555]\n"
        << "  --control <endpoint>  REP endpoint for inbound control [tcp://*:5556]\n"
        << "  --data <endpoint>     PUB endpoint for data plane      [tcp://*:5557]\n"
        << "  --io-threads <N>      ZMQ IO threads                   [1]\n"
        << "  --no-shm-cleanup      Skip stale SHM segment cleanup\n"
        << "  --help                Show this message\n";
}

static OrchestratorCli parse_args(int argc, char* argv[]) {
    OrchestratorCli cli;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            cli.help = true;
        } else if (arg == "--events" && i + 1 < argc) {
            cli.events_endpoint = argv[++i];
        } else if (arg == "--control" && i + 1 < argc) {
            cli.control_endpoint = argv[++i];
        } else if (arg == "--data" && i + 1 < argc) {
            cli.data_endpoint = argv[++i];
        } else if (arg == "--io-threads" && i + 1 < argc) {
            cli.io_threads = std::stoi(argv[++i]);
        } else if (arg == "--no-shm-cleanup") {
            cli.cleanup_stale_shm = false;
        } else {
            std::cerr << "Unknown flag: " << arg << "\n";
            cli.help = true;
        }
    }
    return cli;
}

// ─────────────────────────────────────────────────────────────────────────────
// Banner
// ─────────────────────────────────────────────────────────────────────────────
static void print_banner() {
    std::cout << "\n" << BOLD << CYAN
              << "  ╔══════════════════════════════════════════════╗\n"
              << "  ║       NIKOLA  ORCHESTRATOR  v0.1.11          ║\n"
              << "  ║    ZMQ Spine + Component Lifecycle Mgmt      ║\n"
              << "  ╚══════════════════════════════════════════════╝\n"
              << RST << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    OrchestratorCli cli = parse_args(argc, argv);
    if (cli.help) {
        print_usage();
        return 0;
    }

    print_banner();

    // ── Signal handlers ─────────────────────────────────────────────────
    struct sigaction sa{};
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT,  &sa, nullptr);

    // ── Configure ───────────────────────────────────────────────────────
    OrchestratorConfig ocfg;
    ocfg.events_endpoint   = cli.events_endpoint;
    ocfg.control_endpoint  = cli.control_endpoint;
    ocfg.data_endpoint     = cli.data_endpoint;
    ocfg.io_threads        = cli.io_threads;
    ocfg.cleanup_stale_shm = cli.cleanup_stale_shm;

    std::cout << DIM
              << "  Events:  " << ocfg.events_endpoint << "\n"
              << "  Control: " << ocfg.control_endpoint << "\n"
              << "  Data:    " << ocfg.data_endpoint << "\n"
              << "  IO thds: " << ocfg.io_threads << "\n"
              << RST << "\n";

    // ── Create + start orchestrator ─────────────────────────────────────
    Orchestrator orch(ocfg);

    orch.set_restart_callback([](const std::string& name) {
        std::cerr << RED << "  [RESTART] " << RST
                  << "Component died: " << name << " — restart triggered\n";
    });

    orch.start();
    std::cout << GREEN << "  ✓ " << RST
              << "Orchestrator RUNNING  (PID " << getpid() << ")\n"
              << DIM << "  Ctrl+C or SIGTERM to stop.\n" << RST << "\n";

    // ── Main loop ───────────────────────────────────────────────────────
    uint64_t tick = 0;
    while (!g_shutdown_requested.load(std::memory_order_acquire)) {
        // Process any queued tasks (restart requests, etc.)
        orch.process_pending_tasks();

        // Log status periodically (every 10 seconds)
        if (++tick % 100 == 0) {
            auto comps = orch.components();
            std::size_t alive = 0;
            for (const auto& c : comps) {
                if (c.alive) ++alive;
            }
            std::cout << DIM << "  [tick " << tick << "] "
                      << alive << "/" << comps.size() << " components alive  "
                      << "tasks_dispatched=" << orch.task_stats().dispatched
                      << RST << "\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // ── Shutdown ────────────────────────────────────────────────────────
    std::cout << "\n" << DIM << "  Shutting down..." << RST << "\n";
    orch.stop();
    std::cout << GREEN << "  ✓ " << RST << "Orchestrator STOPPED. Goodbye.\n\n";

    return 0;
}
