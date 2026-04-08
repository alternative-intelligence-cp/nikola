/**
 * @file src/nikola_dap_server.cpp
 * @brief nikola-dap — Debug Adapter Protocol server over stdio
 *
 * Launches a CognitiveTorus + full DecisionLoop, then serves DAP requests
 * over stdin/stdout.  VS Code attaches via a launch.json that invokes this
 * binary.  The server exposes NikolaState, Ψ field stats, metabolic state,
 * and memory database contents as DAP variables.
 *
 * Supported DAP requests:
 *   initialize          — capabilities handshake
 *   configurationDone   — start the physics simulation
 *   threads             — single "Nikola Physics" thread
 *   stackTrace          — current tick as single frame
 *   scopes              — Physics / Metabolic / Memory scopes
 *   variables           — variable tree per scope
 *   evaluate            — expression evaluation (H, drift, node:N, etc.)
 *   continue            — run until next breakpoint tick
 *   next                — advance one tick (step-over)
 *   setBreakpoints      — breakpoint on tick N (source lines = tick numbers)
 *   disconnect          — clean shutdown
 *
 * Nikola extensions:
 *   nikola/gridSnapshot     — full Ψ field snapshot event
 *   nikola/hamiltonianState — energy summary
 *
 * Usage:
 *   nikola-dap [--ticks N] [--steps N] [--memory-lmdb PATH]
 */

#include <nikola/physics/wave_function.hpp>
#include <nikola/physics/hamiltonian.hpp>
#include <nikola/physics/propagator.hpp>
#include <nikola/diag/debug_adapter.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/autonomy/hamiltonian_value.hpp>
#include <nikola/security/bootstrap_manager.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <set>
#include <vector>

using namespace nikola;

// ── Physics constants ────────────────────────────────────────────────────────
static constexpr int   GRID_N = 3;
static constexpr float C0     = 1.0f;
static constexpr float BETA   = 1.0f;
static constexpr float DT     = 0.001f;

// ── DAP variable reference IDs ───────────────────────────────────────────────
static constexpr int SCOPE_PHYSICS   = 1;
static constexpr int SCOPE_METABOLIC = 2;
static constexpr int SCOPE_MEMORY    = 3;
static constexpr int SCOPE_PSI       = 4;

// ── JSON helpers ─────────────────────────────────────────────────────────────

static std::string jf(double v, int prec = 6) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << v;
    return o.str();
}

static std::string mk_var(const std::string& name, const std::string& value,
                           const std::string& type = "float", int ref = 0) {
    return "{\"name\":\"" + name + "\",\"value\":\"" + value
         + "\",\"type\":\"" + type + "\",\"variablesReference\":" + std::to_string(ref) + "}";
}

// ── Minimal JSON request parser ──────────────────────────────────────────────

struct DapRequest {
    int seq = 0;
    std::string command;
    std::string arguments;  // raw JSON substring
};

static std::string extract_string(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\":\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    auto start = pos + needle.size();
    auto end = json.find('"', start);
    if (end == std::string::npos) return "";
    return json.substr(start, end - start);
}

static int extract_int(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\":";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return 0;
    auto start = pos + needle.size();
    while (start < json.size() && json[start] == ' ') ++start;
    return std::atoi(json.c_str() + start);
}

static std::string extract_object(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\":";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "{}";
    auto start = pos + needle.size();
    while (start < json.size() && json[start] == ' ') ++start;
    if (start >= json.size() || json[start] != '{') return "{}";
    int depth = 0;
    auto end = start;
    for (; end < json.size(); ++end) {
        if (json[end] == '{') ++depth;
        else if (json[end] == '}') { --depth; if (depth == 0) { ++end; break; } }
    }
    return json.substr(start, end - start);
}

// Parse breakpoint lines array from setBreakpoints arguments
static std::vector<int> extract_breakpoint_lines(const std::string& json) {
    std::vector<int> lines;
    auto pos = json.find("\"breakpoints\":");
    if (pos == std::string::npos) {
        pos = json.find("\"lines\":");
        if (pos == std::string::npos) return lines;
    }
    pos = json.find('[', pos);
    if (pos == std::string::npos) return lines;
    auto end = json.find(']', pos);
    if (end == std::string::npos) return lines;
    std::string arr = json.substr(pos + 1, end - pos - 1);
    // Find "line":N patterns
    std::string::size_type p = 0;
    while ((p = arr.find("\"line\":", p)) != std::string::npos) {
        p += 7;
        lines.push_back(std::atoi(arr.c_str() + p));
    }
    // Also try bare numbers (simple array of ints)
    if (lines.empty()) {
        std::istringstream stream(arr);
        std::string token;
        while (std::getline(stream, token, ',')) {
            try { lines.push_back(std::stoi(token)); } catch (...) {}
        }
    }
    return lines;
}

static DapRequest parse_request(const std::string& json) {
    DapRequest req;
    req.seq = extract_int(json, "seq");
    req.command = extract_string(json, "command");
    req.arguments = extract_object(json, "arguments");
    return req;
}

// ── DAP response/event builders ──────────────────────────────────────────────

static void send(const std::string& json) {
    std::cout << "Content-Length: " << json.size() << "\r\n\r\n" << json;
    std::cout.flush();
}

static void send_response(int seq, int req_seq, const std::string& command,
                           const std::string& body = "{}", bool success = true,
                           const std::string& msg = "") {
    std::ostringstream o;
    o << "{\"seq\":" << seq
      << ",\"type\":\"response\""
      << ",\"request_seq\":" << req_seq
      << ",\"success\":" << (success ? "true" : "false")
      << ",\"command\":\"" << command << "\"";
    if (!success && !msg.empty())
        o << ",\"message\":\"" << diag::DebugAdapter::escape_json(msg) << "\"";
    if (body != "{}")
        o << ",\"body\":" << body;
    o << "}";
    send(o.str());
}

static void send_event(int seq, const std::string& event,
                        const std::string& body = "{}") {
    std::ostringstream o;
    o << "{\"seq\":" << seq
      << ",\"type\":\"event\""
      << ",\"event\":\"" << event << "\"";
    if (body != "{}")
        o << ",\"body\":" << body;
    o << "}";
    send(o.str());
}

// ── Read a DAP message from stdin ────────────────────────────────────────────

static bool read_message(std::string& out_json) {
    // Read headers until blank line
    std::string line;
    int content_length = -1;

    while (std::getline(std::cin, line)) {
        // Strip trailing \r
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) break;  // header/body separator

        if (line.find("Content-Length: ") == 0) {
            content_length = std::atoi(line.c_str() + 16);
        }
    }

    if (content_length <= 0 || !std::cin.good()) return false;

    out_json.resize(static_cast<std::size_t>(content_length));
    std::cin.read(out_json.data(), content_length);
    return std::cin.good();
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // Parse CLI
    int max_ticks = 10000;
    int steps_per_tick = 50;
    std::string lmdb_path;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--ticks") == 0 && i + 1 < argc)
            max_ticks = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
            steps_per_tick = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--memory-lmdb") == 0 && i + 1 < argc)
            lmdb_path = argv[++i];
    }

    // ── Bootstrap ────────────────────────────────────────────────────────────
    security::BootstrapManager bootstrap;
    bootstrap.set_silent(true);
    bootstrap.get_token();

    // ── Physics setup ────────────────────────────────────────────────────────
    physics::WaveFunction wf;
    wf.seed_manifold(GRID_N, 0, 1, 1.0f, 2026u);

    physics::Hamiltonian ham;
    ham.set_c0(C0).set_beta(BETA);
    ham.verify_initial_conditions(wf);
    const double H0 = ham.compute(wf);

    autonomy::AutonomyConfig acfg;
    acfg.initial_atp = 1.0f;
    acfg.entropy_sample_dt = 0.1f;
    acfg.enable_boredom = true;
    acfg.enable_dream_weave = false;
    autonomy::AutonomyEngine engine(acfg);
    engine.hamiltonian_value().gamma_k  = 1.0f;
    engine.hamiltonian_value().gamma_p  = 1.0f;
    engine.hamiltonian_value().gamma_nl = 1.0f;
    engine.hamiltonian_value().h_max    = static_cast<float>(H0 * 100.0);

    physics::Propagator propagator;
    propagator.set_beta(BETA).set_c0(C0);

    // ── State ────────────────────────────────────────────────────────────────
    int out_seq = 0;         // Outgoing sequence counter
    int current_tick = 0;
    bool running = false;
    std::set<int> breakpoint_ticks;  // tick numbers to break at

    // Helper: advance physics by one tick
    auto do_tick = [&]() {
        for (int s = 0; s < steps_per_tick; ++s) {
            propagator.step(wf, DT);
        }
        const float wall_t = static_cast<float>(current_tick + 1)
                            * static_cast<float>(steps_per_tick) * DT;
        const std::size_t N = wf.num_nodes();
        std::span<const float> psi_r(wf.grid().psi_real(), N);
        std::span<const float> psi_i(wf.grid().psi_imag(), N);
        std::span<const float> vel_r(wf.grid().vel_real(), N);
        std::span<const float> vel_i(wf.grid().vel_imag(), N);
        engine.tick_physics(DT * static_cast<float>(steps_per_tick),
                            psi_r, psi_i, vel_r, vel_i,
                            BETA, autonomy::Reward::NEUTRAL, wall_t);
        ++current_tick;
    };

    // Helper: build physics variables
    auto physics_vars = [&]() -> std::string {
        double h = ham.compute(wf);
        double drift = (H0 > 0) ? (h - H0) / H0 * 100.0 : 0.0;
        std::ostringstream o;
        o << "{\"variables\":["
          << mk_var("H₀", jf(H0), "double") << ","
          << mk_var("H", jf(h), "double") << ","
          << mk_var("drift_pct", jf(drift) + "%", "double") << ","
          << mk_var("total_probability", jf(wf.total_probability()), "double") << ","
          << mk_var("kinetic_energy", jf(wf.total_kinetic_energy()), "double") << ","
          << mk_var("max_amplitude", jf(wf.max_amplitude()), "float") << ","
          << mk_var("mean_curvature", jf(wf.mean_curvature()), "double") << ","
          << mk_var("tick", std::to_string(current_tick), "int") << ","
          << mk_var("nodes", std::to_string(wf.num_nodes()), "size_t")
          << "]}";
        return o.str();
    };

    // Helper: build metabolic variables
    auto metabolic_vars = [&]() -> std::string {
        auto s = engine.snapshot();
        std::ostringstream o;
        o << "{\"variables\":["
          << mk_var("ATP", jf(s.atp, 4)) << ","
          << mk_var("dopamine", jf(s.dopamine, 4)) << ","
          << mk_var("serotonin", jf(engine.serotonin(), 4)) << ","
          << mk_var("norepinephrine", jf(engine.norepinephrine(), 4)) << ","
          << mk_var("boredom", jf(s.boredom, 4)) << ","
          << mk_var("entropy", jf(s.entropy, 4)) << ","
          << mk_var("state", std::to_string(static_cast<int>(s.state)), "AutonomyState") << ","
          << mk_var("nap_count", std::to_string(s.nap_count), "uint32_t")
          << "]}";
        return o.str();
    };

    // Helper: build Ψ field variables (sampled)
    auto psi_vars = [&]() -> std::string {
        std::ostringstream o;
        o << "{\"variables\":[";
        const float* pr = wf.grid().psi_real();
        const float* pi = wf.grid().psi_imag();
        const std::size_t N = wf.num_nodes();
        // Sample 16 evenly-spaced nodes
        constexpr std::size_t MAX_SAMPLE = 16;
        const std::size_t step = std::max<std::size_t>(1, N / MAX_SAMPLE);
        bool first = true;
        for (std::size_t i = 0; i < N && (i / step) < MAX_SAMPLE; i += step) {
            double amp2 = static_cast<double>(pr[i])*pr[i]
                        + static_cast<double>(pi[i])*pi[i];
            if (!first) o << ",";
            first = false;
            o << mk_var("node[" + std::to_string(i) + "]",
                        "ψ=(" + jf(pr[i],4) + "," + jf(pi[i],4) + ") |ψ|²=" + jf(amp2,8),
                        "NodeInspect");
        }
        o << "]}";
        return o.str();
    };

    // ── DAP event loop ───────────────────────────────────────────────────────
    std::string msg_json;
    while (read_message(msg_json)) {
        auto req = parse_request(msg_json);
        const auto& cmd = req.command;

        if (cmd == "initialize") {
            std::string body = R"({)"
                R"("supportsConfigurationDoneRequest":true,)"
                R"("supportsEvaluateForHovers":true,)"
                R"("supportsSetVariable":false,)"
                R"("nikola":{"supportsGridSnapshot":true,"supportsNodeInspect":true,"gridDimensions":9})"
                R"(})";
            send_response(++out_seq, req.seq, cmd, body);
            send_event(++out_seq, "initialized");

        } else if (cmd == "configurationDone") {
            running = false;
            send_response(++out_seq, req.seq, cmd);
            // Send stopped event so IDE shows initial state
            send_event(++out_seq, "stopped",
                       R"({"reason":"entry","threadId":1,"allThreadsStopped":true})");

        } else if (cmd == "threads") {
            send_response(++out_seq, req.seq, cmd,
                std::string(R"--({"threads":[{"id":1,"name":"Nikola Physics (tick )--")
                + std::to_string(current_tick) + R"--()"}]})--");

        } else if (cmd == "stackTrace") {
            std::ostringstream body;
            body << R"({"stackFrames":[{"id":1,"name":"tick )"
                 << current_tick
                 << R"(","source":{"name":"nikola-physics","path":"nikola://torus"})"
                 << R"(,"line":)" << current_tick
                 << R"(,"column":1}],"totalFrames":1})";
            send_response(++out_seq, req.seq, cmd, body.str());

        } else if (cmd == "scopes") {
            send_response(++out_seq, req.seq, cmd,
                "{\"scopes\":["
                "{\"name\":\"Physics\",\"variablesReference\":" + std::to_string(SCOPE_PHYSICS) + ",\"expensive\":false},"
                "{\"name\":\"Metabolic\",\"variablesReference\":" + std::to_string(SCOPE_METABOLIC) + ",\"expensive\":false},"
                "{\"name\":\"Ψ Field\",\"variablesReference\":" + std::to_string(SCOPE_PSI) + ",\"expensive\":false},"
                "{\"name\":\"Memory\",\"variablesReference\":" + std::to_string(SCOPE_MEMORY) + ",\"expensive\":false}"
                "]}");

        } else if (cmd == "variables") {
            int ref = extract_int(req.arguments, "variablesReference");
            std::string body;
            if (ref == SCOPE_PHYSICS)        body = physics_vars();
            else if (ref == SCOPE_METABOLIC) body = metabolic_vars();
            else if (ref == SCOPE_PSI)       body = psi_vars();
            else if (ref == SCOPE_MEMORY) {
                body = "{\"variables\":["
                    + mk_var("lmdb_path", lmdb_path.empty() ? "(none)" : lmdb_path, "string")
                    + "]}";
            } else {
                body = "{\"variables\":[]}";
            }
            send_response(++out_seq, req.seq, cmd, body);

        } else if (cmd == "evaluate") {
            std::string expr = extract_string(req.arguments, "expression");
            std::string result;
            std::string var_type = "double";

            if (expr == "H" || expr == "hamiltonian") {
                result = jf(ham.compute(wf));
            } else if (expr == "drift") {
                double h = ham.compute(wf);
                result = jf((H0 > 0) ? (h - H0) / H0 * 100.0 : 0.0) + "%";
            } else if (expr == "tick") {
                result = std::to_string(current_tick); var_type = "int";
            } else if (expr == "atp") {
                result = jf(engine.snapshot().atp, 4);
            } else if (expr == "dopamine") {
                result = jf(engine.snapshot().dopamine, 4);
            } else if (expr.substr(0, 5) == "node:" && expr.size() > 5) {
                auto idx = static_cast<std::size_t>(std::stoul(expr.substr(5)));
                if (idx < wf.num_nodes()) {
                    const float* pr = wf.grid().psi_real();
                    const float* pi = wf.grid().psi_imag();
                    double a2 = static_cast<double>(pr[idx])*pr[idx]
                              + static_cast<double>(pi[idx])*pi[idx];
                    result = "ψ=(" + jf(pr[idx],4) + "," + jf(pi[idx],4) + ") |ψ|²=" + jf(a2,8);
                    var_type = "NodeInspect";
                } else {
                    result = "index out of range";
                }
            } else {
                send_response(++out_seq, req.seq, cmd, "{}", false,
                              "Unknown expression: " + expr);
                continue;
            }

            send_response(++out_seq, req.seq, cmd,
                "{\"result\":\"" + diag::DebugAdapter::escape_json(result)
                + "\",\"type\":\"" + var_type + "\",\"variablesReference\":0}");

        } else if (cmd == "setBreakpoints") {
            auto lines = extract_breakpoint_lines(req.arguments);
            breakpoint_ticks.clear();
            std::ostringstream body;
            body << "{\"breakpoints\":[";
            for (std::size_t i = 0; i < lines.size(); ++i) {
                breakpoint_ticks.insert(lines[i]);
                if (i > 0) body << ",";
                body << "{\"id\":" << (i + 1)
                     << ",\"verified\":true,\"line\":" << lines[i] << "}";
            }
            body << "]}";
            send_response(++out_seq, req.seq, cmd, body.str());

        } else if (cmd == "continue") {
            send_response(++out_seq, req.seq, cmd,
                          R"({"allThreadsContinued":true})");
            // Run ticks until breakpoint or max
            running = true;
            while (running && current_tick < max_ticks) {
                do_tick();
                if (breakpoint_ticks.count(current_tick)) {
                    running = false;
                    send_event(++out_seq, "stopped",
                        R"({"reason":"breakpoint","threadId":1,"allThreadsStopped":true})");
                }
            }
            if (running && current_tick >= max_ticks) {
                running = false;
                send_event(++out_seq, "stopped",
                    R"({"reason":"pause","threadId":1,"allThreadsStopped":true,"description":"max ticks reached"})");
            }

        } else if (cmd == "next" || cmd == "stepIn" || cmd == "stepOut") {
            send_response(++out_seq, req.seq, cmd);
            if (current_tick < max_ticks) {
                do_tick();
            }
            send_event(++out_seq, "stopped",
                R"({"reason":"step","threadId":1,"allThreadsStopped":true})");

        } else if (cmd == "pause") {
            running = false;
            send_response(++out_seq, req.seq, cmd);
            send_event(++out_seq, "stopped",
                R"({"reason":"pause","threadId":1,"allThreadsStopped":true})");

        } else if (cmd == "disconnect" || cmd == "terminate") {
            send_response(++out_seq, req.seq, cmd);
            break;

        } else if (cmd == "nikola/gridSnapshot") {
            auto snap = diag::GridSnapshot::from(wf, H0,
                static_cast<float>(current_tick), static_cast<uint64_t>(current_tick));
            send_response(++out_seq, req.seq, cmd,
                          diag::DebugAdapter::format_variables(snap));

        } else if (cmd == "nikola/hamiltonianState") {
            auto snap = diag::GridSnapshot::from(wf, H0,
                static_cast<float>(current_tick), static_cast<uint64_t>(current_tick));
            send_response(++out_seq, req.seq, cmd,
                          diag::DebugAdapter::format_hamiltonian_state(snap));

        } else {
            send_response(++out_seq, req.seq, cmd, "{}", false,
                          "Unsupported: " + cmd);
        }
    }

    return 0;
}
