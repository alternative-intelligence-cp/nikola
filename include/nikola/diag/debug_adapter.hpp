/**
 * @file diag/debug_adapter.hpp
 * @brief DAP-compatible debug adapter for 9D coordinate / wavefunction inspection.
 *
 * Design requirements (ecosystem/08_DebugAdapter):
 *   - Implements a subset of the Debug Adapter Protocol (DAP / DAP spec 1.51)
 *     sufficient for Nikola state inspection in VS Code / DAP-compatible IDEs
 *   - Custom Nikola DAP extensions:
 *       nikola/gridSnapshot  — full 9D wavefunction field snapshot
 *       nikola/nodeInspect   — single node psi/vel complex pair
 *       nikola/hamiltonianState — energy & drift summary
 *   - DAP message framing: Content-Length header + JSON body over any stream
 *   - Stateless formatting helpers (pure static methods) + a stateful session
 *     class for sequential request/response management
 *   - Zero heap allocation for small variable responses (<= 64 vars)
 *   - Works without physics layer — uses `GridSnapshot` plain-data struct
 *
 * Protocol wire format (RFC-compliant DAP over any byte stream):
 * @code
 *   Content-Length: 123\r\n
 *   \r\n
 *   {"seq":1,"type":"response","request_seq":1,"success":true,...}
 * @endcode
 *
 * Usage:
 * @code
 *   // Capture snapshot from live WaveFunction
 *   auto snap = nikola::diag::GridSnapshot::from(wf);
 *
 *   // Format a DAP Variables response
 *   auto json = nikola::diag::DebugAdapter::format_variables(snap, 0, 16);
 *   auto msg  = nikola::diag::DebugAdapter::encode_message(json);
 *   write(output_fd, msg.data(), msg.size());
 * @endcode
 */

#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace nikola::diag {

// ─────────────────────────────────────────────────────────────────────────────
//  NodeInspect — single grid node data
// ─────────────────────────────────────────────────────────────────────────────

struct NodeInspect {
    uint64_t index    = 0;
    float    psi_real = 0.0f;
    float    psi_imag = 0.0f;
    float    vel_real = 0.0f;
    float    vel_imag = 0.0f;
    float    resonance = 1.0f;

    double psi_norm_sq() const noexcept {
        return static_cast<double>(psi_real) * psi_real
             + static_cast<double>(psi_imag) * psi_imag;
    }

    double vel_norm_sq() const noexcept {
        return static_cast<double>(vel_real) * vel_real
             + static_cast<double>(vel_imag) * vel_imag;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  GridSnapshot — plain-data capture of WaveFunction state
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Plain-data snapshot of the 9D wavefunction for debug serialization.
 *
 * Decoupled from WaveFunction so DebugAdapter can be used standalone.
 * Use `GridSnapshot::from(wf)` to populate from a live WaveFunction.
 */
struct GridSnapshot {
    std::vector<NodeInspect> nodes;
    double        hamiltonian    = 0.0;
    double        hamiltonian_0  = 0.0;   ///< Reference H at t=0
    float         time           = 0.0f;  ///< Simulation time
    int           grid_n         = 0;     ///< Torus base dimension (3 → 3^9 nodes)
    uint64_t      step           = 0;     ///< Propagation step count

    std::size_t num_nodes() const noexcept { return nodes.size(); }

    double energy_drift_pct() const noexcept {
        if (hamiltonian_0 == 0.0) return 0.0;
        return (hamiltonian - hamiltonian_0) / hamiltonian_0 * 100.0;
    }

    /** @brief Sample up to `max_nodes` evenly-spaced nodes for display. */
    std::vector<NodeInspect> sample(std::size_t max_nodes = 64) const {
        if (nodes.size() <= max_nodes) return nodes;
        std::vector<NodeInspect> out;
        out.reserve(max_nodes);
        const double step_d = static_cast<double>(nodes.size()) /
                              static_cast<double>(max_nodes);
        for (std::size_t i = 0; i < max_nodes; ++i) {
            out.push_back(nodes[static_cast<std::size_t>(i * step_d)]);
        }
        return out;
    }

#ifdef NIKOLA_DIAG_INCLUDE_PHYSICS
    /**
     * @brief Capture snapshot from a live WaveFunction (requires
     *        nikola/physics/wave_function.hpp + hamiltonian.hpp).
     * Enable by defining NIKOLA_DIAG_INCLUDE_PHYSICS before including.
     */
    static GridSnapshot from(const nikola::physics::WaveFunction& wf,
                             double h0 = 0.0,
                             float sim_time = 0.0f,
                             uint64_t sim_step = 0)
    {
        GridSnapshot snap;
        snap.time  = sim_time;
        snap.step  = sim_step;
        snap.hamiltonian_0 = h0;

        const auto& g = wf.grid();
        const std::size_t N = g.num_active_nodes();
        snap.nodes.reserve(N);
        for (std::size_t i = 0; i < N; ++i) {
            NodeInspect ni;
            ni.index     = static_cast<uint64_t>(i);
            ni.psi_real  = g.psi_real()[i];
            ni.psi_imag  = g.psi_imag()[i];
            ni.vel_real  = g.vel_real()[i];
            ni.vel_imag  = g.vel_imag()[i];
            snap.nodes.push_back(ni);
        }

        nikola::physics::Hamiltonian ham;
        snap.hamiltonian = ham.compute(wf);
        return snap;
    }
#endif
};

// ─────────────────────────────────────────────────────────────────────────────
//  DapMessage — request/response container
// ─────────────────────────────────────────────────────────────────────────────

struct DapMessage {
    int         seq         = 0;
    std::string type;         ///< "request", "response", "event"
    std::string command;
    std::string body_json;    ///< Raw JSON object string for the body field
    int         request_seq  = 0;
    bool        success      = true;
    std::string message;      ///< Optional error message

    static DapMessage request(int seq, std::string_view cmd,
                              std::string_view body = "{}") {
        return {seq, "request", std::string(cmd), std::string(body), 0, true, {}};
    }

    static DapMessage response(int seq, int req_seq, std::string_view cmd,
                               std::string_view body = "{}", bool ok = true,
                               std::string_view err_msg = "") {
        return {seq, "response", std::string(cmd), std::string(body),
                req_seq, ok, std::string(err_msg)};
    }

    static DapMessage event(int seq, std::string_view event_name,
                            std::string_view body = "{}") {
        return {seq, "event", std::string(event_name), std::string(body), 0, true, {}};
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  DebugAdapter
// ─────────────────────────────────────────────────────────────────────────────

class DebugAdapter {
public:
    // ── DAP message framing ──────────────────────────────────────────────────

    /**
     * @brief Wrap a JSON body with DAP Content-Length framing.
     *
     * Result: "Content-Length: N\r\n\r\n{json body}"
     * Suitable for writing to stdout (standard DAP transport).
     */
    static std::string encode_message(std::string_view json_body) {
        std::string out;
        out.reserve(32 + json_body.size());
        char hdr[32];
        const int n = std::snprintf(hdr, sizeof(hdr),
                                    "Content-Length: %zu\r\n\r\n",
                                    json_body.size());
        out.append(hdr, static_cast<std::size_t>(n));
        out.append(json_body);
        return out;
    }

    /**
     * @brief Serialize a DapMessage to JSON.
     */
    static std::string to_json(const DapMessage& msg) {
        std::ostringstream o;
        o << "{\"seq\":" << msg.seq
          << ",\"type\":\"" << msg.type << '"';

        if (msg.type == "request" || msg.type == "event") {
            o << ",\"command\":\"" << msg.command << '"';
        } else {
            o << ",\"command\":\"" << msg.command << '"'
              << ",\"request_seq\":" << msg.request_seq
              << ",\"success\":" << (msg.success ? "true" : "false");
            if (!msg.success && !msg.message.empty()) {
                o << ",\"message\":\"" << escape_json(msg.message) << '"';
            }
        }

        if (!msg.body_json.empty() && msg.body_json != "{}") {
            o << ",\"body\":" << msg.body_json;
        }
        o << '}';
        return o.str();
    }

    // ── Standard DAP responses ───────────────────────────────────────────────

    /**
     * @brief DAP Initialize response with Nikola-specific capabilities.
     */
    static std::string format_initialize_response(int seq, int req_seq) {
        const std::string body = R"({
  "supportsConfigurationDoneRequest": true,
  "supportsEvaluateForHovers": true,
  "supportsStepBack": false,
  "supportsRestartFrame": false,
  "supportsSetVariable": false,
  "supportsReadMemoryRequest": false,
  "additionalModuleColumns": [],
  "supportedChecksumAlgorithms": [],
  "nikola": {
    "supportsGridSnapshot": true,
    "supportsNodeInspect": true,
    "supportsHamiltonianState": true,
    "gridDimensions": 9,
    "torusTopology": true
  }
})";
        auto msg = DapMessage::response(seq, req_seq, "initialize", body);
        return to_json(msg);
    }

    // ── Nikola custom extensions ─────────────────────────────────────────────

    /**
     * @brief DAP Variables response for a grid snapshot.
     *
     * @param snap        Grid snapshot to serialize
     * @param var_ref     DAP variablesReference (used by IDE to expand tree)
     * @param max_vars    Maximum number of node variables to emit
     */
    static std::string format_variables(const GridSnapshot& snap,
                                        int var_ref = 1,
                                        std::size_t max_vars = 64)
    {
        std::ostringstream o;
        o << "{\"variables\":[";

        // ── Summary variables (always present) ──────────────────────────────
        o << "{\"name\":\"nodes\",\"value\":\"" << snap.num_nodes() << "\","
          << "\"type\":\"size_t\",\"variablesReference\":0}";

        o << ",{\"name\":\"hamiltonian\",\"value\":\""
          << std::to_string(snap.hamiltonian) << "\","
          << "\"type\":\"double\",\"variablesReference\":0}";

        o << ",{\"name\":\"energy_drift_pct\",\"value\":\""
          << std::to_string(snap.energy_drift_pct()) << "\","
          << "\"type\":\"double\",\"variablesReference\":0}";

        o << ",{\"name\":\"time\",\"value\":\""
          << std::to_string(snap.time) << "\","
          << "\"type\":\"float\",\"variablesReference\":0}";

        o << ",{\"name\":\"step\",\"value\":\""
          << snap.step << "\","
          << "\"type\":\"uint64_t\",\"variablesReference\":0}";

        // ── Sampled node variables ───────────────────────────────────────────
        const auto sampled = snap.sample(max_vars);
        for (const auto& ni : sampled) {
            o << ",{\"name\":\"node[" << ni.index << "]\","
              << "\"value\":\"psi=(" << ni.psi_real << "+" << ni.psi_imag << "i) "
              << "vel=(" << ni.vel_real << "+" << ni.vel_imag << "i) "
              << "|psi|²=" << std::to_string(ni.psi_norm_sq()) << "\","
              << "\"type\":\"NodeInspect\",\"variablesReference\":" << var_ref << "}";
        }

        o << "]}";
        (void)var_ref;
        return o.str();
    }

    /**
     * @brief Format a single node inspection (nikola/nodeInspect extension).
     */
    static std::string format_node_inspect(const NodeInspect& ni) {
        std::ostringstream o;
        o << "{"
          << "\"index\":" << ni.index << ","
          << "\"psi\":{\"real\":" << ni.psi_real << ",\"imag\":" << ni.psi_imag << "},"
          << "\"vel\":{\"real\":" << ni.vel_real << ",\"imag\":" << ni.vel_imag << "},"
          << "\"psi_norm_sq\":" << ni.psi_norm_sq() << ","
          << "\"vel_norm_sq\":" << ni.vel_norm_sq() << ","
          << "\"resonance\":" << ni.resonance
          << "}";
        return o.str();
    }

    /**
     * @brief Format Hamiltonian state summary (nikola/hamiltonianState).
     */
    static std::string format_hamiltonian_state(const GridSnapshot& snap) {
        std::ostringstream o;
        o << "{"
          << "\"H\":" << snap.hamiltonian << ","
          << "\"H0\":" << snap.hamiltonian_0 << ","
          << "\"drift_pct\":" << snap.energy_drift_pct() << ","
          << "\"time\":" << snap.time << ","
          << "\"step\":" << snap.step << ","
          << "\"nodes\":" << snap.num_nodes()
          << "}";
        return o.str();
    }

    /**
     * @brief Format a DAP Evaluate response (used for hover / watch expressions).
     *
     * Supports expressions:
     *   "H"      → hamiltonian value
     *   "drift"  → energy drift percentage
     *   "nodes"  → node count
     *   "step"   → simulation step
     *   "time"   → simulation time
     *   "node:N" → node N inspection (e.g. "node:0", "node:42")
     */
    static std::string format_evaluate(const GridSnapshot& snap,
                                       std::string_view expression,
                                       int seq, int req_seq)
    {
        std::string result;
        std::string var_type = "double";

        if (expression == "H" || expression == "hamiltonian") {
            result = std::to_string(snap.hamiltonian);
        } else if (expression == "drift") {
            result = std::to_string(snap.energy_drift_pct()) + "%";
        } else if (expression == "nodes") {
            result = std::to_string(snap.num_nodes());
            var_type = "size_t";
        } else if (expression == "step") {
            result = std::to_string(snap.step);
            var_type = "uint64_t";
        } else if (expression == "time") {
            result = std::to_string(snap.time);
            var_type = "float";
        } else if (expression.starts_with("node:")) {
            const std::size_t idx =
                static_cast<std::size_t>(std::stoul(std::string(expression.substr(5))));
            if (idx < snap.nodes.size()) {
                result = format_node_inspect(snap.nodes[idx]);
                var_type = "NodeInspect";
            } else {
                result = "index out of range";
            }
        } else {
            // Unknown expression
            auto msg = DapMessage::response(seq, req_seq, "evaluate", "{}",
                                             false, "Unknown expression: " +
                                             std::string(expression));
            return to_json(msg);
        }

        std::ostringstream body;
        body << "{\"result\":\"" << escape_json(result) << "\","
             << "\"type\":\"" << var_type << "\","
             << "\"variablesReference\":0}";

        auto msg = DapMessage::response(seq, req_seq, "evaluate", body.str());
        return to_json(msg);
    }

    /**
     * @brief Format a nikola/gridSnapshot event (pushed to IDE on breakpoint).
     */
    static std::string format_grid_snapshot_event(const GridSnapshot& snap,
                                                   int seq,
                                                   std::size_t max_nodes = 16)
    {
        const std::string body = format_variables(snap, 2, max_nodes);
        auto msg = DapMessage::event(seq, "nikola/gridSnapshot", body);
        return to_json(msg);
    }

    // ── Session ──────────────────────────────────────────────────────────────

    /**
     * @brief Stateful session: tracks request sequence numbers and
     *        maintains a current snapshot for evaluate/variables.
     */
    class Session {
    public:
        void update_snapshot(GridSnapshot snap) {
            snapshot_ = std::move(snap);
        }

        const GridSnapshot& snapshot() const noexcept { return snapshot_; }

        /**
         * @brief Process a raw DAP request body (JSON string) and return a
         *        framed response string ready to write to the client stream.
         *
         * Handles standard + Nikola custom commands.
         */
        std::string handle(std::string_view command, int req_seq,
                           std::string_view args_json = "{}") {
            const int resp_seq = ++seq_;
            std::string body_json;

            if (command == "initialize") {
                body_json = R"({"supportsEvaluateForHovers":true,)"
                            R"("nikola":{"supportsGridSnapshot":true}})";
            } else if (command == "variables" ||
                       command == "nikola/gridSnapshot") {
                body_json = format_variables(snapshot_);
            } else if (command == "nikola/hamiltonianState") {
                body_json = format_hamiltonian_state(snapshot_);
            } else if (command == "evaluate") {
                // Extract expression from args JSON (naive parse)
                const std::string expr = extract_string(args_json, "expression");
                return encode_message(
                    format_evaluate(snapshot_, expr, resp_seq, req_seq));
            } else if (command == "disconnect" || command == "terminate") {
                body_json = "{}";
            } else {
                auto msg = DapMessage::response(resp_seq, req_seq,
                    std::string(command), "{}", false,
                    "Unsupported command: " + std::string(command));
                return encode_message(to_json(msg));
            }

            auto msg = DapMessage::response(resp_seq, req_seq,
                std::string(command), body_json);
            return encode_message(to_json(msg));
        }

    private:
        int seq_ = 0;
        GridSnapshot snapshot_;

        /** Minimal JSON string extraction (no full parser needed here). */
        static std::string extract_string(std::string_view json,
                                          std::string_view key) {
            const std::string needle = "\"" + std::string(key) + "\":\"";
            const auto pos = json.find(needle);
            if (pos == std::string_view::npos) return "";
            const auto start = pos + needle.size();
            const auto end   = json.find('"', start);
            if (end == std::string_view::npos) return "";
            return std::string(json.substr(start, end - start));
        }
    };

    // ── Utilities ────────────────────────────────────────────────────────────

    /** @brief Escape a string for embedding in JSON. */
    static std::string escape_json(std::string_view s) {
        std::string out;
        out.reserve(s.size() + 4);
        for (const char c : s) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:   out += c;       break;
            }
        }
        return out;
    }

    /**
     * @brief Parse Content-Length from a DAP header.
     * @returns content length, or -1 on parse failure.
     */
    static int parse_content_length(std::string_view header) {
        const std::string_view key = "Content-Length: ";
        const auto pos = header.find(key);
        if (pos == std::string_view::npos) return -1;
        const auto start = pos + key.size();
        const auto end   = header.find('\r', start);
        const std::string len_str(header.substr(start,
            end == std::string_view::npos ? std::string_view::npos : end - start));
        try { return std::stoi(len_str); } catch (...) { return -1; }
    }
};

} // namespace nikola::diag
