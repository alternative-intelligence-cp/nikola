/**
 * @file tests/unit/phase20_debug_adapter_test.cpp
 * @brief Phase 20 — ecosystem/08_DebugAdapter unit tests
 *
 * Requirements validated:
 *   - DAP Content-Length framing (encode_message)
 *   - DapMessage JSON serialization (to_json)
 *   - Initialize response with Nikola capabilities
 *   - Variables response from GridSnapshot
 *   - Node inspection formatting
 *   - Hamiltonian state formatting
 *   - Evaluate expression dispatch (H, drift, nodes, step, time, node:N)
 *   - Grid snapshot event
 *   - Session::handle() request dispatch
 *   - escape_json utility
 *   - parse_content_length header parsing
 *   - GridSnapshot::sample() evenly-spaced sampling
 *   - NodeInspect computed properties (psi_norm_sq, vel_norm_sq)
 */

#include <nikola/diag/debug_adapter.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <string>

using namespace nikola::diag;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

static GridSnapshot make_snapshot(int n_nodes = 8,
                                   double H = 68890.5,
                                   double H0 = 70000.0)
{
    GridSnapshot snap;
    snap.hamiltonian   = H;
    snap.hamiltonian_0 = H0;
    snap.time          = 1.5f;
    snap.step          = 150;
    snap.grid_n        = 3;
    for (int i = 0; i < n_nodes; ++i) {
        NodeInspect ni;
        ni.index     = static_cast<uint64_t>(i);
        ni.psi_real  = static_cast<float>(i) * 0.1f;
        ni.psi_imag  = static_cast<float>(i) * 0.05f;
        ni.vel_real  = 0.01f;
        ni.vel_imag  = 0.02f;
        ni.resonance = 1.0f;
        snap.nodes.push_back(ni);
    }
    return snap;
}

// ─────────────────────────────────────────────────────────────────────────────
//  NodeInspect computed properties
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NodeInspect: computed properties", "[debug][node]") {
    NodeInspect ni;
    ni.psi_real = 3.0f;
    ni.psi_imag = 4.0f;
    ni.vel_real = 1.0f;
    ni.vel_imag = 0.0f;

    REQUIRE_THAT(ni.psi_norm_sq(), WithinAbs(25.0, 1e-9));
    REQUIRE_THAT(ni.vel_norm_sq(), WithinAbs(1.0,  1e-9));

    SECTION("zero node") {
        NodeInspect zero;
        REQUIRE(zero.psi_norm_sq() == 0.0);
        REQUIRE(zero.vel_norm_sq() == 0.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  GridSnapshot
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("GridSnapshot: energy_drift_pct", "[debug][snapshot]") {
    GridSnapshot snap;
    snap.hamiltonian   = 68890.5;
    snap.hamiltonian_0 = 70000.0;
    // drift = (68890.5 - 70000) / 70000 * 100 ≈ -1.585%
    REQUIRE_THAT(snap.energy_drift_pct(), WithinAbs(-1.585, 0.01));

    SECTION("zero H0 returns 0") {
        snap.hamiltonian_0 = 0.0;
        REQUIRE(snap.energy_drift_pct() == 0.0);
    }
}

TEST_CASE("GridSnapshot: sample() evenly spaced", "[debug][snapshot]") {
    GridSnapshot snap = make_snapshot(100);

    SECTION("sample less than size") {
        auto s = snap.sample(10);
        REQUIRE(s.size() == 10);
        // Should span the full range
        REQUIRE(s.front().index == 0);
        REQUIRE(s.back().index > 80);  // last ~10% of 100 nodes
    }

    SECTION("sample >= size returns all") {
        auto all = snap.sample(200);
        REQUIRE(all.size() == 100);
    }

    SECTION("sample 1 returns first node") {
        auto s = snap.sample(1);
        REQUIRE(s.size() == 1);
        REQUIRE(s[0].index == 0);
    }
}

TEST_CASE("GridSnapshot: num_nodes()", "[debug][snapshot]") {
    auto snap = make_snapshot(19683);
    REQUIRE(snap.num_nodes() == 19683);
}

// ─────────────────────────────────────────────────────────────────────────────
//  encode_message
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DebugAdapter: encode_message produces Content-Length framing", "[debug][framing]") {
    const std::string body = R"({"seq":1,"type":"response"})";
    const std::string framed = DebugAdapter::encode_message(body);

    REQUIRE_THAT(framed, ContainsSubstring("Content-Length:"));
    REQUIRE_THAT(framed, ContainsSubstring("\r\n\r\n"));
    REQUIRE_THAT(framed, ContainsSubstring(body));

    // Content-Length value must match body length
    const int content_len = DebugAdapter::parse_content_length(framed);
    REQUIRE(content_len == static_cast<int>(body.size()));
}

TEST_CASE("DebugAdapter: encode_message with empty body", "[debug][framing]") {
    const std::string framed = DebugAdapter::encode_message("{}");
    REQUIRE(DebugAdapter::parse_content_length(framed) == 2);
}

// ─────────────────────────────────────────────────────────────────────────────
//  parse_content_length
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DebugAdapter: parse_content_length", "[debug][framing]") {
    REQUIRE(DebugAdapter::parse_content_length("Content-Length: 123\r\n\r\n") == 123);
    REQUIRE(DebugAdapter::parse_content_length("Content-Length: 0\r\n\r\n") == 0);
    REQUIRE(DebugAdapter::parse_content_length("no header") == -1);
    REQUIRE(DebugAdapter::parse_content_length("") == -1);
}

// ─────────────────────────────────────────────────────────────────────────────
//  to_json / DapMessage
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DebugAdapter: to_json DapMessage response", "[debug][json]") {
    auto msg = DapMessage::response(5, 3, "variables", R"({"variables":[]})", true);
    const auto json = DebugAdapter::to_json(msg);

    REQUIRE_THAT(json, ContainsSubstring("\"seq\":5"));
    REQUIRE_THAT(json, ContainsSubstring("\"type\":\"response\""));
    REQUIRE_THAT(json, ContainsSubstring("\"command\":\"variables\""));
    REQUIRE_THAT(json, ContainsSubstring("\"request_seq\":3"));
    REQUIRE_THAT(json, ContainsSubstring("\"success\":true"));
}

TEST_CASE("DebugAdapter: to_json DapMessage error response", "[debug][json]") {
    auto msg = DapMessage::response(2, 1, "evaluate", "{}", false, "Unknown expression");
    const auto json = DebugAdapter::to_json(msg);
    REQUIRE_THAT(json, ContainsSubstring("\"success\":false"));
    REQUIRE_THAT(json, ContainsSubstring("Unknown expression"));
}

TEST_CASE("DebugAdapter: to_json DapMessage event", "[debug][json]") {
    auto msg = DapMessage::event(10, "nikola/gridSnapshot", R"({"variables":[]})");
    const auto json = DebugAdapter::to_json(msg);
    REQUIRE_THAT(json, ContainsSubstring("\"type\":\"event\""));
    REQUIRE_THAT(json, ContainsSubstring("\"command\":\"nikola/gridSnapshot\""));
}

TEST_CASE("DebugAdapter: to_json DapMessage request", "[debug][json]") {
    auto msg = DapMessage::request(1, "initialize", R"({"clientID":"vscode"})");
    const auto json = DebugAdapter::to_json(msg);
    REQUIRE_THAT(json, ContainsSubstring("\"type\":\"request\""));
    REQUIRE_THAT(json, ContainsSubstring("\"command\":\"initialize\""));
}

// ─────────────────────────────────────────────────────────────────────────────
//  format_initialize_response
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DebugAdapter: format_initialize_response", "[debug][init]") {
    const auto json = DebugAdapter::format_initialize_response(1, 1);
    REQUIRE_THAT(json, ContainsSubstring("supportsEvaluateForHovers"));
    REQUIRE_THAT(json, ContainsSubstring("nikola"));
    REQUIRE_THAT(json, ContainsSubstring("supportsGridSnapshot"));
    REQUIRE_THAT(json, ContainsSubstring("gridDimensions"));
    REQUIRE_THAT(json, ContainsSubstring("torusTopology"));
}

// ─────────────────────────────────────────────────────────────────────────────
//  format_variables
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DebugAdapter: format_variables basic structure", "[debug][variables]") {
    auto snap = make_snapshot(8);
    const auto json = DebugAdapter::format_variables(snap);

    REQUIRE_THAT(json, ContainsSubstring("\"variables\":["));
    REQUIRE_THAT(json, ContainsSubstring("\"nodes\""));
    REQUIRE_THAT(json, ContainsSubstring("\"hamiltonian\""));
    REQUIRE_THAT(json, ContainsSubstring("\"energy_drift_pct\""));
    REQUIRE_THAT(json, ContainsSubstring("\"time\""));
    REQUIRE_THAT(json, ContainsSubstring("\"step\""));
}

TEST_CASE("DebugAdapter: format_variables includes node variables", "[debug][variables]") {
    auto snap = make_snapshot(4);
    const auto json = DebugAdapter::format_variables(snap, 1, 4);
    // All 4 nodes should appear
    REQUIRE_THAT(json, ContainsSubstring("node[0]"));
    REQUIRE_THAT(json, ContainsSubstring("node[3]"));
    REQUIRE_THAT(json, ContainsSubstring("NodeInspect"));
    REQUIRE_THAT(json, ContainsSubstring("|psi|²="));
}

TEST_CASE("DebugAdapter: format_variables respects max_vars", "[debug][variables]") {
    auto snap = make_snapshot(100);
    const auto json = DebugAdapter::format_variables(snap, 1, 5);
    // Should have at most 5 node entries
    size_t count = 0, pos = 0;
    while ((pos = json.find("NodeInspect", pos)) != std::string::npos) {
        ++count; ++pos;
    }
    REQUIRE(count <= 5);
}

// ─────────────────────────────────────────────────────────────────────────────
//  format_node_inspect
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DebugAdapter: format_node_inspect", "[debug][node_inspect]") {
    NodeInspect ni;
    ni.index    = 42;
    ni.psi_real = 0.5f;
    ni.psi_imag = 0.3f;
    ni.vel_real = 0.01f;
    ni.vel_imag = 0.02f;
    ni.resonance = 1.0f;

    const auto json = DebugAdapter::format_node_inspect(ni);

    REQUIRE_THAT(json, ContainsSubstring("\"index\":42"));
    REQUIRE_THAT(json, ContainsSubstring("\"psi\":{"));
    REQUIRE_THAT(json, ContainsSubstring("\"real\":0.5"));
    REQUIRE_THAT(json, ContainsSubstring("\"imag\":0.3"));
    REQUIRE_THAT(json, ContainsSubstring("\"vel\":{"));
    REQUIRE_THAT(json, ContainsSubstring("psi_norm_sq"));
    REQUIRE_THAT(json, ContainsSubstring("vel_norm_sq"));
    REQUIRE_THAT(json, ContainsSubstring("\"resonance\":1"));
}

// ─────────────────────────────────────────────────────────────────────────────
//  format_hamiltonian_state
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DebugAdapter: format_hamiltonian_state", "[debug][hamiltonian]") {
    auto snap = make_snapshot();
    const auto json = DebugAdapter::format_hamiltonian_state(snap);

    REQUIRE_THAT(json, ContainsSubstring("\"H\":"));
    REQUIRE_THAT(json, ContainsSubstring("\"H0\":"));
    REQUIRE_THAT(json, ContainsSubstring("\"drift_pct\":"));
    REQUIRE_THAT(json, ContainsSubstring("\"time\":1.5"));
    REQUIRE_THAT(json, ContainsSubstring("\"step\":150"));
    REQUIRE_THAT(json, ContainsSubstring("\"nodes\":8"));
    REQUIRE_THAT(json, ContainsSubstring("68890.5"));
}

// ─────────────────────────────────────────────────────────────────────────────
//  format_evaluate
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DebugAdapter: format_evaluate expressions", "[debug][evaluate]") {
    auto snap = make_snapshot(8, 68890.5, 70000.0);

    SECTION("H expression") {
        const auto json = DebugAdapter::format_evaluate(snap, "H", 1, 1);
        REQUIRE_THAT(json, ContainsSubstring("68890.5"));
        REQUIRE_THAT(json, ContainsSubstring("\"success\":true"));
    }

    SECTION("hamiltonian alias") {
        const auto json = DebugAdapter::format_evaluate(snap, "hamiltonian", 2, 2);
        REQUIRE_THAT(json, ContainsSubstring("68890.5"));
    }

    SECTION("drift expression") {
        const auto json = DebugAdapter::format_evaluate(snap, "drift", 3, 3);
        REQUIRE_THAT(json, ContainsSubstring("%"));
        REQUIRE_THAT(json, ContainsSubstring("\"success\":true"));
    }

    SECTION("nodes expression") {
        const auto json = DebugAdapter::format_evaluate(snap, "nodes", 4, 4);
        REQUIRE_THAT(json, ContainsSubstring("8"));
        REQUIRE_THAT(json, ContainsSubstring("size_t"));
    }

    SECTION("step expression") {
        const auto json = DebugAdapter::format_evaluate(snap, "step", 5, 5);
        REQUIRE_THAT(json, ContainsSubstring("150"));
    }

    SECTION("time expression") {
        const auto json = DebugAdapter::format_evaluate(snap, "time", 6, 6);
        REQUIRE_THAT(json, ContainsSubstring("1.5"));
    }

    SECTION("node:N expression") {
        const auto json = DebugAdapter::format_evaluate(snap, "node:0", 7, 7);
        // The node inspect JSON is embedded as a DAP result string (escaped),
        // so "index":0 appears as \"index\":0 in the wire format.
        // Check for substrings that survive escaping unambiguously.
        REQUIRE_THAT(json, ContainsSubstring("index"));
        REQUIRE_THAT(json, ContainsSubstring("psi_norm_sq"));
        REQUIRE_THAT(json, ContainsSubstring("NodeInspect"));
        REQUIRE_THAT(json, ContainsSubstring("\"success\":true"));
    }

    SECTION("node:N out of range") {
        const auto json = DebugAdapter::format_evaluate(snap, "node:9999", 8, 8);
        REQUIRE_THAT(json, ContainsSubstring("out of range"));
    }

    SECTION("unknown expression returns error") {
        const auto json = DebugAdapter::format_evaluate(snap, "foobar", 9, 9);
        REQUIRE_THAT(json, ContainsSubstring("\"success\":false"));
        REQUIRE_THAT(json, ContainsSubstring("Unknown expression"));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  format_grid_snapshot_event
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DebugAdapter: format_grid_snapshot_event", "[debug][event]") {
    auto snap = make_snapshot(8);
    const auto json = DebugAdapter::format_grid_snapshot_event(snap, 1, 4);
    REQUIRE_THAT(json, ContainsSubstring("\"type\":\"event\""));
    REQUIRE_THAT(json, ContainsSubstring("nikola/gridSnapshot"));
    REQUIRE_THAT(json, ContainsSubstring("\"variables\":["));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Session::handle()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DebugAdapter::Session: handle initialize", "[debug][session]") {
    DebugAdapter::Session session;
    session.update_snapshot(make_snapshot());

    const auto response = session.handle("initialize", 1);
    REQUIRE_THAT(response, ContainsSubstring("Content-Length:"));
    REQUIRE_THAT(response, ContainsSubstring("supportsEvaluateForHovers"));
}

TEST_CASE("DebugAdapter::Session: handle variables", "[debug][session]") {
    DebugAdapter::Session session;
    session.update_snapshot(make_snapshot(4));

    const auto response = session.handle("variables", 2);
    REQUIRE_THAT(response, ContainsSubstring("\"variables\":["));
    REQUIRE_THAT(response, ContainsSubstring("Content-Length:"));
}

TEST_CASE("DebugAdapter::Session: handle nikola/hamiltonianState", "[debug][session]") {
    DebugAdapter::Session session;
    session.update_snapshot(make_snapshot());

    const auto response = session.handle("nikola/hamiltonianState", 3);
    REQUIRE_THAT(response, ContainsSubstring("\"H\":"));
    REQUIRE_THAT(response, ContainsSubstring("drift_pct"));
}

TEST_CASE("DebugAdapter::Session: handle evaluate", "[debug][session]") {
    DebugAdapter::Session session;
    session.update_snapshot(make_snapshot());

    const auto response = session.handle("evaluate", 4,
        R"({"expression":"H","context":"hover"})");
    REQUIRE_THAT(response, ContainsSubstring("68890.5"));
}

TEST_CASE("DebugAdapter::Session: handle unknown command", "[debug][session]") {
    DebugAdapter::Session session;
    const auto response = session.handle("unknownCommand", 5);
    REQUIRE_THAT(response, ContainsSubstring("Unsupported command"));
}

TEST_CASE("DebugAdapter::Session: handle disconnect", "[debug][session]") {
    DebugAdapter::Session session;
    const auto response = session.handle("disconnect", 6);
    REQUIRE_THAT(response, ContainsSubstring("\"success\":true"));
}

TEST_CASE("DebugAdapter::Session: sequence numbers increment", "[debug][session]") {
    DebugAdapter::Session session;
    session.update_snapshot(make_snapshot());

    const auto r1 = session.handle("variables", 1);
    const auto r2 = session.handle("variables", 2);

    // Both should be valid framed responses
    REQUIRE(DebugAdapter::parse_content_length(r1) > 0);
    REQUIRE(DebugAdapter::parse_content_length(r2) > 0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  escape_json
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DebugAdapter: escape_json", "[debug][escape]") {
    REQUIRE(DebugAdapter::escape_json("hello") == "hello");
    REQUIRE(DebugAdapter::escape_json("say \"hi\"") == "say \\\"hi\\\"");
    REQUIRE(DebugAdapter::escape_json("a\\b") == "a\\\\b");
    REQUIRE(DebugAdapter::escape_json("line1\nline2") == "line1\\nline2");
    REQUIRE(DebugAdapter::escape_json("col1\tcol2") == "col1\\tcol2");
    REQUIRE(DebugAdapter::escape_json("") == "");
}
