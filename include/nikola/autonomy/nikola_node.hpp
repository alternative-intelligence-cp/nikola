/**
 * @file include/nikola/autonomy/nikola_node.hpp
 * @brief Phase 30 — ZMQ Spine: Nikola as a distributed autonomous node.
 *
 * NikolaNode wraps the full cognitive stack (CognitiveTorus + AutonomyEngine +
 * DecisionLoop) and binds it to a ZeroMQ message bus:
 *
 *   PUB socket (pub_endpoint, default tcp://-any-:5560)
 *     -> nikola.v1.action  — every non-SILENT DecisionResult as JSON
 *     -> nikola.v1.state   — NikolaState snapshot every N ticks as JSON
 *
 *   PULL socket (pull_endpoint, default tcp://-any-:5561)
 *     <- raw UTF-8 text    — injected directly into DecisionLoop::inject_stimulus()
 *
 * Design rules:
 *   · Core cognitive stack is UNCHANGED — NikolaNode is a pure wrapper.
 *   · ZMQ I/O is non-blocking (dontwait recv on PULL each tick).
 *   · JSON is hand-rolled (no external JSON library required).
 *   · External callers interact only via ZMQ sockets — no shared memory,
 *     no function call coupling.
 *
 * Message schemas:
 *
 *   action  (nikola.v1.action):
 *     {"tick":42,"type":"EMIT_THOUGHT","score":0.7200,"payload":"Some thought"}
 *
 *   state   (nikola.v1.state):
 *     {"tick":42,"time":21.0000,"energy":1.3500,"dopamine":0.8800,
 *      "td_error":-0.0740,"atp":0.9900,"boredom":0.5000,"entropy":0.7100}
 *
 * Usage:
 *   NikolaNodeConfig cfg;    // pub=tcp://0.0.0.0:5560 pull=tcp://0.0.0.0:5561
 *   cfg.pub_endpoint  = "tcp://0.0.0.0:5560";
 *   cfg.pull_endpoint = "tcp://0.0.0.0:5561";
 *
 *   NikolaNode node(cfg);
 *   node.on_action = [](const DecisionResult& r) { ... };
 *   node.inject_stimulus("hello nikola");
 *   node.run();       // blocks until stop() or max_ticks reached
 *
 * Default endpoints:  pub=tcp://0.0.0.0:5560  pull=tcp://0.0.0.0:5561
 */

#pragma once

#include <nikola/infrastructure/spine.hpp>
#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/autonomy/decision_loop.hpp>

#include <algorithm>
#include <atomic>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace nikola::autonomy {

// ============================================================================
// NikolaNodeConfig
// ============================================================================

/**
 * @brief Configuration for NikolaNode.
 *
 *   pub_endpoint           ZMQ PUB bind address (publishes actions + state)
 *   pull_endpoint          ZMQ PULL bind address (receives stimuli)
 *   decision_config        Forwarded to DecisionLoop
 *   state_publish_interval Publish NikolaState every N ticks; 0 = off
 */
struct NikolaNodeConfig {
    std::string pub_endpoint          = "tcp://*:5560";
    std::string pull_endpoint         = "tcp://*:5561";
    DecisionLoopConfig decision_config{};
    int state_publish_interval        = 10;  ///< 0 = never publish state
};

// ============================================================================
// NikolaNode
// ============================================================================

/**
 * @class NikolaNode
 * @brief Full cognitive stack + ZMQ I/O — Nikola as a network node.
 *
 * Ownership: NikolaNode owns all subsystems.
 * Thread safety: run() is intended to be called from one thread only.
 *   stop() may be called from any thread (atomic flag).
 *
 * Extend via callbacks:
 *   node.on_action = [](const DecisionResult& r) { log(r); };
 *   node.on_tick   = [](const NikolaState& s, uint64_t t) { stats(s); };
 */
class NikolaNode {
public:
    // ── Optional user-facing callbacks (mirror DecisionLoop's API) ────────────
    std::function<void(const DecisionResult&)>          on_action;
    std::function<void(const NikolaState&, uint64_t)>   on_tick;

    // ── Construction ─────────────────────────────────────────────────────────

    explicit NikolaNode(const NikolaNodeConfig& cfg)
        : cfg_(cfg)
        , spine_()
#ifdef NIKOLA_HAS_ORT
        , torus_(3, NIKOLA_ORT_TOKENIZER_PATH, NIKOLA_ORT_MODEL_PATH,
                 /*pilot_dim=*/3, /*amplitude=*/0.01f)
#else
        , torus_(3, /*pilot_dim=*/3, /*amplitude=*/0.01f)
#endif
        , engine_([]() {
              AutonomyConfig c;
              c.enable_dream_weave = false;
              return c;
          }())
        , loop_(torus_, engine_, cfg.decision_config)
        , pub_(spine_.make_publisher(cfg.pub_endpoint))
        , pull_(spine_.make_pull(cfg.pull_endpoint))
    {
        // Wire DecisionLoop callbacks into ZMQ publishing
        loop_.on_action = [this](const DecisionResult& r) {
            publish_action(r);
            if (on_action) on_action(r);
        };

        loop_.on_tick = [this](const NikolaState& s) {
            const uint64_t tick = loop_.tick_count();
            if (cfg_.state_publish_interval > 0 &&
                tick % static_cast<uint64_t>(cfg_.state_publish_interval) == 0)
            {
                publish_state(s, tick);
            }
            if (on_tick) on_tick(s, tick);
        };
    }

    NikolaNode(const NikolaNode&)            = delete;
    NikolaNode& operator=(const NikolaNode&) = delete;
    NikolaNode(NikolaNode&&)                 = delete;

    // ── Control ───────────────────────────────────────────────────────────────

    /**
     * @brief Signal the run loop to exit.  Safe to call from any thread.
     */
    void stop() noexcept { running_.store(false); }

    /**
     * @brief Inject a plain text stimulus directly into the DecisionLoop.
     * May be called from the run() thread before or during run().
     */
    void inject_stimulus(const std::string& text) {
        loop_.inject_stimulus(text);
    }

    /**
     * @brief Inject an oracle-scored stimulus.
     *
     * Credibility [0.0–1.0] scales the torus injection amplitude:
     *   1.0 = full strength (Nit{4}), 0.0 = no injection.
     *
     * This is called automatically by poll_stimulus() when it receives a
     * scored stimulus envelope from the LookupFulfillmentAgent.  It can
     * also be called directly for testing.
     */
    void inject_scored_stimulus(const std::string& text, float credibility) {
        loop_.inject_stimulus(text, credibility);
    }

    uint64_t           tick_count()  const noexcept { return loop_.tick_count(); }
    const NikolaState& last_state()  const noexcept { return loop_.last_state(); }

    // ── Main loop ─────────────────────────────────────────────────────────────

    /**
     * @brief Run the cognitive tick loop.
     *
     * Each tick:
     *   1. poll_stimulus()  — check PULL socket for incoming text (non-blocking)
     *   2. loop_.tick()     — run cognitive + autonomy + decision
     *   3. ZMQ callbacks fire automatically (wired in constructor)
     *
     * Returns when stop() is called or max_ticks is reached.
     *
     * @param max_ticks  -1 = run forever, >0 = stop after N ticks.
     */
    void run(int64_t max_ticks = -1) {
        running_.store(true);
        int64_t n = 0;
        while (running_.load()) {
            poll_stimulus();
            loop_.tick();
            ++n;
            if (max_ticks > 0 && n >= max_ticks) break;
        }
        running_.store(false);
    }

    // ── JSON serialization — public for tests and external tooling ────────────

    /**
     * @brief Serialize a DecisionResult to the nikola.v1.action JSON format.
     * {"tick":42,"type":"EMIT_THOUGHT","score":0.7200,"payload":"Some thought"}
     */
    static std::string action_to_json(const DecisionResult& r, uint64_t tick) {
        std::ostringstream o;
        o << std::fixed << std::setprecision(4);
        o << "{\"tick\":"    << tick
          << ",\"type\":\""  << action_name(r.type) << "\""
          << ",\"score\":"   << r.score
          << ",\"payload\":" << json_str(r.payload)
          << "}";
        return o.str();
    }

    /**
     * @brief Serialize a NikolaState to the nikola.v1.state JSON format.
     * {"tick":42,"time":21.0000,"energy":1.3500,"dopamine":0.8800,...}
     */
    static std::string state_to_json(const NikolaState& s, uint64_t tick) {
        std::ostringstream o;
        o << std::fixed << std::setprecision(4);
        o << "{\"tick\":"       << tick
          << ",\"time\":"       << s.time
          << ",\"energy\":"     << s.torus_energy
          << ",\"dopamine\":"   << s.dopamine
          << ",\"td_error\":"   << s.td_error
          << ",\"atp\":"        << s.atp
          << ",\"boredom\":"    << s.boredom
          << ",\"entropy\":"    << s.entropy
          << "}";
        return o.str();
    }

    /**
     * @brief Parse a scored stimulus JSON envelope from the PULL socket.
     *
     * Produced by LookupFulfillmentAgent::to_stimulus_json().
     * Expected format:
     *   {"type":"stimulus","text":"<content>","credibility":0.8700}
     *
     * @return true and populates text/credibility on success; false otherwise.
     * Public so tests and external agents can validate the wire format.
     */
    static bool parse_scored_stimulus(const std::string& json,
                                       std::string& text, float& credibility) {
        // Must be tagged as a stimulus envelope
        if (json.find("\"type\":\"stimulus\"") == std::string::npos) return false;

        // Extract "text":"..."
        {
            const auto k = json.find("\"text\":\"");
            if (k == std::string::npos) return false;
            const auto vs = k + 8;
            std::string val;
            bool escaped = false;
            for (auto i = vs; i < json.size(); ++i) {
                const char c = json[i];
                if (escaped) {
                    if      (c == '"')  val += '"';
                    else if (c == '\\') val += '\\';
                    else if (c == 'n')  val += '\n';
                    else                val += c;
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '"') {
                    break;
                } else {
                    val += c;
                }
            }
            text = val;
        }

        // Extract "credibility":N.NNNN
        {
            const auto k = json.find("\"credibility\":");
            if (k == std::string::npos) return false;
            const auto vs = k + 15;
            try {
                credibility = std::stof(json.substr(vs));
            } catch (...) {
                return false;
            }
        }

        return true;
    }

private:
    // ── Poll PULL socket for incoming stimuli (non-blocking) ──────────────────

    void poll_stimulus() {
        zmq::message_t msg;
        // recv_result_t is optional<size_t> — use msg.size() for the payload length
        const auto res = pull_.recv(msg, zmq::recv_flags::dontwait);
        if (res.has_value() && msg.size() > 0) {
            const std::string raw(static_cast<const char*>(msg.data()), msg.size());

            // Check for scored stimulus envelope: {"type":"stimulus",...}
            // Produced by LookupFulfillmentAgent after oracle scoring.
            if (raw.size() > 2 && raw[0] == '{') {
                std::string text;
                float credibility = -1.f;
                if (parse_scored_stimulus(raw, text, credibility) && !text.empty()) {
                    loop_.inject_stimulus(text, credibility);
                    return;
                }
            }

            // Plain text stimulus (backward compatible — no credibility weight).
            loop_.inject_stimulus(raw);
        }
    }

    // ── Publish helpers ───────────────────────────────────────────────────────

    void publish_action(const DecisionResult& r) {
        const std::string json = action_to_json(r, loop_.tick_count());
        nikola::infrastructure::ZmqSpine::publish(
            pub_, "action", json.data(), json.size());
    }

    void publish_state(const NikolaState& s, uint64_t tick) {
        const std::string json = state_to_json(s, tick);
        nikola::infrastructure::ZmqSpine::publish(
            pub_, "state", json.data(), json.size());
    }

    // ── Minimal JSON string escaping ──────────────────────────────────────────

    static std::string json_str(const std::string& s) {
        std::string out;
        out.reserve(s.size() + 2);
        out += '"';
        for (const char c : s) {
            if      (c == '"')  out += "\\\"";
            else if (c == '\\') out += "\\\\";
            else if (c == '\n') out += "\\n";
            else if (c == '\r') out += "\\r";
            else if (c == '\t') out += "\\t";
            else                out += c;
        }
        out += '"';
        return out;
    }

    // ── Data members — construction order matches initialiser list ────────────

    NikolaNodeConfig                    cfg_;
    nikola::infrastructure::ZmqSpine    spine_;
    nikola::cognitive::CognitiveTorus   torus_;
    nikola::autonomy::AutonomyEngine    engine_;
    nikola::autonomy::DecisionLoop      loop_;
    zmq::socket_t                       pub_;
    zmq::socket_t                       pull_;
    std::atomic<bool>                   running_{false};
};

} // namespace nikola::autonomy
