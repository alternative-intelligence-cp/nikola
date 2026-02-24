#pragma once
/**
 * @file include/nikola/autonomy/lookup_agent.hpp
 * @brief Phase 31 — LookupFulfillmentAgent: closes the REQUEST_LOOKUP loop.
 *
 * Architecture position:
 *
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │  NikolaNode                                                         │
 *   │    PUB :5560  nikola.v1.action  ──────────────────────────────────┐ │
 *   │    PULL:5561  ←── scored stimulus ────────────────────────────────┤ │
 *   └────────────────────────────────────────────────────────────────────┼─┘
 *                                                                        │
 *   ┌────────────────────────────────────────────────────────────────────┘
 *   │  LookupFulfillmentAgent
 *   │    SUB  → connects to 5560 (filtered on "nikola.v1.action")
 *   │    PUSH → connects to 5561
 *   │
 *   │    On each REQUEST_LOOKUP:
 *   │      1. Extract payload (query string) from action JSON
 *   │      2. Call lookup_fn(query) → raw content string
 *   │      3. Run OraclePool.evaluate(query, content) → credibility [0, 1]
 *   │      4. Serialize to stimulus envelope JSON + PUSH to NikolaNode
 *   └──────────────────────────────────────────────────────────────────────
 *
 * Stimulus envelope wire format (PUSH to NikolaNode PULL socket):
 *   {"type":"stimulus","text":"<content>","credibility":0.8700}
 *
 * NikolaNode::poll_stimulus() detects the {-prefix, parses the envelope,
 * and calls loop_.inject_stimulus(text, credibility), which scales the
 * torus injection amplitude by the credibility weight.
 *
 * Plain text stimuli (no envelope) remain fully backward-compatible:
 * NikolaNode falls through to the original inject_stimulus(text) path.
 *
 * Design:
 *   · run() blocks until stop() — intended for a background thread.
 *   · All ZMQ I/O is in the run() thread; no shared-state races.
 *   · OraclePool + lookup_fn are configured before run() is called.
 *   · parse_action_json() and to_stimulus_json() are static + public
 *     so unit tests can verify the JSON pipeline without spinning ZMQ.
 *
 * Extension points:
 *   · Register Tavily oracle: pool.add_oracle(make_shared<TavilyOracle>(key))
 *   · Register Gemini oracle: pool.add_oracle(make_shared<GeminiOracle>(key))
 *   · Set DB-first lookup:    agent.set_lookup_fn(db_then_search)
 *
 * Phase: NIK-LFA-01 (Lookup Fulfillment Agent, Phase 31)
 */

#include <nikola/autonomy/oracle_pool.hpp>
#include <nikola/infrastructure/spine.hpp>

#include <atomic>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace nikola::autonomy {

// ============================================================================
// LookupAgentConfig
// ============================================================================

/**
 * @brief Configuration for LookupFulfillmentAgent.
 *
 *   action_sub_endpoint    ZMQ endpoint to SUB-connect (NikolaNode's PUB bind).
 *   stimulus_push_endpoint ZMQ endpoint to PUSH-connect (NikolaNode's PULL bind).
 *   poll_timeout_ms        How long each ZMQ poll blocks before rechecking
 *                          the running_ flag.
 */
struct LookupAgentConfig {
    std::string action_sub_endpoint    = "tcp://localhost:5560";
    std::string stimulus_push_endpoint = "tcp://localhost:5561";
    int         poll_timeout_ms        = 100;
};

// ============================================================================
// LookupFn
// ============================================================================

/// The lookup function type.
/// Receives a query string; returns the retrieved content (possibly empty).
/// Return empty string if nothing was found — agent will not push a stimulus.
using LookupFn = std::function<std::string(const std::string& query)>;

// ============================================================================
// LookupFulfillmentAgent
// ============================================================================

/**
 * @class LookupFulfillmentAgent
 * @brief Subscribes to Nikola's action feed and fulfills REQUEST_LOOKUP actions.
 *
 * Lifecycle:
 *   1. Construct with LookupAgentConfig.
 *   2. Call set_lookup_fn() and add_oracle() (thread-safe to call before run()).
 *   3. Start run() in a background thread.
 *   4. Call stop() to terminate cleanly.
 *
 * The agent does NOT own NikolaNode — it communicates purely via ZMQ sockets.
 */
class LookupFulfillmentAgent {
public:
    explicit LookupFulfillmentAgent(const LookupAgentConfig& cfg = {})
        : cfg_(cfg)
        , sub_(spine_.make_subscriber(cfg_.action_sub_endpoint, "nikola.v1.action"))
        , push_(spine_.make_push(cfg_.stimulus_push_endpoint))
        , lookup_fn_(nullptr)
    {}

    LookupFulfillmentAgent(const LookupFulfillmentAgent&) = delete;
    LookupFulfillmentAgent& operator=(const LookupFulfillmentAgent&) = delete;
    LookupFulfillmentAgent(LookupFulfillmentAgent&&) = delete;

    // ── Configuration ─────────────────────────────────────────────────────────

    /// Register the lookup function.  Call before run().
    void set_lookup_fn(LookupFn fn) { lookup_fn_ = std::move(fn); }

    /// Add an oracle to the credibility pool.  Call before run().
    void add_oracle(std::shared_ptr<Oracle> oracle) {
        oracle_pool_.add_oracle(std::move(oracle));
    }

    const OraclePool& oracle_pool() const noexcept { return oracle_pool_; }

    // ── Control ───────────────────────────────────────────────────────────────

    /**
     * @brief Run the agent loop.  Blocks until stop() is called.
     *
     * Each iteration:
     *   1. Poll SUB socket for up to poll_timeout_ms milliseconds.
     *   2. On message: receive topic frame + body frame.
     *   3. Parse body as action JSON (REQUEST_LOOKUP → proceed, else skip).
     *   4. Fulfill: lookup → oracle score → push scored stimulus.
     */
    void run() {
        running_.store(true);

        while (running_.load()) {
            zmq_pollitem_t item{};
            item.socket = static_cast<void*>(sub_);
            item.events = ZMQ_POLLIN;

            zmq::poll(&item, 1, std::chrono::milliseconds(cfg_.poll_timeout_ms));
            if (!(item.revents & ZMQ_POLLIN)) continue;

            // Frame 1 — topic
            zmq::message_t topic_msg;
            auto t_res = sub_.recv(topic_msg);
            if (!t_res.has_value() || !topic_msg.more()) continue;

            // Frame 2 — body
            zmq::message_t body_msg;
            auto b_res = sub_.recv(body_msg);
            if (!b_res.has_value() || body_msg.size() == 0) continue;

            const std::string body(static_cast<const char*>(body_msg.data()),
                                   body_msg.size());

            const auto [type, payload] = parse_action_json(body);
            if (type != "REQUEST_LOOKUP" || payload.empty()) continue;

            fulfill_(payload);
        }
    }

    /// Signal the run loop to exit.  Thread-safe.
    void stop() noexcept { running_.store(false); }

    /// Number of successfully fulfilled lookups since construction.
    uint64_t fulfilled_count() const noexcept { return fulfilled_.load(); }

    // ── Static helpers — public for unit testing ──────────────────────────────

    /**
     * @brief Parse an action JSON string produced by NikolaNode.
     *
     * Expected format:
     *   {"tick":42,"type":"REQUEST_LOOKUP","score":0.6500,"payload":"query text"}
     *
     * @return {type_string, payload_string}; both empty on parse failure.
     *
     * Hand-rolled (no external JSON library) — consistent with the rest of the
     * nikola codebase.  Handles \" escapes inside the payload string.
     */
    static std::pair<std::string, std::string>
    parse_action_json(const std::string& json) {
        std::string type;
        std::string payload;

        // Extract "type":"VALUE"
        {
            const auto k = json.find("\"type\":\"");
            if (k != std::string::npos) {
                const auto vs = k + 8;
                const auto ve = json.find('"', vs);
                if (ve != std::string::npos)
                    type = json.substr(vs, ve - vs);
            }
        }

        // Extract "payload":"VALUE"  — respects \" escapes
        {
            const auto k = json.find("\"payload\":\"");
            if (k != std::string::npos) {
                const auto vs = k + 11;
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
                payload = val;
            }
        }

        return { type, payload };
    }

    /**
     * @brief Serialize a (text, credibility) pair to the scored stimulus envelope.
     *
     * Wire format:
     *   {"type":"stimulus","text":"<escaped content>","credibility":0.8700}
     *
     * NikolaNode::poll_stimulus() detects the '{' prefix and parses this
     * envelope to call loop_.inject_stimulus(text, credibility).
     */
    static std::string to_stimulus_json(const std::string& text, float credibility) {
        // JSON-escape the content
        std::string escaped;
        escaped.reserve(text.size() + 4);
        for (const char c : text) {
            if      (c == '"')  escaped += "\\\"";
            else if (c == '\\') escaped += "\\\\";
            else if (c == '\n') escaped += "\\n";
            else if (c == '\r') escaped += "\\r";
            else if (c == '\t') escaped += "\\t";
            else                escaped += c;
        }

        std::ostringstream o;
        o << std::fixed << std::setprecision(4);
        o << "{\"type\":\"stimulus\""
          << ",\"text\":\"" << escaped << "\""
          << ",\"credibility\":" << std::clamp(credibility, 0.f, 1.f)
          << "}";
        return o.str();
    }

private:
    // ── Lookup + oracle scoring + push ────────────────────────────────────────

    void fulfill_(const std::string& query) {
        // 1. Run lookup function (DB / search / any external source)
        const std::string content = lookup_fn_ ? lookup_fn_(query) : std::string{};
        if (content.empty()) return;  // Nothing found — don't inject silence

        // 2. Oracle pool → credibility score
        const float credibility = oracle_pool_.evaluate(query, content);

        // 3. Serialize and push scored stimulus to NikolaNode
        const std::string msg = to_stimulus_json(content, credibility);
        push_.send(zmq::buffer(msg), zmq::send_flags::none);
        ++fulfilled_;
    }

    // ── Data members ──────────────────────────────────────────────────────────

    LookupAgentConfig                   cfg_;
    nikola::infrastructure::ZmqSpine    spine_;
    zmq::socket_t                       sub_;
    zmq::socket_t                       push_;
    OraclePool                          oracle_pool_;
    LookupFn                            lookup_fn_;
    std::atomic<bool>                   running_{false};
    std::atomic<uint64_t>               fulfilled_{0};
};

} // namespace nikola::autonomy
