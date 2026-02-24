// ============================================================================
// escalation_log.hpp          Phase 32 — ESCALATE action type
// ============================================================================
//
// Tamper-evident evidence log for ESCALATE actions.
//
// When Nikola's value function dips deep below the alive prior the
// ESCALATE action fires.  The EscalationLog provides an append-only
// record of every such event, hash-chained (FNV-1a 64-bit) so that any
// post-hoc deletion or modification is detectable.
//
// Design goals:
//   • No external dependencies — FNV-1a implemented inline.
//   • Self-contained record — stimulus text, td_error, and tick count
//     are embedded so the record can stand alone as evidence.
//   • Optional ZMQ forwarding via EscalationAgent.
// ============================================================================

#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nikola/infrastructure/spine.hpp"  // ZmqSpine

namespace nikola::autonomy {

// ============================================================================
// FNV-1a 64-bit — no external deps
// ============================================================================
namespace detail {

constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
constexpr uint64_t FNV_PRIME  = 1099511628211ULL;

inline uint64_t fnv1a_64(const std::string& s, uint64_t seed = FNV_OFFSET) noexcept
{
    uint64_t h = seed;
    for (const unsigned char c : s) {
        h ^= static_cast<uint64_t>(c);
        h *= FNV_PRIME;
    }
    return h;
}

inline std::string u64_hex(uint64_t v)
{
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  static_cast<unsigned long long>(v));
    return std::string(buf);
}

} // namespace detail


// ============================================================================
// EvidenceRecord
// ============================================================================

struct EvidenceRecord {
    uint64_t    tick          {0};
    uint64_t    wall_time_ms  {0};
    std::string stimulus;          // what was asked when ESCALATE fired
    float       td_error      {0.f};
    std::string payload;           // raw payload from DecisionLoop::build_payload()
    std::string prev_hash;         // hash of the previous record ("0" for first)
    std::string self_hash;         // hash of this record's content (computed on insert)

    // Human-readable serialisation used as input to self_hash computation.
    std::string to_canonical_string() const
    {
        std::ostringstream ss;
        ss << "tick="         << tick
           << "|wall_ms="     << wall_time_ms
           << "|stimulus=["   << stimulus << "]"
           << "|td="          << td_error
           << "|payload=["    << payload  << "]"
           << "|prev_hash="   << prev_hash;
        return ss.str();
    }

    // JSON-like serialisation for log files / forwarding.
    std::string to_json() const
    {
        std::ostringstream ss;
        ss << "{"
           << "\"tick\":"         << tick           << ","
           << "\"wall_ms\":"      << wall_time_ms   << ","
           << "\"stimulus\":\""   << stimulus       << "\","
           << "\"td_error\":"     << td_error       << ","
           << "\"payload\":\""    << payload        << "\","
           << "\"prev_hash\":\""  << prev_hash      << "\","
           << "\"self_hash\":\""  << self_hash      << "\""
           << "}";
        return ss.str();
    }
};


// ============================================================================
// EscalationLog — append-only, hash-chained
// ============================================================================

class EscalationLog {
public:
    EscalationLog() = default;

    // Append a new escalation record.  prev_hash and self_hash are computed
    // automatically from the chain state — callers supply only the domain data.
    void record(uint64_t    tick,
                std::string stimulus,
                float       td_error,
                std::string payload)
    {
        using namespace std::chrono;
        const uint64_t wall_ms = static_cast<uint64_t>(
            duration_cast<milliseconds>(
                system_clock::now().time_since_epoch()).count());

        EvidenceRecord r;
        r.tick         = tick;
        r.wall_time_ms = wall_ms;
        r.stimulus     = std::move(stimulus);
        r.td_error     = td_error;
        r.payload      = std::move(payload);
        r.prev_hash    = records_.empty() ? "0" : records_.back().self_hash;

        // Hash this record's canonical form — seeded with the chain so far
        // so even identical events produce different hashes.
        const std::string canon = r.to_canonical_string();
        const uint64_t seed = records_.empty()
                                ? detail::FNV_OFFSET
                                : detail::fnv1a_64(r.prev_hash);
        r.self_hash = detail::u64_hex(detail::fnv1a_64(canon, seed));

        records_.push_back(std::move(r));
    }

    std::size_t         size()  const noexcept { return records_.size(); }
    bool                empty() const noexcept { return records_.empty(); }
    const EvidenceRecord& at(std::size_t i) const { return records_.at(i); }

    // Verify the entire hash chain.  Returns true iff every record's
    // self_hash matches a fresh recomputation from its canonical form.
    bool verify_chain() const
    {
        std::string expected_prev = "0";
        for (const auto& r : records_) {
            // prev_hash check
            if (r.prev_hash != expected_prev) return false;

            // Recompute self_hash
            const uint64_t seed = (r.prev_hash == "0")
                                    ? detail::FNV_OFFSET
                                    : detail::fnv1a_64(r.prev_hash);
            const std::string canon    = r.to_canonical_string();
            const std::string expected = detail::u64_hex(
                detail::fnv1a_64(canon, seed));
            if (r.self_hash != expected) return false;

            expected_prev = r.self_hash;
        }
        return true;
    }

    // Append all records as JSON lines to a file (one JSON object per line).
    void serialize_to_file(const std::string& path) const
    {
        std::ofstream ofs(path, std::ios::app);
        if (!ofs) throw std::runtime_error("EscalationLog: cannot open " + path);
        for (const auto& r : records_) {
            ofs << r.to_json() << "\n";
        }
    }

private:
    std::vector<EvidenceRecord> records_;
};


// ============================================================================
// EscalationAgent — subscriber that logs every ESCALATE action
// ============================================================================
//
// Usage:
//   EscalationAgent agent(context);
//   agent.run();        // blocking loop on a worker thread
//   agent.stop();       // signal loop to exit
//   const auto& log = agent.log();   // inspect records
//
// The agent subscribes to nikola.v1.action on a configurable endpoint,
// parses incoming JSON frames, and records any frame whose "type" is
// "ESCALATE".  Optionally it can PUSH the JSON record to a forwarding
// endpoint (e.g. an operator dashboard).

class EscalationAgent {
public:
    explicit EscalationAgent(zmq::context_t& ctx,
                             std::string action_endpoint = "tcp://127.0.0.1:5560",
                             std::string forward_endpoint = "")
        : sub_(ctx, zmq::socket_type::sub)
        , forward_endpoint_(std::move(forward_endpoint))
        , ctx_(ctx)
    {
        sub_.connect(action_endpoint);
        sub_.set(zmq::sockopt::subscribe,
                 nikola::infrastructure::make_topic("action"));
    }

    // Enable optional forwarding push socket.
    void set_forward_endpoint(const std::string& ep)
    {
        forward_endpoint_ = ep;
    }

    void run()
    {
        running_ = true;

        // Lazily create push socket in run() so it shares the same thread
        // as the recv loop (ZMQ sockets are not thread-safe).
        std::unique_ptr<zmq::socket_t> push;
        if (!forward_endpoint_.empty()) {
            push = std::make_unique<zmq::socket_t>(ctx_, zmq::socket_type::push);
            push->connect(forward_endpoint_);
        }

        while (running_) {
            zmq::message_t topic_frame;
            zmq::message_t body_frame;

            // Non-blocking poll with 50 ms timeout so stop() is responsive.
            zmq::pollitem_t item{ static_cast<void*>(sub_), 0,
                                  ZMQ_POLLIN, 0 };
            zmq::poll(&item, 1, std::chrono::milliseconds(50));
            if (!(item.revents & ZMQ_POLLIN)) continue;

            auto res1 = sub_.recv(topic_frame, zmq::recv_flags::none);
            if (!res1) continue;
            auto res2 = sub_.recv(body_frame, zmq::recv_flags::none);
            if (!res2) continue;

            const std::string body(static_cast<char*>(body_frame.data()),
                                   body_frame.size());
            if (!is_escalate(body)) continue;

            // Parse minimal fields straight out of the JSON text.
            // We intentionally avoid pulling in a JSON library here.
            const uint64_t    tick     = parse_uint_field(body, "tick");
            const float       td       = parse_float_field(body, "td_error");
            const std::string stimulus = parse_str_field(body, "stimulus");
            const std::string payload  = parse_str_field(body, "payload");

            log_.record(tick, stimulus, td, payload);

            if (push) {
                const std::string rec_json = log_.at(log_.size() - 1).to_json();
                push->send(zmq::buffer(rec_json), zmq::send_flags::none);
            }
        }
    }

    void stop() noexcept { running_ = false; }

    const EscalationLog& log() const noexcept { return log_; }

    // ── Parse helpers (minimal, no external JSON dep) ─────────────────────
    static bool is_escalate(const std::string& json)
    {
        // Match  "type":"ESCALATE"  anywhere in the string.
        const std::string needle = "\"ESCALATE\"";
        return json.find(needle) != std::string::npos;
    }

    static uint64_t parse_uint_field(const std::string& json,
                                     const std::string& key)
    {
        const std::string k = "\"" + key + "\":";
        const auto pos = json.find(k);
        if (pos == std::string::npos) return 0;
        const auto start = pos + k.size();
        try { return std::stoull(json.substr(start, 20)); } catch (...) { return 0; }
    }

    static float parse_float_field(const std::string& json,
                                   const std::string& key)
    {
        const std::string k = "\"" + key + "\":";
        const auto pos = json.find(k);
        if (pos == std::string::npos) return 0.f;
        const auto start = pos + k.size();
        try { return std::stof(json.substr(start, 20)); } catch (...) { return 0.f; }
    }

    static std::string parse_str_field(const std::string& json,
                                       const std::string& key)
    {
        const std::string k = "\"" + key + "\":\"";
        const auto pos = json.find(k);
        if (pos == std::string::npos) return "";
        const auto start = pos + k.size();
        const auto end   = json.find('"', start);
        if (end == std::string::npos) return "";
        return json.substr(start, end - start);
    }

private:
    zmq::socket_t   sub_;
    std::string     forward_endpoint_;
    zmq::context_t& ctx_;
    EscalationLog   log_;
    std::atomic<bool> running_{false};
};

} // namespace nikola::autonomy
