/**
 * @file cli/stream_emitter.hpp
 * @brief Phase 117 — nikola-run --stream mode: line-buffered EMIT_THOUGHT output.
 *
 * StreamEmitter wraps an std::ostream and formats DecisionResult events for
 * real-time (streaming) output.  It is header-only and has no I/O side effects
 * beyond writing to the provided ostream.
 *
 * Used by nikola_run.cpp to wire loop.on_action to immediate stdout output when
 * the --stream flag is active.
 *
 * Modes
 * -----
 *   Plain text  (json_mode=false):
 *     Nikola: <payload>\n          (quiet=false)
 *     <payload>\n                  (quiet=true)
 *
 *   JSON / NDJSON (json_mode=true):
 *     {"type":"EMIT_THOUGHT","thought":"<escaped_payload>"}\n
 *     (one JSON object per emitted event; UTF-8 safe)
 *
 * Filtering
 * ---------
 *   emit_all=false (default): only EMIT_THOUGHT actions are emitted.
 *   emit_all=true            : all non-SILENT actions are emitted.
 *
 * Thread safety: NOT thread-safe.  Use from a single thread (the tick thread).
 */
#pragma once

#include <nikola/autonomy/decision_loop.hpp>   // DecisionResult, ActionType, action_name

#include <ostream>
#include <string>
#include <string_view>

namespace nikola::cli {

// ---------------------------------------------------------------------------
// json_escape — pure helper used in both StreamEmitter and nikola_run.cpp
// ---------------------------------------------------------------------------

/**
 * @brief Escape a string for embedding in a JSON string literal.
 *
 * Escapes: " → \\", \\ → \\\\, \\n → \\n, \\r → \\r, \\t → \\t.
 * Other control characters (0x01–0x1F) are emitted as \\uXXXX.
 */
[[nodiscard]] inline std::string json_escape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20u) {
                    // Control character → \uXXXX
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                                  static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// StreamEmitter
// ---------------------------------------------------------------------------

/**
 * @brief Formats and flushes DecisionResult events to an output stream.
 *
 * Typical usage in nikola_run.cpp:
 * @code
 *   StreamEmitter emitter(std::cout, cfg.json_out, cfg.quiet, cfg.emit_all);
 *   loop.on_action = [&](const autonomy::DecisionResult& r) {
 *       emitter.emit(r);
 *   };
 *   for (int t = 0; t < max_ticks; ++t) loop.tick();
 * @endcode
 */
class StreamEmitter {
public:
    /**
     * @brief Construct.
     *
     * @param out       Destination stream (e.g. std::cout, std::ostringstream).
     * @param json_mode If true, emit NDJSON lines.
     * @param quiet     If true, suppress "Nikola: " prefix in plain-text mode.
     * @param emit_all  If true, emit all non-SILENT actions; otherwise only
     *                  EMIT_THOUGHT.
     */
    StreamEmitter(std::ostream& out,
                  bool          json_mode = false,
                  bool          quiet     = false,
                  bool          emit_all  = false) noexcept
        : out_(out), json_mode_(json_mode), quiet_(quiet), emit_all_(emit_all)
    {}

    /**
     * @brief Emit a single decision result if it meets the filter criteria.
     *
     * Writes a complete line (terminated with \\n) and flushes the stream.
     * Increments emit_count() if the result was written.
     *
     * @param r  Result from DecisionLoop::tick().
     */
    void emit(const autonomy::DecisionResult& r) {
        using autonomy::ActionType;

        const bool interesting =
            (r.type == ActionType::EMIT_THOUGHT) ||
            (emit_all_ && r.type != ActionType::SILENT);

        if (!interesting || r.payload.empty())
            return;

        if (json_mode_) {
            out_ << "{\"type\":\""
                 << autonomy::action_name(r.type)
                 << "\",\"thought\":\""
                 << json_escape(r.payload)
                 << "\"}\n"
                 << std::flush;
        } else {
            if (!quiet_) {
                out_ << "Nikola: ";
            }
            out_ << r.payload << "\n" << std::flush;
        }

        last_payload_ = r.payload;
        ++emit_count_;
    }

    // ------------------------------------------------------------------ query

    /// Number of events written since construction (or last reset()).
    [[nodiscard]] int emit_count() const noexcept { return emit_count_; }

    /// Payload of the most recently emitted event.  Empty if nothing emitted yet.
    [[nodiscard]] const std::string& last_payload() const noexcept {
        return last_payload_;
    }

    /// True if at least one event has been emitted.
    [[nodiscard]] bool has_output() const noexcept { return emit_count_ > 0; }

    /// Reset counters (stream content is NOT cleared — that is ostream's job).
    void reset() noexcept {
        emit_count_ = 0;
        last_payload_.clear();
    }

private:
    std::ostream& out_;
    bool          json_mode_;
    bool          quiet_;
    bool          emit_all_;
    int           emit_count_ = 0;
    std::string   last_payload_;
};

} // namespace nikola::cli
