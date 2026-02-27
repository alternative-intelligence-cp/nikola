/**
 * @file phase117_streaming_output_test.cpp
 * @brief Phase 117 — nikola-run --stream mode: StreamEmitter and json_escape.
 *
 * Tests are purely unit-level: no real CognitiveTorus or DecisionLoop is
 * constructed.  DecisionResult structs are built by hand so that every code
 * path in StreamEmitter and json_escape() can be exercised in isolation.
 *
 * Test map  (18 test cases)
 * ─────────────────────────
 *  [P117/init]        initial state of a fresh StreamEmitter
 *  [P117/filter]      filtering logic (SILENT, EMIT_THOUGHT, emit_all)
 *  [P117/plain]       plain-text output format (normal and quiet)
 *  [P117/json]        JSON / NDJSON output format
 *  [P117/escape]      json_escape() correctness
 *  [P117/query]       last_payload, has_output, emit_count, reset
 */

#include <nikola/cli/stream_emitter.hpp>
#include <nikola/autonomy/decision_loop.hpp>

#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

using nikola::autonomy::ActionType;
using nikola::autonomy::DecisionResult;
using nikola::cli::StreamEmitter;
using nikola::cli::json_escape;

/// Build a minimal DecisionResult from type + payload.
static DecisionResult make_result(ActionType type, std::string payload,
                                  float score = 0.5f)
{
    DecisionResult r;
    r.type    = type;
    r.score   = score;
    r.payload = std::move(payload);
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// [P117/init]  Initial state
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase117 StreamEmitter starts with zero emit_count", "[Phase117][P117/init]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss);
    CHECK(emitter.emit_count() == 0);
}

TEST_CASE("Phase117 StreamEmitter has_output is false before any emit", "[Phase117][P117/init]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss);
    CHECK_FALSE(emitter.has_output());
}

TEST_CASE("Phase117 StreamEmitter last_payload is empty before any emit", "[Phase117][P117/init]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss);
    CHECK(emitter.last_payload().empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// [P117/filter]  SILENT is always suppressed; emit_all controls others
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase117 SILENT action not emitted (no payload)", "[Phase117][P117/filter]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss);
    emitter.emit(make_result(ActionType::SILENT, ""));
    CHECK(emitter.emit_count() == 0);
    CHECK(oss.str().empty());
}

TEST_CASE("Phase117 SILENT action not emitted even with non-empty payload", "[Phase117][P117/filter]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss);
    emitter.emit(make_result(ActionType::SILENT, "sneaky payload"));
    CHECK(emitter.emit_count() == 0);
    CHECK(oss.str().empty());
}

TEST_CASE("Phase117 EMIT_THOUGHT is always emitted (emit_all=false default)", "[Phase117][P117/filter]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss);
    emitter.emit(make_result(ActionType::EMIT_THOUGHT, "Hello"));
    CHECK(emitter.emit_count() == 1);
}

TEST_CASE("Phase117 STORE_MEMORY suppressed when emit_all=false", "[Phase117][P117/filter]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss, /*json_mode=*/false, /*quiet=*/false, /*emit_all=*/false);
    emitter.emit(make_result(ActionType::STORE_MEMORY, "remember this"));
    CHECK(emitter.emit_count() == 0);
    CHECK(oss.str().empty());
}

TEST_CASE("Phase117 STORE_MEMORY emitted when emit_all=true", "[Phase117][P117/filter]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss, /*json_mode=*/false, /*quiet=*/false, /*emit_all=*/true);
    emitter.emit(make_result(ActionType::STORE_MEMORY, "remember this"));
    CHECK(emitter.emit_count() == 1);
}

TEST_CASE("Phase117 empty payload suppresses emit even for EMIT_THOUGHT", "[Phase117][P117/filter]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss);
    emitter.emit(make_result(ActionType::EMIT_THOUGHT, ""));
    CHECK(emitter.emit_count() == 0);
    CHECK(oss.str().empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// [P117/plain]  Plain-text output format
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase117 plain-text normal mode: 'Nikola: <payload>\\n'", "[Phase117][P117/plain]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss, /*json_mode=*/false, /*quiet=*/false);
    emitter.emit(make_result(ActionType::EMIT_THOUGHT, "Hello world"));
    CHECK(oss.str() == "Nikola: Hello world\n");
}

TEST_CASE("Phase117 plain-text quiet mode: '<payload>\\n' (no prefix)", "[Phase117][P117/plain]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss, /*json_mode=*/false, /*quiet=*/true);
    emitter.emit(make_result(ActionType::EMIT_THOUGHT, "Hello world"));
    CHECK(oss.str() == "Hello world\n");
}

TEST_CASE("Phase117 plain-text multi-emit appends each thought on its own line", "[Phase117][P117/plain]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss, /*json_mode=*/false, /*quiet=*/true);
    emitter.emit(make_result(ActionType::EMIT_THOUGHT, "first"));
    emitter.emit(make_result(ActionType::EMIT_THOUGHT, "second"));
    const std::string out = oss.str();
    CHECK(out.find("first\n")  != std::string::npos);
    CHECK(out.find("second\n") != std::string::npos);
    CHECK(emitter.emit_count() == 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// [P117/json]  JSON / NDJSON output format
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase117 JSON mode: produces NDJSON object with type and thought fields", "[Phase117][P117/json]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss, /*json_mode=*/true);
    emitter.emit(make_result(ActionType::EMIT_THOUGHT, "deep thought"));
    const std::string out = oss.str();
    CHECK(out.find("\"type\":\"EMIT_THOUGHT\"") != std::string::npos);
    CHECK(out.find("\"thought\":\"deep thought\"") != std::string::npos);
    CHECK(out.back() == '\n');
}

TEST_CASE("Phase117 JSON mode: one complete JSON object per line", "[Phase117][P117/json]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss, /*json_mode=*/true);
    emitter.emit(make_result(ActionType::EMIT_THOUGHT, "line1"));
    emitter.emit(make_result(ActionType::EMIT_THOUGHT, "line2"));
    const std::string out = oss.str();
    // Each line must start with '{' (NDJSON convention)
    CHECK(out.rfind('{', 0)          != std::string::npos);
    // Should contain two newline-terminated objects
    int newlines = 0;
    for (char c : out) if (c == '\n') ++newlines;
    CHECK(newlines == 2);
}

TEST_CASE("Phase117 JSON mode: special chars in payload are escaped", "[Phase117][P117/json]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss, /*json_mode=*/true);
    emitter.emit(make_result(ActionType::EMIT_THOUGHT, "say \"hi\""));
    const std::string out = oss.str();
    // The literal double-quote must be backslash-escaped in the output
    CHECK(out.find("\\\"hi\\\"") != std::string::npos);
    // The raw unescaped pair "" must not appear inside the thought value
    // (the outer {"thought":"..."} structure does use quotes, so we only
    //  check that the literal 4-char sequence  :"say " isn't there)
    CHECK(out.find("\"say \"") == std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────────────
// [P117/escape]  json_escape() correctness
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase117 json_escape: double-quote → backslash-quote", "[Phase117][P117/escape]")
{
    CHECK(json_escape("\"hello\"") == "\\\"hello\\\"");
}

TEST_CASE("Phase117 json_escape: backslash → double-backslash", "[Phase117][P117/escape]")
{
    CHECK(json_escape("a\\b") == "a\\\\b");
}

TEST_CASE("Phase117 json_escape: newline → \\n literal", "[Phase117][P117/escape]")
{
    CHECK(json_escape("a\nb") == "a\\nb");
}

TEST_CASE("Phase117 json_escape: carriage-return → \\r literal", "[Phase117][P117/escape]")
{
    CHECK(json_escape("a\rb") == "a\\rb");
}

TEST_CASE("Phase117 json_escape: tab → \\t literal", "[Phase117][P117/escape]")
{
    CHECK(json_escape("a\tb") == "a\\tb");
}

TEST_CASE("Phase117 json_escape: control char \\x01 → \\u0001", "[Phase117][P117/escape]")
{
    CHECK(json_escape(std::string(1, '\x01')) == "\\u0001");
}

TEST_CASE("Phase117 json_escape: control char \\x1f → \\u001f", "[Phase117][P117/escape]")
{
    CHECK(json_escape(std::string(1, '\x1f')) == "\\u001f");
}

TEST_CASE("Phase117 json_escape: plain ASCII unchanged", "[Phase117][P117/escape]")
{
    CHECK(json_escape("hello world 123!") == "hello world 123!");
}

TEST_CASE("Phase117 json_escape: empty string stays empty", "[Phase117][P117/escape]")
{
    CHECK(json_escape("") == "");
}

// ─────────────────────────────────────────────────────────────────────────────
// [P117/query]  State query methods
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase117 last_payload returns payload of most recently emitted event", "[Phase117][P117/query]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss);
    emitter.emit(make_result(ActionType::EMIT_THOUGHT, "first"));
    emitter.emit(make_result(ActionType::EMIT_THOUGHT, "second"));
    CHECK(emitter.last_payload() == "second");
}

TEST_CASE("Phase117 has_output becomes true after first successful emit", "[Phase117][P117/query]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss);
    CHECK_FALSE(emitter.has_output());
    emitter.emit(make_result(ActionType::EMIT_THOUGHT, "ping"));
    CHECK(emitter.has_output());
}

TEST_CASE("Phase117 reset clears emit_count and last_payload", "[Phase117][P117/query]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss);
    emitter.emit(make_result(ActionType::EMIT_THOUGHT, "data"));
    REQUIRE(emitter.emit_count() == 1);
    emitter.reset();
    CHECK(emitter.emit_count()   == 0);
    CHECK(emitter.last_payload() == "");
    CHECK_FALSE(emitter.has_output());
}

TEST_CASE("Phase117 emit_count accumulates across multiple emits", "[Phase117][P117/query]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss);
    for (int i = 0; i < 5; ++i)
        emitter.emit(make_result(ActionType::EMIT_THOUGHT, "x"));
    CHECK(emitter.emit_count() == 5);
}

TEST_CASE("Phase117 SILENT between EMIT_THOUGHT does not increment count", "[Phase117][P117/query]")
{
    std::ostringstream oss;
    StreamEmitter emitter(oss);
    emitter.emit(make_result(ActionType::EMIT_THOUGHT, "a"));
    emitter.emit(make_result(ActionType::SILENT,       ""));
    emitter.emit(make_result(ActionType::EMIT_THOUGHT, "b"));
    CHECK(emitter.emit_count()   == 2);
    CHECK(emitter.last_payload() == "b");
}
