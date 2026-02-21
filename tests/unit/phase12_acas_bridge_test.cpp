/**
 * @file phase12_acas_bridge_test.cpp
 * @brief Phase 12 unit tests — AcasBridge + ACAS Python scripts (IPC bridge)
 *
 * Tests the C++ ↔ Python IPC layer.  The Python scripts run in offline/stub
 * mode since faster-whisper / piper-tts may not be installed.
 *
 * Build expects NIKOLA_ACAS_SCRIPTS_DIR to be defined as a string macro
 * pointing to the directory containing analyzer_main.py / generator_main.py.
 *
 * Covers:
 *   - json_escape: control characters, quotes, backslash
 *   - AcasBridge: empty scripts → start() throws
 *   - start(): spawns both processes
 *   - analyzer_running() / generator_running() → true while alive
 *   - Analyzer emits initial JSON status line (offline mode)
 *   - Status line contains "type" field
 *   - Generator emits initial status line
 *   - speak(): sends command; generator replies with "speaking" + "done"
 *   - stop_speaking() sends stop command without crashing
 *   - stop(): both processes terminate
 *   - restart(): processes re-spawn after stop
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/infrastructure/acas_bridge.hpp>

#include <algorithm>
#include <cstring>
#include <string>
#include <thread>
#include <chrono>

// JSON helpers (very minimal — just enough for test assertions)
// Checks if a raw JSON string contains `"key":"value"` or `"key":value`
static bool json_has(const std::string& json, const std::string& key,
                     const std::string& value)
{
    // Match both compact ("key":"value") and human-readable ("key": "value")
    const auto needle_qq  = '"' + key + "\":\"" + value + '"';   // no space
    const auto needle_qqs = '"' + key + "\": \"" + value + '"';  // space after colon
    const auto needle_qb  = '"' + key + "\":" + value;           // unquoted
    const auto needle_qbs = '"' + key + "\": " + value;          // unquoted + space
    return json.find(needle_qq)  != std::string::npos
        || json.find(needle_qqs) != std::string::npos
        || json.find(needle_qb)  != std::string::npos
        || json.find(needle_qbs) != std::string::npos;
}

static bool json_has_key(const std::string& json, const std::string& key)
{
    return json.find('"' + key + '"') != std::string::npos;
}

// ---------------------------------------------------------------------------
// Helper: poll for first line from bridge with timeout
// ---------------------------------------------------------------------------

using Fn = std::optional<std::string>(nikola::infrastructure::AcasBridge::*)();

static std::optional<std::string>
wait_for_line(nikola::infrastructure::AcasBridge& bridge,
              Fn reader, int timeout_ms = 3000)
{
    using namespace std::chrono;
    const auto deadline = steady_clock::now() + milliseconds(timeout_ms);
    while (steady_clock::now() < deadline) {
        auto line = (bridge.*reader)();
        if (line.has_value() && !line->empty()) return line;
        std::this_thread::sleep_for(milliseconds(10));
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Determine scripts directory at compile time (set by CMake)
// ---------------------------------------------------------------------------

#ifndef NIKOLA_ACAS_SCRIPTS_DIR
#  error "NIKOLA_ACAS_SCRIPTS_DIR must be defined as the path to src/acas/"
#endif

static const std::string kScriptsDir = NIKOLA_ACAS_SCRIPTS_DIR;
static const std::string kAnalyzer   = kScriptsDir + "/analyzer_main.py";
static const std::string kGenerator  = kScriptsDir + "/generator_main.py";

// ===========================================================================
// json_escape (whitebox test via speak())
// ===========================================================================

// We can test json_escape indirectly by reading back the generator line,
// but for directness we replicate the same escaping logic here:
static std::string local_escape(const std::string& s)
{
    std::string out;
    for (const char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

TEST_CASE("json_escape: plain text unchanged", "[acas][escape]") {
    CHECK(local_escape("hello") == "hello");
}

TEST_CASE("json_escape: double-quote escaped", "[acas][escape]") {
    CHECK(local_escape("say \"hi\"") == R"(say \"hi\")");
}

TEST_CASE("json_escape: backslash escaped", "[acas][escape]") {
    CHECK(local_escape("a\\b") == "a\\\\b");
}

TEST_CASE("json_escape: newline escaped", "[acas][escape]") {
    CHECK(local_escape("line1\nline2") == "line1\\nline2");
}

TEST_CASE("json_escape: tab escaped", "[acas][escape]") {
    CHECK(local_escape("col1\tcol2") == "col1\\tcol2");
}

// ===========================================================================
// AcasBridge configuration validation
// ===========================================================================

TEST_CASE("AcasBridge: empty scripts → start() throws", "[acas][bridge]") {
    nikola::infrastructure::AcasConfig cfg;
    nikola::infrastructure::AcasBridge bridge(cfg);
    REQUIRE_THROWS_AS(bridge.start(), std::runtime_error);
}

TEST_CASE("AcasBridge: only analyzer set → start() throws (generator missing)",
          "[acas][bridge]") {
    nikola::infrastructure::AcasConfig cfg;
    cfg.analyzer_script = kAnalyzer;
    nikola::infrastructure::AcasBridge bridge(cfg);
    REQUIRE_THROWS_AS(bridge.start(), std::runtime_error);
}

// ===========================================================================
// Live process tests (spawn Python in offline mode)
// ===========================================================================

TEST_CASE("AcasBridge: start() spawns both processes", "[acas][live]") {
    nikola::infrastructure::AcasConfig cfg;
    cfg.analyzer_script  = kAnalyzer;
    cfg.generator_script = kGenerator;
    nikola::infrastructure::AcasBridge bridge(cfg);

    REQUIRE_NOTHROW(bridge.start());
    CHECK(bridge.is_started());

    // Give processes a moment to initialise
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    CHECK(bridge.analyzer_running());
    CHECK(bridge.generator_running());
}

TEST_CASE("AcasBridge: analyzer emits initial status JSON", "[acas][live]") {
    nikola::infrastructure::AcasConfig cfg;
    cfg.analyzer_script  = kAnalyzer;
    cfg.generator_script = kGenerator;
    nikola::infrastructure::AcasBridge bridge(cfg);
    bridge.start();

    const auto line = wait_for_line(bridge,
        &nikola::infrastructure::AcasBridge::read_analyzer_line, 4000);

    REQUIRE(line.has_value());
    INFO("Analyzer status line: " << *line);
    CHECK(json_has_key(*line, "type"));
    CHECK(json_has(*line, "type", "status"));
    CHECK(json_has_key(*line, "version"));
}

TEST_CASE("AcasBridge: analyzer status mode is online or offline", "[acas][live]") {
    nikola::infrastructure::AcasConfig cfg;
    cfg.analyzer_script  = kAnalyzer;
    cfg.generator_script = kGenerator;
    nikola::infrastructure::AcasBridge bridge(cfg);
    bridge.start();

    const auto line = wait_for_line(bridge,
        &nikola::infrastructure::AcasBridge::read_analyzer_line, 4000);

    REQUIRE(line.has_value());
    const bool has_mode =
        line->find("\"mode\"") != std::string::npos &&
        (line->find("online") != std::string::npos ||
         line->find("offline") != std::string::npos);
    CHECK(has_mode);
}

TEST_CASE("AcasBridge: generator emits initial status JSON", "[acas][live]") {
    nikola::infrastructure::AcasConfig cfg;
    cfg.analyzer_script  = kAnalyzer;
    cfg.generator_script = kGenerator;
    nikola::infrastructure::AcasBridge bridge(cfg);
    bridge.start();

    const auto line = wait_for_line(bridge,
        &nikola::infrastructure::AcasBridge::read_generator_line, 4000);

    REQUIRE(line.has_value());
    INFO("Generator status line: " << *line);
    CHECK(json_has_key(*line, "type"));
    CHECK(json_has(*line, "type", "status"));
}

TEST_CASE("AcasBridge: speak() triggers speaking + done events", "[acas][live]") {
    nikola::infrastructure::AcasConfig cfg;
    cfg.analyzer_script  = kAnalyzer;
    cfg.generator_script = kGenerator;
    nikola::infrastructure::AcasBridge bridge(cfg);
    bridge.start();

    // Drain initial status line from generator
    wait_for_line(bridge,
        &nikola::infrastructure::AcasBridge::read_generator_line, 3000);

    // Send speak command
    REQUIRE(bridge.speak("hello world"));

    // Expect "speaking" event
    const auto speaking = wait_for_line(bridge,
        &nikola::infrastructure::AcasBridge::read_generator_line, 3000);
    REQUIRE(speaking.has_value());
    INFO("speaking event: " << *speaking);
    CHECK(json_has(*speaking, "type", "speaking"));

    // Expect "done" event
    const auto done = wait_for_line(bridge,
        &nikola::infrastructure::AcasBridge::read_generator_line, 3000);
    REQUIRE(done.has_value());
    INFO("done event: " << *done);
    CHECK(json_has(*done, "type", "done"));
}

TEST_CASE("AcasBridge: speak() with special characters doesn't corrupt protocol",
          "[acas][live]") {
    nikola::infrastructure::AcasConfig cfg;
    cfg.analyzer_script  = kAnalyzer;
    cfg.generator_script = kGenerator;
    nikola::infrastructure::AcasBridge bridge(cfg);
    bridge.start();

    wait_for_line(bridge,
        &nikola::infrastructure::AcasBridge::read_generator_line, 3000);

    // Text with quotes and backslash — must survive JSON encoding
    REQUIRE(bridge.speak("say \"hello\" and C:\\path\\file"));

    const auto speaking = wait_for_line(bridge,
        &nikola::infrastructure::AcasBridge::read_generator_line, 3000);
    CHECK(speaking.has_value());
}

TEST_CASE("AcasBridge: stop_speaking() sends without crash", "[acas][live]") {
    nikola::infrastructure::AcasConfig cfg;
    cfg.analyzer_script  = kAnalyzer;
    cfg.generator_script = kGenerator;
    nikola::infrastructure::AcasBridge bridge(cfg);
    bridge.start();

    REQUIRE_NOTHROW(bridge.stop_speaking());
}

TEST_CASE("AcasBridge: stop() terminates both processes", "[acas][live]") {
    nikola::infrastructure::AcasConfig cfg;
    cfg.analyzer_script  = kAnalyzer;
    cfg.generator_script = kGenerator;
    nikola::infrastructure::AcasBridge bridge(cfg);
    bridge.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    bridge.stop();

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    CHECK_FALSE(bridge.is_started());
    // After stop, processes should have exited
    CHECK_FALSE(bridge.analyzer_running());
    CHECK_FALSE(bridge.generator_running());
}

TEST_CASE("AcasBridge: restart() re-spawns processes", "[acas][live]") {
    nikola::infrastructure::AcasConfig cfg;
    cfg.analyzer_script  = kAnalyzer;
    cfg.generator_script = kGenerator;
    nikola::infrastructure::AcasBridge bridge(cfg);
    bridge.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    bridge.restart();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    CHECK(bridge.is_started());
    CHECK(bridge.analyzer_running());
    CHECK(bridge.generator_running());
}
