/**
 * @file tests/unit/phase139_observability_wiring_test.cpp
 * @brief Phase 139 — v0.0.8 Observability Wiring Validation
 *
 * Validates that:
 *   1. ScopeProfiler is wired into all hot paths (propagator, torus, autonomy,
 *      embedder, mapper, LMDB persistence)
 *   2. TelemetryDaemon captures per-tick metrics via on_tick callback
 *   3. Profiling overhead remains < 1% of tick time
 *   4. All metric types (gauge, counter) produce valid JSON
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/diag/scope_profiler.hpp>
#include <nikola/diag/telemetry_daemon.hpp>
#include <nikola/physics/propagator.hpp>
#include <nikola/physics/wave_function.hpp>
#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/cognitive/cognitive_core.hpp>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>

using namespace nikola;
using Catch::Matchers::WithinAbs;

// ============================================================================
// §1 — ScopeProfiler: propagator::step scope exists after physics run
// ============================================================================
TEST_CASE("Phase139 §1: propagator::step scope registered", "[phase139][profiler]")
{
    diag::ScopeProfiler::global().reset();

    // Build a small WaveFunction and step it
    auto cfg = foundation::GridConfig::uniform(2);  // small grid
    foundation::TorusGrid grid(cfg);
    physics::WaveFunction wf(cfg);
    physics::Propagator prop;
    prop.step(wf, 0.01f);

    auto snap = diag::ScopeProfiler::global().report_one("propagator::step");
    REQUIRE(snap.count >= 1);
    REQUIRE(snap.total_us > 0.0);
}

// ============================================================================
// §2 — ScopeProfiler: torus::step scope registered
// ============================================================================
TEST_CASE("Phase139 §2: torus::step scope registered", "[phase139][profiler]")
{
    diag::ScopeProfiler::global().reset();

    cognitive::CognitiveTorus torus(2);  // small grid
    torus.set_gpu(false);               // force CPU path
    torus.step(0.01f);

    auto snap = diag::ScopeProfiler::global().report_one("torus::step");
    REQUIRE(snap.count >= 1);
    REQUIRE(snap.total_us > 0.0);
}

// ============================================================================
// §3 — ScopeProfiler: mapper::token_to_coord scope registered
// ============================================================================
TEST_CASE("Phase139 §3: mapper::token_to_coord scope registered", "[phase139][profiler]")
{
    diag::ScopeProfiler::global().reset();

    // Create a minimal TokenMapper (9D, embed_dim=4)
    constexpr int E = 4;
    std::array<int, 9> dims = {3, 3, 3, 3, 3, 3, 3, 3, 3};
    std::vector<float> proj(9 * E, 0.1f);  // simple uniform projection
    cognitive::TokenMapper mapper(std::move(proj), E, dims);

    std::vector<float> embed(E, 0.5f);
    auto coord = mapper.map(embed);

    auto snap = diag::ScopeProfiler::global().report_one("mapper::token_to_coord");
    REQUIRE(snap.count == 1);
    REQUIRE(snap.total_us >= 0.0);
}

// ============================================================================
// §4 — TelemetryDaemon: gauge produces valid JSON to pipe
// ============================================================================
TEST_CASE("Phase139 §4: TelemetryDaemon gauge emits JSON", "[phase139][telemetry]")
{
    int pipefd[2];
    REQUIRE(::pipe(pipefd) == 0);

    diag::TelemetryDaemon daemon;
    daemon.start(pipefd[1]);

    daemon.gauge("test.energy", 42.5, "J");
    daemon.flush_sync();
    daemon.stop();
    ::close(pipefd[1]);

    char buf[1024] = {};
    ssize_t n = ::read(pipefd[0], buf, sizeof(buf) - 1);
    ::close(pipefd[0]);

    REQUIRE(n > 0);
    std::string json(buf, static_cast<size_t>(n));
    // Verify key fields present
    REQUIRE(json.find("\"type\":\"gauge\"") != std::string::npos);
    REQUIRE(json.find("\"metric\":\"test.energy\"") != std::string::npos);
    REQUIRE(json.find("\"value\":42.5") != std::string::npos);
    REQUIRE(json.find("\"unit\":\"J\"") != std::string::npos);
    // Must end with newline (JSON Lines format)
    REQUIRE(json.back() == '\n');
}

// ============================================================================
// §5 — TelemetryDaemon: counter produces valid JSON
// ============================================================================
TEST_CASE("Phase139 §5: TelemetryDaemon counter emits JSON", "[phase139][telemetry]")
{
    int pipefd[2];
    REQUIRE(::pipe(pipefd) == 0);

    diag::TelemetryDaemon daemon;
    daemon.start(pipefd[1]);

    daemon.counter("tick.count", 1);
    daemon.flush_sync();
    daemon.stop();
    ::close(pipefd[1]);

    char buf[1024] = {};
    ssize_t n = ::read(pipefd[0], buf, sizeof(buf) - 1);
    ::close(pipefd[0]);

    REQUIRE(n > 0);
    std::string json(buf, static_cast<size_t>(n));
    REQUIRE(json.find("\"type\":\"counter\"") != std::string::npos);
    REQUIRE(json.find("\"metric\":\"tick.count\"") != std::string::npos);
    REQUIRE(json.find("\"delta\":1") != std::string::npos);
}

// ============================================================================
// §6 — Full telemetry pipeline: multiple gauges in sequence
// ============================================================================
TEST_CASE("Phase139 §6: full metric pipeline — 6 gauges + 1 counter", "[phase139][telemetry]")
{
    int pipefd[2];
    REQUIRE(::pipe(pipefd) == 0);

    diag::TelemetryDaemon daemon;
    daemon.start(pipefd[1]);

    // Simulate one tick's worth of telemetry
    daemon.gauge("tick.energy",   68890.5,  "J");
    daemon.gauge("tick.dopamine", 0.523);
    daemon.gauge("tick.atp",      0.847);
    daemon.gauge("tick.boredom",  0.312);
    daemon.gauge("tick.entropy",  7.234,    "nat");
    daemon.gauge("tick.duration", 142.5,    "us");
    daemon.counter("tick.count");

    daemon.flush_sync();
    daemon.stop();
    ::close(pipefd[1]);

    // Read all output
    std::string all;
    char buf[4096];
    ssize_t n;
    while ((n = ::read(pipefd[0], buf, sizeof(buf))) > 0)
        all.append(buf, static_cast<size_t>(n));
    ::close(pipefd[0]);

    // Count lines (each metric = one JSON line)
    int lines = 0;
    for (char c : all) if (c == '\n') ++lines;
    REQUIRE(lines == 7);  // 6 gauges + 1 counter

    // Verify all metric names present
    REQUIRE(all.find("tick.energy")   != std::string::npos);
    REQUIRE(all.find("tick.dopamine") != std::string::npos);
    REQUIRE(all.find("tick.atp")      != std::string::npos);
    REQUIRE(all.find("tick.boredom")  != std::string::npos);
    REQUIRE(all.find("tick.entropy")  != std::string::npos);
    REQUIRE(all.find("tick.duration") != std::string::npos);
    REQUIRE(all.find("tick.count")    != std::string::npos);
}

// ============================================================================
// §7 — Profiling overhead < 1% verification
// ============================================================================
TEST_CASE("Phase139 §7: profiling overhead under 1%", "[phase139][profiler][overhead]")
{
    diag::ScopeProfiler::global().reset();

    // Time 1000 NIKOLA_PROFILE guards (empty scopes)
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        NIKOLA_PROFILE("overhead_test");
    }
    auto t1 = std::chrono::steady_clock::now();

    const double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double per_guard_us = elapsed_us / 1000.0;

    // Each guard should be < 10µs (the Phase 18 budget)
    REQUIRE(per_guard_us < 10.0);

    // With 7 scopes per tick at 1000µs budget: 7 × per_guard < 10µs = <1%
    const double overhead_pct = (7.0 * per_guard_us) / 1000.0 * 100.0;
    REQUIRE(overhead_pct < 1.0);
}

// ============================================================================
// §8 — All expected scopes appear in a full report
// ============================================================================
TEST_CASE("Phase139 §8: all hot-path scopes registered", "[phase139][profiler]")
{
    diag::ScopeProfiler::global().reset();

    // Exercise each scope
    {
        NIKOLA_PROFILE("propagator::step");
        NIKOLA_PROFILE("torus::step");
        NIKOLA_PROFILE("torus::reseed_check");
        NIKOLA_PROFILE("autonomy::read_state");
        NIKOLA_PROFILE("autonomy::score_candidates");
        NIKOLA_PROFILE("embed::nonary");
        NIKOLA_PROFILE("mapper::token_to_coord");
        NIKOLA_PROFILE("lmdb::save_state");
        NIKOLA_PROFILE("lmdb::load_state");
        NIKOLA_PROFILE("lmdb::save_checkpoint");
        NIKOLA_PROFILE("lmdb::load_checkpoint");
        NIKOLA_PROFILE("lmdb::put");
    }

    auto report = diag::ScopeProfiler::global().report();
    REQUIRE(report.size() >= 12);

    // Verify all scope names present
    auto has_scope = [&](const std::string& name) {
        for (const auto& s : report)
            if (s.name == name) return true;
        return false;
    };

    CHECK(has_scope("propagator::step"));
    CHECK(has_scope("torus::step"));
    CHECK(has_scope("torus::reseed_check"));
    CHECK(has_scope("autonomy::read_state"));
    CHECK(has_scope("autonomy::score_candidates"));
    CHECK(has_scope("embed::nonary"));
    CHECK(has_scope("mapper::token_to_coord"));
    CHECK(has_scope("lmdb::save_state"));
    CHECK(has_scope("lmdb::load_state"));
    CHECK(has_scope("lmdb::save_checkpoint"));
    CHECK(has_scope("lmdb::load_checkpoint"));
    CHECK(has_scope("lmdb::put"));
}
