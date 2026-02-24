/**
 * @file tests/unit/phase18_scope_profiler_test.cpp
 * @brief Phase 18 — ecosystem/09_ScopeProfiler unit tests
 *
 * Requirements validated:
 *   - RAII guard records elapsed time correctly
 *   - Global registry accumulates multiple scopes
 *   - Statistics: count, total, min, max, mean
 *   - Thread-safety under concurrent recording
 *   - reset() clears all stats
 *   - Overhead: guard cost < 10µs (<<1% of 1ms 1kHz budget) per call
 *   - NIKOLA_PROFILE macro compiles and records
 *   - within_budget() and budget_fraction_1khz() helpers
 */

#include <nikola/diag/scope_profiler.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

using namespace nikola::diag;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

// ─────────────────────────────────────────────────────────────────────────────
//  Basic recording
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ScopeProfiler: manual record accumulates stats", "[profiler][basic]") {
    ScopeProfiler p;

    SECTION("single record") {
        p.record("test.scope", 5000);  // 5µs
        auto s = p.report_one("test.scope");
        REQUIRE(s.count == 1);
        REQUIRE_THAT(s.total_us, WithinAbs(5.0, 0.01));
        REQUIRE_THAT(s.min_us,   WithinAbs(5.0, 0.01));
        REQUIRE_THAT(s.max_us,   WithinAbs(5.0, 0.01));
        REQUIRE_THAT(s.mean_us(),WithinAbs(5.0, 0.01));
    }

    SECTION("multiple records — min/max/mean") {
        p.record("multi", 1000);   // 1µs
        p.record("multi", 3000);   // 3µs
        p.record("multi", 2000);   // 2µs
        auto s = p.report_one("multi");
        REQUIRE(s.count == 3);
        REQUIRE_THAT(s.min_us,    WithinAbs(1.0, 0.01));
        REQUIRE_THAT(s.max_us,    WithinAbs(3.0, 0.01));
        REQUIRE_THAT(s.mean_us(), WithinAbs(2.0, 0.01));
    }

    SECTION("scope not found returns empty snapshot") {
        auto s = p.report_one("nonexistent.scope");
        REQUIRE(s.count == 0);
        REQUIRE(s.name.empty());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  RAII Guard
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ScopeProfiler: RAII guard records elapsed time", "[profiler][raii]") {
    ScopeProfiler p;

    SECTION("guard measures real time") {
        {
            auto g = p.scope("sleep_scope");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        auto s = p.report_one("sleep_scope");
        REQUIRE(s.count == 1);
        // Should be >= 5ms (5000µs), allow generous upper bound for CI
        REQUIRE(s.total_us >= 4000.0);   // ≥ 4ms (some systems run slow)
        REQUIRE(s.total_us <  50000.0);  // < 50ms (upper sanity)
    }

    SECTION("guard scope count increments") {
        {   auto g1 = p.scope("g.scope"); }
        {   auto g2 = p.scope("g.scope"); }
        {   auto g3 = p.scope("g.scope"); }
        REQUIRE(p.report_one("g.scope").count == 3);
    }

    SECTION("nested scopes are independent") {
        {
            auto outer = p.scope("nest.outer");
            {
                auto inner = p.scope("nest.inner");
            }
        }
        REQUIRE(p.report_one("nest.outer").count == 1);
        REQUIRE(p.report_one("nest.inner").count == 1);
        // outer total >= inner total
        REQUIRE(p.report_one("nest.outer").total_us >=
                p.report_one("nest.inner").total_us);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIKOLA_PROFILE macro
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ScopeProfiler: NIKOLA_PROFILE_ON macro records", "[profiler][macro]") {
    ScopeProfiler p;

    {
        NIKOLA_PROFILE_ON(p, "macro.scope");
        // small computation to avoid zero-time
        volatile int x = 0;
        for (int i = 0; i < 100; ++i) x += i;
        (void)x;
    }

    auto s = p.report_one("macro.scope");
    REQUIRE(s.count == 1);
    REQUIRE(s.total_us >= 0.0);   // non-negative
}

TEST_CASE("ScopeProfiler: NIKOLA_PROFILE macro uses global", "[profiler][macro]") {
    // Reset global so previous tests don't pollute
    ScopeProfiler::global().reset("global.macro.test");

    {
        NIKOLA_PROFILE("global.macro.test");
        volatile int x = 1 + 1; (void)x;
    }

    auto s = ScopeProfiler::global().report_one("global.macro.test");
    REQUIRE(s.count >= 1);
}

// ─────────────────────────────────────────────────────────────────────────────
//  report() and scope_count()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ScopeProfiler: report() returns all scopes", "[profiler][report]") {
    ScopeProfiler p;
    p.record("r.alpha", 1000);
    p.record("r.beta",  2000);
    p.record("r.gamma", 3000);

    auto all = p.report();
    REQUIRE(all.size() == 3);

    // Each name appears exactly once
    std::vector<std::string> names;
    for (const auto& s : all) names.push_back(s.name);
    REQUIRE(std::find(names.begin(), names.end(), "r.alpha") != names.end());
    REQUIRE(std::find(names.begin(), names.end(), "r.beta")  != names.end());
    REQUIRE(std::find(names.begin(), names.end(), "r.gamma") != names.end());
}

TEST_CASE("ScopeProfiler: scope_count()", "[profiler][report]") {
    ScopeProfiler p;
    REQUIRE(p.scope_count() == 0);
    p.record("sc.a", 100);
    REQUIRE(p.scope_count() == 1);
    p.record("sc.b", 200);
    REQUIRE(p.scope_count() == 2);
    p.record("sc.a", 300);   // same name, no new slot
    REQUIRE(p.scope_count() == 2);
}

// ─────────────────────────────────────────────────────────────────────────────
//  reset()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ScopeProfiler: reset() clears all stats", "[profiler][reset]") {
    ScopeProfiler p;
    p.record("rst.a", 1000);
    p.record("rst.b", 2000);
    REQUIRE(p.report_one("rst.a").count == 1);

    p.reset();

    REQUIRE(p.report_one("rst.a").count == 0);
    REQUIRE(p.report_one("rst.b").count == 0);
    REQUIRE(p.scope_count() == 2);  // slots still exist, just zeroed
}

TEST_CASE("ScopeProfiler: reset(name) clears single scope", "[profiler][reset]") {
    ScopeProfiler p;
    p.record("r1.a", 1000);
    p.record("r1.b", 2000);
    p.reset("r1.a");
    REQUIRE(p.report_one("r1.a").count == 0);
    REQUIRE(p.report_one("r1.b").count == 1);  // untouched
}

// ─────────────────────────────────────────────────────────────────────────────
//  Budget helpers
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ScopeProfiler: budget helpers", "[profiler][budget]") {
    ScopeProfiler p;

    SECTION("within_budget: fast scope") {
        p.record("budget.fast", 500);   // 0.5µs — well under any budget
        auto s = p.report_one("budget.fast");
        REQUIRE(s.within_budget(10.0));   // 10µs budget
        REQUIRE(s.within_budget(1.0));    // 1µs budget
    }

    SECTION("within_budget: slow scope") {
        p.record("budget.slow", 500'000);  // 500µs
        auto s = p.report_one("budget.slow");
        REQUIRE(!s.within_budget(10.0));  // exceeds 10µs budget
    }

    SECTION("budget_fraction_1khz is proportional to mean") {
        p.record("bfrac", 100'000);  // 100µs = 10% of 1ms
        auto s = p.report_one("bfrac");
        REQUIRE_THAT(s.budget_fraction_1khz(), WithinAbs(0.1, 0.001));
    }

    SECTION("mean_us is 0 for count=0") {
        ScopeSnapshot empty;
        REQUIRE(empty.mean_us() == 0.0);
        REQUIRE(empty.budget_fraction_1khz() == 0.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Thread safety
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ScopeProfiler: thread-safe concurrent recording", "[profiler][thread]") {
    ScopeProfiler p;
    constexpr int N_THREADS = 8;
    constexpr int N_RECS    = 500;

    std::vector<std::thread> threads;
    threads.reserve(N_THREADS);
    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&p, t] {
            for (int i = 0; i < N_RECS; ++i) {
                p.record("concurrent.scope", static_cast<uint64_t>((t + 1) * 1000));
            }
        });
    }
    for (auto& th : threads) th.join();

    auto s = p.report_one("concurrent.scope");
    REQUIRE(s.count == static_cast<uint64_t>(N_THREADS * N_RECS));
}

TEST_CASE("ScopeProfiler: thread-safe multi-scope concurrent", "[profiler][thread]") {
    ScopeProfiler p;
    constexpr int N_THREADS = 4;
    constexpr int N_RECS    = 200;

    std::vector<std::thread> threads;
    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&p, t] {
            const std::string name = "ts.scope." + std::to_string(t);
            for (int i = 0; i < N_RECS; ++i) {
                auto g = p.scope(name.c_str());
                volatile int x = i * 2; (void)x;
            }
        });
    }
    for (auto& th : threads) th.join();

    for (int t = 0; t < N_THREADS; ++t) {
        const std::string name = "ts.scope." + std::to_string(t);
        auto s = p.report_one(name);
        REQUIRE(s.count == static_cast<uint64_t>(N_RECS));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Overhead benchmark
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ScopeProfiler: guard overhead < 1% of 1kHz budget", "[profiler][overhead]") {
    ScopeProfiler p;
    constexpr int ITERATIONS = 10'000;

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        auto g = p.scope("overhead.bench");
        // empty body — measures raw guard overhead
    }
    const auto t1 = std::chrono::steady_clock::now();

    const double total_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double per_guard_us = total_us / static_cast<double>(ITERATIONS);

    // Budget: <1% of 1ms = <10µs per guard
    // On any reasonable hardware this is <1µs; 10µs is very conservative
    REQUIRE(per_guard_us < 10.0);

    // Counts recorded correctly
    REQUIRE(p.report_one("overhead.bench").count ==
            static_cast<uint64_t>(ITERATIONS));
}

// ─────────────────────────────────────────────────────────────────────────────
//  SlotStats unit tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("SlotStats: atomic accumulation", "[profiler][slot]") {
    SlotStats s;

    SECTION("initial state") {
        REQUIRE(s.count.load() == 0);
        REQUIRE(s.total_ns.load() == 0);
        REQUIRE(s.max_ns.load() == 0);
    }

    SECTION("record three values") {
        s.record(100);
        s.record(200);
        s.record(50);
        REQUIRE(s.count.load() == 3);
        REQUIRE(s.total_ns.load() == 350);
        REQUIRE(s.min_ns.load() == 50);
        REQUIRE(s.max_ns.load() == 200);
    }

    SECTION("reset restores initial state") {
        s.record(999);
        s.reset();
        REQUIRE(s.count.load() == 0);
        REQUIRE(s.total_ns.load() == 0);
        REQUIRE(s.max_ns.load() == 0);
    }
}
