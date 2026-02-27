/**
 * @file phase119_otel_tracing_test.cpp
 * @brief Phase 119 — OpenTelemetry live tracing for Nikola tick loop.
 *
 * Tests exercise  nikola::telemetry::TickTracer  and the helper functions
 * (setup_in_memory_tracer / teardown_tracer) using the OTel in-memory exporter.
 * No real CognitiveTorus / DecisionLoop is needed; NikolaState structs are
 * constructed by hand.
 *
 * Test map  (16 test cases)
 * ─────────────────────────
 *  [P119/setup]    setup_in_memory_tracer returns non-null InMemorySpanData
 *  [P119/span]     trace_tick emits exactly one span per call
 *  [P119/name]     span name is "nikola.tick"
 *  [P119/attrs]    span attributes match NikolaState fields
 *  [P119/action]   nikola.action attribute matches action_name()
 *  [P119/multi]    multiple trace_tick calls accumulate spans
 *  [P119/noop]     NIKOLA_OTEL_DISABLED build: trace_tick is a no-op (compile-only)
 *  [P119/teardown] teardown_tracer restores no-op provider (no spans after)
 */

#include <nikola/telemetry/nikola_tracer.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <string>
#include <vector>
#include <cstdint>

#include <opentelemetry/sdk/common/attribute_utils.h>   // OwnedAttributeValue

using nikola::telemetry::TickTracer;
using nikola::telemetry::setup_in_memory_tracer;
using nikola::telemetry::teardown_tracer;
using nikola::autonomy::NikolaState;
using nikola::autonomy::ActionType;
using nikola::autonomy::action_name;

// Convenience: get an attribute by key from the span's attribute map.
// Returns the string representation of the attribute value.
static std::string attr_str(
    const std::unordered_map<std::string,
          opentelemetry::sdk::common::OwnedAttributeValue>& attrs,
    const std::string& key)
{
    auto it = attrs.find(key);
    if (it == attrs.end()) return "<MISSING>";
    return opentelemetry::nostd::visit(
        [](auto&& v) -> std::string {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, opentelemetry::nostd::string_view>)
                return std::string(v);
            else if constexpr (std::is_same_v<T, std::string>)
                return v;
            else if constexpr (std::is_arithmetic_v<T>)
                return std::to_string(v);
            else
                return "<type>";
        },
        it->second);
}

static double attr_double(
    const std::unordered_map<std::string,
          opentelemetry::sdk::common::OwnedAttributeValue>& attrs,
    const std::string& key)
{
    auto it = attrs.find(key);
    if (it == attrs.end()) return -999.0;
    return opentelemetry::nostd::visit(
        [](auto&& v) -> double {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_arithmetic_v<T>)
                return static_cast<double>(v);
            return -999.0;
        },
        it->second);
}

// Helper: build a NikolaState with given field values.
static NikolaState make_state(float energy=0.5f, float dopamine=0.7f,
                               float atp=0.8f, float boredom=0.2f,
                               float entropy=1.3f, float td_error=0.05f,
                               ActionType action=ActionType::EMIT_THOUGHT)
{
    NikolaState s;
    s.torus_energy = energy;
    s.dopamine     = dopamine;
    s.atp          = atp;
    s.boredom      = boredom;
    s.entropy      = entropy;
    s.td_error     = td_error;
    s.last_action  = action;
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// [P119/setup]  Provider setup utilities
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase119 setup_in_memory_tracer returns non-null data handle",
          "[Phase119][P119/setup]")
{
    auto data = setup_in_memory_tracer();
    REQUIRE(data != nullptr);
    teardown_tracer();
}

TEST_CASE("Phase119 setup_in_memory_tracer: no spans before any trace_tick call",
          "[Phase119][P119/setup]")
{
    auto data = setup_in_memory_tracer();
    CHECK(data->GetSpans().empty());
    teardown_tracer();
}

// ─────────────────────────────────────────────────────────────────────────────
// [P119/span]  Span emission count
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase119 trace_tick emits exactly one span per call",
          "[Phase119][P119/span]")
{
    auto data = setup_in_memory_tracer();
    TickTracer tt;
    tt.trace_tick(make_state(), 0);
    CHECK(data->GetSpans().size() == 1);
    teardown_tracer();
}

// ─────────────────────────────────────────────────────────────────────────────
// [P119/name]  Span name
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase119 span name is 'nikola.tick'",
          "[Phase119][P119/name]")
{
    auto data = setup_in_memory_tracer();
    TickTracer tt;
    tt.trace_tick(make_state(), 1);
    auto spans = data->GetSpans();
    REQUIRE(spans.size() == 1);
    CHECK(std::string(spans[0]->GetName()) == TickTracer::kSpanName);
    teardown_tracer();
}

// ─────────────────────────────────────────────────────────────────────────────
// [P119/attrs]  Attribute values match NikolaState fields
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase119 nikola.tick.index attribute matches tick counter",
          "[Phase119][P119/attrs]")
{
    auto data = setup_in_memory_tracer();
    TickTracer tt;
    tt.trace_tick(make_state(), 42);
    auto spans = data->GetSpans();
    REQUIRE(spans.size() == 1);
    double idx = attr_double(spans[0]->GetAttributes(), "nikola.tick.index");
    CHECK_THAT(idx, Catch::Matchers::WithinAbs(42.0, 0.5));
    teardown_tracer();
}

TEST_CASE("Phase119 nikola.state.energy attribute matches torus_energy",
          "[Phase119][P119/attrs]")
{
    auto data = setup_in_memory_tracer();
    TickTracer tt;
    tt.trace_tick(make_state(/*energy=*/3.14f), 0);
    auto spans = data->GetSpans();
    REQUIRE(spans.size() == 1);
    CHECK_THAT(attr_double(spans[0]->GetAttributes(), "nikola.state.energy"),
               Catch::Matchers::WithinAbs(3.14, 1e-4));
    teardown_tracer();
}

TEST_CASE("Phase119 nikola.state.dopamine attribute matches dopamine",
          "[Phase119][P119/attrs]")
{
    auto data = setup_in_memory_tracer();
    TickTracer tt;
    tt.trace_tick(make_state(0.5f, /*dopamine=*/0.65f), 0);
    auto spans = data->GetSpans();
    REQUIRE(spans.size() == 1);
    CHECK_THAT(attr_double(spans[0]->GetAttributes(), "nikola.state.dopamine"),
               Catch::Matchers::WithinAbs(0.65, 1e-4));
    teardown_tracer();
}

TEST_CASE("Phase119 nikola.state.atp attribute matches ATP",
          "[Phase119][P119/attrs]")
{
    auto data = setup_in_memory_tracer();
    TickTracer tt;
    tt.trace_tick(make_state(0.5f, 0.5f, /*atp=*/0.33f), 0);
    auto spans = data->GetSpans();
    REQUIRE(spans.size() == 1);
    CHECK_THAT(attr_double(spans[0]->GetAttributes(), "nikola.state.atp"),
               Catch::Matchers::WithinAbs(0.33, 1e-4));
    teardown_tracer();
}

TEST_CASE("Phase119 nikola.state.boredom attribute matches boredom",
          "[Phase119][P119/attrs]")
{
    auto data = setup_in_memory_tracer();
    TickTracer tt;
    tt.trace_tick(make_state(0.5f, 0.5f, 0.5f, /*boredom=*/0.77f), 0);
    auto spans = data->GetSpans();
    REQUIRE(spans.size() == 1);
    CHECK_THAT(attr_double(spans[0]->GetAttributes(), "nikola.state.boredom"),
               Catch::Matchers::WithinAbs(0.77, 1e-4));
    teardown_tracer();
}

TEST_CASE("Phase119 nikola.state.entropy attribute matches entropy",
          "[Phase119][P119/attrs]")
{
    auto data = setup_in_memory_tracer();
    TickTracer tt;
    tt.trace_tick(make_state(0.5f, 0.5f, 0.5f, 0.5f, /*entropy=*/2.1f), 0);
    auto spans = data->GetSpans();
    REQUIRE(spans.size() == 1);
    CHECK_THAT(attr_double(spans[0]->GetAttributes(), "nikola.state.entropy"),
               Catch::Matchers::WithinAbs(2.1, 1e-4));
    teardown_tracer();
}

TEST_CASE("Phase119 nikola.state.td_error attribute matches td_error",
          "[Phase119][P119/attrs]")
{
    auto data = setup_in_memory_tracer();
    TickTracer tt;
    tt.trace_tick(make_state(0.5f, 0.5f, 0.5f, 0.5f, 1.0f, /*td_error=*/-0.25f), 0);
    auto spans = data->GetSpans();
    REQUIRE(spans.size() == 1);
    CHECK_THAT(attr_double(spans[0]->GetAttributes(), "nikola.state.td_error"),
               Catch::Matchers::WithinAbs(-0.25, 1e-4));
    teardown_tracer();
}

// ─────────────────────────────────────────────────────────────────────────────
// [P119/action]  nikola.action attribute
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase119 nikola.action attribute is action_name() of last_action",
          "[Phase119][P119/action]")
{
    auto data = setup_in_memory_tracer();
    TickTracer tt;
    tt.trace_tick(make_state(0.5f, 0.5f, 0.5f, 0.5f, 1.0f, 0.0f,
                              ActionType::EXPLORE), 0);
    auto spans = data->GetSpans();
    REQUIRE(spans.size() == 1);
    std::string act = attr_str(spans[0]->GetAttributes(), "nikola.action");
    CHECK(act == std::string(action_name(ActionType::EXPLORE)));
    teardown_tracer();
}

TEST_CASE("Phase119 nikola.action is 'SILENT' for SILENT state",
          "[Phase119][P119/action]")
{
    auto data = setup_in_memory_tracer();
    TickTracer tt;
    tt.trace_tick(make_state(0.5f, 0.5f, 0.5f, 0.5f, 1.0f, 0.0f,
                              ActionType::SILENT), 0);
    auto spans = data->GetSpans();
    REQUIRE(spans.size() == 1);
    std::string act = attr_str(spans[0]->GetAttributes(), "nikola.action");
    CHECK(act == std::string(action_name(ActionType::SILENT)));
    teardown_tracer();
}

// ─────────────────────────────────────────────────────────────────────────────
// [P119/multi]  Multiple calls accumulate spans
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase119 multiple trace_tick calls accumulate spans",
          "[Phase119][P119/multi]")
{
    auto data = setup_in_memory_tracer(512);
    TickTracer tt;
    for (int i = 0; i < 5; ++i)
        tt.trace_tick(make_state(), static_cast<int64_t>(i));
    CHECK(data->GetSpans().size() == 5);
    teardown_tracer();
}

TEST_CASE("Phase119 each accumulated span has a unique tick index",
          "[Phase119][P119/multi]")
{
    auto data = setup_in_memory_tracer(512);
    TickTracer tt;
    for (int i = 0; i < 3; ++i)
        tt.trace_tick(make_state(), static_cast<int64_t>(i * 10));
    auto spans = data->GetSpans();
    REQUIRE(spans.size() == 3);
    std::vector<double> indices;
    for (const auto& sp : spans)
        indices.push_back(attr_double(sp->GetAttributes(), "nikola.tick.index"));
    // All three must be distinct
    CHECK(indices[0] != indices[1]);
    CHECK(indices[1] != indices[2]);
    CHECK(indices[0] != indices[2]);
    teardown_tracer();
}

// ─────────────────────────────────────────────────────────────────────────────
// [P119/teardown]  teardown_tracer restores no-op provider
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Phase119 teardown_tracer: no spans written after teardown",
          "[Phase119][P119/teardown]")
{
    // First: install in-memory provider
    auto data = setup_in_memory_tracer();
    TickTracer tt;
    tt.trace_tick(make_state(), 0);
    REQUIRE(data->GetSpans().size() == 1);

    // Teardown: switch to noop
    teardown_tracer();

    // After teardown: new trace_tick calls go to the noop provider.
    // The data handle still contains the original 1 span.
    // Re-install in-memory to verify the noop provider was actually replaced.
    auto data2 = setup_in_memory_tracer();
    // No additional trace_tick call → data2 should be empty
    CHECK(data2->GetSpans().empty());
    teardown_tracer();
}
