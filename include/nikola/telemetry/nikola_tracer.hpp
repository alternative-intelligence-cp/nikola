/**
 * @file telemetry/nikola_tracer.hpp
 * @brief Phase 119 — OpenTelemetry live tracing for the Nikola tick loop.
 *
 * Usage (nikola_run.cpp or any caller)
 * ─────────────────────────────────────
 * @code
 *   // 1. Set up a provider once at startup (e.g. ostream to stderr):
 *   nikola::telemetry::setup_ostream_tracer();        // → stderr
 *   // or for tests / library consumers:
 *   auto data = nikola::telemetry::setup_in_memory_tracer();
 *
 *   // 2. Wire the tick tracer into the decision loop callback:
 *   nikola::telemetry::TickTracer tracer;
 *   loop.on_tick = [&](const nikola::autonomy::NikolaState& s) {
 *       tracer.trace_tick(s, loop.tick_count());
 *   };
 * @endcode
 *
 * Compile-time opt-out
 * ─────────────────────
 * Define NIKOLA_OTEL_DISABLED before including this header to compile out
 * all OTel code.  TickTracer becomes a no-op struct; setup functions do nothing
 * and return nullptr.
 *
 * Span schema (name: "nikola.tick")
 * ──────────────────────────────────
 *   nikola.tick.index     int64  monotonic tick counter
 *   nikola.state.energy   double torus |ψ|² energy
 *   nikola.state.dopamine double dopamine level ∈ [0,1]
 *   nikola.state.atp      double ATP level ∈ [0,1]
 *   nikola.state.boredom  double boredom ∈ [0,1]
 *   nikola.state.entropy  double Shannon entropy of ψ-field
 *   nikola.state.td_error double TD prediction error
 *   nikola.action         string last selected ActionType name
 */
#pragma once

#ifndef NIKOLA_OTEL_DISABLED

#include <nikola/autonomy/decision_loop.hpp>   // NikolaState, action_name

#include <opentelemetry/trace/provider.h>
#include <opentelemetry/trace/scope.h>
#include <opentelemetry/trace/tracer_provider.h>
#include <opentelemetry/sdk/trace/provider.h>
#include <opentelemetry/sdk/trace/simple_processor_factory.h>
#include <opentelemetry/sdk/trace/tracer_provider_factory.h>
#include <opentelemetry/exporters/memory/in_memory_span_data.h>
#include <opentelemetry/exporters/memory/in_memory_span_exporter_factory.h>
#include <opentelemetry/exporters/ostream/span_exporter_factory.h>

#include <memory>
#include <cstdint>
#include <ostream>

namespace nikola::telemetry {

// ─────────────────────────────────────────────────────────────────────────────
// Helpers — singleton provider swap
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Install an OStream exporter as the global TracerProvider.
 *
 * Spans are written as human-readable text to `out` (default: std::cerr).
 * Calling this again replaces the previous provider.
 */
inline void setup_ostream_tracer(std::ostream& out = std::cerr)
{
    namespace sdktrace = opentelemetry::sdk::trace;
    namespace ostream  = opentelemetry::exporter::trace;
    namespace trace    = opentelemetry::trace;

    auto exporter  = ostream::OStreamSpanExporterFactory::Create(out);
    auto processor = sdktrace::SimpleSpanProcessorFactory::Create(std::move(exporter));
    auto provider  = sdktrace::TracerProviderFactory::Create(std::move(processor));

    // Install into the global singleton via the SDK provider.
    sdktrace::Provider::SetTracerProvider(
        opentelemetry::nostd::shared_ptr<trace::TracerProvider>(provider.release()));
}

/**
 * @brief Install an in-memory exporter as the global TracerProvider.
 *
 * Returns the shared InMemorySpanData handle so the caller can inspect
 * captured spans.  Primarily used for testing.
 *
 * @param buffer_size Maximum number of spans buffered (default: 512).
 * @return shared_ptr to InMemorySpanData; call GetSpans() to read.
 */
inline std::shared_ptr<opentelemetry::exporter::memory::InMemorySpanData>
setup_in_memory_tracer(std::size_t buffer_size = 512)
{
    namespace sdktrace = opentelemetry::sdk::trace;
    namespace memory   = opentelemetry::exporter::memory;
    namespace trace    = opentelemetry::trace;

    std::shared_ptr<memory::InMemorySpanData> span_data;
    auto exporter  = memory::InMemorySpanExporterFactory::Create(span_data, buffer_size);
    auto processor = sdktrace::SimpleSpanProcessorFactory::Create(std::move(exporter));
    auto provider  = sdktrace::TracerProviderFactory::Create(std::move(processor));

    sdktrace::Provider::SetTracerProvider(
        opentelemetry::nostd::shared_ptr<trace::TracerProvider>(provider.release()));

    return span_data;
}

/**
 * @brief Restore a no-op TracerProvider (clean up after tests).
 */
inline void teardown_tracer()
{
    namespace trace = opentelemetry::trace;
    opentelemetry::sdk::trace::Provider::SetTracerProvider(
        opentelemetry::nostd::shared_ptr<trace::TracerProvider>(
            new trace::NoopTracerProvider));
}

// ─────────────────────────────────────────────────────────────────────────────
// TickTracer — stateless RAII wrapper
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Emits one OTel span per DecisionLoop tick via trace_tick().
 *
 * Designed to be called from loop.on_tick:
 * @code
 *   TickTracer tt;
 *   loop.on_tick = [&](const NikolaState& s) { tt.trace_tick(s, tick_index); };
 * @endcode
 *
 * Each call starts a "nikola.tick" span, adds all NikolaState fields as
 * double attributes, then immediately ends it.  The span duration therefore
 * reflects only the tracing overhead — it is NOT a wall-clock measurement of
 * the tick itself (that responsibility belongs to the tick loop, not the
 * tracer).
 *
 * Thread-safe: the underlying OTel SDK handles concurrent access.
 */
class TickTracer {
public:
    static constexpr char kSpanName[] = "nikola.tick";

    /**
     * @brief Emit one "nikola.tick" span for the given state.
     *
     * @param s     Snapshot of NikolaState after tick completes.
     * @param index Monotonic tick counter (loop.tick_count()).
     */
    void trace_tick(const autonomy::NikolaState& s, int64_t index) const
    {
        namespace trace = opentelemetry::trace;

        auto provider = trace::Provider::GetTracerProvider();
        auto tracer   = provider->GetTracer("nikola", "0.0.4");
        auto span     = tracer->StartSpan(kSpanName);

        span->SetAttribute("nikola.tick.index",     index);
        span->SetAttribute("nikola.state.energy",   static_cast<double>(s.torus_energy));
        span->SetAttribute("nikola.state.dopamine", static_cast<double>(s.dopamine));
        span->SetAttribute("nikola.state.atp",      static_cast<double>(s.atp));
        span->SetAttribute("nikola.state.boredom",  static_cast<double>(s.boredom));
        span->SetAttribute("nikola.state.entropy",  static_cast<double>(s.entropy));
        span->SetAttribute("nikola.state.td_error", static_cast<double>(s.td_error));
        span->SetAttribute("nikola.action",
                           opentelemetry::nostd::string_view{
                               autonomy::action_name(s.last_action)});
        span->End();
    }
};

} // namespace nikola::telemetry

#else  // NIKOLA_OTEL_DISABLED ──────────────────────────────────────────────────

#include <nikola/autonomy/decision_loop.hpp>
#include <memory>
#include <ostream>
#include <cstdint>

namespace nikola::telemetry {

// Null implementations for builds without OTel.
inline void setup_ostream_tracer(std::ostream& = std::cerr) noexcept {}
inline std::nullptr_t setup_in_memory_tracer(std::size_t = 512) noexcept { return nullptr; }
inline void teardown_tracer() noexcept {}

struct TickTracer {
    static constexpr char kSpanName[] = "nikola.tick";
    void trace_tick(const autonomy::NikolaState&, int64_t) const noexcept {}
};

} // namespace nikola::telemetry

#endif // NIKOLA_OTEL_DISABLED
