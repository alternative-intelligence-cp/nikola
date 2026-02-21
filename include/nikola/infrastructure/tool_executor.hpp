/**
 * @file include/nikola/infrastructure/tool_executor.hpp
 * @brief Tool Executor: function-calling interface for Nikola agents.
 *
 * Provides a strongly-typed registry for callable "tools" — actions that the
 * Orchestrator's AI loop may invoke in response to a reasoning cycle.
 *
 * Tools are registered as C++ callables; the executor dispatches by name,
 * passes a JSON argument string, and returns a JSON result string.
 *
 * This design mirrors the "function-calling" API pattern in modern LLM SDKs
 * (OpenAI, Gemini) while staying transport-agnostic.
 *
 * Integration with Phase 4:
 *   • ToolExecutor lives inside the Orchestrator; the cogntive core emits
 *     ToolCall structs, the executor dispatches them.
 *   • Results are fed back to the LLMBridge or directly into the wavefunction
 *     as an injected emitter field.
 *   • A CircuitBreaker guards each tool so repeated failures isolate gracefully.
 */

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nikola/infrastructure/circuit_breaker.hpp>

namespace nikola::infrastructure {

// ---------------------------------------------------------------------------
// ToolCall / ToolResult
// ---------------------------------------------------------------------------

/// A function-call request emitted by the cognitive core.
struct ToolCall {
    std::string call_id;      ///< Unique correlator
    std::string tool_name;    ///< Registered tool identifier
    std::string args_json;    ///< Arguments serialised as JSON
    int         proto_version = 1; ///< Gap 4.5 — schema version
};

/// Result returned to the reasoning engine.
struct ToolResult {
    std::string call_id;
    std::string tool_name;
    std::string result_json; ///< Output serialised as JSON; empty on error
    bool        ok = false;
    std::string error_msg;
    std::chrono::microseconds elapsed{0};
};

// ---------------------------------------------------------------------------
// ToolDefinition
// ---------------------------------------------------------------------------

/// Metadata describing a registered tool.
struct ToolDefinition {
    std::string name;
    std::string description;
    std::string args_schema_json;   ///< JSON Schema of the expected args
    std::string result_schema_json; ///< JSON Schema of the result
};

// ---------------------------------------------------------------------------
// ToolFn
// ---------------------------------------------------------------------------

/// The callable type for tool implementations.
/// Receives args_json, returns result_json.  Throws on hard errors.
using ToolFn = std::function<std::string(const std::string& /*args_json*/)>;

// ---------------------------------------------------------------------------
// ToolExecutor
// ---------------------------------------------------------------------------

/**
 * @class ToolExecutor
 * @brief Registry and dispatcher for named callable tools.
 *
 * Features:
 *   • Named registration with optional JSON schema metadata.
 *   • Per-tool CircuitBreaker (threshold = 3, cool-down = 500 ms).
 *   • Execution timing tracked in ToolResult::elapsed.
 *   • Unknown tool calls return an error ToolResult (no exception propagation).
 */
class ToolExecutor {
public:
    ToolExecutor() = default;

    // -----------------------------------------------------------------------
    // Registration
    // -----------------------------------------------------------------------

    /**
     * @brief Register a new tool.
     * @param def   Metadata (name, description, schemas).
     * @param fn    The implementation callable.
     */
    void register_tool(ToolDefinition def, ToolFn fn) {
        const std::string key = def.name;
        auto breaker = std::make_unique<CircuitBreaker>(CircuitBreaker::Config{
            .failure_threshold = ZMQ_MAX_RETRIES,
            .cool_down         = std::chrono::milliseconds(500),
            .component_name    = key
        });
        tools_.emplace(key, Entry{std::move(def), std::move(fn), std::move(breaker)});
    }

    /// Remove a tool by name.  No-op if not found.
    void deregister_tool(const std::string& name) {
        tools_.erase(name);
    }

    // -----------------------------------------------------------------------
    // Execution
    // -----------------------------------------------------------------------

    /**
     * @brief Execute a ToolCall synchronously.
     * Never throws; errors are encoded in ToolResult::ok / error_msg.
     */
    [[nodiscard]] ToolResult execute(const ToolCall& call) {
        ToolResult result;
        result.call_id   = call.call_id;
        result.tool_name = call.tool_name;

        auto it = tools_.find(call.tool_name);
        if (it == tools_.end()) {
            result.ok        = false;
            result.error_msg = "Unknown tool: " + call.tool_name;
            return result;
        }

        auto& entry = it->second;
        auto& cb    = *entry.breaker;

        RetryPolicy policy{1, ZMQ_BACKOFF_BASE_MS, MessagePriority::CONTROL};

        auto t0 = std::chrono::steady_clock::now();
        bool ok = retry_with_circuit_breaker([&]() -> bool {
            try {
                result.result_json = entry.fn(call.args_json);
                result.ok = true;
                return true;
            } catch (const std::exception& ex) {
                result.error_msg = ex.what();
                return false;
            } catch (...) {
                result.error_msg = "tool threw unknown exception";
                return false;
            }
        }, cb, policy);

        result.elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0);
        result.ok = ok;
        return result;
    }

    /**
     * @brief Execute a batch of calls.
     * Calls are dispatched sequentially for now (parallel dispatch in Phase 7).
     */
    [[nodiscard]] std::vector<ToolResult> execute_batch(const std::vector<ToolCall>& calls) {
        std::vector<ToolResult> results;
        results.reserve(calls.size());
        for (const auto& call : calls) {
            results.push_back(execute(call));
        }
        return results;
    }

    // -----------------------------------------------------------------------
    // Inspection
    // -----------------------------------------------------------------------

    [[nodiscard]] bool has_tool(const std::string& name) const {
        return tools_.count(name) > 0;
    }

    [[nodiscard]] std::size_t tool_count() const noexcept { return tools_.size(); }

    [[nodiscard]] std::vector<ToolDefinition> list_tools() const {
        std::vector<ToolDefinition> out;
        out.reserve(tools_.size());
        for (const auto& [name, e] : tools_) {
            out.push_back(e.def);
        }
        return out;
    }

    /// Get the circuit-breaker state for a tool.
    [[nodiscard]] CBState tool_cb_state(const std::string& name) const {
        auto it = tools_.find(name);
        if (it == tools_.end()) return CBState::OPEN; // unknown = circuit open
        return it->second.breaker->state();
    }

    /// Reset the circuit breaker for a tool (e.g., after component restart).
    void reset_tool_cb(const std::string& name) {
        auto it = tools_.find(name);
        if (it != tools_.end()) it->second.breaker->reset();
    }

private:
    struct Entry {
        ToolDefinition   def;
        ToolFn           fn;
        std::unique_ptr<CircuitBreaker> breaker;  // unique_ptr: Entry is movable
    };

    std::unordered_map<std::string, Entry> tools_;
};

} // namespace nikola::infrastructure
