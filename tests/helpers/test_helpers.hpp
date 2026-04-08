// ============================================================
// Shared Test Helpers for Nikola Integration & End-to-End Tests
// tests/helpers/test_helpers.hpp
// ============================================================
#pragma once

#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/cognitive/cognitive_core.hpp>
#include <nikola/physics/propagator.hpp>
#include <nikola/physics/wave_function.hpp>
#include <nikola/foundation/vector9d.hpp>

#include <cmath>
#include <string>
#include <vector>

namespace nikola::test {

// ── Factory: CognitiveTorus (no ONNX, non-GPU) ─────────────────────────────
inline cognitive::CognitiveTorus make_torus(int n = 3) {
    return cognitive::CognitiveTorus(n);
}

// ── Factory: AutonomyEngine with sensible test defaults ─────────────────────
inline autonomy::AutonomyEngine make_engine() {
    autonomy::AutonomyConfig cfg;
    cfg.enable_dream_weave = false;
    cfg.enable_boredom     = true;
    return autonomy::AutonomyEngine(cfg);
}

// ── Factory: DecisionLoopConfig with small vocabulary ───────────────────────
inline autonomy::DecisionLoopConfig make_loop_config() {
    autonomy::DecisionLoopConfig cfg;
    cfg.steps_per_tick      = 10;
    cfg.action_threshold    = 0.05f;
    cfg.min_emit_interval_s = 0.0f;
    cfg.decode_top_k        = 5;
    cfg.vocabulary          = {
        "hello", "curious", "wave", "energy", "nikola",
        "thought", "explore", "memory", "reason", "signal",
        "physics", "torus", "field", "quantum", "entropy",
        "dream", "nap", "reward", "spike", "calm"
    };
    return cfg;
}

// ── Factory: full pipeline (torus + engine + loop) ──────────────────────────
struct Pipeline {
    cognitive::CognitiveTorus torus;
    autonomy::AutonomyEngine  engine;
    autonomy::DecisionLoop    loop;

    Pipeline()
        : torus(make_torus()),
          engine(make_engine()),
          loop(torus, engine, make_loop_config()) {}
};

// ── Utility: check all floats in a NikolaState are finite ───────────────────
inline bool state_is_finite(const autonomy::NikolaState& s) {
    return std::isfinite(s.atp) && std::isfinite(s.boredom);
}

// ── Utility: run N ticks and collect results ────────────────────────────────
inline std::vector<autonomy::DecisionResult> run_ticks(
        autonomy::DecisionLoop& loop, int n) {
    std::vector<autonomy::DecisionResult> results;
    results.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        results.push_back(loop.tick());
    }
    return results;
}

// ── Utility: count a specific action type in results ────────────────────────
inline size_t count_action(const std::vector<autonomy::DecisionResult>& results,
                           autonomy::ActionType type) {
    size_t count = 0;
    for (const auto& r : results) {
        if (r.type == type) ++count;
    }
    return count;
}

}  // namespace nikola::test
