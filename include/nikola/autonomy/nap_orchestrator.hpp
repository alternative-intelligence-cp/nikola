/**
 * @file autonomy/nap_orchestrator.hpp
 * @brief v0.1.18 — NAP System orchestrator: wires NapController, consolidation,
 *        dream-weave, dream engine, and checkpointing into a unified nap lifecycle.
 *
 * During NAP:
 *   1. Entry:    checkpoint cognitive state, copy psi to dream buffers,
 *                z-normalize dream content
 *   2. Per-tick: recharge ATP, run consolidation (once), dream-weave (once),
 *                dream engine cycle (once)
 *   3. Exit:     finalize NapCycleReport, restore AWAKE state
 *
 * The orchestrator doesn't own heavy subsystems — it accepts pointers and
 * callbacks so callers can wire whatever combination they need.
 *
 * Reference:
 *   RELEASE_0.1.18.md Phases 1–3
 *   Engineering Report §6.4 (NAP System), §5.1 (ENGS — ATP depletion triggers)
 *   Integration Report TASK-013 (NAP scheduling algorithm)
 */

#pragma once

#include <nikola/autonomy/nap_controller.hpp>
#include <nikola/autonomy/dream_weave.hpp>
#include <nikola/interior/dream_engine.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace nikola::autonomy {

// ============================================================================
// Configuration
// ============================================================================

struct NapOrchestratorConfig {
    /// ATP recharge per second during NAP ([0,1] normalized).
    float atp_recharge_rate    = 0.05f;

    /// Norepinephrine level during nap.  Low = explore mode (diverse replay).
    float norepinephrine_level = 0.3f;

    /// Whether to save a DMC checkpoint on nap entry.
    bool  checkpoint_on_entry  = true;

    /// Whether to run DreamWeaveEngine during nap.
    bool  run_dream_weave      = true;

    /// Whether to run DreamEngine cycle during nap.
    bool  run_dream_engine     = true;

    /// Whether to run consolidation callback during nap.
    bool  run_consolidation    = true;
};

// ============================================================================
// Consolidation result (returned by user callback)
// ============================================================================

struct ConsolidationResult {
    size_t pruned   = 0;   ///< Records pruned during consolidation
    int    replayed = 0;   ///< Memories replayed (Hebbian strengthening)
};

// ============================================================================
// NAP cycle report
// ============================================================================

struct NapCycleReport {
    float       duration_s            = 0.f;
    float       atp_at_entry          = 0.f;
    float       atp_at_exit           = 0.f;
    bool        checkpointed          = false;

    // Consolidation
    size_t      memories_pruned       = 0;
    int         memories_replayed     = 0;

    // Dream-weave (Frobenius convergence)
    int         dream_iterations      = 0;
    bool        dream_converged       = false;
    float       dream_final_delta     = 0.f;

    // Dream engine (fragment discovery)
    size_t      dream_fragments       = 0;
    size_t      dream_memories_formed = 0;

    const char* exit_reason           = "";
};

// ============================================================================
// NapOrchestrator
// ============================================================================

/**
 * @class NapOrchestrator
 * @brief Drives the full NAP lifecycle by wiring NapController to subsystems.
 *
 * Usage:
 * @code
 *   NapOrchestrator orch;
 *   orch.set_consolidation_fn([&]() -> ConsolidationResult {
 *       auto s = consolidation.nap_cycle(mem, wf);
 *       return {s.pruned, s.replayed};
 *   });
 *   orch.set_dream_weave(&dream_weave);
 *   orch.set_dream_stepper([&](auto r, auto i) { prop.step(r, i); });
 *   orch.set_dream_engine(&dream_engine);
 *   orch.set_checkpoint_fn([&]() { return save_checkpoint(path, snap); });
 *   orch.init_dream_buffers(psi_r, psi_i);
 *
 *   // Per tick:
 *   orch.update(atp, time, dt);
 * @endcode
 */
class NapOrchestrator {
public:
    using ConsolidationFn = std::function<ConsolidationResult()>;
    using CheckpointFn    = std::function<bool()>;
    using Stepper         = DreamWeaveEngine::Stepper;

    explicit NapOrchestrator(const NapOrchestratorConfig& cfg = {})
        : cfg_(cfg)
    {
        wire_callbacks_();
    }

    // ── subsystem wiring ─────────────────────────────────────────────────

    void set_dream_weave(DreamWeaveEngine* dw)           { dream_weave_ = dw; }
    void set_dream_engine(interior::DreamEngine* de)      { dream_engine_ = de; }
    void set_dream_stepper(Stepper s)                     { stepper_ = std::move(s); }
    void set_consolidation_fn(ConsolidationFn fn)         { consolidate_fn_ = std::move(fn); }
    void set_checkpoint_fn(CheckpointFn fn)               { checkpoint_fn_ = std::move(fn); }

    /**
     * @brief Copy waking psi buffers for dream isolation.
     *
     * Called once before the nap might begin (e.g., every tick, or when
     * the waking state changes).  On nap entry, these are copied into
     * internal dream buffers — the originals remain untouched.
     */
    void init_dream_buffers(std::span<const float> real,
                            std::span<const float> imag) {
        waking_psi_real_.assign(real.begin(), real.end());
        waking_psi_imag_.assign(imag.begin(), imag.end());
    }

    // ── main update ──────────────────────────────────────────────────────

    /**
     * @brief Drive the nap state machine — call every simulation tick.
     *
     * @param atp           In/out ATP level [0, 1].  Recharged during nap.
     * @param current_time  Monotonic wall-clock seconds (any epoch).
     * @param dt            Time delta since last call (seconds).
     */
    void update(float& atp, float current_time, float dt) {
        // Recharge ATP while napping (BEFORE controller check so that
        // the controller sees the updated level for exit threshold).
        if (controller_.is_napping()) {
            atp = std::min(1.0f, atp + cfg_.atp_recharge_rate * dt);
        }

        current_atp_ = atp;
        controller_.update(atp, current_time);
    }

    // ── observers ────────────────────────────────────────────────────────

    [[nodiscard]] bool        is_napping()   const noexcept { return controller_.is_napping(); }
    [[nodiscard]] NapState    state()        const noexcept { return controller_.state(); }
    [[nodiscard]] uint32_t    nap_count()    const noexcept { return controller_.nap_count(); }

    [[nodiscard]] const NapCycleReport&        last_report() const noexcept { return last_report_; }
    [[nodiscard]] const NapController&         controller()  const noexcept { return controller_; }
    [[nodiscard]] NapController&               controller()        noexcept { return controller_; }
    [[nodiscard]] const NapOrchestratorConfig&  config()     const noexcept { return cfg_; }

    /// Access dream buffers (read-only) — useful for diagnostics.
    [[nodiscard]] const std::vector<float>& dream_psi_real() const noexcept { return dream_psi_real_; }
    [[nodiscard]] const std::vector<float>& dream_psi_imag() const noexcept { return dream_psi_imag_; }

    // ── z-score normalization utility ──────────────────────────────────────

    /**
     * @brief Z-score normalize an array in place: x_i ← (x_i − μ) / σ.
     *
     * If variance is near zero (< 1e-12), stddev defaults to 1.0 (no scaling).
     */
    static void z_normalize(std::span<float> data) noexcept {
        if (data.empty()) return;
        const auto N = static_cast<float>(data.size());
        float sum = 0.f, sq_sum = 0.f;
        for (float v : data) { sum += v; sq_sum += v * v; }
        float mean   = sum / N;
        float var    = sq_sum / N - mean * mean;
        float stddev = (var > 1e-12f) ? std::sqrt(var) : 1.0f;
        for (float& v : data) v = (v - mean) / stddev;
    }

private:
    NapOrchestratorConfig  cfg_;
    NapController          controller_;

    // ── subsystem pointers (non-owning) ──────────────────────────────────
    DreamWeaveEngine*       dream_weave_   = nullptr;
    interior::DreamEngine*  dream_engine_  = nullptr;
    Stepper                 stepper_;
    ConsolidationFn         consolidate_fn_;
    CheckpointFn            checkpoint_fn_;

    // ── psi buffers ──────────────────────────────────────────────────────
    std::vector<float> waking_psi_real_;   ///< snapshot at init_dream_buffers()
    std::vector<float> waking_psi_imag_;
    std::vector<float> dream_psi_real_;    ///< working copy during nap
    std::vector<float> dream_psi_imag_;

    // ── per-nap state ────────────────────────────────────────────────────
    float    current_atp_   = 1.0f;
    bool     consolidated_  = false;
    bool     dreamed_       = false;
    bool     dream_cycled_  = false;
    uint64_t tick_counter_  = 0;

    NapCycleReport current_report_{};
    NapCycleReport last_report_{};

    // ── callback wiring ──────────────────────────────────────────────────

    void wire_callbacks_() {
        controller_.on_enter_nap = [this]() {
            current_report_ = {};
            current_report_.atp_at_entry = current_atp_;

            // ── 1. Checkpoint ────────────────────────────────────────────
            if (cfg_.checkpoint_on_entry && checkpoint_fn_) {
                current_report_.checkpointed = checkpoint_fn_();
            }

            // ── 2. Copy waking psi → dream buffers (isolation) ──────────
            dream_psi_real_ = waking_psi_real_;
            dream_psi_imag_ = waking_psi_imag_;

            // ── 3. Z-normalize dream content ─────────────────────────────
            if (!dream_psi_real_.empty()) {
                z_normalize(dream_psi_real_);
                z_normalize(dream_psi_imag_);
            }

            consolidated_ = false;
            dreamed_      = false;
            dream_cycled_ = false;
        };

        controller_.on_nap_tick = [this](float /*elapsed*/) {
            ++tick_counter_;

            // ── Consolidation (once per nap) ─────────────────────────────
            if (cfg_.run_consolidation && !consolidated_ && consolidate_fn_) {
                auto result = consolidate_fn_();
                current_report_.memories_pruned  = result.pruned;
                current_report_.memories_replayed = result.replayed;
                consolidated_ = true;
            }

            // ── Dream-weave: Frobenius convergence (once per nap) ────────
            if (cfg_.run_dream_weave && !dreamed_
                && dream_weave_ && stepper_
                && !dream_psi_real_.empty()) {
                auto result = dream_weave_->run(
                    dream_psi_real_, dream_psi_imag_, stepper_);
                current_report_.dream_iterations  = result.iterations;
                current_report_.dream_converged   = result.converged;
                current_report_.dream_final_delta = result.final_delta;
                dreamed_ = true;
            }

            // ── Dream engine cycle: fragment discovery (once per nap) ────
            if (cfg_.run_dream_engine && !dream_cycled_ && dream_engine_) {
                auto cycle = dream_engine_->dream(tick_counter_);
                current_report_.dream_fragments       = cycle.fragments_found;
                current_report_.dream_memories_formed  = cycle.memories_formed;
                dream_cycled_ = true;
            }
        };

        controller_.on_exit_nap = [this]() {
            current_report_.atp_at_exit  = current_atp_;
            current_report_.duration_s   = controller_.last_nap_duration();
            current_report_.exit_reason  = controller_.last_exit_reason();
            last_report_ = current_report_;
        };
    }
};

} // namespace nikola::autonomy
