/**
 * @file autonomy/autonomy_engine.hpp
 * @brief Phase 5 integration facade: combines all autonomous-system components.
 *
 * AutonomyEngine owns and drives:
 *   - DopamineSystem    (Gap 5.1 — TD prediction error)
 *   - EntropyEstimator  (Gap 5.2 — Monte Carlo H)
 *   - BoredomRegulator  (Gap 5.2 — explore trigger)
 *   - MetabolicSimulator (Gap 5.3 — ATP cost)
 *   - NapController     (Gap 5.4 — hysteresis nap cycle)
 *   - DreamWeaveEngine  (Gap 5.5 — Frobenius consolidation)
 *
 * Designed for header-only use (no separate .cpp) behind a Pimpl guard
 * for ABI stability — same pattern as Orchestrator.
 *
 * Public API (no heavy includes — just basic stdlib):
 *   engine.tick(dt, psi_r, psi_i, reward, wall_time)
 *   engine.atp(), .boredom(), .dopamine()
 *   engine.is_napping(), .is_exploring()
 *
 * Impl (behind NIKOLA_AUTONOMY_ENGINE_IMPL):
 *   Full implementation includes all 5 gap headers.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>

// All autonomy gap headers are lightweight (stdlib only) — include upfront
#include <nikola/autonomy/dopamine_system.hpp>
#include <nikola/autonomy/entropy_estimator.hpp>
#include <nikola/autonomy/hamiltonian_value.hpp>   // NIK-005
#include <nikola/autonomy/metabolic_simulator.hpp>
#include <nikola/autonomy/nap_controller.hpp>
#include <nikola/autonomy/dream_weave.hpp>

namespace nikola::autonomy {

// ── AutonomyState ─────────────────────────────────────────────────────────────

enum class AutonomyState : uint8_t {
    ACTIVE,     ///< Normal operation — processing inputs, emitting outputs
    EXPLORING,  ///< Boredom-triggered — injecting stochastic novelty
    NAPPING,    ///< ATP depleted — Dream-Weave consolidation running
    INHIBITED,  ///< External override — all autonomy suppressed
};

[[nodiscard]] inline const char* autonomy_state_name(AutonomyState s) noexcept {
    switch (s) {
        case AutonomyState::ACTIVE:    return "ACTIVE";
        case AutonomyState::EXPLORING: return "EXPLORING";
        case AutonomyState::NAPPING:   return "NAPPING";
        case AutonomyState::INHIBITED: return "INHIBITED";
    }
    return "UNKNOWN";
}

// ── AutonomyConfig ────────────────────────────────────────────────────────────

struct AutonomyConfig {
    float initial_atp          = 1.0f;
    float entropy_sample_dt    = 0.1f;   ///< how often to re-sample entropy (s)
    bool  enable_dream_weave   = true;
    bool  enable_boredom       = true;
    uint32_t entropy_rng_seed  = 42u;
};

// ── AutonomySnapshot (telemetry) ───────────────────────────────────────────────

struct AutonomySnapshot {
    float         atp;
    float         dopamine;
    float         boredom;
    float         entropy;
    AutonomyState state;
    uint32_t      nap_count;
    uint32_t      dream_convergences;
};

// ── forward declaration for Pimpl ────────────────────────────────────────────

struct AutonomyEngineImpl;

// ── AutonomyEngine ────────────────────────────────────────────────────────────

/**
 * @class AutonomyEngine
 * @brief ABI-stable facade orchestrating all Phase 5 autonomous systems.
 *
 * Thread safety: NOT thread-safe — drive from a single control thread.
 * The NapController callbacks execute on the same thread.
 */
class AutonomyEngine {
public:
    explicit AutonomyEngine(AutonomyConfig cfg = {});
    ~AutonomyEngine();

    // non-copyable; movable
    AutonomyEngine(const AutonomyEngine&)            = delete;
    AutonomyEngine& operator=(const AutonomyEngine&) = delete;
    AutonomyEngine(AutonomyEngine&&)                 noexcept;
    AutonomyEngine& operator=(AutonomyEngine&&)      noexcept;

    // ── primary interface ─────────────────────────────────────────────────

    /**
     * @brief Advance all autonomy systems by one timestep.
     *
     * @note  The dopamine signal in this overload uses Σ|Ψ|² as the value
     *        estimate.  For a standing wave this oscillates at 2ω, causing
     *        spurious negative TD errors ("Stroboscopic Value Collapse",
     *        NIK-005).  Prefer tick_physics() when vel spans are available.
     *
     * @param dt          Elapsed seconds since last tick.
     * @param psi_real    Re(Ψ) span (may be empty — entropy/ATP use last).
     * @param psi_imag    Im(Ψ) span.
     * @param reward      External reward signal for dopamine update.
     * @param wall_time   Monotonic wall-clock seconds (for NapController).
     */
    void tick(float                  dt,
              std::span<const float> psi_real,
              std::span<const float> psi_imag,
              Reward                 reward    = Reward::NEUTRAL,
              float                  wall_time = 0.0f);

    /**
     * @brief Advance all autonomy systems using the full Hamiltonian value.
     *
     * Resolves NIK-005: uses HamiltonianValue::compute_spans() which includes
     * kinetic energy (|V|²), making the dopamine signal invariant for stable
     * standing waves — TD error δ → 0 for conserved states.
     *
     * @param dt          Elapsed seconds since last tick.
     * @param psi_real    Re(Ψ) span.
     * @param psi_imag    Im(Ψ) span.
     * @param vel_real    Re(∂_t Ψ) span — Störmer-Verlet velocity field.
     * @param vel_imag    Im(∂_t Ψ) span.
     * @param beta        Nonlinear coupling constant β.
     * @param reward      External reward signal for dopamine update.
     * @param wall_time   Monotonic wall-clock seconds (for NapController).
     */
    void tick_physics(float                  dt,
                      std::span<const float> psi_real,
                      std::span<const float> psi_imag,
                      std::span<const float> vel_real,
                      std::span<const float> vel_imag,
                      float                  beta      = 0.0f,
                      Reward                 reward    = Reward::NEUTRAL,
                      float                  wall_time = 0.0f);

    /// Expose the HamiltonianValue config (γ weights, H_max) for tuning.
    [[nodiscard]] HamiltonianValue&       hamiltonian_value()       noexcept;
    [[nodiscard]] const HamiltonianValue& hamiltonian_value() const noexcept;

    /// Override: allow the nap stepper used by DreamWeaveEngine during naps.
    void set_dream_stepper(DreamWeaveEngine::Stepper stepper);

    /// Callback fired when boredom-driven exploration begins.
    std::function<void()> on_explore;

    /// Callback fired on nap entry.
    std::function<void()> on_nap_enter;

    /// Callback fired on nap exit.
    std::function<void()> on_nap_exit;

    // ── observers ────────────────────────────────────────────────────────────

    [[nodiscard]] float         atp()        const noexcept;
    [[nodiscard]] float         dopamine()   const noexcept;
    [[nodiscard]] float         boredom()    const noexcept;
    [[nodiscard]] float         entropy()    const noexcept;
    [[nodiscard]] AutonomyState state()      const noexcept;
    [[nodiscard]] bool          is_napping()   const noexcept;
    [[nodiscard]] bool          is_exploring() const noexcept;

    [[nodiscard]] AutonomySnapshot snapshot() const noexcept;

    // ── direct component access (for tests / telemetry) ───────────────────

    [[nodiscard]] const DopamineSystem&     dopamine_system()    const noexcept;
    [[nodiscard]] const BoredomRegulator&   boredom_regulator()  const noexcept;
    [[nodiscard]] const MetabolicSimulator& metabolic()          const noexcept;
    [[nodiscard]] const NapController&      nap_controller()     const noexcept;
    [[nodiscard]] const DreamWeaveEngine&   dream_weave()        const noexcept;

private:
    std::unique_ptr<AutonomyEngineImpl> impl_;
};

} // namespace nikola::autonomy

// ─────────────────────────────────────────────────────────────────────────────
//  IMPLEMENTATION  (header-only, behind guard — same pattern as Orchestrator)
// ─────────────────────────────────────────────────────────────────────────────

#ifdef NIKOLA_AUTONOMY_ENGINE_IMPL

// All gap headers already included above; impl just adds function bodies.
namespace nikola::autonomy {

struct AutonomyEngineImpl {
    AutonomyConfig     cfg;
    DopamineSystem     dopamine;
    EntropyEstimator   entropy_est;
    BoredomRegulator   boredom;
    MetabolicSimulator metabolic;
    NapController      nap;
    DreamWeaveEngine   dream;
    HamiltonianValue   hamiltonian_value_fn;  // NIK-005

    float last_entropy = 0.0f;
    float entropy_acc  = 0.0f;   // accumulates dt for subset sampling

    DreamWeaveEngine::Stepper dream_stepper;  // optional user override

    explicit AutonomyEngineImpl(AutonomyConfig c)
        : cfg(c)
        , entropy_est(c.entropy_rng_seed)
        , metabolic(c.initial_atp)
    {}
};

// ── AutonomyEngine constructor / destructor ───────────────────────────────────

AutonomyEngine::AutonomyEngine(AutonomyConfig cfg)
    : impl_(std::make_unique<AutonomyEngineImpl>(std::move(cfg)))
{
    // Wire NapController callbacks
    impl_->nap.on_enter_nap = [this]() {
        if (on_nap_enter) on_nap_enter();
    };
    impl_->nap.on_exit_nap = [this]() {
        if (on_nap_exit) on_nap_exit();
    };
    impl_->nap.on_nap_tick = [this](float /*elapsed*/) {
        if (!impl_->cfg.enable_dream_weave) return;
        if (!impl_->dream_stepper)           return;
        // DreamWeave runs lazily — caller provides psi buffers via tick()
    };
}

AutonomyEngine::~AutonomyEngine() = default;

AutonomyEngine::AutonomyEngine(AutonomyEngine&&) noexcept = default;
AutonomyEngine& AutonomyEngine::operator=(AutonomyEngine&&) noexcept = default;

// ── tick ──────────────────────────────────────────────────────────────────────

void AutonomyEngine::tick(float                  dt,
                          std::span<const float> psi_real,
                          std::span<const float> psi_imag,
                          Reward                 reward,
                          float                  wall_time)
{
    auto& I = *impl_;

    // --- 1. Total energy (Σ|Ψ|²) for dopamine — NOTE: stroboscopic (NIK-005)
    //         Use tick_physics() to pass vel spans for the invariant Hamiltonian.
    float total_energy = 0.0f;
    for (std::size_t i = 0, n = std::min(psi_real.size(), psi_imag.size()); i < n; ++i) {
        float r = psi_real[i], im = psi_imag[i];
        total_energy += r*r + im*im;
    }

    // --- 2. Dopamine TD update + decay ---
    I.dopamine.update(total_energy, reward);
    I.dopamine.decay(dt);

    // --- 3. Entropy + boredom (sampled at cfg.entropy_sample_dt rate) ---
    if (I.cfg.enable_boredom && !psi_real.empty()) {
        I.entropy_acc += dt;
        if (I.entropy_acc >= I.cfg.entropy_sample_dt) {
            I.last_entropy = I.entropy_est.estimate(psi_real, psi_imag);
            I.entropy_acc  = 0.0f;
        }
        I.boredom.update(I.last_entropy, dt);

        if (I.boredom.should_explore() && on_explore) {
            on_explore();
        }
    }

    // --- 4. Metabolic cost — proxy energy rate from total energy ---
    if (!I.nap.is_napping()) {
        I.metabolic.consume_by_rate(total_energy, dt);
    } else {
        I.metabolic.recharge(dt);
    }

    // --- 5. Nap state machine ---
    I.nap.update(I.metabolic.atp(), wall_time);
}

// ── tick_physics — NIK-005: Hamiltonian-based stable dopamine ─────────────────

void AutonomyEngine::tick_physics(
        float                  dt,
        std::span<const float> psi_real,
        std::span<const float> psi_imag,
        std::span<const float> vel_real,
        std::span<const float> vel_imag,
        float                  beta,
        Reward                 reward,
        float                  wall_time)
{
    auto& I = *impl_;

    // --- 1. Full Hamiltonian H = γ_K|V|² + γ_P|Ψ|² + γ_NL β/2|Ψ|⁴ (no 2ω flicker)
    const float total_energy = I.hamiltonian_value_fn.compute_spans(
            psi_real, psi_imag, vel_real, vel_imag, beta);

    // --- 2. Dopamine TD update + decay ---
    I.dopamine.update(total_energy, reward);
    I.dopamine.decay(dt);

    // --- 3. Entropy + boredom ---
    if (I.cfg.enable_boredom && !psi_real.empty()) {
        I.entropy_acc += dt;
        if (I.entropy_acc >= I.cfg.entropy_sample_dt) {
            I.last_entropy = I.entropy_est.estimate(psi_real, psi_imag);
            I.entropy_acc  = 0.0f;
        }
        I.boredom.update(I.last_entropy, dt);

        if (I.boredom.should_explore() && on_explore) {
            on_explore();
        }
    }

    // --- 4. Metabolic cost ---
    if (!I.nap.is_napping()) {
        I.metabolic.consume_by_rate(total_energy, dt);
    } else {
        I.metabolic.recharge(dt);
    }

    // --- 5. Nap state machine ---
    I.nap.update(I.metabolic.atp(), wall_time);
}

HamiltonianValue&       AutonomyEngine::hamiltonian_value()       noexcept { return impl_->hamiltonian_value_fn; }
const HamiltonianValue& AutonomyEngine::hamiltonian_value() const noexcept { return impl_->hamiltonian_value_fn; }

void AutonomyEngine::set_dream_stepper(DreamWeaveEngine::Stepper s) {
    impl_->dream_stepper = std::move(s);
}

// ── observers ─────────────────────────────────────────────────────────────────

float         AutonomyEngine::atp()       const noexcept { return impl_->metabolic.atp(); }
float         AutonomyEngine::dopamine()  const noexcept { return impl_->dopamine.level(); }
float         AutonomyEngine::boredom()   const noexcept { return impl_->boredom.level(); }
float         AutonomyEngine::entropy()   const noexcept { return impl_->last_entropy; }

AutonomyState AutonomyEngine::state() const noexcept {
    if (impl_->nap.is_napping())        return AutonomyState::NAPPING;
    if (impl_->boredom.should_explore()) return AutonomyState::EXPLORING;
    return AutonomyState::ACTIVE;
}

bool AutonomyEngine::is_napping()   const noexcept { return impl_->nap.is_napping(); }
bool AutonomyEngine::is_exploring() const noexcept { return impl_->boredom.should_explore(); }

AutonomySnapshot AutonomyEngine::snapshot() const noexcept {
    return {
        impl_->metabolic.atp(),
        impl_->dopamine.level(),
        impl_->boredom.level(),
        impl_->last_entropy,
        state(),
        impl_->nap.nap_count(),
        impl_->dream.convergence_count(),
    };
}

const DopamineSystem&     AutonomyEngine::dopamine_system()   const noexcept { return impl_->dopamine; }
const BoredomRegulator&   AutonomyEngine::boredom_regulator() const noexcept { return impl_->boredom; }
const MetabolicSimulator& AutonomyEngine::metabolic()         const noexcept { return impl_->metabolic; }
const NapController&      AutonomyEngine::nap_controller()    const noexcept { return impl_->nap; }
const DreamWeaveEngine&   AutonomyEngine::dream_weave()       const noexcept { return impl_->dream; }

} // namespace nikola::autonomy

#endif // NIKOLA_AUTONOMY_ENGINE_IMPL
