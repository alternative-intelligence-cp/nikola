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

#include <algorithm>   // std::min, std::max (Phase 51 ring buffer)
#include <array>        // std::array (Phase 51 mania ring)
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>

// All autonomy gap headers are lightweight (stdlib only) — include upfront
#include <nikola/autonomy/dopamine_system.hpp>
#include <nikola/autonomy/entropy_estimator.hpp>
#include <nikola/autonomy/goal_system.hpp>
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

// ── Phase 52: GAP-005 Cross-Coupling Matrix constants ──────────────────────────

/// M[0,1] = -κ_DS: Serotonin inhibits Dopamine (Opponent Process Theory, Daw et al.)
inline constexpr float COUPLING_M01 = -0.10f;
/// M[0,2] = +κ_DN: Norepinephrine amplifies Dopamine (Adaptive Gain Theory, Aston-Jones)
inline constexpr float COUPLING_M02 =  0.08f;
/// M[1,0] = +κ_SD: Dopamine stimulates Serotonin (Success → Confidence)
inline constexpr float COUPLING_M10 =  0.05f;
/// M[1,2] = -κ_SN: Norepinephrine inhibits Serotonin
inline constexpr float COUPLING_M12 = -0.07f;
/// M[2,1] = -κ_NS: Serotonin inhibits Norepinephrine (Stability calms arousal)
inline constexpr float COUPLING_M21 = -0.06f;
/// λ_S: Serotonin homeostatic decay rate back to equilibrium (0.5).
inline constexpr float COUPLING_LAMBDA_S = 0.15f;
/// λ_N: Norepinephrine homeostatic decay rate back to equilibrium (0.5).
inline constexpr float COUPLING_LAMBDA_N = 0.15f;
/// Equilibrium baseline for Serotonin and Norepinephrine [0, 1].
inline constexpr float COUPLING_EQ = 0.5f;

// ── Phase 51: Failure Mode Guard constants ──────────────────────────────────────

/// §9.1 Anhedonia Trap: D(t) threshold — consecutive cycles below this count as "low dopamine".
inline constexpr float    ANHEDONIA_D_THRESHOLD    = 0.1f;
/// §9.1 Default consecutive low-D cycles before Emergency Stimulus fires (override via config).
inline constexpr uint32_t ANHEDONIA_WINDOW_CYCLES  = 1000u;
/// §9.1 Emergency Stimulus amplitude — synthetic Reward::POSITIVE injected to restart plasticity.
inline constexpr float    EMERGENCY_STIMULUS_VALUE = 0.5f;
/// §9.2 Mania Loop: ring-buffer depth (consecutive goal firings tracked for rate detection).
inline constexpr uint32_t MANIA_GUARD_RING_SIZE    = 3u;
/// §9.2 If MANIA_GUARD_RING_SIZE goals fire within this many ticks → Mania detected.
inline constexpr uint32_t MANIA_DETECT_WINDOW      = 10u;
/// §9.2 Default serotonin suppression duration (seconds) after Mania detected.
inline constexpr float    MANIA_SUPPRESSION_SECS   = 30.0f;
/// §9.2 Serotonin boost amplitude on Mania (spec: "artificially boost Serotonin, simulating a sedative").
inline constexpr float    MANIA_SEROTONIN_BOOST    = 0.4f;

// ── AutonomyConfig ────────────────────────────────────────────────────────────

struct AutonomyConfig {
    float initial_atp          = 1.0f;
    float entropy_sample_dt    = 0.1f;   ///< how often to re-sample entropy (s)
    bool  enable_dream_weave   = true;
    bool  enable_boredom       = true;
    uint32_t entropy_rng_seed  = 42u;

    // Phase 51: Failure Mode Guard tunables
    uint32_t anhedonia_window      = ANHEDONIA_WINDOW_CYCLES; ///< low-D cycles before Emergency Stimulus
    float    anhedonia_d_threshold = ANHEDONIA_D_THRESHOLD;   ///< D below this = anhedonic cycle
    float    mania_suppression_secs = MANIA_SUPPRESSION_SECS; ///< suppression duration (s) after Mania
    uint32_t mania_detect_window   = MANIA_DETECT_WINDOW;     ///< tick window for Mania Loop detection
};

// ── Phase 50: CuriosityGoal ───────────────────────────────────────────────────

/**
 * @brief Structured goal emitted by AutonomyEngine when boredom exceeds θ_explore.
 *
 * Implements spec §8.3 "inject a CuriosityGoal":
 *   If Boredom > 0.8 → pause task queue, inject CuriosityGoal.
 *
 * priority tiers (spec §6.3):
 *   0 = LOW    boredom ∈ [0.80, 0.90)
 *   1 = MEDIUM boredom ∈ [0.90, 0.95)
 *   2 = HIGH   boredom ≥ 0.95
 */
struct CuriosityGoal {
    uint32_t id       = 0;    ///< Monotonic goal ID within this engine instance
    float    boredom  = 0.0f; ///< B(t) at time of generation
    float    entropy  = 0.0f; ///< H(Ψ) at time of generation
    uint8_t  priority = 0;    ///< 0=LOW, 1=MEDIUM, 2=HIGH

    /// Derive priority tier from boredom level.
    [[nodiscard]] static uint8_t tier_from_boredom(float b) noexcept {
        if (b >= 0.95f) return 2;
        if (b >= 0.90f) return 1;
        return 0;
    }
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

    std::function<void()>               on_explore;          ///< legacy bare callback (Phase 5 compat)
    std::function<void(CuriosityGoal)>  on_curiosity_goal;   ///< Phase 50: typed goal payload

    /// Callback fired on nap entry.
    std::function<void()> on_nap_enter;

    /// Callback fired on nap exit.
    std::function<void()> on_nap_exit;

    // ── observers ────────────────────────────────────────────────────────────

    [[nodiscard]] float         atp()              const noexcept;
    [[nodiscard]] float         dopamine()         const noexcept;
    [[nodiscard]] float         serotonin()        const noexcept;  ///< Phase 46 stub: [0,1] default 0.5
    [[nodiscard]] float         norepinephrine()   const noexcept;  ///< Phase 47 stub: [0,1] default 0.5
    [[nodiscard]] float         boredom()          const noexcept;
    [[nodiscard]] float         entropy()    const noexcept;
    [[nodiscard]] AutonomyState state()      const noexcept;
    [[nodiscard]] bool          is_napping()     const noexcept;
    [[nodiscard]] bool          is_exploring()   const noexcept;

    /// Phase 50: true when ATP < NAP_ENTER_THRESHOLD (0.15).
    /// Spec §8.3: "If ATP < 15%, reject all external queries."
    [[nodiscard]] bool          is_query_gated() const noexcept;

    /// Phase 50: total CuriosityGoals emitted since construction.
    [[nodiscard]] uint32_t      curiosity_goal_count() const noexcept;

    /// Phase 50: total ticks where is_query_gated() was true.
    [[nodiscard]] uint32_t      query_gate_count()     const noexcept;

    /// Phase 51 §9.1: true when dopamine is currently below anhedonia threshold (D < θ_anh).
    [[nodiscard]] bool          is_anhedonic()              const noexcept;
    /// Phase 51 §9.2: true while Mania Loop suppression timer is active.
    [[nodiscard]] bool          is_mania_suppressed()        const noexcept;
    /// Phase 51 §9.1: total Emergency Stimulus events fired since construction.
    [[nodiscard]] uint32_t      emergency_stimulus_count()   const noexcept;
    /// Phase 51 §9.2: total Mania Loop suppression events triggered.
    [[nodiscard]] uint32_t      mania_suppress_count()       const noexcept;

    [[nodiscard]] AutonomySnapshot snapshot() const noexcept;

    // ── direct component access (for tests / telemetry) ───────────────────

    [[nodiscard]] const DopamineSystem&     dopamine_system()    const noexcept;
    [[nodiscard]] const BoredomRegulator&   boredom_regulator()  const noexcept;
    [[nodiscard]] GoalSystem&               goal_system()              noexcept;
    [[nodiscard]] const GoalSystem&         goal_system()        const noexcept;
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
    float              serotonin_        = 0.5f;  ///< Phase 46: metric elasticity modulator [0,1]
    float              norepinephrine_   = 0.5f;  ///< Phase 47: arousal / refractive index [0,1]
    EntropyEstimator   entropy_est;
    BoredomRegulator   boredom;
    MetabolicSimulator metabolic;
    NapController      nap;
    DreamWeaveEngine   dream;
    GoalSystem         goal_system;           // Phase 33: GoalSystem
    HamiltonianValue   hamiltonian_value_fn;  // NIK-005

    float    last_entropy  = 0.0f;
    float    entropy_acc   = 0.0f;   // accumulates dt for subset sampling
    uint32_t curiosity_goal_count_ = 0u;  ///< Phase 50 telemetry
    uint32_t query_gate_count_     = 0u;  ///< Phase 50 telemetry
    bool     exploring_active_     = false; ///< Phase 50: cooldown flag (Mania guard)

    /// Phase 50: boredom fraction drained when a CuriosityGoal is emitted.
    /// Prevents immediate re-fire on next tick (early Mania Loop guard, spec §9.2).
    static constexpr float CURIOSITY_BOREDOM_DRAIN = 0.3f;

    // Phase 51 §9.1 — Anhedonia Trap
    uint32_t anhedonia_cycle_           = 0u;   ///< consecutive low-D cycles
    uint32_t emergency_stimulus_count_  = 0u;   ///< total Emergency Stimulus events

    // Phase 51 §9.2 — Mania Loop
    uint32_t tick_count_                = 0u;   ///< monotonic tick counter
    float    mania_suppression_timer_   = 0.0f; ///< remaining suppression (s); 0 = inactive
    uint32_t mania_suppress_count_      = 0u;   ///< total Mania Loop events triggered
    std::array<uint32_t, static_cast<std::size_t>(MANIA_GUARD_RING_SIZE)>
             goal_tick_ring_{}; ///< ring buffer of recent CuriosityGoal tick stamps
    uint8_t  goal_ring_write_           = 0u;   ///< next write slot (0..RING_SIZE-1)
    uint8_t  goal_ring_count_           = 0u;   ///< valid entries in ring (0..RING_SIZE)

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
    // Wire GoalSystem reward callback to DopamineSystem::adjust()
    impl_->goal_system.set_reward_fn([this](float delta, const std::string& /*desc*/) {
        impl_->dopamine.adjust(delta);
    });

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
    ++I.tick_count_;   // Phase 51 §9.2: monotonic tick counter for Mania Loop detection

    // --- 1. Total energy (Σ|Ψ|²) for dopamine — NOTE: stroboscopic (NIK-005)
    //         Use tick_physics() to pass vel spans for the invariant Hamiltonian.
    float total_energy = 0.0f;
    const std::size_t n_nodes = std::min(psi_real.size(), psi_imag.size());
    for (std::size_t i = 0; i < n_nodes; ++i) {
        float r = psi_real[i], im = psi_imag[i];
        total_energy += r*r + im*im;
    }

    // --- 2. Dopamine TD update + decay ---
    I.dopamine.update(total_energy, reward);
    I.dopamine.decay(dt);

    // Phase 51 §9.1: Anhedonia Trap — count consecutive low-D cycles
    if (I.dopamine.level() < I.cfg.anhedonia_d_threshold) {
        ++I.anhedonia_cycle_;
    } else {
        I.anhedonia_cycle_ = 0u;
    }
    if (I.anhedonia_cycle_ >= I.cfg.anhedonia_window) {
        // Emergency Stimulus: synthetic reward injection to jumpstart plasticity engine
        I.dopamine.update(EMERGENCY_STIMULUS_VALUE, Reward::POSITIVE);
        ++I.emergency_stimulus_count_;
        I.anhedonia_cycle_ = 0u;
    }

    // --- Phase 52: GAP-005 Cross-Coupling Matrix (off-diagonal M·N update) ---
    // dN/dt = M·N + F_nl; diagonal handled by existing decay; ATP row handled by metabolic.
    //   dD += (M[0,1]·S + M[0,2]·N) · dt
    //   dS += (M[1,0]·D + M[1,2]·N) · dt  +  diagonal homeostatic decay
    //   dN += (M[2,1]·S)             · dt  +  diagonal homeostatic decay
    {
        const float D = I.dopamine.level();
        const float S = I.serotonin_;
        const float N = I.norepinephrine_;
        // Off-diagonal cross-coupling
        const float dD = (COUPLING_M01 * S + COUPLING_M02 * N) * dt;
        const float dS = (COUPLING_M10 * D + COUPLING_M12 * N) * dt;
        const float dN = (COUPLING_M21 * S)                    * dt;
        // Homeostatic decay: S and N drift back to equilibrium (0.5)
        const float dS_decay = -COUPLING_LAMBDA_S * (S - COUPLING_EQ) * dt;
        const float dN_decay = -COUPLING_LAMBDA_N * (N - COUPLING_EQ) * dt;
        // Apply
        I.dopamine.adjust(dD);
        I.serotonin_      = std::clamp(I.serotonin_      + dS + dS_decay, 0.0f, 1.0f);
        I.norepinephrine_ = std::clamp(I.norepinephrine_ + dN + dN_decay, 0.0f, 1.0f);
    }

    // --- 3. Entropy + boredom (sampled at cfg.entropy_sample_dt rate) ---
    if (I.cfg.enable_boredom && !psi_real.empty()) {
        I.entropy_acc += dt;
        if (I.entropy_acc >= I.cfg.entropy_sample_dt) {
            I.last_entropy = I.entropy_est.estimate(psi_real, psi_imag);
            I.entropy_acc  = 0.0f;
        }
        I.boredom.update(I.last_entropy, dt);

        // Phase 51 §9.2: countdown Mania Loop suppression timer
        if (I.mania_suppression_timer_ > 0.0f) {
            I.mania_suppression_timer_ = std::max(0.0f, I.mania_suppression_timer_ - dt);
        }

        // Phase 50/51: emit CuriosityGoal once per boredom episode.
        //   Phase 51 §9.2: skip goal emission during active Mania Loop suppression.
        if (I.boredom.should_explore() && !I.exploring_active_
                && I.mania_suppression_timer_ <= 0.0f) {
            I.exploring_active_ = true;
            ++I.curiosity_goal_count_;
            CuriosityGoal goal{
                I.curiosity_goal_count_,
                I.boredom.level(),
                I.last_entropy,
                CuriosityGoal::tier_from_boredom(I.boredom.level())
            };
            // Drain boredom to prevent immediate re-fire (Phase 50 early Mania guard)
            I.boredom.drain(AutonomyEngineImpl::CURIOSITY_BOREDOM_DRAIN);
            if (on_curiosity_goal) on_curiosity_goal(goal);
            if (on_explore)        on_explore();   // legacy Phase-5 compat

            // Phase 51 §9.2: record goal tick in ring buffer; detect Mania Loop rate
            I.goal_tick_ring_[I.goal_ring_write_] = I.tick_count_;
            I.goal_ring_write_ = static_cast<uint8_t>(
                    (I.goal_ring_write_ + 1u) % MANIA_GUARD_RING_SIZE);
            if (I.goal_ring_count_ < static_cast<uint8_t>(MANIA_GUARD_RING_SIZE))
                ++I.goal_ring_count_;
            if (I.goal_ring_count_ == static_cast<uint8_t>(MANIA_GUARD_RING_SIZE)) {
                // oldest entry = ring[write_] (circular FIFO; tick_count_ is strictly monotone)
                const uint32_t oldest_tick = I.goal_tick_ring_[I.goal_ring_write_];
                if (I.tick_count_ - oldest_tick < I.cfg.mania_detect_window) {
                    // Mania Loop detected: boost serotonin (sedative) + enter suppression
                    I.mania_suppression_timer_ = I.cfg.mania_suppression_secs;
                    I.serotonin_ = std::min(1.0f, I.serotonin_ + MANIA_SEROTONIN_BOOST);
                    ++I.mania_suppress_count_;
                    I.goal_ring_count_ = 0u;  // clear ring; fresh detection window
                }
            }
        } else if (!I.boredom.should_explore()) {
            I.exploring_active_ = false;  // reset cooldown flag once boredom falls
        }
    }

    // --- 4. Metabolic cost — proxy energy rate from total energy ---
    // Phase 33: GoalSystem autonomous motivation check
    I.goal_system.check_motivation(I.boredom.level(), I.dopamine.level(), I.tick_count_);

    // Phase 50: track query-gate ticks (spec §8.3: ATP < 15% → reject queries)
    if (I.metabolic.atp() < NAP_ENTER_THRESHOLD) ++I.query_gate_count_;
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
    ++I.tick_count_;   // Phase 51 §9.2: monotonic tick counter for Mania Loop detection

    // --- 1. Full Hamiltonian H = γ_K|V|² + γ_P|Ψ|² + γ_NL β/2|Ψ|⁴ (no 2ω flicker)
    const float total_energy = I.hamiltonian_value_fn.compute_spans(
            psi_real, psi_imag, vel_real, vel_imag, beta);

    // --- 2. Dopamine TD update + decay ---
    I.dopamine.update(total_energy, reward);
    I.dopamine.decay(dt);

    // Phase 51 §9.1: Anhedonia Trap — count consecutive low-D cycles
    if (I.dopamine.level() < I.cfg.anhedonia_d_threshold) {
        ++I.anhedonia_cycle_;
    } else {
        I.anhedonia_cycle_ = 0u;
    }
    if (I.anhedonia_cycle_ >= I.cfg.anhedonia_window) {
        // Emergency Stimulus: synthetic reward injection to jumpstart plasticity engine
        I.dopamine.update(EMERGENCY_STIMULUS_VALUE, Reward::POSITIVE);
        ++I.emergency_stimulus_count_;
        I.anhedonia_cycle_ = 0u;
    }

    // --- Phase 52: GAP-005 Cross-Coupling Matrix (off-diagonal M·N update) ---
    {
        const float D = I.dopamine.level();
        const float S = I.serotonin_;
        const float N = I.norepinephrine_;
        const float dD = (COUPLING_M01 * S + COUPLING_M02 * N) * dt;
        const float dS = (COUPLING_M10 * D + COUPLING_M12 * N) * dt;
        const float dN = (COUPLING_M21 * S)                    * dt;
        const float dS_decay = -COUPLING_LAMBDA_S * (S - COUPLING_EQ) * dt;
        const float dN_decay = -COUPLING_LAMBDA_N * (N - COUPLING_EQ) * dt;
        I.dopamine.adjust(dD);
        I.serotonin_      = std::clamp(I.serotonin_      + dS + dS_decay, 0.0f, 1.0f);
        I.norepinephrine_ = std::clamp(I.norepinephrine_ + dN + dN_decay, 0.0f, 1.0f);
    }

    // --- 3. Entropy + boredom ---
    if (I.cfg.enable_boredom && !psi_real.empty()) {
        I.entropy_acc += dt;
        if (I.entropy_acc >= I.cfg.entropy_sample_dt) {
            I.last_entropy = I.entropy_est.estimate(psi_real, psi_imag);
            I.entropy_acc  = 0.0f;
        }
        I.boredom.update(I.last_entropy, dt);

        // Phase 51 §9.2: countdown Mania Loop suppression timer
        if (I.mania_suppression_timer_ > 0.0f) {
            I.mania_suppression_timer_ = std::max(0.0f, I.mania_suppression_timer_ - dt);
        }

        // Phase 50/51: emit CuriosityGoal once per boredom episode.
        //   Phase 51 §9.2: skip goal emission during active Mania Loop suppression.
        if (I.boredom.should_explore() && !I.exploring_active_
                && I.mania_suppression_timer_ <= 0.0f) {
            I.exploring_active_ = true;
            ++I.curiosity_goal_count_;
            CuriosityGoal goal{
                I.curiosity_goal_count_,
                I.boredom.level(),
                I.last_entropy,
                CuriosityGoal::tier_from_boredom(I.boredom.level())
            };
            I.boredom.drain(AutonomyEngineImpl::CURIOSITY_BOREDOM_DRAIN);
            if (on_curiosity_goal) on_curiosity_goal(goal);
            if (on_explore)        on_explore();   // legacy Phase-5 compat

            // Phase 51 §9.2: record goal tick in ring buffer; detect Mania Loop rate
            I.goal_tick_ring_[I.goal_ring_write_] = I.tick_count_;
            I.goal_ring_write_ = static_cast<uint8_t>(
                    (I.goal_ring_write_ + 1u) % MANIA_GUARD_RING_SIZE);
            if (I.goal_ring_count_ < static_cast<uint8_t>(MANIA_GUARD_RING_SIZE))
                ++I.goal_ring_count_;
            if (I.goal_ring_count_ == static_cast<uint8_t>(MANIA_GUARD_RING_SIZE)) {
                const uint32_t oldest_tick = I.goal_tick_ring_[I.goal_ring_write_];
                if (I.tick_count_ - oldest_tick < I.cfg.mania_detect_window) {
                    I.mania_suppression_timer_ = I.cfg.mania_suppression_secs;
                    I.serotonin_ = std::min(1.0f, I.serotonin_ + MANIA_SEROTONIN_BOOST);
                    ++I.mania_suppress_count_;
                    I.goal_ring_count_ = 0u;
                }
            }
        } else if (!I.boredom.should_explore()) {
            I.exploring_active_ = false;
        }
    }

    // --- 4. Metabolic cost ---
    // Phase 33: GoalSystem autonomous motivation check
    I.goal_system.check_motivation(I.boredom.level(), I.dopamine.level(), I.tick_count_);

    // Phase 50: track query-gate ticks (spec §8.3: ATP < 15% → reject queries)
    if (I.metabolic.atp() < NAP_ENTER_THRESHOLD) ++I.query_gate_count_;
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

float         AutonomyEngine::atp()             const noexcept { return impl_->metabolic.atp(); }
float         AutonomyEngine::dopamine()        const noexcept { return impl_->dopamine.level(); }
float         AutonomyEngine::serotonin()       const noexcept { return impl_->serotonin_; }
float         AutonomyEngine::norepinephrine()  const noexcept { return impl_->norepinephrine_; }
float         AutonomyEngine::boredom()         const noexcept { return impl_->boredom.level(); }
float         AutonomyEngine::entropy()   const noexcept { return impl_->last_entropy; }

AutonomyState AutonomyEngine::state() const noexcept {
    if (impl_->nap.is_napping())        return AutonomyState::NAPPING;
    if (impl_->boredom.should_explore()) return AutonomyState::EXPLORING;
    return AutonomyState::ACTIVE;
}

bool AutonomyEngine::is_napping()     const noexcept { return impl_->nap.is_napping(); }
bool AutonomyEngine::is_exploring()   const noexcept { return impl_->boredom.should_explore(); }
bool AutonomyEngine::is_query_gated() const noexcept { return impl_->metabolic.atp() < NAP_ENTER_THRESHOLD; }

uint32_t AutonomyEngine::curiosity_goal_count() const noexcept { return impl_->curiosity_goal_count_; }
uint32_t AutonomyEngine::query_gate_count()     const noexcept { return impl_->query_gate_count_; }

bool     AutonomyEngine::is_anhedonic()             const noexcept { return impl_->dopamine.level() < impl_->cfg.anhedonia_d_threshold; }
bool     AutonomyEngine::is_mania_suppressed()      const noexcept { return impl_->mania_suppression_timer_ > 0.0f; }
uint32_t AutonomyEngine::emergency_stimulus_count() const noexcept { return impl_->emergency_stimulus_count_; }
uint32_t AutonomyEngine::mania_suppress_count()     const noexcept { return impl_->mania_suppress_count_; }

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
GoalSystem&               AutonomyEngine::goal_system()             noexcept { return impl_->goal_system; }
const GoalSystem&         AutonomyEngine::goal_system()       const noexcept { return impl_->goal_system; }
const MetabolicSimulator& AutonomyEngine::metabolic()         const noexcept { return impl_->metabolic; }
const NapController&      AutonomyEngine::nap_controller()    const noexcept { return impl_->nap; }
const DreamWeaveEngine&   AutonomyEngine::dream_weave()       const noexcept { return impl_->dream; }

} // namespace nikola::autonomy

#endif // NIKOLA_AUTONOMY_ENGINE_IMPL
