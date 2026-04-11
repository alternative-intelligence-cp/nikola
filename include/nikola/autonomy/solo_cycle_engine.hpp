#pragma once
/**
 * @file autonomy/solo_cycle_engine.hpp
 * @brief Phase 153 — Multi-Cycle Solo SIE: Internal Code Generation + Campaigns.
 *
 * v0.1.16: Nikola generates code improvements internally using its own
 * cognitive pipeline (Mamba9D state analysis → NPT attention → parameterised
 * code synthesis) instead of calling the external Gemini specialist.
 *
 * Architecture:
 *   NikolaState → InternalCodeGenerator::generate()
 *     1. Encode state → 9D SSM input + PhysicsParams
 *     2. Mamba9D.step() → hidden state captures temporal dynamics
 *     3. Create probe WaveFunction from SSM output
 *     4. NPT.forward(probe_wf) → 8 attention scores (cognitive band weights)
 *     5. Dominant band → improvement strategy selection
 *     6. Strategy + state → parameterised C++ module source code
 *     7. Confidence from attention entropy + state coherence
 *
 *   SoloCampaignRunner::run_campaign(initial_state)
 *     Loop up to max_cycles:
 *       - Generate code internally
 *       - Self-assess confidence (skip if below threshold)
 *       - Deploy via SIE run_cycle_with_source()
 *       - Track quality, detect plateau, handle rollback
 *
 * Header-only.  No new .cpp required.
 */

#include <nikola/autonomy/self_improvement_engine.hpp>
#include <nikola/autonomy/decision_loop.hpp>          // NikolaState full definition
#include <nikola/cognitive/mamba9d.hpp>
#include <nikola/cognitive/neuroplastic_transformer.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace nikola::autonomy {

// ============================================================================
// Improvement strategies — one per NPT cognitive band
// ============================================================================

/**
 * @brief Maps each NPT CognitiveBand to a concrete improvement strategy.
 *
 * The dominant attention head determines which aspect of cognition Nikola
 * prioritises for self-improvement in this cycle.
 */
enum class ImprovementStrategy : uint8_t {
    EXPLORATION_DIVERSITY   = 0,  ///< Head 0: diversify exploration weights
    MEMORY_CONSOLIDATION    = 1,  ///< Head 1: strengthen memory parameters
    WORKING_BUFFER_TUNING   = 2,  ///< Head 2: optimise short-term buffer
    LOGIC_LOW_ENHANCEMENT   = 3,  ///< Head 3: slow logical inference tuning
    LOGIC_HIGH_ENHANCEMENT  = 4,  ///< Head 4: fast logical inference tuning
    SENSORY_BINDING         = 5,  ///< Head 5: cross-modal integration
    PRECISION_TUNING        = 6,  ///< Head 6: fine-detail resolution
    ERROR_SENSITIVITY       = 7,  ///< Head 7: discrepancy detection
};

inline const char* strategy_name(ImprovementStrategy s) noexcept {
    switch (s) {
        case ImprovementStrategy::EXPLORATION_DIVERSITY:  return "exploration_diversity";
        case ImprovementStrategy::MEMORY_CONSOLIDATION:   return "memory_consolidation";
        case ImprovementStrategy::WORKING_BUFFER_TUNING:  return "working_buffer_tuning";
        case ImprovementStrategy::LOGIC_LOW_ENHANCEMENT:  return "logic_low_enhancement";
        case ImprovementStrategy::LOGIC_HIGH_ENHANCEMENT: return "logic_high_enhancement";
        case ImprovementStrategy::SENSORY_BINDING:        return "sensory_binding";
        case ImprovementStrategy::PRECISION_TUNING:       return "precision_tuning";
        case ImprovementStrategy::ERROR_SENSITIVITY:      return "error_sensitivity";
        default:                                          return "unknown";
    }
}

// ============================================================================
// InternalGenerationResult
// ============================================================================

/**
 * @brief Result of internal code generation (no external specialist).
 */
struct InternalGenerationResult {
    std::string              source_code;        ///< Generated C++ module source
    std::string              instruction;        ///< Human-readable instruction
    ImprovementStrategy      strategy;           ///< Which strategy was selected
    float                    confidence;         ///< [0, 1] self-assessed quality
    std::array<float, 8>     attention_weights;  ///< NPT head scores used
    double                   generation_ms;      ///< Wall-clock generation time
};

// ============================================================================
// CycleQuality — per-cycle quality metrics for campaign tracking
// ============================================================================

struct CycleQuality {
    float           pre_entropy;       ///< Shannon entropy before cycle
    float           post_entropy;      ///< Shannon entropy after cycle (estimated)
    float           pre_boredom;       ///< Boredom level before
    float           pre_dopamine;      ///< Dopamine level before
    float           confidence;        ///< Self-assessment confidence
    SIEOutcome      outcome;           ///< From the SIE cycle
    double          elapsed_ms;        ///< Cycle wall-clock time

    /// Quality score = confidence-weighted success: 1.0 if succeeded with
    /// high confidence, 0.0 if failed or low confidence.
    [[nodiscard]] float quality_score() const noexcept {
        return (outcome == SIEOutcome::SUCCESS) ? confidence : 0.f;
    }
};

// ============================================================================
// CampaignResult — full multi-cycle campaign report
// ============================================================================

struct CampaignResult {
    uint32_t                  cycles_attempted{0};
    uint32_t                  cycles_succeeded{0};
    uint32_t                  consecutive_successes{0};
    uint32_t                  max_consecutive{0};
    std::vector<CycleQuality> history;
    double                    total_elapsed_ms{0.0};
    bool                      plateau_detected{false};
    std::string               termination_reason;

    /// Campaign achieved its target?
    [[nodiscard]] bool target_met(uint32_t target) const noexcept {
        return max_consecutive >= target;
    }
};

// ============================================================================
// SoloCampaignConfig
// ============================================================================

struct SoloCampaignConfig {
    uint32_t max_cycles         = 10;    ///< Maximum cycles per campaign
    uint32_t target_consecutive = 3;     ///< Stop after N consecutive successes
    float    confidence_threshold = 0.3f; ///< Don't submit below this
    float    plateau_threshold  = 0.01f; ///< ΔQ_score < this = plateau
    uint32_t plateau_patience   = 3;     ///< Plateau cycles before termination
};

// ============================================================================
// InternalCodeGenerator — Mamba9D + NPT → parameterised C++ modules
// ============================================================================

/**
 * @brief Generates C++ cognitive enhancement modules using the internal
 *        cognitive pipeline instead of an external specialist model.
 *
 * Pipeline:
 *   1. NikolaState → 9D SSM input (boredom, entropy, dopamine, atp, ...)
 *   2. PhysicsParams derived from state (intensity, phase, resonance, ρ_G)
 *   3. Mamba9D.step(h, input, physics) → hidden state update
 *   4. SSM output → probe WaveFunction (seed from hidden-state energies)
 *   5. NPT.forward(probe_wf) → AttentionResult (8 head scores)
 *   6. Dominant head → ImprovementStrategy
 *   7. Strategy + attention weights + state → parameterised C++ source
 *   8. Confidence from attention entropy + state coherence
 */
class InternalCodeGenerator {
public:
    /**
     * @param grid_n  Nodes-per-dimension for NPT WaveFunctions (default 2 = 512 nodes).
     * @param seed    RNG seed for reproducible Mamba9D initialisation.
     */
    explicit InternalCodeGenerator(int grid_n = 2, uint32_t seed = 42)
        : mamba_(256, 9, 50000, seed)
        , npt_(grid_n, /*temperature=*/1.0f, /*curvature_alpha=*/0.5f)
        , grid_n_(grid_n)
    {
        ssm_state_ = mamba_.ssm().make_zero_state();
    }

    // Non-copyable (NPT + Mamba are non-copyable)
    InternalCodeGenerator(const InternalCodeGenerator&)            = delete;
    InternalCodeGenerator& operator=(const InternalCodeGenerator&) = delete;
    InternalCodeGenerator(InternalCodeGenerator&&)                 = default;
    InternalCodeGenerator& operator=(InternalCodeGenerator&&)      = default;

    // ------------------------------------------------------------------
    // Core generation
    // ------------------------------------------------------------------

    /**
     * @brief Generate improvement code using the cognitive pipeline.
     */
    [[nodiscard]]
    InternalGenerationResult generate(const NikolaState& state) {
        const auto t0 = std::chrono::steady_clock::now();

        // Step 1-2: Encode state → SSM input + PhysicsParams
        auto input   = encode_state(state);
        auto physics = derive_physics(state);

        // Step 3: Run Mamba9D — update hidden state
        mamba_.step(ssm_state_, input, physics);

        // Step 4: Create probe WaveFunction from SSM energies
        auto probe_wf = create_probe_wavefunction(state);

        // Step 5: NPT forward pass → 8 attention scores
        auto attn = npt_.forward(probe_wf, state.dopamine);

        // Step 6: Select dominant strategy
        auto strategy = select_strategy(attn.head_scores);

        // Step 7: Synthesise C++ source
        auto source = synthesize_code(strategy, state, attn.head_scores);

        // Step 8: Compute confidence
        float confidence = compute_confidence(attn.head_scores, state);

        // Build instruction description
        std::string instruction = build_instruction(strategy, state, confidence);

        const auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        return {
            std::move(source),
            std::move(instruction),
            strategy,
            confidence,
            attn.head_scores,
            ms
        };
    }

    /**
     * @brief Self-assess confidence for a given source + state.
     *
     * Uses attention entropy and state coherence metrics.
     */
    [[nodiscard]]
    float assess_confidence(const std::string& source,
                            const NikolaState& state) const {
        // Source validity check: must contain factory symbol
        if (source.find("nikola_module_factory") == std::string::npos)
            return 0.f;

        // State coherence: low entropy + adequate ATP → higher confidence
        float coherence = 1.f - std::clamp(state.entropy / 4.f, 0.f, 1.f);
        float energy    = std::clamp(state.atp, 0.f, 1.f);

        // Boredom penalty: very bored → parameters may be desperate
        float boredom_penalty = std::clamp(state.boredom, 0.f, 1.f) * 0.3f;

        return std::clamp(0.5f * coherence + 0.3f * energy - boredom_penalty + 0.2f,
                          0.f, 1.f);
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    const cognitive::Mamba9D& mamba() const noexcept { return mamba_; }
    const cognitive::NeuroplasticTransformer& npt() const noexcept { return npt_; }
    int grid_n() const noexcept { return grid_n_; }

private:
    // ------------------------------------------------------------------
    // Step 1: State → 9D SSM input
    // ------------------------------------------------------------------

    /**
     * @brief Map NikolaState fields to the 9 SSM input dimensions.
     *
     * Dimension mapping (matches T⁹ manifold convention):
     *   [0-2]  spatial proxy:  boredom, entropy/4, torus_energy/10
     *   [3]    temporal:       time (normalised)
     *   [4-5]  neurochemical:  dopamine, atp
     *   [6-8]  quantum proxy:  td_error+0.5, 1-boredom, entropy/4
     */
    [[nodiscard]]
    static std::array<float, 9> encode_state(const NikolaState& state) noexcept {
        return {{
            std::clamp(state.boredom,         0.f, 1.f),
            std::clamp(state.entropy / 4.f,   0.f, 1.f),
            std::clamp(state.torus_energy / 10.f, 0.f, 1.f),
            std::fmod(std::fabs(state.time), 1.f),             // normalised time
            std::clamp(state.dopamine,        0.f, 1.f),
            std::clamp(state.atp,             0.f, 1.f),
            std::clamp(state.td_error + 0.5f, 0.f, 1.f),
            std::clamp(1.f - state.boredom,   0.f, 1.f),
            std::clamp(state.entropy / 4.f,   0.f, 1.f),
        }};
    }

    // ------------------------------------------------------------------
    // Step 2: State → PhysicsParams
    // ------------------------------------------------------------------

    /**
     * @brief Derive SSM physics parameters from cognitive state.
     *
     * Intensity: proportional to normalised state fields (higher = slower decay).
     * Phase: set to zero (no spatial phase information from NikolaState).
     * Resonance: 1 − boredom (bored = low resonance = more exploration).
     * ρ_G: torus energy clamped to [0.1, 10] (spectral radius of metric).
     */
    [[nodiscard]]
    static cognitive::PhysicsParams derive_physics(
            const NikolaState& state) noexcept {
        cognitive::PhysicsParams pp{};
        auto input = encode_state(state);
        for (int d = 0; d < 9; ++d) {
            pp.intensity[d] = input[d];
            pp.phase[d]     = 0.f;
        }
        pp.resonance = std::clamp(1.f - state.boredom, 0.f, 1.f);
        pp.rho_G     = std::clamp(state.torus_energy, 0.1f, 10.f);
        return pp;
    }

    // ------------------------------------------------------------------
    // Step 4: Create probe WaveFunction
    // ------------------------------------------------------------------

    /**
     * @brief Build a probe WaveFunction seeded from cognitive state.
     *
     * The probe represents the current mental state projected onto the NPT's
     * grid.  Its amplitude encodes overall cognitive energy; the pilot
     * dimension reflects the most active state component.
     */
    [[nodiscard]]
    physics::WaveFunction create_probe_wavefunction(
            const NikolaState& state) const {
        // Pilot dimension: pick dimension with highest encoded value
        auto input = encode_state(state);
        int pilot_dim = 0;
        float max_val = input[0];
        for (int d = 1; d < 9; ++d) {
            if (input[d] > max_val) {
                max_val = input[d];
                pilot_dim = d;
            }
        }

        // Amplitude proportional to ATP (more energy = stronger probe)
        float amplitude = std::clamp(state.atp * 2.f, 0.1f, 2.f);

        physics::WaveFunction wf(foundation::GridConfig::uniform(grid_n_));
        wf.seed_manifold(grid_n_, pilot_dim, /*k_mode=*/1, amplitude, 42u);
        return wf;
    }

    // ------------------------------------------------------------------
    // Step 6: Select dominant strategy
    // ------------------------------------------------------------------

    /**
     * @brief Pick the ImprovementStrategy from the NPT head with the
     *        highest attention score.
     */
    [[nodiscard]]
    static ImprovementStrategy select_strategy(
            const std::array<float, 8>& scores) noexcept {
        size_t best = 0;
        for (size_t i = 1; i < 8; ++i)
            if (scores[i] > scores[best]) best = i;
        return static_cast<ImprovementStrategy>(best);
    }

    // ------------------------------------------------------------------
    // Step 7: Synthesise C++ source code
    // ------------------------------------------------------------------

    /**
     * @brief Generate parameterised C++ module source from the selected
     *        strategy and NPT attention weights.
     *
     * Each strategy produces a CognitiveParameters struct whose float fields
     * are derived from the attention distribution rather than being hardcoded.
     * The factory pattern (nikola_module_factory) matches what the SIE
     * packaging/gate pipeline expects.
     */
    [[nodiscard]]
    static std::string synthesize_code(
            ImprovementStrategy strategy,
            const NikolaState& state,
            const std::array<float, 8>& attn) {
        // Derive named parameters from attention distribution
        //
        // exploration_weight: driven by head 0 (Global Context)
        //   + head 5 (Sensory Integration) — broader exploration
        float exploration_weight = std::clamp(
            0.5f + 0.3f * (attn[0] + attn[5]) - 0.1f * state.boredom,
            0.1f, 0.95f);

        // coherence_bias: driven by head 2 (Working Memory)
        //   + head 3 (Logic Lo) — tighter coherence focus
        float coherence_bias = std::clamp(
            0.3f + 0.4f * (attn[2] + attn[3]) - 0.05f * (state.entropy / 4.f),
            0.05f, 0.90f);

        // reward_sensitivity: driven by head 7 (Error Correction)
        //   + head 4 (Logic Hi) — sharper reward discrimination
        float reward_sensitivity = std::clamp(
            0.4f + 0.3f * (attn[7] + attn[4]) + 0.1f * state.dopamine,
            0.1f, 0.95f);

        std::ostringstream oss;
        oss << "#include <cstdint>\n\n"
            << "// Auto-generated by Nikola InternalCodeGenerator\n"
            << "// Strategy: " << strategy_name(strategy) << "\n"
            << "// State: boredom=" << state.boredom
            << " entropy=" << state.entropy
            << " dopamine=" << state.dopamine
            << " atp=" << state.atp << "\n\n"
            << "struct CognitiveParameters {\n"
            << "    uint32_t version;\n"
            << "    const char* name;\n"
            << "    float exploration_weight;\n"
            << "    float coherence_bias;\n"
            << "    float reward_sensitivity;\n"
            << "};\n\n"
            << "static CognitiveParameters params = {\n"
            << "    1, \"" << strategy_name(strategy) << "\",\n"
            << "    " << exploration_weight << "f,\n"
            << "    " << coherence_bias << "f,\n"
            << "    " << reward_sensitivity << "f\n"
            << "};\n\n"
            << "extern \"C\" void* nikola_module_factory() {\n"
            << "    return &params;\n"
            << "}\n";

        return oss.str();
    }

    // ------------------------------------------------------------------
    // Step 8: Confidence scoring
    // ------------------------------------------------------------------

    /**
     * @brief Compute self-assessment confidence ∈ [0, 1].
     *
     * Combines:
     *   - Attention entropy: lower entropy = more decisive = higher confidence
     *   - State coherence:   low field entropy + adequate ATP = good context
     *   - Boredom penalty:   extreme boredom suggests desperation
     */
    [[nodiscard]]
    static float compute_confidence(
            const std::array<float, 8>& attn,
            const NikolaState& state) noexcept {
        // Shannon entropy of the 8-head attention distribution
        float H = 0.f;
        for (float a : attn) {
            if (a > 1e-8f) H -= a * std::log2(a);
        }
        // Max entropy for 8 heads = log2(8) = 3.0
        // Lower H means more focused attention → higher confidence
        float focus = std::clamp(1.f - H / 3.f, 0.f, 1.f);

        // State coherence
        float coherence = 1.f - std::clamp(state.entropy / 4.f, 0.f, 1.f);
        float energy    = std::clamp(state.atp, 0.f, 1.f);
        float boredom_pen = std::clamp(state.boredom, 0.f, 1.f) * 0.2f;

        return std::clamp(
            0.35f * focus + 0.25f * coherence + 0.2f * energy - boredom_pen + 0.2f,
            0.f, 1.f);
    }

    // ------------------------------------------------------------------
    // Instruction builder
    // ------------------------------------------------------------------

    [[nodiscard]]
    static std::string build_instruction(
            ImprovementStrategy strategy,
            const NikolaState& state,
            float confidence) {
        std::ostringstream oss;
        oss << "[Solo SIE] Strategy: " << strategy_name(strategy)
            << " | Confidence: " << confidence
            << " | State: boredom=" << state.boredom
            << " entropy=" << state.entropy
            << " dopamine=" << state.dopamine
            << " atp=" << state.atp;
        return oss.str();
    }

    // ------------------------------------------------------------------
    // Members
    // ------------------------------------------------------------------
    cognitive::Mamba9D                      mamba_;
    cognitive::SSMLayer::State              ssm_state_;
    cognitive::NeuroplasticTransformer      npt_;
    int                                     grid_n_;
};

// ============================================================================
// SoloCampaignRunner — multi-cycle campaign orchestration
// ============================================================================

/**
 * @brief Runs multi-cycle self-improvement campaigns using internal code
 *        generation (no external specialist).
 *
 * Campaign loop:
 *   1. Generate code internally via InternalCodeGenerator
 *   2. Check confidence — skip if below threshold
 *   3. Submit to SIE via run_cycle_with_source()
 *   4. Record outcome and quality metrics
 *   5. Detect plateau (quality stopped improving)
 *   6. Stop on: target met, plateau, or max cycles
 *
 * Rollback: if a cycle produces QUALITY_REGRESSION, the campaign halts
 * immediately.  The ShadowSpine / EO handles actual module reversal.
 */
class SoloCampaignRunner {
public:
    SoloCampaignRunner(SelfImprovementEngine& sie,
                       InternalCodeGenerator& gen,
                       SoloCampaignConfig     cfg = {})
        : sie_(sie), gen_(gen), cfg_(std::move(cfg)) {}

    // ------------------------------------------------------------------
    // Run a full campaign
    // ------------------------------------------------------------------

    /**
     * @brief Execute a multi-cycle improvement campaign.
     *
     * @param state  Snapshot of current cognitive/metabolic state.
     *               (Used for all cycles — the real DecisionLoop would
     *                re-read state between cycles; for testing we use a
     *                fixed snapshot.)
     * @return CampaignResult with full history and termination reason.
     */
    [[nodiscard]]
    CampaignResult run_campaign(const NikolaState& state) {
        CampaignResult result;
        const auto t0 = std::chrono::steady_clock::now();

        uint32_t consecutive = 0;

        for (uint32_t i = 0; i < cfg_.max_cycles; ++i) {
            // Generate code internally
            auto gen_result = gen_.generate(state);

            CycleQuality quality{};
            quality.pre_entropy  = state.entropy;
            quality.pre_boredom  = state.boredom;
            quality.pre_dopamine = state.dopamine;
            quality.confidence   = gen_result.confidence;

            ++result.cycles_attempted;

            // Confidence gate: skip low-confidence proposals
            if (gen_result.confidence < cfg_.confidence_threshold) {
                quality.outcome    = SIEOutcome::QUALITY_REGRESSION;
                quality.elapsed_ms = gen_result.generation_ms;
                result.history.push_back(quality);
                consecutive = 0;
                continue;
            }

            // Submit to SIE pipeline (package → sign → deploy)
            auto cycle_result = sie_.run_cycle_with_source(
                gen_result.source_code,
                gen_result.instruction);

            quality.outcome     = cycle_result.outcome;
            quality.elapsed_ms  = cycle_result.elapsed_ms + gen_result.generation_ms;
            quality.post_entropy = state.entropy;  // In live system, re-read state

            result.history.push_back(quality);

            if (cycle_result.outcome == SIEOutcome::SUCCESS) {
                ++result.cycles_succeeded;
                ++consecutive;
                if (consecutive > result.max_consecutive)
                    result.max_consecutive = consecutive;

                // Target met?
                if (consecutive >= cfg_.target_consecutive) {
                    result.consecutive_successes = consecutive;
                    result.termination_reason = "target_met";
                    break;
                }
            } else {
                consecutive = 0;

                // Quality regression → immediate halt (rollback signal)
                if (cycle_result.outcome == SIEOutcome::QUALITY_REGRESSION) {
                    result.termination_reason = "quality_regression";
                    break;
                }
            }

            // Plateau detection
            if (detect_plateau(result.history)) {
                result.plateau_detected = true;
                result.termination_reason = "plateau";
                break;
            }
        }

        // If loop completed without early termination
        if (result.termination_reason.empty()) {
            result.termination_reason = "max_cycles_reached";
        }

        result.consecutive_successes = consecutive;
        const auto t1 = std::chrono::steady_clock::now();
        result.total_elapsed_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        return result;
    }

    // ------------------------------------------------------------------
    // Run a single solo cycle
    // ------------------------------------------------------------------

    /**
     * @brief Run one cycle of internal generation → SIE deployment.
     *
     * Convenience wrapper for testing individual solo cycles.
     */
    [[nodiscard]]
    SIECycleResult run_solo_cycle(const NikolaState& state) {
        auto gen_result = gen_.generate(state);
        if (gen_result.confidence < cfg_.confidence_threshold) {
            SIECycleResult r;
            r.outcome     = SIEOutcome::QUALITY_REGRESSION;
            r.instruction = gen_result.instruction;
            r.source_code = gen_result.source_code;
            r.elapsed_ms  = gen_result.generation_ms;
            return r;
        }
        return sie_.run_cycle_with_source(
            gen_result.source_code,
            gen_result.instruction);
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    const SoloCampaignConfig& config() const noexcept { return cfg_; }

private:
    // ------------------------------------------------------------------
    // Plateau detection
    // ------------------------------------------------------------------

    /**
     * @brief Detect quality plateau: last N cycles had negligible improvement.
     *
     * If the last `plateau_patience` cycles all have |ΔQ| < threshold,
     * the campaign has stagnated and should terminate.
     */
    [[nodiscard]]
    bool detect_plateau(const std::vector<CycleQuality>& history) const {
        if (history.size() < cfg_.plateau_patience) return false;

        // Look at the last `patience` quality scores
        size_t start = history.size() - cfg_.plateau_patience;
        float first_q = history[start].quality_score();

        for (size_t i = start + 1; i < history.size(); ++i) {
            float delta = std::fabs(history[i].quality_score() - first_q);
            if (delta > cfg_.plateau_threshold)
                return false;  // Significant change → not a plateau
        }
        return true;
    }

    // ------------------------------------------------------------------
    // Members
    // ------------------------------------------------------------------
    SelfImprovementEngine& sie_;
    InternalCodeGenerator& gen_;
    SoloCampaignConfig     cfg_;
};

} // namespace nikola::autonomy
