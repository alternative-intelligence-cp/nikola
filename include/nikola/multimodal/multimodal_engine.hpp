/**
 * @file multimodal/multimodal_engine.hpp
 * @brief Phase 6 integration facade: combines all multimodal + persistence components.
 *
 * MultimodalEngine owns and drives:
 *   - AudioEmitterLayout   (Gap 6.1 — 8-emitter circular audio array)
 *   - LogPolarTransform    (Gap 6.2 — OpenCV-free visual transduction)
 *   - CheckpointManager    (Gap 6.3 — periodic + event-driven persistence)
 *   - GGUFExporter         (Gap 6.4 — GGUF-compatible binary export)
 *   - AdaptiveQuantizer    (Gap 6.5 — Q9_0/FP16 mixed precision)
 *
 * Designed for header-only use (no separate .cpp) behind a Pimpl guard
 * for ABI stability — same pattern as AutonomyEngine and Orchestrator.
 *
 * Public API (no heavy includes):
 *   engine.tick_audio(pcm, time_idx)        → emitter coordinates
 *   engine.tick_visual(image, w, h, t)      → injection coordinates
 *   engine.checkpoint_if_needed(napping)    → bool
 *   engine.export_gguf(path, real, imag)    → void
 *   engine.compress(real, imag)             → Q9Block vector
 *
 * Impl (behind NIKOLA_MULTIMODAL_ENGINE_IMPL):
 *   Full implementation includes all 5 gap headers.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

// All multimodal gap headers are lightweight (stdlib only) — include upfront
#include <nikola/multimodal/audio_emitter.hpp>
#include <nikola/multimodal/log_polar_transform.hpp>
#include <nikola/multimodal/checkpoint_manager.hpp>
#include <nikola/multimodal/gguf_exporter.hpp>
#include <nikola/multimodal/adaptive_quantizer.hpp>

namespace nikola::multimodal {

// ============================================================================
// Config
// ============================================================================

struct MultimodalConfig {
    std::string checkpoint_dir   = "/var/lib/nikola/checkpoints";
    int         grid_nx          = 64;
    int         grid_ny          = 64;
    int         grid_nr          = 16;
    int         grid_ns          = 16;
    int         grid_nt          = 128;
    bool        enable_audio     = true;
    bool        enable_visual    = true;
    bool        enable_checkpoints = true;
};

// ============================================================================
// Snapshot (telemetry)
// ============================================================================

struct MultimodalSnapshot {
    uint64_t audio_ticks{0};
    uint64_t visual_ticks{0};
    uint64_t checkpoint_count{0};
    uint64_t gguf_exports{0};
    size_t   last_injection_count{0};
    float    last_compression_ratio{1.0f};
};

// ============================================================================
// Forward declaration for Pimpl
// ============================================================================

struct MultimodalEngineImpl;

// ============================================================================
// MultimodalEngine
// ============================================================================

/**
 * Phase 6 facade for sensory transduction + persistence.
 *
 * All heavy state lives in the Pimpl struct (compiled only when
 * NIKOLA_MULTIMODAL_ENGINE_IMPL is defined).
 */
class MultimodalEngine {
public:
    using EmitterArray    = std::array<EmitterPosition, NUM_EMITTERS>;
    using InjectionList   = LogPolarTransform::InjectionList;
    using CompressedPsi   = std::vector<Q9Block>;

    explicit MultimodalEngine(MultimodalConfig cfg = {});
    ~MultimodalEngine();

    // Non-copyable, movable
    MultimodalEngine(const MultimodalEngine&)            = delete;
    MultimodalEngine& operator=(const MultimodalEngine&) = delete;
    MultimodalEngine(MultimodalEngine&&)                 noexcept;
    MultimodalEngine& operator=(MultimodalEngine&&)      noexcept;

    // ── Sensory ──────────────────────────────────────────────────────────────

    /**
     * Compute all 8 audio emitter positions for the given time index.
     * PCM samples currently unused (placeholder for future spectral analysis).
     */
    EmitterArray tick_audio(std::span<const float> pcm_samples, int time_index);

    /**
     * Run log-polar transform on a grayscale float image and return injection coords.
     */
    InjectionList tick_visual(std::span<const float> image,
                               int width, int height, int time_index);

    // ── Persistence ──────────────────────────────────────────────────────────

    /**
     * Trigger a checkpoint if any condition is met.
     * @param is_napping  Current NAP state (from AutonomyEngine)
     * @return True if a checkpoint was written
     */
    bool checkpoint_if_needed(bool is_napping);

    /** Force an immediate checkpoint with the given reason. */
    void force_checkpoint(CheckpointReason reason = CheckpointReason::PERIODIC);

    /**
     * Export wavefunction to GGUF format.
     * @param filename  Output path
     * @param psi_real  Real part of Ψ
     * @param psi_imag  Imaginary part of Ψ
     * @param meta      Override default metadata if needed
     */
    void export_gguf(const std::string&     filename,
                     std::span<const float> psi_real = {},
                     std::span<const float> psi_imag = {},
                     const NikolaGGUFMeta&  meta     = {});

    // ── Compression ──────────────────────────────────────────────────────────

    /** Compress wavefunction using adaptive Q9_0/FP16 quantizer. */
    CompressedPsi compress_psi(std::span<const float> psi_real,
                                std::span<const float> psi_imag);

    /** Decompress back to float arrays. */
    void decompress_psi(const CompressedPsi& blocks,
                         std::vector<float>&  psi_real_out,
                         std::vector<float>&  psi_imag_out);

    // ── Telemetry ─────────────────────────────────────────────────────────────

    MultimodalSnapshot snapshot() const;

private:
    std::unique_ptr<MultimodalEngineImpl> impl_;
};

// ============================================================================
// Implementation (compiled only with NIKOLA_MULTIMODAL_ENGINE_IMPL)
// ============================================================================

#ifdef NIKOLA_MULTIMODAL_ENGINE_IMPL

struct MultimodalEngineImpl {
    MultimodalConfig         cfg;
    AudioEmitterLayout       audio_layout;
    CheckpointManager        checkpoint_mgr;
    MultimodalSnapshot       snap;

    explicit MultimodalEngineImpl(MultimodalConfig c)
        : cfg(std::move(c))
        , checkpoint_mgr(cfg.checkpoint_dir)
    {}
};

// ── Constructor / Destructor ─────────────────────────────────────────────────

MultimodalEngine::MultimodalEngine(MultimodalConfig cfg)
    : impl_(std::make_unique<MultimodalEngineImpl>(std::move(cfg)))
{}

MultimodalEngine::~MultimodalEngine() = default;

MultimodalEngine::MultimodalEngine(MultimodalEngine&&) noexcept = default;
MultimodalEngine& MultimodalEngine::operator=(MultimodalEngine&&) noexcept = default;

// ── Sensory ──────────────────────────────────────────────────────────────────

MultimodalEngine::EmitterArray
MultimodalEngine::tick_audio(std::span<const float> /*pcm_samples*/, int time_index)
{
    auto& d = *impl_;
    if (!d.cfg.enable_audio) return {};
    ++d.snap.audio_ticks;
    return AudioEmitterLayout::all_positions(
        d.cfg.grid_nx, d.cfg.grid_ny,
        d.cfg.grid_nr, d.cfg.grid_ns,
        d.cfg.grid_nt, time_index);
}

MultimodalEngine::InjectionList
MultimodalEngine::tick_visual(std::span<const float> image,
                               int width, int height, int time_index)
{
    auto& d = *impl_;
    if (!d.cfg.enable_visual) return {};
    ++d.snap.visual_ticks;

    const auto lp = LogPolarTransform::transform(image, width, height);
    auto result   = LogPolarTransform::inject_coords(lp, time_index, d.cfg.grid_nt);
    d.snap.last_injection_count = result.size();
    return result;
}

// ── Persistence ──────────────────────────────────────────────────────────────

bool MultimodalEngine::checkpoint_if_needed(bool is_napping)
{
    auto& d = *impl_;
    if (!d.cfg.enable_checkpoints) return false;
    const bool triggered = d.checkpoint_mgr.update(is_napping);
    if (triggered) ++d.snap.checkpoint_count;
    return triggered;
}

void MultimodalEngine::force_checkpoint(CheckpointReason reason)
{
    auto& d = *impl_;
    d.checkpoint_mgr.force_checkpoint(reason);
    ++d.snap.checkpoint_count;
}

void MultimodalEngine::export_gguf(const std::string&     filename,
                                    std::span<const float> psi_real,
                                    std::span<const float> psi_imag,
                                    const NikolaGGUFMeta&  meta)
{
    GGUFExporter::export_metadata(filename, psi_real, psi_imag, meta);
    ++impl_->snap.gguf_exports;
}

// ── Compression ──────────────────────────────────────────────────────────────

MultimodalEngine::CompressedPsi
MultimodalEngine::compress_psi(std::span<const float> psi_real,
                                std::span<const float> psi_imag)
{
    auto blocks = AdaptiveQuantizer::compress(psi_real, psi_imag);
    impl_->snap.last_compression_ratio = AdaptiveQuantizer::compression_ratio(blocks);
    return blocks;
}

void MultimodalEngine::decompress_psi(const CompressedPsi& blocks,
                                       std::vector<float>&  psi_real_out,
                                       std::vector<float>&  psi_imag_out)
{
    AdaptiveQuantizer::decompress(blocks, psi_real_out, psi_imag_out);
}

// ── Telemetry ─────────────────────────────────────────────────────────────────

MultimodalSnapshot MultimodalEngine::snapshot() const
{
    return impl_->snap;
}

#endif // NIKOLA_MULTIMODAL_ENGINE_IMPL

} // namespace nikola::multimodal
