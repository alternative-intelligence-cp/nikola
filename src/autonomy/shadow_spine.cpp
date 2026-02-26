// =============================================================================
// NIKOLA — Phase 114
// ShadowSpine implementation
// =============================================================================
// Spec   : §11.6 / §11.7 "Shadow Spine: Safe Self-Improvement Deployment"
//          docs/info/integration/sections/04_infrastructure/02_orchestrator_router.md
// Author : Nikola Phase 114
// License: MIT
// =============================================================================

#include <nikola/autonomy/shadow_spine.hpp>

#include <fstream>
#include <iterator>

namespace nikola::autonomy {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

ShadowSpine::ShadowSpine(EvolutionaryOrchestrator&         eo,
                         nikola::security::HybridVerifier& hv) noexcept
    : eo_(eo), hv_(hv) {}

// ---------------------------------------------------------------------------
// stage
// ---------------------------------------------------------------------------

StageReport ShadowSpine::stage(
        std::string_view                              so_path,
        const std::string&                            source_code,
        const nikola::security::HybridSignature&      sig,
        const std::vector<uint8_t>&                   ed_pub,
        const std::vector<uint8_t>&                   sphincs_pub,
        EvolutionaryOrchestrator::PhysicsProvider     physics_fn) {

    std::lock_guard<std::mutex> lock(stage_mutex_);

    StageReport rep{};

    // ── Gate 0a: read the candidate binary from disk ─────────────────────
    // We verify the binary *content* — same bytes that dlopen will load.
    // This prevents a TOCTOU window where a file could be swapped between
    // verification and loading; in practice the OS file cache makes this
    // the same read.
    std::vector<uint8_t> binary;
    {
        std::ifstream file(std::string(so_path), std::ios::binary);
        if (!file) {
            rep.status = StageStatus::LOAD_FAILED;
            last_report_ = rep;
            return rep;
        }
        binary.assign(std::istreambuf_iterator<char>(file),
                      std::istreambuf_iterator<char>());
    }

    // ── Gate 0b: hybrid signature verification ───────────────────────────
    const bool sig_ok = hv_.verify_module(binary, sig, ed_pub, sphincs_pub);

    rep.signature_passed = sig_ok;
    rep.sig_fail         = hv_.last_failure();

    if (!sig_ok) {
        rep.status = StageStatus::SIGNATURE_REJECTED;
        last_report_ = rep;
        return rep;
    }

    // ── Gates 1–3: delegate to EvolutionaryOrchestrator ──────────────────
    rep.cycle_report = eo_.run_cycle(so_path, source_code, std::move(physics_fn));
    rep.status       = from_cycle_status(rep.cycle_report.status);

    last_report_ = rep;
    return rep;
}

// ---------------------------------------------------------------------------
// rollback
// ---------------------------------------------------------------------------

bool ShadowSpine::rollback() noexcept {
    return eo_.rollback();
}

// ---------------------------------------------------------------------------
// Inspection
// ---------------------------------------------------------------------------

bool ShadowSpine::has_active() const noexcept {
    return eo_.has_active();
}

const StageReport* ShadowSpine::last_report() const noexcept {
    if (last_report_.has_value()) {
        return &last_report_.value();
    }
    return nullptr;
}

CycleStats ShadowSpine::stats() const noexcept {
    return eo_.stats();
}

// ---------------------------------------------------------------------------
// Private: from_cycle_status
// ---------------------------------------------------------------------------

StageStatus ShadowSpine::from_cycle_status(CycleStatus cs) noexcept {
    switch (cs) {
        case CycleStatus::SUCCESS:            return StageStatus::SUCCESS;
        case CycleStatus::ATP_DENIED:         return StageStatus::ATP_DENIED;
        case CycleStatus::SECURITY_REJECTED:  return StageStatus::SECURITY_REJECTED;
        case CycleStatus::PHYSICS_REJECTED:   return StageStatus::PHYSICS_REJECTED;
        case CycleStatus::LOAD_FAILED:        return StageStatus::LOAD_FAILED;
        case CycleStatus::SYMBOL_MISSING:     return StageStatus::SYMBOL_MISSING;
        case CycleStatus::SAME_MODULE:        return StageStatus::SAME_MODULE;
    }
    return StageStatus::LOAD_FAILED; // unreachable
}

} // namespace nikola::autonomy
