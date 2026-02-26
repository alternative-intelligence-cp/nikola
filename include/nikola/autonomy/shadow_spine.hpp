// =============================================================================
// NIKOLA — Phase 114
// ShadowSpine — final SIE safety gate: signatures → EO pipeline
// =============================================================================
// Spec   : §11.6 "Shadow Spine: Safe Self-Improvement Deployment"
//          docs/info/integration/sections/04_infrastructure/02_orchestrator_router.md
//          §4.1 "Hybrid Signature Architecture (GAP-047)"
//          docs/info/integration/sections/05_autonomous_systems/04_self_improvement.md
// Author : Nikola Phase 114
// License: MIT
//
// ShadowSpine is the outermost safety gate in the SIE (Self-Improvement Engine)
// deploy path.  It adds hybrid-signature verification as Gate 0 before handing
// control to the EvolutionaryOrchestrator (which runs Gate 1 security scan,
// Gate 2 physics validation, and Gate 3 dlopen hot-swap).
//
// Full SIE pipeline through ShadowSpine:
//
//   [Candidate .so + HybridSignature]
//        │
//   Gate 0: HybridVerifier.verify_module()       ← Phase 114 (this file)
//        │  Ed25519 fast-path + SPHINCS+ slow-path
//        │  Reject → SIGNATURE_REJECTED
//        ↓
//   [EvolutionaryOrchestrator.run_cycle()]        ← Phase 113
//        │
//   Gate 1: CodePatternBlacklist.is_safe()
//        │  Reject → SECURITY_REJECTED
//        ↓
//   Gate 2: PhysicsOracle sandbox
//        │  Reject → PHYSICS_REJECTED
//        ↓
//   Gate 3: ModuleSwapper.swap_in()               ← Phase 112
//        │  dlopen + symbol resolution
//        │  Reject → LOAD_FAILED / SYMBOL_MISSING
//        ↓
//   [MetabolicLock.commit()] → ATP consumed
//        ↓
//   SUCCESS — new module is live
//
// Key Guarantee (from §11.6.2):
//   "User NEVER waits for candidate response.  Production availability is
//    preserved even if candidate code hangs or crashes."
// ShadowSpine honours this by reporting failures through StageReport without
// propagating exceptions to the caller.
//
// Thread-safety: stage() is protected by an internal mutex; rollback(),
// has_active(), and last_report() are safe to call concurrently.
// =============================================================================
#pragma once

#include <nikola/autonomy/evolutionary_orchestrator.hpp>
#include <nikola/security/hybrid_verifier.hpp>

#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nikola::autonomy {

// ---------------------------------------------------------------------------
// StageStatus
// ---------------------------------------------------------------------------

/// Result classification for a ShadowSpine::stage() call.
enum class StageStatus : int {
    SUCCESS,              ///< All gates passed; new module is active
    SIGNATURE_REJECTED,   ///< Gate 0: hybrid signature verification failed
    ATP_DENIED,           ///< Gate 1 (EO): insufficient ATP
    SECURITY_REJECTED,    ///< Gate 1 (EO): code blacklist rejected source
    PHYSICS_REJECTED,     ///< Gate 2 (EO): physics oracle rejected module
    LOAD_FAILED,          ///< Gate 3 (EO): dlopen / module load failed
    SYMBOL_MISSING,       ///< Gate 3 (EO): factory symbol not found
    SAME_MODULE,          ///< Gate 3 (EO): candidate is the same module already loaded
};

/// Returns a human-readable string for a StageStatus.
[[nodiscard]] constexpr std::string_view stage_status_str(StageStatus s) noexcept {
    switch (s) {
        case StageStatus::SUCCESS:            return "SUCCESS";
        case StageStatus::SIGNATURE_REJECTED: return "SIGNATURE_REJECTED";
        case StageStatus::ATP_DENIED:         return "ATP_DENIED";
        case StageStatus::SECURITY_REJECTED:  return "SECURITY_REJECTED";
        case StageStatus::PHYSICS_REJECTED:   return "PHYSICS_REJECTED";
        case StageStatus::LOAD_FAILED:        return "LOAD_FAILED";
        case StageStatus::SYMBOL_MISSING:     return "SYMBOL_MISSING";
        case StageStatus::SAME_MODULE:        return "SAME_MODULE";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// StageReport
// ---------------------------------------------------------------------------

/// Detailed report from ShadowSpine::stage().
struct StageReport {
    /// Top-level outcome.
    StageStatus status{StageStatus::LOAD_FAILED};

    /// True when Gate 0 (hybrid signatures) passed.
    bool signature_passed{false};

    /// If signature_passed == false, the specific reason.
    nikola::security::VerifyFailReason sig_fail{
        nikola::security::VerifyFailReason::NONE};

    /// Full report from the EvolutionaryOrchestrator (populated when
    /// signature_passed == true, even on EO failure).
    CycleReport cycle_report{};

    /// Convenience bool: true iff status == SUCCESS.
    [[nodiscard]] explicit operator bool() const noexcept {
        return status == StageStatus::SUCCESS;
    }
};

// ---------------------------------------------------------------------------
// ShadowSpine
// ---------------------------------------------------------------------------

/// Final SIE deploy gate: hybrid-signature verification + EO pipeline.
///
/// Owns references to an EvolutionaryOrchestrator and a HybridVerifier —
/// both must outlive this ShadowSpine instance.
///
/// Typical usage:
/// @code
///   ShadowSpine spine{eo, hv};
///   auto rep = spine.stage(path, source, sig, ed_pub, sphincs_pub, physics_fn);
///   if (rep) {
///       // Module is now live — inspect rep.cycle_report for metrics
///   } else if (!rep.signature_passed) {
///       // Crypto rejected — rep.sig_fail has the reason
///   } else {
///       // EO pipeline failure — rep.cycle_report.status has the reason
///   }
///   spine.rollback();  // optional: revert to previous module
/// @endcode
class ShadowSpine {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Construct a ShadowSpine wrapping the given EO and verifier.
    /// @param eo  EvolutionaryOrchestrator to delegate hot-swap to.
    /// @param hv  HybridVerifier that performs Gate 0 signature checks.
    explicit ShadowSpine(EvolutionaryOrchestrator&          eo,
                         nikola::security::HybridVerifier&  hv) noexcept;

    // Non-copyable, non-movable (std::mutex is neither copyable nor movable)
    ShadowSpine(const ShadowSpine&)            = delete;
    ShadowSpine& operator=(const ShadowSpine&) = delete;
    ShadowSpine(ShadowSpine&&)                 = delete;
    ShadowSpine& operator=(ShadowSpine&&)      = delete;

    // -----------------------------------------------------------------------
    // Core operation
    // -----------------------------------------------------------------------

    /// Attempt to deploy a candidate module through the full SIE pipeline.
    ///
    /// @param so_path        Path to the candidate shared object.
    /// @param source_code    Source code for CodePatternBlacklist scan (may be empty).
    /// @param sig            HybridSignature (Ed25519 + SPHINCS+).
    /// @param ed_pub         32-byte Ed25519 public key.
    /// @param sphincs_pub    64-byte SPHINCS+-shake-256f public key.
    /// @param physics_fn     Optional physics provider for Gate 2 (may be {}).
    /// @returns StageReport  Full results; operator bool() true on success.
    [[nodiscard]] StageReport stage(
            std::string_view                                   so_path,
            const std::string&                                 source_code,
            const nikola::security::HybridSignature&           sig,
            const std::vector<uint8_t>&                        ed_pub,
            const std::vector<uint8_t>&                        sphincs_pub,
            EvolutionaryOrchestrator::PhysicsProvider          physics_fn = {});

    // -----------------------------------------------------------------------
    // Rollback
    // -----------------------------------------------------------------------

    /// Roll back to the previously active module.
    /// Delegates to EvolutionaryOrchestrator::rollback().
    /// @returns true if rollback succeeded; false if no previous module exists.
    bool rollback() noexcept;

    // -----------------------------------------------------------------------
    // Inspection
    // -----------------------------------------------------------------------

    /// True if the EO currently has an active module loaded.
    [[nodiscard]] bool has_active() const noexcept;

    /// Pointer to the most recent StageReport, or nullptr if stage() was never
    /// called.
    [[nodiscard]] const StageReport* last_report() const noexcept;

    /// Cumulative statistics from the underlying EvolutionaryOrchestrator.
    [[nodiscard]] CycleStats stats() const noexcept;

private:
    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Map a CycleStatus to the corresponding StageStatus.
    [[nodiscard]] static StageStatus from_cycle_status(CycleStatus cs) noexcept;

    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    EvolutionaryOrchestrator&          eo_;
    nikola::security::HybridVerifier&  hv_;
    mutable std::mutex                 stage_mutex_;
    std::optional<StageReport>         last_report_;
};

} // namespace nikola::autonomy
