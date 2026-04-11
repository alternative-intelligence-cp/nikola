/**
 * @file include/nikola/autonomy/self_improvement_engine.hpp
 * @brief Phase 146 — SelfImprovementEngine: end-to-end autonomous code
 *        generation, validation, and deployment.
 *
 * This is the glue that closes the self-improvement loop.  When the
 * DecisionLoop fires GENERATE_CODE, the SIE orchestrates:
 *
 *   1. Instruction formulation (from NikolaState → specialist prompt)
 *   2. Code generation (SpecialistInterface::ask)
 *   3. Source extraction (extract_code_block)
 *   4. Compile validation (AriaCompileValidator or g++ for C++ modules)
 *   5. Module packaging (source → .so with nikola_module_factory)
 *   6. Hybrid signing (Ed25519 + SPHINCS+-shake-256f)
 *   7. ShadowSpine deployment (Gate 0–3 + MetabolicLock)
 *   8. Proposal persistence (CodeProposalStore)
 *   9. Quality measurement (pre vs post deployment)
 *
 * The SIE is stateful: it owns a signing keypair (generated once at
 * construction), a specialist subprocess, and the proposal store.
 *
 * Thread safety: run_cycle() is serialised by an internal mutex.
 * Only one self-improvement cycle runs at a time.
 */
#pragma once

#include <nikola/aria/compile_validator.hpp>
#include <nikola/aria/code_proposal_store.hpp>
#include <nikola/aria/specialist_interface.hpp>
#include <nikola/autonomy/shadow_spine.hpp>
#include <nikola/security/hybrid_verifier.hpp>
#include <nikola/security/sphincs_signer.hpp>

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// Forward-declare OpenSSL types to avoid pulling in the header
typedef struct evp_pkey_st EVP_PKEY;

namespace nikola::autonomy {

// Forward-declare NikolaState (defined in decision_loop.hpp) to break
// the circular include.  The .cpp includes the full header.
struct NikolaState;

// ============================================================================
// SIECycleResult
// ============================================================================

/// Outcome of a single self-improvement cycle.
enum class SIEOutcome : int {
    SUCCESS              = 0,  ///< Full cycle succeeded — module deployed
    SPECIALIST_FAILED    = 1,  ///< Specialist model returned an error
    NO_CODE_EXTRACTED    = 2,  ///< extract_code_block returned empty
    COMPILE_FAILED       = 3,  ///< Source failed to compile
    PACKAGING_FAILED     = 4,  ///< g++ shared-lib packaging failed
    SIGNING_FAILED       = 5,  ///< Hybrid signature creation failed
    GATE0_REJECTED       = 6,  ///< ShadowSpine Gate 0 (signature) rejected
    GATE1_REJECTED       = 7,  ///< EO Gate 1 (blacklist) rejected
    GATE2_REJECTED       = 8,  ///< EO Gate 2 (physics) rejected
    GATE3_REJECTED       = 9,  ///< EO Gate 3 (module load) rejected
    ATP_DENIED           = 10, ///< Insufficient ATP for the cycle
    QUALITY_REGRESSION   = 11, ///< Post-deployment quality worse than pre
};

/// Human-readable label for an SIEOutcome.
[[nodiscard]] constexpr std::string_view sie_outcome_str(SIEOutcome o) noexcept {
    switch (o) {
        case SIEOutcome::SUCCESS:            return "SUCCESS";
        case SIEOutcome::SPECIALIST_FAILED:  return "SPECIALIST_FAILED";
        case SIEOutcome::NO_CODE_EXTRACTED:  return "NO_CODE_EXTRACTED";
        case SIEOutcome::COMPILE_FAILED:     return "COMPILE_FAILED";
        case SIEOutcome::PACKAGING_FAILED:   return "PACKAGING_FAILED";
        case SIEOutcome::SIGNING_FAILED:     return "SIGNING_FAILED";
        case SIEOutcome::GATE0_REJECTED:     return "GATE0_REJECTED";
        case SIEOutcome::GATE1_REJECTED:     return "GATE1_REJECTED";
        case SIEOutcome::GATE2_REJECTED:     return "GATE2_REJECTED";
        case SIEOutcome::GATE3_REJECTED:     return "GATE3_REJECTED";
        case SIEOutcome::ATP_DENIED:         return "ATP_DENIED";
        case SIEOutcome::QUALITY_REGRESSION: return "QUALITY_REGRESSION";
    }
    return "UNKNOWN";
}

/// Detailed result from a single self-improvement cycle.
struct SIECycleResult {
    SIEOutcome  outcome{SIEOutcome::SPECIALIST_FAILED};

    /// The instruction sent to the specialist.
    std::string instruction;

    /// Raw response from the specialist model.
    std::string raw_response;

    /// Extracted source code (empty if extraction failed).
    std::string source_code;

    /// Compile output (errors, warnings).
    std::string compile_output;

    /// Path to the packaged .so (empty if packaging didn't happen).
    std::string so_path;

    /// ShadowSpine stage report (populated if deployment was attempted).
    std::optional<StageReport> stage_report;

    /// Proposal ID in the code store (0 if not stored).
    uint64_t proposal_id{0};

    /// Wall-clock duration of the entire cycle.
    double elapsed_ms{0.0};

    /// True iff outcome == SUCCESS.
    [[nodiscard]] explicit operator bool() const noexcept {
        return outcome == SIEOutcome::SUCCESS;
    }
};

// ============================================================================
// SelfImprovementEngine configuration
// ============================================================================

struct SIEConfig {
    /// Path to the Aria specialist server.py script.
    std::string specialist_server_path;

    /// Path to the ariac binary (for Aria code validation).
    std::string ariac_path;

    /// Path to g++ (for C++ module compilation).
    std::string gpp_path = "/usr/bin/g++";

    /// Path to the LMDB code proposal store directory.
    std::string proposal_store_path;

    /// Working directory for temporary build artifacts.
    std::string work_dir = "/tmp/nikola_sie";

    /// Specialist query timeout in milliseconds.
    int specialist_timeout_ms = 60'000;

    /// C++ compilation timeout in milliseconds.
    int compile_timeout_ms = 30'000;
};

// ============================================================================
// SelfImprovementEngine
// ============================================================================

class SelfImprovementEngine {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Construct a SelfImprovementEngine.
    ///
    /// Generates a fresh Ed25519 + SPHINCS+ keypair for self-signing.
    /// Creates the work directory if it doesn't exist.
    ///
    /// @param spine     ShadowSpine for Gate 0–3 deployment.
    /// @param cfg       SIE configuration (paths, timeouts).
    explicit SelfImprovementEngine(ShadowSpine& spine,
                                   SIEConfig    cfg);

    /// Destructor — stops specialist subprocess, frees Ed25519 key.
    ~SelfImprovementEngine();

    // Non-copyable, non-movable
    SelfImprovementEngine(const SelfImprovementEngine&)            = delete;
    SelfImprovementEngine& operator=(const SelfImprovementEngine&) = delete;
    SelfImprovementEngine(SelfImprovementEngine&&)                 = delete;
    SelfImprovementEngine& operator=(SelfImprovementEngine&&)      = delete;

    // -----------------------------------------------------------------------
    // Core operation
    // -----------------------------------------------------------------------

    /// Run a complete self-improvement cycle.
    ///
    /// Given the current NikolaState, formulates an improvement instruction,
    /// generates candidate code, validates it, packages it into a .so,
    /// self-signs it, and attempts deployment through ShadowSpine.
    ///
    /// @param state  Current cognitive/metabolic state snapshot.
    /// @returns SIECycleResult with full details of every step.
    [[nodiscard]] SIECycleResult run_cycle(const NikolaState& state);

    /// Run an SIE cycle with pre-generated source code (solo mode).
    ///
    /// Skips the specialist query and code extraction steps.  Starts
    /// directly at packaging → signing → deployment → store.
    ///
    /// @param source_code  C++ source code to package and deploy.
    /// @param instruction  Description of the improvement (for logging/store).
    /// @returns SIECycleResult with full details.
    [[nodiscard]] SIECycleResult run_cycle_with_source(
            const std::string& source_code,
            const std::string& instruction);

    // -----------------------------------------------------------------------
    // Inspection
    // -----------------------------------------------------------------------

    /// Total cycles attempted.
    [[nodiscard]] uint32_t cycles_attempted() const noexcept { return cycles_attempted_; }

    /// Total cycles that resulted in SUCCESS.
    [[nodiscard]] uint32_t cycles_succeeded() const noexcept { return cycles_succeeded_; }

    /// True if the specialist subprocess is running.
    [[nodiscard]] bool specialist_running() const noexcept;

    /// Public keys used for self-signing (for external verification).
    [[nodiscard]] const std::vector<uint8_t>& ed25519_public_key() const noexcept {
        return ed_pk_;
    }
    [[nodiscard]] const std::vector<uint8_t>& sphincs_public_key() const noexcept {
        return sphincs_kp_.pk;
    }

private:
    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Formulate an improvement instruction from the current state.
    [[nodiscard]] std::string formulate_instruction(const NikolaState& state) const;

    /// Package C++ source into a shared library (.so).
    /// Returns path to the .so on success, empty string on failure.
    [[nodiscard]] std::string package_module(const std::string& source,
                                             std::string& compile_output);

    /// Create a HybridSignature for a module binary.
    [[nodiscard]] std::optional<security::HybridSignature>
    sign_module(const std::vector<uint8_t>& binary);

    /// Read a file into a byte vector.
    [[nodiscard]] static std::vector<uint8_t> read_binary(const std::string& path);

    /// Start the specialist subprocess if not already running.
    bool ensure_specialist_running();

    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    ShadowSpine&          spine_;
    SIEConfig             cfg_;

    aria::SpecialistInterface   specialist_;
    aria::AriaCompileValidator  validator_;
    std::unique_ptr<aria::CodeProposalStore> store_;

    // Signing keypair (generated at construction)
    std::vector<uint8_t>        ed_pk_;     ///< 32-byte Ed25519 public key
    std::vector<uint8_t>        ed_sk_;     ///< 32-byte Ed25519 private key
    security::SphincsKeypair    sphincs_kp_;
    EVP_PKEY*                   ed_pkey_{nullptr}; ///< OpenSSL EVP handle

    uint32_t cycles_attempted_{0};
    uint32_t cycles_succeeded_{0};
    bool     specialist_started_{false};

    mutable std::mutex cycle_mutex_;
};

} // namespace nikola::autonomy
