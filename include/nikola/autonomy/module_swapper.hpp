/**
 * @file include/nikola/autonomy/module_swapper.hpp
 * @brief dlopen-based hot-swap engine for the SIE Phase-4 loop.
 *
 * The ModuleSwapper provides ABI-stable, one-step-rollback module replacement
 * using POSIX dlopen / dlsym / dlclose.  It is the deployment layer of the
 * Self-Improvement Evolution (SIE) lifecycle described in:
 *   docs/info/integration/sections/05_autonomous_systems/04_self_improvement.md
 *   §2.4 Deployment Protocol
 *
 * Design contract
 * ─────────────────────────────────────────────────────────────────────────────
 * • Manages two slots — *active* and *previous* — enabling a single rollback.
 * • swap_in() promotes a validated candidate; the current active is demoted to
 *   previous (displacing and dlclose-ing whatever was there before).
 * • rollback() restores the previous slot to active; the displaced module is
 *   dlclose-d immediately.
 * • All public methods are serialised via an internal std::mutex.
 * • Does NOT acquire a MetabolicLock.  Per spec §7, the caller (Evolutionary
 *   Orchestrator) acquires the lock before starting the SIE cycle.
 *
 * Expected plugin ABI
 * ─────────────────────────────────────────────────────────────────────────────
 * Every candidate .so must export a C symbol named "nikola_module_factory" (or
 * the name supplied to the constructor).  The factory signature is intentionally
 * left void* — the SIE orchestrator casts it to the appropriate PIMPL factory
 * type after consulting the Physics Oracle.
 *
 * Usage example
 * ─────────────────────────────────────────────────────────────────────────────
 * @code
 *   // 1. Caller acquires MetabolicLock(controller, SWAP_ATP_COST).
 *   nikola::autonomy::ModuleSwapper swapper{"nikola_module_factory"};
 *
 *   auto res = swapper.swap_in("/tmp/sandbox/candidate.so",
 *       [&oracle](void* factory) {
 *           return oracle.standard_candle_test(factory);
 *       });
 *
 *   if (res != nikola::autonomy::SwapResult::SUCCESS) {
 *       log_error("swap failed: {}", swap_result_str(res));
 *       return;
 *   }
 *
 *   // 2. Later, if something diverges:
 *   if (!swapper.rollback())
 *       log_error("no previous module to roll back to");
 * @endcode
 */

#pragma once

#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>

namespace nikola::autonomy {

// ============================================================================
// SwapResult
// ============================================================================

/// Result codes returned by ModuleSwapper::swap_in().
enum class SwapResult : int {
    SUCCESS           = 0, ///< Candidate promoted; it is now the active module.
    LOAD_FAILED       = 1, ///< dlopen() failed (bad path, missing deps, etc.).
    SYMBOL_MISSING    = 2, ///< Factory symbol not exported by the candidate .so.
    VALIDATION_FAILED = 3, ///< Caller-supplied validator rejected the candidate.
    SAME_MODULE       = 4, ///< Candidate path hash matches the active module.
};

/// Human-readable label for a SwapResult — useful for logging.
[[nodiscard]] constexpr std::string_view swap_result_str(SwapResult r) noexcept {
    switch (r) {
        case SwapResult::SUCCESS:           return "SUCCESS";
        case SwapResult::LOAD_FAILED:       return "LOAD_FAILED";
        case SwapResult::SYMBOL_MISSING:    return "SYMBOL_MISSING";
        case SwapResult::VALIDATION_FAILED: return "VALIDATION_FAILED";
        case SwapResult::SAME_MODULE:       return "SAME_MODULE";
        default:                            return "UNKNOWN";
    }
}

// ============================================================================
// LoadedModule
// ============================================================================

/// Internal descriptor for a single loaded shared-library slot.
///
/// Lifecycle: constructed empty (all zero/null), populated by ModuleSwapper,
/// and released via unload().  Not thread-safe on its own; callers must hold
/// the ModuleSwapper's mutex while touching a slot.
struct LoadedModule {
    void*       dl_handle{nullptr};   ///< Handle returned by ::dlopen().
    void*       factory_sym{nullptr}; ///< Resolved factory symbol pointer.
    std::string path;                 ///< Path string passed to ::dlopen().
    std::size_t path_hash{0};         ///< std::hash<std::string>{}(path).

    /// True iff a library is currently loaded in this slot.
    [[nodiscard]] bool occupied() const noexcept { return dl_handle != nullptr; }

    /// Dlclose the library and zero all fields.  Safe on an empty slot.
    void unload() noexcept;
};

// ============================================================================
// ModuleSwapper
// ============================================================================

/// Thread-safe dlopen-based hot-swap engine.
///
/// Owns OS handles for up to two shared libraries (active + previous).
/// Move-only; copies are deleted.
class ModuleSwapper {
public:
    // ── Types ────────────────────────────────────────────────────────────────

    /// Validator callback type.
    ///
    /// Receives the resolved factory symbol pointer for the candidate module.
    /// Return true to allow promotion; false to reject (VALIDATION_FAILED).
    ///
    /// The callback is invoked while the internal mutex is held — keep it fast
    /// and do not call back into ModuleSwapper from inside the validator.
    using ValidatorFn = std::function<bool(void* factory_sym)>;

    // ── Construction / destruction ────────────────────────────────────────────

    /// Construct with the name of the factory symbol to locate in each .so.
    ///
    /// @param factory_symbol  C symbol exported by candidate modules.
    ///                        Defaults to "nikola_module_factory".
    explicit ModuleSwapper(
        std::string factory_symbol = "nikola_module_factory");

    /// Dlclose both active and previous modules on destruction.
    ~ModuleSwapper();

    // Non-copyable (owns POSIX handles).
    ModuleSwapper(const ModuleSwapper&)            = delete;
    ModuleSwapper& operator=(const ModuleSwapper&) = delete;

    // Movable.
    ModuleSwapper(ModuleSwapper&&) noexcept;
    ModuleSwapper& operator=(ModuleSwapper&&) noexcept;

    // ── Core operations ───────────────────────────────────────────────────────

    /// Load a candidate .so, validate it, and promote it to the active slot.
    ///
    /// Sequence:
    ///   1. Hash so_path; reject immediately if it matches the active module.
    ///   2. dlopen(so_path, RTLD_NOW | RTLD_LOCAL).
    ///   3. dlsym(handle, factory_symbol_).
    ///   4. If validator supplied, call it; dlclose + return on false.
    ///   5. Move current active → previous (dlclose old previous if present).
    ///   6. Install new module as active.
    ///
    /// @param so_path    Filesystem path to the candidate shared library.
    /// @param validator  Optional acceptance predicate (see ValidatorFn).
    /// @returns SwapResult indicating outcome.
    [[nodiscard]] SwapResult swap_in(std::string_view so_path,
                                     ValidatorFn      validator = {});

    /// Roll back to the previous module.
    ///
    /// Dlcloses the current active module, promotes previous → active, and
    /// clears the previous slot.  Returns false (and makes no changes) if the
    /// previous slot is empty.
    bool rollback();

    /// Unload all modules and reset both slots to empty.
    void reset() noexcept;

    // ── Inspection ────────────────────────────────────────────────────────────

    /// True iff the active slot holds a loaded library.
    [[nodiscard]] bool has_active()   const noexcept;

    /// True iff the previous slot holds a library ready for rollback.
    [[nodiscard]] bool has_previous() const noexcept;

    /// Factory symbol pointer from the active module (nullptr if empty).
    [[nodiscard]] void* active_factory() const noexcept;

    /// Path of the active module (empty string if none).
    [[nodiscard]] std::string active_path() const noexcept;

    /// Factory symbol pointer from the previous module (nullptr if empty).
    [[nodiscard]] void* previous_factory() const noexcept;

    /// Path of the previous module (empty string if none).
    [[nodiscard]] std::string previous_path() const noexcept;

private:
    std::string        factory_symbol_; ///< Symbol name to resolve via dlsym.
    LoadedModule       active_{};       ///< Currently active module slot.
    LoadedModule       previous_{};     ///< Previous module (rollback target).
    mutable std::mutex mtx_;            ///< Protects all slot mutations.
};

} // namespace nikola::autonomy
