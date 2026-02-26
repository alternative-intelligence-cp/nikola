/**
 * @file src/autonomy/module_swapper.cpp
 * @brief Implementation of ModuleSwapper — dlopen-based hot-swap engine.
 *
 * SIE Phase-4 deployment layer.  See the header for the full design contract.
 *
 * Platform notes:
 *   • Requires POSIX dlopen/dlsym/dlclose (Linux, macOS, FreeBSD …).
 *   • Links against -ldl on Linux (added in CMakeLists.txt).
 *   • RTLD_LOCAL is used deliberately: candidate modules must not pollute the
 *     global symbol table even if they export symbols with common names.
 */

#include <nikola/autonomy/module_swapper.hpp>

#include <dlfcn.h>        // dlopen, dlsym, dlclose, dlerror
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>        // std::move, std::exchange

namespace nikola::autonomy {

// ============================================================================
// LoadedModule
// ============================================================================

void LoadedModule::unload() noexcept {
    if (dl_handle) {
        ::dlclose(dl_handle);
        dl_handle   = nullptr;
        factory_sym = nullptr;
        path.clear();
        path_hash = 0;
    }
}

// ============================================================================
// ModuleSwapper — construction / destruction
// ============================================================================

ModuleSwapper::ModuleSwapper(std::string factory_symbol)
    : factory_symbol_{std::move(factory_symbol)} {}

ModuleSwapper::~ModuleSwapper() {
    active_.unload();
    previous_.unload();
}

ModuleSwapper::ModuleSwapper(ModuleSwapper&& other) noexcept
    : factory_symbol_{std::move(other.factory_symbol_)}
    , active_        {std::exchange(other.active_,   {})}
    , previous_      {std::exchange(other.previous_, {})} {}

ModuleSwapper& ModuleSwapper::operator=(ModuleSwapper&& other) noexcept {
    if (this == &other) return *this;

    // Release our current handles before taking ownership of other's.
    active_.unload();
    previous_.unload();

    factory_symbol_ = std::move(other.factory_symbol_);
    active_         = std::exchange(other.active_,   {});
    previous_       = std::exchange(other.previous_, {});
    return *this;
}

// ============================================================================
// swap_in
// ============================================================================

SwapResult ModuleSwapper::swap_in(std::string_view so_path,
                                   ValidatorFn      validator) {
    std::string  path_str{so_path};
    std::size_t  hash = std::hash<std::string>{}(path_str);

    std::lock_guard<std::mutex> lk{mtx_};

    // ── 1. Detect exact-same module (compare path hashes) ──────────────────
    if (active_.occupied() && hash == active_.path_hash)
        return SwapResult::SAME_MODULE;

    // ── 2. dlopen the candidate ────────────────────────────────────────────
    // RTLD_NOW  – resolve all symbols immediately so we catch link errors here.
    // RTLD_LOCAL – do not expose symbols to the global namespace.
    void* handle = ::dlopen(path_str.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle)
        return SwapResult::LOAD_FAILED;   // dlerror() has more detail if needed

    // ── 3. Resolve factory symbol ──────────────────────────────────────────
    // Clear any pre-existing error before calling dlsym.
    ::dlerror();
    void* factory = ::dlsym(handle, factory_symbol_.c_str());
    if (!factory || ::dlerror() != nullptr) {
        ::dlclose(handle);
        return SwapResult::SYMBOL_MISSING;
    }

    // ── 4. Optional caller validation ──────────────────────────────────────
    if (validator && !validator(factory)) {
        ::dlclose(handle);
        return SwapResult::VALIDATION_FAILED;
    }

    // ── 5. Demote current active → previous (release old previous) ─────────
    previous_.unload();
    previous_ = std::move(active_);

    // ── 6. Install new module as active ────────────────────────────────────
    active_.dl_handle   = handle;
    active_.factory_sym = factory;
    active_.path        = std::move(path_str);
    active_.path_hash   = hash;

    return SwapResult::SUCCESS;
}

// ============================================================================
// rollback
// ============================================================================

bool ModuleSwapper::rollback() {
    std::lock_guard<std::mutex> lk{mtx_};

    if (!previous_.occupied())
        return false;

    // Unload the current active (if any).
    active_.unload();

    // Promote previous → active.
    active_   = std::move(previous_);
    previous_ = {};   // explicitly empty (default-constructed)

    return true;
}

// ============================================================================
// reset
// ============================================================================

void ModuleSwapper::reset() noexcept {
    std::lock_guard<std::mutex> lk{mtx_};
    active_.unload();
    previous_.unload();
}

// ============================================================================
// Inspection
// ============================================================================

bool ModuleSwapper::has_active() const noexcept {
    std::lock_guard<std::mutex> lk{mtx_};
    return active_.occupied();
}

bool ModuleSwapper::has_previous() const noexcept {
    std::lock_guard<std::mutex> lk{mtx_};
    return previous_.occupied();
}

void* ModuleSwapper::active_factory() const noexcept {
    std::lock_guard<std::mutex> lk{mtx_};
    return active_.factory_sym;
}

std::string ModuleSwapper::active_path() const noexcept {
    std::lock_guard<std::mutex> lk{mtx_};
    return active_.path;
}

void* ModuleSwapper::previous_factory() const noexcept {
    std::lock_guard<std::mutex> lk{mtx_};
    return previous_.factory_sym;
}

std::string ModuleSwapper::previous_path() const noexcept {
    std::lock_guard<std::mutex> lk{mtx_};
    return previous_.path;
}

} // namespace nikola::autonomy
