/**
 * @file tests/unit/phase112_test_module.cpp
 * @brief Minimal shared library compiled as a MODULE for Phase 112 tests.
 *
 * This file is compiled with CMake's MODULE library type, producing a .so that
 * ModuleSwapper can dlopen.  It exports two factory symbols so the test suite
 * can cover custom factory-symbol lookup.
 *
 *   nikola_module_factory   — default symbol; returns a stable sentinel value.
 *   nikola_alt_factory      — alternative symbol; returns a different sentinel.
 *
 * The sentinel values are deliberately non-null and non-aligned addresses that
 * are easy to recognise in test assertions without dereferencing.
 */

#include <cstddef>   // std::size_t

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Default factory symbol expected by ModuleSwapper's default constructor.
 * Returns a sentinel void* that tests can compare for non-null identity.
 */
void* nikola_module_factory() {
    /* Casting an integer literal to void* is a recognised pattern for
     * producing a cheap sentinel; the resulting pointer is never dereferenced
     * in tests.  Suppress -Wint-to-pointer-size warnings on 64-bit builds. */
    return (void*)static_cast<std::size_t>(0xDEAD'C0DE'0000'0001ULL);
}

/**
 * Alternate factory symbol — used by the custom-symbol-name test case.
 */
void* nikola_alt_factory() {
    return (void*)static_cast<std::size_t>(0xDEAD'C0DE'0000'0002ULL);
}

#ifdef __cplusplus
}  // extern "C"
#endif
