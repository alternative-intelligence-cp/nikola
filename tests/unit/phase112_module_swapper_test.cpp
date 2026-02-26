/**
 * @file tests/unit/phase112_module_swapper_test.cpp
 * @brief Phase 112 — ModuleSwapper dlopen hot-swap engine test suite.
 *
 * Tests the full swap_in / rollback lifecycle, error paths, and thread safety
 * of nikola::autonomy::ModuleSwapper.
 *
 * The CMake build compiles phase112_test_module.cpp as a MODULE library and
 * writes it to ${CMAKE_BINARY_DIR}/test_plugins/phase112_test_plugin.so.
 * The directory is injected at compile time via PHASE112_PLUGIN_DIR.
 */

#include <nikola/autonomy/module_swapper.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <thread>
#include <vector>

// ── Plugin path ────────────────────────────────────────────────────────────
// PHASE112_PLUGIN_DIR is defined via target_compile_definitions in CMakeLists.
#ifndef PHASE112_PLUGIN_DIR
#  define PHASE112_PLUGIN_DIR "."
#endif

static const std::string k_plugin_path =
    std::string(PHASE112_PLUGIN_DIR) + "/phase112_test_plugin.so";

static const std::string k_bad_path =
    "/nonexistent/surely/absent/module.so";

// ── Convenience alias ──────────────────────────────────────────────────────
using nikola::autonomy::ModuleSwapper;
using nikola::autonomy::SwapResult;
using nikola::autonomy::swap_result_str;

// ==========================================================================
// Helpers
// ==========================================================================

/// Confirm the test plugin file actually exists before trying to load it.
/// If it is missing, the build is broken — fail loudly.
static void require_plugin_exists() {
    REQUIRE(std::filesystem::exists(k_plugin_path));
}

// ==========================================================================
// Section 1 — Construction & initial state
// ==========================================================================

TEST_CASE("ModuleSwapper — default-constructed state", "[phase112][swapper]") {
    ModuleSwapper sw;
    CHECK_FALSE(sw.has_active());
    CHECK_FALSE(sw.has_previous());
    CHECK(sw.active_factory() == nullptr);
    CHECK(sw.previous_factory() == nullptr);
    CHECK(sw.active_path().empty());
    CHECK(sw.previous_path().empty());
}

// ==========================================================================
// Section 2 — Successful load
// ==========================================================================

TEST_CASE("ModuleSwapper — swap_in valid .so returns SUCCESS", "[phase112][swapper]") {
    require_plugin_exists();
    ModuleSwapper sw;
    auto res = sw.swap_in(k_plugin_path);
    REQUIRE(res == SwapResult::SUCCESS);
}

TEST_CASE("ModuleSwapper — after successful swap_in has_active is true", "[phase112][swapper]") {
    require_plugin_exists();
    ModuleSwapper sw;
    REQUIRE(sw.swap_in(k_plugin_path) == SwapResult::SUCCESS);
    CHECK(sw.has_active());
    CHECK_FALSE(sw.has_previous());
}

TEST_CASE("ModuleSwapper — active_factory is non-null after swap_in", "[phase112][swapper]") {
    require_plugin_exists();
    ModuleSwapper sw;
    REQUIRE(sw.swap_in(k_plugin_path) == SwapResult::SUCCESS);
    CHECK(sw.active_factory() != nullptr);
}

TEST_CASE("ModuleSwapper — active_path matches loaded path", "[phase112][swapper]") {
    require_plugin_exists();
    ModuleSwapper sw;
    REQUIRE(sw.swap_in(k_plugin_path) == SwapResult::SUCCESS);
    CHECK(sw.active_path() == k_plugin_path);
}

// ==========================================================================
// Section 3 — Second swap (previous slot)
// ==========================================================================

TEST_CASE("ModuleSwapper — second swap_in keeps previous slot occupied", "[phase112][swapper]") {
    require_plugin_exists();
    ModuleSwapper sw;

    // Load once.
    REQUIRE(sw.swap_in(k_plugin_path) == SwapResult::SUCCESS);

    // To force a second distinct load, use a symlink or an alternate
    // realpath-distinct name.  Here we use a copy to create a unique path.
    const std::string copy_path =
        std::string(PHASE112_PLUGIN_DIR) + "/phase112_test_plugin_copy.so";

    std::filesystem::copy_file(
        k_plugin_path, copy_path,
        std::filesystem::copy_options::overwrite_existing);

    auto res = sw.swap_in(copy_path);
    REQUIRE(res == SwapResult::SUCCESS);

    CHECK(sw.has_active());
    CHECK(sw.has_previous());
    CHECK(sw.active_path() == copy_path);
    CHECK(sw.previous_path() == k_plugin_path);

    // Clean up copy.
    std::filesystem::remove(copy_path);
}

// ==========================================================================
// Section 4 — Rollback
// ==========================================================================

TEST_CASE("ModuleSwapper — rollback with no previous returns false", "[phase112][swapper]") {
    ModuleSwapper sw;
    CHECK_FALSE(sw.rollback());
}

TEST_CASE("ModuleSwapper — rollback with only active (no previous) returns false", "[phase112][swapper]") {
    require_plugin_exists();
    ModuleSwapper sw;
    REQUIRE(sw.swap_in(k_plugin_path) == SwapResult::SUCCESS);
    // Has active but no previous — rollback must return false.
    CHECK_FALSE(sw.rollback());
    // Active should still be there.
    CHECK(sw.has_active());
}

TEST_CASE("ModuleSwapper — rollback after two swaps restores first module", "[phase112][swapper]") {
    require_plugin_exists();

    const std::string copy_path =
        std::string(PHASE112_PLUGIN_DIR) + "/phase112_rollback_copy.so";

    std::filesystem::copy_file(
        k_plugin_path, copy_path,
        std::filesystem::copy_options::overwrite_existing);

    ModuleSwapper sw;
    REQUIRE(sw.swap_in(k_plugin_path) == SwapResult::SUCCESS);
    REQUIRE(sw.swap_in(copy_path)     == SwapResult::SUCCESS);

    // Rollback should restore the first module.
    REQUIRE(sw.rollback());
    CHECK(sw.active_path() == k_plugin_path);
    CHECK_FALSE(sw.has_previous());

    std::filesystem::remove(copy_path);
}

TEST_CASE("ModuleSwapper — rollback clears previous slot", "[phase112][swapper]") {
    require_plugin_exists();

    const std::string copy_path =
        std::string(PHASE112_PLUGIN_DIR) + "/phase112_rollback_clear.so";

    std::filesystem::copy_file(
        k_plugin_path, copy_path,
        std::filesystem::copy_options::overwrite_existing);

    ModuleSwapper sw;
    REQUIRE(sw.swap_in(k_plugin_path) == SwapResult::SUCCESS);
    REQUIRE(sw.swap_in(copy_path)     == SwapResult::SUCCESS);
    sw.rollback();

    CHECK_FALSE(sw.has_previous());

    std::filesystem::remove(copy_path);
}

// ==========================================================================
// Section 5 — Error paths
// ==========================================================================

TEST_CASE("ModuleSwapper — bad path returns LOAD_FAILED", "[phase112][swapper]") {
    ModuleSwapper sw;
    auto res = sw.swap_in(k_bad_path);
    CHECK(res == SwapResult::LOAD_FAILED);
    CHECK_FALSE(sw.has_active());
}

TEST_CASE("ModuleSwapper — validator rejection returns VALIDATION_FAILED", "[phase112][swapper]") {
    require_plugin_exists();
    ModuleSwapper sw;
    auto res = sw.swap_in(k_plugin_path, [](void*) { return false; });
    CHECK(res == SwapResult::VALIDATION_FAILED);
    // Rejected module must NOT become active.
    CHECK_FALSE(sw.has_active());
}

TEST_CASE("ModuleSwapper — validator accepting returns SUCCESS", "[phase112][swapper]") {
    require_plugin_exists();
    ModuleSwapper sw;
    bool validator_called = false;
    auto res = sw.swap_in(k_plugin_path, [&](void* sym) {
        validator_called = true;
        return sym != nullptr;
    });
    CHECK(res == SwapResult::SUCCESS);
    CHECK(validator_called);
}

TEST_CASE("ModuleSwapper — loading same path twice returns SAME_MODULE", "[phase112][swapper]") {
    require_plugin_exists();
    ModuleSwapper sw;
    REQUIRE(sw.swap_in(k_plugin_path) == SwapResult::SUCCESS);
    auto res = sw.swap_in(k_plugin_path);
    CHECK(res == SwapResult::SAME_MODULE);
    // Active should still be the first load.
    CHECK(sw.active_path() == k_plugin_path);
    CHECK_FALSE(sw.has_previous());
}

// ==========================================================================
// Section 6 — Custom factory symbol
// ==========================================================================

TEST_CASE("ModuleSwapper — custom factory symbol resolves correctly", "[phase112][swapper]") {
    require_plugin_exists();
    ModuleSwapper sw{"nikola_alt_factory"};
    auto res = sw.swap_in(k_plugin_path);
    REQUIRE(res == SwapResult::SUCCESS);
    CHECK(sw.active_factory() != nullptr);
}

TEST_CASE("ModuleSwapper — missing symbol returns SYMBOL_MISSING", "[phase112][swapper]") {
    require_plugin_exists();
    ModuleSwapper sw{"symbol_that_does_not_exist_xyzzy"};
    auto res = sw.swap_in(k_plugin_path);
    CHECK(res == SwapResult::SYMBOL_MISSING);
    CHECK_FALSE(sw.has_active());
}

// ==========================================================================
// Section 7 — reset()
// ==========================================================================

TEST_CASE("ModuleSwapper — reset clears both slots", "[phase112][swapper]") {
    require_plugin_exists();

    const std::string copy_path =
        std::string(PHASE112_PLUGIN_DIR) + "/phase112_reset_copy.so";

    std::filesystem::copy_file(
        k_plugin_path, copy_path,
        std::filesystem::copy_options::overwrite_existing);

    ModuleSwapper sw;
    REQUIRE(sw.swap_in(k_plugin_path) == SwapResult::SUCCESS);
    REQUIRE(sw.swap_in(copy_path)     == SwapResult::SUCCESS);

    sw.reset();
    CHECK_FALSE(sw.has_active());
    CHECK_FALSE(sw.has_previous());
    CHECK(sw.active_factory()   == nullptr);
    CHECK(sw.previous_factory() == nullptr);

    std::filesystem::remove(copy_path);
}

// ==========================================================================
// Section 8 — Move semantics
// ==========================================================================

TEST_CASE("ModuleSwapper — move constructor transfers ownership", "[phase112][swapper]") {
    require_plugin_exists();
    ModuleSwapper sw;
    REQUIRE(sw.swap_in(k_plugin_path) == SwapResult::SUCCESS);
    const std::string original_path = sw.active_path();

    ModuleSwapper sw2{std::move(sw)};
    CHECK(sw2.has_active());
    CHECK(sw2.active_path() == original_path);
    // Source should be vacated.
    CHECK_FALSE(sw.has_active());
}

TEST_CASE("ModuleSwapper — swap_result_str covers all enumerators", "[phase112][swapper]") {
    using SwapResult = nikola::autonomy::SwapResult;
    CHECK(swap_result_str(SwapResult::SUCCESS)           == "SUCCESS");
    CHECK(swap_result_str(SwapResult::LOAD_FAILED)       == "LOAD_FAILED");
    CHECK(swap_result_str(SwapResult::SYMBOL_MISSING)    == "SYMBOL_MISSING");
    CHECK(swap_result_str(SwapResult::VALIDATION_FAILED) == "VALIDATION_FAILED");
    CHECK(swap_result_str(SwapResult::SAME_MODULE)       == "SAME_MODULE");
}

// ==========================================================================
// Section 9 — Concurrent swap (basic thread-safety smoke test)
// ==========================================================================

TEST_CASE("ModuleSwapper — concurrent swap_in calls do not crash",
          "[phase112][swapper][thread]") {
    require_plugin_exists();

    // N threads each try to swap_in the same path — only one should succeed
    // per iteration; the rest will get SAME_MODULE after the first one lands.
    // The goal is: no crash, no UB, the object remains coherent.
    ModuleSwapper sw;
    constexpr int N = 8;
    std::vector<SwapResult> results(N, SwapResult::LOAD_FAILED);

    {
        std::vector<std::thread> threads;
        threads.reserve(N);
        for (int i = 0; i < N; ++i) {
            threads.emplace_back([&sw, &results, i]() {
                results[i] = sw.swap_in(k_plugin_path);
            });
        }
        for (auto& t : threads) t.join();
    }

    // At least one thread should have succeeded.
    int success_count = 0;
    for (auto r : results)
        if (r == SwapResult::SUCCESS) ++success_count;

    CHECK(success_count >= 1);
    CHECK(sw.has_active());
}
