/**
 * @file phase128_polymorphic_defense_test.cpp
 * @brief Phase 128 — PolymorphicDefense unit tests
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/security/polymorphic_defense.hpp>

using namespace nikola::security;
using Catch::Approx;

// ---------------------------------------------------------------------------
// register_entry / current_token / validate_token
// ---------------------------------------------------------------------------

TEST_CASE("PolymorphicDefense::register_entry — returns non-zero token",
          "[Phase128][register]") {
    PolymorphicDefense pd;
    const uint64_t tok = pd.register_entry("behavior_alpha");
    REQUIRE(tok != 0u);
}

TEST_CASE("PolymorphicDefense::register_entry — dedup returns same token",
          "[Phase128][register]") {
    PolymorphicDefense pd;
    const uint64_t t1 = pd.register_entry("alpha");
    const uint64_t t2 = pd.register_entry("alpha");  // duplicate
    REQUIRE(t1 == t2);
    REQUIRE(pd.entry_count() == 1);
}

TEST_CASE("PolymorphicDefense::register_entry — distinct names get distinct tokens",
          "[Phase128][register]") {
    PolymorphicDefense pd;
    const uint64_t a = pd.register_entry("a");
    const uint64_t b = pd.register_entry("b");
    REQUIRE(a != b);
    REQUIRE(pd.entry_count() == 2);
}

TEST_CASE("PolymorphicDefense::current_token — returns correct token",
          "[Phase128][register]") {
    PolymorphicDefense pd;
    const uint64_t tok = pd.register_entry("module_a");
    REQUIRE(pd.current_token("module_a") == tok);
}

TEST_CASE("PolymorphicDefense::current_token — unknown name returns 0",
          "[Phase128][register]") {
    PolymorphicDefense pd;
    REQUIRE(pd.current_token("ghost") == 0u);
}

TEST_CASE("PolymorphicDefense::validate_token — valid token accepted",
          "[Phase128][validate]") {
    PolymorphicDefense pd;
    const uint64_t tok = pd.register_entry("secure_mod");
    REQUIRE(pd.validate_token("secure_mod", tok) == true);
}

TEST_CASE("PolymorphicDefense::validate_token — wrong token rejected",
          "[Phase128][validate]") {
    PolymorphicDefense pd;
    const uint64_t tok = pd.register_entry("secure_mod");
    REQUIRE(pd.validate_token("secure_mod", tok + 1) == false);
}

TEST_CASE("PolymorphicDefense::validate_token — unknown name rejected",
          "[Phase128][validate]") {
    PolymorphicDefense pd;
    REQUIRE(pd.validate_token("ghost", 12345) == false);
}

// ---------------------------------------------------------------------------
// remove_entry / remove_named / clear
// ---------------------------------------------------------------------------

TEST_CASE("PolymorphicDefense::remove_named — deletes entry",
          "[Phase128][remove]") {
    PolymorphicDefense pd;
    pd.register_entry("alpha");
    pd.register_entry("beta");
    REQUIRE(pd.entry_count() == 2);

    pd.remove_named("alpha");
    REQUIRE(pd.entry_count() == 1);
    REQUIRE(pd.current_token("alpha") == 0u);
    REQUIRE(pd.current_token("beta")  != 0u);
}

TEST_CASE("PolymorphicDefense::remove_named — unknown name is no-op",
          "[Phase128][remove]") {
    PolymorphicDefense pd;
    pd.register_entry("x");
    REQUIRE_NOTHROW(pd.remove_named("ghost"));
    REQUIRE(pd.entry_count() == 1);
}

TEST_CASE("PolymorphicDefense::remove_entry — removes by id",
          "[Phase128][remove]") {
    PolymorphicDefense pd;
    pd.register_entry("alpha");
    const auto* e = pd.find("alpha");
    REQUIRE(e != nullptr);
    const uint64_t id = e->id;

    pd.remove_entry(id);
    REQUIRE(pd.find("alpha") == nullptr);
}

TEST_CASE("PolymorphicDefense::clear — empties pool",
          "[Phase128][remove]") {
    PolymorphicDefense pd;
    pd.register_entry("a"); pd.register_entry("b"); pd.register_entry("c");
    pd.clear();
    REQUIRE(pd.entry_count() == 0);
}

// ---------------------------------------------------------------------------
// randomize
// ---------------------------------------------------------------------------

TEST_CASE("PolymorphicDefense::randomize — token changes after mutation",
          "[Phase128][randomize]") {
    PolymorphicDefense pd;
    const uint64_t old_tok = pd.register_entry("target");

    // Force full mutation
    pd.randomize(1.0);

    const uint64_t new_tok = pd.current_token("target");
    // Token must have changed (statistically certain with good RNG)
    REQUIRE(new_tok != old_tok);
}

TEST_CASE("PolymorphicDefense::randomize — old token invalidated after mutation",
          "[Phase128][randomize]") {
    PolymorphicDefense pd;
    const uint64_t old_tok = pd.register_entry("guard");

    pd.randomize(1.0);

    REQUIRE(pd.validate_token("guard", old_tok) == false);
    REQUIRE(pd.validate_token("guard", pd.current_token("guard")) == true);
}

TEST_CASE("PolymorphicDefense::randomize — zero rate mutates at least 1",
          "[Phase128][randomize]") {
    PolymorphicDefense pd;
    pd.register_entry("a");
    // rate 0.0 → clamps to max(1, 0%) = 1 entry mutated
    const uint64_t before = pd.stats().total_mutations;
    pd.randomize(0.0);
    REQUIRE(pd.stats().total_mutations > before);
}

TEST_CASE("PolymorphicDefense::randomize — full rate mutates all entries",
          "[Phase128][randomize]") {
    PolymorphicDefense pd;
    pd.register_entry("a");
    pd.register_entry("b");
    pd.register_entry("c");

    const uint64_t before = pd.stats().total_mutations;
    pd.randomize(1.0);
    REQUIRE(pd.stats().total_mutations == before + 3);
}

TEST_CASE("PolymorphicDefense::randomize — empty pool is no-op",
          "[Phase128][randomize]") {
    PolymorphicDefense pd;
    REQUIRE_NOTHROW(pd.randomize(1.0));
    REQUIRE(pd.stats().total_mutations == 0);
}

// ---------------------------------------------------------------------------
// remutate
// ---------------------------------------------------------------------------

TEST_CASE("PolymorphicDefense::remutate — force-remutates specific entry",
          "[Phase128][remutate]") {
    PolymorphicDefense pd;
    pd.register_entry("alpha");
    pd.register_entry("beta");

    const uint64_t beta_before = pd.current_token("beta");

    pd.remutate("alpha");

    // beta unchanged
    REQUIRE(pd.current_token("beta") == beta_before);
    // alpha's mutation count incremented
    REQUIRE(pd.find("alpha")->mutation_count == 1);
}

TEST_CASE("PolymorphicDefense::remutate — unknown name is no-op",
          "[Phase128][remutate]") {
    PolymorphicDefense pd;
    REQUIRE_NOTHROW(pd.remutate("ghost"));
    REQUIRE(pd.stats().total_mutations == 0);
}

// ---------------------------------------------------------------------------
// Callback
// ---------------------------------------------------------------------------

TEST_CASE("PolymorphicDefense::on_mutation — fires on randomize",
          "[Phase128][callback]") {
    PolymorphicDefense pd;
    pd.register_entry("watched");

    bool fired = false;
    std::string fired_name;

    pd.on_mutation([&](const ProtectedEntry& e) {
        fired      = true;
        fired_name = e.name;
    });

    pd.randomize(1.0);
    REQUIRE(fired == true);
    REQUIRE(fired_name == "watched");
}

TEST_CASE("PolymorphicDefense::on_mutation — fires on remutate",
          "[Phase128][callback]") {
    PolymorphicDefense pd;
    pd.register_entry("target");

    int count = 0;
    pd.on_mutation([&](const ProtectedEntry&) { ++count; });

    pd.remutate("target");
    pd.remutate("target");
    REQUIRE(count == 2);
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

TEST_CASE("PolymorphicDefense::stats — correct counts",
          "[Phase128][stats]") {
    PolymorphicDefense pd;
    pd.register_entry("a");
    pd.register_entry("b");
    pd.randomize(1.0);  // 2 mutations

    const auto s = pd.stats();
    REQUIRE(s.total_entries   == 2);
    REQUIRE(s.total_mutations == 2);
    REQUIRE(s.mean_mutations_per_entry == Approx(1.0));
    REQUIRE(s.continuous_active == false);
}

TEST_CASE("PolymorphicDefense::stats — empty stats",
          "[Phase128][stats]") {
    PolymorphicDefense pd;
    const auto s = pd.stats();
    REQUIRE(s.total_entries            == 0);
    REQUIRE(s.total_mutations          == 0);
    REQUIRE(s.mean_mutations_per_entry == Approx(0.0));
}

// ---------------------------------------------------------------------------
// Eviction at POLY_MAX_ENTRIES
// ---------------------------------------------------------------------------

TEST_CASE("PolymorphicDefense — evicts oldest at POLY_MAX_ENTRIES",
          "[Phase128][evict]") {
    PolymorphicDefense pd;

    for (size_t i = 0; i < POLY_MAX_ENTRIES; ++i) {
        pd.register_entry("e" + std::to_string(i));
    }
    REQUIRE(pd.entry_count() == POLY_MAX_ENTRIES);
    REQUIRE(pd.current_token("e0") != 0u);   // first entry present

    pd.register_entry("newcomer");
    REQUIRE(pd.entry_count() == POLY_MAX_ENTRIES);
    REQUIRE(pd.current_token("e0") == 0u);      // evicted
    REQUIRE(pd.current_token("newcomer") != 0u); // new entry present
}
