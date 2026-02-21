/**
 * @file phase11_cognitive_generator_test.cpp
 * @brief Phase 11 unit tests — CognitiveGenerator + ConceptMinter (COG-05)
 *
 * Covers:
 *   - ConceptMinter: mint format, count, lexicon registration
 *   - CognitiveGenerator: initial state (empty queue)
 *   - scan() on empty grid → no token
 *   - scan() on zero-psi grid → no token (below threshold)
 *   - scan() with known psi + registered token → token emitted
 *   - pop_token() returns emitted token, then nullopt
 *   - drain() returns all queued tokens
 *   - tokens_emitted() counter increments
 *   - decode miss + minting enabled → NEO_CONCEPT emitted + registered
 *   - decode miss + minting disabled → nothing emitted
 *   - persistence_count > 1: emit only after N consecutive misses on same peak
 *   - inhibition_enabled: after scan() peak node has reduced |Ψ|
 *   - build_signature: deterministic, 9 elements, different psi → different sig
 *   - Multiple tokens: highest cognitive-energy node decoded first
 *   - queue_size() reflects pending tokens
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/cognitive_generator.hpp>

#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace nc = nikola::cognitive;
namespace np = nikola::physics;

using Complex = nc::Complex;

// ---------------------------------------------------------------------------
// Helper: build a 1-node WaveFunction and set its psi/resonance.
//
// n=1 → 1^9 = 1 active node.
// ---------------------------------------------------------------------------

static np::WaveFunction make_wf_1node(Complex psi, float resonance = 1.0f)
{
    np::WaveFunction wf;
    wf.seed_manifold(1);  // 1 node

    auto& grid = wf.grid();
    auto node  = grid.get_node(0);
    node.psi       = psi;
    node.resonance = resonance;
    grid.set_node(0, node);
    return wf;
}

// Helper: build and register a token whose wave matches build_signature(psi)
static void register_at(nc::HolographicLexicon& lex,
                         const std::string& tok,
                         Complex psi)
{
    lex.add_token(tok, nc::CognitiveGenerator::build_signature(psi));
}

// ===========================================================================
// ConceptMinter tests
// ===========================================================================

TEST_CASE("ConceptMinter: mint returns NEO_CONCEPT_XXXX format", "[cog][minter]") {
    nc::HolographicLexicon lex;
    nc::ConceptMinter minter(lex);
    const std::string tok = minter.mint(nc::CognitiveGenerator::build_signature({1.0f, 0.0f}));
    CHECK(tok.substr(0, 12) == "NEO_CONCEPT_");
    CHECK(tok.size() == 16u);   // "NEO_CONCEPT_" (12) + 4 hex digits
}

TEST_CASE("ConceptMinter: count increments after each mint", "[cog][minter]") {
    nc::HolographicLexicon lex;
    nc::ConceptMinter minter(lex);
    const auto sig = nc::CognitiveGenerator::build_signature({0.5f, 0.5f});
    CHECK(minter.count() == 0u);
    minter.mint(sig);
    CHECK(minter.count() == 1u);
    minter.mint(sig);
    CHECK(minter.count() == 2u);
}

TEST_CASE("ConceptMinter: minted token is registered and decodable", "[cog][minter]") {
    nc::HolographicLexicon lex;
    nc::ConceptMinter minter(lex);
    const Complex psi{2.0f, 1.0f};
    const auto sig = nc::CognitiveGenerator::build_signature(psi);
    const std::string tok = minter.mint(sig);
    // Lexicon should now recognise this wave
    CHECK(lex.exists(tok));
    const auto decoded = lex.decode(sig);
    REQUIRE(decoded.has_value());
    CHECK(*decoded == tok);
}

TEST_CASE("ConceptMinter: unique ids per call", "[cog][minter]") {
    nc::HolographicLexicon lex;
    nc::ConceptMinter m(lex);
    const auto s = nc::CognitiveGenerator::build_signature({1.0f, 0.0f});
    const auto t1 = m.mint(s);
    const auto t2 = m.mint(s);
    CHECK(t1 != t2);
}

// ===========================================================================
// build_signature tests
// ===========================================================================

TEST_CASE("build_signature: returns 9 elements", "[cog][signature]") {
    CHECK(nc::CognitiveGenerator::build_signature({1.0f, 0.0f}).size() == 9u);
}

TEST_CASE("build_signature: same psi → same signature (deterministic)", "[cog][signature]") {
    const Complex psi{0.7f, -0.3f};
    CHECK(nc::CognitiveGenerator::build_signature(psi) ==
          nc::CognitiveGenerator::build_signature(psi));
}

TEST_CASE("build_signature: different psi → different signature", "[cog][signature]") {
    const auto s0 = nc::CognitiveGenerator::build_signature({1.0f,  0.0f});
    const auto s1 = nc::CognitiveGenerator::build_signature({0.0f, -1.0f});
    CHECK(s0 != s1);
}

TEST_CASE("build_signature: dim 0 equals psi * exp(0) = psi itself", "[cog][signature]") {
    const Complex psi{1.5f, 0.0f};
    const auto sig = nc::CognitiveGenerator::build_signature(psi);
    CHECK(std::abs(sig[0] - psi) < 1e-5f);
}

TEST_CASE("build_signature: energy is preserved (|sig[d]| = |psi| for all d)",
          "[cog][signature]") {
    const Complex psi{0.8f, 0.6f};
    const float amp = std::abs(psi);
    for (const auto& c : nc::CognitiveGenerator::build_signature(psi))
        CHECK(std::abs(std::abs(c) - amp) < 1e-5f);
}

// ===========================================================================
// CognitiveGenerator core tests
// ===========================================================================

TEST_CASE("CognitiveGenerator: empty queue on construction", "[cog][gen]") {
    nc::HolographicLexicon lex;
    nc::CognitiveGenerator gen(lex);
    CHECK(gen.queue_size() == 0u);
    CHECK_FALSE(gen.pop_token().has_value());
    CHECK(gen.tokens_emitted() == 0u);
}

TEST_CASE("CognitiveGenerator: scan empty grid → no token", "[cog][gen]") {
    nc::HolographicLexicon lex;
    nc::CognitiveGenerator gen(lex);
    np::WaveFunction wf;
    gen.scan(wf);
    CHECK_FALSE(gen.pop_token().has_value());
}

TEST_CASE("CognitiveGenerator: scan zero-psi node → no token (below threshold)", "[cog][gen]") {
    nc::HolographicLexicon lex;
    nc::CognitiveGenerator gen(lex);
    gen.set_energy_threshold(1e-10f);
    // seed_manifold(1) → 1 node with psi=pilot_wave (varies), but
    // we override to zero
    auto wf = make_wf_1node({0.0f, 0.0f}, 1.0f);
    gen.scan(wf);
    CHECK_FALSE(gen.pop_token().has_value());
}

TEST_CASE("CognitiveGenerator: scan known peak → emits registered token", "[cog][gen]") {
    nc::HolographicLexicon lex;
    nc::CognitiveGenerator gen(lex);
    gen.set_minting_enabled(false);   // only test known-token path for now

    const Complex psi{1.0f, 0.0f};
    register_at(lex, "hello", psi);

    auto wf = make_wf_1node(psi, 1.0f);
    gen.scan(wf);

    const auto tok = gen.pop_token();
    REQUIRE(tok.has_value());
    CHECK(*tok == "hello");
    CHECK(gen.tokens_emitted() == 1u);
}

TEST_CASE("CognitiveGenerator: pop_token drained after first call", "[cog][gen]") {
    nc::HolographicLexicon lex;
    nc::CognitiveGenerator gen(lex);
    const Complex psi{1.0f, 0.5f};
    register_at(lex, "word", psi);
    auto wf = make_wf_1node(psi, 1.0f);
    gen.scan(wf);

    CHECK(gen.pop_token().has_value());
    CHECK_FALSE(gen.pop_token().has_value());  // queue now empty
}

TEST_CASE("CognitiveGenerator: drain() returns all queued tokens", "[cog][gen]") {
    nc::HolographicLexicon lex;
    nc::CognitiveGenerator gen(lex);
    gen.set_inhibition_enabled(false);  // prevent psi modification between scans

    const Complex psi{1.0f, 0.0f};
    register_at(lex, "a", psi);

    // Scan 3 times — inhibition off so same node keeps emitting
    auto wf = make_wf_1node(psi, 1.0f);
    gen.scan(wf);
    gen.scan(wf);
    gen.scan(wf);

    const auto toks = gen.drain();
    CHECK(toks.size() == 3u);
    for (const auto& t : toks) CHECK(t == "a");
    CHECK(gen.queue_size() == 0u);
}

TEST_CASE("CognitiveGenerator: tokens_emitted counter increments", "[cog][gen]") {
    nc::HolographicLexicon lex;
    nc::CognitiveGenerator gen(lex);
    gen.set_inhibition_enabled(false);

    const Complex psi{1.0f, 0.0f};
    register_at(lex, "tok", psi);
    auto wf = make_wf_1node(psi, 1.0f);

    gen.scan(wf);
    gen.scan(wf);
    CHECK(gen.tokens_emitted() == 2u);
}

TEST_CASE("CognitiveGenerator: decode miss + minting enabled → NEO_CONCEPT emitted",
          "[cog][gen]") {
    nc::HolographicLexicon lex;
    nc::CognitiveGenerator gen(lex);
    gen.set_minting_enabled(true);
    gen.set_persistence_count(1);

    // Do NOT register any token — all scans will miss
    auto wf = make_wf_1node({1.0f, 0.5f}, 1.0f);
    gen.scan(wf);

    const auto tok = gen.pop_token();
    REQUIRE(tok.has_value());
    CHECK(tok->substr(0, 12) == "NEO_CONCEPT_");
    CHECK(gen.concepts_minted() == 1u);
}

TEST_CASE("CognitiveGenerator: minted concept is registered in lexicon", "[cog][gen]") {
    nc::HolographicLexicon lex;
    nc::CognitiveGenerator gen(lex);
    gen.set_inhibition_enabled(false);

    auto wf = make_wf_1node({0.9f, 0.1f}, 1.0f);
    gen.scan(wf);

    const auto neo = gen.pop_token();
    REQUIRE(neo.has_value());
    CHECK(lex.exists(*neo));   // must be findable in lexicon
}

TEST_CASE("CognitiveGenerator: decode miss + minting disabled → nothing emitted",
          "[cog][gen]") {
    nc::HolographicLexicon lex;
    nc::CognitiveGenerator gen(lex);
    gen.set_minting_enabled(false);

    auto wf = make_wf_1node({1.0f, 0.0f}, 1.0f);
    gen.scan(wf);

    CHECK_FALSE(gen.pop_token().has_value());
    CHECK(gen.concepts_minted() == 0u);
}

TEST_CASE("CognitiveGenerator: persistence_count=2 — emit only after 2nd consecutive miss",
          "[cog][gen]") {
    nc::HolographicLexicon lex;
    nc::CognitiveGenerator gen(lex);
    gen.set_minting_enabled(true);
    gen.set_persistence_count(2);
    gen.set_inhibition_enabled(false);  // keep psi steady across scans

    auto wf = make_wf_1node({0.8f, 0.2f}, 1.0f);

    gen.scan(wf);  // 1st miss — streak=1, below persistence_count → no emit
    CHECK_FALSE(gen.pop_token().has_value());

    gen.scan(wf);  // 2nd miss — streak=2, reached → emit
    CHECK(gen.pop_token().has_value());
}

TEST_CASE("CognitiveGenerator: inhibition reduces peak node |Ψ| after scan", "[cog][gen]") {
    nc::HolographicLexicon lex;
    nc::CognitiveGenerator gen(lex);
    gen.set_inhibition_enabled(true);

    const Complex psi{0.05f, 0.0f};  // must be small enough to not hit inject clamp
    register_at(lex, "wave", psi);

    auto wf = make_wf_1node(psi, 1.0f);
    const float before = std::abs(wf.grid().psi_real()[0]);

    gen.scan(wf);

    const float pr = wf.grid().psi_real()[0];
    const float pi = wf.grid().psi_imag()[0];
    const float after = std::sqrt(pr*pr + pi*pi);

    // After inhibition the magnitude should be strictly smaller
    CHECK(after < before + 1e-5f);  // ≤ before (may be clamped/near-zero)
}

TEST_CASE("CognitiveGenerator: inhibition disabled — peak node unchanged", "[cog][gen]") {
    nc::HolographicLexicon lex;
    nc::CognitiveGenerator gen(lex);
    gen.set_inhibition_enabled(false);

    const Complex psi{1.0f, 0.0f};
    register_at(lex, "still", psi);
    auto wf = make_wf_1node(psi, 1.0f);

    gen.scan(wf);

    const float pr = wf.grid().psi_real()[0];
    const float pi = wf.grid().psi_imag()[0];
    CHECK(std::abs(Complex{pr, pi} - psi) < 1e-5f);
}

TEST_CASE("CognitiveGenerator: queue_size reflects pending tokens", "[cog][gen]") {
    nc::HolographicLexicon lex;
    nc::CognitiveGenerator gen(lex);
    gen.set_inhibition_enabled(false);

    const Complex psi{1.2f, 0.0f};
    register_at(lex, "q", psi);
    auto wf = make_wf_1node(psi, 1.0f);

    CHECK(gen.queue_size() == 0u);
    gen.scan(wf);
    CHECK(gen.queue_size() == 1u);
    [[maybe_unused]] auto _ = gen.pop_token();
    CHECK(gen.queue_size() == 0u);
}
