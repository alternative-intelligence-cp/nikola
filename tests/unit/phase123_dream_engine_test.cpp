/**
 * @file phase123_dream_engine_test.cpp
 * @brief Phase 123 — DreamEngine unit tests
 *
 * §1  Constants
 * §2  is_nightmare_state()
 * §3  is_idle_enough()
 * §4  state_similarity()
 * §5  compute_novelty()
 * §6  generate_insight()
 * §7  record_experience() — buffer management
 * §8  dream() — fragment discovery and consolidation
 * §9  dream() — callback
 * §10 nightmare_count() and process_nightmares()
 * §11 recall()
 * §12 stats()
 * §13 dream_log() accumulation
 * §14 FIFO eviction at DREAM_BUFFER_SIZE
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/interior/dream_engine.hpp>
#include <nikola/autonomy/decision_loop.hpp>

using namespace nikola::interior;
using namespace nikola::autonomy;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static NikolaState make_state(float dopamine    = 0.5f,
                               float atp        = 0.7f,
                               float entropy    = 0.8f,
                               float torus_e    = 0.5f,
                               float boredom    = 0.3f) {
    NikolaState s;
    s.dopamine     = dopamine;
    s.atp          = atp;
    s.entropy      = entropy;
    s.torus_energy = torus_e;
    s.boredom      = boredom;
    return s;
}

/// State that triggers nightmare classification
static NikolaState nightmare_state() {
    return make_state(0.1f /*dopamine*/, 0.3f /*atp*/,
                      1.6f /*entropy — above 1.40*/,
                      0.2f, 0.2f);
}

/// State that is clearly safe (normal operation)
static NikolaState normal_state() {
    return make_state(0.7f, 0.8f, 0.5f, 0.5f, 0.3f);
}

/// Two states that are effectively identical
static NikolaState identical_state() { return make_state(0.6f, 0.6f, 0.6f, 0.6f, 0.6f); }

// ---------------------------------------------------------------------------
// §1 — Constants
// ---------------------------------------------------------------------------

TEST_CASE("Phase123 §1 constants", "[Phase123]") {
    CHECK(DREAM_IDLE_THRESHOLD       > 0.0);
    CHECK(DREAM_IDLE_THRESHOLD       < 1.0);
    CHECK(DREAM_SIMILARITY_THRESHOLD > 0.0);
    CHECK(DREAM_SIMILARITY_THRESHOLD < 1.0);
    CHECK(DREAM_NIGHTMARE_ENTROPY    > 0.0f);
    CHECK(DREAM_NIGHTMARE_DOPAMINE   > 0.0f);
    CHECK(DREAM_NIGHTMARE_DOPAMINE   < 1.0f);
    CHECK(DREAM_CONSOLIDATION_MIN    > 0.0);
    CHECK(DREAM_CONSOLIDATION_MIN    < 1.0);
    CHECK(DREAM_BUFFER_SIZE          > 0u);
    CHECK(DREAM_MAX_RECALL           > 0u);
}

// ---------------------------------------------------------------------------
// §2 — is_nightmare_state
// ---------------------------------------------------------------------------

TEST_CASE("Phase123 §2 is_nightmare_state()", "[Phase123]") {
    SECTION("high entropy + low dopamine -> true") {
        NikolaState s = make_state(0.1f, 0.4f, 1.6f);
        CHECK(DreamEngine::is_nightmare_state(s));
    }

    SECTION("high entropy only (dopamine normal) -> false") {
        NikolaState s = make_state(0.6f, 0.7f, 1.6f);
        CHECK(!DreamEngine::is_nightmare_state(s));
    }

    SECTION("low dopamine only (entropy normal) -> false") {
        NikolaState s = make_state(0.1f, 0.7f, 0.5f);
        CHECK(!DreamEngine::is_nightmare_state(s));
    }

    SECTION("normal state -> false") {
        CHECK(!DreamEngine::is_nightmare_state(normal_state()));
    }

    SECTION("boundary: exactly at thresholds -> false (exclusive)") {
        NikolaState s = make_state(
            DREAM_NIGHTMARE_DOPAMINE,        // == not <
            0.4f,
            DREAM_NIGHTMARE_ENTROPY,         // == not >
            0.5f);
        CHECK(!DreamEngine::is_nightmare_state(s));
    }
}

// ---------------------------------------------------------------------------
// §3 — is_idle_enough
// ---------------------------------------------------------------------------

TEST_CASE("Phase123 §3 is_idle_enough()", "[Phase123]") {
    SECTION("boredom >= DREAM_IDLE_THRESHOLD -> true") {
        NikolaState s = make_state();
        s.boredom = static_cast<float>(DREAM_IDLE_THRESHOLD);
        CHECK(DreamEngine::is_idle_enough(s));
    }

    SECTION("boredom above threshold -> true") {
        NikolaState s = make_state();
        s.boredom = 0.9f;
        CHECK(DreamEngine::is_idle_enough(s));
    }

    SECTION("boredom below threshold -> false") {
        NikolaState s = make_state();
        s.boredom = 0.1f;
        CHECK(!DreamEngine::is_idle_enough(s));
    }
}

// ---------------------------------------------------------------------------
// §4 — state_similarity
// ---------------------------------------------------------------------------

TEST_CASE("Phase123 §4 state_similarity()", "[Phase123]") {
    SECTION("identical states -> 1.0") {
        auto a = identical_state();
        CHECK_THAT(DreamEngine::state_similarity(a, a), WithinAbs(1.0, 1e-9));
    }

    SECTION("result always in [0, 1]") {
        auto a = make_state(0.0f, 0.0f, 0.0f, 0.0f);
        auto b = make_state(1.0f, 1.0f, 2.0f, 1.0f);
        double sim = DreamEngine::state_similarity(a, b);
        CHECK(sim >= 0.0);
        CHECK(sim <= 1.0);
    }

    SECTION("symmetric: sim(a,b) == sim(b,a)") {
        auto a = make_state(0.2f, 0.8f, 1.0f, 0.3f);
        auto b = make_state(0.7f, 0.4f, 0.3f, 0.8f);
        double ab = DreamEngine::state_similarity(a, b);
        double ba = DreamEngine::state_similarity(b, a);
        CHECK_THAT(ab, WithinAbs(ba, 1e-9));
    }

    SECTION("more similar pair scores higher than dissimilar pair") {
        auto ref  = make_state(0.5f, 0.5f, 0.5f, 0.5f);
        auto near = make_state(0.6f, 0.5f, 0.5f, 0.5f); // small delta
        auto far  = make_state(0.0f, 1.0f, 2.0f, 0.0f); // large delta
        double sim_near = DreamEngine::state_similarity(ref, near);
        double sim_far  = DreamEngine::state_similarity(ref, far);
        CHECK(sim_near > sim_far);
    }
}

// ---------------------------------------------------------------------------
// §5 — compute_novelty
// ---------------------------------------------------------------------------

TEST_CASE("Phase123 §5 compute_novelty()", "[Phase123]") {
    SECTION("always in [0, 1]") {
        double n = DreamEngine::compute_novelty(0.6, 0.4);
        CHECK(n >= 0.0);
        CHECK(n <= 1.0);
    }

    SECTION("low similarity + low mean -> moderate novelty") {
        double n = DreamEngine::compute_novelty(0.2, 0.2);
        CHECK(n > 0.0);
    }

    SECTION("high similarity (boring pair) -> low novelty") {
        double n_high = DreamEngine::compute_novelty(0.95, 0.5);
        double n_low  = DreamEngine::compute_novelty(0.30, 0.5);
        CHECK(n_high < n_low);
    }

    SECTION("boundary: similarity=1.0 -> novelty=0") {
        CHECK_THAT(DreamEngine::compute_novelty(1.0, 0.5), WithinAbs(0.0, 1e-9));
    }
}

// ---------------------------------------------------------------------------
// §6 — generate_insight
// ---------------------------------------------------------------------------

TEST_CASE("Phase123 §6 generate_insight()", "[Phase123]") {
    NikolaState sa = normal_state();
    NikolaState sb = nightmare_state();
    Experience a; a.tag = "reward_spike"; a.state = sa; a.tick = 1;
    Experience b; b.tag = "error_burst";  b.state = sb; b.tick = 5;

    SECTION("returns non-empty string") {
        std::string insight = DreamEngine::generate_insight(a, b, 0.7);
        CHECK(!insight.empty());
    }

    SECTION("contains both tag strings") {
        std::string insight = DreamEngine::generate_insight(a, b, 0.7);
        CHECK(insight.find("reward_spike") != std::string::npos);
        CHECK(insight.find("error_burst")  != std::string::npos);
    }

    SECTION("contains similarity percentage") {
        std::string insight = DreamEngine::generate_insight(a, b, 0.75);
        // 0.75 * 100 = 75%
        CHECK(insight.find("75") != std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// §7 — record_experience
// ---------------------------------------------------------------------------

TEST_CASE("Phase123 §7 record_experience()", "[Phase123]") {
    SECTION("increments count") {
        DreamEngine d;
        d.record_experience("test", normal_state());
        CHECK(d.experience_count() == 1);
        d.record_experience("test2", normal_state());
        CHECK(d.experience_count() == 2);
    }

    SECTION("nightmare auto-set for nightmare state") {
        DreamEngine d;
        d.record_experience("crash", nightmare_state());
        REQUIRE(d.experience_count() == 1);
        CHECK(d.experiences().back().is_nightmare == true);
    }

    SECTION("not a nightmare for normal state") {
        DreamEngine d;
        d.record_experience("normal", normal_state());
        REQUIRE(d.experience_count() == 1);
        CHECK(d.experiences().back().is_nightmare == false);
    }

    SECTION("tag stored correctly") {
        DreamEngine d;
        d.record_experience("my_event", normal_state(), 0.5f);
        CHECK(d.experiences().back().tag == "my_event");
    }

    SECTION("reward_signal stored") {
        DreamEngine d;
        d.record_experience("r", normal_state(), 1.5f);
        CHECK_THAT(d.experiences().back().reward_signal, WithinAbs(1.5f, 1e-5f));
    }
}

// ---------------------------------------------------------------------------
// §8 — dream() fragment discovery and consolidation
// ---------------------------------------------------------------------------

TEST_CASE("Phase123 §8 dream()", "[Phase123]") {
    SECTION("empty buffer -> DreamCycle with zero fragments") {
        DreamEngine d;
        auto cycle = d.dream(1);
        CHECK(cycle.fragments_found == 0);
        CHECK(cycle.memories_formed == 0);
    }

    SECTION("single experience -> zero fragments") {
        DreamEngine d;
        d.record_experience("solo", normal_state());
        auto cycle = d.dream(1);
        CHECK(cycle.fragments_found == 0);
    }

    SECTION("two identical-state experiences -> at least one fragment") {
        DreamEngine d;
        auto s = identical_state();
        d.record_experience("a", s);
        d.record_experience("b", s);
        auto cycle = d.dream(1);
        CHECK(cycle.fragments_found >= 1);
    }

    SECTION("two maximally dissimilar states -> zero fragments") {
        DreamEngine d;
        d.record_experience("low",  make_state(0.0f, 0.0f, 0.0f, 0.0f));
        d.record_experience("high", make_state(1.0f, 1.0f, 2.0f, 1.0f));
        auto cycle = d.dream(1);
        // similarity should be below threshold; no fragments expected
        CHECK(cycle.fragments_found == 0);
    }

    SECTION("start_tick and end_tick recorded") {
        DreamEngine d;
        auto cycle = d.dream(42);
        CHECK(cycle.start_tick == 42);
        CHECK(cycle.end_tick   == 42);
    }

    SECTION("nightmare count in cycle matches buffer nightmares") {
        DreamEngine d;
        d.record_experience("nm1", nightmare_state());
        d.record_experience("nm2", nightmare_state());
        d.record_experience("ok",  normal_state());
        auto cycle = d.dream(1);
        CHECK(cycle.nightmares_processed == 2);
    }
}

// ---------------------------------------------------------------------------
// §9 — dream() callback
// ---------------------------------------------------------------------------

TEST_CASE("Phase123 §9 dream callback", "[Phase123]") {
    SECTION("callback fires after dream()") {
        DreamEngine d;
        bool fired = false;
        DreamCycle captured;
        d.on_dream_complete([&](const DreamCycle& c) {
            fired    = true;
            captured = c;
        });
        d.record_experience("x", normal_state());
        d.dream(55);
        CHECK(fired);
        CHECK(captured.start_tick == 55);
    }

    SECTION("callback not set -> dream() does not crash") {
        DreamEngine d;
        d.record_experience("x", normal_state());
        REQUIRE_NOTHROW(d.dream(1));
    }
}

// ---------------------------------------------------------------------------
// §10 — nightmare_count and process_nightmares
// ---------------------------------------------------------------------------

TEST_CASE("Phase123 §10 nightmares", "[Phase123]") {
    SECTION("nightmare_count == 0 with no nightmare experiences") {
        DreamEngine d;
        d.record_experience("ok", normal_state());
        CHECK(d.nightmare_count() == 0);
    }

    SECTION("nightmare_count tracks nightmare experiences") {
        DreamEngine d;
        d.record_experience("nm", nightmare_state());
        d.record_experience("ok", normal_state());
        d.record_experience("nm2", nightmare_state());
        CHECK(d.nightmare_count() == 2);
    }

    SECTION("process_nightmares returns empty when none") {
        DreamEngine d;
        d.record_experience("fine", normal_state());
        CHECK(d.process_nightmares().empty());
    }

    SECTION("process_nightmares returns at least one pattern when nightmares present") {
        DreamEngine d;
        d.record_experience("crash", nightmare_state());
        d.record_experience("crash2", nightmare_state());
        auto patterns = d.process_nightmares();
        CHECK(!patterns.empty());
    }

    SECTION("each pattern is non-empty string") {
        DreamEngine d;
        d.record_experience("nm", nightmare_state());
        for (const auto& p : d.process_nightmares()) {
            CHECK(!p.empty());
        }
    }
}

// ---------------------------------------------------------------------------
// §11 — recall
// ---------------------------------------------------------------------------

TEST_CASE("Phase123 §11 recall()", "[Phase123]") {
    SECTION("no memories -> empty result") {
        DreamEngine d;
        CHECK(d.recall("anything").empty());
    }

    SECTION("recall with empty query -> empty result") {
        DreamEngine d;
        auto s = identical_state();
        d.record_experience("reward", s);
        d.record_experience("reward2", s);
        d.dream(1);
        CHECK(d.recall("").empty());
    }

    SECTION("max limits result count") {
        DreamEngine d;
        // Create many similar experiences so many memories consolidate
        auto s = identical_state();
        for (int i = 0; i < 10; ++i)
            d.record_experience("event_" + std::to_string(i), s);
        d.dream(1);
        auto res = d.recall("event", 3);
        CHECK(res.size() <= 3);
    }

    SECTION("returned pointers are non-null") {
        DreamEngine d;
        auto s = identical_state();
        d.record_experience("alpha", s);
        d.record_experience("beta",  s);
        d.dream(1);
        if (!d.memories().empty()) {
            auto res = d.recall("alpha beta");
            for (const auto* m : res) CHECK(m != nullptr);
        }
    }
}

// ---------------------------------------------------------------------------
// §12 — stats
// ---------------------------------------------------------------------------

TEST_CASE("Phase123 §12 stats()", "[Phase123]") {
    SECTION("fresh instance -> all zeros") {
        DreamEngine d;
        auto s = d.stats();
        CHECK(s.total_experiences      == 0);
        CHECK(s.total_nightmares       == 0);
        CHECK(s.total_fragments        == 0);
        CHECK(s.total_memories         == 0);
        CHECK(s.total_dream_cycles     == 0);
        CHECK_THAT(s.mean_memory_confidence, WithinAbs(0.0, 1e-9));
    }

    SECTION("stats reflect recorded experiences") {
        DreamEngine d;
        d.record_experience("ok",  normal_state());
        d.record_experience("nm",  nightmare_state());
        auto s = d.stats();
        CHECK(s.total_experiences == 2);
        CHECK(s.total_nightmares  == 1);
    }

    SECTION("stats reflect dream cycle") {
        DreamEngine d;
        d.record_experience("a", identical_state());
        d.record_experience("b", identical_state());
        d.dream(1);
        auto s = d.stats();
        CHECK(s.total_dream_cycles >= 1);
    }
}

// ---------------------------------------------------------------------------
// §13 — dream_log accumulation
// ---------------------------------------------------------------------------

TEST_CASE("Phase123 §13 dream_log", "[Phase123]") {
    SECTION("each dream() call adds one entry to dream_log") {
        DreamEngine d;
        d.record_experience("x", normal_state());
        d.dream(10);
        d.dream(20);
        d.dream(30);
        CHECK(d.dream_log().size() == 3);
    }

    SECTION("dream_log entries have correct ticks") {
        DreamEngine d;
        d.dream(100);
        d.dream(200);
        CHECK(d.dream_log()[0].start_tick == 100);
        CHECK(d.dream_log()[1].start_tick == 200);
    }
}

// ---------------------------------------------------------------------------
// §14 — FIFO eviction
// ---------------------------------------------------------------------------

TEST_CASE("Phase123 §14 FIFO eviction at DREAM_BUFFER_SIZE", "[Phase123]") {
    SECTION("count never exceeds DREAM_BUFFER_SIZE") {
        DreamEngine d;
        for (size_t i = 0; i < DREAM_BUFFER_SIZE + 10; ++i)
            d.record_experience("e", normal_state());
        CHECK(d.experience_count() <= DREAM_BUFFER_SIZE);
    }

    SECTION("oldest entries are evicted first") {
        DreamEngine d;
        // Fill to capacity
        for (size_t i = 0; i < DREAM_BUFFER_SIZE; ++i)
            d.record_experience("old_" + std::to_string(i), normal_state());
        // Add one more
        d.record_experience("newest", normal_state());
        CHECK(d.experience_count() <= DREAM_BUFFER_SIZE);
        // The last element should be the newest one
        CHECK(d.experiences().back().tag == "newest");
    }
}
