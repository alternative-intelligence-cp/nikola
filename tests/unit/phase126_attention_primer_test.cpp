/**
 * @file phase126_attention_primer_test.cpp
 * @brief Phase 126 — AttentionPrimer unit tests
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/attention_primer.hpp>
#include <nikola/autonomy/decision_loop.hpp>

using namespace nikola::cognitive;
using Catch::Approx;

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

TEST_CASE("AttentionPrimer::topic_overlap — identical", "[Phase126][static]") {
    REQUIRE(AttentionPrimer::topic_overlap("explore novel ideas", "explore novel ideas")
            == Approx(1.0));
}

TEST_CASE("AttentionPrimer::topic_overlap — disjoint", "[Phase126][static]") {
    REQUIRE(AttentionPrimer::topic_overlap("cat dog", "fish bird") == Approx(0.0));
}

TEST_CASE("AttentionPrimer::topic_overlap — partial", "[Phase126][static]") {
    // "a b c" vs "b c d" → inter=2, union=4 → 0.5
    REQUIRE(AttentionPrimer::topic_overlap("a b c", "b c d") == Approx(0.5));
}

TEST_CASE("AttentionPrimer::topic_overlap — case-insensitive", "[Phase126][static]") {
    REQUIRE(AttentionPrimer::topic_overlap("HELLO WORLD", "hello world") == Approx(1.0));
}

TEST_CASE("AttentionPrimer::topic_overlap — empty strings", "[Phase126][static]") {
    REQUIRE(AttentionPrimer::topic_overlap("", "") == Approx(1.0));
    REQUIRE(AttentionPrimer::topic_overlap("x", "") == Approx(0.0));
    REQUIRE(AttentionPrimer::topic_overlap("", "y") == Approx(0.0));
}

TEST_CASE("AttentionPrimer::merged_activation — additive clamped", "[Phase126][static]") {
    REQUIRE(AttentionPrimer::merged_activation(0.4, 0.3) == Approx(0.7));
    REQUIRE(AttentionPrimer::merged_activation(0.8, 0.6) == Approx(1.0));  // clamped
}

// ---------------------------------------------------------------------------
// prime / weight_of / is_primed
// ---------------------------------------------------------------------------

TEST_CASE("AttentionPrimer::prime — new topic added", "[Phase126][prime]") {
    AttentionPrimer ap;
    REQUIRE(ap.topic_count() == 0);

    ap.prime("reward", 0.7);

    REQUIRE(ap.topic_count() == 1);
    REQUIRE(ap.weight_of("reward") == Approx(0.7));
}

TEST_CASE("AttentionPrimer::prime — boosts existing topic", "[Phase126][prime]") {
    AttentionPrimer ap;
    ap.prime("explore", 0.4);
    ap.prime("explore", 0.3);

    // merged = 0.4 + 0.3 = 0.7
    REQUIRE(ap.weight_of("explore") == Approx(0.7));
    REQUIRE(ap.topic_count() == 1);  // still one entry
}

TEST_CASE("AttentionPrimer::prime — activation clamped at 1.0", "[Phase126][prime]") {
    AttentionPrimer ap;
    ap.prime("goal", 0.8);
    ap.prime("goal", 0.8);  // merged = 1.0 (clamped)

    REQUIRE(ap.weight_of("goal") == Approx(1.0));
}

TEST_CASE("AttentionPrimer::prime — case-insensitive dedup", "[Phase126][prime]") {
    AttentionPrimer ap;
    ap.prime("Novel", 0.5);
    ap.prime("novel", 0.2);  // same tag, different case

    REQUIRE(ap.topic_count() == 1);
    REQUIRE(ap.weight_of("novel") == Approx(0.7));
}

TEST_CASE("AttentionPrimer::prime — NikolaState context captured",
          "[Phase126][prime]") {
    AttentionPrimer ap;
    nikola::autonomy::NikolaState st;
    st.dopamine = 0.9f;
    st.entropy  = 0.2f;

    ap.prime("reward_signal", 0.6, ATTENTION_DECAY_RATE, 42, &st);

    const auto all = ap.all_primed();
    REQUIRE(all.size() == 1);
    REQUIRE(all[0].dopamine_ctx == Approx(0.9f));
    REQUIRE(all[0].prime_tick   == 42u);
}

TEST_CASE("AttentionPrimer::weight_of — returns 0 for unknown tag",
          "[Phase126][prime]") {
    AttentionPrimer ap;
    REQUIRE(ap.weight_of("nonexistent") == Approx(0.0));
}

TEST_CASE("AttentionPrimer::is_primed — basic threshold check",
          "[Phase126][prime]") {
    AttentionPrimer ap;
    ap.prime("curiosity", 0.6);

    REQUIRE(ap.is_primed("curiosity") == true);
    REQUIRE(ap.is_primed("curiosity", 0.7) == false);
    REQUIRE(ap.is_primed("other") == false);
}

// ---------------------------------------------------------------------------
// decay_all
// ---------------------------------------------------------------------------

TEST_CASE("AttentionPrimer::decay_all — reduces activation", "[Phase126][decay]") {
    AttentionPrimer ap;
    ap.prime("goal", 0.8, 0.9);
    ap.decay_all();

    REQUIRE(ap.weight_of("goal") == Approx(0.8 * 0.9));
}

TEST_CASE("AttentionPrimer::decay_all — prunes below min weight",
          "[Phase126][decay]") {
    AttentionPrimer ap;
    ap.prime("fading_topic", ATTENTION_MIN_WEIGHT * 1.01, 0.01);
    REQUIRE(ap.topic_count() == 1);

    // One decay step: activation * 0.01 << ATTENTION_MIN_WEIGHT → pruned
    ap.decay_all();
    REQUIRE(ap.topic_count() == 0);
}

TEST_CASE("AttentionPrimer::decay_all — keeps topics above threshold",
          "[Phase126][decay]") {
    AttentionPrimer ap;
    ap.prime("active", 1.0, ATTENTION_DECAY_RATE);
    ap.decay_all();
    ap.decay_all();
    ap.decay_all();

    // After 3 decays: 1.0 * 0.85^3 = 0.614 > 0.05 → still present
    REQUIRE(ap.topic_count() == 1);
    REQUIRE(ap.weight_of("active") == Approx(std::pow(ATTENTION_DECAY_RATE, 3)).margin(0.001));
}

TEST_CASE("AttentionPrimer::decay_all — per-entry decay rates",
          "[Phase126][decay]") {
    AttentionPrimer ap;
    ap.prime("fast_decay",  0.8, 0.1);  // will drop fast
    ap.prime("slow_decay",  0.8, 0.99); // will persist

    ap.decay_all();
    ap.decay_all();
    ap.decay_all();

    // fast: 0.8 * 0.1^3 = 0.0008 < ATTENTION_MIN_WEIGHT → pruned
    REQUIRE(ap.is_primed("fast_decay") == false);
    // slow: 0.8 * 0.99^3 ≈ 0.776 > 0.05 → present
    REQUIRE(ap.is_primed("slow_decay") == true);
}

// ---------------------------------------------------------------------------
// remove / clear
// ---------------------------------------------------------------------------

TEST_CASE("AttentionPrimer::remove — deletes specific tag", "[Phase126][remove]") {
    AttentionPrimer ap;
    ap.prime("a", 0.7);
    ap.prime("b", 0.6);
    REQUIRE(ap.topic_count() == 2);

    ap.remove("a");
    REQUIRE(ap.topic_count() == 1);
    REQUIRE(ap.weight_of("a") == Approx(0.0));
    REQUIRE(ap.weight_of("b") == Approx(0.6));
}

TEST_CASE("AttentionPrimer::remove — unknown tag is no-op", "[Phase126][remove]") {
    AttentionPrimer ap;
    ap.prime("exists", 0.5);
    REQUIRE_NOTHROW(ap.remove("ghost"));
    REQUIRE(ap.topic_count() == 1);
}

TEST_CASE("AttentionPrimer::clear — empties pool", "[Phase126][remove]") {
    AttentionPrimer ap;
    ap.prime("x", 0.5);
    ap.prime("y", 0.6);
    ap.prime("z", 0.7);

    ap.clear();
    REQUIRE(ap.topic_count() == 0);
}

// ---------------------------------------------------------------------------
// most_primed / all_primed
// ---------------------------------------------------------------------------

TEST_CASE("AttentionPrimer::most_primed — returns highest activation",
          "[Phase126][query]") {
    AttentionPrimer ap;
    ap.prime("low",    0.2);
    ap.prime("mid",    0.5);
    ap.prime("high",   0.9);
    ap.prime("medium", 0.4);

    const auto top = ap.most_primed();
    REQUIRE(top.has_value() == true);
    REQUIRE(top->tag == "high");
    REQUIRE(top->activation == Approx(0.9));
}

TEST_CASE("AttentionPrimer::most_primed — empty pool returns nullopt",
          "[Phase126][query]") {
    AttentionPrimer ap;
    REQUIRE(ap.most_primed().has_value() == false);
}

TEST_CASE("AttentionPrimer::all_primed — sorted descending", "[Phase126][query]") {
    AttentionPrimer ap;
    ap.prime("c", 0.3);
    ap.prime("a", 0.9);
    ap.prime("b", 0.6);

    const auto sorted = ap.all_primed();
    REQUIRE(sorted.size() == 3);
    REQUIRE(sorted[0].activation >= sorted[1].activation);
    REQUIRE(sorted[1].activation >= sorted[2].activation);
    REQUIRE(sorted[0].tag == "a");
}

// ---------------------------------------------------------------------------
// predict_focus
// ---------------------------------------------------------------------------

TEST_CASE("AttentionPrimer::predict_focus — empty pool returns nullopt",
          "[Phase126][predict]") {
    AttentionPrimer ap;
    nikola::autonomy::NikolaState st;
    REQUIRE(ap.predict_focus(st).has_value() == false);
}

TEST_CASE("AttentionPrimer::predict_focus — dopamine boosts reward topic",
          "[Phase126][predict]") {
    AttentionPrimer ap;
    // prime a reward topic and a neutral one with equal activation
    ap.prime("goal achievement",  0.5);
    ap.prime("background noise",  0.5);

    nikola::autonomy::NikolaState st;
    st.dopamine = 0.8f;   // high dopamine
    st.boredom  = 0.0f;
    st.entropy  = 0.0f;

    const auto pred = ap.predict_focus(st);
    REQUIRE(pred.has_value() == true);
    // "goal achievement" contains "goal" → gets bonus → should win
    REQUIRE(pred->tag == "goal achievement");
}

TEST_CASE("AttentionPrimer::predict_focus — boredom boosts explore topic",
          "[Phase126][predict]") {
    AttentionPrimer ap;
    ap.prime("explore novel ideas", 0.5);
    ap.prime("routine task",        0.5);

    nikola::autonomy::NikolaState st;
    st.dopamine = 0.0f;
    st.boredom  = 0.8f;
    st.entropy  = 0.0f;

    const auto pred = ap.predict_focus(st);
    REQUIRE(pred.has_value() == true);
    REQUIRE(pred->tag == "explore novel ideas");
}

TEST_CASE("AttentionPrimer::predict_focus — entropy boosts uncertainty topic",
          "[Phase126][predict]") {
    AttentionPrimer ap;
    ap.prime("resolve conflict", 0.5);
    ap.prime("known fact",       0.5);

    nikola::autonomy::NikolaState st;
    st.dopamine = 0.0f;
    st.boredom  = 0.0f;
    st.entropy  = 0.9f;

    const auto pred = ap.predict_focus(st);
    REQUIRE(pred.has_value() == true);
    REQUIRE(pred->tag == "resolve conflict");
}

TEST_CASE("AttentionPrimer::predict_focus — neutral state: highest activation wins",
          "[Phase126][predict]") {
    AttentionPrimer ap;
    ap.prime("alpha",  0.9);
    ap.prime("beta",   0.3);
    ap.prime("gamma",  0.6);

    nikola::autonomy::NikolaState st;
    // All state fields zero/default → no bonuses
    st.dopamine = 0.0f;
    st.boredom  = 0.0f;
    st.entropy  = 0.0f;

    const auto pred = ap.predict_focus(st);
    REQUIRE(pred.has_value() == true);
    REQUIRE(pred->tag == "alpha");
}

// ---------------------------------------------------------------------------
// Callback
// ---------------------------------------------------------------------------

TEST_CASE("AttentionPrimer::on_prime — fires on new topic", "[Phase126][cb]") {
    AttentionPrimer ap;
    bool fired = false;
    std::string fired_tag;

    ap.on_prime([&](const PrimedFocus& f) {
        fired     = true;
        fired_tag = f.tag;
    });

    ap.prime("trigger", 0.7);
    REQUIRE(fired == true);
    REQUIRE(fired_tag == "trigger");
}

TEST_CASE("AttentionPrimer::on_prime — fires on boost of existing topic",
          "[Phase126][cb]") {
    AttentionPrimer ap;
    int fire_count = 0;
    ap.on_prime([&](const PrimedFocus&) { ++fire_count; });

    ap.prime("topic", 0.4);
    ap.prime("topic", 0.2);  // boost → callback fires again
    REQUIRE(fire_count == 2);
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

TEST_CASE("AttentionPrimer::stats — correct counts and mean", "[Phase126][stats]") {
    AttentionPrimer ap;
    ap.prime("a", 0.2);
    ap.prime("b", 0.6);
    ap.prime("c", 1.0);

    const auto s = ap.stats();
    REQUIRE(s.topic_count == 3);
    REQUIRE(s.mean_activation == Approx((0.2 + 0.6 + 1.0) / 3.0));
    REQUIRE(s.max_activation  == Approx(1.0));
    REQUIRE(s.min_activation  == Approx(0.2));
}

TEST_CASE("AttentionPrimer::stats — empty primer", "[Phase126][stats]") {
    AttentionPrimer ap;
    const auto s = ap.stats();
    REQUIRE(s.topic_count    == 0);
    REQUIRE(s.mean_activation == Approx(0.0));
}

// ---------------------------------------------------------------------------
// Eviction at ATTENTION_MAX_TOPICS
// ---------------------------------------------------------------------------

TEST_CASE("AttentionPrimer — evicts lowest-weight topic at cap",
          "[Phase126][evict]") {
    AttentionPrimer ap;

    // Fill to exactly MAX with activation = index/MAX (1/64 .. 64/64)
    for (size_t i = 0; i < ATTENTION_MAX_TOPICS; ++i) {
        const double w = static_cast<double>(i + 1) / ATTENTION_MAX_TOPICS;
        ap.prime("topic_" + std::to_string(i), w);
    }
    REQUIRE(ap.topic_count() == ATTENTION_MAX_TOPICS);

    // topic_0 has the lowest activation (1/64 = 0.0156, below is_primed
    // threshold — but it IS in the pool, so weight_of returns > 0)
    REQUIRE(ap.weight_of("topic_0") > 0.0);

    // Adding one more should evict topic_0
    ap.prime("newcomer", 0.5);
    REQUIRE(ap.topic_count() == ATTENTION_MAX_TOPICS);
    REQUIRE(ap.weight_of("topic_0") == Approx(0.0));  // evicted
    REQUIRE(ap.is_primed("newcomer") == true);
}
