/**
 * @file phase125_scratchpad_test.cpp
 * @brief Phase 125 — Scratchpad unit tests
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/cognitive/scratchpad.hpp>
#include <nikola/autonomy/decision_loop.hpp>

using namespace nikola::cognitive;
using Catch::Approx;

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

TEST_CASE("Scratchpad::word_overlap — identical text", "[Phase125][static]") {
    const double score = Scratchpad::word_overlap("the quick brown fox",
                                                   "the quick brown fox");
    REQUIRE(score == Approx(1.0));
}

TEST_CASE("Scratchpad::word_overlap — no shared words", "[Phase125][static]") {
    const double score = Scratchpad::word_overlap("cat dog", "fish bird");
    REQUIRE(score == Approx(0.0));
}

TEST_CASE("Scratchpad::word_overlap — partial overlap", "[Phase125][static]") {
    // "a b c" vs "b c d" → intersection {b,c}=2, union {a,b,c,d}=4 → 0.5
    const double score = Scratchpad::word_overlap("a b c", "b c d");
    REQUIRE(score == Approx(0.5));
}

TEST_CASE("Scratchpad::word_overlap — case-insensitive", "[Phase125][static]") {
    const double score = Scratchpad::word_overlap("Hello World", "hello world");
    REQUIRE(score == Approx(1.0));
}

TEST_CASE("Scratchpad::word_overlap — empty strings", "[Phase125][static]") {
    REQUIRE(Scratchpad::word_overlap("", "") == Approx(1.0));
    REQUIRE(Scratchpad::word_overlap("hello", "") == Approx(0.0));
    REQUIRE(Scratchpad::word_overlap("", "world") == Approx(0.0));
}

TEST_CASE("Scratchpad::score_against_pool — empty pool", "[Phase125][static]") {
    std::vector<CommittedEntry> pool;
    REQUIRE(Scratchpad::score_against_pool("anything", pool) == Approx(0.0));
}

TEST_CASE("Scratchpad::score_against_pool — best match selected",
          "[Phase125][static]") {
    std::vector<CommittedEntry> pool;

    CommittedEntry e1;
    e1.id = 1; e1.text = "dog cat"; e1.confidence = 1.0;
    pool.push_back(e1);

    CommittedEntry e2;
    e2.id = 2; e2.text = "the quick brown fox"; e2.confidence = 1.0;
    pool.push_back(e2);

    // "the quick brown dog" overlaps more with e2 than e1
    const double s = Scratchpad::score_against_pool("the quick brown dog", pool);
    REQUIRE(s > 0.5);

    // "dog cat bird" overlaps best with e1
    const double s2 = Scratchpad::score_against_pool("dog cat bird", pool);
    // e1: overlap("dog cat bird","dog cat") = 2/3 = 0.667; e2 = 0 → best is e1
    REQUIRE(s2 > 0.0);
}

// ---------------------------------------------------------------------------
// Committed pool
// ---------------------------------------------------------------------------

TEST_CASE("Scratchpad::commit — basic", "[Phase125][committed]") {
    Scratchpad sp;
    REQUIRE(sp.committed_count() == 0);

    sp.commit("the sky is blue", 1.0);
    sp.commit("water is wet", 0.9);

    REQUIRE(sp.committed_count() == 2);
    REQUIRE(sp.committed()[0].text == "the sky is blue");
    REQUIRE(sp.committed()[1].text == "water is wet");
}

TEST_CASE("Scratchpad::commit — confidence clamped", "[Phase125][committed]") {
    Scratchpad sp;
    sp.commit("fact", 2.5);
    REQUIRE(sp.committed()[0].confidence == Approx(1.0));

    sp.commit("another", -0.5);
    REQUIRE(sp.committed()[1].confidence == Approx(0.0));
}

// ---------------------------------------------------------------------------
// inject
// ---------------------------------------------------------------------------

TEST_CASE("Scratchpad::inject — assigns ascending ids", "[Phase125][inject]") {
    Scratchpad sp;
    const uint64_t id1 = sp.inject("hypothesis one");
    const uint64_t id2 = sp.inject("hypothesis two");
    const uint64_t id3 = sp.inject("hypothesis three");

    REQUIRE(id2 > id1);
    REQUIRE(id3 > id2);
}

TEST_CASE("Scratchpad::inject — new entries are PENDING", "[Phase125][inject]") {
    Scratchpad sp;
    const uint64_t id = sp.inject("some hypothesis", 0.7);

    const HypothesisEntry* e = sp.find(id);
    REQUIRE(e != nullptr);
    REQUIRE(e->status == HypothesisStatus::PENDING);
    REQUIRE(e->confidence == Approx(0.7));
}

TEST_CASE("Scratchpad::inject — NikolaState context captured",
          "[Phase125][inject]") {
    Scratchpad sp;

    nikola::autonomy::NikolaState st;
    st.dopamine = 0.8f;
    st.entropy  = 0.3f;

    const uint64_t id = sp.inject("state-bound hypothesis", 0.6, &st);
    const HypothesisEntry* e = sp.find(id);
    REQUIRE(e != nullptr);
    REQUIRE(e->dopamine_ctx == Approx(0.8f));
    REQUIRE(e->entropy_ctx  == Approx(0.3f));
}

// ---------------------------------------------------------------------------
// measure_resonance
// ---------------------------------------------------------------------------

TEST_CASE("Scratchpad::measure_resonance — empty pool returns 0",
          "[Phase125][resonance]") {
    Scratchpad sp;
    const uint64_t id = sp.inject("some hypothesis");
    REQUIRE(sp.measure_resonance(id) == Approx(0.0));
}

TEST_CASE("Scratchpad::measure_resonance — unknown id returns 0",
          "[Phase125][resonance]") {
    Scratchpad sp;
    sp.commit("fact");
    REQUIRE(sp.measure_resonance(999) == Approx(0.0));
}

TEST_CASE("Scratchpad::measure_resonance — scores against committed pool",
          "[Phase125][resonance]") {
    Scratchpad sp;
    sp.commit("the cat sat on the mat", 1.0);

    const uint64_t id = sp.inject("the cat was on the mat");
    const double score = sp.measure_resonance(id);
    REQUIRE(score > 0.0);

    // Score is stored on the entry
    REQUIRE(sp.find(id)->resonance == Approx(score));
}

// ---------------------------------------------------------------------------
// collapse_if_resonant / discard
// ---------------------------------------------------------------------------

TEST_CASE("Scratchpad::collapse_if_resonant — collapses high-resonance entry",
          "[Phase125][collapse]") {
    Scratchpad sp;
    sp.commit("nikola is an artificial intelligence system", 1.0);

    // Very similar text — should exceed SCRATCHPAD_RESONANCE_THRESHOLD (0.40)
    const uint64_t id = sp.inject("nikola is an artificial intelligence", 0.9);
    const bool collapsed = sp.collapse_if_resonant(id);

    REQUIRE(collapsed == true);
    REQUIRE(sp.find(id)->status == HypothesisStatus::COLLAPSED);
}

TEST_CASE("Scratchpad::collapse_if_resonant — discards low-resonance entry",
          "[Phase125][collapse]") {
    Scratchpad sp;
    sp.commit("the weather in london is cold and rainy", 1.0);

    // Completely different topic
    const uint64_t id = sp.inject("quantum physics equations", 0.9);
    const bool collapsed = sp.collapse_if_resonant(id);

    REQUIRE(collapsed == false);
    REQUIRE(sp.find(id)->status == HypothesisStatus::DISCARDED);
}

TEST_CASE("Scratchpad::collapse_if_resonant — respects custom threshold",
          "[Phase125][collapse]") {
    Scratchpad sp;
    sp.commit("cat dog", 1.0);

    // "cat fish" → overlap with "cat dog": intersect=1(cat), union=3 → 0.333
    const uint64_t id = sp.inject("cat fish");
    // Default threshold 0.40 → discard
    REQUIRE(sp.collapse_if_resonant(id) == false);

    // With low threshold 0.20 → should collapse
    const uint64_t id2 = sp.inject("cat fish");
    REQUIRE(sp.collapse_if_resonant(id2, 0.20) == true);
}

TEST_CASE("Scratchpad::discard — explicitly discards", "[Phase125][discard]") {
    Scratchpad sp;
    const uint64_t id = sp.inject("to be discarded");
    REQUIRE(sp.find(id)->status == HypothesisStatus::PENDING);

    sp.discard(id);
    REQUIRE(sp.find(id)->status == HypothesisStatus::DISCARDED);
}

TEST_CASE("Scratchpad::discard — unknown id is no-op", "[Phase125][discard]") {
    Scratchpad sp;
    // Should not throw
    REQUIRE_NOTHROW(sp.discard(9999));
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

TEST_CASE("Scratchpad — queries return correct pointers", "[Phase125][queries]") {
    Scratchpad sp;
    sp.commit("ground truth", 1.0);

    const uint64_t pa = sp.inject("pending a");
    const uint64_t pb = sp.inject("pending b");
    const uint64_t pc = sp.inject("ground truth matching text");

    sp.discard(pb);
    sp.collapse_if_resonant(pc);

    const auto pending_v   = sp.pending();
    const auto collapsed_v = sp.collapsed();
    const auto discarded_v = sp.discarded();

    REQUIRE(pending_v.size()   == 1);
    REQUIRE(collapsed_v.size() == 1);
    REQUIRE(discarded_v.size() == 1);

    REQUIRE(pending_v[0]->id   == pa);
    REQUIRE(collapsed_v[0]->id == pc);
    REQUIRE(discarded_v[0]->id == pb);
}

TEST_CASE("Scratchpad::find — returns nullptr for missing id",
          "[Phase125][queries]") {
    Scratchpad sp;
    REQUIRE(sp.find(0) == nullptr);
    REQUIRE(sp.find(1234) == nullptr);
}

// ---------------------------------------------------------------------------
// clear
// ---------------------------------------------------------------------------

TEST_CASE("Scratchpad::clear_pending — only removes PENDING",
          "[Phase125][clear]") {
    Scratchpad sp;
    sp.commit("fact", 1.0);

    sp.inject("keep me — will be collapsed");
    const uint64_t p_id = sp.inject("pending that will be cleared");
    const uint64_t d_id = sp.inject("will be discarded");

    // Need to collapse first entry before clearing pending
    const uint64_t c_id = sp.inject("fact exact");
    sp.collapse_if_resonant(c_id);
    sp.discard(d_id);

    sp.clear_pending();

    REQUIRE(sp.find(p_id) == nullptr);           // cleared
    REQUIRE(sp.find(d_id) != nullptr);           // discarded — kept
    REQUIRE(sp.find(c_id) != nullptr);           // collapsed — kept
}

TEST_CASE("Scratchpad::clear_all — removes all hypotheses",
          "[Phase125][clear]") {
    Scratchpad sp;
    sp.commit("fact", 1.0);
    sp.inject("one");
    sp.inject("two");
    sp.inject("three");
    REQUIRE(sp.hypothesis_count() == 3);

    sp.clear_all();
    REQUIRE(sp.hypothesis_count() == 0);
    // Committed pool survives
    REQUIRE(sp.committed_count() == 1);
}

// ---------------------------------------------------------------------------
// Callback
// ---------------------------------------------------------------------------

TEST_CASE("Scratchpad::on_collapse — fires when hypothesis collapses",
          "[Phase125][callback]") {
    Scratchpad sp;
    sp.commit("nikola cognitive architecture system awareness", 1.0);

    bool fired = false;
    uint64_t fired_id = 0;
    sp.on_collapse([&](const HypothesisEntry& e) {
        fired    = true;
        fired_id = e.id;
    });

    const uint64_t id = sp.inject("nikola cognitive architecture system");
    const bool ok = sp.collapse_if_resonant(id);

    REQUIRE(ok == true);
    REQUIRE(fired == true);
    REQUIRE(fired_id == id);
}

TEST_CASE("Scratchpad::on_collapse — does NOT fire on discard",
          "[Phase125][callback]") {
    Scratchpad sp;
    sp.commit("unrelated topic here", 1.0);

    bool fired = false;
    sp.on_collapse([&](const HypothesisEntry&) { fired = true; });

    const uint64_t id = sp.inject("completely different words");
    sp.collapse_if_resonant(id);

    REQUIRE(fired == false);
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

TEST_CASE("Scratchpad::stats — counts reflect operations",
          "[Phase125][stats]") {
    Scratchpad sp;
    sp.commit("the cat sat on the mat", 1.0);
    sp.commit("dogs are loyal animals", 1.0);

    const uint64_t id1 = sp.inject("cat mat sat");      // should collapse
    const uint64_t id2 = sp.inject("random words xyz"); // should discard
    [[maybe_unused]] const uint64_t id3 = sp.inject("still pending");

    sp.collapse_if_resonant(id1);
    sp.collapse_if_resonant(id2);
    // id3 stays pending

    const auto s = sp.stats();

    REQUIRE(s.total_injected  == 3);
    REQUIRE(s.total_collapsed == 1);
    REQUIRE(s.total_discarded == 1);
    REQUIRE(s.total_pending   == 1);
    REQUIRE(s.total_committed == 2);
    REQUIRE(s.mean_resonance  > 0.0);
}

TEST_CASE("Scratchpad::stats — empty scratchpad", "[Phase125][stats]") {
    Scratchpad sp;
    const auto s = sp.stats();
    REQUIRE(s.total_injected  == 0);
    REQUIRE(s.total_committed == 0);
    REQUIRE(s.mean_resonance  == Approx(0.0));
}

// ---------------------------------------------------------------------------
// FIFO eviction
// ---------------------------------------------------------------------------

TEST_CASE("Scratchpad — FIFO eviction at SCRATCHPAD_MAX_HYPOTHESES",
          "[Phase125][fifo]") {
    Scratchpad sp;

    // Fill to exactly MAX
    std::vector<uint64_t> ids;
    for (size_t i = 0; i < SCRATCHPAD_MAX_HYPOTHESES; ++i) {
        ids.push_back(sp.inject("entry " + std::to_string(i)));
    }
    REQUIRE(sp.hypothesis_count() == SCRATCHPAD_MAX_HYPOTHESES);

    // One more inject — oldest PENDING should be evicted
    const uint64_t new_id = sp.inject("newcomer");
    REQUIRE(sp.hypothesis_count() == SCRATCHPAD_MAX_HYPOTHESES);
    REQUIRE(sp.find(ids.front()) == nullptr);   // oldest evicted
    REQUIRE(sp.find(new_id) != nullptr);        // newcomer present
}

TEST_CASE("Scratchpad — FIFO eviction for committed pool",
          "[Phase125][fifo]") {
    Scratchpad sp;
    for (size_t i = 0; i < SCRATCHPAD_MAX_COMMITTED; ++i) {
        sp.commit("commitment " + std::to_string(i), 1.0);
    }
    REQUIRE(sp.committed_count() == SCRATCHPAD_MAX_COMMITTED);

    // First commit text
    const std::string first_text = sp.committed().front().text;

    sp.commit("one more committed fact");
    REQUIRE(sp.committed_count() == SCRATCHPAD_MAX_COMMITTED);
    REQUIRE(sp.committed().front().text != first_text); // oldest evicted
}

// ---------------------------------------------------------------------------
// Alias
// ---------------------------------------------------------------------------

TEST_CASE("QuantumScratchpad alias works", "[Phase125][alias]") {
    QuantumScratchpad sp;
    sp.commit("fact", 1.0);
    const uint64_t id = sp.inject("fact based reasoning");
    REQUIRE(sp.find(id) != nullptr);
    REQUIRE(sp.measure_resonance(id) > 0.0);
}
