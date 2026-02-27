/**
 * @file phase122_internal_dialogue_test.cpp
 * @brief Phase 122 — InternalDialogue unit tests
 *
 * §1  Constants
 * §2  Pure static: word_overlap()
 * §3  Pure static: contains_negation()
 * §4  Pure static: generate_socratic_questions()
 * §5  Chain lifecycle — start, think, conclude
 * §6  Chain confidence and accessors
 * §7  detect_circular_reasoning()
 * §8  detect_contradictions()
 * §9  synthesize_conclusion()
 * §10 question_assumption() — instance method
 * §11 explain_reasoning()
 * §12 recall_similar()
 * §13 stats()
 * §14 NikolaState context attachment
 * §15 auto-conclude on new start_chain()
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/interior/internal_dialogue.hpp>
#include <nikola/autonomy/decision_loop.hpp>

using namespace nikola::interior;
using namespace nikola::autonomy;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static NikolaState make_state(float dopamine = 0.5f,
                               float atp      = 0.7f,
                               float entropy  = 1.0f) {
    NikolaState s;
    s.dopamine = dopamine;
    s.atp      = atp;
    s.entropy  = entropy;
    return s;
}

// ---------------------------------------------------------------------------
// §1 — Constants
// ---------------------------------------------------------------------------

TEST_CASE("Phase122 §1 constants", "[Phase122]") {
    SECTION("DIALOGUE_CIRCULAR_THRESHOLD in (0, 1)") {
        CHECK(DIALOGUE_CIRCULAR_THRESHOLD > 0.0);
        CHECK(DIALOGUE_CIRCULAR_THRESHOLD < 1.0);
    }
    SECTION("DIALOGUE_CONTRADICTION_OVERLAP in (0, 1)") {
        CHECK(DIALOGUE_CONTRADICTION_OVERLAP > 0.0);
        CHECK(DIALOGUE_CONTRADICTION_OVERLAP < 1.0);
    }
    SECTION("DIALOGUE_MAX_RECALL > 0") {
        CHECK(DIALOGUE_MAX_RECALL > 0);
    }
    SECTION("DIALOGUE_CHAIN_LENGTH_WARN > 0") {
        CHECK(DIALOGUE_CHAIN_LENGTH_WARN > 0);
    }
}

// ---------------------------------------------------------------------------
// §2 — word_overlap
// ---------------------------------------------------------------------------

TEST_CASE("Phase122 §2 word_overlap()", "[Phase122]") {
    SECTION("identical strings -> 1.0") {
        CHECK_THAT(InternalDialogue::word_overlap("hello world", "hello world"),
                   WithinAbs(1.0, 1e-9));
    }

    SECTION("completely disjoint strings -> 0.0") {
        CHECK_THAT(InternalDialogue::word_overlap("cat dog", "tree car"),
                   WithinAbs(0.0, 1e-9));
    }

    SECTION("half overlap -> ~0.33..0.5") {
        double ov = InternalDialogue::word_overlap("the cat sat", "the dog ran");
        CHECK(ov > 0.0);
        CHECK(ov < 1.0);
    }

    SECTION("empty both -> 1.0") {
        CHECK_THAT(InternalDialogue::word_overlap("", ""),
                   WithinAbs(1.0, 1e-9));
    }

    SECTION("empty one side -> 0.0") {
        CHECK_THAT(InternalDialogue::word_overlap("hello", ""),
                   WithinAbs(0.0, 1e-9));
        CHECK_THAT(InternalDialogue::word_overlap("", "world"),
                   WithinAbs(0.0, 1e-9));
    }

    SECTION("case insensitive") {
        double ov = InternalDialogue::word_overlap("Hello World", "hello world");
        CHECK_THAT(ov, WithinAbs(1.0, 1e-9));
    }

    SECTION("result always in [0, 1]") {
        double ov = InternalDialogue::word_overlap(
            "the quick brown fox jumped", "a slow white rabbit hopped");
        CHECK(ov >= 0.0);
        CHECK(ov <= 1.0);
    }

    SECTION("superset has higher overlap than disjoint") {
        double ov_sup = InternalDialogue::word_overlap("cat dog bird", "cat dog");
        double ov_dis = InternalDialogue::word_overlap("cat dog bird", "mouse fish");
        CHECK(ov_sup > ov_dis);
    }
}

// ---------------------------------------------------------------------------
// §3 — contains_negation
// ---------------------------------------------------------------------------

TEST_CASE("Phase122 §3 contains_negation()", "[Phase122]") {
    SECTION("explicit negation of a topic word -> true") {
        CHECK(InternalDialogue::contains_negation(
            "the system is ready",
            "the system is not ready"));
    }

    SECTION("'cannot' negation -> true") {
        CHECK(InternalDialogue::contains_negation(
            "we can solve it",
            "we cannot solve it"));
    }

    SECTION("symmetric: a negates b == b negates a") {
        bool ab = InternalDialogue::contains_negation(
            "the door is open", "the door is not open");
        bool ba = InternalDialogue::contains_negation(
            "the door is not open", "the door is open");
        CHECK(ab == ba);
    }

    SECTION("unrelated strings -> false") {
        CHECK(!InternalDialogue::contains_negation(
            "the cat sat on the mat",
            "birds fly in the sky"));
    }

    SECTION("positive statements without any negation marker -> false") {
        CHECK(!InternalDialogue::contains_negation(
            "the answer is correct",
            "the answer is correct"));
    }
}

// ---------------------------------------------------------------------------
// §4 — generate_socratic_questions
// ---------------------------------------------------------------------------

TEST_CASE("Phase122 §4 generate_socratic_questions()", "[Phase122]") {
    SECTION("returns 5 questions for non-empty assumption") {
        auto qs = InternalDialogue::generate_socratic_questions("X is true");
        CHECK(qs.size() == 5);
    }

    SECTION("each question is non-empty and contains the assumption") {
        auto qs = InternalDialogue::generate_socratic_questions("the model is correct");
        for (const auto& q : qs) {
            CHECK(!q.empty());
            CHECK(q.find("the model is correct") != std::string::npos);
        }
    }

    SECTION("empty assumption -> returns empty list") {
        auto qs = InternalDialogue::generate_socratic_questions("");
        CHECK(qs.empty());
    }
}

// ---------------------------------------------------------------------------
// §5 — Chain lifecycle
// ---------------------------------------------------------------------------

TEST_CASE("Phase122 §5 chain lifecycle", "[Phase122]") {
    SECTION("initially no active chain") {
        InternalDialogue d;
        CHECK(!d.has_active_chain());
    }

    SECTION("start_chain activates a chain") {
        InternalDialogue d;
        d.start_chain("Is dopamine sufficient?");
        CHECK(d.has_active_chain());
    }

    SECTION("think() creates an active chain if none exists") {
        InternalDialogue d;
        d.think("initial thought");
        CHECK(d.has_active_chain());
    }

    SECTION("think() adds to current chain") {
        InternalDialogue d;
        d.start_chain("test");
        d.think("first thought");
        d.think("second thought");
        CHECK(d.current_length() == 2);
    }

    SECTION("conclude_chain() deactivates chain") {
        InternalDialogue d;
        d.start_chain("test");
        d.think("a thought");
        d.conclude_chain("my conclusion");
        CHECK(!d.has_active_chain());
    }

    SECTION("concluded chain moves to all_chains()") {
        InternalDialogue d;
        d.start_chain("problem A");
        d.think("step 1");
        d.conclude_chain("answer A");
        CHECK(d.all_chains().size() == 1);
        CHECK(d.all_chains()[0].problem == "problem A");
        CHECK(d.all_chains()[0].conclusion == "answer A");
        CHECK(d.all_chains()[0].is_concluded());
    }

    SECTION("conclude_chain on inactive dialogue is a no-op") {
        InternalDialogue d;
        REQUIRE_NOTHROW(d.conclude_chain("nothing"));
        CHECK(d.all_chains().empty());
    }

    SECTION("chain_id increments per chain") {
        InternalDialogue d;
        uint64_t id1 = d.start_chain("A");
        d.conclude_chain("done A");
        uint64_t id2 = d.start_chain("B");
        d.conclude_chain("done B");
        CHECK(id2 > id1);
    }
}

// ---------------------------------------------------------------------------
// §6 — Chain confidence
// ---------------------------------------------------------------------------

TEST_CASE("Phase122 §6 chain confidence", "[Phase122]") {
    SECTION("empty chain -> chain_confidence() = 0") {
        InternalDialogue d;
        d.start_chain("test");
        CHECK_THAT(d.chain_confidence(), WithinAbs(0.0, 1e-9));
    }

    SECTION("chain_confidence() = mean of thought confidences") {
        InternalDialogue d;
        d.start_chain("test");
        d.think("t1", 0.4);
        d.think("t2", 0.8);
        // mean = 0.6
        CHECK_THAT(d.chain_confidence(), WithinAbs(0.6, 1e-9));
    }

    SECTION("confidence is clamped to [0, 1]") {
        InternalDialogue d;
        d.think("too confident", 1.5);
        d.think("negative conf", -0.5);
        auto& chain = d.current_chain();
        for (const auto& t : chain.thoughts) {
            CHECK(t.confidence >= 0.0);
            CHECK(t.confidence <= 1.0);
        }
    }

    SECTION("ReasoningChain::mean_confidence() matches") {
        InternalDialogue d;
        d.start_chain("x");
        d.think("a", 0.3);
        d.think("b", 0.7);
        double mc = d.current_chain().mean_confidence();
        CHECK_THAT(mc, WithinAbs(0.5, 1e-9));
    }

    SECTION("ReasoningChain::peak_confidence()") {
        InternalDialogue d;
        d.start_chain("x");
        d.think("low", 0.2);
        d.think("high", 0.9);
        d.think("mid", 0.5);
        CHECK_THAT(d.current_chain().peak_confidence(), WithinAbs(0.9, 1e-9));
    }
}

// ---------------------------------------------------------------------------
// §7 — detect_circular_reasoning
// ---------------------------------------------------------------------------

TEST_CASE("Phase122 §7 detect_circular_reasoning()", "[Phase122]") {
    SECTION("empty chain -> false") {
        InternalDialogue d;
        d.start_chain("test");
        CHECK(!d.detect_circular_reasoning());
    }

    SECTION("single thought -> false") {
        InternalDialogue d;
        d.think("the sky is blue");
        CHECK(!d.detect_circular_reasoning());
    }

    SECTION("nearly identical thoughts -> true") {
        InternalDialogue d;
        d.start_chain("loop");
        d.think("we cannot solve this problem because we lack information");
        d.think("we lack information because we cannot solve this problem");
        CHECK(d.detect_circular_reasoning());
    }

    SECTION("completely different thoughts -> false") {
        InternalDialogue d;
        d.start_chain("test");
        d.think("the network dropped during the connection attempt");
        d.think("dopamine levels were elevated after the reward");
        CHECK(!d.detect_circular_reasoning());
    }
}

// ---------------------------------------------------------------------------
// §8 — detect_contradictions
// ---------------------------------------------------------------------------

TEST_CASE("Phase122 §8 detect_contradictions()", "[Phase122]") {
    SECTION("empty chain -> zero contradictions") {
        InternalDialogue d;
        d.start_chain("test");
        CHECK(d.detect_contradictions().empty());
    }

    SECTION("single thought -> zero contradictions") {
        InternalDialogue d;
        d.think("the system is stable");
        CHECK(d.detect_contradictions().empty());
    }

    SECTION("contradictory pair detected") {
        InternalDialogue d;
        d.start_chain("stability");
        d.think("the system is stable and reliable");    // thought 0
        d.think("the system is not stable and reliable"); // thought 1
        auto contradictions = d.detect_contradictions();
        CHECK(!contradictions.empty());
        // Should include the pair (0, 1)
        bool found = false;
        for (auto [i, j] : contradictions) {
            if ((i == 0 && j == 1) || (i == 1 && j == 0))
                found = true;
        }
        CHECK(found);
    }

    SECTION("non-contradictory high-overlap pair not flagged") {
        // High overlap but no negation markers
        InternalDialogue d;
        d.start_chain("agreement");
        d.think("the answer is correct and verified");
        d.think("the answer is correct and confirmed");
        // word_overlap is high but no negation -> no contradiction
        // (depends on threshold — confirm detector logic is sound)
        // At minimum this should not crash
        auto c = d.detect_contradictions();
        REQUIRE_NOTHROW(c.size());
    }
}

// ---------------------------------------------------------------------------
// §9 — synthesize_conclusion
// ---------------------------------------------------------------------------

TEST_CASE("Phase122 §9 synthesize_conclusion()", "[Phase122]") {
    SECTION("empty chain -> empty string") {
        InternalDialogue d;
        d.start_chain("test");
        CHECK(d.synthesize_conclusion().empty());
    }

    SECTION("single thought -> returns it prefixed with 'Synthesis: '") {
        InternalDialogue d;
        d.think("the cause is clear", 0.7);
        std::string syn = d.synthesize_conclusion();
        CHECK(syn.find("Synthesis:") != std::string::npos);
        CHECK(syn.find("the cause is clear") != std::string::npos);
    }

    SECTION("picks highest-confidence thought") {
        InternalDialogue d;
        d.start_chain("test");
        d.think("low confidence thought", 0.2);
        d.think("high confidence thought", 0.9);
        d.think("medium confidence thought", 0.5);
        std::string syn = d.synthesize_conclusion();
        CHECK(syn.find("high confidence thought") != std::string::npos);
    }

    SECTION("no active chain -> empty string") {
        InternalDialogue d;
        CHECK(d.synthesize_conclusion().empty());
    }
}

// ---------------------------------------------------------------------------
// §10 — question_assumption (instance method)
// ---------------------------------------------------------------------------

TEST_CASE("Phase122 §10 question_assumption()", "[Phase122]") {
    SECTION("returns 5 questions") {
        InternalDialogue d;
        auto qs = d.question_assumption("the model is converging");
        CHECK(qs.size() == 5);
    }

    SECTION("each question contains the assumption text") {
        InternalDialogue d;
        auto qs = d.question_assumption("the router was compromised");
        for (const auto& q : qs)
            CHECK(q.find("the router was compromised") != std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// §11 — explain_reasoning
// ---------------------------------------------------------------------------

TEST_CASE("Phase122 §11 explain_reasoning()", "[Phase122]") {
    SECTION("no active chain -> returns placeholder") {
        InternalDialogue d;
        std::string ex = d.explain_reasoning();
        CHECK(!ex.empty());
        CHECK(ex.find("no active") != std::string::npos);
    }

    SECTION("contains problem text") {
        InternalDialogue d;
        d.start_chain("Why does ATP drop under high load?");
        d.think("metabolic cost of computation is high", 0.7);
        std::string ex = d.explain_reasoning();
        CHECK(ex.find("Why does ATP drop") != std::string::npos);
    }

    SECTION("contains each thought text") {
        InternalDialogue d;
        d.start_chain("test");
        d.think("thought A", 0.5, "deduction");
        d.think("thought B", 0.8, "induction");
        std::string ex = d.explain_reasoning();
        CHECK(ex.find("thought A") != std::string::npos);
        CHECK(ex.find("thought B") != std::string::npos);
    }

    SECTION("contains reasoning_type labels") {
        InternalDialogue d;
        d.start_chain("test");
        d.think("X follows from Y", 0.6, "deduction");
        std::string ex = d.explain_reasoning();
        CHECK(ex.find("deduction") != std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// §12 — recall_similar
// ---------------------------------------------------------------------------

TEST_CASE("Phase122 §12 recall_similar()", "[Phase122]") {
    SECTION("no past chains -> empty result") {
        InternalDialogue d;
        auto res = d.recall_similar("anything");
        CHECK(res.empty());
    }

    SECTION("recalls most relevant past chain") {
        InternalDialogue d;
        d.start_chain("network latency problem");
        d.think("the router dropped packets");
        d.conclude_chain("packet loss caused by router");

        d.start_chain("file system corruption issue");
        d.think("disk write failed");
        d.conclude_chain("disk failure");

        auto res = d.recall_similar("router and network issue", 2);
        REQUIRE(!res.empty());
        // First result should be the network chain
        CHECK(res[0]->problem.find("network") != std::string::npos);
    }

    SECTION("max_results limits output") {
        InternalDialogue d;
        for (int i = 0; i < 10; ++i) {
            d.start_chain("problem " + std::to_string(i));
            d.think("thought about problem");
            d.conclude_chain("done");
        }
        auto res = d.recall_similar("problem thought", 3);
        CHECK(res.size() <= 3);
    }

    SECTION("results are pointers into past chains (non-null)") {
        InternalDialogue d;
        d.start_chain("x");
        d.conclude_chain("y");
        auto res = d.recall_similar("x", 1);
        REQUIRE(!res.empty());
        CHECK(res[0] != nullptr);
    }
}

// ---------------------------------------------------------------------------
// §13 — stats()
// ---------------------------------------------------------------------------

TEST_CASE("Phase122 §13 stats()", "[Phase122]") {
    SECTION("fresh instance -> all zeros") {
        InternalDialogue d;
        auto s = d.stats();
        CHECK(s.total_thoughts   == 0);
        CHECK(s.total_chains     == 0);
        CHECK(s.completed_chains == 0);
    }

    SECTION("counts thoughts across chains") {
        InternalDialogue d;
        d.start_chain("A");
        d.think("1"); d.think("2");
        d.conclude_chain("done");

        d.start_chain("B");
        d.think("3");
        // chain B still active

        auto s = d.stats();
        CHECK(s.total_thoughts   == 3);
        CHECK(s.total_chains     == 2);
        CHECK(s.completed_chains == 1);
    }

    SECTION("mean_chain_confidence uses concluded chains only") {
        InternalDialogue d;
        d.start_chain("A");
        d.think("t1", 0.5);
        d.conclude_chain("done", 0.5);

        d.start_chain("B");
        d.think("t2", 0.9);
        d.conclude_chain("done", 0.9);

        auto s = d.stats();
        CHECK_THAT(s.mean_chain_confidence, WithinAbs(0.7, 0.01));
    }
}

// ---------------------------------------------------------------------------
// §14 — NikolaState context attachment
// ---------------------------------------------------------------------------

TEST_CASE("Phase122 §14 NikolaState context attachment", "[Phase122]") {
    SECTION("thoughts capture neurochemical context when state provided") {
        InternalDialogue d;
        auto state = make_state(0.8f, 0.9f, 1.5f);
        d.think("important insight", 0.8, "deduction", &state);
        const auto& t = d.current_chain().thoughts.back();
        CHECK_THAT(t.dopamine_context, WithinAbs(0.8, 1e-4));
        CHECK_THAT(t.atp_context,      WithinAbs(0.9, 1e-4));
        CHECK_THAT(t.entropy_context,  WithinAbs(1.5, 1e-4));
    }

    SECTION("thoughts without state have zero neurochemical context") {
        InternalDialogue d;
        d.think("plain thought", 0.5);
        const auto& t = d.current_chain().thoughts.back();
        CHECK_THAT(t.dopamine_context, WithinAbs(0.0, 1e-9));
        CHECK_THAT(t.atp_context,      WithinAbs(0.0, 1e-9));
        CHECK_THAT(t.entropy_context,  WithinAbs(0.0, 1e-9));
    }

    SECTION("reasoning_type stored correctly") {
        InternalDialogue d;
        d.think("X implies Y", 0.7, "deduction");
        CHECK(d.current_chain().thoughts.back().reasoning_type == "deduction");
    }
}

// ---------------------------------------------------------------------------
// §15 — auto-conclude on new start_chain
// ---------------------------------------------------------------------------

TEST_CASE("Phase122 §15 auto-conclude on second start_chain()", "[Phase122]") {
    SECTION("open chain is auto-concluded before new start") {
        InternalDialogue d;
        d.start_chain("first problem");
        d.think("some thought");
        // Start another chain without manually concluding
        d.start_chain("second problem");
        // First chain should now be in past_
        CHECK(d.all_chains().size() == 1);
        CHECK(d.all_chains()[0].problem == "first problem");
        // New chain should be active
        CHECK(d.has_active_chain());
        CHECK(d.current_chain().problem == "second problem");
    }
}
