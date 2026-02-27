/**
 * @file phase121_wave_mirror_test.cpp
 * @brief Phase 121 — WaveMirror unit tests
 *
 * §1  Constants and AttentionFocus labels
 * §2  Construction and initial state
 * §3  Pure static: compute_confidence()
 * §4  Pure static: compute_confusion()
 * §5  Pure static: compute_cognitive_load()
 * §6  Pure static: compute_coherence()
 * §7  Pure static: compute_spectral_signature()
 * §8  Pure static: compute_attention_focus()
 * §9  update() — metric derivation
 * §10 Rolling-window smoothing
 * §11 snapshot() / individual accessors consistency
 * §12 describe() — human-readable output
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/interior/wave_mirror.hpp>
#include <nikola/autonomy/decision_loop.hpp>

using namespace nikola::interior;
using namespace nikola::autonomy;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static NikolaState make_state(float dopamine    = 0.5f,
                               float td_error    = 0.0f,
                               float atp         = 0.7f,
                               float boredom     = 0.1f,
                               float entropy     = 1.0f,
                               float torus_energy = 0.4f) {
    NikolaState s;
    s.dopamine     = dopamine;
    s.td_error     = td_error;
    s.atp          = atp;
    s.boredom      = boredom;
    s.entropy      = entropy;
    s.torus_energy = torus_energy;
    return s;
}

// ---------------------------------------------------------------------------
// §1 — Constants and labels
// ---------------------------------------------------------------------------

TEST_CASE("Phase121 §1 Constants and AttentionFocus labels", "[Phase121]") {
    SECTION("MIRROR_HISTORY_WINDOW is positive") {
        CHECK(MIRROR_HISTORY_WINDOW > 0);
    }

    SECTION("MIRROR_ENTROPY_CEILING is positive") {
        CHECK(MIRROR_ENTROPY_CEILING > 0.0);
    }

    SECTION("MIRROR_COHERENCE_HIGH is in (0, 1)") {
        CHECK(MIRROR_COHERENCE_HIGH > 0.0);
        CHECK(MIRROR_COHERENCE_HIGH < 1.0);
    }

    SECTION("AttentionFocus mode_name returns non-empty strings for all modes") {
        using Mode = AttentionFocus::Mode;
        for (auto m : {Mode::CURIOUS, Mode::REWARD, Mode::THREAT,
                       Mode::FATIGUE, Mode::IDLE}) {
            const char* name = AttentionFocus::mode_name(m);
            CHECK(name != nullptr);
            CHECK(std::string(name).size() > 0);
        }
    }

    SECTION("mode_name member matches static") {
        AttentionFocus f;
        f.mode = AttentionFocus::Mode::REWARD;
        CHECK(std::string(f.mode_name()) == "reward");
    }
}

// ---------------------------------------------------------------------------
// §2 — Construction
// ---------------------------------------------------------------------------

TEST_CASE("Phase121 §2 Default construction", "[Phase121]") {
    WaveMirror wm;

    SECTION("confidence is 0") {
        CHECK_THAT(wm.confidence(), WithinAbs(0.0, 1e-9));
    }
    SECTION("confusion is 0") {
        CHECK_THAT(wm.confusion(), WithinAbs(0.0, 1e-9));
    }
    SECTION("cognitive_load is 0") {
        CHECK_THAT(wm.cognitive_load(), WithinAbs(0.0, 1e-9));
    }
    SECTION("coherence is 0") {
        CHECK_THAT(wm.coherence(), WithinAbs(0.0, 1e-9));
    }
    SECTION("metacognitive is 0") {
        CHECK_THAT(wm.metacognitive(), WithinAbs(0.0, 1e-9));
    }
    SECTION("initial focus mode is IDLE") {
        CHECK(wm.attention_focus().mode == AttentionFocus::Mode::IDLE);
    }
    SECTION("all spectral bins are 0") {
        for (double v : wm.spectral_signature())
            CHECK_THAT(v, WithinAbs(0.0, 1e-9));
    }
}

// ---------------------------------------------------------------------------
// §3 — compute_confidence
// ---------------------------------------------------------------------------

TEST_CASE("Phase121 §3 compute_confidence", "[Phase121]") {
    SECTION("Equilibrium D=0.5, td=0, A=1.0 -> ~0.45") {
        // 0.5*0.5 + 0 + 1.0*0.2 = 0.25 + 0.2 = 0.45
        double c = WaveMirror::compute_confidence(0.5, 0.0, 1.0);
        CHECK_THAT(c, WithinAbs(0.45, 0.01));
    }

    SECTION("High D + positive td + high A -> high confidence") {
        double c = WaveMirror::compute_confidence(0.9, 0.3, 0.9);
        CHECK(c > 0.6);
    }

    SECTION("Low D + negative td + low A -> low confidence") {
        double c = WaveMirror::compute_confidence(0.1, -0.3, 0.05);
        CHECK(c < 0.15);
    }

    SECTION("Positive td boosts above zero-td equivalent") {
        double c_zero = WaveMirror::compute_confidence(0.5, 0.0, 0.7);
        double c_pos  = WaveMirror::compute_confidence(0.5, 0.2, 0.7);
        CHECK(c_pos > c_zero);
    }

    SECTION("Negative td reduces below zero-td equivalent") {
        double c_zero = WaveMirror::compute_confidence(0.5, 0.0, 0.7);
        double c_neg  = WaveMirror::compute_confidence(0.5, -0.2, 0.7);
        CHECK(c_neg < c_zero);
    }

    SECTION("Result clamped to [0, 1]") {
        CHECK(WaveMirror::compute_confidence(0.0, -1.0, 0.0) >= 0.0);
        CHECK(WaveMirror::compute_confidence(1.0,  1.0, 1.0) <= 1.0);
    }
}

// ---------------------------------------------------------------------------
// §4 — compute_confusion
// ---------------------------------------------------------------------------

TEST_CASE("Phase121 §4 compute_confusion", "[Phase121]") {
    SECTION("Zero entropy + zero td -> zero confusion") {
        CHECK_THAT(WaveMirror::compute_confusion(0.0, 0.0), WithinAbs(0.0, 1e-9));
    }

    SECTION("High entropy increases confusion") {
        double c_low  = WaveMirror::compute_confusion(0.5, 0.0);
        double c_high = WaveMirror::compute_confusion(2.5, 0.0);
        CHECK(c_high > c_low);
    }

    SECTION("Strong negative td increases confusion") {
        double c_zero = WaveMirror::compute_confusion(1.0, 0.0);
        double c_neg  = WaveMirror::compute_confusion(1.0, -0.4);
        CHECK(c_neg > c_zero);
    }

    SECTION("Positive td does not increase confusion") {
        double c_zero = WaveMirror::compute_confusion(1.0, 0.0);
        double c_pos  = WaveMirror::compute_confusion(1.0, 0.4);
        CHECK(c_pos <= c_zero + 1e-9);
    }

    SECTION("Result clamped to [0, 1]") {
        CHECK(WaveMirror::compute_confusion(100.0, -100.0) <= 1.0);
        CHECK(WaveMirror::compute_confusion(0.0, 0.0) >= 0.0);
    }
}

// ---------------------------------------------------------------------------
// §5 — compute_cognitive_load
// ---------------------------------------------------------------------------

TEST_CASE("Phase121 §5 compute_cognitive_load", "[Phase121]") {
    SECTION("Zero entropy + zero energy -> zero load") {
        CHECK_THAT(WaveMirror::compute_cognitive_load(0.0, 0.0),
                   WithinAbs(0.0, 1e-9));
    }

    SECTION("High entropy increases load") {
        double l_low  = WaveMirror::compute_cognitive_load(0.5, 0.2);
        double l_high = WaveMirror::compute_cognitive_load(2.5, 0.2);
        CHECK(l_high > l_low);
    }

    SECTION("High torus energy increases load") {
        double l_low  = WaveMirror::compute_cognitive_load(1.0, 0.1);
        double l_high = WaveMirror::compute_cognitive_load(1.0, 0.9);
        CHECK(l_high > l_low);
    }

    SECTION("Entropy contributes more than energy (weight 0.7 vs 0.3)") {
        // At ENTROPY_CEILING entropy alone -> load = 0.7
        // At full torus energy alone -> load = 0.3
        double l_entr = WaveMirror::compute_cognitive_load(
                            MIRROR_ENTROPY_CEILING, 0.0);
        double l_ener = WaveMirror::compute_cognitive_load(0.0, 1.0);
        CHECK(l_entr > l_ener);
    }

    SECTION("Result clamped to [0, 1]") {
        CHECK(WaveMirror::compute_cognitive_load(100.0, 100.0) <= 1.0);
        CHECK(WaveMirror::compute_cognitive_load(0.0, 0.0) >= 0.0);
    }
}

// ---------------------------------------------------------------------------
// §6 — compute_coherence
// ---------------------------------------------------------------------------

TEST_CASE("Phase121 §6 compute_coherence", "[Phase121]") {
    SECTION("High D + high A + low H -> high coherence") {
        double c = WaveMirror::compute_coherence(0.9, 0.9, 0.1);
        CHECK(c > 0.7);
    }

    SECTION("Low D + low A + high H -> low coherence") {
        double c = WaveMirror::compute_coherence(0.05, 0.05, 2.5);
        CHECK(c < 0.2);
    }

    SECTION("Increasing D raises coherence") {
        double c_low  = WaveMirror::compute_coherence(0.2, 0.5, 1.0);
        double c_high = WaveMirror::compute_coherence(0.8, 0.5, 1.0);
        CHECK(c_high > c_low);
    }

    SECTION("Increasing entropy lowers coherence") {
        double c_low_H  = WaveMirror::compute_coherence(0.6, 0.6, 0.2);
        double c_high_H = WaveMirror::compute_coherence(0.6, 0.6, 2.5);
        CHECK(c_low_H > c_high_H);
    }

    SECTION("Result clamped to [0, 1]") {
        CHECK(WaveMirror::compute_coherence(0.0, 0.0, 100.0) >= 0.0);
        CHECK(WaveMirror::compute_coherence(1.0, 1.0, 0.0) <= 1.0);
    }
}

// ---------------------------------------------------------------------------
// §7 — compute_spectral_signature
// ---------------------------------------------------------------------------

TEST_CASE("Phase121 §7 compute_spectral_signature", "[Phase121]") {
    SECTION("All 9 bins are in [0, 1]") {
        for (auto s : {
            make_state(0.5f, 0.0f, 0.7f, 0.2f, 1.0f, 0.4f),
            make_state(0.9f, 0.4f, 0.9f, 0.1f, 0.2f, 0.8f),
            make_state(0.1f,-0.3f, 0.05f,0.8f, 2.5f, 0.1f)
        }) {
            auto sig = WaveMirror::compute_spectral_signature(s);
            for (int i = 0; i < 9; ++i) {
                CHECK(sig[i] >= 0.0);
                CHECK(sig[i] <= 1.0);
            }
        }
    }

    SECTION("Band 0 equals clamp(dopamine, 0, 1)") {
        auto s = make_state(0.75f);
        auto sig = WaveMirror::compute_spectral_signature(s);
        CHECK_THAT(sig[0], WithinAbs(0.75, 1e-6));
    }

    SECTION("Band 4 equals boredom") {
        auto s = make_state(0.5f, 0.0f, 0.7f, 0.65f, 1.0f);
        auto sig = WaveMirror::compute_spectral_signature(s);
        CHECK_THAT(sig[4], WithinAbs(0.65, 1e-6));
    }

    SECTION("Band 6 equals 1 - atp") {
        auto s = make_state(0.5f, 0.0f, 0.8f);
        auto sig = WaveMirror::compute_spectral_signature(s);
        CHECK_THAT(sig[6], WithinAbs(0.2, 1e-6));
    }

    SECTION("Band 7 is 0 when td >= 0") {
        auto s = make_state(0.5f, 0.2f);
        auto sig = WaveMirror::compute_spectral_signature(s);
        CHECK_THAT(sig[7], WithinAbs(0.0, 1e-9));
    }

    SECTION("Band 7 is positive when td < 0") {
        auto s = make_state(0.5f, -0.3f);
        auto sig = WaveMirror::compute_spectral_signature(s);
        CHECK(sig[7] > 0.0);
    }
}

// ---------------------------------------------------------------------------
// §8 — compute_attention_focus
// ---------------------------------------------------------------------------

TEST_CASE("Phase121 §8 compute_attention_focus priority rules", "[Phase121]") {
    SECTION("FATIGUE when atp < 0.15") {
        auto s = make_state(0.5f, 0.0f, 0.05f, 0.2f, 1.0f);
        auto f = WaveMirror::compute_attention_focus(s);
        CHECK(f.mode == AttentionFocus::Mode::FATIGUE);
        CHECK(f.salience > 0.0);
        CHECK(f.salience <= 1.0);
    }

    SECTION("THREAT when td < -0.10 and atp ok") {
        auto s = make_state(0.5f, -0.35f, 0.7f, 0.2f, 1.0f);
        auto f = WaveMirror::compute_attention_focus(s);
        CHECK(f.mode == AttentionFocus::Mode::THREAT);
        CHECK(f.salience > 0.0);
    }

    SECTION("FATIGUE wins over THREAT when both conditions met") {
        auto s = make_state(0.5f, -0.40f, 0.05f, 0.2f, 1.0f);
        auto f = WaveMirror::compute_attention_focus(s);
        CHECK(f.mode == AttentionFocus::Mode::FATIGUE); // priority
    }

    SECTION("REWARD when td > 0.05 and D > 0.55") {
        auto s = make_state(0.80f, 0.20f, 0.7f, 0.1f, 0.5f);
        auto f = WaveMirror::compute_attention_focus(s);
        CHECK(f.mode == AttentionFocus::Mode::REWARD);
        CHECK(f.salience > 0.0);
    }

    SECTION("IDLE when boredom > 0.70 and entropy < 0.50") {
        auto s = make_state(0.5f, 0.0f, 0.7f, 0.85f, 0.20f);
        auto f = WaveMirror::compute_attention_focus(s);
        CHECK(f.mode == AttentionFocus::Mode::IDLE);
    }

    SECTION("CURIOUS is default when no other condition met") {
        auto s = make_state(0.4f, 0.0f, 0.6f, 0.3f, 1.2f);
        auto f = WaveMirror::compute_attention_focus(s);
        CHECK(f.mode == AttentionFocus::Mode::CURIOUS);
        CHECK(f.salience >= MIRROR_MIN_SALIENCE);
    }

    SECTION("Salience always in [0, 1] for all modes") {
        for (auto s : {
            make_state(0.5f, 0.0f, 0.05f, 0.2f, 1.0f),   // FATIGUE
            make_state(0.5f,-0.35f, 0.7f, 0.2f, 1.0f),   // THREAT
            make_state(0.8f, 0.20f, 0.7f, 0.1f, 0.5f),   // REWARD
            make_state(0.5f, 0.0f, 0.7f, 0.85f, 0.20f),  // IDLE
            make_state(0.4f, 0.0f, 0.6f, 0.3f, 1.2f)     // CURIOUS
        }) {
            auto f = WaveMirror::compute_attention_focus(s);
            CHECK(f.salience >= 0.0);
            CHECK(f.salience <= 1.0);
        }
    }
}

// ---------------------------------------------------------------------------
// §9 — update() metric derivation
// ---------------------------------------------------------------------------

TEST_CASE("Phase121 §9 update() metric derivation", "[Phase121]") {
    SECTION("Confident state: high D + high A + positive td") {
        WaveMirror wm;
        auto s = make_state(0.85f, 0.2f, 0.9f, 0.1f, 0.5f, 0.3f);
        wm.update(s);
        CHECK(wm.confidence() > 0.5);
    }

    SECTION("Confused state: high entropy + negative td") {
        WaveMirror wm;
        auto s = make_state(0.3f, -0.3f, 0.5f, 0.2f, 2.5f, 0.5f);
        wm.update(s);
        CHECK(wm.confusion() > 0.5);
    }

    SECTION("High load: high entropy + high torus energy") {
        WaveMirror wm;
        auto s = make_state(0.5f, 0.0f, 0.7f, 0.2f, 2.8f, 0.9f);
        wm.update(s);
        CHECK(wm.cognitive_load() > 0.5);
    }

    SECTION("Coherent state: high D + high A + low entropy") {
        WaveMirror wm;
        auto s = make_state(0.9f, 0.1f, 0.9f, 0.1f, 0.1f, 0.3f);
        wm.update(s);
        CHECK(wm.coherence() > 0.6);
    }

    SECTION("All metrics are in [0, 1] after any update") {
        WaveMirror wm;
        for (auto s : {
            make_state(0.9f, 0.4f, 0.9f, 0.05f, 0.1f, 0.2f),
            make_state(0.1f,-0.4f, 0.05f, 0.9f, 2.9f, 0.9f),
            make_state(0.5f, 0.0f, 0.5f, 0.5f, 1.5f, 0.5f)
        }) {
            wm.update(s);
            CHECK(wm.confidence()     >= 0.0); CHECK(wm.confidence()     <= 1.0);
            CHECK(wm.confusion()      >= 0.0); CHECK(wm.confusion()      <= 1.0);
            CHECK(wm.cognitive_load() >= 0.0); CHECK(wm.cognitive_load() <= 1.0);
            CHECK(wm.coherence()      >= 0.0); CHECK(wm.coherence()      <= 1.0);
            CHECK(wm.metacognitive()  >= 0.0); CHECK(wm.metacognitive()  <= 1.0);
        }
    }

    SECTION("Focus changes to THREAT after negative-TD update") {
        WaveMirror wm;
        wm.update(make_state(0.3f, -0.5f, 0.7f, 0.1f, 1.0f));
        CHECK(wm.attention_focus().mode == AttentionFocus::Mode::THREAT);
    }

    SECTION("Focus changes to FATIGUE after low-ATP update") {
        WaveMirror wm;
        wm.update(make_state(0.5f, 0.0f, 0.05f));
        CHECK(wm.attention_focus().mode == AttentionFocus::Mode::FATIGUE);
    }

    SECTION("Spectral signature has 9 elements all in [0, 1]") {
        WaveMirror wm;
        wm.update(make_state());
        auto sig = wm.spectral_signature();
        CHECK(sig.size() == 9);
        for (double v : sig) {
            CHECK(v >= 0.0);
            CHECK(v <= 1.0);
        }
    }
}

// ---------------------------------------------------------------------------
// §10 — Rolling-window smoothing
// ---------------------------------------------------------------------------

TEST_CASE("Phase121 §10 rolling-window smoothing", "[Phase121]") {
    SECTION("confidence smooths toward steady-state value") {
        WaveMirror wm;
        auto s = make_state(0.8f, 0.2f, 0.9f, 0.1f, 0.4f);
        // Feed same state for the full window
        for (int i = 0; i < MIRROR_HISTORY_WINDOW; ++i)
            wm.update(s);
        double steady = WaveMirror::compute_confidence(0.8, 0.2, 0.9);
        CHECK_THAT(wm.confidence(), WithinAbs(steady, 0.01));
    }

    SECTION("coherence smooths toward steady-state value") {
        WaveMirror wm;
        auto s = make_state(0.85f, 0.1f, 0.85f, 0.1f, 0.2f);
        for (int i = 0; i < MIRROR_HISTORY_WINDOW; ++i)
            wm.update(s);
        double steady = WaveMirror::compute_coherence(0.85, 0.85, 0.2);
        CHECK_THAT(wm.coherence(), WithinAbs(steady, 0.01));
    }

    SECTION("repeated identical updates are idempotent after window fills") {
        WaveMirror wm;
        auto s = make_state(0.6f, 0.05f, 0.8f, 0.2f, 1.0f);
        for (int i = 0; i < MIRROR_HISTORY_WINDOW + 2; ++i)
            wm.update(s);
        double v1 = wm.confidence();
        wm.update(s);
        CHECK_THAT(wm.confidence(), WithinAbs(v1, 1e-9));
    }
}

// ---------------------------------------------------------------------------
// §11 — snapshot() / accessor consistency
// ---------------------------------------------------------------------------

TEST_CASE("Phase121 §11 snapshot() accessor consistency", "[Phase121]") {
    WaveMirror wm;
    auto s = make_state(0.7f, 0.15f, 0.75f, 0.2f, 1.2f, 0.5f);
    wm.update(s);

    auto snap = wm.snapshot();

    SECTION("snapshot confidence matches confidence()") {
        CHECK_THAT(snap.confidence, WithinAbs(wm.confidence(), 1e-9));
    }
    SECTION("snapshot confusion matches confusion()") {
        CHECK_THAT(snap.confusion, WithinAbs(wm.confusion(), 1e-9));
    }
    SECTION("snapshot cognitive_load matches cognitive_load()") {
        CHECK_THAT(snap.cognitive_load, WithinAbs(wm.cognitive_load(), 1e-9));
    }
    SECTION("snapshot coherence matches coherence()") {
        CHECK_THAT(snap.coherence, WithinAbs(wm.coherence(), 1e-9));
    }
    SECTION("snapshot metacognitive matches metacognitive()") {
        CHECK_THAT(snap.metacognitive, WithinAbs(wm.metacognitive(), 1e-9));
    }
    SECTION("snapshot focus mode matches attention_focus()") {
        CHECK(snap.focus.mode == wm.attention_focus().mode);
    }
    SECTION("metacognitive = coherence * confidence * (1-confusion)") {
        double expected = snap.coherence * snap.confidence * (1.0 - snap.confusion);
        // clamp to [0,1]
        expected = expected < 0.0 ? 0.0 : (expected > 1.0 ? 1.0 : expected);
        CHECK_THAT(snap.metacognitive, WithinAbs(expected, 1e-6));
    }
}

// ---------------------------------------------------------------------------
// §12 — describe()
// ---------------------------------------------------------------------------

TEST_CASE("Phase121 §12 describe()", "[Phase121]") {
    SECTION("Returns non-empty string") {
        WaveMirror wm;
        CHECK(!wm.describe().empty());
    }

    SECTION("Contains 'conf=' metric label") {
        WaveMirror wm;
        CHECK(wm.describe().find("conf=") != std::string::npos);
    }

    SECTION("Contains 'coh=' metric label") {
        WaveMirror wm;
        CHECK(wm.describe().find("coh=") != std::string::npos);
    }

    SECTION("Contains the current focus mode name") {
        WaveMirror wm;
        auto s = make_state(0.3f, -0.4f, 0.7f, 0.2f, 1.0f); // THREAT
        wm.update(s);
        std::string desc = wm.describe();
        CHECK(desc.find("threat") != std::string::npos);
    }

    SECTION("Saturated state described as [saturated]") {
        WaveMirror wm;
        // entropy at ceiling + high torus energy -> load near 1
        auto s = make_state(0.5f, 0.0f, 0.7f, 0.2f,
                            static_cast<float>(MIRROR_ENTROPY_CEILING), 1.0f);
        wm.update(s);
        if (wm.cognitive_load() >= MIRROR_LOAD_SATURATED)
            CHECK(wm.describe().find("[saturated]") != std::string::npos);
    }

    SECTION("Coherent state described as [coherent] or [nominal]") {
        WaveMirror wm;
        auto s = make_state(0.9f, 0.1f, 0.9f, 0.05f, 0.1f, 0.1f);
        for (int i = 0; i < MIRROR_HISTORY_WINDOW; ++i)
            wm.update(s);
        std::string desc = wm.describe();
        bool has_label = desc.find("[coherent]") != std::string::npos ||
                         desc.find("[nominal]")  != std::string::npos;
        CHECK(has_label);
    }
}
