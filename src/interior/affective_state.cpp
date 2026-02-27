/**
 * @file affective_state.cpp
 * @brief AffectiveState implementation -- Phase 120
 *
 * Soft membership scoring for 11 Affect labels derived from the ENGS
 * neurochemical state [D, td_error, ATP, boredom, entropy].
 */

#include <nikola/interior/affective_state.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace nikola::interior {

// ============================================================================
// Constructor
// ============================================================================

AffectiveState::AffectiveState() noexcept {
    scores_.fill(0.0);
    induced_.fill(0.0);
    scores_[static_cast<int>(Affect::NEUTRAL)] = 1.0;
    dominant_ = Affect::NEUTRAL;
}

// ============================================================================
// Pure static helpers
// ============================================================================

double AffectiveState::compute_valence(double d,
                                        double td_error,
                                        double atp) noexcept {
    double v = 2.0 * (d - DOPAMINE_EQUILIBRIUM)
             + 1.0 * td_error
             - 0.4 * (atp < ATP_ANXIETY_THRESHOLD
                      ? (ATP_ANXIETY_THRESHOLD - atp) / ATP_ANXIETY_THRESHOLD
                      : 0.0);
    return clamp(v, -1.0, 1.0);
}

double AffectiveState::compute_arousal(double entropy, double boredom) noexcept {
    double n_eff = clamp01(entropy / ENTROPY_AROUSAL_CEILING);
    return clamp01(n_eff + 0.25 * boredom);
}

AffectiveState::IntensityMap
AffectiveState::compute_scores(const nikola::autonomy::NikolaState& s) noexcept {
    const double d  = static_cast<double>(s.dopamine);
    const double td = static_cast<double>(s.td_error);
    const double a  = static_cast<double>(s.atp);
    const double b  = static_cast<double>(s.boredom);
    const double H  = static_cast<double>(s.entropy);

    const double N_eff = clamp01(H / ENTROPY_AROUSAL_CEILING);
    const double S_eff = clamp01(a * 0.7 + d * 0.3);

    IntensityMap m{};

    // CURIOSITY: high boredom + adequate entropy + ATP ok
    {
        double boredom_drive = soft_step(b, BOREDOM_THRESHOLD, 8.0);
        double entropy_ok    = soft_step(H, 0.3, 5.0);
        double atp_ok        = soft_step(a, 0.15, 10.0);
        m[0] = clamp01(boredom_drive * entropy_ok * atp_ok);
    }

    // FRUSTRATION: low dopamine + negative td_error
    {
        double low_d  = 1.0 - soft_step(d, 0.35, 10.0);
        double neg_td = soft_step(-td, 0.10, 8.0);
        m[1] = clamp01(low_d * 0.6 + neg_td * 0.4);
    }

    // SATISFACTION: high dopamine + positive td_error
    {
        double high_d = soft_step(d, 0.60, 10.0);
        double pos_td = soft_step(td, 0.05, 8.0);
        m[2] = clamp01(high_d * 0.6 + pos_td * 0.4);
    }

    // CONCERN: negative td + high entropy
    {
        double neg_td = soft_step(-td, 0.05, 8.0);
        double high_H = soft_step(H, 1.0, 3.0);
        m[3] = clamp01(neg_td * 0.5 + high_H * neg_td * 0.5);
    }

    // BOREDOM: very high boredom + low entropy
    {
        double high_b = soft_step(b, 0.70, 10.0);
        double low_H  = 1.0 - soft_step(H, 0.8, 4.0);
        m[4] = clamp01(high_b * low_H);
    }

    // INTEREST: moderate-high dopamine + high entropy
    {
        double mod_d  = soft_step(d, 0.40, 6.0);
        double high_H = soft_step(H, 1.2, 3.0);
        m[5] = clamp01(mod_d * high_H);
    }

    // CONFUSION: low dopamine + very high entropy
    {
        double low_d  = 1.0 - soft_step(d, 0.35, 10.0);
        double very_H = soft_step(H, 2.0, 4.0);
        m[6] = clamp01(low_d * very_H);
    }

    // CONFIDENCE: high dopamine + high serotonin proxy + high ATP
    {
        double high_d = soft_step(d, 0.60, 10.0);
        double high_S = soft_step(S_eff, 0.65, 8.0);
        double high_a = soft_step(a, 0.40, 8.0);
        m[7] = clamp01(high_d * 0.4 + high_S * 0.3 + high_a * 0.3);
    }

    // ANXIETY: low ATP + high norepinephrine proxy
    {
        double low_a  = 1.0 - soft_step(a, ATP_ANXIETY_THRESHOLD, 12.0);
        double high_N = soft_step(N_eff, 0.50, 6.0);
        m[8] = clamp01(low_a * 0.6 + low_a * high_N * 0.4);
    }

    // EXCITEMENT: high dopamine + high entropy + positive td
    {
        double high_d = soft_step(d, 0.55, 8.0);
        double high_H = soft_step(H, 1.5, 3.0);
        double pos_td = soft_step(td, 0.05, 8.0);
        m[9] = clamp01(high_d * high_H * pos_td);
    }

    // NEUTRAL: active when nothing else is
    {
        double max_score = 0.0;
        for (int i = 0; i < 10; ++i) {
            if (m[i] > max_score) max_score = m[i];
        }
        m[10] = clamp01(1.0 - max_score);
    }

    return m;
}

// ============================================================================
// update()
// ============================================================================

void AffectiveState::update(const nikola::autonomy::NikolaState& s) noexcept {
    const Affect prev = dominant_;

    valence_ = compute_valence(
        static_cast<double>(s.dopamine),
        static_cast<double>(s.td_error),
        static_cast<double>(s.atp));
    arousal_ = compute_arousal(
        static_cast<double>(s.entropy),
        static_cast<double>(s.boredom));

    scores_ = compute_scores(s);

    // Decay induced weights
    for (int i = 0; i < AFFECT_COUNT; ++i) {
        induced_[i] *= INDUCED_AFFECT_DECAY;
        if (induced_[i] < INDUCED_AFFECT_MIN) induced_[i] = 0.0;
    }

    // Dominant = argmax(base + induced)
    double best = -1.0;
    dominant_ = Affect::NEUTRAL;
    for (int i = 0; i < AFFECT_COUNT; ++i) {
        double total = clamp01(scores_[i] + induced_[i]);
        if (total > best) {
            best      = total;
            dominant_ = static_cast<Affect>(i);
        }
    }

    if (dominant_ != prev && on_affect_change) {
        on_affect_change(prev, dominant_, best);
    }
}

// ============================================================================
// Discrete affect queries
// ============================================================================

Affect AffectiveState::current_affect() const noexcept {
    return dominant_;
}

double AffectiveState::get_affect_intensity(Affect a) const noexcept {
    auto idx = static_cast<int>(a);
    if (idx < 0 || idx >= AFFECT_COUNT) return 0.0;
    return clamp01(scores_[idx] + induced_[idx]);
}

std::map<Affect, double> AffectiveState::get_all_affects() const {
    std::map<Affect, double> out;
    for (int i = 0; i < AFFECT_COUNT; ++i) {
        out[static_cast<Affect>(i)] = clamp01(scores_[i] + induced_[i]);
    }
    return out;
}

// ============================================================================
// External induction
// ============================================================================

void AffectiveState::induce_affect(Affect a, double intensity) {
    if (intensity < 0.0 || intensity > 1.0) {
        throw std::invalid_argument(
            "AffectiveState::induce_affect: intensity must be in [0, 1]");
    }
    auto idx = static_cast<int>(a);
    if (idx < 0 || idx >= AFFECT_COUNT) {
        throw std::invalid_argument(
            "AffectiveState::induce_affect: invalid Affect value");
    }
    induced_[idx] = std::min(1.0, induced_[idx] + intensity);
}

// ============================================================================
// Neurochemical consequence table
// ============================================================================

std::map<std::string, double>
AffectiveState::affect_to_neurochemistry(Affect a) {
    switch (a) {
        case Affect::CURIOSITY:
            return {{"dopamine", +0.10}, {"serotonin", -0.05}, {"norepinephrine", +0.15}};
        case Affect::FRUSTRATION:
            return {{"dopamine", -0.15}, {"serotonin", -0.10}, {"norepinephrine", +0.20}};
        case Affect::SATISFACTION:
            return {{"dopamine", +0.20}, {"serotonin", +0.15}, {"norepinephrine", -0.05}};
        case Affect::CONCERN:
            return {{"dopamine", -0.05}, {"serotonin", -0.08}, {"norepinephrine", +0.10}};
        case Affect::BOREDOM:
            return {{"dopamine", -0.10}, {"serotonin", +0.05}, {"norepinephrine", -0.10}};
        case Affect::INTEREST:
            return {{"dopamine", +0.10}, {"serotonin", +0.05}, {"norepinephrine", +0.10}};
        case Affect::CONFUSION:
            return {{"dopamine", -0.10}, {"serotonin", -0.05}, {"norepinephrine", +0.15}};
        case Affect::CONFIDENCE:
            return {{"dopamine", +0.10}, {"serotonin", +0.20}, {"norepinephrine", -0.10}};
        case Affect::ANXIETY:
            return {{"dopamine", -0.10}, {"serotonin", -0.15}, {"norepinephrine", +0.25}};
        case Affect::EXCITEMENT:
            return {{"dopamine", +0.25}, {"serotonin", -0.05}, {"norepinephrine", +0.20}};
        case Affect::NEUTRAL:
        default:
            return {{"dopamine", 0.0}, {"serotonin", 0.0}, {"norepinephrine", 0.0}};
    }
}

// ============================================================================
// Attention modulation
// ============================================================================

double AffectiveState::attention_weight(double entropy) const noexcept {
    const double high_entropy_bonus =
        get_affect_intensity(Affect::CURIOSITY)  * 0.4 +
        get_affect_intensity(Affect::INTEREST)   * 0.3 +
        get_affect_intensity(Affect::EXCITEMENT) * 0.3;

    const double high_entropy_penalty =
        get_affect_intensity(Affect::ANXIETY)  * 0.3 +
        get_affect_intensity(Affect::CONCERN)  * 0.2;

    const double h_norm = clamp01(entropy / ENTROPY_AROUSAL_CEILING);

    double w = 1.0
             + (high_entropy_bonus   * h_norm)
             - (high_entropy_penalty * h_norm);

    return clamp(w, 0.5, 2.0);
}

// ============================================================================
// Description
// ============================================================================

std::string AffectiveState::describe_state() const {
    struct Pair { int idx; double val; };
    std::array<Pair, AFFECT_COUNT> ranked;
    for (int i = 0; i < AFFECT_COUNT; ++i) {
        ranked[i] = {i, clamp01(scores_[i] + induced_[i])};
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const Pair& x, const Pair& y){ return x.val > y.val; });

    std::ostringstream oss;
    oss << affect_name(static_cast<Affect>(ranked[0].idx));

    if (ranked[1].val > 0.25) {
        oss << " and slightly "
            << affect_name(static_cast<Affect>(ranked[1].idx));
    }

    oss << " (valence=" << std::fixed << std::setprecision(2) << valence_
        << ", arousal="  << std::fixed << std::setprecision(2) << arousal_
        << ")";
    return oss.str();
}

} // namespace nikola::interior
