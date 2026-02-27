/**
 * @file wave_mirror.cpp
 * @brief Phase 121 -- WaveMirror implementation
 */

#include <nikola/interior/wave_mirror.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace nikola::interior {

const char* AttentionFocus::mode_name(Mode m) noexcept {
    switch (m) {
        case Mode::CURIOUS:  return "curious";
        case Mode::REWARD:   return "reward";
        case Mode::THREAT:   return "threat";
        case Mode::FATIGUE:  return "fatigue";
        case Mode::IDLE:     return "idle";
        default:             return "unknown";
    }
}

// ============================================================================
// Constructor
// ============================================================================

WaveMirror::WaveMirror() noexcept {
    conf_hist_.fill(0.0);
    coh_hist_.fill(0.0);
    current_ = MirrorSnapshot{};
}

// ============================================================================
// Rolling-window helper
// ============================================================================

double WaveMirror::smoothed_mean(
    const std::array<double, MIRROR_HISTORY_WINDOW>& h,
    int filled) const noexcept
{
    if (filled <= 0) return 0.0;
    int n = std::min(filled, MIRROR_HISTORY_WINDOW);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += h[i];
    return sum / static_cast<double>(n);
}

// ============================================================================
// Pure static metrics
// ============================================================================

double WaveMirror::compute_confidence(double D, double td, double A) noexcept {
    double pos_td = td > 0.0 ? td : 0.0;
    double neg_td = td < 0.0 ? -td : 0.0;
    return clamp01(D * 0.5 + pos_td * 0.3 - neg_td * 0.2 + A * 0.2);
}

double WaveMirror::compute_confusion(double H, double td) noexcept {
    double h_norm = clamp01(H / MIRROR_ENTROPY_CEILING);
    double neg_td = td < 0.0 ? clamp01(-td) : 0.0;
    return clamp01(h_norm * 0.6 + neg_td * 0.4);
}

double WaveMirror::compute_cognitive_load(double H, double E) noexcept {
    double h_norm = clamp01(H / MIRROR_ENTROPY_CEILING);
    double e_norm = clamp01(E);
    return clamp01(h_norm * 0.7 + e_norm * 0.3);
}

double WaveMirror::compute_coherence(double D, double A, double H) noexcept {
    double h_inv = 1.0 - clamp01(H / MIRROR_ENTROPY_CEILING);
    return clamp01(D * 0.4 + A * 0.4 + h_inv * 0.2);
}

std::array<double, 9> WaveMirror::compute_spectral_signature(
    const NikolaState& s) noexcept
{
    double D  = static_cast<double>(s.dopamine);
    double td = static_cast<double>(s.td_error);
    double A  = static_cast<double>(s.atp);
    double B  = static_cast<double>(s.boredom);
    double H  = static_cast<double>(s.entropy);
    double E  = static_cast<double>(s.torus_energy);

    double conf = compute_confidence(D, td, A);
    double coh  = compute_coherence(D, A, H);

    std::array<double, 9> sig{};
    sig[0] = clamp01(D);                                   // dopamine
    sig[1] = clamp01(A * 0.7 + D * 0.3);                  // serotonin proxy
    sig[2] = clamp01(H / MIRROR_ENTROPY_CEILING);          // NE proxy (arousal)
    sig[3] = clamp01(std::fabs(td));                       // |prediction error|
    sig[4] = clamp01(B);                                   // boredom
    sig[5] = clamp01(E);                                   // torus activity
    sig[6] = clamp01(1.0 - A);                             // depletion
    sig[7] = td < 0.0 ? clamp01(-td) : 0.0;               // aversive signal
    sig[8] = clamp01(conf * (1.0 - conf) * (1.0 - coh));  // metacog uncertainty
    return sig;
}

AttentionFocus WaveMirror::compute_attention_focus(
    const NikolaState& s) noexcept
{
    double D  = static_cast<double>(s.dopamine);
    double td = static_cast<double>(s.td_error);
    double A  = static_cast<double>(s.atp);
    double B  = static_cast<double>(s.boredom);
    double H  = static_cast<double>(s.entropy);

    AttentionFocus f;

    // Priority: FATIGUE > THREAT > REWARD > IDLE > CURIOUS
    if (A < 0.15) {
        f.mode     = AttentionFocus::Mode::FATIGUE;
        f.salience = clamp01(1.0 - A / 0.15);
        return f;
    }

    if (td < -0.10) {
        f.mode     = AttentionFocus::Mode::THREAT;
        f.salience = clamp01(-td);
        return f;
    }

    if (td > 0.05 && D > 0.55) {
        f.mode     = AttentionFocus::Mode::REWARD;
        f.salience = clamp01(td * 2.0 + (D - 0.55) * 2.0);
        return f;
    }

    if (B > 0.70 && H < 0.50) {
        f.mode     = AttentionFocus::Mode::IDLE;
        f.salience = clamp01((B - 0.70) * 3.0 + (0.50 - H) * 0.5);
        return f;
    }

    // Default: curious / exploratory
    f.mode     = AttentionFocus::Mode::CURIOUS;
    f.salience = clamp01(H / MIRROR_ENTROPY_CEILING * 0.6 + B * 0.4);
    if (f.salience < MIRROR_MIN_SALIENCE)
        f.salience = MIRROR_MIN_SALIENCE;
    return f;
}

// ============================================================================
// update() -- main entry point
// ============================================================================

void WaveMirror::update(const NikolaState& s) noexcept {
    double D  = static_cast<double>(s.dopamine);
    double td = static_cast<double>(s.td_error);
    double A  = static_cast<double>(s.atp);
    double H  = static_cast<double>(s.entropy);
    double E  = static_cast<double>(s.torus_energy);

    // Instantaneous metrics
    double inst_conf = compute_confidence(D, td, A);
    double inst_coh  = compute_coherence(D, A, H);

    // Update rolling window
    conf_hist_[hist_idx_] = inst_conf;
    coh_hist_[hist_idx_]  = inst_coh;
    hist_idx_ = (hist_idx_ + 1) % MIRROR_HISTORY_WINDOW;
    if (tick_count_ < std::numeric_limits<int>::max())
        ++tick_count_;

    int filled = std::min(tick_count_, MIRROR_HISTORY_WINDOW);

    // Smoothed values
    current_.confidence     = smoothed_mean(conf_hist_, filled);
    current_.coherence      = smoothed_mean(coh_hist_,  filled);
    current_.confusion      = compute_confusion(H, td);
    current_.cognitive_load = compute_cognitive_load(H, E);
    current_.metacognitive  =
        clamp01(current_.coherence
                * current_.confidence
                * (1.0 - current_.confusion));
    current_.focus              = compute_attention_focus(s);
    current_.spectral_signature = compute_spectral_signature(s);
}

// ============================================================================
// Accessors
// ============================================================================

double WaveMirror::confidence()     const noexcept { return current_.confidence; }
double WaveMirror::confusion()      const noexcept { return current_.confusion; }
double WaveMirror::cognitive_load() const noexcept { return current_.cognitive_load; }
double WaveMirror::coherence()      const noexcept { return current_.coherence; }
double WaveMirror::metacognitive()  const noexcept { return current_.metacognitive; }

AttentionFocus WaveMirror::attention_focus() const noexcept {
    return current_.focus;
}

std::array<double, 9> WaveMirror::spectral_signature() const noexcept {
    return current_.spectral_signature;
}

MirrorSnapshot WaveMirror::snapshot() const noexcept {
    return current_;
}

// ============================================================================
// describe()
// ============================================================================

std::string WaveMirror::describe() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "WaveMirror["
        << "focus=" << current_.focus.mode_name()
        << " sal=" << current_.focus.salience
        << " conf="  << current_.confidence
        << " coh="   << current_.coherence
        << " load="  << current_.cognitive_load
        << " confus=" << current_.confusion
        << " meta="  << current_.metacognitive
        << "]";

    // Append a plain-language assessment
    if (current_.cognitive_load >= MIRROR_LOAD_SATURATED)
        oss << " [saturated]";
    else if (current_.coherence >= MIRROR_COHERENCE_HIGH)
        oss << " [coherent]";
    else if (current_.confusion > 0.60)
        oss << " [confused]";
    else
        oss << " [nominal]";

    return oss.str();
}

} // namespace nikola::interior
