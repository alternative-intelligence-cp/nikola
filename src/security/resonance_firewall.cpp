/**
 * @file src/security/resonance_firewall.cpp
 * @brief v0.3.0 — ResonanceFirewall implementation.
 *
 * Self-contained radix-2 Cooley-Tukey FFT (no external dependencies).
 * Spectral entropy, autocorrelation, and hazardous pattern cross-correlation.
 */

#include <nikola/security/resonance_firewall.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace nikola::security {

// ── Construction ────────────────────────────────────────────────────────────

ResonanceFirewall::ResonanceFirewall()
    : cfg_{}
{
    if (cfg_.max_autocorr_lags == 0)
        cfg_.max_autocorr_lags = cfg_.window_size / 4;
}

ResonanceFirewall::ResonanceFirewall(ResonanceFirewallConfig config)
    : cfg_(std::move(config))
{
    if (cfg_.max_autocorr_lags == 0)
        cfg_.max_autocorr_lags = cfg_.window_size / 4;
}

// ── Main validation ─────────────────────────────────────────────────────────

FirewallVerdict ResonanceFirewall::validate(
    const std::vector<std::complex<double>>& waveform) const
{
    ++total_validations_;
    FirewallVerdict v;

    // ── Check 1: Amplitude bounds ────────────────────────────────────────────
    double max_amp = 0.0;
    for (const auto& sample : waveform) {
        double amp = std::abs(sample);
        if (amp > max_amp) max_amp = amp;
    }
    v.max_amplitude = max_amp;

    if (max_amp > cfg_.max_amplitude) {
        v.action = FirewallAction::REJECT;
        v.reason = "Amplitude overflow: " + std::to_string(max_amp)
                 + " > " + std::to_string(cfg_.max_amplitude);
        ++total_rejections_;
        return v;
    }

    // ── Check 2: Spectral entropy ────────────────────────────────────────────
    v.spectral_entropy = compute_spectral_entropy(waveform);

    if (v.spectral_entropy < cfg_.min_entropy) {
        v.action = FirewallAction::REJECT;
        v.reason = "Low entropy (Siren Attack): H=" + std::to_string(v.spectral_entropy)
                 + " < " + std::to_string(cfg_.min_entropy);
        ++total_rejections_;
        return v;
    }

    if (v.spectral_entropy > cfg_.max_entropy) {
        if (cfg_.dampen_high_entropy) {
            v.action = FirewallAction::DAMPEN;
            v.damping_factor = cfg_.high_entropy_damping;
            v.reason = "High entropy (Thermal Attack): H=" + std::to_string(v.spectral_entropy)
                     + " > " + std::to_string(cfg_.max_entropy)
                     + "; applying " + std::to_string(cfg_.high_entropy_damping * 100.0)
                     + "% damping";
        } else {
            v.action = FirewallAction::REJECT;
            v.reason = "High entropy (Thermal Attack): H=" + std::to_string(v.spectral_entropy)
                     + " > " + std::to_string(cfg_.max_entropy);
            ++total_rejections_;
            return v;
        }
    }

    // ── Check 3: Temporal autocorrelation ────────────────────────────────────
    double peak_R = 0.0;
    size_t peak_lag = 0;
    size_t max_lags = std::min(cfg_.max_autocorr_lags, waveform.size() / 2);

    for (size_t lag = 1; lag <= max_lags; ++lag) {
        double R = compute_autocorrelation(waveform, lag);
        if (R > peak_R) {
            peak_R   = R;
            peak_lag = lag;
        }
    }

    v.peak_autocorrelation = peak_R;
    v.peak_autocorr_lag    = peak_lag;

    if (peak_R > cfg_.autocorr_threshold) {
        // If already dampening, upgrade to reject; otherwise penalize
        if (v.action == FirewallAction::DAMPEN) {
            v.action = FirewallAction::REJECT;
            v.reason += "; also repeating loop R_xx=" + std::to_string(peak_R)
                     + " at lag " + std::to_string(peak_lag);
            ++total_rejections_;
            return v;
        }
        v.action = FirewallAction::PENALIZE;
        v.reason = "Repeating loop detected: R_xx(" + std::to_string(peak_lag)
                 + ")=" + std::to_string(peak_R) + " > "
                 + std::to_string(cfg_.autocorr_threshold);
    }

    // ── Check 4: Hazardous pattern database ──────────────────────────────────
    if (!patterns_.empty()) {
        auto input_spectrum = compute_fft(waveform);

        double best_corr = 0.0;
        int    best_idx  = -1;

        for (size_t i = 0; i < patterns_.size(); ++i) {
            double corr = cross_correlate(input_spectrum, patterns_[i].spectrum);
            if (corr > best_corr) {
                best_corr = corr;
                best_idx  = static_cast<int>(i);
            }
        }

        v.hazard_correlation   = best_corr;
        v.matched_pattern_idx  = best_idx;

        if (best_corr > cfg_.pattern_threshold) {
            v.action = FirewallAction::REJECT;
            v.reason = "Matched hazardous pattern '"
                     + patterns_[static_cast<size_t>(best_idx)].name
                     + "' (corr=" + std::to_string(best_corr) + ")";
            ++total_rejections_;
            return v;
        }
    }

    return v;
}

bool ResonanceFirewall::is_safe(
    const std::vector<std::complex<double>>& waveform) const
{
    return validate(waveform).allowed();
}

// ── Hazardous pattern management ────────────────────────────────────────────

void ResonanceFirewall::add_hazardous_pattern(
    const std::string& name,
    const std::string& source,
    const std::vector<std::complex<double>>& signal)
{
    HazardousPattern pat;
    pat.name     = name;
    pat.source   = source;
    pat.spectrum = compute_fft(signal);
    patterns_.push_back(std::move(pat));
}

// ── Spectral entropy ────────────────────────────────────────────────────────

double ResonanceFirewall::compute_spectral_entropy(
    const std::vector<std::complex<double>>& signal)
{
    if (signal.empty()) return 0.0;

    auto spectrum = compute_fft(signal);

    // Compute PSD: P[k] = |X[k]|²
    std::vector<double> psd(spectrum.size());
    double total_power = 0.0;

    for (size_t k = 0; k < spectrum.size(); ++k) {
        psd[k] = std::norm(spectrum[k]);  // |X[k]|²
        total_power += psd[k];
    }

    if (total_power < 1e-30) return 0.0;  // silence

    // Normalize to probability distribution and compute Shannon entropy
    double H = 0.0;
    for (size_t k = 0; k < psd.size(); ++k) {
        double p = psd[k] / total_power;
        if (p > 1e-30) {
            H -= p * std::log2(p);
        }
    }

    return H;
}

// ── Temporal autocorrelation ────────────────────────────────────────────────

double ResonanceFirewall::compute_autocorrelation(
    const std::vector<std::complex<double>>& signal, size_t lag)
{
    if (signal.empty() || lag >= signal.size()) return 0.0;

    // Compute ||x||² (normalization)
    double norm_sq = 0.0;
    for (const auto& s : signal) {
        norm_sq += std::norm(s);
    }
    if (norm_sq < 1e-30) return 0.0;

    // R_xx(τ) = Σ x[n] · conj(x[n+τ]) / ||x||²
    std::complex<double> sum{0.0, 0.0};
    for (size_t n = 0; n + lag < signal.size(); ++n) {
        sum += signal[n] * std::conj(signal[n + lag]);
    }

    // Return real part of normalized autocorrelation
    return std::abs(sum) / norm_sq;
}

// ── FFT (radix-2 Cooley-Tukey) ──────────────────────────────────────────────

size_t ResonanceFirewall::next_pow2(size_t n) noexcept {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

void ResonanceFirewall::bit_reverse_permute(
    std::vector<std::complex<double>>& v)
{
    size_t n = v.size();
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) std::swap(v[i], v[j]);
    }
}

std::vector<std::complex<double>> ResonanceFirewall::compute_fft(
    const std::vector<std::complex<double>>& signal)
{
    size_t n = next_pow2(signal.size());
    std::vector<std::complex<double>> buf(n, {0.0, 0.0});

    // Copy signal into buffer (zero-pad if needed)
    for (size_t i = 0; i < signal.size(); ++i) {
        buf[i] = signal[i];
    }

    bit_reverse_permute(buf);

    // Iterative radix-2 Cooley-Tukey
    for (size_t len = 2; len <= n; len <<= 1) {
        double angle = -2.0 * M_PI / static_cast<double>(len);
        std::complex<double> w_step{std::cos(angle), std::sin(angle)};

        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w{1.0, 0.0};
            for (size_t j = 0; j < len / 2; ++j) {
                auto u = buf[i + j];
                auto t = w * buf[i + j + len / 2];
                buf[i + j]           = u + t;
                buf[i + j + len / 2] = u - t;
                w *= w_step;
            }
        }
    }

    return buf;
}

std::vector<std::complex<double>> ResonanceFirewall::compute_ifft(
    const std::vector<std::complex<double>>& spectrum)
{
    // IFFT = conj(FFT(conj(X))) / N
    size_t n = spectrum.size();
    std::vector<std::complex<double>> conj_input(n);
    for (size_t i = 0; i < n; ++i) {
        conj_input[i] = std::conj(spectrum[i]);
    }

    auto result = compute_fft(conj_input);
    double inv_n = 1.0 / static_cast<double>(n);
    for (auto& v : result) {
        v = std::conj(v) * inv_n;
    }
    return result;
}

// ── Cross-correlation ───────────────────────────────────────────────────────

double ResonanceFirewall::cross_correlate(
    const std::vector<std::complex<double>>& spectrum_a,
    const std::vector<std::complex<double>>& spectrum_b)
{
    // Ensure same size
    size_t n = std::max(spectrum_a.size(), spectrum_b.size());
    n = next_pow2(n);

    // Zero-pad spectra to same length
    auto a = spectrum_a;
    auto b = spectrum_b;
    a.resize(n, {0.0, 0.0});
    b.resize(n, {0.0, 0.0});

    // Cross-correlation in frequency domain: F* · G
    std::vector<std::complex<double>> product(n);
    for (size_t k = 0; k < n; ++k) {
        product[k] = std::conj(a[k]) * b[k];
    }

    // IFFT to get cross-correlation in time domain
    auto xcorr = compute_ifft(product);

    // Find peak magnitude
    double peak = 0.0;
    for (const auto& v : xcorr) {
        double mag = std::abs(v);
        if (mag > peak) peak = mag;
    }

    // Normalize: sqrt(||a||² · ||b||²)
    double norm_a = 0.0, norm_b = 0.0;
    for (size_t k = 0; k < n; ++k) {
        norm_a += std::norm(a[k]);
        norm_b += std::norm(b[k]);
    }

    double denom = std::sqrt(norm_a * norm_b);
    if (denom < 1e-30) return 0.0;

    // IFFT already divides by N, but norms are in freq domain (N× time energy).
    // Correct normalization: multiply peak by N to compensate.
    return peak * static_cast<double>(n) / denom;
}

}  // namespace nikola::security
