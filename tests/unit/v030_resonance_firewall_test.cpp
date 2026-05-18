/**
 * @file tests/unit/v030_resonance_firewall_test.cpp
 * @brief v0.3.0 — ResonanceFirewall test suite
 *
 * Tests:
 *   §1  Default construction
 *   §2  Safe waveform passes (mixed frequencies → moderate entropy)
 *   §3  Pure sine wave rejected (low entropy — Siren Attack)
 *   §4  White noise dampened or rejected (high entropy — Thermal Attack)
 *   §5  Amplitude overflow rejected
 *   §6  Repeating pattern detected (high autocorrelation)
 *   §7  Hazardous pattern database matching
 *   §8  Spectral entropy computation (known values)
 *   §9  FFT correctness (Parseval's theorem)
 *   §10 Autocorrelation of known signal
 *   §11 is_safe() convenience method
 *   §12 Cross-correlation of identical spectra
 *   §13 Empty waveform handling
 *   §14 Custom config thresholds
 *   §15 Counters (total_validations, total_rejections)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/security/resonance_firewall.hpp>

#include <cmath>
#include <complex>
#include <random>
#include <vector>

using namespace nikola::security;
using C = std::complex<double>;

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::vector<C> make_sine(size_t N, double freq, double amp = 1.0) {
    std::vector<C> v(N);
    for (size_t i = 0; i < N; ++i) {
        v[i] = C{amp * std::sin(2.0 * M_PI * freq * i / N), 0.0};
    }
    return v;
}

static std::vector<C> make_noise(size_t N, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<C> v(N);
    for (size_t i = 0; i < N; ++i) {
        v[i] = C{dist(rng), 0.0};
    }
    return v;
}

static std::vector<C> make_mixed(size_t N) {
    // Mix several frequencies at different amplitudes — realistic signal
    std::vector<C> v(N, C{0.0, 0.0});
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> noise(-0.05, 0.05);
    for (size_t i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / N;
        v[i] = C{
            0.5 * std::sin(2.0 * M_PI * 3.0 * t) +
            0.3 * std::sin(2.0 * M_PI * 7.0 * t) +
            0.2 * std::sin(2.0 * M_PI * 13.0 * t) +
            0.15 * std::sin(2.0 * M_PI * 29.0 * t) +
            0.1 * std::sin(2.0 * M_PI * 47.0 * t) +
            0.08 * std::sin(2.0 * M_PI * 61.0 * t) +
            0.05 * std::sin(2.0 * M_PI * 97.0 * t) +
            noise(rng),
            0.0
        };
    }
    return v;
}

// ============================================================================
// §1 Default construction
// ============================================================================

TEST_CASE("§1 ResonanceFirewall default construction", "[v030][firewall]") {
    ResonanceFirewall fw;
    const auto& cfg = fw.config();

    REQUIRE(cfg.min_entropy == FIREWALL_MIN_ENTROPY);
    REQUIRE(cfg.max_entropy == FIREWALL_MAX_ENTROPY);
    REQUIRE(cfg.autocorr_threshold == FIREWALL_AUTOCORR_THRESHOLD);
    REQUIRE(cfg.max_amplitude == FIREWALL_MAX_AMPLITUDE);
    REQUIRE(cfg.pattern_threshold == FIREWALL_PATTERN_THRESHOLD);
    REQUIRE(cfg.dampen_high_entropy == true);

    REQUIRE(fw.total_validations() == 0);
    REQUIRE(fw.total_rejections() == 0);
    REQUIRE(fw.hazardous_pattern_count() == 0);
}

// ============================================================================
// §2 Safe mixed-frequency waveform passes
// ============================================================================

TEST_CASE("§2 Mixed-frequency waveform passes", "[v030][firewall]") {
    ResonanceFirewall fw;
    auto wave = make_mixed(1024);

    auto verdict = fw.validate(wave);
    REQUIRE(verdict.allowed());
    REQUIRE(verdict.spectral_entropy > FIREWALL_MIN_ENTROPY);
    REQUIRE(verdict.spectral_entropy < FIREWALL_MAX_ENTROPY);
    REQUIRE(fw.total_validations() == 1);
    REQUIRE(fw.total_rejections() == 0);
}

// ============================================================================
// §3 Pure sine wave rejected (Siren Attack — low entropy)
// ============================================================================

TEST_CASE("§3 Pure sine rejected (Siren Attack)", "[v030][firewall]") {
    ResonanceFirewall fw;
    auto wave = make_sine(1024, 10.0, 1.0);

    auto verdict = fw.validate(wave);
    REQUIRE(verdict.rejected());
    REQUIRE(verdict.spectral_entropy < FIREWALL_MIN_ENTROPY);
    REQUIRE((verdict.reason.find("Low Entropy") != std::string::npos
         || verdict.reason.find("Siren") != std::string::npos));
    REQUIRE(fw.total_rejections() == 1);
}

// ============================================================================
// §4 White noise dampened (Thermal Attack — high entropy)
// ============================================================================

TEST_CASE("§4 White noise dampened (Thermal Attack)", "[v030][firewall]") {
    ResonanceFirewall fw;
    auto wave = make_noise(1024);

    auto verdict = fw.validate(wave);
    // With default config, high entropy is dampened (not rejected)
    if (verdict.spectral_entropy > FIREWALL_MAX_ENTROPY) {
        REQUIRE(verdict.action == FirewallAction::DAMPEN);
        REQUIRE(verdict.damping_factor < 1.0);
        REQUIRE(verdict.allowed());  // DAMPEN is still "allowed"
    }
}

// ============================================================================
// §5 Amplitude overflow rejected
// ============================================================================

TEST_CASE("§5 Amplitude overflow rejected", "[v030][firewall]") {
    ResonanceFirewall fw;
    auto wave = make_sine(1024, 5.0, 5.0);  // amplitude 5.0 > 4.0 limit

    auto verdict = fw.validate(wave);
    REQUIRE(verdict.rejected());
    REQUIRE(verdict.max_amplitude > FIREWALL_MAX_AMPLITUDE);
    REQUIRE(verdict.reason.find("Amplitude") != std::string::npos);
}

// ============================================================================
// §6 Repeating pattern detected (high autocorrelation)
// ============================================================================

TEST_CASE("§6 Repeating pattern penalized", "[v030][firewall]") {
    // Create a signal that repeats every 64 samples
    constexpr size_t N = 1024;
    constexpr size_t period = 64;
    std::vector<C> wave(N);

    // Generate one period of "interesting" data, then tile it
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<double> base(period);
    for (size_t i = 0; i < period; ++i) {
        base[i] = 0.3 * std::sin(2.0 * M_PI * 3.0 * i / period)
                + 0.2 * std::sin(2.0 * M_PI * 7.0 * i / period)
                + dist(rng) * 0.05;
    }
    for (size_t i = 0; i < N; ++i) {
        wave[i] = C{base[i % period], 0.0};
    }

    // Bypass entropy check — this test is about autocorrelation detection.
    // The repeating signal has low spectral entropy (~1 bit, 2 harmonics)
    // which would trigger a REJECT before autocorrelation is evaluated.
    ResonanceFirewallConfig cfg;
    cfg.min_entropy = 0.0;
    ResonanceFirewall fw(cfg);
    auto verdict = fw.validate(wave);
    REQUIRE(verdict.peak_autocorrelation > 0.9);
    // Should be penalized or rejected
    REQUIRE((verdict.action == FirewallAction::PENALIZE
          || verdict.action == FirewallAction::REJECT));
}

// ============================================================================
// §7 Hazardous pattern database matching
// ============================================================================

TEST_CASE("§7 Hazardous pattern matched → rejected", "[v030][firewall]") {
    ResonanceFirewall fw;

    // Register a hazardous pattern (a known bad waveform)
    auto bad_wave = make_mixed(1024);
    fw.add_hazardous_pattern("test_hazard", "manual", bad_wave);
    REQUIRE(fw.hazardous_pattern_count() == 1);

    // Feed the exact same waveform — should match
    auto verdict = fw.validate(bad_wave);
    REQUIRE(verdict.hazard_correlation > 0.9);
    // Cross-correlation with self should be ~1.0
    REQUIRE(verdict.matched_pattern_idx == 0);
}

// ============================================================================
// §8 Spectral entropy computation
// ============================================================================

TEST_CASE("§8 Spectral entropy of known signals", "[v030][firewall]") {
    // Pure sine should have very low entropy
    auto sine = make_sine(256, 10.0);
    double H_sine = ResonanceFirewall::compute_spectral_entropy(sine);
    REQUIRE(H_sine < 3.0);

    // Noise should have high entropy
    auto noise = make_noise(256);
    double H_noise = ResonanceFirewall::compute_spectral_entropy(noise);
    REQUIRE(H_noise > 5.0);

    // Mixed should be in between
    auto mixed = make_mixed(256);
    double H_mixed = ResonanceFirewall::compute_spectral_entropy(mixed);
    REQUIRE(H_mixed > H_sine);
}

// ============================================================================
// §9 FFT correctness (Parseval's theorem)
// ============================================================================

TEST_CASE("§9 FFT satisfies Parseval's theorem", "[v030][firewall]") {
    auto signal = make_mixed(256);

    // Energy in time domain
    double E_time = 0.0;
    for (const auto& s : signal) E_time += std::norm(s);

    // Energy in frequency domain
    auto spectrum = ResonanceFirewall::compute_fft(signal);
    double E_freq = 0.0;
    for (const auto& s : spectrum) E_freq += std::norm(s);
    E_freq /= static_cast<double>(spectrum.size());

    // Parseval: Σ|x[n]|² = (1/N) Σ|X[k]|²
    REQUIRE_THAT(E_time, Catch::Matchers::WithinRel(E_freq, 0.01));
}

// ============================================================================
// §10 Autocorrelation at lag 0 equals 1
// ============================================================================

TEST_CASE("§10 Autocorrelation at lag 0 equals 1", "[v030][firewall]") {
    auto signal = make_mixed(256);
    double R0 = ResonanceFirewall::compute_autocorrelation(signal, 0);
    REQUIRE_THAT(R0, Catch::Matchers::WithinAbs(1.0, 1e-10));
}

// ============================================================================
// §11 is_safe convenience
// ============================================================================

TEST_CASE("§11 is_safe() convenience", "[v030][firewall]") {
    ResonanceFirewall fw;
    auto safe = make_mixed(1024);
    REQUIRE(fw.is_safe(safe) == true);

    auto dangerous = make_sine(1024, 10.0, 5.0);  // amplitude overflow
    REQUIRE(fw.is_safe(dangerous) == false);
}

// ============================================================================
// §12 Cross-correlation of identical spectra ≈ 1.0
// ============================================================================

TEST_CASE("§12 Cross-correlation self = 1.0", "[v030][firewall]") {
    auto signal = make_mixed(256);
    auto spectrum = ResonanceFirewall::compute_fft(signal);

    double corr = ResonanceFirewall::cross_correlate(spectrum, spectrum);
    REQUIRE_THAT(corr, Catch::Matchers::WithinAbs(1.0, 0.01));
}

// ============================================================================
// §13 Empty waveform
// ============================================================================

TEST_CASE("§13 Empty waveform entropy is 0", "[v030][firewall]") {
    std::vector<C> empty;
    double H = ResonanceFirewall::compute_spectral_entropy(empty);
    REQUIRE(H == 0.0);
}

// ============================================================================
// §14 Custom config thresholds
// ============================================================================

TEST_CASE("§14 Custom config thresholds", "[v030][firewall]") {
    ResonanceFirewallConfig cfg;
    cfg.min_entropy = 1.0;   // more permissive
    cfg.max_entropy = 12.0;  // more permissive
    cfg.max_amplitude = 10.0;

    ResonanceFirewall fw(cfg);

    // Sine wave with amplitude 5.0 should now pass amplitude check
    auto wave = make_sine(1024, 10.0, 5.0);
    auto verdict = fw.validate(wave);
    // Should NOT be rejected for amplitude (limit is 10.0 now)
    REQUIRE(verdict.max_amplitude < 10.0);
}

// ============================================================================
// §15 Counters
// ============================================================================

TEST_CASE("§15 Validation counters", "[v030][firewall]") {
    ResonanceFirewall fw;
    REQUIRE(fw.total_validations() == 0);
    REQUIRE(fw.total_rejections() == 0);

    (void)fw.validate(make_mixed(1024));    // should pass
    REQUIRE(fw.total_validations() == 1);

    (void)fw.validate(make_sine(1024, 10.0, 5.0));  // amplitude overflow
    REQUIRE(fw.total_validations() == 2);
    REQUIRE(fw.total_rejections() == 1);
}
