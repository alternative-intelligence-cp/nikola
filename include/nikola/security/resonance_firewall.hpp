/**
 * @file security/resonance_firewall.hpp
 * @brief v0.3.0 — Resonance Firewall (Ingress Protection)
 *
 * The digital immune system of the Nikola Model.  Sits at the perimeter
 * of the Ingestion Pipeline and analyzes every incoming waveform before
 * it interacts with the Torus Manifold.
 *
 * Filters based on three spectral properties:
 *
 *   1. **Spectral Entropy** (H_spec):
 *      - H < 2.0  → too ordered ("Siren Attack") → REJECT
 *      - H > 8.0  → pure noise  ("Thermal Attack") → DAMPEN or REJECT
 *      - 2.0 ≤ H ≤ 8.0 → "Edge of Chaos" → PASS
 *
 *   2. **Temporal Autocorrelation** (R_xx):
 *      - R_xx(τ>0) > 0.95 → repeating loop → apply boredom penalty
 *
 *   3. **Hazardous Pattern Database**:
 *      - Cross-correlation against known-bad waveform signatures
 *      - Signatures from crash logs + Adversarial Dojo GA
 *
 * Spec: §3.1 "The Resonance Firewall (Ingress Protection)"
 *        docs/info/integration/sections/04_infrastructure/05_security_subsystem.md
 *
 * FFT: Uses self-contained radix-2 DFT by default.
 *       FFTW3 can be enabled via NIKOLA_ENABLE_FFTW3 for performance.
 */
#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace nikola::security {

// ============================================================================
// Constants — spec §3.1.1
// ============================================================================

/// Minimum spectral entropy — below this is "Siren Attack" (too ordered).
inline constexpr double FIREWALL_MIN_ENTROPY = 2.0;

/// Maximum spectral entropy — above this is "Thermal Attack" (pure noise).
inline constexpr double FIREWALL_MAX_ENTROPY = 8.0;

/// Autocorrelation threshold — above this is dangerous periodicity.
inline constexpr double FIREWALL_AUTOCORR_THRESHOLD = 0.95;

/// Maximum safe amplitude — Balanced Nonary limit.
inline constexpr double FIREWALL_MAX_AMPLITUDE = 4.0;

/// Cross-correlation threshold for hazardous pattern matching.
inline constexpr double FIREWALL_PATTERN_THRESHOLD = 0.90;

/// Default FFT window size.
inline constexpr size_t FIREWALL_DEFAULT_WINDOW = 1024;

// ============================================================================
// FirewallVerdict
// ============================================================================

enum class FirewallAction : uint8_t {
    PASS,         ///< Waveform is safe — allow through
    DAMPEN,       ///< Apply damping factor before allowing
    REJECT,       ///< Block entirely
    PENALIZE,     ///< Apply boredom/attention penalty
};

inline const char* firewall_action_str(FirewallAction a) {
    switch (a) {
        case FirewallAction::PASS:     return "PASS";
        case FirewallAction::DAMPEN:   return "DAMPEN";
        case FirewallAction::REJECT:   return "REJECT";
        case FirewallAction::PENALIZE: return "PENALIZE";
    }
    return "UNKNOWN";
}

// ============================================================================
// FirewallVerdict — detailed result of waveform analysis
// ============================================================================

struct FirewallVerdict {
    FirewallAction  action{FirewallAction::PASS};
    std::string     reason;

    /// Spectral entropy of the input waveform.
    double          spectral_entropy{0.0};

    /// Peak autocorrelation at lag > 0.
    double          peak_autocorrelation{0.0};

    /// Lag at which peak autocorrelation occurs.
    size_t          peak_autocorr_lag{0};

    /// Maximum amplitude in the input.
    double          max_amplitude{0.0};

    /// Cross-correlation score against hazardous pattern DB (-1 if not checked).
    double          hazard_correlation{-1.0};

    /// Index of matched hazardous pattern (-1 if none matched).
    int             matched_pattern_idx{-1};

    /// Damping factor to apply (1.0 = no damping, 0.1 = 90% damping).
    double          damping_factor{1.0};

    /// True if the waveform should be allowed through (PASS or DAMPEN).
    [[nodiscard]] bool allowed() const noexcept {
        return action == FirewallAction::PASS
            || action == FirewallAction::DAMPEN
            || action == FirewallAction::PENALIZE;
    }

    /// True if the waveform was rejected.
    [[nodiscard]] bool rejected() const noexcept {
        return action == FirewallAction::REJECT;
    }
};

// ============================================================================
// ResonanceFirewallConfig
// ============================================================================

struct ResonanceFirewallConfig {
    /// FFT window size (must be power of 2 for radix-2 FFT).
    size_t window_size = FIREWALL_DEFAULT_WINDOW;

    /// Entropy thresholds.
    double min_entropy = FIREWALL_MIN_ENTROPY;
    double max_entropy = FIREWALL_MAX_ENTROPY;

    /// Autocorrelation threshold.
    double autocorr_threshold = FIREWALL_AUTOCORR_THRESHOLD;

    /// Maximum safe amplitude.
    double max_amplitude = FIREWALL_MAX_AMPLITUDE;

    /// Cross-correlation threshold for pattern matching.
    double pattern_threshold = FIREWALL_PATTERN_THRESHOLD;

    /// If true, high-entropy signals are dampened (90%) instead of rejected.
    bool dampen_high_entropy = true;

    /// High-entropy damping factor (applied when H > max_entropy).
    double high_entropy_damping = 0.1;

    /// Maximum autocorrelation lags to check (0 = window_size/4).
    size_t max_autocorr_lags = 0;
};

// ============================================================================
// HazardousPattern — a known-bad waveform signature
// ============================================================================

struct HazardousPattern {
    std::string                          name;
    std::string                          source;  ///< "crash_log", "dojo", "manual"
    std::vector<std::complex<double>>    spectrum; ///< FFT of the hazardous signal
};

// ============================================================================
// ResonanceFirewall
// ============================================================================

/**
 * @class ResonanceFirewall
 * @brief Ingress waveform filter — spectral entropy + autocorrelation + pattern DB.
 *
 * Usage:
 *   ResonanceFirewall fw;
 *   auto verdict = fw.validate(waveform);
 *   if (verdict.rejected()) { block_input(); }
 *   else if (verdict.action == FirewallAction::DAMPEN) {
 *       apply_damping(waveform, verdict.damping_factor);
 *   }
 *
 * Thread safety: validate() is const and thread-safe.
 * add_hazardous_pattern() is NOT thread-safe — call during setup only.
 */
class ResonanceFirewall {
public:
    ResonanceFirewall();
    explicit ResonanceFirewall(ResonanceFirewallConfig config);

    /**
     * Validate an incoming waveform against all firewall rules.
     *
     * @param waveform  Complex-valued input signal.
     * @return FirewallVerdict with action, metrics, and reason.
     */
    [[nodiscard]] FirewallVerdict validate(
        const std::vector<std::complex<double>>& waveform) const;

    /**
     * Quick boolean check — true if waveform would be allowed.
     */
    [[nodiscard]] bool is_safe(
        const std::vector<std::complex<double>>& waveform) const;

    /**
     * Add a hazardous waveform pattern to the database.
     * The pattern's FFT is computed and stored for cross-correlation matching.
     *
     * @param name    Human-readable name for the pattern.
     * @param source  Origin: "crash_log", "dojo", "manual".
     * @param signal  Time-domain waveform of the hazardous pattern.
     */
    void add_hazardous_pattern(const std::string& name,
                               const std::string& source,
                               const std::vector<std::complex<double>>& signal);

    /// Number of patterns in the hazardous pattern database.
    [[nodiscard]] size_t hazardous_pattern_count() const noexcept {
        return patterns_.size();
    }

    /// Total waveforms validated.
    [[nodiscard]] uint64_t total_validations() const noexcept {
        return total_validations_;
    }

    /// Total waveforms rejected.
    [[nodiscard]] uint64_t total_rejections() const noexcept {
        return total_rejections_;
    }

    /// Access config.
    [[nodiscard]] const ResonanceFirewallConfig& config() const noexcept {
        return cfg_;
    }

    // ── Static utility functions (public for testing) ────────────────────────

    /**
     * Compute spectral entropy of a signal.
     * H = -Σ p_k log2(p_k) where p_k = P[k] / Σ P[j].
     */
    [[nodiscard]] static double compute_spectral_entropy(
        const std::vector<std::complex<double>>& signal);

    /**
     * Compute normalized temporal autocorrelation at a given lag.
     * R_xx(τ) = Σ x[n]·x̄[n+τ] / (||x||²)
     */
    [[nodiscard]] static double compute_autocorrelation(
        const std::vector<std::complex<double>>& signal, size_t lag);

    /**
     * Compute the FFT of a signal (radix-2 Cooley-Tukey).
     * Input is zero-padded to the next power of 2 if necessary.
     */
    [[nodiscard]] static std::vector<std::complex<double>> compute_fft(
        const std::vector<std::complex<double>>& signal);

    /**
     * Compute frequency-domain cross-correlation between two spectra.
     * Returns max |IFFT(F* · G)| / sqrt(||F||² · ||G||²).
     */
    [[nodiscard]] static double cross_correlate(
        const std::vector<std::complex<double>>& spectrum_a,
        const std::vector<std::complex<double>>& spectrum_b);

private:
    ResonanceFirewallConfig              cfg_;
    std::vector<HazardousPattern>        patterns_;
    mutable uint64_t                     total_validations_{0};
    mutable uint64_t                     total_rejections_{0};

    /// Compute inverse FFT.
    [[nodiscard]] static std::vector<std::complex<double>> compute_ifft(
        const std::vector<std::complex<double>>& spectrum);

    /// Next power of 2 >= n.
    [[nodiscard]] static size_t next_pow2(size_t n) noexcept;

    /// Bit-reversal permutation for FFT.
    static void bit_reverse_permute(std::vector<std::complex<double>>& v);
};

}  // namespace nikola::security
