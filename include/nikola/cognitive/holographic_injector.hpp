#pragma once

// HolographicInjector — Nit[128] → wave chords → torus emitter injection
//
// Chunks the 128-Nit embedding into 9-Nit groups (14 full chords + 2 remainder).
// Each group creates a frequency-multiplexed chord across 8 emitters, then
// injects the accumulated interference pattern into the TorusGrid.
//
// Emitter frequencies: f_n = π·φⁿ (golden-ratio-spaced, avoids resonance lock-in)
// Phase encoding:      each Nit amplitude modulates the n-th emitter sinusoid
//
// Spec: docs/info/engineering/03_cognitive_systems.txt §3.4.1 "Holographic Multiplexing"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <nikola/cognitive/nonary_embedder.hpp>
#include <nikola/foundation/nit.hpp>

namespace nikola::cognitive {

using nikola::foundation::Nit;

// Forward declaration — TorusGrid interface used for injection
// Defined in nikola/physics/torus_grid.hpp
template<typename Grid>
class HolographicInjector {
public:
    static constexpr size_t NUM_EMITTERS  = 8;
    static constexpr size_t CHORD_SIZE    = 9;    // Nits per chord
    static constexpr double PHI           = 1.6180339887498948482;
    static constexpr double PI            = 3.1415926535897932385;

    // Emitter frequencies: f_n = π·φⁿ for n in [0,7]
    // These are spectrally orthogonal — no integer ratios → no resonance lock-in
    static std::array<double, NUM_EMITTERS> emitter_frequencies() {
        std::array<double, NUM_EMITTERS> freqs;
        double phi_n = 1.0;
        for (size_t n = 0; n < NUM_EMITTERS; ++n) {
            freqs[n] = PI * phi_n;
            phi_n *= PHI;
        }
        return freqs;
    }

    explicit HolographicInjector(Grid& grid) : grid_(grid) {
        freqs_ = emitter_frequencies();
        std::cout << "[HolographicInjector] Emitter freqs: ";
        for (auto f : freqs_) std::cout << f << " ";
        std::cout << "\n";
    }

    // Inject a 128-Nit embedding as holographic wave energy into the torus
    // time: current physics time (used for phase calculation)
    void inject(const std::vector<Nit>& nit_vec, double time = 0.0) {
        if (nit_vec.empty()) return;

        // Pad to multiple of CHORD_SIZE
        std::vector<Nit> padded = nit_vec;
        while (padded.size() % CHORD_SIZE != 0) padded.push_back(0);

        size_t num_chords = padded.size() / CHORD_SIZE;
        double chord_time_step = (2.0 * PI) / static_cast<double>(num_chords + 1);

        for (size_t c = 0; c < num_chords; ++c) {
            // Extract 9-Nit chord
            std::array<Nit, CHORD_SIZE> chord;
            for (size_t i = 0; i < CHORD_SIZE; ++i) {
                chord[i] = padded[c * CHORD_SIZE + i];
            }

            double t = time + static_cast<double>(c) * chord_time_step;
            auto wave = compute_chord(chord, t);

            // Inject chord wave energy into grid at emitter positions
            inject_chord_to_grid(wave, c, num_chords);
        }

        if constexpr (requires { grid_.mark_injected(); }) {
            grid_.mark_injected();
        }
    }

    // Inject text directly (convenience wrapper with embedded NonaryEmbedder)
    void inject_text(const NonaryEmbedder& embedder,
                     const std::string& text,
                     double time = 0.0) {
        auto nit_vec = embedder.embed(text);
        inject(nit_vec, time);
        std::cout << "[HolographicInjector] Injected \"" << text.substr(0, 40)
                  << (text.size() > 40 ? "..." : "")
                  << "\" → " << nit_vec.size() << " Nits → "
                  << (nit_vec.size() / CHORD_SIZE) << " chords\n";
    }

    /**
     * @brief Inject a Nit vector with each amplitude scaled by @p weight.
     *
     * Used by the gated injection path to attenuate marginal signals without
     * discarding them entirely.  A weight of 1.0 is identical to inject().
     * A weight of 0.0 is a no-op (nothing enters the torus).
     *
     * @param nit_vec  128-Nit embedding vector.
     * @param weight   Salience weight in [0, 1].  Nit amplitudes are multiplied
     *                 by this value before chord computation.
     * @param time     Physics time for phase.
     */
    void inject_scaled(const std::vector<Nit>& nit_vec,
                       float weight,
                       double time = 0.0) {
        if (nit_vec.empty() || weight <= 0.f) return;
        if (weight >= 1.f) { inject(nit_vec, time); return; }

        // Scale each Nit amplitude by weight (Nit is an int8 alias; round to nearest)
        std::vector<Nit> scaled;
        scaled.reserve(nit_vec.size());
        for (Nit n : nit_vec) {
            int val = static_cast<int>(std::round(static_cast<float>(n) * weight));
            scaled.push_back(static_cast<Nit>(std::clamp(val, -4, 4)));
        }
        inject(scaled, time);
    }

    // Return the last computed chord energy for diagnostics
    double last_chord_energy() const { return last_energy_; }

    // -----------------------------------------------------------------------
    // Analytic injection signature  (Phase 28 — time-correct warm decode)
    // -----------------------------------------------------------------------

    /**
     * @brief Compute the chord-amplitude signature for a Nit vector at time t.
     *
     * Returns a vector of `num_chords` complex values: element c is the
     * scaled chord amplitude that inject() would write to the grid at chord
     * position c.  The computation is PURELY ARITHMETIC — no GridState is
     * read or modified, and the result is identical to what the actual inject
     * call would have produced.
     *
     * This is the mathematical heart of Phase 28 emitter-phase-aware decode:
     * because the emitter frequencies are π·φⁿ (incommensurate, Weyl-
     * equidistributed), the same Nit vector produces completely different
     * grid amplitudes at different times.  Rather than storing a snapshot at
     * calibration time, we evaluate the closed-form expression at the actual
     * injection time, making the comparison mathematically exact regardless
     * of how long the loop has been running.
     *
     * @param nit_vec  128-Nit embedding (will be padded internally).
     * @param time     Physics time at the moment of injection (torus_.time()).
     * @return         Vector of complex chord amplitudes, length == num_chords.
     */
    static std::vector<std::complex<double>> analytic_signature(
            const std::vector<Nit>& nit_vec, double time)
    {
        const auto freqs = emitter_frequencies();

        // Mirror padding logic from inject()
        std::vector<Nit> padded = nit_vec;
        while (padded.size() % CHORD_SIZE != 0) padded.push_back(0);
        const size_t num_chords = padded.size() / CHORD_SIZE;
        const double chord_time_step = (2.0 * PI) / static_cast<double>(num_chords + 1);

        // Scaling: mirror inject_chord_to_grid()
        static constexpr double NIT_MAX_         = 4.0;
        static constexpr double INJECTION_SCALE_ = 0.05;
        const double scale = INJECTION_SCALE_ / (static_cast<double>(NUM_EMITTERS) * NIT_MAX_);

        std::vector<std::complex<double>> sig(num_chords);
        for (size_t c = 0; c < num_chords; ++c) {
            std::array<Nit, CHORD_SIZE> chord;
            for (size_t i = 0; i < CHORD_SIZE; ++i) chord[i] = padded[c * CHORD_SIZE + i];

            double t = time + static_cast<double>(c) * chord_time_step;
            std::complex<double> sum = 0.0;
            for (size_t n = 0; n < NUM_EMITTERS; ++n) {
                double amplitude = static_cast<double>(chord[n % CHORD_SIZE]);
                double freq      = freqs[n];
                sum += amplitude * std::exp(std::complex<double>(0.0, freq * t));
            }
            // Apply same safety scale as inject_chord_to_grid
            const double mag = std::abs(sum);
            if (mag > 0.0) sum *= scale;
            sig[c] = sum;
        }
        return sig;
    }

    /**
     * @brief Cosine similarity between two injection signatures.
     *
     * Uses the complex inner product |<a,b>| / (||a|| · ||b||), so phases
     * cancel and only amplitude-pattern similarity remains.  Returns 0 when
     * either vector is zero-norm, and 1.0 for identical signatures.
     *
     * @param a  First signature (from analytic_signature()).
     * @param b  Second signature.
     * @return   Similarity in [0, 1].
     */
    static float signature_cosine(const std::vector<std::complex<double>>& a,
                                   const std::vector<std::complex<double>>& b)
    {
        const size_t len = std::min(a.size(), b.size());
        std::complex<double> dot = 0.0;
        double norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < len; ++i) {
            dot    += std::conj(a[i]) * b[i];
            norm_a += std::norm(a[i]);
            norm_b += std::norm(b[i]);
        }
        if (norm_a < 1e-30 || norm_b < 1e-30) return 0.f;
        return static_cast<float>(std::abs(dot) / std::sqrt(norm_a * norm_b));
    }

private:
    Grid& grid_;
    std::array<double, NUM_EMITTERS> freqs_;
    mutable double last_energy_ = 0.0;

    // Compute superposition amplitude for one 9-Nit chord at time t
    // Uses 8 emitters, cycling through chord Nits as amplitudes
    std::complex<double> compute_chord(const std::array<Nit, CHORD_SIZE>& chord,
                                       double t) const {
        std::complex<double> sum = 0.0;
        for (size_t n = 0; n < NUM_EMITTERS; ++n) {
            // Map emitter → chord index (wrap around 9 Nits)
            double amplitude = static_cast<double>(chord[n % CHORD_SIZE]);
            double freq      = freqs_[n];
            // ψ_n(t) = A_n · e^{i·f_n·t}
            sum += amplitude * std::exp(std::complex<double>(0.0, freq * t));
        }
        last_energy_ = std::abs(sum);
        return sum;
    }

    // Distribute chord energy to grid nodes at evenly-spaced torus positions
    void inject_chord_to_grid(std::complex<double> wave,
                              size_t chord_idx,
                              size_t total_chords) {
        // Map chord index to a torus position (0..N-1 evenly distributed)
        size_t grid_n = grid_.grid_n();
        size_t stride = (grid_n > total_chords) ? (grid_n / total_chords) : 1;
        size_t base   = (chord_idx * stride) % (grid_n * grid_n * grid_n);

        // Normalise to safe injection amplitude.
        // Raw chord magnitude: up to NUM_EMITTERS × MAX_NIT = 8 × 4 = 32.
        // Scale → max |perturbation| ≈ INJECTION_SCALE (≈ 0.05), safely below
        // the nonlinear blow-up threshold sqrt(0.1 / beta) ≈ 0.316.
        static constexpr double NIT_MAX        = 4.0;
        static constexpr double INJECTION_SCALE= 0.05;
        double mag = std::abs(wave);
        if (mag > 0.0) {
            double scale = INJECTION_SCALE / (static_cast<double>(NUM_EMITTERS) * NIT_MAX);
            wave *= scale;
        }

        // Apply wave energy as perturbation to wavefunction at target node
        grid_.perturb_wavefunction(base, wave.real(), wave.imag());
    }
};

} // namespace nikola::cognitive
