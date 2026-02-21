/**
 * @file complex_field.hpp
 * @brief Complex number utilities for 9D toroidal wave physics.
 *
 * Provides type aliases, Kahan-compensated summation, thermal bath sampling,
 * and small helpers used throughout the UFIE physics engine.
 *
 * All operations are float-precision (FP32).  Double is intentionally avoided
 * in the hot path to ensure SIMD-friendly data layouts.
 *
 * Reference: nikola engineering report, Phase 1 (Gap 1.1, Gap 1.2)
 */
#pragma once

#include <complex>
#include <cmath>
#include <random>
#include <span>
#include <array>
#include <cassert>
#include <limits>

namespace nikola::foundation {

// ============================================================================
// Type aliases
// ============================================================================

/// Primary complex type used by the wave engine  (FP32 components).
using Complex = std::complex<float>;

/// Convenience: real part of complex.
inline float real_part(Complex c) noexcept { return c.real(); }

/// Convenience: imaginary part.
inline float imag_part(Complex c) noexcept { return c.imag(); }

/// |z|² — avoids the sqrt in std::abs.
inline float magnitude_sq(Complex c) noexcept {
    return c.real() * c.real() + c.imag() * c.imag();
}

/// |z| = sqrt(|z|²).
inline float magnitude(Complex c) noexcept {
    return std::sqrt(magnitude_sq(c));
}

// ============================================================================
// Kahan compensated summation  (numerical stability for 18-point Laplacian)
// ============================================================================

/**
 * @brief Kahan–Neumaier compensated sum of an array of Complex values.
 *
 * Reduces floating-point cancellation error from O(n·ε) to O(ε).  This is
 * mandatory when accumulating 18 neighbours in the 9D discrete Laplacian.
 *
 * @param values  Contiguous span of complex values to sum.
 * @return        Compensated sum.
 */
[[nodiscard]]
inline Complex kahan_sum(std::span<const Complex> values) noexcept {
    Complex sum{0.f, 0.f};
    Complex compensation{0.f, 0.f};   // running error term

    for (const Complex& v : values) {
        Complex y = v - compensation;
        Complex t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    return sum;
}

/**
 * @brief Kahan sum of a fixed-size array (template overload, zero overhead).
 */
template<std::size_t N>
[[nodiscard]]
inline Complex kahan_sum(const std::array<Complex, N>& values) noexcept {
    return kahan_sum(std::span<const Complex>{values.data(), N});
}

// ============================================================================
// Thermal bath sampling  (for velocity field initialization, Gap 1.2)
// ============================================================================

/**
 * @brief Sample a complex Gaussian random variable with standard deviation σ.
 *
 * Used to initialise the velocity field ∂Ψ/∂t to quantum vacuum fluctuations:
 *   σ_T = 1e-6 · √(Tr(g(x)) / 9)   (from Phase 1 spec, Gap 1.2)
 *
 * @param sigma  Standard deviation (positive).
 * @param rng    Seeded Mersenne-Twister generator (state mutated).
 */
[[nodiscard]]
inline Complex sample_thermal(float sigma, std::mt19937& rng) noexcept {
    std::normal_distribution<float> dist(0.f, sigma);
    return {dist(rng), dist(rng)};
}

/**
 * @brief Thermal noise floor from local metric trace.
 *
 *   σ_T = 1e-6 · √(trace / 9)
 *
 * @param metric_trace  Tr(g) at the node.  Should be ≈ 9 for near-identity.
 */
[[nodiscard]]
inline float thermal_sigma(float metric_trace) noexcept {
    assert(metric_trace > 0.f && "Metric trace must be positive (SPD)");
    return 1e-6f * std::sqrt(metric_trace / 9.f);
}

// ============================================================================
// Miscellaneous utilities
// ============================================================================

/**
 * @brief Clamp amplitude to a maximum safe value.
 *
 * The injection amplitude must satisfy |Ψ| ≤ √(MAX_E / β) to prevent
 * the cubic nonlinearity from causing numerical explosion.
 *
 * @param z          Complex amplitude to be clamped.
 * @param max_amp    Maximum allowed |z|.
 * @return           z scaled down (if necessary) to |z| ≤ max_amp, else z.
 */
[[nodiscard]]
inline Complex clamp_amplitude(Complex z, float max_amp) noexcept {
    const float mag = magnitude(z);
    if (mag > max_amp && mag > 0.f) {
        return z * (max_amp / mag);
    }
    return z;
}

/**
 * @brief Apply a Perfectly Matched Layer (PML) ghost cell extrapolation.
 *
 * When a neighbour lookup returns vacuum (node not allocated), the wave
 * must continue propagating outward rather than reflecting.
 *
 *   Ψ_ghost = Ψ_self · e^(-ik·Δx) · α_absorb
 *
 * For a simple CPU implementation we approximate exp(-ik·Δx) ≈ 1,
 * keeping only the absorption factor.
 *
 * @param psi_self   Wavefunction at center node.
 * @param alpha_abs  Absorption coefficient (default 0.9 from spec).
 * @return           Ghost cell value.
 */
[[nodiscard]]
inline Complex pml_ghost(Complex psi_self, float alpha_abs = 0.9f) noexcept {
    return psi_self * alpha_abs;
}

/**
 * @brief Check if complex value is numerically finite.
 */
[[nodiscard]]
inline bool is_finite(Complex c) noexcept {
    return std::isfinite(c.real()) && std::isfinite(c.imag());
}

/**
 * @brief Compute the pilot wave at position x along a single dimension.
 *
 *   Ψ_pilot(x) = A₀ · e^(i·k·x)
 *
 * @param x          Coordinate in [0, N).
 * @param N          Grid size in this dimension.
 * @param k_mode     Wavenumber mode (integer for periodic BC).
 * @param amplitude  A₀ (default 1.0 from spec, activates nonlinearity).
 */
[[nodiscard]]
inline Complex pilot_wave(int x, int N, int k_mode = 1, float amplitude = 1.f) noexcept {
    const float phase = 2.f * static_cast<float>(M_PI) * k_mode * x / N;
    return {amplitude * std::cos(phase), amplitude * std::sin(phase)};
}

} // namespace nikola::foundation
