/**
 * @file audio_emitter.hpp
 * @brief Gap 6.1 — AudioEmitterLayout
 *
 * 8 emitters in a circular array on the z=0 plane of the 9D torus.
 * Spatial placement uses golden-ratio frequencies for harmonic separation.
 *
 * Dimension indices (per ANISOTROPIC_RESOLUTION = {64,64,64,128,16,16,32,32,32}):
 *   c[0]=x(64)  c[1]=y(64)  c[2]=z(64)  c[3]=t(128)
 *   c[4]=r(16)  c[5]=s(16)  c[6]=u(32)  c[7]=v(32)  c[8]=w(32)
 */
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "nikola/spatial/topology_manager.hpp"
#include "nikola/multimodal/cymatic_transduction.hpp"

namespace nikola::multimodal {

// Bring Coord9DInt into this namespace
using nikola::spatial::Coord9DInt;

// ============================================================================
// Constants
// ============================================================================

inline constexpr int    NUM_EMITTERS          = 8;
// GOLDEN_RATIO defined in cymatic_transduction.hpp
inline constexpr double EMITTER_RADIAL_FRAC   = 0.8;  // r = 0.8 * Nr
inline constexpr double AUDIO_RING_FRAC       = 0.5;  // R = Nx/2

// ============================================================================
// Gap 6.1 — AudioEmitterLayout
// ============================================================================

/**
 * Emitter position as 9D integer coordinate + associated frequency.
 */
struct EmitterPosition {
    Coord9DInt coord;
    double     frequency_hz{0.0}; ///< f_n = π · φⁿ
    int        emitter_id{0};
};

/**
 * Computes the spatial layout for NUM_EMITTERS audio injection points.
 */
class AudioEmitterLayout {
public:
    /**
     * Compute the torus position for emitter n.
     *
     * θ_n = 2π·n/NUM_EMITTERS
     * x_n = cx + R·cos(θ_n)      (cx = Nx/2, R = Nx/2)
     * y_n = cy + R·sin(θ_n)      (cy = Ny/2)
     * z_n = 0
     * r_n = clamp(round(0.8·Nr), 0, Nr-1)
     * s_n = Nr-1  (Ns clamped to max index)
     * u=v=w = 0
     * t   = time_index % Nt
     *
     * @param n          Emitter index [0, NUM_EMITTERS)
     * @param grid_nx    Grid size in x dimension (typically 64)
     * @param grid_ny    Grid size in y dimension (typically 64)
     * @param grid_nr    Grid size in r dimension (typically 16)
     * @param grid_ns    Grid size in s dimension (typically 16)
     * @param grid_nt    Grid size in t dimension (typically 128)
     * @param time_index Current time step
     * @return EmitterPosition with full 9D coordinate and frequency
     */
    static EmitterPosition compute_position(int n,
                                             int grid_nx   = 64,
                                             int grid_ny   = 64,
                                             int grid_nr   = 16,
                                             int grid_ns   = 16,
                                             int grid_nt   = 128,
                                             int time_index = 0)
    {
        if (n < 0 || n >= NUM_EMITTERS) {
            throw std::out_of_range("AudioEmitterLayout: emitter index out of range");
        }

        const double theta  = 2.0 * std::numbers::pi * n / NUM_EMITTERS;
        const double R      = static_cast<double>(grid_nx) * AUDIO_RING_FRAC;
        const double cx     = static_cast<double>(grid_nx) / 2.0;
        const double cy     = static_cast<double>(grid_ny) / 2.0;

        const double xd     = cx + R * std::cos(theta);
        const double yd     = cy + R * std::sin(theta);

        // Clamp to valid grid range
        auto clamp_to = [](double v, int max_exclusive) -> uint16_t {
            int iv = static_cast<int>(std::round(v));
            if (iv < 0)              iv = 0;
            if (iv >= max_exclusive) iv = max_exclusive - 1;
            return static_cast<uint16_t>(iv);
        };

        const uint16_t x_coord = clamp_to(xd, grid_nx);
        const uint16_t y_coord = clamp_to(yd, grid_ny);
        const uint16_t z_coord = 0; // z=0 plane
        const uint16_t t_coord = static_cast<uint16_t>(time_index % grid_nt);
        const uint16_t r_coord = clamp_to(EMITTER_RADIAL_FRAC * grid_nr, grid_nr);
        const uint16_t s_coord = static_cast<uint16_t>(grid_ns - 1); // Ns max index
        const uint16_t u_coord = 0;
        const uint16_t v_coord = 0;
        const uint16_t w_coord = 0;

        Coord9DInt coord{};
        coord.c[0] = x_coord;
        coord.c[1] = y_coord;
        coord.c[2] = z_coord;
        coord.c[3] = t_coord;
        coord.c[4] = r_coord;
        coord.c[5] = s_coord;
        coord.c[6] = u_coord;
        coord.c[7] = v_coord;
        coord.c[8] = w_coord;

        const double freq = std::numbers::pi * std::pow(GOLDEN_RATIO, static_cast<double>(n));

        return EmitterPosition{coord, freq, n};
    }

    /**
     * Generate all NUM_EMITTERS positions for the current time step.
     */
    static std::array<EmitterPosition, NUM_EMITTERS>
    all_positions(int grid_nx   = 64,
                  int grid_ny   = 64,
                  int grid_nr   = 16,
                  int grid_ns   = 16,
                  int grid_nt   = 128,
                  int time_index = 0)
    {
        std::array<EmitterPosition, NUM_EMITTERS> out{};
        for (int i = 0; i < NUM_EMITTERS; ++i) {
            out[i] = compute_position(i, grid_nx, grid_ny, grid_nr, grid_ns, grid_nt, time_index);
        }
        return out;
    }

    /**
     * Minimum Euclidean (2D XY) distance between any two emitters.
     * Spec requires > 10 grid cells for valid layout.
     */
    static double min_emitter_separation(int grid_nx = 64, int grid_ny = 64)
    {
        auto positions = all_positions(grid_nx, grid_ny);
        double min_dist = std::numeric_limits<double>::max();
        for (int i = 0; i < NUM_EMITTERS; ++i) {
            for (int j = i + 1; j < NUM_EMITTERS; ++j) {
                double dx = static_cast<double>(positions[i].coord.c[0])
                          - static_cast<double>(positions[j].coord.c[0]);
                double dy = static_cast<double>(positions[i].coord.c[1])
                          - static_cast<double>(positions[j].coord.c[1]);
                double d = std::hypot(dx, dy);
                if (d < min_dist) min_dist = d;
            }
        }
        return min_dist;
    }

    /**
     * Emitter frequency f_n = π·φⁿ
     */
    static double emitter_frequency(int n) {
        return std::numbers::pi * std::pow(GOLDEN_RATIO, static_cast<double>(n));
    }
};

} // namespace nikola::multimodal
