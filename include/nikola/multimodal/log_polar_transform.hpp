/**
 * @file log_polar_transform.hpp
 * @brief Gap 6.2 — LogPolarTransform (OpenCV-free visual sensory transduction)
 *
 * Transforms a grayscale image into 64×64 log-polar coordinates,
 * then generates 9D injection coordinates for each activated pixel.
 *
 * No OpenCV dependency — accepts a flat float span.
 */
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

#include "nikola/spatial/topology_manager.hpp"

namespace nikola::multimodal {

// Bring Coord9DInt into this namespace
using nikola::spatial::Coord9DInt;

// ============================================================================
// Constants
// ============================================================================

inline constexpr int   LP_ANGULAR_BINS    = 64;
inline constexpr int   LP_RADIAL_BINS     = 64;
inline constexpr float LP_INJECT_THRESHOLD = 0.01f;

// Injection layer: visual sits above audio (z=1), mid-range neurochemical state
inline constexpr uint16_t LP_INJECT_Z = 1;
inline constexpr uint16_t LP_INJECT_R = 8; // mid-range resonance
inline constexpr uint16_t LP_INJECT_S = 8; // mid-range state

// ============================================================================
// Gap 6.2 — LogPolarTransform
// ============================================================================

/**
 * OpenCV-free log-polar image transform.
 *
 *   log_r   = (rad_idx / RADIAL_BINS) * ln(max_radius)
 *   radius  = exp(log_r)
 *   angle   = (ang_idx / ANGULAR_BINS) * 2π
 *   src_x   = cx + radius * cos(angle)
 *   src_y   = cy + radius * sin(angle)
 *
 * Injection coordinate layout (9D):
 *   c[0]=x → radial bin idx   (0…63)
 *   c[1]=y → angular bin idx  (0…63)
 *   c[2]=z → LP_INJECT_Z (1)
 *   c[3]=t → time_index % 128
 *   c[4]=r → LP_INJECT_R (8)
 *   c[5]=s → LP_INJECT_S (8)
 *   c[6]=u, c[7]=v, c[8]=w → 0
 */
class LogPolarTransform {
public:
    using LPOutput = std::array<float, LP_RADIAL_BINS * LP_ANGULAR_BINS>;
    using InjectionList = std::vector<std::pair<Coord9DInt, float>>;

    /**
     * Compute the log-polar transform of a grayscale image.
     *
     * @param image  Flattened grayscale float image [0,1], row-major (width × height)
     * @param width  Image width in pixels
     * @param height Image height in pixels
     * @return 64×64 log-polar output, indexed [radial * LP_ANGULAR_BINS + angular]
     */
    static LPOutput transform(std::span<const float> image, int width, int height)
    {
        LPOutput out{};
        out.fill(0.0f);

        if (image.size() != static_cast<size_t>(width * height)) return out;
        if (width <= 0 || height <= 0) return out;

        const double cx = static_cast<double>(width)  / 2.0;
        const double cy = static_cast<double>(height) / 2.0;
        const double max_radius = std::hypot(cx, cy);
        if (max_radius < 1.0) return out;

        const double log_max = std::log(max_radius);

        for (int r_idx = 0; r_idx < LP_RADIAL_BINS; ++r_idx) {
            const double log_r  = (static_cast<double>(r_idx) / LP_RADIAL_BINS) * log_max;
            const double radius = std::exp(log_r);

            for (int a_idx = 0; a_idx < LP_ANGULAR_BINS; ++a_idx) {
                const double angle = (static_cast<double>(a_idx) / LP_ANGULAR_BINS)
                                   * 2.0 * std::numbers::pi;

                const double sx = cx + radius * std::cos(angle);
                const double sy = cy + radius * std::sin(angle);

                // Bilinear sample (or nearest-neighbour when on boundary)
                const int ix = static_cast<int>(sx);
                const int iy = static_cast<int>(sy);

                if (ix < 0 || ix >= width || iy < 0 || iy >= height) continue;

                // Bilinear weights
                const double fx = sx - ix;
                const double fy = sy - iy;

                auto sample = [&](int px, int py) -> float {
                    if (px < 0 || px >= width || py < 0 || py >= height) return 0.0f;
                    return image[static_cast<size_t>(py * width + px)];
                };

                const float v = static_cast<float>(
                    (1.0 - fx) * (1.0 - fy) * sample(ix,   iy)
                  +        fx  * (1.0 - fy) * sample(ix+1, iy)
                  + (1.0 - fx) *        fy  * sample(ix,   iy+1)
                  +        fx  *        fy  * sample(ix+1, iy+1)
                );

                out[static_cast<size_t>(r_idx * LP_ANGULAR_BINS + a_idx)] = v;
            }
        }
        return out;
    }

    /**
     * Convert log-polar output to a list of 9D injection coordinates and intensities.
     * Only pixels above LP_INJECT_THRESHOLD are included.
     *
     * @param lp_output   Output from transform()
     * @param time_index  Current time step (used for t dimension)
     * @param grid_nt     t dimension size (default 128)
     * @return List of (Coord9DInt, intensity) pairs
     */
    static InjectionList inject_coords(const LPOutput& lp_output,
                                        int time_index = 0,
                                        int grid_nt    = 128)
    {
        InjectionList result;
        result.reserve(128);

        const uint16_t t_coord = static_cast<uint16_t>(time_index % grid_nt);

        for (int r_idx = 0; r_idx < LP_RADIAL_BINS; ++r_idx) {
            for (int a_idx = 0; a_idx < LP_ANGULAR_BINS; ++a_idx) {
                const float v = lp_output[static_cast<size_t>(r_idx * LP_ANGULAR_BINS + a_idx)];
                if (v < LP_INJECT_THRESHOLD) continue;

                Coord9DInt coord{};
                coord.c[0] = static_cast<uint16_t>(r_idx);  // x → radial bin
                coord.c[1] = static_cast<uint16_t>(a_idx);  // y → angular bin
                coord.c[2] = LP_INJECT_Z;                   // z=1 visual layer
                coord.c[3] = t_coord;
                coord.c[4] = LP_INJECT_R;
                coord.c[5] = LP_INJECT_S;
                coord.c[6] = 0;
                coord.c[7] = 0;
                coord.c[8] = 0;

                result.emplace_back(coord, v);
            }
        }
        return result;
    }
};

} // namespace nikola::multimodal
