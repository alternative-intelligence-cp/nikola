/**
 * @file include/nikola/spatial/morton_encoder.hpp
 * @brief 9D Morton (Z-order) encoding for T⁹ toroidal manifold.
 *
 * Morton keys interleave the bits of 9 coordinates into a single integer,
 * producing a Z-order curve that maps 9D grid points to 1D keys.
 *
 * With 14 bits/dimension, the full key is 9 × 14 = 126 bits, fitting in a
 * 128-bit integer.  The bit layout (MSB → LSB) interleaves dimension bits
 * from the most significant coordinate bit down:
 *
 *   bit 125  = dim 8 bit 13   (z MSB)
 *   bit 124  = dim 7 bit 13   (y MSB)
 *   bit 123  = dim 6 bit 13   (x MSB)
 *   ...
 *   bit 8    = dim 8 bit 0    (z LSB)
 *   ...
 *   bit 0    = dim 0 bit 0    (r LSB)
 *
 * This ordering groups spatially nearby points under shared key prefixes,
 * enabling efficient range queries and LMDB prefix scans.
 *
 * Dimension semantics (from coordinate_semantics.hpp):
 *   dim 0 = Resonance (r)     dim 3 = U    dim 6 = X
 *   dim 1 = State     (s)     dim 4 = V    dim 7 = Y
 *   dim 2 = Time      (t)     dim 5 = W    dim 8 = Z
 */
#pragma once

#include <array>
#include <cstdint>

namespace nikola::spatial {

// ============================================================================
// Constants
// ============================================================================

/// Number of dimensions in the T⁹ manifold.
inline constexpr int MORTON_DIMS = 9;

/// Bits per coordinate dimension.
inline constexpr int MORTON_BITS_PER_DIM = 14;

/// Maximum coordinate value per dimension (exclusive): 2^14 = 16384.
inline constexpr uint32_t MORTON_COORD_MAX = 1u << MORTON_BITS_PER_DIM;

/// Total Morton key width in bits: 9 × 14 = 126.
inline constexpr int MORTON_TOTAL_BITS = MORTON_DIMS * MORTON_BITS_PER_DIM;

/// Coordinate type — same layout as HilbertScanner::Coord9D.
using Coord9D = std::array<uint32_t, MORTON_DIMS>;

/// 128-bit Morton key type.
using MortonKey = __uint128_t;

// ============================================================================
// Core API
// ============================================================================

/**
 * @brief Encode 9D coordinates into a 126-bit Morton key.
 *
 * Each coordinate must be in [0, 16384).  Bits are interleaved from MSB
 * down: for each bit position b = 13..0, the key receives dim 8's bit b,
 * then dim 7's bit b, ..., then dim 0's bit b.
 *
 * @param coords  9D coordinate array, each in [0, 2^14).
 * @return 126-bit Morton key.
 */
[[nodiscard]] inline constexpr MortonKey
morton_encode(const Coord9D& coords) noexcept {
    MortonKey key = 0;
    for (int bit = MORTON_BITS_PER_DIM - 1; bit >= 0; --bit) {
        for (int dim = MORTON_DIMS - 1; dim >= 0; --dim) {
            key <<= 1;
            key |= static_cast<MortonKey>((coords[dim] >> bit) & 1u);
        }
    }
    return key;
}

/**
 * @brief Decode a 126-bit Morton key back to 9D coordinates.
 *
 * Inverse of morton_encode(): extracts interleaved bits and reconstructs
 * each coordinate.
 *
 * @param key  126-bit Morton key.
 * @return 9D coordinate array, each in [0, 2^14).
 */
[[nodiscard]] inline constexpr Coord9D
morton_decode(MortonKey key) noexcept {
    Coord9D coords{};
    for (int bit = 0; bit < MORTON_BITS_PER_DIM; ++bit) {
        for (int dim = 0; dim < MORTON_DIMS; ++dim) {
            coords[dim] |= static_cast<uint32_t>(key & 1) << bit;
            key >>= 1;
        }
    }
    return coords;
}

/**
 * @brief Serialize a 128-bit Morton key to a 16-byte big-endian buffer.
 *
 * Big-endian encoding ensures lexicographic byte order matches numeric
 * order — required for LMDB prefix scans and range queries.
 *
 * @param key  The Morton key.
 * @param out  Pointer to at least 16 bytes of output storage.
 */
inline void morton_key_to_bytes(MortonKey key,
                                uint8_t out[16]) noexcept {
    for (int i = 15; i >= 0; --i) {
        out[i] = static_cast<uint8_t>(key & 0xFF);
        key >>= 8;
    }
}

/**
 * @brief Deserialize a 16-byte big-endian buffer to a 128-bit Morton key.
 *
 * @param in  Pointer to 16 bytes of big-endian Morton key data.
 * @return Reconstructed Morton key.
 */
[[nodiscard]] inline MortonKey
morton_key_from_bytes(const uint8_t in[16]) noexcept {
    MortonKey key = 0;
    for (int i = 0; i < 16; ++i) {
        key = (key << 8) | static_cast<MortonKey>(in[i]);
    }
    return key;
}

/**
 * @brief Check whether all coordinates are within the valid range.
 *
 * @param coords  9D coordinate array.
 * @return true if every element is in [0, 2^14).
 */
[[nodiscard]] inline constexpr bool
morton_coords_valid(const Coord9D& coords) noexcept {
    for (int d = 0; d < MORTON_DIMS; ++d) {
        if (coords[d] >= MORTON_COORD_MAX) return false;
    }
    return true;
}

} // namespace nikola::spatial
