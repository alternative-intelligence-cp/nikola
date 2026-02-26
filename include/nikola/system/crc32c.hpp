#pragma once
/**
 * @file nikola/system/crc32c.hpp
 * @brief Phase 95 — GAP-038: CRC32C hardware intrinsics
 *
 * Hardware-accelerated CRC32C (Castagnoli polynomial, 0x1EDC6F41) using
 * SSE4.2 _mm_crc32_u{8,16,32,64} intrinsics on x86-64.  Falls back to a
 * portable Sarwate table-lookup implementation when SSE4.2 is unavailable.
 *
 * Spec references:
 *   RFC 3720 §B.4  — iSCSI CRC32C reference test vectors
 *   Nikola §6.1    — DMC persistence page/WAL checksum validation
 *   Nikola §GAP-019 — partition table migration integrity (CRC32C field)
 *   Nikola §GAP-038 — "10× speedup over software" requirement (SSE4.2)
 *
 * Build requirement: compile with -msse4.2 to activate the hardware path.
 * The #ifdef __SSE4_2__ guard ensures graceful degradation on older CPUs.
 *
 * Public API summary:
 *   crc32c(buf, len, seed=0)      — auto-select HW or SW
 *   crc32c_hw(buf, len, seed=0)   — explicit HW path (SSE4.2 only)
 *   crc32c_sw(buf, len, seed=0)   — explicit SW path (always available)
 *
 * Chaining example:
 *   uint32_t crc = crc32c(header, hlen);
 *   crc = crc32c(payload, plen, crc);   // same as crc32c(header+payload, total)
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#ifdef __SSE4_2__
#  include <nmmintrin.h>
#endif

namespace nikola::system {

// ============================================================================
// Detail: compile-time Sarwate lookup table
// ============================================================================
namespace detail {

/// Reflected Castagnoli CRC32C polynomial.
inline constexpr uint32_t CRC32C_POLY = 0x82F63B78U;

/// Generates the 256-entry Sarwate lookup table at compile time.
[[nodiscard]] consteval std::array<uint32_t, 256> make_crc32c_table() noexcept {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256U; ++i) {
        uint32_t crc = i;
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (CRC32C_POLY & static_cast<uint32_t>(-(crc & 1U)));
        t[i] = crc;
    }
    return t;
}

inline constexpr auto CRC32C_TABLE = make_crc32c_table();

} // namespace detail

// ============================================================================
// Capability flag
// ============================================================================

/// True when the build was compiled with -msse4.2 (hardware CRC32C available).
#ifdef __SSE4_2__
inline constexpr bool CRC32C_HW_AVAILABLE = true;
#else
inline constexpr bool CRC32C_HW_AVAILABLE = false;
#endif

// ============================================================================
// RFC 3720 §B.4 reference constants (useful in static_asserts and tests)
// ============================================================================

/// crc32c("") == 0x00000000  (universally agreed)
inline constexpr uint32_t CRC32C_EMPTY    = 0x00000000U;
/// crc32c("123456789") == 0xE3069283  (CRC catalog standard / RFC check value)
inline constexpr uint32_t CRC32C_DIGITS   = 0xE3069283U;
/// crc32c(32 × 0x00) == 0x8A9136AA  (verified against CRC catalog)
inline constexpr uint32_t CRC32C_32ZEROS  = 0x8A9136AAU;
/// crc32c(32 × 0xFF) == 0x62A8AB43  (verified against CRC catalog)
inline constexpr uint32_t CRC32C_32FF     = 0x62A8AB43U;
/// crc32c(0x00,0x01,...,0x1F) == 0x46DD794E  (32-byte sequential, verified)
inline constexpr uint32_t CRC32C_32SEQ    = 0x46DD794EU;

// ============================================================================
// Software path (always available)
// ============================================================================

/**
 * @brief CRC32C via portable Sarwate table lookup.
 *
 * Supports incremental/chained computation: pass the return value of a prior
 * call as @p seed to continue the checksum across multiple buffers.  Passing
 * seed=0 (default) is equivalent to starting a fresh computation.
 *
 * @param buf   Pointer to data (may be nullptr iff len == 0).
 * @param len   Number of bytes to process.
 * @param seed  Prior CRC32C result, or 0 to start fresh.
 * @return      CRC32C checksum of the logical byte stream seed+buf[0..len).
 */
[[nodiscard]] inline uint32_t
crc32c_sw(const void* buf, size_t len, uint32_t seed = 0U) noexcept {
    const auto* p = static_cast<const uint8_t*>(buf);
    // XOR with 0xFFFFFFFF "un-finalises" the seed so we reenter mid-stream.
    uint32_t crc = seed ^ 0xFFFFFFFFU;
    while (len--)
        crc = (crc >> 8) ^ detail::CRC32C_TABLE[(crc ^ *p++) & 0xFFU];
    return crc ^ 0xFFFFFFFFU;
}

// ============================================================================
// Hardware path (SSE4.2 only)
// ============================================================================

#ifdef __SSE4_2__
/**
 * @brief CRC32C via SSE4.2 _mm_crc32_u{64,32,16,8} intrinsics.
 *
 * Processes 8 bytes per instruction in the main loop; handles 4-, 2-, and
 * 1-byte tails.  Gives identical results to crc32c_sw() for all inputs.
 *
 * Only compiled when -msse4.2 is passed; the __SSE4_2__ macro is set by GCC,
 * Clang, and ICC when that flag is present.
 *
 * @param buf   Pointer to data.
 * @param len   Number of bytes.
 * @param seed  Prior CRC32C, or 0 to start fresh.
 * @return      CRC32C checksum.
 */
[[nodiscard]] inline uint32_t
crc32c_hw(const void* buf, size_t len, uint32_t seed = 0U) noexcept {
    const auto* p = static_cast<const uint8_t*>(buf);
    // Use uint64_t accumulator — _mm_crc32_u64 returns uint64_t but only
    // the lower 32 bits are significant.
    uint64_t crc = static_cast<uint64_t>(seed ^ 0xFFFFFFFFU);

    // 8-byte chunks (main throughput path)
    while (len >= 8U) {
        uint64_t block;
        std::memcpy(&block, p, 8U);
        crc = _mm_crc32_u64(crc, block);
        p += 8U; len -= 8U;
    }
    // 4-byte chunk
    if (len >= 4U) {
        uint32_t block;
        std::memcpy(&block, p, 4U);
        crc = _mm_crc32_u32(static_cast<uint32_t>(crc), block);
        p += 4U; len -= 4U;
    }
    // 2-byte chunk
    if (len >= 2U) {
        uint16_t block;
        std::memcpy(&block, p, 2U);
        crc = _mm_crc32_u16(static_cast<uint32_t>(crc), block);
        p += 2U; len -= 2U;
    }
    // Remaining byte
    if (len) {
        crc = _mm_crc32_u8(static_cast<uint32_t>(crc), *p);
    }
    return static_cast<uint32_t>(crc) ^ 0xFFFFFFFFU;
}
#endif // __SSE4_2__

// ============================================================================
// Unified entry point (auto-selects HW or SW)
// ============================================================================

/**
 * @brief Compute CRC32C, automatically using hardware when available.
 *
 * This is the function that should be called by all Nikola subsystems
 * (DMC persistence, partition table protocol, ZMQ spine checksum, etc.).
 *
 * Performance note (from Nikola §6 persistence spec):
 *   Hardware path ≈10× faster than software for large buffers (~8 GB/s with
 *   SSE4.2 on a modern CPU vs ~800 MB/s for table lookup).
 *
 * @param buf   Data to checksum (may be nullptr iff len == 0).
 * @param len   Number of bytes.
 * @param seed  Prior CRC32C result, or 0 to start fresh.
 * @return      CRC32C checksum.
 */
[[nodiscard]] inline uint32_t
crc32c(const void* buf, size_t len, uint32_t seed = 0U) noexcept {
#ifdef __SSE4_2__
    return crc32c_hw(buf, len, seed);
#else
    return crc32c_sw(buf, len, seed);
#endif
}

// ============================================================================
// Convenience overloads
// ============================================================================

/// Compute CRC32C of a typed array (e.g., uint8_t buffer, struct, etc.).
template<typename T>
[[nodiscard]] inline uint32_t crc32c_of(const T& value, uint32_t seed = 0U) noexcept {
    return crc32c(std::addressof(value), sizeof(T), seed);
}

} // namespace nikola::system
