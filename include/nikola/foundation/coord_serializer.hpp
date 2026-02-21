/**
 * @file foundation/coord_serializer.hpp
 * @brief NIK-003 — Portable little-endian serialization for Coord9D and wave data.
 *
 * Motivation:
 *   The internal grid code uses `Morton128 = __uint128_t` for fast Z-order
 *   encoding.  This type is not available on MSVC and has implementation-defined
 *   binary layout on other compilers.  C++ bitfields have implementation-defined
 *   packing, bit ordering, and alignment — they cannot be used for portable wire
 *   formats or on-disk checkpoint files (DMC format).
 *
 * Solution:
 *   This header provides a portable, endian-safe serialization layer that:
 *     - Explicitly writes IEEE 754 floats/doubles as little-endian bytes
 *     - Round-trips floats identically on any IEEE 754 platform
 *     - Works on both little-endian (x86-64, ARM LE) and big-endian (SPARC,
 *       s390x, MIPS BE) hosts using std::endian detection
 *     - Does NOT depend on __uint128_t, compiler bitfield tricks, or UB
 *
 * Coord9D wire layout (from TASK_COORD9D_PORTABILITY spec):
 *   9 floats × 4 bytes = 36 bytes, all little-endian
 *   [x0][x1][x2][x3][x4][x5][x6][x7][x8]
 *   where xi = float for dimension i of the 9D torus coordinates
 *
 *   Compact (bit-packed) layout is available via CoordWords using two uint64_t:
 *     lo  [r:4][s:4][t:14][u:8][v:8][w:8][x:14][pad:4]
 *     hi  [y:14][z:14][pad:36]
 *
 * Usage:
 * @code
 *   uint8_t buf[36];
 *   std::array<float,9> coord = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
 *                                 6.0f, 7.0f, 8.0f, 9.0f};
 *   nikola::foundation::CoordSerializer::serialize_coord(buf, coord);
 *   auto restored = nikola::foundation::CoordSerializer::deserialize_coord(buf);
 *   // restored == coord
 * @endcode
 *
 * @see TASKS.md  NIK-003
 * @see toroidal_grid.hpp  (Morton128 remains for internal fast-path)
 */

#pragma once

#include <array>
#include <bit>        // std::endian, std::bit_cast (C++20)
#include <complex>
#include <cstdint>
#include <cstring>    // std::memcpy
#include <stdexcept>

namespace nikola::foundation {

// ─────────────────────────────────────────────────────────────────────────────
//  CoordWords — compact bit-packed Coord9D representation (two uint64_t)
//  From TASK_COORD9D_PORTABILITY spec.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Compact integer representation of a 9D grid coordinate.
 *
 * Packs nine bounded integer indices into two portable uint64_t words,
 * avoiding __uint128_t and implementation-defined bitfield layout.
 *
 * Dimension ranges and bit widths:
 *   r, s            :  [0, 15]    4 bits each   (resonance/state, 16 levels)
 *   t, x            :  [0, 16383] 14 bits each  (temporal/spatial, grid dims)
 *   u, v, w         :  [0, 255]   8 bits each   (sub-dimensions)
 *   y, z            :  [0, 16383] 14 bits each
 *
 * Word layout (lo):  r[0-3] s[4-7] t[8-21] u[22-29] v[30-37] w[38-45] x[46-59] pad[60-63]
 * Word layout (hi):  y[0-13] z[14-27] pad[28-63]
 */
struct CoordWords {
    uint64_t lo = 0;
    uint64_t hi = 0;

    // ── bit masks ────────────────────────
    static constexpr uint64_t MASK_4  = 0x000000FULL;
    static constexpr uint64_t MASK_8  = 0x0000FFULL;
    static constexpr uint64_t MASK_14 = 0x3FFFULL;

    // ── bit shifts in 'lo' ───────────────
    static constexpr int SHR_R = 0;
    static constexpr int SHR_S = 4;
    static constexpr int SHR_T = 8;
    static constexpr int SHR_U = 22;
    static constexpr int SHR_V = 30;
    static constexpr int SHR_W = 38;
    static constexpr int SHR_X = 46;

    // ── bit shifts in 'hi' ───────────────
    static constexpr int SHH_Y = 0;
    static constexpr int SHH_Z = 14;

    // ── constructors ──────────────────────

    constexpr CoordWords() noexcept = default;

    /**
     * @brief Encode nine dimension indices into CoordWords.
     *
     * @param r  [0,15]     resonance level index
     * @param s  [0,15]     state level index
     * @param t  [0,16383]  temporal index
     * @param u  [0,255]    u sub-dimension
     * @param v  [0,255]    v sub-dimension
     * @param w  [0,255]    w sub-dimension
     * @param x  [0,16383]  x spatial index
     * @param y  [0,16383]  y spatial index
     * @param z  [0,16383]  z spatial index
     */
    constexpr CoordWords(
            uint64_t r, uint64_t s, uint64_t t,
            uint64_t u, uint64_t v, uint64_t w,
            uint64_t x, uint64_t y, uint64_t z) noexcept
    {
        lo = ((r & MASK_4)  << SHR_R)
           | ((s & MASK_4)  << SHR_S)
           | ((t & MASK_14) << SHR_T)
           | ((u & MASK_8)  << SHR_U)
           | ((v & MASK_8)  << SHR_V)
           | ((w & MASK_8)  << SHR_W)
           | ((x & MASK_14) << SHR_X);

        hi = ((y & MASK_14) << SHH_Y)
           | ((z & MASK_14) << SHH_Z);
    }

    // ── accessors ─────────────────────────

    constexpr uint64_t r() const noexcept { return (lo >> SHR_R) & MASK_4;  }
    constexpr uint64_t s() const noexcept { return (lo >> SHR_S) & MASK_4;  }
    constexpr uint64_t t() const noexcept { return (lo >> SHR_T) & MASK_14; }
    constexpr uint64_t u() const noexcept { return (lo >> SHR_U) & MASK_8;  }
    constexpr uint64_t v() const noexcept { return (lo >> SHR_V) & MASK_8;  }
    constexpr uint64_t w() const noexcept { return (lo >> SHR_W) & MASK_8;  }
    constexpr uint64_t x() const noexcept { return (lo >> SHR_X) & MASK_14; }
    constexpr uint64_t y() const noexcept { return (hi >> SHH_Y) & MASK_14; }
    constexpr uint64_t z() const noexcept { return (hi >> SHH_Z) & MASK_14; }

    constexpr bool operator==(const CoordWords& o) const noexcept {
        return lo == o.lo && hi == o.hi;
    }
    constexpr bool operator!=(const CoordWords& o) const noexcept {
        return !(*this == o);
    }
};

} // namespace nikola::foundation

// ── std::hash specialization ──────────────────────────────────────────────────

template<>
struct std::hash<nikola::foundation::CoordWords> {
    [[nodiscard]]
    std::size_t operator()(const nikola::foundation::CoordWords& c) const noexcept {
        const uint64_t mixed = c.lo ^ std::rotl(c.hi, 32);
        return std::hash<uint64_t>{}(mixed);
    }
};

namespace nikola::foundation {

// ─────────────────────────────────────────────────────────────────────────────
//  CoordSerializer — low-level IEEE 754 / little-endian I/O primitives
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Static utility class for portable byte-level serialization.
 *
 * All write_* functions store data in little-endian IEEE 754 byte order,
 * regardless of host endianness.  All read_* functions reverse the conversion.
 *
 * No alignment requirements are imposed on buf pointers.
 * All operations are noexcept where a valid buf is provided.
 */
class CoordSerializer {
public:
    CoordSerializer() = delete;

    // ── float (f32, 4 bytes) ─────────────────────────────────────────────────

    /**
     * @brief Write a float to buf[0..3] in little-endian IEEE 754 order.
     * @param buf  Destination — must have space for 4 bytes.
     * @param v    Value to serialize.
     */
    static void write_le_f32(uint8_t* buf, float v) noexcept {
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        buf[0] = static_cast<uint8_t>(bits       & 0xFF);
        buf[1] = static_cast<uint8_t>((bits >> 8) & 0xFF);
        buf[2] = static_cast<uint8_t>((bits >>16) & 0xFF);
        buf[3] = static_cast<uint8_t>((bits >>24) & 0xFF);
    }

    /**
     * @brief Read a float from buf[0..3] stored in little-endian IEEE 754.
     * @param buf  Source — must hold at least 4 bytes.
     * @return     Deserialized float.
     */
    [[nodiscard]]
    static float read_le_f32(const uint8_t* buf) noexcept {
        const uint32_t bits =
              static_cast<uint32_t>(buf[0])
            | (static_cast<uint32_t>(buf[1]) <<  8)
            | (static_cast<uint32_t>(buf[2]) << 16)
            | (static_cast<uint32_t>(buf[3]) << 24);
        float v;
        std::memcpy(&v, &bits, 4);
        return v;
    }

    // ── double (f64, 8 bytes) ────────────────────────────────────────────────

    /**
     * @brief Write a double to buf[0..7] in little-endian IEEE 754 order.
     */
    static void write_le_f64(uint8_t* buf, double v) noexcept {
        uint64_t bits;
        std::memcpy(&bits, &v, 8);
        for (int i = 0; i < 8; ++i)
            buf[i] = static_cast<uint8_t>((bits >> (8*i)) & 0xFF);
    }

    /**
     * @brief Read a double from buf[0..7] stored in little-endian IEEE 754.
     */
    [[nodiscard]]
    static double read_le_f64(const uint8_t* buf) noexcept {
        uint64_t bits = 0;
        for (int i = 0; i < 8; ++i)
            bits |= static_cast<uint64_t>(buf[i]) << (8*i);
        double v;
        std::memcpy(&v, &bits, 8);
        return v;
    }

    // ── std::complex<float> (c64, 8 bytes) ─────────────────────────────────

    /**
     * @brief Write complex<float> to buf[0..7]: re (4 bytes LE), im (4 bytes LE).
     */
    static void write_le_c64(uint8_t* buf, std::complex<float> v) noexcept {
        write_le_f32(buf,     v.real());
        write_le_f32(buf + 4, v.imag());
    }

    /**
     * @brief Read complex<float> from buf[0..7].
     */
    [[nodiscard]]
    static std::complex<float> read_le_c64(const uint8_t* buf) noexcept {
        return { read_le_f32(buf), read_le_f32(buf + 4) };
    }

    // ── uint64_t (8 bytes) ───────────────────────────────────────────────────

    /**
     * @brief Write uint64_t to buf[0..7] in little-endian order.
     */
    static void write_le_u64(uint8_t* buf, uint64_t v) noexcept {
        for (int i = 0; i < 8; ++i)
            buf[i] = static_cast<uint8_t>((v >> (8*i)) & 0xFF);
    }

    /**
     * @brief Read uint64_t from buf[0..7] stored in little-endian order.
     */
    [[nodiscard]]
    static uint64_t read_le_u64(const uint8_t* buf) noexcept {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i)
            v |= static_cast<uint64_t>(buf[i]) << (8*i);
        return v;
    }

    // ── CoordWords (16 bytes) ────────────────────────────────────────────────

    /**
     * @brief Serialize a CoordWords to buf[0..15] (lo then hi, both LE).
     */
    static void write_coord_words(uint8_t* buf, const CoordWords& c) noexcept {
        write_le_u64(buf,     c.lo);
        write_le_u64(buf + 8, c.hi);
    }

    /**
     * @brief Deserialize a CoordWords from buf[0..15].
     */
    [[nodiscard]]
    static CoordWords read_coord_words(const uint8_t* buf) noexcept {
        CoordWords c;
        c.lo = read_le_u64(buf);
        c.hi = read_le_u64(buf + 8);
        return c;
    }

    // ── Coord9D float array (36 bytes) ───────────────────────────────────────

    /**
     * @brief Serialize a 9D float coordinate to 36 bytes (9 × 4-byte LE floats).
     *
     * Wire format:  [dim0: 4 bytes LE][dim1: 4 bytes LE] ... [dim8: 4 bytes LE]
     *
     * @param buf    Destination buffer — must hold at least 36 bytes.
     * @param coord  Input array of 9 floats (one per torus dimension).
     */
    static void serialize_coord(uint8_t* buf, const std::array<float, 9>& coord) noexcept {
        for (int i = 0; i < 9; ++i)
            write_le_f32(buf + 4*i, coord[static_cast<std::size_t>(i)]);
    }

    /**
     * @brief Deserialize a 9D float coordinate from 36 bytes.
     *
     * @param buf  Source buffer — must hold at least 36 bytes.
     * @return     Array of 9 floats.
     */
    [[nodiscard]]
    static std::array<float, 9> deserialize_coord(const uint8_t* buf) noexcept {
        std::array<float, 9> coord{};
        for (int i = 0; i < 9; ++i)
            coord[static_cast<std::size_t>(i)] = read_le_f32(buf + 4*i);
        return coord;
    }

    // ── Validation ───────────────────────────────────────────────────────────

    /**
     * @brief Return true iff v is a finite, non-denormal IEEE 754 float.
     *
     * Rejects: NaN, ±Inf, and subnormals (denormals).  Useful for validating
     * wavefunction data before serialization.
     *
     * @param v  Value to check.
     * @return   true if safe to write; false if degenerate.
     */
    [[nodiscard]]
    static bool is_valid_f32(float v) noexcept {
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        const uint32_t exp  = (bits >> 23) & 0xFF;
        const uint32_t mant =  bits        & 0x7FFFFF;

        if (exp == 0xFF) return false;   // NaN or Inf
        if (exp == 0x00 && mant != 0) return false;   // denormal (subnormal)
        return true;
    }

    /**
     * @brief Return true iff v is a finite, non-denormal IEEE 754 double.
     */
    [[nodiscard]]
    static bool is_valid_f64(double v) noexcept {
        uint64_t bits;
        std::memcpy(&bits, &v, 8);
        const uint64_t exp  = (bits >> 52) & 0x7FF;
        const uint64_t mant =  bits        & 0x000FFFFFFFFFFFFFULL;

        if (exp == 0x7FF) return false;   // NaN or Inf
        if (exp == 0x000 && mant != 0) return false;   // denormal
        return true;
    }

    /**
     * @brief Return true iff all 9 components of coord are valid (finite, normal).
     */
    [[nodiscard]]
    static bool is_valid_coord(const std::array<float, 9>& coord) noexcept {
        for (const float v : coord)
            if (!is_valid_f32(v)) return false;
        return true;
    }
};

} // namespace nikola::foundation
