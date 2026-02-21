/**
 * @file tests/unit/phase16_coord_serializer_test.cpp
 * @brief NIK-003 — Coord9D Portability test suite (Catch2 v3).
 *
 * Background:
 *   toroidal_grid.hpp uses `Morton128 = __uint128_t` for internal Z-order
 *   encoding.  This is non-portable (not available on MSVC) and C++ bitfields
 *   have implementation-defined binary layout, making checkpoint files
 *   non-portable across compilers and architectures.
 *
 *   CoordSerializer provides an explicit little-endian IEEE 754 serialization
 *   layer that round-trips correctly on all IEEE 754 platforms.
 *
 * Tests:
 *   NIK-003-A  known byte patterns for specific float values
 *   NIK-003-B  read_le_f32 / write_le_f32 round-trip
 *   NIK-003-C  read_le_f64 / write_le_f64 round-trip
 *   NIK-003-D  read_le_c64 / write_le_c64 round-trip
 *   NIK-003-E  serialize_coord / deserialize_coord round-trip (array<float,9>)
 *   NIK-003-F  little-endian byte order verification (known values)
 *   NIK-003-G  is_valid_f32 / is_valid_f64 — NaN, Inf, denormal rejection
 *   NIK-003-H  is_valid_coord — all-valid and mixed coord arrays
 *   NIK-003-I  CoordWords — encode/decode round-trip for all 9 dimensions
 *   NIK-003-J  CoordWords — std::hash specialization produces consistent value
 *   NIK-003-K  CoordWords — equality and inequality operators
 *   NIK-003-L  read_le_u64 / write_le_u64 round-trip
 *   NIK-003-M  write_coord_words / read_coord_words round-trip
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <nikola/foundation/coord_serializer.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <unordered_set>

using Catch::Approx;
using Catch::Matchers::WithinAbs;
using nikola::foundation::CoordSerializer;
using nikola::foundation::CoordWords;

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-003-A  Known byte patterns
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-003-A — 1.0f serializes to known LE bytes",
          "[coord_serializer][nik003][bytes]")
{
    // IEEE 754 single 1.0f = 0x3F800000
    // LE bytes: [0x00, 0x00, 0x80, 0x3F]
    uint8_t buf[4] = {0};
    CoordSerializer::write_le_f32(buf, 1.0f);
    CHECK(buf[0] == 0x00);
    CHECK(buf[1] == 0x00);
    CHECK(buf[2] == 0x80);
    CHECK(buf[3] == 0x3F);
}

TEST_CASE("NIK-003-A — -1.0f serializes to known LE bytes",
          "[coord_serializer][nik003][bytes]")
{
    // -1.0f = 0xBF800000 → LE: [0x00, 0x00, 0x80, 0xBF]
    uint8_t buf[4] = {0};
    CoordSerializer::write_le_f32(buf, -1.0f);
    CHECK(buf[0] == 0x00);
    CHECK(buf[1] == 0x00);
    CHECK(buf[2] == 0x80);
    CHECK(buf[3] == 0xBF);
}

TEST_CASE("NIK-003-A — 0.0f serializes to all-zero bytes",
          "[coord_serializer][nik003][bytes]")
{
    uint8_t buf[4] = {0xFF, 0xFF, 0xFF, 0xFF};
    CoordSerializer::write_le_f32(buf, 0.0f);
    CHECK(buf[0] == 0x00);
    CHECK(buf[1] == 0x00);
    CHECK(buf[2] == 0x00);
    CHECK(buf[3] == 0x00);
}

TEST_CASE("NIK-003-A — 1.0 (double) serializes to known LE bytes",
          "[coord_serializer][nik003][bytes]")
{
    // 1.0 double = 0x3FF0000000000000
    // LE: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F]
    uint8_t buf[8] = {0};
    CoordSerializer::write_le_f64(buf, 1.0);
    CHECK(buf[0] == 0x00);
    CHECK(buf[1] == 0x00);
    CHECK(buf[2] == 0x00);
    CHECK(buf[3] == 0x00);
    CHECK(buf[4] == 0x00);
    CHECK(buf[5] == 0x00);
    CHECK(buf[6] == 0xF0);
    CHECK(buf[7] == 0x3F);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-003-B  Float round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-003-B — float round-trips through write/read",
          "[coord_serializer][nik003][roundtrip]")
{
    const std::array<float, 8> values = {
        0.0f, 1.0f, -1.0f, 3.14159265f,
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::min(),   // smallest positive normal
        -0.00123456f,
        1.234567891e10f
    };

    uint8_t buf[4];
    for (float v : values) {
        CoordSerializer::write_le_f32(buf, v);
        const float restored = CoordSerializer::read_le_f32(buf);
        // Bit-exact round-trip required
        CHECK(restored == v);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-003-C  Double round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-003-C — double round-trips through write/read",
          "[coord_serializer][nik003][roundtrip]")
{
    const std::array<double, 6> values = {
        0.0, 1.0, -1.0,
        3.141592653589793,
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::min()
    };

    uint8_t buf[8];
    for (double v : values) {
        CoordSerializer::write_le_f64(buf, v);
        const double restored = CoordSerializer::read_le_f64(buf);
        CHECK(restored == v);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-003-D  complex<float> round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-003-D — complex<float> round-trips through write/read",
          "[coord_serializer][nik003][roundtrip]")
{
    using CF = std::complex<float>;
    const std::array<CF, 4> values = {
        CF{1.0f, 0.0f}, CF{0.0f, 1.0f}, CF{-1.0f, -1.0f}, CF{3.5f, -2.7f}
    };

    uint8_t buf[8];
    for (const CF& v : values) {
        CoordSerializer::write_le_c64(buf, v);
        const CF restored = CoordSerializer::read_le_c64(buf);
        CHECK(restored.real() == v.real());
        CHECK(restored.imag() == v.imag());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-003-E  serialize_coord / deserialize_coord round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-003-E — Coord9D array round-trips (36 bytes)",
          "[coord_serializer][nik003][roundtrip][coord9d]")
{
    const std::array<float, 9> coord = {
        1.0f, 2.5f, -3.14f, 0.0f, 1e6f,
        -1e-6f, 0.333333f, 100.0f, -200.0f
    };

    uint8_t buf[36];
    CoordSerializer::serialize_coord(buf, coord);
    const auto restored = CoordSerializer::deserialize_coord(buf);

    for (int i = 0; i < 9; ++i)
        CHECK(restored[static_cast<std::size_t>(i)]
              == coord[static_cast<std::size_t>(i)]);
}

TEST_CASE("NIK-003-E — all-zero coord round-trips",
          "[coord_serializer][nik003][roundtrip][coord9d]")
{
    const std::array<float, 9> coord = {};
    uint8_t buf[36];
    CoordSerializer::serialize_coord(buf, coord);
    const auto restored = CoordSerializer::deserialize_coord(buf);
    for (int i = 0; i < 9; ++i)
        CHECK(restored[static_cast<std::size_t>(i)] == 0.0f);
}

TEST_CASE("NIK-003-E — coord consumes exactly 36 bytes",
          "[coord_serializer][nik003][roundtrip][coord9d]")
{
    std::array<uint8_t, 40> buf;
    buf.fill(0xAA);  // sentinel

    const std::array<float, 9> coord = {1,2,3,4,5,6,7,8,9};
    CoordSerializer::serialize_coord(buf.data(), coord);

    // sentinel bytes at offset 36,37,38,39 should be untouched
    CHECK(buf[36] == 0xAA);
    CHECK(buf[37] == 0xAA);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-003-F  Little-endian byte order verification
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-003-F — bytes are written in little-endian order",
          "[coord_serializer][nik003][endian]")
{
    // Write 0x12345678 as a float using bit_cast
    constexpr uint32_t pattern = 0x12345678u;
    float v;
    std::memcpy(&v, &pattern, 4);

    uint8_t buf[4];
    CoordSerializer::write_le_f32(buf, v);

    // LE: LSB first
    CHECK(buf[0] == 0x78);
    CHECK(buf[1] == 0x56);
    CHECK(buf[2] == 0x34);
    CHECK(buf[3] == 0x12);
}

TEST_CASE("NIK-003-F — reading pre-built LE bytes gives expected float",
          "[coord_serializer][nik003][endian]")
{
    // 0x3F000000 = 0.5f in IEEE 754
    // LE bytes: [0x00, 0x00, 0x00, 0x3F]
    const uint8_t buf[4] = {0x00, 0x00, 0x00, 0x3F};
    const float v = CoordSerializer::read_le_f32(buf);
    CHECK(v == Approx(0.5f));
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-003-G  is_valid_f32 / is_valid_f64
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-003-G — normal floats pass is_valid_f32",
          "[coord_serializer][nik003][validation]")
{
    CHECK(CoordSerializer::is_valid_f32(0.0f));
    CHECK(CoordSerializer::is_valid_f32(1.0f));
    CHECK(CoordSerializer::is_valid_f32(-1.0f));
    CHECK(CoordSerializer::is_valid_f32(std::numeric_limits<float>::max()));
    CHECK(CoordSerializer::is_valid_f32(std::numeric_limits<float>::min()));
}

TEST_CASE("NIK-003-G — NaN rejected by is_valid_f32",
          "[coord_serializer][nik003][validation]")
{
    CHECK_FALSE(CoordSerializer::is_valid_f32(std::numeric_limits<float>::quiet_NaN()));
    CHECK_FALSE(CoordSerializer::is_valid_f32(std::numeric_limits<float>::signaling_NaN()));
}

TEST_CASE("NIK-003-G — Inf rejected by is_valid_f32",
          "[coord_serializer][nik003][validation]")
{
    CHECK_FALSE(CoordSerializer::is_valid_f32(std::numeric_limits<float>::infinity()));
    CHECK_FALSE(CoordSerializer::is_valid_f32(-std::numeric_limits<float>::infinity()));
}

TEST_CASE("NIK-003-G — denormals rejected by is_valid_f32",
          "[coord_serializer][nik003][validation]")
{
    CHECK_FALSE(CoordSerializer::is_valid_f32(std::numeric_limits<float>::denorm_min()));
}

TEST_CASE("NIK-003-G — normal doubles pass is_valid_f64",
          "[coord_serializer][nik003][validation]")
{
    CHECK(CoordSerializer::is_valid_f64(0.0));
    CHECK(CoordSerializer::is_valid_f64(1.0));
    CHECK(CoordSerializer::is_valid_f64(3.14159265358979));
}

TEST_CASE("NIK-003-G — NaN rejected by is_valid_f64",
          "[coord_serializer][nik003][validation]")
{
    CHECK_FALSE(CoordSerializer::is_valid_f64(std::numeric_limits<double>::quiet_NaN()));
}

TEST_CASE("NIK-003-G — Inf rejected by is_valid_f64",
          "[coord_serializer][nik003][validation]")
{
    CHECK_FALSE(CoordSerializer::is_valid_f64(std::numeric_limits<double>::infinity()));
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-003-H  is_valid_coord
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-003-H — all-normal coord passes is_valid_coord",
          "[coord_serializer][nik003][validation][coord9d]")
{
    const std::array<float, 9> coord = {1,2,3,4,5,6,7,8,9};
    CHECK(CoordSerializer::is_valid_coord(coord));
}

TEST_CASE("NIK-003-H — coord with one NaN fails is_valid_coord",
          "[coord_serializer][nik003][validation][coord9d]")
{
    std::array<float, 9> coord = {1,2,3,4,5,6,7,8,9};
    coord[4] = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(CoordSerializer::is_valid_coord(coord));
}

TEST_CASE("NIK-003-H — coord with one Inf fails is_valid_coord",
          "[coord_serializer][nik003][validation][coord9d]")
{
    std::array<float, 9> coord = {0,0,0,0,0,0,0,0,0};
    coord[8] = std::numeric_limits<float>::infinity();
    CHECK_FALSE(CoordSerializer::is_valid_coord(coord));
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-003-I  CoordWords encode / decode round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-003-I — CoordWords encodes and decodes all 9 dimensions",
          "[coord_serializer][nik003][coord_words]")
{
    const CoordWords c{
        /*r=*/12, /*s=*/ 5, /*t=*/ 8192,
        /*u=*/200, /*v=*/ 50, /*w=*/ 127,
        /*x=*/4000, /*y=*/ 100, /*z=*/ 16383
    };

    CHECK(c.r() == 12);
    CHECK(c.s() ==  5);
    CHECK(c.t() == 8192);
    CHECK(c.u() == 200);
    CHECK(c.v() ==  50);
    CHECK(c.w() == 127);
    CHECK(c.x() == 4000);
    CHECK(c.y() == 100);
    CHECK(c.z() == 16383);
}

TEST_CASE("NIK-003-I — CoordWords max values don't bleed into adjacent fields",
          "[coord_serializer][nik003][coord_words]")
{
    // r and s are 4-bit; t and x are 14-bit; u,v,w are 8-bit
    const CoordWords c{15, 15, 16383, 255, 255, 255, 16383, 16383, 16383};

    CHECK(c.r() == 15);
    CHECK(c.s() == 15);
    CHECK(c.t() == 16383);
    CHECK(c.u() == 255);
    CHECK(c.v() == 255);
    CHECK(c.w() == 255);
    CHECK(c.x() == 16383);
    CHECK(c.y() == 16383);
    CHECK(c.z() == 16383);
}

TEST_CASE("NIK-003-I — default CoordWords is all zeros",
          "[coord_serializer][nik003][coord_words]")
{
    const CoordWords c;
    CHECK(c.r() == 0);
    CHECK(c.s() == 0);
    CHECK(c.t() == 0);
    CHECK(c.u() == 0);
    CHECK(c.v() == 0);
    CHECK(c.w() == 0);
    CHECK(c.x() == 0);
    CHECK(c.y() == 0);
    CHECK(c.z() == 0);
    CHECK(c.lo == 0);
    CHECK(c.hi == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-003-J  std::hash specialization
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-003-J — hash is consistent for equal CoordWords",
          "[coord_serializer][nik003][hash]")
{
    const CoordWords a{1, 2, 100, 50, 60, 70, 3000, 200, 300};
    const CoordWords b{1, 2, 100, 50, 60, 70, 3000, 200, 300};

    const auto hash_a = std::hash<CoordWords>{}(a);
    const auto hash_b = std::hash<CoordWords>{}(b);
    CHECK(hash_a == hash_b);
}

TEST_CASE("NIK-003-J — hash differs for distinct CoordWords (basic sanity)",
          "[coord_serializer][nik003][hash]")
{
    const CoordWords a{1, 0, 0, 0, 0, 0, 0, 0, 0};
    const CoordWords b{2, 0, 0, 0, 0, 0, 0, 0, 0};

    const auto hash_a = std::hash<CoordWords>{}(a);
    const auto hash_b = std::hash<CoordWords>{}(b);
    // Different coords — hashes should differ (not a proof, but basic sanity)
    CHECK(hash_a != hash_b);
}

TEST_CASE("NIK-003-J — CoordWords usable in unordered_set",
          "[coord_serializer][nik003][hash]")
{
    std::unordered_set<CoordWords> s;
    s.insert(CoordWords{1, 2, 3, 4, 5, 6, 7, 8, 9});
    s.insert(CoordWords{9, 8, 7, 6, 5, 4, 3, 2, 1});
    s.insert(CoordWords{1, 2, 3, 4, 5, 6, 7, 8, 9});  // duplicate

    CHECK(s.size() == 2);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-003-K  Equality and inequality operators
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-003-K — equal CoordWords compare equal",
          "[coord_serializer][nik003][equality]")
{
    const CoordWords a{3, 14, 159, 26, 53, 58, 979, 323, 846};
    const CoordWords b{3, 14, 159, 26, 53, 58, 979, 323, 846};
    CHECK(a == b);
    CHECK_FALSE(a != b);
}

TEST_CASE("NIK-003-K — different CoordWords compare not equal",
          "[coord_serializer][nik003][equality]")
{
    const CoordWords a{0, 0, 0, 0, 0, 0, 0, 0, 1};
    const CoordWords b{0, 0, 0, 0, 0, 0, 0, 0, 2};
    CHECK(a != b);
    CHECK_FALSE(a == b);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-003-L  uint64_t round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-003-L — uint64_t round-trips through write/read",
          "[coord_serializer][nik003][roundtrip]")
{
    const std::array<uint64_t, 4> values = {
        0ULL, 0xDEADBEEFCAFEBABEULL,
        std::numeric_limits<uint64_t>::max(),
        0x0102030405060708ULL
    };

    uint8_t buf[8];
    for (uint64_t v : values) {
        CoordSerializer::write_le_u64(buf, v);
        const uint64_t restored = CoordSerializer::read_le_u64(buf);
        CHECK(restored == v);
    }
}

TEST_CASE("NIK-003-L — uint64_t writes in little-endian order",
          "[coord_serializer][nik003][endian]")
{
    uint8_t buf[8];
    CoordSerializer::write_le_u64(buf, 0x0102030405060708ULL);
    // LE: LSB first
    CHECK(buf[0] == 0x08);
    CHECK(buf[1] == 0x07);
    CHECK(buf[2] == 0x06);
    CHECK(buf[3] == 0x05);
    CHECK(buf[4] == 0x04);
    CHECK(buf[5] == 0x03);
    CHECK(buf[6] == 0x02);
    CHECK(buf[7] == 0x01);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NIK-003-M  CoordWords serialization round-trip (16 bytes)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NIK-003-M — CoordWords round-trips through write_coord_words/read",
          "[coord_serializer][nik003][coord_words][roundtrip]")
{
    const CoordWords original{5, 10, 1234, 99, 77, 55, 8000, 12345, 9999};

    uint8_t buf[16];
    CoordSerializer::write_coord_words(buf, original);
    const CoordWords restored = CoordSerializer::read_coord_words(buf);

    CHECK(restored == original);
    CHECK(restored.r() == original.r());
    CHECK(restored.t() == original.t());
    CHECK(restored.z() == original.z());
}

TEST_CASE("NIK-003-M — CoordWords serialize uses exactly 16 bytes",
          "[coord_serializer][nik003][coord_words]")
{
    std::array<uint8_t, 20> buf;
    buf.fill(0xAA);

    CoordSerializer::write_coord_words(buf.data(), CoordWords{1,2,3,4,5,6,7,8,9});

    // Sentinel bytes at offset 16..19 must be untouched
    CHECK(buf[16] == 0xAA);
    CHECK(buf[17] == 0xAA);
}
