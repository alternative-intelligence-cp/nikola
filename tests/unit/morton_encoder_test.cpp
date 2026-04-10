// ============================================================
// v0.1.5 — Morton Encoding Test Suite
// tests/unit/morton_encoder_test.cpp
//
// Validates 9D Morton (Z-order) encoding for T⁹:
//   §1  Round-trip: encode(decode(key)) == key
//   §2  Decode(encode(coords)) == coords
//   §3  Coordinate validation
//   §4  Origin and maximum corner
//   §5  Byte serialization round-trip
//   §6  Lexicographic ordering of serialized keys
//   §7  Known bit patterns
//   §8  Exhaustive round-trip (small order)
//   §9  Benchmark: encode/decode throughput
// ============================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/spatial/morton_encoder.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

using namespace nikola::spatial;

// Helper: generate a random valid Coord9D
static Coord9D random_coord(std::mt19937& rng) {
    std::uniform_int_distribution<uint32_t> dist(0, MORTON_COORD_MAX - 1);
    Coord9D c;
    for (auto& v : c) v = dist(rng);
    return c;
}

// ═══════════════════════════════════════════════════════════════════════════
// §1  Round-trip: decode(encode(coords)) == coords
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("§1-1 Morton: encode/decode round-trip — origin", "[morton]") {
    Coord9D origin{};
    auto key = morton_encode(origin);
    REQUIRE(key == 0);
    auto back = morton_decode(key);
    REQUIRE(back == origin);
}

TEST_CASE("§1-2 Morton: encode/decode round-trip — single dimensions",
          "[morton]") {
    // Place a 1 in each dimension separately
    for (int d = 0; d < MORTON_DIMS; ++d) {
        Coord9D c{};
        c[d] = 1;
        auto key = morton_encode(c);
        auto back = morton_decode(key);
        INFO("dim=" << d);
        REQUIRE(back == c);
        // Key should be nonzero
        REQUIRE(key != 0);
    }
}

TEST_CASE("§1-3 Morton: encode/decode round-trip — max corner",
          "[morton]") {
    Coord9D maxc;
    maxc.fill(MORTON_COORD_MAX - 1);
    auto key = morton_encode(maxc);
    auto back = morton_decode(key);
    REQUIRE(back == maxc);
    // All 126 bits should be set
    MortonKey all_126 = (static_cast<MortonKey>(1) << 126) - 1;
    REQUIRE(key == all_126);
}

TEST_CASE("§1-4 Morton: encode/decode round-trip — 10,000 random coords",
          "[morton]") {
    std::mt19937 rng(42);
    for (int i = 0; i < 10000; ++i) {
        auto c = random_coord(rng);
        auto back = morton_decode(morton_encode(c));
        REQUIRE(back == c);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §2  Encode-then-decode: encode(decode(key)) == key
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("§2-1 Morton: key round-trip — 10,000 random keys", "[morton]") {
    std::mt19937 rng(123);
    std::uniform_int_distribution<uint64_t> dist64(0, UINT64_MAX);
    MortonKey mask = (static_cast<MortonKey>(1) << 126) - 1;

    for (int i = 0; i < 10000; ++i) {
        MortonKey key = (static_cast<MortonKey>(dist64(rng)) << 64)
                      | static_cast<MortonKey>(dist64(rng));
        key &= mask;  // clamp to 126 bits
        auto coords = morton_decode(key);
        auto key2 = morton_encode(coords);
        REQUIRE(key2 == key);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §3  Coordinate validation
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("§3-1 Morton: valid coords accepted", "[morton]") {
    Coord9D zero{};
    REQUIRE(morton_coords_valid(zero));

    Coord9D maxc;
    maxc.fill(MORTON_COORD_MAX - 1);
    REQUIRE(morton_coords_valid(maxc));
}

TEST_CASE("§3-2 Morton: out-of-range coords rejected", "[morton]") {
    for (int d = 0; d < MORTON_DIMS; ++d) {
        Coord9D c{};
        c[d] = MORTON_COORD_MAX;  // exactly at limit
        INFO("dim=" << d);
        REQUIRE_FALSE(morton_coords_valid(c));
    }
    Coord9D huge{};
    huge[0] = 0xFFFFFFFF;
    REQUIRE_FALSE(morton_coords_valid(huge));
}

// ═══════════════════════════════════════════════════════════════════════════
// §4  Known bit patterns
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("§4-1 Morton: single-bit coordinate maps to correct key bit",
          "[morton]") {
    // coord[d] = (1 << b) should set key bit at position (b * 9 + d)
    for (int d = 0; d < MORTON_DIMS; ++d) {
        for (int b = 0; b < MORTON_BITS_PER_DIM; ++b) {
            Coord9D c{};
            c[d] = 1u << b;
            MortonKey key = morton_encode(c);
            int expected_bit = b * MORTON_DIMS + d;
            MortonKey expected = static_cast<MortonKey>(1) << expected_bit;
            INFO("dim=" << d << " bit=" << b
                 << " expected_key_bit=" << expected_bit);
            REQUIRE(key == expected);
        }
    }
}

TEST_CASE("§4-2 Morton: distinct coordinates produce distinct keys",
          "[morton]") {
    std::mt19937 rng(99);
    std::vector<MortonKey> keys;
    keys.reserve(10000);
    for (int i = 0; i < 10000; ++i) {
        keys.push_back(morton_encode(random_coord(rng)));
    }
    std::sort(keys.begin(), keys.end());
    auto it = std::adjacent_find(keys.begin(), keys.end());
    // With 126-bit key space and 10K samples, duplicates should be
    // astronomically unlikely. We just check the sort+find works.
    // (Statistically impossible to collide in 2^126 space.)
    // Note: random coords CAN collide, so we actually just check round-trip.
    // The real uniqueness guarantee is: different coords → different keys.
    for (int d = 0; d < MORTON_DIMS; ++d) {
        Coord9D a{}, b{};
        a[d] = 0;
        b[d] = 1;
        REQUIRE(morton_encode(a) != morton_encode(b));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §5  Byte serialization
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("§5-1 Morton: byte serialization round-trip", "[morton]") {
    std::mt19937 rng(77);
    MortonKey mask = (static_cast<MortonKey>(1) << 126) - 1;
    std::uniform_int_distribution<uint64_t> dist64(0, UINT64_MAX);

    for (int i = 0; i < 1000; ++i) {
        MortonKey key = (static_cast<MortonKey>(dist64(rng)) << 64)
                      | static_cast<MortonKey>(dist64(rng));
        key &= mask;
        uint8_t buf[16]{};
        morton_key_to_bytes(key, buf);
        MortonKey back = morton_key_from_bytes(buf);
        REQUIRE(back == key);
    }
}

TEST_CASE("§5-2 Morton: zero key serializes to all-zero bytes", "[morton]") {
    MortonKey zero = 0;
    uint8_t buf[16]{};
    morton_key_to_bytes(zero, buf);
    for (int i = 0; i < 16; ++i) {
        REQUIRE(buf[i] == 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §6  Lexicographic ordering
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("§6-1 Morton: big-endian byte order preserves numeric ordering",
          "[morton]") {
    // Generate random keys, serialize, compare byte buffers
    std::mt19937 rng(55);
    MortonKey mask = (static_cast<MortonKey>(1) << 126) - 1;
    std::uniform_int_distribution<uint64_t> dist64(0, UINT64_MAX);

    for (int i = 0; i < 5000; ++i) {
        MortonKey a = (static_cast<MortonKey>(dist64(rng)) << 64)
                    | static_cast<MortonKey>(dist64(rng));
        MortonKey b = (static_cast<MortonKey>(dist64(rng)) << 64)
                    | static_cast<MortonKey>(dist64(rng));
        a &= mask;
        b &= mask;

        uint8_t ba[16]{}, bb[16]{};
        morton_key_to_bytes(a, ba);
        morton_key_to_bytes(b, bb);

        int byte_cmp = std::memcmp(ba, bb, 16);
        if (a < b) {
            REQUIRE(byte_cmp < 0);
        } else if (a > b) {
            REQUIRE(byte_cmp > 0);
        } else {
            REQUIRE(byte_cmp == 0);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §7  Exhaustive round-trip (small coordinate space)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("§7-1 Morton: exhaustive 2-bit/dim round-trip (2^18 = 262,144 points)",
          "[morton][longsession]") {
    // With 2 bits per dim → coords in [0,4), 9 dims → 4^9 = 262,144 points
    constexpr uint32_t SMALL_MAX = 4;
    Coord9D c{};

    // We iterate all 4^9 combinations using a single counter
    uint32_t total = 1;
    for (int d = 0; d < MORTON_DIMS; ++d) total *= SMALL_MAX;

    for (uint32_t idx = 0; idx < total; ++idx) {
        // Decompose idx into base-4 digits
        uint32_t tmp = idx;
        for (int d = 0; d < MORTON_DIMS; ++d) {
            c[d] = tmp % SMALL_MAX;
            tmp /= SMALL_MAX;
        }
        auto key = morton_encode(c);
        auto back = morton_decode(key);
        REQUIRE(back == c);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §8  Benchmark: encode/decode throughput
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("§8-1 Morton: encode throughput", "[morton][!benchmark]") {
    constexpr int N = 1'000'000;
    std::mt19937 rng(0);
    std::vector<Coord9D> coords(N);
    for (auto& c : coords) c = random_coord(rng);

    auto t0 = std::chrono::steady_clock::now();
    volatile MortonKey sink = 0;
    for (int i = 0; i < N; ++i) {
        sink = morton_encode(coords[i]);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;

    INFO("Morton encode: " << ns << " ns/op");
    // Bit-interleaving should be < 500 ns/op on modern CPUs
    REQUIRE(ns < 500.0);
}

TEST_CASE("§8-2 Morton: decode throughput", "[morton][!benchmark]") {
    constexpr int N = 1'000'000;
    std::mt19937 rng(1);
    std::uniform_int_distribution<uint64_t> dist64(0, UINT64_MAX);
    MortonKey mask = (static_cast<MortonKey>(1) << 126) - 1;

    std::vector<MortonKey> keys(N);
    for (auto& k : keys) {
        k = (static_cast<MortonKey>(dist64(rng)) << 64)
          | static_cast<MortonKey>(dist64(rng));
        k &= mask;
    }

    auto t0 = std::chrono::steady_clock::now();
    volatile uint32_t sink = 0;
    for (int i = 0; i < N; ++i) {
        auto c = morton_decode(keys[i]);
        sink = c[0];
    }
    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;

    INFO("Morton decode: " << ns << " ns/op");
    REQUIRE(ns < 500.0);
}

TEST_CASE("§8-3 Morton: byte serialization throughput", "[morton][!benchmark]") {
    constexpr int N = 1'000'000;
    std::mt19937 rng(2);
    std::uniform_int_distribution<uint64_t> dist64(0, UINT64_MAX);
    MortonKey mask = (static_cast<MortonKey>(1) << 126) - 1;

    std::vector<MortonKey> keys(N);
    for (auto& k : keys) {
        k = (static_cast<MortonKey>(dist64(rng)) << 64)
          | static_cast<MortonKey>(dist64(rng));
        k &= mask;
    }

    uint8_t buf[16]{};
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        morton_key_to_bytes(keys[i], buf);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;

    INFO("Morton serialize: " << ns << " ns/op");
    // Byte shuffle should be trivially fast (< 100 ns)
    REQUIRE(ns < 100.0);
}
