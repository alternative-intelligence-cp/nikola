/**
 * @file phase95_crc32c_test.cpp
 * @brief Phase 95 — GAP-038: CRC32C hardware intrinsics test suite
 *
 * Tests the crc32c.hpp implementation against:
 *   - RFC 3720 §B.4 reference vectors (iSCSI / SCTP standard)
 *   - HW/SW equivalence (when SSE4.2 is available)
 *   - Incremental (chained) computation correctness
 *   - Alignment robustness (various start offsets)
 *   - Corruption detection
 *   - Nikola usage patterns (DMC persistence, partition migration)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <nikola/system/crc32c.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace nikola::system;

// ============================================================================
// Test helpers
// ============================================================================

namespace {

// Build a buffer of 'n' bytes all set to 'value'.
std::vector<uint8_t> filled(size_t n, uint8_t value) {
    return std::vector<uint8_t>(n, value);
}

// Build a buffer 0x00, 0x01, ..., 0xFF, 0x00, ... (wrapping at 0x100) of length 'n'.
std::vector<uint8_t> sequential(size_t n) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(i & 0xFFU);
    return v;
}

} // namespace

// ============================================================================
// RFC 3720 §B.4 reference vectors (auto-select HW/SW via crc32c)
// ============================================================================

TEST_CASE("crc32c - known-good reference vectors", "[crc32c][rfc3720]") {

    SECTION("Empty input yields 0x00000000") {
        CHECK(crc32c(nullptr, 0) == 0x00000000U);
        CHECK(crc32c("", 0)      == 0x00000000U);
        CHECK(crc32c(nullptr, 0) == CRC32C_EMPTY);
    }

    SECTION("32 bytes of 0x00 yields 0x8A9136AA") {
        auto buf = filled(32, 0x00);
        CHECK(crc32c(buf.data(), 32) == 0x8A9136AAU);
        CHECK(crc32c(buf.data(), 32) == CRC32C_32ZEROS);
    }

    SECTION("32 bytes of 0xFF yields 0x62A8AB43") {
        auto buf = filled(32, 0xFF);
        CHECK(crc32c(buf.data(), 32) == 0x62A8AB43U);
        CHECK(crc32c(buf.data(), 32) == CRC32C_32FF);
    }

    SECTION("32-byte sequential 0x00..0x1F yields 0x46DD794E") {
        uint8_t seq[32];
        for (int i = 0; i < 32; ++i) seq[i] = static_cast<uint8_t>(i);
        CHECK(crc32c(seq, 32) == 0x46DD794EU);
        CHECK(crc32c(seq, 32) == CRC32C_32SEQ);
    }

    SECTION("\"123456789\" yields 0xE3069283 (CRC catalog / universal standard)") {
        const char* data = "123456789";
        CHECK(crc32c(data, 9) == 0xE3069283U);
        CHECK(crc32c(data, 9) == CRC32C_DIGITS);
    }
}

// ============================================================================
// Software path explicit verification
// ============================================================================

TEST_CASE("crc32c_sw - known-good vectors explicit SW path", "[crc32c][sw]") {

    SECTION("Empty input") {
        CHECK(crc32c_sw(nullptr, 0) == 0x00000000U);
    }

    SECTION("32 zeros") {
        auto buf = filled(32, 0x00);
        CHECK(crc32c_sw(buf.data(), 32) == 0x8A9136AAU);
    }

    SECTION("32 × 0xFF") {
        auto buf = filled(32, 0xFF);
        CHECK(crc32c_sw(buf.data(), 32) == 0x62A8AB43U);
    }

    SECTION("Sequential 0x00..0x1F (32 bytes)") {
        uint8_t seq[32];
        for (int i = 0; i < 32; ++i) seq[i] = static_cast<uint8_t>(i);
        CHECK(crc32c_sw(seq, 32) == 0x46DD794EU);
    }

    SECTION("\"123456789\"") {
        const char* data = "123456789";
        CHECK(crc32c_sw(data, 9) == 0xE3069283U);
    }
}

// ============================================================================
// Hardware path explicit verification (SSE4.2)
// ============================================================================

TEST_CASE("crc32c_hw - RFC 3720 §B.4 vectors explicit HW path", "[crc32c][hw]") {

#ifdef __SSE4_2__
    SECTION("Empty input") {
        CHECK(crc32c_hw(nullptr, 0) == 0x00000000U);
    }

    SECTION("32 zeros") {
        auto buf = filled(32, 0x00);
        CHECK(crc32c_hw(buf.data(), 32) == 0x8A9136AAU);
    }

    SECTION("32 × 0xFF") {
        auto buf = filled(32, 0xFF);
        CHECK(crc32c_hw(buf.data(), 32) == 0x62A8AB43U);
    }

    SECTION("Sequential 0x00..0x1F (32 bytes)") {
        uint8_t seq[32];
        for (int i = 0; i < 32; ++i) seq[i] = static_cast<uint8_t>(i);
        CHECK(crc32c_hw(seq, 32) == 0x46DD794EU);
    }

    SECTION("\"123456789\"") {
        const char* data = "123456789";
        CHECK(crc32c_hw(data, 9) == 0xE3069283U);
    }

    SECTION("CRC32C_HW_AVAILABLE flag is true") {
        CHECK(CRC32C_HW_AVAILABLE == true);
    }
#else
    SECTION("CRC32C_HW_AVAILABLE flag is false when no SSE4.2") {
        CHECK(CRC32C_HW_AVAILABLE == false);
    }
    SUCCEED("SSE4.2 not available; HW path tests skipped");
#endif
}

// ============================================================================
// HW / SW equivalence
// ============================================================================

TEST_CASE("crc32c HW == SW equivalence", "[crc32c][equivalence]") {

#ifndef __SSE4_2__
    SUCCEED("SSE4.2 not compiled; equivalence test skipped");
#else
    SECTION("All lengths 0..64") {
        auto large = sequential(64);
        for (size_t len = 0; len <= 64; ++len) {
            uint32_t hw = crc32c_hw(large.data(), len);
            uint32_t sw = crc32c_sw(large.data(), len);
            CHECK(hw == sw);
        }
    }

    SECTION("All-zero buffers 1..64 bytes") {
        auto zeros = filled(64, 0x00);
        for (size_t len = 1; len <= 64; ++len) {
            CHECK(crc32c_hw(zeros.data(), len) == crc32c_sw(zeros.data(), len));
        }
    }

    SECTION("All-0xFF buffers 1..64 bytes") {
        auto ff = filled(64, 0xFF);
        for (size_t len = 1; len <= 64; ++len) {
            CHECK(crc32c_hw(ff.data(), len) == crc32c_sw(ff.data(), len));
        }
    }

    SECTION("Large buffer (4096 bytes)") {
        auto large = sequential(4096);
        CHECK(crc32c_hw(large.data(), 4096) == crc32c_sw(large.data(), 4096));
    }

    SECTION("Non-power-of-two lengths") {
        auto buf = sequential(200);
        for (size_t len : {1U, 3U, 5U, 7U, 9U, 11U, 13U, 15U, 17U, 100U, 199U}) {
            CHECK(crc32c_hw(buf.data(), len) == crc32c_sw(buf.data(), len));
        }
    }
#endif
}

// ============================================================================
// Incremental (chained) computation
// ============================================================================

TEST_CASE("crc32c incremental chaining (split buffer == whole buffer)", "[crc32c][incremental]") {

    auto buf = sequential(64);

    SECTION("Split at byte 1") {
        uint32_t chained = crc32c(buf.data(), 1);
        chained = crc32c(buf.data() + 1, 63, chained);
        CHECK(chained == crc32c(buf.data(), 64));
    }

    SECTION("Split at byte 8") {
        uint32_t chained = crc32c(buf.data(), 8);
        chained = crc32c(buf.data() + 8, 56, chained);
        CHECK(chained == crc32c(buf.data(), 64));
    }

    SECTION("Three-way split") {
        uint32_t crc = crc32c(buf.data(),      16);
        crc          = crc32c(buf.data() + 16, 16, crc);
        crc          = crc32c(buf.data() + 32, 32, crc);
        CHECK(crc == crc32c(buf.data(), 64));
    }

    SECTION("Single-byte chain across 32 bytes") {
        // Feed buf one byte at a time
        uint32_t crc = 0;
        for (size_t i = 0; i < 32; ++i)
            crc = crc32c(buf.data() + i, 1, crc);
        CHECK(crc == crc32c(buf.data(), 32));
    }

    SECTION("SW explicit three-way split") {
        uint32_t crc = crc32c_sw(buf.data(),      16);
        crc          = crc32c_sw(buf.data() + 16, 16, crc);
        crc          = crc32c_sw(buf.data() + 32, 32, crc);
        CHECK(crc == crc32c_sw(buf.data(), 64));
    }
}

// ============================================================================
// Alignment robustness
// ============================================================================

TEST_CASE("crc32c alignment robustness", "[crc32c][alignment]") {

    // Allocate an oversized buffer with 8-byte head room so we can test all
    // byte offsets 0..7.
    std::vector<uint8_t> storage(64 + 8);
    for (size_t i = 0; i < storage.size(); ++i)
        storage[i] = static_cast<uint8_t>((i * 37 + 13) & 0xFF);  // pseudo-random

    SECTION("Offsets 0..7 give same result for equal content") {
        // Compare CRC32C of 32 bytes of identical data regardless of alignment.
        std::vector<uint8_t> ref_data(32);
        for (size_t i = 0; i < 32; ++i) ref_data[i] = static_cast<uint8_t>(i);
        uint32_t expected = crc32c(ref_data.data(), 32);

        // Copy into storage at each offset and recompute.
        for (size_t offset = 0; offset < 8; ++offset) {
            std::memcpy(storage.data() + offset, ref_data.data(), 32);
            CHECK(crc32c(storage.data() + offset, 32) == expected);
        }
    }

    SECTION("Buffer size 1..9 at non-zero offset") {
        for (size_t offset = 1; offset <= 3; ++offset) {
            for (size_t len = 1; len <= 9; ++len) {
#ifdef __SSE4_2__
                CHECK(crc32c_hw(storage.data() + offset, len) ==
                      crc32c_sw(storage.data() + offset, len));
#else
                // SW-only: just verify it produces a consistent result.
                uint32_t a = crc32c_sw(storage.data() + offset, len);
                uint32_t b = crc32c_sw(storage.data() + offset, len);
                CHECK(a == b);
#endif
            }
        }
    }
}

// ============================================================================
// Zero-length edge cases
// ============================================================================

TEST_CASE("crc32c zero-length edge cases", "[crc32c][zero_len]") {

    SECTION("crc32c with len=0 is identity (seed passes through)") {
        // CRC of empty input with seed X should return X (identity property).
        CHECK(crc32c_sw("unused", 0, 0x12345678U) == 0x12345678U);
#ifdef __SSE4_2__
        CHECK(crc32c_hw("unused", 0, 0x12345678U) == 0x12345678U);
#endif
        CHECK(crc32c("unused", 0, 0xDEADBEEFU) == 0xDEADBEEFU);
    }

    SECTION("crc32c of empty twice == crc32c of empty once") {
        uint32_t first  = crc32c(nullptr, 0);
        uint32_t second = crc32c(nullptr, 0, first);
        CHECK(first == 0U);
        CHECK(second == 0U);
    }
}

// ============================================================================
// Corruption detection
// ============================================================================

TEST_CASE("crc32c detects data corruption", "[crc32c][corruption]") {

    auto buf = sequential(64);
    uint32_t original = crc32c(buf.data(), 64);

    SECTION("Flipping bit 0 of first byte changes CRC") {
        buf[0] ^= 0x01;
        CHECK(crc32c(buf.data(), 64) != original);
        buf[0] ^= 0x01;  // restore
    }

    SECTION("Flipping last byte changes CRC") {
        buf[63] ^= 0xFF;
        CHECK(crc32c(buf.data(), 64) != original);
        buf[63] ^= 0xFF;
    }

    SECTION("Zeroing middle byte changes CRC") {
        uint8_t saved = buf[32];
        buf[32] = 0x00;
        if (saved != 0x00)
            CHECK(crc32c(buf.data(), 64) != original);
        buf[32] = saved;
    }

    SECTION("Restored buffer gives original CRC") {
        CHECK(crc32c(buf.data(), 64) == original);
    }

    SECTION("Swapping two bytes changes CRC (unless they're identical)") {
        uint8_t a = buf[10], b = buf[20];
        if (a != b) {
            std::swap(buf[10], buf[20]);
            CHECK(crc32c(buf.data(), 64) != original);
            std::swap(buf[10], buf[20]);
        }
        CHECK(crc32c(buf.data(), 64) == original);
    }
}

// ============================================================================
// Single-byte values
// ============================================================================

TEST_CASE("crc32c single-byte values are distinct from zero-length", "[crc32c][single_byte]") {

    uint32_t empty_crc = crc32c(nullptr, 0);

    // Every non-zero byte should produce a CRC different from the empty CRC.
    for (int b = 1; b <= 255; ++b) {
        uint8_t byte = static_cast<uint8_t>(b);
        CHECK(crc32c(&byte, 1) != empty_crc);
    }
}

// ============================================================================
// All 256 single-byte inputs produce distinct CRCs
// ============================================================================

TEST_CASE("crc32c all 256 single-byte inputs are distinct", "[crc32c][collision]") {

    std::set<uint32_t> seen;
    for (int b = 0; b < 256; ++b) {
        uint8_t byte = static_cast<uint8_t>(b);
        seen.insert(crc32c(&byte, 1));
    }
    // All 256 CRC values are unique (no collisions over the 1-byte range).
    CHECK(seen.size() == 256U);
}

// ============================================================================
// Compile-time constant validation
// ============================================================================

TEST_CASE("crc32c compile-time constants match runtime computation", "[crc32c][constants]") {

    SECTION("CRC32C_EMPTY") {
        CHECK(crc32c_sw(nullptr, 0) == CRC32C_EMPTY);
    }
    SECTION("CRC32C_DIGITS") {
        CHECK(crc32c_sw("123456789", 9) == CRC32C_DIGITS);
    }
    SECTION("CRC32C_32ZEROS") {
        auto z = filled(32, 0x00);
        CHECK(crc32c_sw(z.data(), 32) == CRC32C_32ZEROS);
    }
    SECTION("CRC32C_32FF") {
        auto ff = filled(32, 0xFF);
        CHECK(crc32c_sw(ff.data(), 32) == CRC32C_32FF);
    }
    SECTION("CRC32C_32SEQ") {
        uint8_t seq[32];
        for (int i = 0; i < 32; ++i) seq[i] = static_cast<uint8_t>(i);
        CHECK(crc32c_sw(seq, 32) == CRC32C_32SEQ);
    }
}

// ============================================================================
// Lookup table sanity
// ============================================================================

TEST_CASE("crc32c Sarwate table properties", "[crc32c][table]") {

    SECTION("Table entry 0 is 0 (byte value 0x00 has CRC 0)") {
        CHECK(detail::CRC32C_TABLE[0] == 0U);
    }

    SECTION("Table entry 1 is crc32c of single byte 0x01") {
        // TABLE[1] = CRC32C of the byte 0x01 with initial state 0xFFFFFFFF,
        // which processes bits LSB-first through the Castagnoli polynomial.
        // Correct value: 0xF26B8303
        CHECK(detail::CRC32C_TABLE[1] == 0xF26B8303U);
    }

    SECTION("Table has 256 distinct entries would be overly strong — but all nonzero for nonzero idx") {
        // Entries don't have to be unique but entry[i] for i>0 should be nonzero
        // because CRC32C is a proper degree-32 polynomial.
        bool any_zero = false;
        for (int i = 1; i < 256; ++i)
            if (detail::CRC32C_TABLE[i] == 0U) { any_zero = true; break; }
        CHECK(!any_zero);
    }
}

// ============================================================================
// Nikola DMC persistence use-case
// ============================================================================

TEST_CASE("crc32c DMC persistence use-case: page checksum", "[crc32c][persistence]") {

    // Simulate a DMC WAL page: header + compressed payload.
    struct PageHeader {
        uint64_t sequence_number{42};
        uint64_t timestamp_us{1234567890};
        uint32_t payload_size{128};
        uint32_t checksum{0};   // filled in after computation
    };

    std::vector<uint8_t> payload = sequential(128);

    SECTION("Checksum covers payload only") {
        uint32_t cs = crc32c(payload.data(), payload.size());
        CHECK(cs != 0U);  // non-trivial checksum
    }

    SECTION("Checksum covers header then payload (chained)") {
        PageHeader hdr;
        hdr.checksum = 0;

        uint32_t cs = crc32c(std::addressof(hdr), sizeof(hdr));
        cs           = crc32c(payload.data(), 128, cs);
        CHECK(cs != 0U);

        // Same result as computing over the whole flat buffer.
        std::vector<uint8_t> flat(sizeof(hdr) + 128);
        std::memcpy(flat.data(),               std::addressof(hdr), sizeof(hdr));
        std::memcpy(flat.data() + sizeof(hdr), payload.data(),      128);
        CHECK(crc32c(flat.data(), flat.size()) == cs);
    }

    SECTION("Payload corruption changes checksum") {
        uint32_t good = crc32c(payload.data(), 128);
        payload[64] ^= 0x01;
        uint32_t bad = crc32c(payload.data(), 128);
        CHECK(good != bad);
    }
}

// ============================================================================
// Nikola partition table protocol use-case
// ============================================================================

TEST_CASE("crc32c partition table migration integrity", "[crc32c][partition]") {

    // Simulate a batch of 10 migrated SoA nodes (232 bytes each per §GAP-019).
    constexpr size_t NODE_SIZE   = 232;
    constexpr size_t BATCH_COUNT = 10;
    constexpr size_t BATCH_SIZE  = NODE_SIZE * BATCH_COUNT;

    auto batch = sequential(BATCH_SIZE);
    uint32_t migration_crc = crc32c(batch.data(), BATCH_SIZE);

    SECTION("Integrity check passes on clean transfer") {
        // Receiver re-computes — should match.
        uint32_t receiver_crc = crc32c(batch.data(), BATCH_SIZE);
        CHECK(receiver_crc == migration_crc);
    }

    SECTION("Single-bit corruption detected in middle of batch") {
        batch[NODE_SIZE * 5 + 100] ^= 0x80;
        CHECK(crc32c(batch.data(), BATCH_SIZE) != migration_crc);
    }

    SECTION("Per-node incremental matches whole-batch CRC") {
        uint32_t crc = 0;
        for (size_t i = 0; i < BATCH_COUNT; ++i)
            crc = crc32c(batch.data() + i * NODE_SIZE, NODE_SIZE, crc);
        // We ^'d bit in previous section — recompute on fresh sequential batch.
        auto clean = sequential(BATCH_SIZE);
        uint32_t expected = crc32c(clean.data(), BATCH_SIZE);
        uint32_t incremental = 0;
        for (size_t i = 0; i < BATCH_COUNT; ++i)
            incremental = crc32c(clean.data() + i * NODE_SIZE, NODE_SIZE, incremental);
        CHECK(incremental == expected);
    }
}

// ============================================================================
// crc32c_of<T> typed helper
// ============================================================================

TEST_CASE("crc32c_of<T> typed overload", "[crc32c][typed]") {

    SECTION("uint32_t value matches raw pointer call") {
        uint32_t val = 0xDEADBEEFU;
        CHECK(crc32c_of(val) == crc32c(&val, sizeof(val)));
    }

    SECTION("Chaining with typed helper") {
        uint64_t a = 0x0102030405060708ULL;
        uint64_t b = 0x090A0B0C0D0E0F10ULL;
        uint32_t chained = crc32c_of(b, crc32c_of(a));
        uint8_t flat[16];
        std::memcpy(flat,     &a, 8);
        std::memcpy(flat + 8, &b, 8);
        CHECK(chained == crc32c(flat, 16));
    }
}

// ============================================================================
// Performance: hardware path >= 2× faster than software on 1 MB
// ============================================================================

TEST_CASE("crc32c hardware path performance >= 2x software (1 MB)", "[crc32c][performance]") {

#ifndef __SSE4_2__
    SUCCEED("SSE4.2 not compiled; performance test skipped");
#else
    constexpr size_t BUF_SIZE = 1U << 20;  // 1 MB
    std::vector<uint8_t> buf(BUF_SIZE);
    for (size_t i = 0; i < BUF_SIZE; ++i) buf[i] = static_cast<uint8_t>(i);

    // Warm-up
    volatile uint32_t dummy = crc32c_hw(buf.data(), BUF_SIZE);
    dummy = crc32c_sw(buf.data(), BUF_SIZE);
    (void)dummy;

    auto t0 = std::chrono::high_resolution_clock::now();
    volatile uint32_t hw_result = crc32c_hw(buf.data(), BUF_SIZE);
    auto t1 = std::chrono::high_resolution_clock::now();
    volatile uint32_t sw_result = crc32c_sw(buf.data(), BUF_SIZE);
    auto t2 = std::chrono::high_resolution_clock::now();

    CHECK(hw_result == sw_result);  // Correctness first

    double hw_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double sw_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    double ratio = sw_us / hw_us;

    // Spec §6: "10× speedup" — require at minimum 2× to pass in debug builds.
    INFO("HW: " << hw_us << " µs, SW: " << sw_us << " µs, ratio: " << ratio);
    CHECK(ratio >= 2.0);
#endif
}
