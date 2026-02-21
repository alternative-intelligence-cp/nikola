/**
 * @file security/vm_image_manager.hpp
 * @brief Gap 7.1 — VMImageManager
 *
 * Manages creation verification and lifecycle of the KVM sandbox base image.
 *
 * Verification uses a pure C++ SHA-256 implementation (no OpenSSL dependency)
 * to hash gold.qcow2 and compare against the expected digest stored in a
 * read-only file (default: /boot/nikola_checksums.txt).
 *
 * KVM-specific operations (create_snapshot, launch_vm, destroy_vm) are
 * compiled only when NIKOLA_ENABLE_KVM is defined. The pure SHA-256 path
 * is always available for testing.
 */
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nikola::security {

// ============================================================================
// Pure C++ SHA-256 (RFC 6234 / FIPS 180-4)
// ============================================================================

namespace detail {

inline constexpr std::array<uint32_t, 64> SHA256_K = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

struct SHA256Ctx {
    std::array<uint32_t, 8> state{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    uint64_t bitcount{0};
    std::array<uint8_t, 64> buf{};
    size_t buflen{0};

    void transform(const uint8_t* block) {
        std::array<uint32_t, 64> W{};
        for (int i = 0; i < 16; ++i)
            W[i] = (static_cast<uint32_t>(block[i*4])   << 24) |
                   (static_cast<uint32_t>(block[i*4+1]) << 16) |
                   (static_cast<uint32_t>(block[i*4+2]) <<  8) |
                   (static_cast<uint32_t>(block[i*4+3]));
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr32(W[i-15], 7) ^ rotr32(W[i-15], 18) ^ (W[i-15] >> 3);
            uint32_t s1 = rotr32(W[i-2],  17) ^ rotr32(W[i-2],  19) ^ (W[i-2]  >> 10);
            W[i] = W[i-16] + s0 + W[i-7] + s1;
        }
        auto [a,b,c,d,e,f,g,h] = state;
        for (int i = 0; i < 64; ++i) {
            uint32_t S1  = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
            uint32_t ch  = (e & f) ^ (~e & g);
            uint32_t t1  = h + S1 + ch + SHA256_K[i] + W[i];
            uint32_t S0  = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2  = S0 + maj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
        state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    }

    void update(const uint8_t* data, size_t len) {
        bitcount += static_cast<uint64_t>(len) * 8;
        while (len > 0) {
            size_t room = 64 - buflen;
            size_t take = (len < room) ? len : room;
            std::memcpy(buf.data() + buflen, data, take);
            buflen += take;
            data   += take;
            len    -= take;
            if (buflen == 64) { transform(buf.data()); buflen = 0; }
        }
    }

    std::array<uint8_t, 32> finalise() {
        buf[buflen++] = 0x80;
        if (buflen > 56) {
            while (buflen < 64) buf[buflen++] = 0;
            transform(buf.data()); buflen = 0;
        }
        while (buflen < 56) buf[buflen++] = 0;
        for (int i = 7; i >= 0; --i)
            buf[buflen++] = static_cast<uint8_t>((bitcount >> (i*8)) & 0xFF);
        transform(buf.data());

        std::array<uint8_t, 32> digest{};
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j)
                digest[i*4+j] = static_cast<uint8_t>((state[i] >> (24 - j*8)) & 0xFF);
        return digest;
    }
};

} // namespace detail

// ============================================================================
// Gap 7.1 — VMImageManager
// ============================================================================

inline constexpr char GOLD_IMAGE_DEFAULT_PATH[]    = "/var/lib/nikola/gold.qcow2";
inline constexpr char CHECKSUMS_DEFAULT_PATH[]     = "/boot/nikola_checksums.txt";
inline constexpr char ALPINE_BASE_VERSION[]        = "3.19";
inline constexpr uint64_t VM_DISK_SIZE_MB          = 512;
inline constexpr uint64_t VM_MEMORY_MB             = 512;

using SHA256Digest = std::array<uint8_t, 32>;

/** Convert a SHA256 digest to lowercase hex string. */
inline std::string digest_to_hex(const SHA256Digest& d) {
    std::ostringstream oss;
    for (uint8_t b : d) oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
    return oss.str();
}

/** Parse a 64-char hex string into a SHA256Digest. Returns false on error. */
inline bool hex_to_digest(const std::string& hex, SHA256Digest& out) {
    if (hex.size() != 64) return false;
    for (size_t i = 0; i < 32; ++i) {
        out[i] = static_cast<uint8_t>(std::stoul(hex.substr(i*2, 2), nullptr, 16));
    }
    return true;
}

/**
 * Compute SHA-256 of a file using the pure C++ implementation.
 * Returns an empty digest and sets ok=false if the file cannot be read.
 */
inline SHA256Digest sha256_file(const std::string& path, bool& ok) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { ok = false; return {}; }
    ok = true;
    detail::SHA256Ctx ctx{};
    char buf[8192];
    while (f.read(buf, sizeof(buf)) || f.gcount() > 0) {
        ctx.update(reinterpret_cast<const uint8_t*>(buf),
                   static_cast<size_t>(f.gcount()));
    }
    return ctx.finalise();
}

/** Compute SHA-256 of raw bytes in memory. */
inline SHA256Digest sha256_bytes(const uint8_t* data, size_t len) {
    detail::SHA256Ctx ctx{};
    ctx.update(data, len);
    return ctx.finalise();
}

inline SHA256Digest sha256_string(const std::string& s) {
    return sha256_bytes(reinterpret_cast<const uint8_t*>(s.data()), s.size());
}

/**
 * Result of a verification operation.
 */
struct ImageVerifyResult {
    bool     ok{false};
    std::string actual_hex;
    std::string expected_hex;
    std::string error_msg;
};

/**
 * Manages the Alpine-based gold.qcow2 sandbox base image.
 *
 * Instantiate once on startup and call verify_integrity() before
 * launching any execution VM.
 */
class VMImageManager {
public:
    struct Config {
        std::string gold_image_path   = GOLD_IMAGE_DEFAULT_PATH;
        std::string checksums_path    = CHECKSUMS_DEFAULT_PATH;
        bool        strict_mode       = true;  ///< Throw on verification failure
    };

    VMImageManager() : cfg_{} {}
    explicit VMImageManager(Config cfg) : cfg_(std::move(cfg)) {}

    /**
     * Set the expected digest directly (e.g. from build-time constant).
     * Overrides reading from the checksums file.
     */
    void set_expected_digest(const SHA256Digest& d) {
        expected_       = d;
        expected_loaded_ = true;
    }

    void set_expected_hex(const std::string& hex) {
        SHA256Digest d{};
        if (!hex_to_digest(hex, d))
            throw std::invalid_argument("VMImageManager: invalid hex digest");
        set_expected_digest(d);
    }

    /**
     * Verify the gold image hash against the stored checksum.
     * If checksums_path doesn't exist the check is skipped (returns ok=true,
     * skipped=true) unless strict_mode is enabled.
     */
    ImageVerifyResult verify_integrity() {
        ImageVerifyResult result;

        // Load expected hash if not yet set
        if (!expected_loaded_) {
            load_expected_from_file(result);
            if (!result.error_msg.empty() && cfg_.strict_mode)
                return result;
        }

        // Hash the image file
        bool file_ok = false;
        const SHA256Digest actual = sha256_file(cfg_.gold_image_path, file_ok);

        if (!file_ok) {
            result.ok        = false;
            result.error_msg = "Cannot read gold image: " + cfg_.gold_image_path;
            return result;
        }

        result.actual_hex   = digest_to_hex(actual);
        result.expected_hex = expected_loaded_ ? digest_to_hex(expected_) : "";

        if (expected_loaded_) {
            result.ok = (actual == expected_);
            if (!result.ok)
                result.error_msg = "SHA-256 mismatch: image may be corrupted or tampered";
        } else {
            // No expected hash available — just record the actual hash
            result.ok = !cfg_.strict_mode; // strict: fail; lenient: pass
        }

        return result;
    }

    /**
     * Compute and return the SHA-256 of the gold image (no comparison).
     * Returns empty string if the file cannot be read.
     */
    std::string compute_image_hex() {
        bool ok = false;
        auto d = sha256_file(cfg_.gold_image_path, ok);
        return ok ? digest_to_hex(d) : "";
    }

    const Config& config() const { return cfg_; }

#ifdef NIKOLA_ENABLE_KVM
    /**
     * Create a copy-on-write snapshot of the gold image for a new VM instance.
     * Requires qemu-img to be installed on the host.
     */
    bool create_snapshot(const std::string& snapshot_path) {
        const std::string cmd =
            "qemu-img create -f qcow2 -b " + cfg_.gold_image_path +
            " -F qcow2 " + snapshot_path + " 2>/dev/null";
        return std::system(cmd.c_str()) == 0; // NOLINT
    }

    /** Delete a snapshot created with create_snapshot(). */
    bool destroy_snapshot(const std::string& snapshot_path) {
        return std::filesystem::remove(snapshot_path);
    }
#endif // NIKOLA_ENABLE_KVM

private:
    Config       cfg_;
    SHA256Digest expected_{};
    bool         expected_loaded_{false};

    void load_expected_from_file(ImageVerifyResult& result) {
        std::ifstream f(cfg_.checksums_path);
        if (!f) {
            result.error_msg = "Cannot read checksums file: " + cfg_.checksums_path;
            return;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("gold.qcow2") == std::string::npos) continue;
            // Format: "<hex64>  gold.qcow2" or "gold.qcow2: <hex64>"
            for (size_t i = 0; i + 64 <= line.size(); ++i) {
                bool all_hex = true;
                for (size_t j = i; j < i + 64; ++j) {
                    char c = line[j];
                    if (!((c>='0'&&c<='9')||(c>='a'&&c<='f')||(c>='A'&&c<='F')))
                    { all_hex = false; break; }
                }
                if (all_hex) {
                    std::string hex = line.substr(i, 64);
                    // normalise to lowercase
                    for (auto& c : hex) if (c>='A'&&c<='F') c += 32;
                    SHA256Digest d{};
                    if (hex_to_digest(hex, d)) {
                        expected_       = d;
                        expected_loaded_ = true;
                        return;
                    }
                }
            }
        }
        result.error_msg = "gold.qcow2 entry not found in " + cfg_.checksums_path;
    }
};

} // namespace nikola::security
