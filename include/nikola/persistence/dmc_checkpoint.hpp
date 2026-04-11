// ============================================================
// include/nikola/persistence/dmc_checkpoint.hpp
// Phase 154 — GAP-6.1  Differential Manifold Checkpointing
// ============================================================
// Implements the .nik binary format for saving and restoring
// Nikola's full cognitive state:
//
//   ┌────────────────────────────┐
//   │  Global Header (64 bytes)  │  magic 0x4E494B4F ("NIKO")
//   ├────────────────────────────┤
//   │  Section: NikolaState      │  37+ bytes (packed scalars + tokens)
//   ├────────────────────────────┤
//   │  Section: TorusGrid        │  Hyper-page blocks (NRLE-compressed)
//   ├────────────────────────────┤
//   │  Section: SSM Weights      │  A, B, C, D, W_delta, W_Bsel
//   ├────────────────────────────┤
//   │  Section: NPT Heads        │  8 × {Q, K, V} wavefunctions
//   ├────────────────────────────┤
//   │  Section: Metric Tensor    │  45 doubles (lower-triangle)
//   ├────────────────────────────┤
//   │  Footer (128 bytes)        │  Merkle root + metadata
//   └────────────────────────────┘
//
// Write-Ahead Log (WAL) provides crash safety.
// Event-driven: 300s periodic + NAP trigger (Gap 6.3).
// ============================================================
#pragma once

#include <nikola/persistence/nrle_codec.hpp>
#include <nikola/system/crc32c.hpp>
#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/foundation/toroidal_grid.hpp>
#include <nikola/cognitive/cognitive_core.hpp>
#include <nikola/cognitive/neuroplastic_transformer.hpp>
#include <nikola/physics/metric_tensor.hpp>
#include <nikola/physics/wave_function.hpp>

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <stdexcept>
#include <algorithm>

namespace nikola::persistence {

// Cross-namespace type aliases for convenience
using autonomy::NikolaState;
using autonomy::ActionType;
using system::crc32c;

// ────────────────────────────────────────────────────────────────────────────
// §1  Format constants
// ────────────────────────────────────────────────────────────────────────────

inline constexpr uint32_t NIK_MAGIC         = 0x4E494B4Fu;  // "NIKO"
inline constexpr uint16_t NIK_VERSION_MAJOR  = 0;
inline constexpr uint16_t NIK_VERSION_MINOR  = 4;
inline constexpr uint8_t  NIK_DIM_ENCODING   = 0x09;  // nonary
inline constexpr uint8_t  NIK_CIPHER_NONE    = 0x00;
inline constexpr uint8_t  NIK_CIPHER_CHACHA  = 0x01;

inline constexpr size_t NIK_HEADER_SIZE  = 64;
inline constexpr size_t NIK_FOOTER_SIZE  = 128;

// Section type tags
inline constexpr uint32_t SEC_NIKOLA_STATE  = 0x4E535441u;  // "NSTA"
inline constexpr uint32_t SEC_TORUS_GRID    = 0x54475244u;  // "TGRD"
inline constexpr uint32_t SEC_SSM_WEIGHTS   = 0x53534D57u;  // "SSMW"
inline constexpr uint32_t SEC_NPT_HEADS     = 0x4E505448u;  // "NPTH"
inline constexpr uint32_t SEC_METRIC_TENSOR = 0x4D455452u;  // "METR"

// Page flags
inline constexpr uint8_t PAGE_DIRTY      = 0x01;
inline constexpr uint8_t PAGE_COMPRESSED = 0x02;
inline constexpr uint8_t PAGE_ENCRYPTED  = 0x04;
inline constexpr uint8_t PAGE_DELETED    = 0x08;

// WAL entry types
inline constexpr uint8_t WAL_INSERT = 0x01;
inline constexpr uint8_t WAL_UPDATE = 0x02;
inline constexpr uint8_t WAL_COMMIT = 0x03;

// Checkpoint trigger
inline constexpr float CHECKPOINT_INTERVAL_SEC = 300.f;

// ────────────────────────────────────────────────────────────────────────────
// §2  On-disk structures (packed)
// ────────────────────────────────────────────────────────────────────────────

#pragma pack(push, 1)

struct NikHeader {
    uint32_t magic;            // 0x4E494B4F
    uint16_t version_major;
    uint16_t version_minor;
    uint64_t creation_time;    // Unix timestamp (ms)
    uint64_t last_snap_time;   // Timestamp of this snapshot
    uint8_t  dim_encoding;     // 0x09 (nonary)
    uint8_t  cipher_type;      // 0x00 = none, 0x01 = ChaCha20
    uint8_t  reserved[38];     // Pad to 64 bytes
};
static_assert(sizeof(NikHeader) == NIK_HEADER_SIZE,
              ".nik header must be exactly 64 bytes");

struct SectionHeader {
    uint32_t section_type;     // SEC_* tag
    uint32_t checksum;         // CRC32C of payload
    uint64_t payload_len;      // Byte count of payload after this header
    uint8_t  flags;            // PAGE_* bitmask
    uint8_t  reserved[7];      // Pad to 24 bytes
};
static_assert(sizeof(SectionHeader) == 24,
              "Section header must be exactly 24 bytes");

struct NikFooter {
    uint8_t  merkle_root[32];  // SHA-256 of all section CRCs
    uint64_t total_sections;   // Number of sections in file
    uint64_t total_nodes;      // TorusGrid node count
    uint64_t ssm_hidden_dim;   // SSM H dimension
    uint64_t ssm_output_dim;   // SSM O dimension
    uint64_t npt_num_heads;    // NPT head count
    uint8_t  reserved[56];     // Pad to 128 bytes
};
static_assert(sizeof(NikFooter) == NIK_FOOTER_SIZE,
              ".nik footer must be exactly 128 bytes");

struct WALEntry {
    uint64_t timestamp;        // Unix timestamp (ms)
    uint8_t  entry_type;       // WAL_INSERT / WAL_UPDATE / WAL_COMMIT
    uint32_t payload_size;
    uint32_t checksum;         // CRC32C of payload
};

#pragma pack(pop)

// ────────────────────────────────────────────────────────────────────────────
// §3  Serialisation helpers
// ────────────────────────────────────────────────────────────────────────────

namespace detail {

/// Get current Unix timestamp in milliseconds.
inline uint64_t now_ms() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count());
}

/// Append raw bytes to a buffer.
inline void append_raw(std::vector<uint8_t>& buf,
                       const void* data, size_t len) {
    const auto* p = static_cast<const uint8_t*>(data);
    buf.insert(buf.end(), p, p + len);
}

/// Append a typed value to a buffer.
template<typename T>
void append_value(std::vector<uint8_t>& buf, const T& val) {
    append_raw(buf, &val, sizeof(T));
}

/// Append a float vector to a buffer.
inline void append_float_vec(std::vector<uint8_t>& buf,
                             const std::vector<float>& v) {
    const uint64_t count = v.size();
    append_value(buf, count);
    if (!v.empty()) {
        append_raw(buf, v.data(), v.size() * sizeof(float));
    }
}

/// Read a typed value from a byte buffer at pos, advancing pos.
template<typename T>
T read_value(const uint8_t* data, size_t len, size_t& pos) {
    if (pos + sizeof(T) > len)
        throw std::runtime_error("dmc: truncated read");
    T val;
    std::memcpy(&val, data + pos, sizeof(T));
    pos += sizeof(T);
    return val;
}

/// Read a float vector from buffer (prefixed by uint64_t count).
inline std::vector<float> read_float_vec(const uint8_t* data,
                                         size_t len, size_t& pos) {
    const uint64_t count = read_value<uint64_t>(data, len, pos);
    if (pos + count * sizeof(float) > len)
        throw std::runtime_error("dmc: truncated float vector");
    std::vector<float> v(static_cast<size_t>(count));
    if (count > 0) {
        std::memcpy(v.data(), data + pos, count * sizeof(float));
        pos += count * sizeof(float);
    }
    return v;
}

/// Append a string (uint32_t len + chars, no null).
inline void append_string(std::vector<uint8_t>& buf,
                          const std::string& s) {
    const uint32_t len = static_cast<uint32_t>(s.size());
    append_value(buf, len);
    if (!s.empty()) {
        append_raw(buf, s.data(), s.size());
    }
}

/// Read a string from buffer.
inline std::string read_string(const uint8_t* data,
                               size_t len, size_t& pos) {
    const uint32_t slen = read_value<uint32_t>(data, len, pos);
    if (pos + slen > len)
        throw std::runtime_error("dmc: truncated string");
    std::string s(reinterpret_cast<const char*>(data + pos), slen);
    pos += slen;
    return s;
}

// Simple Merkle root: XOR-chain of section CRC32C values.
// (Full SHA-256 Merkle tree deferred to hardening phase.)
inline void compute_merkle_root(const std::vector<uint32_t>& crcs,
                                uint8_t out[32]) {
    std::memset(out, 0, 32);
    for (size_t i = 0; i < crcs.size(); ++i) {
        const size_t offset = (i * 4) % 32;
        uint32_t existing;
        std::memcpy(&existing, out + offset, 4);
        existing ^= crcs[i];
        std::memcpy(out + offset, &existing, 4);
    }
}

}  // namespace detail

// ────────────────────────────────────────────────────────────────────────────
// §4  Section serialisers
// ────────────────────────────────────────────────────────────────────────────

/// Pack NikolaState into binary payload.
[[nodiscard]] inline std::vector<uint8_t>
pack_nikola_state(const NikolaState& state) {
    std::vector<uint8_t> buf;
    buf.reserve(128);

    detail::append_value(buf, state.time);
    detail::append_value(buf, state.torus_energy);
    detail::append_value(buf, state.dopamine);
    detail::append_value(buf, state.td_error);
    detail::append_value(buf, state.atp);
    detail::append_value(buf, state.boredom);
    detail::append_value(buf, state.entropy);
    detail::append_value(buf, static_cast<uint8_t>(state.last_action));

    // Tokens
    const uint32_t num_tokens = static_cast<uint32_t>(state.tokens.size());
    detail::append_value(buf, num_tokens);
    for (const auto& tok : state.tokens) {
        detail::append_string(buf, tok);
    }

    return buf;
}

/// Unpack NikolaState from binary payload.
inline void unpack_nikola_state(const uint8_t* data, size_t len,
                                NikolaState& state) {
    size_t pos = 0;
    state.time         = detail::read_value<float>(data, len, pos);
    state.torus_energy = detail::read_value<float>(data, len, pos);
    state.dopamine     = detail::read_value<float>(data, len, pos);
    state.td_error     = detail::read_value<float>(data, len, pos);
    state.atp          = detail::read_value<float>(data, len, pos);
    state.boredom      = detail::read_value<float>(data, len, pos);
    state.entropy      = detail::read_value<float>(data, len, pos);
    state.last_action  = static_cast<ActionType>(
        detail::read_value<uint8_t>(data, len, pos));

    const uint32_t num_tokens = detail::read_value<uint32_t>(data, len, pos);
    state.tokens.clear();
    state.tokens.reserve(num_tokens);
    for (uint32_t i = 0; i < num_tokens; ++i) {
        state.tokens.push_back(detail::read_string(data, len, pos));
    }
}

/// Pack TorusGrid SoA arrays.
/// If compress=true, uses NRLE on sparse fields.
[[nodiscard]] inline std::vector<uint8_t>
pack_torus_grid(const foundation::TorusGrid& grid, bool compress = true) {
    const size_t N = grid.num_active_nodes();
    std::vector<uint8_t> buf;
    buf.reserve(N * 6 * sizeof(float) + 64);

    const uint64_t node_count = N;
    detail::append_value(buf, node_count);

    // Pack 6 SoA arrays: psi_real, psi_imag, vel_real, vel_imag,
    //                     resonance, state_field
    const float* arrays[] = {
        grid.psi_real(), grid.psi_imag(),
        grid.vel_real(), grid.vel_imag(),
        grid.resonance(), grid.state_field()
    };

    for (int a = 0; a < 6; ++a) {
        if (compress) {
            auto compressed = nrle_compress_floats(arrays[a], N);
            detail::append_value(buf, compressed.scale);
            const uint64_t clen = compressed.data.size();
            detail::append_value(buf, clen);
            detail::append_raw(buf, compressed.data.data(), clen);
        } else {
            // Raw float storage
            const float neg_scale = -1.f;  // sentinel: uncompressed
            detail::append_value(buf, neg_scale);
            const uint64_t raw_len = N * sizeof(float);
            detail::append_value(buf, raw_len);
            detail::append_raw(buf, arrays[a], raw_len);
        }
    }

    return buf;
}

/// Unpack TorusGrid SoA arrays from payload. Populates existing grid nodes.
inline void unpack_torus_grid(const uint8_t* data, size_t len,
                              foundation::TorusGrid& grid) {
    size_t pos = 0;
    const uint64_t node_count = detail::read_value<uint64_t>(data, len, pos);
    const size_t N = static_cast<size_t>(node_count);

    if (grid.num_active_nodes() != N) {
        throw std::runtime_error(
            "dmc: TorusGrid node count mismatch: expected " +
            std::to_string(grid.num_active_nodes()) +
            " got " + std::to_string(N));
    }

    // Unpack 6 SoA arrays in same order
    float* arrays[] = {
        grid.psi_real(), grid.psi_imag(),
        grid.vel_real(), grid.vel_imag(),
        grid.resonance(), grid.state_field()
    };

    for (int a = 0; a < 6; ++a) {
        const float scale = detail::read_value<float>(data, len, pos);
        const uint64_t clen = detail::read_value<uint64_t>(data, len, pos);

        if (scale < 0.f) {
            // Uncompressed raw floats
            if (pos + clen > len)
                throw std::runtime_error("dmc: truncated torus raw data");
            std::memcpy(arrays[a], data + pos, clen);
            pos += static_cast<size_t>(clen);
        } else {
            // NRLE-compressed
            if (pos + clen > len)
                throw std::runtime_error("dmc: truncated torus NRLE data");
            NrleCompressedFloat cf;
            cf.scale = scale;
            cf.data.assign(data + pos, data + pos + clen);
            cf.original_count = N;
            pos += static_cast<size_t>(clen);

            auto decompressed = nrle_decompress_floats(cf);
            if (decompressed.size() < N)
                throw std::runtime_error("dmc: NRLE decompress count mismatch");
            std::memcpy(arrays[a], decompressed.data(), N * sizeof(float));
        }
    }
}

/// Pack SSMLayer weights (A, B, C, D, W_delta, W_Bsel + dimensions).
[[nodiscard]] inline std::vector<uint8_t>
pack_ssm_weights(const cognitive::SSMLayer& ssm) {
    std::vector<uint8_t> buf;
    buf.reserve(ssm.C().size() * sizeof(float) + 256);

    // Dimensions
    const int32_t H = ssm.hidden_dim();
    const int32_t I = ssm.input_dim();
    const int32_t O = ssm.output_dim();
    detail::append_value(buf, H);
    detail::append_value(buf, I);
    detail::append_value(buf, O);

    // Weight matrices
    detail::append_float_vec(buf, ssm.A());
    detail::append_float_vec(buf, ssm.B());
    detail::append_float_vec(buf, ssm.C());
    detail::append_float_vec(buf, ssm.D());
    detail::append_float_vec(buf, ssm.W_delta());
    detail::append_float_vec(buf, ssm.W_Bsel());

    return buf;
}

/// Unpack SSMLayer weights from payload.
inline void unpack_ssm_weights(const uint8_t* data, size_t len,
                               cognitive::SSMLayer& ssm) {
    size_t pos = 0;
    const int32_t H = detail::read_value<int32_t>(data, len, pos);
    const int32_t I = detail::read_value<int32_t>(data, len, pos);
    const int32_t O = detail::read_value<int32_t>(data, len, pos);

    if (H != ssm.hidden_dim() || I != ssm.input_dim() || O != ssm.output_dim())
        throw std::runtime_error("dmc: SSM dimension mismatch");

    ssm.A()       = detail::read_float_vec(data, len, pos);
    ssm.B()       = detail::read_float_vec(data, len, pos);
    ssm.C()       = detail::read_float_vec(data, len, pos);
    ssm.D()       = detail::read_float_vec(data, len, pos);
    ssm.W_delta() = detail::read_float_vec(data, len, pos);
    ssm.W_Bsel()  = detail::read_float_vec(data, len, pos);
}

/// Pack NPT heads (8 heads × {Q, K, V} wavefunctions).
/// Each WF is serialised as its TorusGrid SoA arrays.
[[nodiscard]] inline std::vector<uint8_t>
pack_npt_heads(const cognitive::NeuroplasticTransformer& npt) {
    std::vector<uint8_t> buf;
    buf.reserve(16 * 1024 * 1024);  // ~15 MiB expected

    const uint32_t num_heads = static_cast<uint32_t>(
        cognitive::NPT_NUM_HEADS);
    detail::append_value(buf, num_heads);

    // Access heads via const ref — NPT exposes them via forward()
    // We need to serialize Q, K, V grids from each head.
    // NPT provides head access for serialisation.
    for (uint32_t h = 0; h < num_heads; ++h) {
        const auto& head = npt.head(h);
        auto q_data = pack_torus_grid(head.Q.grid());
        auto k_data = pack_torus_grid(head.K.grid());
        auto v_data = pack_torus_grid(head.V.grid());

        const uint64_t q_len = q_data.size();
        const uint64_t k_len = k_data.size();
        const uint64_t v_len = v_data.size();

        detail::append_value(buf, h);
        detail::append_value(buf, head.frequency);
        detail::append_value(buf, q_len);
        detail::append_raw(buf, q_data.data(), q_len);
        detail::append_value(buf, k_len);
        detail::append_raw(buf, k_data.data(), k_len);
        detail::append_value(buf, v_len);
        detail::append_raw(buf, v_data.data(), v_len);
    }

    return buf;
}

/// Unpack NPT heads from payload.
inline void unpack_npt_heads(const uint8_t* data, size_t len,
                             cognitive::NeuroplasticTransformer& npt) {
    size_t pos = 0;
    const uint32_t num_heads = detail::read_value<uint32_t>(data, len, pos);

    if (num_heads != cognitive::NPT_NUM_HEADS)
        throw std::runtime_error("dmc: NPT head count mismatch");

    for (uint32_t h = 0; h < num_heads; ++h) {
        const uint32_t head_idx = detail::read_value<uint32_t>(data, len, pos);
        const double freq = detail::read_value<double>(data, len, pos);
        (void)freq;  // frequency is derived, not restored

        auto& head = npt.head(head_idx);

        const uint64_t q_len = detail::read_value<uint64_t>(data, len, pos);
        if (pos + q_len > len) throw std::runtime_error("dmc: truncated NPT Q");
        unpack_torus_grid(data + pos, static_cast<size_t>(q_len), head.Q.grid());
        pos += static_cast<size_t>(q_len);

        const uint64_t k_len = detail::read_value<uint64_t>(data, len, pos);
        if (pos + k_len > len) throw std::runtime_error("dmc: truncated NPT K");
        unpack_torus_grid(data + pos, static_cast<size_t>(k_len), head.K.grid());
        pos += static_cast<size_t>(k_len);

        const uint64_t v_len = detail::read_value<uint64_t>(data, len, pos);
        if (pos + v_len > len) throw std::runtime_error("dmc: truncated NPT V");
        unpack_torus_grid(data + pos, static_cast<size_t>(v_len), head.V.grid());
        pos += static_cast<size_t>(v_len);
    }
}

/// Pack MetricTensorCache (45 doubles, lower triangle of 9×9).
[[nodiscard]] inline std::vector<uint8_t>
pack_metric_tensor(const physics::MetricTensorCache& mtc) {
    std::vector<uint8_t> buf;
    buf.reserve(45 * sizeof(double) + 8);

    const uint8_t valid = mtc.is_valid() ? 1 : 0;
    detail::append_value(buf, valid);

    const auto& g = mtc.metric();
    detail::append_raw(buf, g.data(), g.size() * sizeof(double));

    return buf;
}

/// Unpack MetricTensorCache from payload.
inline void unpack_metric_tensor(const uint8_t* data, size_t len,
                                 physics::MetricTensorCache& mtc) {
    size_t pos = 0;
    const uint8_t valid = detail::read_value<uint8_t>(data, len, pos);

    std::array<double, physics::METRIC_LOWER_SIZE> g{};
    if (pos + g.size() * sizeof(double) > len)
        throw std::runtime_error("dmc: truncated metric tensor");
    std::memcpy(g.data(), data + pos, g.size() * sizeof(double));
    pos += g.size() * sizeof(double);

    if (valid) {
        mtc.force_update(g);
    } else {
        mtc.invalidate();
    }
}

// ────────────────────────────────────────────────────────────────────────────
// §5  Write-Ahead Log
// ────────────────────────────────────────────────────────────────────────────

class WriteAheadLog {
public:
    explicit WriteAheadLog(const std::string& path)
        : wal_path_(path)
    {}

    /// Open WAL file for append.
    bool open() {
        wal_stream_.open(wal_path_,
            std::ios::binary | std::ios::app);
        return wal_stream_.is_open();
    }

    /// Write a section payload as a WAL entry.
    void append(uint8_t entry_type, const std::vector<uint8_t>& payload) {
        if (!wal_stream_.is_open()) return;

        WALEntry entry{};
        entry.timestamp    = detail::now_ms();
        entry.entry_type   = entry_type;
        entry.payload_size = static_cast<uint32_t>(payload.size());
        entry.checksum     = crc32c(payload.data(), payload.size());

        wal_stream_.write(reinterpret_cast<const char*>(&entry),
                          sizeof(entry));
        wal_stream_.write(reinterpret_cast<const char*>(payload.data()),
                          payload.size());

        wal_size_ += sizeof(entry) + payload.size();

        // Periodic fsync every 1 MiB
        if (wal_size_ >= WAL_SYNC_INTERVAL) {
            wal_stream_.flush();
            wal_size_ = 0;
        }
    }

    /// Write a commit marker.
    void commit() {
        std::vector<uint8_t> empty;
        append(WAL_COMMIT, empty);
        wal_stream_.flush();
    }

    /// Close and optionally remove the WAL (after successful checkpoint).
    void close_and_remove() {
        if (wal_stream_.is_open()) {
            wal_stream_.close();
        }
        std::error_code ec;
        std::filesystem::remove(wal_path_, ec);
    }

    /// Check if a WAL exists (for recovery).
    [[nodiscard]] bool exists() const {
        return std::filesystem::exists(wal_path_);
    }

    [[nodiscard]] const std::string& path() const noexcept {
        return wal_path_;
    }

private:
    std::string wal_path_;
    std::ofstream wal_stream_;
    size_t wal_size_ = 0;
    static constexpr size_t WAL_SYNC_INTERVAL = 1024 * 1024;
};

// ────────────────────────────────────────────────────────────────────────────
// §6  DmcCheckpoint — full save/restore
// ────────────────────────────────────────────────────────────────────────────

/// Cognitive state snapshot — everything needed to restore Nikola.
struct CognitiveSnapshot {
    NikolaState              state;
    foundation::TorusGrid*   grid       = nullptr;
    cognitive::SSMLayer*     ssm        = nullptr;
    cognitive::NeuroplasticTransformer* npt = nullptr;
    physics::MetricTensorCache*        metric = nullptr;
};

/// Save a full cognitive state to a .nik file.
/// Returns total bytes written.
[[nodiscard]] inline size_t
save_checkpoint(const std::string& path,
                const CognitiveSnapshot& snap,
                uint64_t creation_time = 0) {
    if (creation_time == 0) creation_time = detail::now_ms();

    // WAL for crash safety
    WriteAheadLog wal(path + ".wal");
    wal.open();

    std::vector<uint8_t> file_buf;
    file_buf.reserve(64 * 1024 * 1024);  // 64 MiB initial

    // ── Global header ──
    NikHeader hdr{};
    hdr.magic         = NIK_MAGIC;
    hdr.version_major = NIK_VERSION_MAJOR;
    hdr.version_minor = NIK_VERSION_MINOR;
    hdr.creation_time = creation_time;
    hdr.last_snap_time = detail::now_ms();
    hdr.dim_encoding  = NIK_DIM_ENCODING;
    hdr.cipher_type   = NIK_CIPHER_NONE;
    std::memset(hdr.reserved, 0, sizeof(hdr.reserved));
    detail::append_raw(file_buf, &hdr, sizeof(hdr));

    std::vector<uint32_t> section_crcs;

    // Helper: write a section
    auto write_section = [&](uint32_t type, const std::vector<uint8_t>& payload,
                             uint8_t flags = 0) {
        wal.append(WAL_UPDATE, payload);

        SectionHeader shdr{};
        shdr.section_type = type;
        shdr.checksum     = crc32c(payload.data(), payload.size());
        shdr.payload_len  = payload.size();
        shdr.flags        = flags;
        std::memset(shdr.reserved, 0, sizeof(shdr.reserved));

        detail::append_raw(file_buf, &shdr, sizeof(shdr));
        detail::append_raw(file_buf, payload.data(), payload.size());
        section_crcs.push_back(shdr.checksum);
    };

    // ── Section 1: NikolaState ──
    write_section(SEC_NIKOLA_STATE, pack_nikola_state(snap.state));

    // ── Section 2: TorusGrid ──
    if (snap.grid) {
        write_section(SEC_TORUS_GRID,
                      pack_torus_grid(*snap.grid),
                      PAGE_COMPRESSED);
    }

    // ── Section 3: SSM Weights ──
    if (snap.ssm) {
        write_section(SEC_SSM_WEIGHTS, pack_ssm_weights(*snap.ssm));
    }

    // ── Section 4: NPT Heads ──
    if (snap.npt) {
        write_section(SEC_NPT_HEADS,
                      pack_npt_heads(*snap.npt),
                      PAGE_COMPRESSED);
    }

    // ── Section 5: Metric Tensor ──
    if (snap.metric) {
        write_section(SEC_METRIC_TENSOR, pack_metric_tensor(*snap.metric));
    }

    // ── Footer ──
    NikFooter footer{};
    detail::compute_merkle_root(section_crcs, footer.merkle_root);
    footer.total_sections = section_crcs.size();
    footer.total_nodes    = snap.grid ? snap.grid->num_active_nodes() : 0;
    footer.ssm_hidden_dim = snap.ssm ? static_cast<uint64_t>(snap.ssm->hidden_dim()) : 0;
    footer.ssm_output_dim = snap.ssm ? static_cast<uint64_t>(snap.ssm->output_dim()) : 0;
    footer.npt_num_heads  = cognitive::NPT_NUM_HEADS;
    std::memset(footer.reserved, 0, sizeof(footer.reserved));
    detail::append_raw(file_buf, &footer, sizeof(footer));

    // WAL commit before writing final file
    wal.commit();

    // ── Atomic write: write to .tmp then rename ──
    const std::string tmp_path = path + ".tmp";
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open())
            throw std::runtime_error("dmc: cannot open " + tmp_path);
        out.write(reinterpret_cast<const char*>(file_buf.data()),
                  static_cast<std::streamsize>(file_buf.size()));
        out.flush();
        if (!out.good())
            throw std::runtime_error("dmc: write failed to " + tmp_path);
    }

    std::filesystem::rename(tmp_path, path);
    wal.close_and_remove();

    return file_buf.size();
}

/// Load a full cognitive state from a .nik file.
inline void load_checkpoint(const std::string& path,
                            CognitiveSnapshot& snap) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open())
        throw std::runtime_error("dmc: cannot open " + path);

    const size_t file_size = static_cast<size_t>(in.tellg());
    if (file_size < NIK_HEADER_SIZE + NIK_FOOTER_SIZE)
        throw std::runtime_error("dmc: file too small");

    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(file_size);
    in.read(reinterpret_cast<char*>(buf.data()),
            static_cast<std::streamsize>(file_size));

    const uint8_t* data = buf.data();
    size_t pos = 0;

    // ── Read header ──
    NikHeader hdr;
    std::memcpy(&hdr, data, sizeof(hdr));
    pos += sizeof(hdr);

    if (hdr.magic != NIK_MAGIC)
        throw std::runtime_error("dmc: invalid magic");
    if (hdr.version_major != NIK_VERSION_MAJOR)
        throw std::runtime_error("dmc: unsupported version");

    // ── Read footer (at end of file) ──
    NikFooter footer;
    std::memcpy(&footer, data + file_size - NIK_FOOTER_SIZE,
                sizeof(footer));

    const size_t sections_end = file_size - NIK_FOOTER_SIZE;
    std::vector<uint32_t> section_crcs;

    // ── Read sections ──
    while (pos + sizeof(SectionHeader) <= sections_end) {
        SectionHeader shdr;
        std::memcpy(&shdr, data + pos, sizeof(shdr));
        pos += sizeof(shdr);

        if (pos + shdr.payload_len > sections_end)
            throw std::runtime_error("dmc: section overflow");

        const uint8_t* payload = data + pos;
        const size_t plen = static_cast<size_t>(shdr.payload_len);

        // Verify CRC
        const uint32_t actual_crc = crc32c(payload, plen);
        if (actual_crc != shdr.checksum)
            throw std::runtime_error("dmc: CRC mismatch in section");

        section_crcs.push_back(shdr.checksum);

        switch (shdr.section_type) {
            case SEC_NIKOLA_STATE:
                unpack_nikola_state(payload, plen, snap.state);
                break;

            case SEC_TORUS_GRID:
                if (snap.grid)
                    unpack_torus_grid(payload, plen, *snap.grid);
                break;

            case SEC_SSM_WEIGHTS:
                if (snap.ssm)
                    unpack_ssm_weights(payload, plen, *snap.ssm);
                break;

            case SEC_NPT_HEADS:
                if (snap.npt)
                    unpack_npt_heads(payload, plen, *snap.npt);
                break;

            case SEC_METRIC_TENSOR:
                if (snap.metric)
                    unpack_metric_tensor(payload, plen, *snap.metric);
                break;

            default:
                // Skip unknown sections (forward-compatibility)
                break;
        }

        pos += plen;
    }

    // Verify Merkle root
    uint8_t computed_root[32];
    detail::compute_merkle_root(section_crcs, computed_root);
    if (std::memcmp(computed_root, footer.merkle_root, 32) != 0)
        throw std::runtime_error("dmc: Merkle root mismatch");
}

// ────────────────────────────────────────────────────────────────────────────
// §7  Event-driven checkpoint controller
// ────────────────────────────────────────────────────────────────────────────

/// Controls when checkpoints should fire.
/// 300s periodic + NAP trigger (Gap 6.3).
class CheckpointController {
public:
    explicit CheckpointController(
        float interval_sec = CHECKPOINT_INTERVAL_SEC)
        : interval_sec_(interval_sec)
    {}

    /// Returns true if a checkpoint should be taken now.
    [[nodiscard]] bool should_checkpoint(float sim_time,
                                         ActionType action) noexcept {
        // NAP trigger: always checkpoint before sleep
        if (action == ActionType::NAP)
            return true;

        // Periodic trigger
        if (sim_time - last_checkpoint_time_ >= interval_sec_) {
            return true;
        }

        return false;
    }

    /// Mark that a checkpoint was just taken.
    void record_checkpoint(float sim_time) noexcept {
        last_checkpoint_time_ = sim_time;
        checkpoint_count_++;
    }

    [[nodiscard]] uint64_t checkpoint_count() const noexcept {
        return checkpoint_count_;
    }

    [[nodiscard]] float last_checkpoint_time() const noexcept {
        return last_checkpoint_time_;
    }

private:
    float interval_sec_;
    float last_checkpoint_time_ = 0.f;
    uint64_t checkpoint_count_  = 0;
};

}  // namespace nikola::persistence
