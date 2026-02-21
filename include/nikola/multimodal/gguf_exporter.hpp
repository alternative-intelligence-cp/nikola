/**
 * @file gguf_exporter.hpp
 * @brief Gap 6.4 — GGUFExporter (custom binary writer, no gguf.h dependency)
 *
 * Writes a GGUF-compatible binary file containing Nikola topology metadata
 * and (optionally) wavefunction tensor data.
 *
 * GGUF format reference (https://github.com/ggerganov/ggml/blob/master/docs/gguf.md):
 *   Magic   : uint32  0x46554747 ("GGUF")
 *   Version : uint32  3
 *   n_tensors: uint64
 *   n_kv    : uint64
 *   [KV pairs]
 *   [Tensor info]
 *   [Tensor data – aligned to 32 bytes]
 *
 * This implementation writes the KV metadata correctly and exports the
 * wavefunction as raw FP32 tensors (file_type = 0 = ALL_F32 for tensor data,
 * while the KV metadata records general.file_type = 9 = Q9_0 intent).
 */
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace nikola::multimodal {

// ============================================================================
// GGUF format constants
// ============================================================================

inline constexpr uint32_t GGUF_MAGIC   = 0x46554747u; // "GGUF" LE
inline constexpr uint32_t GGUF_VERSION = 3u;

// GGUF KV type IDs
enum class GGUFValueType : uint32_t {
    UINT8    = 0,
    INT8     = 1,
    UINT16   = 2,
    INT16    = 3,
    UINT32   = 4,
    INT32    = 5,
    FLOAT32  = 6,
    BOOL     = 7,
    STRING   = 8,
    ARRAY    = 9,
    UINT64   = 10,
    INT64    = 11,
    FLOAT64  = 12,
};

// GGUF tensor type IDs
enum class GGUFTensorType : uint32_t {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    // ... (only F32/F16 used here)
};

// ============================================================================
// Gap 6.4 — Nikola GGUF metadata
// ============================================================================

/**
 * Required KV metadata for a Nikola GGUF export.
 */
struct NikolaGGUFMeta {
    // nikola.topology.dims  = [16, 16, 128, 32, 32, 32, 64, 64, 64]  (r,s,t,u,v,w,x,y,z)
    std::array<int64_t, 9> topology_dims  = {16, 16, 128, 32, 32, 32, 64, 64, 64};

    // nikola.topology.names = ["r","s","t","u","v","w","x","y","z"]
    std::array<std::string, 9> topology_names = {"r","s","t","u","v","w","x","y","z"};

    // nikola.topology.semantics
    std::array<std::string, 9> topology_semantics = {
        "resonance","state","time",
        "quantum_u","quantum_v","quantum_w",
        "spatial_x","spatial_y","spatial_z"
    };

    // general.*
    std::string architecture = "nikola_v0";
    int32_t     file_type    = 9; // Q9_0
};

// ============================================================================
// GGUFExporter
// ============================================================================

class GGUFExporter {
public:
    /**
     * Export a GGUF file with Nikola metadata + optional wavefunction tensors.
     *
     * @param filename   Output file path (e.g. "nikola_checkpoint.gguf")
     * @param psi_real   Real part of wavefunction (may be empty → no tensor data)
     * @param psi_imag   Imaginary part of wavefunction (same size as psi_real)
     * @param meta       Nikola topology metadata
     */
    static void export_metadata(const std::string&       filename,
                                 std::span<const float>   psi_real = {},
                                 std::span<const float>   psi_imag = {},
                                 const NikolaGGUFMeta&    meta     = {})
    {
        std::ofstream f(filename, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("GGUFExporter: cannot open " + filename);

        // Determine tensor count
        const uint64_t n_tensors = (psi_real.empty()) ? 0u : 2u; // real + imag
        // (+1 if we export metric, but we keep it simple here)

        // Count KV pairs:
        //   nikola.topology.dims     (array of INT64)
        //   nikola.topology.names    (array of STRING)
        //   nikola.topology.semantics(array of STRING)
        //   general.architecture     (string)
        //   general.file_type        (int32)
        const uint64_t n_kv = 5u;

        // ---- Header ----
        write_u32(f, GGUF_MAGIC);
        write_u32(f, GGUF_VERSION);
        write_u64(f, n_tensors);
        write_u64(f, n_kv);

        // ---- KV: nikola.topology.dims ----
        write_kv_key(f, "nikola.topology.dims");
        write_u32(f, static_cast<uint32_t>(GGUFValueType::ARRAY));
        // Array: type of elements + count + data
        write_u32(f, static_cast<uint32_t>(GGUFValueType::INT64)); // element type
        write_u64(f, static_cast<uint64_t>(meta.topology_dims.size()));
        for (int64_t d : meta.topology_dims) write_i64(f, d);

        // ---- KV: nikola.topology.names ----
        write_kv_key(f, "nikola.topology.names");
        write_u32(f, static_cast<uint32_t>(GGUFValueType::ARRAY));
        write_u32(f, static_cast<uint32_t>(GGUFValueType::STRING));
        write_u64(f, static_cast<uint64_t>(meta.topology_names.size()));
        for (const auto& s : meta.topology_names) write_gguf_string(f, s);

        // ---- KV: nikola.topology.semantics ----
        write_kv_key(f, "nikola.topology.semantics");
        write_u32(f, static_cast<uint32_t>(GGUFValueType::ARRAY));
        write_u32(f, static_cast<uint32_t>(GGUFValueType::STRING));
        write_u64(f, static_cast<uint64_t>(meta.topology_semantics.size()));
        for (const auto& s : meta.topology_semantics) write_gguf_string(f, s);

        // ---- KV: general.architecture ----
        write_kv_key(f, "general.architecture");
        write_u32(f, static_cast<uint32_t>(GGUFValueType::STRING));
        write_gguf_string(f, meta.architecture);

        // ---- KV: general.file_type ----
        write_kv_key(f, "general.file_type");
        write_u32(f, static_cast<uint32_t>(GGUFValueType::INT32));
        write_i32(f, meta.file_type);

        // ---- Tensor info (if any) ----
        if (n_tensors > 0) {
            write_tensor_info(f, "wavefunction.real", psi_real.size());
            write_tensor_info(f, "wavefunction.imag", psi_imag.size());

            // Align to 32 bytes
            const auto cur = static_cast<size_t>(f.tellp());
            const size_t pad = (32 - (cur % 32)) % 32;
            for (size_t i = 0; i < pad; ++i) f.put(0);

            // Write tensor data
            if (!psi_real.empty()) {
                f.write(reinterpret_cast<const char*>(psi_real.data()),
                        static_cast<std::streamsize>(psi_real.size() * sizeof(float)));
            }
            if (!psi_imag.empty()) {
                f.write(reinterpret_cast<const char*>(psi_imag.data()),
                        static_cast<std::streamsize>(psi_imag.size() * sizeof(float)));
            }
        }

        if (!f) throw std::runtime_error("GGUFExporter: write error");
    }

    /**
     * Read back the general.architecture string from an exported file.
     * Useful for validation in tests.
     */
    static std::string read_architecture(const std::string& filename)
    {
        std::ifstream f(filename, std::ios::binary);
        if (!f) throw std::runtime_error("GGUFExporter: cannot open " + filename);

        uint32_t magic = 0;
        f.read(reinterpret_cast<char*>(&magic), 4);
        if (magic != GGUF_MAGIC)
            throw std::runtime_error("GGUFExporter: invalid GGUF magic");

        uint32_t version = 0; f.read(reinterpret_cast<char*>(&version), 4);
        uint64_t n_tensors = 0; f.read(reinterpret_cast<char*>(&n_tensors), 8);
        uint64_t n_kv = 0; f.read(reinterpret_cast<char*>(&n_kv), 8);

        // Scan KV pairs looking for general.architecture
        for (uint64_t i = 0; i < n_kv; ++i) {
            const std::string key = read_gguf_string(f);
            uint32_t vtype = 0; f.read(reinterpret_cast<char*>(&vtype), 4);

            if (key == "general.architecture" &&
                vtype == static_cast<uint32_t>(GGUFValueType::STRING))
            {
                return read_gguf_string(f);
            }

            // Skip this value
            skip_value(f, static_cast<GGUFValueType>(vtype));
        }
        return "";
    }

    /**
     * Read back topology dims array from an exported file.
     */
    static std::vector<int64_t> read_topology_dims(const std::string& filename)
    {
        std::ifstream f(filename, std::ios::binary);
        if (!f) throw std::runtime_error("GGUFExporter: cannot open " + filename);

        uint32_t magic = 0; f.read(reinterpret_cast<char*>(&magic), 4);
        if (magic != GGUF_MAGIC)
            throw std::runtime_error("GGUFExporter: invalid GGUF magic");

        uint32_t version = 0; f.read(reinterpret_cast<char*>(&version), 4);
        uint64_t n_tensors = 0; f.read(reinterpret_cast<char*>(&n_tensors), 8);
        uint64_t n_kv = 0; f.read(reinterpret_cast<char*>(&n_kv), 8);

        for (uint64_t i = 0; i < n_kv; ++i) {
            const std::string key = read_gguf_string(f);
            uint32_t vtype = 0; f.read(reinterpret_cast<char*>(&vtype), 4);

            if (key == "nikola.topology.dims" &&
                vtype == static_cast<uint32_t>(GGUFValueType::ARRAY))
            {
                uint32_t elem_type = 0; f.read(reinterpret_cast<char*>(&elem_type), 4);
                uint64_t count = 0;     f.read(reinterpret_cast<char*>(&count), 8);
                std::vector<int64_t> dims(count);
                for (auto& d : dims) f.read(reinterpret_cast<char*>(&d), 8);
                return dims;
            }

            skip_value(f, static_cast<GGUFValueType>(vtype));
        }
        return {};
    }

private:
    // ---- Binary write helpers ------------------------------------------------

    static void write_u32(std::ofstream& f, uint32_t v) {
        f.write(reinterpret_cast<const char*>(&v), 4);
    }
    static void write_u64(std::ofstream& f, uint64_t v) {
        f.write(reinterpret_cast<const char*>(&v), 8);
    }
    static void write_i32(std::ofstream& f, int32_t v) {
        f.write(reinterpret_cast<const char*>(&v), 4);
    }
    static void write_i64(std::ofstream& f, int64_t v) {
        f.write(reinterpret_cast<const char*>(&v), 8);
    }

    /// GGUF string: uint64 length + UTF-8 bytes (no null terminator)
    static void write_gguf_string(std::ofstream& f, const std::string& s) {
        const uint64_t len = s.size();
        f.write(reinterpret_cast<const char*>(&len), 8);
        f.write(s.data(), static_cast<std::streamsize>(len));
    }

    /// KV key = GGUF string
    static void write_kv_key(std::ofstream& f, const std::string& key) {
        write_gguf_string(f, key);
    }

    /// Minimal tensor info: name, n_dims=1, shape[0], type=F32, offset=0
    static void write_tensor_info(std::ofstream& f,
                                   const std::string& name,
                                   size_t elem_count)
    {
        write_gguf_string(f, name);
        write_u32(f, 1u); // n_dims
        const uint64_t ne0 = elem_count;
        f.write(reinterpret_cast<const char*>(&ne0), 8);
        write_u32(f, static_cast<uint32_t>(GGUFTensorType::F32));
        write_u64(f, 0u); // offset (placeholder)
    }

    // ---- Binary read helpers -------------------------------------------------

    static std::string read_gguf_string(std::ifstream& f) {
        uint64_t len = 0;
        f.read(reinterpret_cast<char*>(&len), 8);
        if (!f || len > (1u << 20)) return "";
        std::string s(len, '\0');
        f.read(s.data(), static_cast<std::streamsize>(len));
        return s;
    }

    static void skip_value(std::ifstream& f, GGUFValueType vtype) {
        switch (vtype) {
            case GGUFValueType::UINT8:
            case GGUFValueType::INT8:
            case GGUFValueType::BOOL:  f.seekg(1, std::ios::cur); break;
            case GGUFValueType::UINT16:
            case GGUFValueType::INT16:  f.seekg(2, std::ios::cur); break;
            case GGUFValueType::UINT32:
            case GGUFValueType::INT32:
            case GGUFValueType::FLOAT32: f.seekg(4, std::ios::cur); break;
            case GGUFValueType::UINT64:
            case GGUFValueType::INT64:
            case GGUFValueType::FLOAT64: f.seekg(8, std::ios::cur); break;
            case GGUFValueType::STRING:  read_gguf_string(f); break;
            case GGUFValueType::ARRAY: {
                uint32_t elem_type = 0; f.read(reinterpret_cast<char*>(&elem_type), 4);
                uint64_t count = 0;     f.read(reinterpret_cast<char*>(&count), 8);
                for (uint64_t i = 0; i < count; ++i)
                    skip_value(f, static_cast<GGUFValueType>(elem_type));
                break;
            }
            default: break;
        }
    }
};

} // namespace nikola::multimodal
