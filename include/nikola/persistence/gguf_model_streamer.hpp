/// @file   gguf_model_streamer.hpp
/// @brief  Streaming GGUF model reader for Nikola (GAP-014)
///
/// Provides header-only, no-allocation iteration over GGUF v3 tensor metadata
/// using the ggml-base C API (libggml-base.so).  Tensor data is never loaded
/// into RAM — only the tensor ``ne[]`` shape, type, and byte-offset are read.
///
/// Intended use: inspect models, validate tensor naming conventions, map
/// Nikola's DMC node-tensor layout, and integrate with the Q9_0 quantization
/// pipeline defined in §06_persistence/02_gguf_interoperability.md.

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// ggml-base C API
#include <ggml.h>
#include <gguf.h>

namespace nikola::persistence {

// ---------------------------------------------------------------------------
// Type mirror (avoids leaking ggml_type into call-site enums)
// ---------------------------------------------------------------------------
enum class GgmlType : uint32_t {
    F32    = 0,
    F16    = 1,
    Q4_0   = 2,
    Q4_1   = 3,
    Q5_0   = 6,
    Q5_1   = 7,
    Q8_0   = 8,
    Q8_1   = 9,
    Q2_K   = 10,
    Q3_K   = 11,
    Q4_K   = 12,
    Q5_K   = 13,
    Q6_K   = 14,
    Q8_K   = 15,
    BF16   = 30,
    UNKNOWN = 0xFFFFFFFFu,
};

/// @brief Convert ggml_type to our GgmlType enum.
inline GgmlType to_ggml_type(enum ggml_type t) noexcept
{
    return static_cast<GgmlType>(static_cast<uint32_t>(t));
}

/// @brief Return a human-readable name for a GgmlType.
inline std::string_view ggml_type_name(GgmlType t) noexcept
{
    switch (t) {
    case GgmlType::F32:  return "F32";
    case GgmlType::F16:  return "F16";
    case GgmlType::Q4_0: return "Q4_0";
    case GgmlType::Q4_1: return "Q4_1";
    case GgmlType::Q5_0: return "Q5_0";
    case GgmlType::Q5_1: return "Q5_1";
    case GgmlType::Q8_0: return "Q8_0";
    case GgmlType::Q8_1: return "Q8_1";
    case GgmlType::Q2_K: return "Q2_K";
    case GgmlType::Q3_K: return "Q3_K";
    case GgmlType::Q4_K: return "Q4_K";
    case GgmlType::Q5_K: return "Q5_K";
    case GgmlType::Q6_K: return "Q6_K";
    case GgmlType::Q8_K: return "Q8_K";
    case GgmlType::BF16: return "BF16";
    default:             return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Per-tensor metadata read from the GGUF header.
struct GgufTensorInfo {
    std::string           name;           ///< Tensor name (e.g. "blk.0.attn_q.weight")
    GgmlType              type;           ///< Quantization / element type
    uint32_t              n_dims;         ///< Number of occupied dimensions (1–4)
    std::array<int64_t,4> ne;             ///< Shape; unused slots are 1
    int64_t               n_elements;     ///< Total element count = Π ne[i]
    size_t                data_offset;    ///< Byte offset from GGUF data section start

    /// Number of bytes if converted to F32 (useful for size estimates).
    size_t bytes_as_f32() const noexcept
    {
        return static_cast<size_t>(n_elements) * sizeof(float);
    }
};

/// GGUF file-level metadata.
struct GgufMeta {
    uint32_t version;       ///< GGUF format version (expected: 3)
    size_t   alignment;     ///< Data-section alignment (default: 32)
    size_t   data_offset;   ///< Byte offset in file where tensor data begins
    int64_t  n_kv;          ///< Number of key-value metadata entries
    int64_t  n_tensors;     ///< Number of tensors in the file
};

// ---------------------------------------------------------------------------
// GgufModelStreamer
// ---------------------------------------------------------------------------

/// @brief Opens a GGUF file and lazily streams tensor metadata.
///
/// No tensor data is ever mapped into memory.  The ggml_context is allocated
/// with headers-only (no_alloc=true) so ggml can report tensor shapes.
///
/// Lifecycle: open → query meta/tensors → (optional) for_each_tensor → destroy
class GgufModelStreamer {
public:
    // Default GGML pool size — enough for ≈40 000 tensor structs
    static constexpr size_t DEFAULT_GGML_MEM = 32u * 1024u * 1024u;

    /// @brief Open the given GGUF file.
    /// @throws std::runtime_error if the file cannot be opened or is invalid.
    explicit GgufModelStreamer(const std::filesystem::path& path,
                               size_t ggml_mem = DEFAULT_GGML_MEM)
        : path_(path)
    {
        // Step 1: Initialise a ggml_context large enough for tensor structs.
        ggml_init_params gparams{};
        gparams.mem_size   = ggml_mem;
        gparams.mem_buffer = nullptr;   // ggml mallocs internally
        gparams.no_alloc   = true;      // tensor structs only — no data pages
        ggml_ctx_ = ggml_init(gparams);
        if (!ggml_ctx_)
            throw std::runtime_error(
                "GgufModelStreamer: ggml_init failed (out of memory?)");

        // Step 2: Open the GGUF file; fill ggml_ctx with tensor headers.
        gguf_init_params uparams{};
        uparams.no_alloc = true;
        uparams.ctx      = &ggml_ctx_;
        gguf_ctx_ = gguf_init_from_file(path_.c_str(), uparams);
        if (!gguf_ctx_) {
            ggml_free(ggml_ctx_);
            ggml_ctx_ = nullptr;
            throw std::runtime_error(
                std::string("GgufModelStreamer: cannot open '") +
                path_.string() + "' as GGUF");
        }

        // Step 3: Cache tensor infos for O(1) random access.
        int64_t nt = gguf_get_n_tensors(gguf_ctx_);
        tensors_.reserve(static_cast<size_t>(nt));
        for (int64_t i = 0; i < nt; ++i)
            tensors_.push_back(read_tensor(i));
    }

    ~GgufModelStreamer()
    {
        if (gguf_ctx_) gguf_free(gguf_ctx_);
        if (ggml_ctx_) ggml_free(ggml_ctx_);
    }

    // Non-copyable; moveable in principle but not needed right now.
    GgufModelStreamer(const GgufModelStreamer&)            = delete;
    GgufModelStreamer& operator=(const GgufModelStreamer&) = delete;

    // -----------------------------------------------------------------------
    // Meta accessors
    // -----------------------------------------------------------------------

    /// @brief Return file-level metadata.
    GgufMeta meta() const noexcept
    {
        return GgufMeta{
            gguf_get_version(gguf_ctx_),
            gguf_get_alignment(gguf_ctx_),
            gguf_get_data_offset(gguf_ctx_),
            gguf_get_n_kv(gguf_ctx_),
            gguf_get_n_tensors(gguf_ctx_),
        };
    }

    int64_t n_tensors() const noexcept { return gguf_get_n_tensors(gguf_ctx_); }
    int64_t n_kv()      const noexcept { return gguf_get_n_kv(gguf_ctx_);      }

    const std::filesystem::path& path() const noexcept { return path_; }

    // -----------------------------------------------------------------------
    // Tensor accessors
    // -----------------------------------------------------------------------

    /// @brief Return metadata for tensor at zero-based @p idx.
    /// @throws std::out_of_range if idx >= n_tensors().
    const GgufTensorInfo& tensor_info(int64_t idx) const
    {
        range_check(idx);
        return tensors_[static_cast<size_t>(idx)];
    }

    /// @brief Search tensor index by name.
    /// @returns index, or std::nullopt if not found.
    std::optional<int64_t> find_tensor(std::string_view name) const
    {
        int64_t idx = gguf_find_tensor(gguf_ctx_, std::string(name).c_str());
        if (idx < 0) return std::nullopt;
        return idx;
    }

    // -----------------------------------------------------------------------
    // KV metadata accessors
    // -----------------------------------------------------------------------

    /// @brief Return the key string for KV entry @p idx.
    std::string kv_key(int64_t idx) const
    {
        kv_range_check(idx);
        const char* k = gguf_get_key(gguf_ctx_, idx);
        return k ? k : "";
    }

    /// @brief Return the GGUF type tag for KV entry @p idx.
    enum gguf_type kv_type(int64_t idx) const
    {
        kv_range_check(idx);
        return gguf_get_kv_type(gguf_ctx_, idx);
    }

    /// @brief Return the string value for a GGUF_TYPE_STRING KV.
    std::string kv_string_value(int64_t idx) const
    {
        kv_range_check(idx);
        const char* v = gguf_get_val_str(gguf_ctx_, idx);
        return v ? v : "";
    }

    // -----------------------------------------------------------------------
    // Streaming callback
    // -----------------------------------------------------------------------

    /// @brief Visit all tensors in file order.
    void for_each_tensor(
        const std::function<void(int64_t idx, const GgufTensorInfo&)>& fn) const
    {
        for (int64_t i = 0; i < n_tensors(); ++i)
            fn(i, tensors_[static_cast<size_t>(i)]);
    }

private:
    // Build a GgufTensorInfo for one tensor index.
    GgufTensorInfo read_tensor(int64_t idx) const
    {
        GgufTensorInfo info{};
        info.name        = safe_str(gguf_get_tensor_name(gguf_ctx_, idx));
        info.type        = to_ggml_type(gguf_get_tensor_type(gguf_ctx_, idx));
        info.data_offset = gguf_get_tensor_offset(gguf_ctx_, idx);

        // Shape from ggml_tensor (populated by gguf_init_from_file with ctx)
        const ggml_tensor* t =
            ggml_get_tensor(ggml_ctx_, info.name.c_str());
        if (t) {
            // Determine real n_dims: find last axis where ne > 1 (or min 1)
            info.n_dims = 1;
            for (int d = GGML_MAX_DIMS - 1; d >= 1; --d)
                if (t->ne[d] > 1) { info.n_dims = static_cast<uint32_t>(d + 1); break; }
            info.n_elements = ggml_nelements(t);
            for (int d = 0; d < GGML_MAX_DIMS; ++d)
                info.ne[static_cast<size_t>(d)] = t->ne[d];
        } else {
            // Fallback: ggml_ctx not available or tensor not found
            info.n_dims     = 0;
            info.n_elements = 0;
            info.ne         = {0, 0, 0, 0};
        }
        return info;
    }

    static std::string safe_str(const char* p) { return p ? p : ""; }

    void range_check(int64_t idx) const
    {
        if (idx < 0 || idx >= n_tensors())
            throw std::out_of_range(
                "GgufModelStreamer::tensor_info: index " +
                std::to_string(idx) + " out of range [0," +
                std::to_string(n_tensors()) + ")");
    }
    void kv_range_check(int64_t idx) const
    {
        if (idx < 0 || idx >= n_kv())
            throw std::out_of_range(
                "GgufModelStreamer::kv_*: index " +
                std::to_string(idx) + " out of range [0," +
                std::to_string(n_kv()) + ")");
    }

    gguf_context*             gguf_ctx_ = nullptr;
    ggml_context*             ggml_ctx_ = nullptr;
    std::filesystem::path     path_;
    std::vector<GgufTensorInfo> tensors_;
};

} // namespace nikola::persistence
