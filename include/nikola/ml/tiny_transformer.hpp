/// @file   tiny_transformer.hpp
/// @brief  BERT-Tiny ONNX-Runtime inference wrapper (GAP-015)
///
/// Wraps an ONNX-Runtime Session to expose a minimal BERT-Tiny forward pass:
///   token_ids (int64 [1, seq_len])  →  embedding (float32 [1, hidden_dim])
///
/// Architecture constants reflect the BERT-Tiny configuration:
///   4 transformer layers, 128 hidden dimensions, 2 attention heads.
///
/// Thread-compatibility: one forward() at a time per TinyTransformer instance.

#pragma once

#include <array>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

// ORT C++ API (header-only bindings over the C API)
#include <onnxruntime_cxx_api.h>

namespace nikola::ml {

// ---------------------------------------------------------------------------
// BERT-Tiny architecture constants
// ---------------------------------------------------------------------------
constexpr int BERT_TINY_LAYERS     = 4;    ///< Number of transformer encoder layers
constexpr int BERT_TINY_HIDDEN_DIM = 128;  ///< Hidden / embedding dimension
constexpr int BERT_TINY_ATTN_HEADS = 2;    ///< Multi-head attention heads
constexpr int MAX_SEQ_LEN          = 512;  ///< Maximum token-sequence length

// ---------------------------------------------------------------------------
// TinyTransformer
// ---------------------------------------------------------------------------

/// @brief Lightweight ONNXRuntime wrapper for BERT-Tiny inference.
///
/// Constructs an Ort::Env + Ort::Session from a serialised *.onnx model file.
/// The expected model I/O convention used by Nikola is:
///   - Input  "input_ids"  int64  [1, seq_len]
///   - Output "embedding"  float  [1, BERT_TINY_HIDDEN_DIM]
///
/// In production the BERT-Tiny model is loaded once and reused across
/// inference calls; the embedding is treated as the CLS-pooled representation.
class TinyTransformer {
public:
    /// @brief Load and JIT-compile the ONNX model at @p model_path.
    ///
    /// @throws Ort::Exception (inherits std::exception) if the model file
    ///         cannot be opened, parsed, or compiled.
    explicit TinyTransformer(const std::filesystem::path& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "nikola_tt")
        , session_(env_, model_path.c_str(), make_session_opts())
    {
        Ort::AllocatorWithDefaultOptions alloc;
        // Cache I/O names as std::string so callers can hold references.
        input_name_  = session_.GetInputNameAllocated(0, alloc).get();
        output_name_ = session_.GetOutputNameAllocated(0, alloc).get();
    }

    // Non-copyable; moveable via Ort::Session move semantics.
    TinyTransformer(const TinyTransformer&)            = delete;
    TinyTransformer& operator=(const TinyTransformer&) = delete;
    TinyTransformer(TinyTransformer&&)                 = default;
    TinyTransformer& operator=(TinyTransformer&&)      = default;

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /// @brief Run one forward pass through the model.
    ///
    /// @param token_ids  Sequence of integer token IDs (length must match the
    ///                   model's static or dynamic sequence dimension).
    /// @returns          Flat float vector of length (batch × BERT_TINY_HIDDEN_DIM).
    ///                   For batch=1: exactly BERT_TINY_HIDDEN_DIM elements.
    /// @throws std::invalid_argument if @p token_ids is empty.
    /// @throws Ort::Exception on any ORT runtime error.
    std::vector<float> forward(const std::vector<int64_t>& token_ids)
    {
        if (token_ids.empty())
            throw std::invalid_argument(
                "TinyTransformer::forward: token_ids must not be empty");

        // Build input tensor — shape [1, seq_len]
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                                   OrtMemTypeDefault);
        std::array<int64_t, 2> shape{1, static_cast<int64_t>(token_ids.size())};

        // CreateTensor requires a non-const pointer; copy into local buffer.
        std::vector<int64_t> token_buf = token_ids;
        auto input_val = Ort::Value::CreateTensor<int64_t>(
            mem_info, token_buf.data(), token_buf.size(),
            shape.data(), shape.size());

        // Run
        const char* in_name  = input_name_.c_str();
        const char* out_name = output_name_.c_str();
        Ort::RunOptions run_opts{nullptr};
        auto outputs = session_.Run(run_opts, &in_name, &input_val, 1,
                                    &out_name, 1);

        // Extract output as std::vector<float>
        const float* ptr    = outputs[0].GetTensorData<float>();
        const auto   n_elem = outputs[0].GetTensorTypeAndShapeInfo()
                                         .GetElementCount();
        return std::vector<float>(ptr, ptr + n_elem);
    }

    // -----------------------------------------------------------------------
    // Accessors (useful for testing and diagnostics)
    // -----------------------------------------------------------------------

    const std::string& input_name()  const noexcept { return input_name_;  }
    const std::string& output_name() const noexcept { return output_name_; }
    std::size_t        num_inputs()  const { return session_.GetInputCount();  }
    std::size_t        num_outputs() const { return session_.GetOutputCount(); }

private:
    // Build a minimal SessionOptions: single-threaded, basic optimisations.
    static Ort::SessionOptions make_session_opts()
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_BASIC);
        return opts;
    }

    Ort::Env     env_;        ///< ORT runtime environment (must outlive session_)
    Ort::Session session_;    ///< Compiled inference session
    std::string  input_name_;
    std::string  output_name_;
};

// ---------------------------------------------------------------------------
// Standalone helpers
// ---------------------------------------------------------------------------

/// @brief Convenience factory for a standalone Ort::Env.
///
/// Useful in unit tests that want to verify ORT initialisation
/// independently of TinyTransformer.
[[nodiscard]] inline Ort::Env
create_ort_env(const std::string& log_id = "nikola",
               OrtLoggingLevel    level   = ORT_LOGGING_LEVEL_WARNING)
{
    return Ort::Env(level, log_id.c_str());
}

} // namespace nikola::ml
