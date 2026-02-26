#pragma once

// TinyTransformer — BERT-Tiny (4L/128H/2A) via ONNX Runtime C++ API
// Loads model.onnx exported from prajjwal1/bert-tiny and runs forward pass.
// token_ids → 128-dim float [CLS] embedding
//
// Spec: docs/info/engineering/03_cognitive_systems.txt §3.4.1
// ONNX Runtime: /home/randy/Workspace/SYSTEM/onnxruntime/cpp/
// Model:        /home/randy/Workspace/SYSTEM/onnxruntime/bert-tiny-onnx/model.onnx

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace nikola::cognitive {

class TinyTransformer {
public:
    // BERT-Tiny dimensions
    static constexpr int64_t HIDDEN_DIM  = 128;
    static constexpr int64_t MAX_SEQ_LEN = 512;

    explicit TinyTransformer(const std::string& model_path)
        : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NikolaTinyTransformer");

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), opts);

        std::cout << "[TinyTransformer] Loaded: " << model_path << "\n";
        std::cout << "[TinyTransformer] Architecture: BERT-Tiny (4L/128H/2A)\n";

        // Validate output count
        size_t out_count = session_->GetOutputCount();
        if (out_count == 0) {
            throw std::runtime_error("[TinyTransformer] Model has no outputs");
        }

        // Introspect model input names — some exports omit attention_mask
        Ort::AllocatorWithDefaultOptions alloc;
        size_t in_count = session_->GetInputCount();
        input_names_storage_.clear();
        input_names_.clear();
        has_attention_mask_ = false;
        for (size_t i = 0; i < in_count; ++i) {
            auto name = session_->GetInputNameAllocated(i, alloc);
            input_names_storage_.emplace_back(name.get());
            if (input_names_storage_.back() == "attention_mask") {
                has_attention_mask_ = true;
            }
        }
        for (const auto& s : input_names_storage_) {
            input_names_.push_back(s.c_str());
        }

        // Introspect fixed vs dynamic sequence length from input shape
        // Shape is [batch, seq_len]; if seq_len > 0 it is a fixed requirement
        {
            auto type_info = session_->GetInputTypeInfo(0);
            auto ti = type_info.GetTensorTypeAndShapeInfo();
            auto shape = ti.GetShape();
            if (shape.size() >= 2 && shape[1] > 0) {
                fixed_seq_len_ = static_cast<size_t>(shape[1]);
            }
        }

        // Introspect output name
        output_names_storage_.clear();
        output_names_.clear();
        size_t out_count2 = session_->GetOutputCount();
        for (size_t i = 0; i < out_count2; ++i) {
            auto name = session_->GetOutputNameAllocated(i, alloc);
            output_names_storage_.emplace_back(name.get());
        }
        for (const auto& s : output_names_storage_) {
            output_names_.push_back(s.c_str());
        }
    }

    // Forward pass: token IDs → 128-dim [CLS] embedding
    // attention_mask: 1 for real tokens, 0 for padding (auto-generated if empty)
    std::vector<float> forward(const std::vector<int64_t>& token_ids,
                               const std::vector<int64_t>& attention_mask = {}) const {

        // Determine effective sequence length (respect fixed model shape if any)
        size_t max_len = (fixed_seq_len_ > 0) ? fixed_seq_len_
                                               : static_cast<size_t>(MAX_SEQ_LEN);
        size_t seq_len = std::min(token_ids.size(), max_len);

        // Truncate token IDs, then zero-pad to max_len if model requires fixed shape
        std::vector<int64_t> ids(max_len, 0);
        std::copy(token_ids.begin(), token_ids.begin() + seq_len, ids.begin());

        // Build attention mask (1 for real tokens, 0 for padding)
        std::vector<int64_t> mask(max_len, 0);
        if (attention_mask.empty()) {
            std::fill(mask.begin(), mask.begin() + seq_len, 1);
        } else {
            size_t copy_len = std::min(attention_mask.size(), seq_len);
            std::copy(attention_mask.begin(), attention_mask.begin() + copy_len, mask.begin());
        }

        seq_len = max_len;  // tensor shape must match model expectation

        std::array<int64_t, 2> shape{1, static_cast<int64_t>(seq_len)};

        // Create input tensors
        Ort::Value ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info_, ids.data(), ids.size(), shape.data(), shape.size());
        Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info_, mask.data(), mask.size(), shape.data(), shape.size());

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(ids_tensor));
        if (has_attention_mask_) {
            inputs.push_back(std::move(mask_tensor));
        }

        // Allocate output tensor
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), inputs.data(), inputs.size(),
            output_names_.data(), output_names_.size());

        // Extract [CLS] token embedding: output shape [1, seq_len, 128]
        // [CLS] is at position 0 → first HIDDEN_DIM floats
        const float* out_data = outputs[0].GetTensorData<float>();
        return std::vector<float>(out_data, out_data + HIDDEN_DIM);
    }

private:
    std::unique_ptr<Ort::Env>     env_;
    std::unique_ptr<Ort::Session> session_;
    mutable Ort::MemoryInfo       memory_info_;

    // Populated at construction by introspecting the loaded model
    std::vector<std::string>     input_names_storage_;
    std::vector<std::string>     output_names_storage_;
    std::vector<const char*>     input_names_;
    std::vector<const char*>     output_names_;
    bool                         has_attention_mask_ = false;
    size_t                       fixed_seq_len_      = 0;  // 0 = dynamic
};

} // namespace nikola::cognitive
