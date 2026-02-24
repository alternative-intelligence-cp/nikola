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
    }

    // Forward pass: token IDs → 128-dim [CLS] embedding
    // attention_mask: 1 for real tokens, 0 for padding (auto-generated if empty)
    std::vector<float> forward(const std::vector<int64_t>& token_ids,
                               const std::vector<int64_t>& attention_mask = {}) const {

        size_t seq_len = std::min(token_ids.size(), static_cast<size_t>(MAX_SEQ_LEN));

        // Truncate token IDs
        std::vector<int64_t> ids(token_ids.begin(), token_ids.begin() + seq_len);

        // Build attention mask (all 1s for real tokens)
        std::vector<int64_t> mask;
        if (attention_mask.empty()) {
            mask.assign(seq_len, 1);
        } else {
            mask.assign(attention_mask.begin(),
                        attention_mask.begin() + std::min(attention_mask.size(), seq_len));
        }

        std::array<int64_t, 2> shape{1, static_cast<int64_t>(seq_len)};

        // Create input tensors
        Ort::Value ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info_, ids.data(), ids.size(), shape.data(), shape.size());
        Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info_, mask.data(), mask.size(), shape.data(), shape.size());

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(ids_tensor));
        inputs.push_back(std::move(mask_tensor));

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

    // BERT-Tiny input/output names
    std::vector<const char*> input_names_  {"input_ids", "attention_mask"};
    std::vector<const char*> output_names_ {"last_hidden_state"};
};

} // namespace nikola::cognitive
