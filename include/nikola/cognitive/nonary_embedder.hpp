#pragma once

// NonaryEmbedder — text → balanced nonary wave vector
// Pipeline: text → BPETokenizer → TinyTransformer → tanh quantize → Nit[128]
//
// Spec: docs/info/engineering/03_cognitive_systems.txt §3.4.1
// quantize_to_nit: tanh(val) scaled to [-4,+4] representing wave amplitude states

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nikola/cognitive/bpe_tokenizer.hpp>
#include <nikola/cognitive/tiny_transformer.hpp>
#include <nikola/foundation/nit.hpp>    // for Nit type
#include <nikola/diag/scope_profiler.hpp>

namespace nikola::cognitive {

// Import Nit from foundation namespace
using nikola::foundation::Nit;

class NonaryEmbedder {
public:
    static constexpr size_t EMBEDDING_DIM = 128;  // BERT-Tiny hidden size

    // Initialize with tokenizer vocab and ONNX model path
    NonaryEmbedder(const std::string& tokenizer_path,
                   const std::string& model_path)
        : tokenizer_(tokenizer_path)
        , transformer_(model_path) {
        std::cout << "[NonaryEmbedder] Initialized — BPE + TinyTransformer (128-dim→Nit)\n";
    }

    // text → 128 Nit balanced-nonary wave vector
    std::vector<Nit> embed(const std::string& text) const {
        NIKOLA_PROFILE("embed::nonary");
        if (text.empty()) {
            return std::vector<Nit>(EMBEDDING_DIM, 0);
        }

        // 1. Tokenize
        auto token_ids = tokenizer_.encode(text);

        // 2. Forward pass → 128-dim float [CLS] embedding
        auto float_vec = transformer_.forward(token_ids);

        if (float_vec.size() < EMBEDDING_DIM) {
            float_vec.resize(EMBEDDING_DIM, 0.0f);
        }

        // 3. Quantize to balanced nonary [-4, +4]
        std::vector<Nit> nit_vec;
        nit_vec.reserve(EMBEDDING_DIM);
        for (size_t i = 0; i < EMBEDDING_DIM; ++i) {
            nit_vec.push_back(quantize_to_nit(float_vec[i]));
        }

        return nit_vec;
    }

    // Embed and return raw float vector (for diagnostics/distance comparison)
    std::vector<float> embed_float(const std::string& text) const {
        auto token_ids = tokenizer_.encode(text);
        auto vec = transformer_.forward(token_ids);
        vec.resize(EMBEDDING_DIM, 0.0f);
        return vec;
    }

    // Cosine similarity between two float embeddings (for resonance matching)
    static float cosine_similarity(const std::vector<float>& a,
                                   const std::vector<float>& b) {
        if (a.size() != b.size() || a.empty()) return 0.0f;
        float dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        float denom = std::sqrt(na) * std::sqrt(nb);
        return denom < 1e-9f ? 0.0f : dot / denom;
    }

private:
    BPETokenizer    tokenizer_;
    TinyTransformer transformer_;

    // Quantize float → Nit using tanh to compress to [-1,1], scale to [-4,+4]
    static Nit quantize_to_nit(float val) {
        float normalized = std::tanh(val);                   // → [-1, 1]
        int   quantized  = static_cast<int>(std::round(normalized * 4.0f));
        return static_cast<Nit>(std::clamp(quantized, -4, 4));
    }
};

} // namespace nikola::cognitive
