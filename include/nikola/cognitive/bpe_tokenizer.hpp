#pragma once

// BPETokenizer — Phase 1 (whitespace approximation)
// Phase 1 splits on whitespace and maps to token IDs via a simple vocab lookup.
// When full BPE vocab is loaded from tokenizer.json this is a direct drop-in.
//
// Spec: docs/info/engineering/03_cognitive_systems.txt §3.4.1
//       "BPE tokenization — whitespace approximation for Phase 1"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nikola::cognitive {

class BPETokenizer {
public:
    // Special token IDs matching BERT convention
    static constexpr int64_t TOKEN_CLS     = 101;
    static constexpr int64_t TOKEN_SEP     = 102;
    static constexpr int64_t TOKEN_PAD     = 0;
    static constexpr int64_t TOKEN_UNK     = 100;
    static constexpr size_t  MAX_SEQ_LEN   = 512;

    // Default constructor — uses fallback hash-based IDs (no vocab file)
    BPETokenizer() = default;

    // Load vocab from tokenizer.json (HuggingFace format)
    // Falls back gracefully if file not found — uses hash mapping
    explicit BPETokenizer(const std::string& tokenizer_json_path) {
        load_vocab(tokenizer_json_path);
    }

    // Encode text → token IDs with CLS/SEP framing, truncated to MAX_SEQ_LEN
    std::vector<int64_t> encode(const std::string& text) const {
        std::vector<int64_t> ids;
        ids.reserve(64);
        ids.push_back(TOKEN_CLS);

        auto words = split_whitespace(lowercase(text));
        for (const auto& word : words) {
            if (ids.size() >= MAX_SEQ_LEN - 1) break;  // leave room for SEP
            ids.push_back(word_to_id(word));
        }

        ids.push_back(TOKEN_SEP);
        return ids;
    }

    // Encode with fixed-length padding
    std::vector<int64_t> encode_padded(const std::string& text, size_t length) const {
        auto ids = encode(text);
        ids.resize(std::min(ids.size(), length), TOKEN_PAD);
        while (ids.size() < length) ids.push_back(TOKEN_PAD);
        return ids;
    }

    size_t vocab_size() const { return vocab_.empty() ? 30522 : vocab_.size(); }

private:
    std::unordered_map<std::string, int64_t> vocab_;

    void load_vocab(const std::string& path) {
        // Minimal tokenizer.json parser — extracts "vocab" map from HuggingFace format
        // Format: {"model": {"vocab": {"[CLS]": 101, "word": id, ...}}}
        // Accepts either a direct file path or a directory containing tokenizer.json
        std::ifstream f(path);
        if (!f.is_open()) {
            // Try treating path as a directory and appending /tokenizer.json
            std::ifstream f2(path + "/tokenizer.json");
            if (!f2.is_open()) return;  // graceful fallback to hash mode
            f2.close();
            f.open(path + "/tokenizer.json");
        }
        if (!f.is_open()) return;

        std::string line;
        bool in_vocab = false;
        while (std::getline(f, line)) {
            // Detect vocab section start
            if (line.find("\"vocab\"") != std::string::npos &&
                line.find('{') != std::string::npos) {
                in_vocab = true;
                continue;
            }
            if (!in_vocab) continue;
            if (line.find('}') != std::string::npos && in_vocab) {
                in_vocab = false;
                break;
            }

            // Parse: "token": id
            auto q1 = line.find('"');
            if (q1 == std::string::npos) continue;
            auto q2 = line.find('"', q1 + 1);
            if (q2 == std::string::npos) continue;
            auto colon = line.find(':', q2);
            if (colon == std::string::npos) continue;

            std::string token = line.substr(q1 + 1, q2 - q1 - 1);
            try {
                int64_t id = std::stoll(line.substr(colon + 1));
                vocab_[token] = id;
            } catch (...) {}
        }
    }

    int64_t word_to_id(const std::string& word) const {
        if (!vocab_.empty()) {
            auto it = vocab_.find(word);
            if (it != vocab_.end()) return it->second;
            // Try ##word (continuation token)
            auto it2 = vocab_.find("##" + word);
            if (it2 != vocab_.end()) return it2->second;
            return TOKEN_UNK;
        }
        // Fallback: deterministic hash into [1000, 30000]
        size_t h = std::hash<std::string>{}(word);
        return static_cast<int64_t>(1000 + (h % 29000));
    }

    static std::string lowercase(const std::string& s) {
        std::string out = s;
        std::transform(out.begin(), out.end(), out.begin(), ::tolower);
        return out;
    }

    static std::vector<std::string> split_whitespace(const std::string& s) {
        std::vector<std::string> tokens;
        std::istringstream ss(s);
        std::string token;
        while (ss >> token) tokens.push_back(token);
        return tokens;
    }
};

} // namespace nikola::cognitive
