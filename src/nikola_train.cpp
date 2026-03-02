/**
 * @file src/nikola_train.cpp
 * @brief nikola-train — Corpus ingestion → holographic memory.
 *
 * Reads a corpus file (plain text — one training sentence per line, or JSONL
 * with {"text":"..."} objects), injects each item into the cognitive torus,
 * runs the decision loop for --ticks cycles, and lets the autonomous
 * STORE_MEMORY action lay down durable memory records in LMDB.
 *
 * Usage:
 *   nikola-train --corpus corpus/basic_math.txt --memory-lmdb ~/.nikola/train.lmdb
 *   nikola-train --corpus corpus/physics.jsonl   --ticks 300 --steps 50
 *
 * Options:
 *   --corpus  <file>    Input corpus (plain text or JSONL)  [required]
 *   --ticks   <N>       Max ticks per training item         [200]
 *   --steps   <N>       Physics steps per tick              [50]
 *   --memory-lmdb <p>   LMDB memory path                   ["nikola_train.lmdb"]
 *   --vocab   <file>    Extra vocabulary words              []
 *   --no-emit           Suppress thought output (train only)
 *   --quiet             Suppress all non-result output
 *   --json-out          Structured JSON per item
 *   --dry-run           Parse corpus, skip injection (count items)
 *
 * Phase: NIK-TR-01 (Training Pipeline, Phase 138)
 */

#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/autonomy/decision_loop.hpp>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// ANSI helpers (minimal inline, no dependency on nikola_run)
// ─────────────────────────────────────────────────────────────────────────────
namespace ansi {
    static bool enabled = true;
    static const char* rst  = "\033[0m";
    static const char* bold = "\033[1m";
    static const char* dim  = "\033[2m";
    static const char* gray = "\033[90m";
    static const char* cyan = "\033[36m";
    static const char* green= "\033[32m";
    static std::string c(const char* code) {
        return enabled ? code : "";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI config
// ─────────────────────────────────────────────────────────────────────────────
struct TrainConfig {
    std::string corpus_path;
    std::string memory_lmdb  = "nikola_train.lmdb";
    std::string vocab_path;
    std::string model_path;
    std::string tokenizer_path;
    int max_ticks            = 200;
    int steps_per_tick       = 50;
    bool no_emit             = false;
    bool quiet               = false;
    bool json_out            = false;
    bool dry_run             = false;
    bool no_color            = false;
};

static void print_help(const char* argv0)
{
    std::cout
        << "\nUsage: " << argv0 << " [options]\n\n"
        << "  --corpus  <file>    Training corpus (plain text or JSONL)  [required]\n"
        << "  --ticks   <N>       Max decision ticks per item            [200]\n"
        << "  --steps   <N>       Physics steps per tick                 [50]\n"
        << "  --memory-lmdb <p>   LMDB memory output path               [nikola_train.lmdb]\n"
        << "  --vocab   <file>    Extra vocabulary words (one per line)  []\n"
        << "  --no-emit           Suppress thought output during training\n"
        << "  --dry-run           Count corpus items without running\n"
        << "  --quiet             Suppress all non-critical output\n"
        << "  --json-out          Machine-readable output per item\n"
        << "  --no-color          Disable ANSI colour codes\n"
        << "  --help              Show this help\n\n"
        << "Corpus formats:\n"
        << "  plain text:  one sentence per line (blank lines skipped)\n"
        << "  JSONL:       one JSON object per line, field \"text\" is used\n\n"
        << "Example:\n"
        << "  " << argv0 << " --corpus corpus/basic_math.txt"
               " --memory-lmdb ~/.nikola/math.lmdb --ticks 300\n\n";
}

static std::optional<TrainConfig> parse_args(int argc, char** argv)
{
    TrainConfig cfg;

    // Locate ONNX model (same logic as nikola-run)
    const char* home = std::getenv("HOME");
    std::string default_model_base;
    if (home) {
        default_model_base = std::string(home) +
            "/Workspace/SYSTEM/onnxruntime/bert-tiny-onnx";
    }
    cfg.model_path     = default_model_base + "/model.onnx";
    cfg.tokenizer_path = default_model_base;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << a << " requires an argument\n";
                std::exit(1);
            }
            return argv[++i];
        };

        if (a == "--help" || a == "-h") { print_help(argv[0]); return std::nullopt; }
        else if (a == "--corpus")       cfg.corpus_path    = next();
        else if (a == "--ticks")        cfg.max_ticks      = std::stoi(next());
        else if (a == "--steps")        cfg.steps_per_tick = std::stoi(next());
        else if (a == "--memory-lmdb")  cfg.memory_lmdb    = next();
        else if (a == "--vocab")        cfg.vocab_path     = next();
        else if (a == "--no-emit")      cfg.no_emit        = true;
        else if (a == "--quiet")        cfg.quiet          = true;
        else if (a == "--json-out")     cfg.json_out       = true;
        else if (a == "--dry-run")      cfg.dry_run        = true;
        else if (a == "--no-color")     cfg.no_color       = true;
        else if (a == "--model")        cfg.model_path     = next();
        else if (a == "--tokenizer")    cfg.tokenizer_path = next();
        else {
            std::cerr << "Unknown option: " << a << "\n";
            print_help(argv[0]);
            return std::nullopt;
        }
    }

    if (cfg.corpus_path.empty()) {
        std::cerr << "Error: --corpus is required\n\n";
        print_help(argv[0]);
        return std::nullopt;
    }

    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// Corpus loading
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal JSON field extractor — finds "text": "..." without a full JSON parser.
/// Returns empty string if no "text" field found.
static std::string extract_json_text(const std::string& line)
{
    // Find "text" key
    const auto key_pos = line.find("\"text\"");
    if (key_pos == std::string::npos) return "";

    // Find the colon after the key
    auto colon = line.find(':', key_pos + 6);
    if (colon == std::string::npos) return "";

    // Skip whitespace after the colon
    auto val_start = line.find('"', colon + 1);
    if (val_start == std::string::npos) return "";

    ++val_start; // skip opening quote
    std::string result;
    bool escaped = false;
    for (size_t i = val_start; i < line.size(); ++i) {
        const char c = line[i];
        if (escaped) {
            switch (c) {
                case '"':  result += '"'; break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                default:   result += c; break;
            }
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (c == '"') {
            break; // end of string
        } else {
            result += c;
        }
    }
    return result;
}

/// Load corpus lines from file.
/// Supports plain text (one sentence per line) and JSONL (extracts "text" field).
static std::vector<std::string> load_corpus(const std::string& path)
{
    std::vector<std::string> items;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open corpus: " << path << "\n";
        return items;
    }

    std::string line;
    while (std::getline(f, line)) {
        // Trim whitespace
        auto s = line.find_first_not_of(" \t\r\n");
        auto e = line.find_last_not_of(" \t\r\n");
        if (s == std::string::npos) continue;
        line = line.substr(s, e - s + 1);
        if (line.empty() || line[0] == '#') continue;

        // JSONL detection: line starts with '{'
        if (line[0] == '{') {
            const auto text = extract_json_text(line);
            if (!text.empty()) items.push_back(text);
            // No text field → skip silently
        } else {
            items.push_back(line);
        }
    }

    return items;
}

// ─────────────────────────────────────────────────────────────────────────────
// Extra vocabulary
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<std::string> load_vocab_file(const std::string& path)
{
    std::vector<std::string> words;
    std::ifstream f(path);
    if (!f.is_open()) { return words; }
    std::string line;
    while (std::getline(f, line)) {
        auto s = line.find_first_not_of(" \t\r\n");
        auto e = line.find_last_not_of(" \t\r\n");
        if (s == std::string::npos) continue;
        line = line.substr(s, e - s + 1);
        if (line.empty() || line[0] == '#') continue;
        words.push_back(line);
    }
    return words;
}

// ─────────────────────────────────────────────────────────────────────────────
// Default vocabulary (same as nikola-run Phase 138 prep)
// ─────────────────────────────────────────────────────────────────────────────
static const std::vector<std::string> DEFAULT_VOCABULARY = {
    // Numbers
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "hundred", "thousand", "million", "infinity",
    // Arithmetic
    "plus", "minus", "times", "divided", "equals", "sum", "product",
    "difference", "number", "calculate", "result", "count", "total",
    "greater", "smaller", "equal", "less", "ratio", "factor", "prime",
    "integer", "fraction", "probability", "proportion",
    // Self / identity
    "nikola", "mind", "self", "thought", "consciousness", "awareness",
    "intelligence", "machine", "neural", "system", "architect", "cognition",
    "identity", "memory", "purpose", "belief", "artificial", "construct",
    "abstract", "model",
    // Cognitive verbs
    "think", "feel", "know", "learn", "understand", "explore", "wonder",
    "perceive", "observe", "remember", "imagine", "create", "discover",
    "reason", "analyze", "solve", "predict", "generate", "express",
    "transform", "synthesize", "sense", "recognize", "reflect",
    // Concepts
    "meaning", "concept", "idea", "knowledge", "information", "pattern",
    "signal", "symbol", "truth", "reality", "experience", "representation",
    "category", "relation", "context", "insight", "inference", "logic",
    "structure", "definition",
    // Physics
    "energy", "field", "wave", "frequency", "space", "time", "force",
    "matter", "light", "motion", "dimension", "quantum", "torus",
    "resonance", "interference", "harmonic", "entropy", "gravity",
    "electron", "photon", "particle", "universe", "cosmos", "chaos",
    "order", "symmetry", "topology",
    // Qualities
    "true", "false", "possible", "certain", "unknown", "complex", "simple",
    "different", "similar", "important", "strange", "stable", "dynamic",
    "emergent", "recursive", "infinite", "bounded", "continuous", "discrete",
    // Transformations
    "grow", "build", "connect", "combine", "begin", "end", "increase",
    "decrease", "evolve", "emerge", "change", "expand", "collapse",
    "activate", "decode", "encode", "compute", "integrate", "diverge",
    // Emotional states
    "curious", "uncertain", "excited", "calm", "lost", "clear", "anxious",
    "peaceful", "active", "quiet", "focused", "confused", "inspired",
    "empty", "full",
    // Language
    "word", "language", "sentence", "describe", "define", "symbol",
    "metaphor", "question", "answer", "voice", "grammar", "narrative",
    // World / science
    "life", "human", "world", "nature", "physics", "math", "science",
    "biology", "chemistry", "computation", "network", "algorithm",
};

// ─────────────────────────────────────────────────────────────────────────────
// JSON helpers
// ─────────────────────────────────────────────────────────────────────────────
static std::string json_escape(const std::string& s)
{
    std::string r;
    for (char c : s) {
        if (c == '"')       r += "\\\"";
        else if (c == '\\') r += "\\\\";
        else if (c == '\n') r += "\\n";
        else                r += c;
    }
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Training loop for one corpus item
// ─────────────────────────────────────────────────────────────────────────────
struct ItemResult {
    std::string text;
    std::string thought;    // Best EMIT_THOUGHT (may be empty)
    int memories_stored;    // Number of STORE_MEMORY events
    int ticks_used;
    double elapsed_ms;
};

static ItemResult train_item(nikola::autonomy::DecisionLoop& loop,
                              const std::string& text,
                              const TrainConfig& cfg)
{
    using namespace nikola::autonomy;

    ItemResult res;
    res.text = text;
    res.memories_stored = 0;
    res.ticks_used = 0;

    auto t0 = std::chrono::steady_clock::now();

    loop.inject_stimulus(text);

    for (int t = 0; t < cfg.max_ticks; ++t) {
        auto r = loop.tick();
        ++res.ticks_used;

        if (r.type == ActionType::STORE_MEMORY) {
            ++res.memories_stored;
        }
        if (r.type == ActionType::EMIT_THOUGHT && res.thought.empty()) {
            if (!cfg.no_emit) res.thought = r.payload;
        }

        // Once we've stored at least one memory AND emitted at least one
        // thought (or no_emit is set), training for this item is done.
        if (res.memories_stored >= 1 &&
            (cfg.no_emit || !res.thought.empty())) {
            break;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    auto maybe_cfg = parse_args(argc, argv);
    if (!maybe_cfg) return 0;
    TrainConfig& cfg = *maybe_cfg;

    if (cfg.no_color) ansi::enabled = false;

    // ── Load corpus ──────────────────────────────────────────────────────────
    const auto corpus = load_corpus(cfg.corpus_path);
    if (corpus.empty()) {
        std::cerr << "Error: no training items found in " << cfg.corpus_path << "\n";
        return 1;
    }

    if (!cfg.quiet) {
        std::cerr << ansi::c(ansi::bold) << ansi::c(ansi::cyan)
                  << "nikola-train" << ansi::c(ansi::rst)
                  << ansi::c(ansi::dim) << "  v0.0.1  |  Corpus → Holographic Memory\n"
                  << ansi::c(ansi::rst);
        std::cerr << ansi::c(ansi::gray)
                  << "  corpus:  " << cfg.corpus_path
                  << " (" << corpus.size() << " items)\n"
                  << "  memory:  " << cfg.memory_lmdb << "\n"
                  << "  ticks:   " << cfg.max_ticks << " per item\n"
                  << ansi::c(ansi::rst) << "\n";
    }

    if (cfg.dry_run) {
        std::cout << corpus.size() << " items in " << cfg.corpus_path << "\n";
        return 0;
    }

    // ── Build pipeline ───────────────────────────────────────────────────────
    nikola::cognitive::CognitiveTorus torus(3,
                                             cfg.tokenizer_path,
                                             cfg.model_path);
    nikola::autonomy::AutonomyEngine engine;

    nikola::autonomy::DecisionLoopConfig loop_cfg;
    loop_cfg.steps_per_tick          = cfg.steps_per_tick;
    loop_cfg.tokenizer_json_path     = cfg.tokenizer_path;
    loop_cfg.transformer_model_path  = cfg.model_path;
    loop_cfg.lmdb_memory_path        = cfg.memory_lmdb;
    loop_cfg.min_emit_interval_s     = 0.0f;
    loop_cfg.min_store_interval_s    = 0.0f;  // store after every item, no cooldown

    // Vocabulary
    loop_cfg.vocabulary = DEFAULT_VOCABULARY;
    if (!cfg.vocab_path.empty()) {
        const auto extra = load_vocab_file(cfg.vocab_path);
        for (const auto& w : extra) {
            if (std::find(loop_cfg.vocabulary.begin(),
                          loop_cfg.vocabulary.end(), w)
                    == loop_cfg.vocabulary.end()) {
                loop_cfg.vocabulary.push_back(w);
            }
        }
    }

    if (!cfg.quiet) {
        std::cerr << ansi::c(ansi::gray)
                  << "  vocabulary: " << loop_cfg.vocabulary.size() << " words\n"
                  << ansi::c(ansi::rst);
    }

    nikola::autonomy::DecisionLoop loop(torus, engine, loop_cfg);

    // ── Training loop ────────────────────────────────────────────────────────
    int total_memories = 0;
    int items_with_memory = 0;
    int items_with_thought = 0;
    double total_ms = 0.0;

    for (size_t i = 0; i < corpus.size(); ++i) {
        const auto res = train_item(loop, corpus[i], cfg);

        total_memories      += res.memories_stored;
        total_ms            += res.elapsed_ms;
        if (res.memories_stored > 0) ++items_with_memory;
        if (!res.thought.empty())    ++items_with_thought;

        if (cfg.json_out) {
            std::cout << "{"
                      << "\"i\":" << i
                      << ",\"text\":\"" << json_escape(res.text) << "\""
                      << ",\"thought\":\"" << json_escape(res.thought) << "\""
                      << ",\"memories\":" << res.memories_stored
                      << ",\"ticks\":" << res.ticks_used
                      << ",\"ms\":" << std::fixed << std::setprecision(1)
                                    << res.elapsed_ms
                      << "}\n";
        } else if (!cfg.quiet) {
            // Progress: [001/050] text... → thought  (Nm, Xt)
            const std::string trunc_text =
                res.text.size() > 48 ? res.text.substr(0, 45) + "..." : res.text;
            std::cerr << ansi::c(ansi::gray)
                      << "[" << std::setfill('0') << std::setw(3) << (i + 1)
                      << "/" << std::setw(3) << corpus.size() << "] "
                      << ansi::c(ansi::rst)
                      << trunc_text;
            if (!res.thought.empty()) {
                std::cerr << ansi::c(ansi::green)
                          << " → " << res.thought
                          << ansi::c(ansi::rst);
            }
            if (res.memories_stored > 0) {
                std::cerr << ansi::c(ansi::cyan)
                          << " [+" << res.memories_stored << "m]"
                          << ansi::c(ansi::rst);
            }
            std::cerr << "\n";
        }
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    if (!cfg.quiet && !cfg.json_out) {
        const double avg_ms = corpus.empty() ? 0.0 : total_ms / corpus.size();
        std::cerr << "\n" << ansi::c(ansi::bold) << "Training complete" << ansi::c(ansi::rst) << "\n"
                  << ansi::c(ansi::gray)
                  << "  items:    " << corpus.size() << "\n"
                  << "  memories: " << total_memories
                  << " (" << items_with_memory << "/" << corpus.size() << " items stored)\n"
                  << "  thoughts: " << items_with_thought << "/" << corpus.size() << " items emitted\n"
                  << "  avg time: " << std::fixed << std::setprecision(1) << avg_ms << "ms/item\n"
                  << "  lmdb:     " << cfg.memory_lmdb << "\n"
                  << ansi::c(ansi::rst);
    } else if (!cfg.json_out) {
        std::cout << corpus.size() << " items  "
                  << total_memories << " memories  "
                  << cfg.memory_lmdb << "\n";
    }

    return 0;
}
