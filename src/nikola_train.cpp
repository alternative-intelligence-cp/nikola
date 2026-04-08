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
 * Phase: NIK-TR-02 (Training Pipeline v2 — EqProp + Metrics)
 */

#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/cognitive/plasticity.hpp>
#include <nikola/spatial/topology_manager.hpp>

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
    std::string metric_out   = "nikola_metric.bin";
    std::string metric_in;
    std::string vocab_path;
    std::string model_path;
    std::string tokenizer_path;
    int max_ticks            = 200;
    int steps_per_tick       = 50;
    int epochs               = 1;
    int limit                = 0;      // 0 = no limit
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
        << "  --epochs  <N>       Training epochs over full corpus       [1]\n"
        << "  --memory-lmdb <p>   LMDB memory output path               [nikola_train.lmdb]\n"
        << "  --metric-out <p>    Save trained metric to file            [nikola_metric.bin]\n"
        << "  --metric-in  <p>    Load initial metric from file          []\n"
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
               " --memory-lmdb ~/.nikola/math.lmdb --ticks 300 --epochs 3\n\n";
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
        else if (a == "--metric-out")   cfg.metric_out     = next();
        else if (a == "--metric-in")    cfg.metric_in      = next();
        else if (a == "--vocab")        cfg.vocab_path     = next();
        else if (a == "--epochs")       cfg.epochs         = std::stoi(next());
        else if (a == "--limit")        cfg.limit          = std::stoi(next());
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
// Training metrics
// ─────────────────────────────────────────────────────────────────────────────
struct TrainingMetrics {
    double loss       = 0.0;  ///< EqProp energy diff E⁺ − E⁻
    double energy_pos = 0.0;  ///< Free phase energy
    double energy_neg = 0.0;  ///< Clamped phase energy
    bool   converged  = false;///< E⁻ < E⁺ (physics prefers clamped state)
    float  dopamine   = 0.5f; ///< Post-item dopamine level
    float  atp        = 1.0f; ///< Post-item ATP level
    float  entropy    = 0.0f; ///< Post-item field entropy
    int    memories_stored = 0;
    double elapsed_ms = 0.0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Metric persistence (9×9 = 81 floats)
// ─────────────────────────────────────────────────────────────────────────────
static bool save_metric(const nikola::spatial::TopologyManager& topo,
                        const std::string& path)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.write(reinterpret_cast<const char*>(topo.metric()), 81 * sizeof(float));
    return f.good();
}

static bool load_metric(nikola::spatial::TopologyManager& topo,
                        const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    float g[81];
    f.read(reinterpret_cast<char*>(g), 81 * sizeof(float));
    if (!f.good()) return false;
    topo.set_metric(g);
    topo.validate_metric();
    return true;
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
                              nikola::cognitive::CognitiveTorus& torus,
                              nikola::autonomy::AutonomyEngine& engine,
                              nikola::cognitive::PlasticityEngine& plasticity,
                              const std::string& text,
                              const std::string& contrastive_text,
                              const TrainConfig& cfg,
                              TrainingMetrics& metrics)
{
    using namespace nikola::autonomy;

    ItemResult res;
    res.text = text;
    res.memories_stored = 0;
    res.ticks_used = 0;

    auto t0 = std::chrono::steady_clock::now();

    // ── Inject corpus text + spike dopamine for STORE_MEMORY ─────────────
    loop.inject_stimulus(text);

    for (int t = 0; t < cfg.max_ticks; ++t) {
        // Spike dopamine each tick until a memory is stored.
        // The POSITIVE reward is consumed by engine_.tick() inside loop.tick()
        // and resets to NEUTRAL, so we re-arm it every iteration.
        if (res.memories_stored == 0)
            loop.set_pending_reward(Reward::POSITIVE);

        auto r = loop.tick();
        ++res.ticks_used;

        if (r.type == ActionType::STORE_MEMORY) {
            ++res.memories_stored;
        }
        if (r.type == ActionType::EMIT_THOUGHT && res.thought.empty()) {
            if (!cfg.no_emit) res.thought = r.payload;
        }

        // Once we've stored at least one memory AND emitted at least one
        // thought (or no_emit is set), cognitive pass for this item is done.
        if (res.memories_stored >= 1 &&
            (cfg.no_emit || !res.thought.empty())) {
            break;
        }
    }

    // ── Force-store memory if the scoring loop didn't get to it ──────────
    // EMIT_THOUGHT frequently outcompetes STORE_MEMORY because its score
    // scales with boredom × dopamine × atp.  For training we guarantee
    // every corpus item produces a durable memory record.
    if (res.memories_stored == 0) {
        loop.force_store_wavefield();
        ++res.memories_stored;
    }

    // ── EqProp training step — update the 9×9 Riemannian metric ──────────
    auto inject_input = [&](nikola::physics::WaveFunction& /*w*/) {
        torus.inject_text(text);
    };
    auto inject_target = [&](nikola::physics::WaveFunction& /*w*/) {
        torus.inject_text(contrastive_text);  // Different text → different spatial pattern
    };

    bool conv = plasticity.eqprop().train_step(
        torus.wave_function(), inject_input, inject_target);

    // ── Record metrics ───────────────────────────────────────────────────
    metrics.loss       = plasticity.eqprop().last_energy_diff();
    metrics.energy_pos = plasticity.eqprop().last_energy_positive();
    metrics.energy_neg = plasticity.eqprop().last_energy_negative();
    metrics.converged  = conv;
    metrics.dopamine   = engine.dopamine();
    metrics.atp        = engine.atp();
    metrics.entropy    = engine.entropy();
    metrics.memories_stored = res.memories_stored;

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    metrics.elapsed_ms = res.elapsed_ms;

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
                  << ansi::c(ansi::dim) << "  v0.2.0  |  Corpus → EqProp + Holographic Memory\n"
                  << ansi::c(ansi::rst);
        std::cerr << ansi::c(ansi::gray)
                  << "  corpus:  " << cfg.corpus_path
                  << " (" << corpus.size() << " items)\n"
                  << "  memory:  " << cfg.memory_lmdb << "\n"
                  << "  metric:  " << cfg.metric_out << "\n"
                  << "  ticks:   " << cfg.max_ticks << " per item\n"
                  << "  epochs:  " << cfg.epochs << "\n"
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

    // Topology manager + plasticity engine (EqProp + Hebbian)
    nikola::spatial::TopologyManager topo;
    if (!cfg.metric_in.empty()) {
        if (load_metric(topo, cfg.metric_in)) {
            if (!cfg.quiet)
                std::cerr << ansi::c(ansi::gray) << "  metric loaded from: "
                          << cfg.metric_in << "\n" << ansi::c(ansi::rst);
        } else {
            std::cerr << "Warning: could not load metric from "
                      << cfg.metric_in << ", using identity\n";
        }
    }
    nikola::cognitive::EqPropConfig eq_cfg;
    eq_cfg.phase_steps = cfg.steps_per_tick;  // Match physics steps
    nikola::cognitive::PlasticityEngine plasticity(topo, eq_cfg);

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

    // ── Training loop (with epochs) ──────────────────────────────────────────
    int    grand_total_memories   = 0;
    int    grand_items_with_mem   = 0;
    int    grand_items_with_tht   = 0;
    int    grand_converged_items  = 0;
    double grand_total_ms         = 0.0;
    double grand_total_loss       = 0.0;

    const size_t n_items = (cfg.limit > 0)
        ? std::min(static_cast<size_t>(cfg.limit), corpus.size())
        : corpus.size();

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        int    epoch_memories     = 0;
        int    epoch_with_mem     = 0;
        int    epoch_with_tht     = 0;
        int    epoch_converged    = 0;
        double epoch_total_ms     = 0.0;
        double epoch_total_loss   = 0.0;

        if (!cfg.quiet && !cfg.json_out && cfg.epochs > 1) {
            std::cerr << ansi::c(ansi::bold) << "\n── Epoch "
                      << (epoch + 1) << "/" << cfg.epochs
                      << " ──" << ansi::c(ansi::rst) << "\n";
        }

        for (size_t i = 0; i < n_items; ++i) {
            TrainingMetrics metrics;
            // Contrastive: pair each item with the next (wrap-around)
            const std::string& contrastive =
                corpus[(i + 1) % n_items];
            const auto res = train_item(loop, torus, engine, plasticity,
                                        corpus[i], contrastive, cfg, metrics);

            epoch_memories     += res.memories_stored;
            epoch_total_ms     += res.elapsed_ms;
            epoch_total_loss   += metrics.loss;
            if (res.memories_stored > 0) ++epoch_with_mem;
            if (!res.thought.empty())    ++epoch_with_tht;
            if (metrics.converged)       ++epoch_converged;

            if (cfg.json_out) {
                std::cout << "{"
                          << "\"epoch\":" << epoch
                          << ",\"i\":" << i
                          << ",\"text\":\"" << json_escape(res.text) << "\""
                          << ",\"thought\":\"" << json_escape(res.thought) << "\""
                          << ",\"memories\":" << res.memories_stored
                          << ",\"ticks\":" << res.ticks_used
                          << ",\"loss\":" << std::scientific << std::setprecision(6)
                                          << metrics.loss
                          << ",\"E_pos\":" << metrics.energy_pos
                          << ",\"E_neg\":" << metrics.energy_neg
                          << ",\"converged\":" << (metrics.converged ? "true" : "false")
                          << ",\"dopamine\":" << std::fixed << std::setprecision(4)
                                              << metrics.dopamine
                          << ",\"atp\":" << metrics.atp
                          << ",\"entropy\":" << metrics.entropy
                          << ",\"ms\":" << std::fixed << std::setprecision(1)
                                        << res.elapsed_ms
                          << "}\n";
            } else if (!cfg.quiet) {
                // Progress: [001/050] text... → thought  (+1m, L=1.2e-3)
                const size_t global_idx = static_cast<size_t>(epoch) * n_items + i;
                const size_t total_items = static_cast<size_t>(cfg.epochs) * n_items;
                const std::string trunc_text =
                    res.text.size() > 40 ? res.text.substr(0, 37) + "..." : res.text;
                std::cerr << ansi::c(ansi::gray)
                          << "[" << std::setfill('0') << std::setw(4) << (global_idx + 1)
                          << "/" << std::setw(4) << total_items << "] "
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
                // Compact loss indicator
                std::cerr << ansi::c(ansi::dim)
                          << " L=" << std::scientific << std::setprecision(1)
                          << metrics.loss
                          << (metrics.converged ? "*" : "")
                          << ansi::c(ansi::rst)
                          << "\n";
            }
        }

        // ── Per-epoch summary ────────────────────────────────────────────────
        grand_total_memories  += epoch_memories;
        grand_items_with_mem  += epoch_with_mem;
        grand_items_with_tht  += epoch_with_tht;
        grand_converged_items += epoch_converged;
        grand_total_ms        += epoch_total_ms;
        grand_total_loss      += epoch_total_loss;

        if (!cfg.quiet && !cfg.json_out && cfg.epochs > 1) {
            const double avg_loss = n_items == 0 ? 0.0 : epoch_total_loss / static_cast<double>(n_items);
            const double avg_ms   = n_items == 0 ? 0.0 : epoch_total_ms / static_cast<double>(n_items);
            const double conv_pct = n_items == 0 ? 0.0
                : 100.0 * static_cast<double>(epoch_converged) / static_cast<double>(n_items);
            std::cerr << ansi::c(ansi::gray)
                      << "  epoch " << (epoch + 1) << ": "
                      << epoch_with_mem << "/" << n_items << " stored, "
                      << epoch_converged << "/" << n_items << " converged ("
                      << std::fixed << std::setprecision(1) << conv_pct << "%), "
                      << "avg loss=" << std::scientific << std::setprecision(2) << avg_loss
                      << ", avg " << std::fixed << std::setprecision(1) << avg_ms << "ms/item\n"
                      << ansi::c(ansi::rst);
        }
    }

    // ── Save trained metric ──────────────────────────────────────────────────
    if (!cfg.metric_out.empty()) {
        if (save_metric(topo, cfg.metric_out)) {
            if (!cfg.quiet)
                std::cerr << ansi::c(ansi::gray)
                          << "\n  metric saved to: " << cfg.metric_out << "\n"
                          << ansi::c(ansi::rst);
        } else {
            std::cerr << "Warning: could not save metric to " << cfg.metric_out << "\n";
        }
    }

    // ── Final summary ────────────────────────────────────────────────────────
    const size_t total_item_passes = static_cast<size_t>(cfg.epochs) * n_items;
    if (!cfg.quiet && !cfg.json_out) {
        const double avg_ms   = total_item_passes == 0 ? 0.0
            : grand_total_ms / static_cast<double>(total_item_passes);
        const double avg_loss = total_item_passes == 0 ? 0.0
            : grand_total_loss / static_cast<double>(total_item_passes);
        const double conv_pct = total_item_passes == 0 ? 0.0
            : 100.0 * static_cast<double>(grand_converged_items) / static_cast<double>(total_item_passes);
        std::cerr << "\n" << ansi::c(ansi::bold) << "Training complete" << ansi::c(ansi::rst) << "\n"
                  << ansi::c(ansi::gray)
                  << "  corpus:     " << n_items << " items × "
                  << cfg.epochs << " epoch" << (cfg.epochs > 1 ? "s" : "")
                  << " = " << total_item_passes << " passes\n"
                  << "  memories:   " << grand_total_memories
                  << " (" << grand_items_with_mem << "/" << total_item_passes << " stored)\n"
                  << "  thoughts:   " << grand_items_with_tht << "/" << total_item_passes << " emitted\n"
                  << "  converged:  " << grand_converged_items << "/" << total_item_passes
                  << " (" << std::fixed << std::setprecision(1) << conv_pct << "%)\n"
                  << "  avg loss:   " << std::scientific << std::setprecision(4) << avg_loss << "\n"
                  << "  avg time:   " << std::fixed << std::setprecision(1) << avg_ms << "ms/item\n"
                  << "  metric:     " << cfg.metric_out << "\n"
                  << "  lmdb:       " << cfg.memory_lmdb << "\n"
                  << ansi::c(ansi::rst);
    } else if (!cfg.json_out) {
        std::cout << total_item_passes << " passes  "
                  << grand_total_memories << " memories  "
                  << grand_converged_items << " converged  "
                  << cfg.memory_lmdb << "\n";
    }

    return 0;
}
