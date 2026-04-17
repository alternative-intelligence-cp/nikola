/**
 * @file src/nikola_infer.cpp
 * @brief nikola-infer — Lightweight inference CLI (no autonomy).
 *
 * Runs the Nikola inference pipeline (Torus → Mamba → NPT → decode → compose)
 * without the full autonomy/scoring/personality stack.  Also serves an optional
 * HTTP API for external integration.
 *
 * USAGE
 * ─────
 *   # Single prompt:
 *   nikola-infer --prompt "What is consciousness?"
 *
 *   # With checkpoint:
 *   nikola-infer --checkpoint model.nik --prompt "Hello"
 *
 *   # Interactive REPL:
 *   nikola-infer --interactive
 *
 *   # Batch mode:
 *   nikola-infer --batch < prompts.txt
 *
 *   # HTTP API server:
 *   nikola-infer --serve --port 8080
 *
 * FLAGS
 * ─────
 *   --prompt <text>      Input text for single-shot mode
 *   --checkpoint <path>  Load .nik checkpoint file
 *   --ticks  <N>         Max ticks per inference (default: 200)
 *   --steps  <N>         Torus steps per tick    (default: 50)
 *   --interactive        REPL mode
 *   --batch              One prompt per stdin line
 *   --serve              Start HTTP API server
 *   --port  <N>          HTTP port (default: 8080)
 *   --model  <path>      Override ONNX model path
 *   --tokenizer <path>   Override tokenizer.json path
 *   --vocab  <file>      Load extra vocabulary words
 *   --stream             Print thoughts as they emerge
 *   --no-npt             Disable NPT reasoning pass
 *   --gpu                Force GPU propagation
 *   --no-gpu             Force CPU propagation
 *   --no-color           Disable ANSI colour
 *   --quiet              Suppress status lines
 *   --json               Machine-readable JSON output
 *   --help               Show usage
 *
 * v0.2.5
 */

#include <nikola/inference/nikola_inference.hpp>
#include <nikola/inference/http_server.hpp>
#include <nikola/diag/scope_profiler.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// ANSI helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace ansi {
    static bool enabled = true;
    enum Code { rst=0, bold=1, dim=2, red=31, green=32, yellow=33, blue=34,
                cyan=36, gray=90 };
    inline std::string c(Code code) {
        if (!enabled) return {};
        return "\033[" + std::to_string(static_cast<int>(code)) + "m";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI config
// ─────────────────────────────────────────────────────────────────────────────
struct CliConfig {
    std::string prompt;
    std::string checkpoint_path;
    std::string model_path      = NIKOLA_ORT_MODEL_PATH;
    std::string tokenizer_path  = NIKOLA_ORT_TOKENIZER_PATH;
    std::string vocab_path;
    int         max_ticks       = 200;
    int         steps_per_tick  = 50;
    int         port            = 8080;
    bool        interactive     = false;
    bool        batch           = false;
    bool        serve           = false;
    bool        stream          = false;
    bool        enable_npt      = true;
    bool        gpu             = true;
    bool        no_color        = false;
    bool        quiet           = false;
    bool        json_out        = false;
    bool        profile         = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// Default vocabulary (same as nikola_run.cpp)
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
    // Concepts / epistemology
    "meaning", "concept", "idea", "knowledge", "information", "pattern",
    "signal", "symbol", "truth", "reality", "experience", "representation",
    "category", "relation", "context", "insight", "inference", "logic",
    "structure", "definition",
    // Physics / field theory
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
    // Emotional / experiential
    "curious", "uncertain", "excited", "calm", "lost", "clear", "anxious",
    "peaceful", "active", "quiet", "focused", "confused", "inspired",
    "empty", "full",
    // Language
    "word", "language", "sentence", "describe", "define", "symbol",
    "metaphor", "question", "answer", "voice", "meaning", "grammar",
    "narrative", "logic",
    // World / science
    "life", "human", "world", "nature", "physics", "math", "science",
    "biology", "chemistry", "computation", "network", "algorithm",
};

/// Load supplemental words from a plain-text file (one word per line).
static std::vector<std::string> load_vocab_file(const std::string& path) {
    std::vector<std::string> words;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Warning: could not open vocab file: " << path << "\n";
        return words;
    }
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
// Usage / arg parsing
// ─────────────────────────────────────────────────────────────────────────────
static void print_help(const char* prog) {
    std::cout
        << ansi::c(ansi::bold) << "nikola-infer"
        << ansi::c(ansi::rst) << " — Lightweight Nikola inference (no autonomy)\n\n"
        << "USAGE\n  " << prog << " [OPTIONS]\n\n"
        << "OPTIONS\n"
        << "  --prompt <text>      Input text for single-shot mode\n"
        << "  --checkpoint <path>  Load .nik checkpoint file\n"
        << "  --ticks  <N>         Max ticks per inference [200]\n"
        << "  --steps  <N>         Torus steps per tick    [50]\n"
        << "  --interactive        REPL mode\n"
        << "  --batch              Read one prompt per stdin line\n"
        << "  --serve              Start HTTP API server\n"
        << "  --port  <N>          HTTP port [8080]\n"
        << "  --model  <path>      Override ONNX model path\n"
        << "  --tokenizer <path>   Override tokenizer.json path\n"
        << "  --vocab  <file>      Load extra vocabulary words\n"
        << "  --stream             Print thoughts as they emerge\n"
        << "  --no-npt             Disable NPT reasoning pass\n"
        << "  --gpu                Force GPU propagation\n"
        << "  --no-gpu             Disable CUDA, force CPU\n"
        << "  --no-color           Disable ANSI colours\n"
        << "  --quiet              Suppress status lines\n"
        << "  --json               Machine-readable JSON output\n"
        << "  --profile            Print scope profiler report at exit\n"
        << "  --help               Show this message\n\n";
}

static std::optional<CliConfig> parse_args(int argc, char** argv) {
    CliConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << a << " requires an argument\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if      (a == "--help")         { print_help(argv[0]); return std::nullopt; }
        else if (a == "--interactive")   cfg.interactive      = true;
        else if (a == "--batch")         cfg.batch            = true;
        else if (a == "--serve")         cfg.serve            = true;
        else if (a == "--stream")        cfg.stream           = true;
        else if (a == "--no-npt")        cfg.enable_npt       = false;
        else if (a == "--gpu")           cfg.gpu              = true;
        else if (a == "--no-gpu")        cfg.gpu              = false;
        else if (a == "--no-color")      cfg.no_color         = true;
        else if (a == "--quiet")         cfg.quiet            = true;
        else if (a == "--json")          cfg.json_out         = true;
        else if (a == "--profile")       cfg.profile          = true;
        else if (a == "--prompt")        cfg.prompt           = next();
        else if (a == "--checkpoint")    cfg.checkpoint_path  = next();
        else if (a == "--model")         cfg.model_path       = next();
        else if (a == "--tokenizer")     cfg.tokenizer_path   = next();
        else if (a == "--vocab")         cfg.vocab_path       = next();
        else if (a == "--ticks")         cfg.max_ticks        = std::stoi(next());
        else if (a == "--steps")         cfg.steps_per_tick   = std::stoi(next());
        else if (a == "--port")          cfg.port             = std::stoi(next());
        else {
            std::cerr << "Unknown option: " << a << "  (try --help)\n";
            std::exit(1);
        }
    }
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON helpers
// ─────────────────────────────────────────────────────────────────────────────
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if      (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else                out += c;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Inference helpers
// ─────────────────────────────────────────────────────────────────────────────
static std::string run_prompt(
    nikola::inference::NikolaInference& engine,
    const std::string&                  prompt,
    const CliConfig&                    cfg)
{
    engine.inject(prompt);

    std::string result;
    int thinking_dots = 0;
    auto t0 = std::chrono::steady_clock::now();

    if (cfg.stream) {
        // Streaming mode: emit each thought as it appears.
        for (int t = 0; t < cfg.max_ticks; ++t) {
            auto r = engine.tick();
            if (!r.thought.empty()) {
                if (cfg.json_out) {
                    std::cout << "{\"tick\":" << r.tick
                              << ",\"thought\":\"" << json_escape(r.thought)
                              << "\",\"energy\":" << r.energy
                              << "}\n" << std::flush;
                } else {
                    std::cout << r.thought << "\n" << std::flush;
                }
                result = r.thought;
            }
        }
    } else {
        // Non-streaming: return first thought.
        for (int t = 0; t < cfg.max_ticks; ++t) {
            auto r = engine.tick();

            if (!cfg.quiet && !cfg.json_out && !cfg.batch) {
                ++thinking_dots;
                if (thinking_dots % 10 == 0) {
                    std::cerr << ansi::c(ansi::gray) << "." << ansi::c(ansi::rst) << std::flush;
                }
            }

            if (!r.thought.empty()) {
                result = r.thought;
                break;
            }
        }
    }

    if (!cfg.quiet && !cfg.json_out && !cfg.batch && thinking_dots >= 10)
        std::cerr << "\n";

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!cfg.quiet && !cfg.json_out) {
        std::cerr << ansi::c(ansi::gray) << "  ["
                  << engine.tick_count() << " ticks / "
                  << std::fixed << std::setprecision(0) << ms << "ms]"
                  << ansi::c(ansi::rst) << "\n";
    }

    return result;
}

static void print_result(const std::string& prompt,
                         const std::string& thought,
                         const CliConfig&   cfg)
{
    if (cfg.json_out) {
        std::cout << "{\"prompt\":\"" << json_escape(prompt)
                  << "\",\"thought\":\"" << json_escape(thought)
                  << "\"}\n" << std::flush;
        return;
    }

    if (thought.empty()) {
        if (!cfg.quiet)
            std::cout << ansi::c(ansi::yellow)
                      << "  (silent — no thought emitted within "
                      << cfg.max_ticks << " ticks)"
                      << ansi::c(ansi::rst) << "\n";
        return;
    }

    if (!cfg.quiet) {
        std::cout << ansi::c(ansi::bold) << ansi::c(ansi::cyan)
                  << "Nikola: " << ansi::c(ansi::rst);
    }
    std::cout << thought << "\n" << std::flush;
}

// ─────────────────────────────────────────────────────────────────────────────
// Interactive REPL
// ─────────────────────────────────────────────────────────────────────────────
static void run_repl(nikola::inference::NikolaInference& engine,
                     const CliConfig& cfg)
{
    if (!cfg.quiet) {
        std::cout << ansi::c(ansi::bold) << ansi::c(ansi::cyan)
                  << "\n  Nikola inference session"
                  << ansi::c(ansi::rst)
                  << ansi::c(ansi::dim)
                  << "  (Ctrl-D or 'exit' to quit)\n\n"
                  << ansi::c(ansi::rst);
    }

    std::string line;
    while (true) {
        if (!cfg.quiet && !cfg.json_out)
            std::cout << ansi::c(ansi::green) << "> " << ansi::c(ansi::rst) << std::flush;

        if (!std::getline(std::cin, line)) break;

        auto start = line.find_first_not_of(" \t\r\n");
        auto end   = line.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start, end - start + 1);

        if (line == "exit" || line == "quit") break;

        auto thought = run_prompt(engine, line, cfg);
        print_result(line, thought, cfg);
    }

    if (!cfg.quiet)
        std::cout << ansi::c(ansi::dim) << "\n  Session ended.\n" << ansi::c(ansi::rst);
}

// ─────────────────────────────────────────────────────────────────────────────
// Signal handling for clean shutdown
// ─────────────────────────────────────────────────────────────────────────────
static std::atomic<bool> g_shutdown{false};
static void signal_handler(int) { g_shutdown.store(true); }

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    auto maybe_cfg = parse_args(argc, argv);
    if (!maybe_cfg) return 0;

    CliConfig& cfg = *maybe_cfg;
    if (cfg.no_color) ansi::enabled = false;

    // ── Header ──────────────────────────────────────────────────────────────
    if (!cfg.quiet && !cfg.json_out) {
        std::cerr << ansi::c(ansi::bold) << ansi::c(ansi::blue)
                  << "nikola-infer" << ansi::c(ansi::rst)
                  << ansi::c(ansi::dim)
                  << "  v0.2.5  |  Lightweight Inference (no autonomy)\n"
                  << ansi::c(ansi::rst);
    }

    // ── Build inference engine ──────────────────────────────────────────────
    nikola::inference::InferenceConfig icfg;
    icfg.grid_n          = 3;
    icfg.steps_per_tick  = cfg.steps_per_tick;
    icfg.enable_npt      = cfg.enable_npt;
    icfg.enable_gpu      = cfg.gpu;
    icfg.tokenizer_json_path = cfg.tokenizer_path;
    icfg.model_path          = cfg.model_path;

    // Vocabulary: built-in default + optional file.
    icfg.vocabulary = DEFAULT_VOCABULARY;
    if (!cfg.vocab_path.empty()) {
        const auto extra = load_vocab_file(cfg.vocab_path);
        for (const auto& w : extra) {
            if (std::find(icfg.vocabulary.begin(), icfg.vocabulary.end(), w)
                    == icfg.vocabulary.end()) {
                icfg.vocabulary.push_back(w);
            }
        }
    }

    if (!cfg.quiet && !cfg.json_out) {
        std::cerr << ansi::c(ansi::gray)
                  << "  vocabulary: " << icfg.vocabulary.size() << " words\n"
                  << "  NPT:        " << (icfg.enable_npt ? "enabled" : "disabled") << "\n"
                  << ansi::c(ansi::rst);
    }

    nikola::inference::NikolaInference engine(icfg);

    // ── Load checkpoint if provided ─────────────────────────────────────────
    if (!cfg.checkpoint_path.empty()) {
        if (!engine.load_checkpoint(cfg.checkpoint_path)) {
            std::cerr << "Error: failed to load checkpoint\n";
            return 1;
        }
    }

    // ── GPU status ──────────────────────────────────────────────────────────
    if (!cfg.quiet && !cfg.json_out) {
        std::cerr << ansi::c(ansi::gray)
                  << "  propagator: " << (engine.torus().gpu_enabled() ? "GPU (CUDA)" : "CPU")
                  << "\n" << ansi::c(ansi::rst);
    }

    // ── Warm-up: one tick to populate caches ────────────────────────────────
    engine.warmup();

    if (!cfg.quiet && !cfg.json_out) {
        std::cerr << ansi::c(ansi::gray) << "  ready.\n\n" << ansi::c(ansi::rst);
    }

    // ── HTTP server mode ────────────────────────────────────────────────────
    if (cfg.serve) {
        std::signal(SIGINT,  signal_handler);
        std::signal(SIGTERM, signal_handler);

        if (!cfg.quiet)
            std::cerr << ansi::c(ansi::green)
                      << "  HTTP API listening on port " << cfg.port << "\n"
                      << "  Endpoints: POST /v1/generate, POST /v1/embed, GET /v1/health\n"
                      << "  Press Ctrl-C to stop.\n"
                      << ansi::c(ansi::rst) << "\n";

        nikola::inference::HttpServer server(engine, cfg.port);
        server.run(g_shutdown);
        return 0;
    }

    // ── Interactive mode ────────────────────────────────────────────────────
    if (cfg.interactive) {
        run_repl(engine, cfg);
        return 0;
    }

    // ── Batch mode ──────────────────────────────────────────────────────────
    if (cfg.batch) {
        std::string line;
        while (std::getline(std::cin, line)) {
            auto s = line.find_first_not_of(" \t\r\n");
            auto e = line.find_last_not_of(" \t\r\n");
            if (s == std::string::npos) continue;
            line = line.substr(s, e - s + 1);
            if (line.empty()) continue;

            auto thought = run_prompt(engine, line, cfg);
            print_result(line, thought, cfg);
        }
        return 0;
    }

    // ── Single prompt mode ──────────────────────────────────────────────────
    std::string prompt = cfg.prompt;
    if (prompt.empty()) {
        // Read from stdin.
        std::string buf;
        while (std::getline(std::cin, buf))
            prompt += buf + "\n";
        // Trim trailing newline.
        while (!prompt.empty() && (prompt.back() == '\n' || prompt.back() == '\r'))
            prompt.pop_back();
    }

    if (prompt.empty()) {
        std::cerr << "No prompt provided. Use --prompt, --interactive, --batch, or pipe stdin.\n";
        return 1;
    }

    auto thought = run_prompt(engine, prompt, cfg);
    print_result(prompt, thought, cfg);

    // ── Profile report ──────────────────────────────────────────────────────
    if (cfg.profile) {
        auto report = nikola::diag::ScopeProfiler::global().report();
        std::sort(report.begin(), report.end(),
                  [](const auto& a, const auto& b){ return a.total_us > b.total_us; });
        std::cerr << "\n--- Profile report (" << report.size() << " scopes) ---\n";
        for (const auto& s : report) {
            std::cerr << "  " << s.name
                      << "  n="    << s.count
                      << "  mean=" << std::fixed << std::setprecision(2)
                      << s.mean_us() << "us"
                      << "  min="  << s.min_us  << "us"
                      << "  max="  << s.max_us  << "us\n";
        }
    }

    return thought.empty() ? 1 : 0;
}
