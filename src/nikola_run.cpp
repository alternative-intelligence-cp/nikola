/**
 * @file src/nikola_run.cpp
 * @brief nikola-run — Nikola inference CLI
 *
 * Feeds text through the full Nikola pipeline (CognitiveTorus → DecisionLoop)
 * and prints the resulting thought(s) to stdout.
 *
 * USAGE
 * ─────
 *   # Single prompt, wait for first EMIT_THOUGHT (up to --ticks ticks):
 *   nikola-run --prompt "What is consciousness?"
 *
 *   # Pipe from stdin (one prompt):
 *   echo "Hello Nikola" | nikola-run
 *
 *   # Read multiple lines from stdin (batch):
 *   nikola-run --batch < prompts.txt
 *
 *   # Interactive REPL (Ctrl-D or 'exit' to quit):
 *   nikola-run --interactive
 *
 *   # Persist memory across runs:
 *   nikola-run --memory ~/.nikola_memory.bin --interactive
 *
 * FLAGS
 * ─────
 *   --prompt <text>      Input text for single-shot mode (default: read stdin)
 *   --ticks  <N>         Max decision ticks before giving up  (default: 200)
 *   --steps  <N>         Torus steps per tick                 (default: 50)
 *   --interactive        REPL mode — type prompts, get thoughts
 *   --batch              Read one prompt per stdin line, print one result each
 *   --memory <path>      SemanticMemory snapshot file (persists across runs)
 *   --model  <path>      Override ONNX model path
 *   --tokenizer <path>   Override tokenizer.json/dir path
 *   --emit-all           Print ALL non-SILENT actions, not just EMIT_THOUGHT
 *   --no-color           Disable ANSI colour output
 *   --quiet              Suppress headers and status lines
 *   --json               Machine-readable JSON output (one object per result)
 *   --help               Show this message
 */

#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/autonomy/autonomy_engine.hpp>
#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/cli/stream_emitter.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Compile-time defaults (overridable at build time, same macros as nikola_core)
// ─────────────────────────────────────────────────────────────────────────────
#ifndef NIKOLA_RUN_MODEL_PATH
#  ifdef NIKOLA_ORT_MODEL_PATH
#    define NIKOLA_RUN_MODEL_PATH NIKOLA_ORT_MODEL_PATH
#  else
#    define NIKOLA_RUN_MODEL_PATH ""
#  endif
#endif

#ifndef NIKOLA_RUN_TOKENIZER_PATH
#  ifdef NIKOLA_ORT_TOKENIZER_PATH
#    define NIKOLA_RUN_TOKENIZER_PATH NIKOLA_ORT_TOKENIZER_PATH
#  else
#    define NIKOLA_RUN_TOKENIZER_PATH ""
#  endif
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ANSI helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace ansi {
    static bool enabled = true;
    static const char* rst   = "\033[0m";
    static const char* bold  = "\033[1m";
    static const char* dim   = "\033[2m";
    static const char* cyan  = "\033[36m";
    static const char* green = "\033[32m";
    static const char* yellow= "\033[33m";
    static const char* blue  = "\033[34m";
    static const char* gray  = "\033[90m";

    inline const char* c(const char* code) { return enabled ? code : ""; }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI config
// ─────────────────────────────────────────────────────────────────────────────
struct CliConfig {
    std::string              prompt;
    std::string              memory_path;
    std::string              model_path    = NIKOLA_RUN_MODEL_PATH;
    std::string              tokenizer_path= NIKOLA_RUN_TOKENIZER_PATH;
    int                      max_ticks     = 200;
    int                      steps_per_tick= 50;
    bool                     interactive   = false;
    bool                     batch         = false;
    bool                     emit_all      = false;
    bool                     no_color      = false;
    bool                     quiet         = false;
    bool                     json_out      = false;
    bool                     stream        = false;  ///< --stream: line-buffered EMIT_THOUGHT during tick loop
};

static void print_help(const char* argv0) {
    std::cout
        << "\nUsage: " << argv0 << " [options]\n\n"
        << "  --prompt <text>      Input text (default: read from stdin)\n"
        << "  --ticks  <N>         Max decision ticks before giving up  [200]\n"
        << "  --steps  <N>         Torus steps per tick                 [50]\n"
        << "  --interactive        REPL mode — type prompts, Ctrl-D to quit\n"
        << "  --batch              Read one prompt per stdin line\n"
        << "  --memory <path>      Persist SemanticMemory across runs\n"
        << "  --model  <path>      Override ONNX model.onnx path\n"
        << "  --tokenizer <path>   Override tokenizer.json/dir path\n"
        << "  --emit-all           Print ALL non-SILENT actions\n"
        << "  --stream             Print each EMIT_THOUGHT immediately (line-buffered)\n"
        << "  --no-color           Disable ANSI colour\n"
        << "  --quiet              Suppress status headers\n"
        << "  --json               Machine-readable JSON output\n"
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
        if      (a == "--help")        { print_help(argv[0]); return std::nullopt; }
        else if (a == "--interactive")  cfg.interactive    = true;
        else if (a == "--batch")        cfg.batch          = true;
        else if (a == "--emit-all")     cfg.emit_all       = true;
        else if (a == "--stream")       cfg.stream         = true;
        else if (a == "--no-color")     cfg.no_color       = true;
        else if (a == "--quiet")        cfg.quiet          = true;
        else if (a == "--json")         cfg.json_out       = true;
        else if (a == "--prompt")       cfg.prompt         = next();
        else if (a == "--memory")       cfg.memory_path    = next();
        else if (a == "--model")        cfg.model_path     = next();
        else if (a == "--tokenizer")    cfg.tokenizer_path = next();
        else if (a == "--ticks")        cfg.max_ticks      = std::stoi(next());
        else if (a == "--steps")        cfg.steps_per_tick = std::stoi(next());
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
// Run one prompt through the pipeline
//
// Returns: the emitted thought (or best-effort payload), or empty string if
//          nothing was emitted within max_ticks.
// ─────────────────────────────────────────────────────────────────────────────
static std::string run_prompt(
    nikola::cognitive::CognitiveTorus& torus,
    nikola::autonomy::DecisionLoop&    loop,
    const std::string&                 prompt,
    const CliConfig&                   cfg)
{
    using namespace nikola::autonomy;

    // Inject text into the torus at t=0 relative to current tick
    if (!prompt.empty()) {
        torus.inject_text(prompt, 0.0);
    }

    // Tick until EMIT_THOUGHT fires, or we hit the limit
    std::string result;
    int thinking_dots = 0;

    auto t0 = std::chrono::steady_clock::now();

    if (cfg.stream) {
        // ── Streaming path ────────────────────────────────────────────────
        // Wire on_action to immediate flushed output; run all max_ticks ticks
        // without early exit so all EMIT_THOUGHT events are surfaced.
        nikola::cli::StreamEmitter emitter(std::cout,
                                           cfg.json_out,
                                           cfg.quiet,
                                           cfg.emit_all);
        loop.on_action = [&](const DecisionResult& r) {
            emitter.emit(r);
            if (emitter.has_output())
                result = emitter.last_payload();
        };

        for (int t = 0; t < cfg.max_ticks; ++t)
            loop.tick();   // emit via on_action callback; no dot progress

        loop.on_action = nullptr;   // reset for next call to run_prompt
    } else {
        // ── Non-streaming path ────────────────────────────────────────────
        // Break on first interesting result (original behaviour).
        for (int t = 0; t < cfg.max_ticks; ++t) {
            auto r = loop.tick();

            // Progress indicator (not in quiet/json/batch mode)
            if (!cfg.quiet && !cfg.json_out && !cfg.batch) {
                ++thinking_dots;
                if (thinking_dots % 10 == 0) {
                    std::cerr << ansi::c(ansi::gray) << "." << ansi::c(ansi::rst) << std::flush;
                }
            }

            bool interesting = (r.type == ActionType::EMIT_THOUGHT) ||
                               (cfg.emit_all && r.type != ActionType::SILENT);

            if (interesting && !r.payload.empty()) {
                result = r.payload;
                break;
            }
        }
    }

    if (!cfg.quiet && !cfg.json_out && !cfg.batch && thinking_dots >= 10) {
        std::cerr << "\n";
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!cfg.quiet && !cfg.json_out) {
        std::cerr << ansi::c(ansi::gray) << "  [" << cfg.max_ticks
                  << " tick ceiling / " << std::fixed << std::setprecision(0)
                  << ms << "ms]" << ansi::c(ansi::rst) << "\n";
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Print one result
// ─────────────────────────────────────────────────────────────────────────────
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
            std::cout << ansi::c(ansi::yellow) << "  (silent — no thought emitted within "
                      << cfg.max_ticks << " ticks)" << ansi::c(ansi::rst) << "\n";
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
static void run_repl(nikola::cognitive::CognitiveTorus& torus,
                     nikola::autonomy::DecisionLoop&    loop,
                     const CliConfig&                   cfg)
{
    if (!cfg.quiet) {
        std::cout << ansi::c(ansi::bold) << ansi::c(ansi::cyan)
                  << "\n  Nikola interactive session"
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

        // Trim whitespace
        auto start = line.find_first_not_of(" \t\r\n");
        auto end   = line.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start, end - start + 1);

        if (line == "exit" || line == "quit") break;

        auto thought = run_prompt(torus, loop, line, cfg);
        print_result(line, thought, cfg);
    }

    if (!cfg.quiet)
        std::cout << ansi::c(ansi::dim) << "\n  Session ended.\n" << ansi::c(ansi::rst);
}

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
                  << "nikola-run" << ansi::c(ansi::rst)
                  << ansi::c(ansi::dim) << "  v0.0.4  |  9D Toroidal Waveform Intelligence\n"
                  << ansi::c(ansi::rst);

        if (!cfg.model_path.empty())
            std::cerr << ansi::c(ansi::gray)
                      << "  model:     " << cfg.model_path << "\n"
                      << "  tokenizer: " << cfg.tokenizer_path << "\n"
                      << ansi::c(ansi::rst);
        else
            std::cerr << ansi::c(ansi::yellow)
                      << "  (no ONNX model — using heuristic thought composer)\n"
                      << ansi::c(ansi::rst);
        std::cerr << "\n";
    }

    // ── Build pipeline ───────────────────────────────────────────────────────
    // CognitiveTorus(n, tok_path, model_path) — n=3 gives 3^9 = 19,683 nodes
    nikola::cognitive::CognitiveTorus torus(3,
                                            cfg.tokenizer_path,
                                            cfg.model_path);

    nikola::autonomy::AutonomyEngine engine;

    nikola::autonomy::DecisionLoopConfig loop_cfg;
    loop_cfg.steps_per_tick        = cfg.steps_per_tick;
    loop_cfg.tokenizer_json_path   = cfg.tokenizer_path;
    loop_cfg.transformer_model_path= cfg.model_path;
    loop_cfg.memory_path           = cfg.memory_path;
    loop_cfg.min_emit_interval_s   = 0.0f;  // CLI: no rate limit between prompts

    nikola::autonomy::DecisionLoop loop(torus, engine, loop_cfg);

    // ── Dispatch ─────────────────────────────────────────────────────────────

    // Interactive REPL
    if (cfg.interactive) {
        run_repl(torus, loop, cfg);
        return 0;
    }

    // Batch: read one prompt per stdin line
    if (cfg.batch) {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            auto thought = run_prompt(torus, loop, line, cfg);
            print_result(line, thought, cfg);
        }
        return 0;
    }

    // Single-shot: --prompt flag
    if (!cfg.prompt.empty()) {
        if (!cfg.quiet && !cfg.json_out)
            std::cerr << ansi::c(ansi::dim) << "Prompt: " << cfg.prompt
                      << "\n" << ansi::c(ansi::rst);

        auto thought = run_prompt(torus, loop, cfg.prompt, cfg);
        print_result(cfg.prompt, thought, cfg);
        return 0;
    }

    // Default: read single prompt from stdin
    if (!cfg.quiet && !cfg.json_out)
        std::cerr << ansi::c(ansi::dim) << "Reading prompt from stdin...\n"
                  << ansi::c(ansi::rst);

    std::string line;
    if (std::getline(std::cin, line) && !line.empty()) {
        auto thought = run_prompt(torus, loop, line, cfg);
        print_result(line, thought, cfg);
    } else {
        // No input: go interactive
        cfg.interactive = true;
        run_repl(torus, loop, cfg);
    }

    return 0;
}
