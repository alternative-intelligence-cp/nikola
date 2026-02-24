/**
 * @file src/nikola_cognitive_echo.cpp
 * @brief Nikola Path B demo — text → torus resonance → decoded output
 *
 * Full Path B pipeline demonstration:
 *   1. Build a 3^9 = 19,683-node CognitiveTorus with BERT-Tiny ONNX embedder
 *   2. Pre-populate a ResonanceDecoder vocabulary (common + cognitive terms)
 *   3. inject_text(prompt) → holographic wave injection into torus
 *   4. run(N) steps → let the waveform propagate and resonate
 *   5. Decode resonance → token sequence → joined response text
 *   6. Print diagnostics: hot nodes, energy, decoded tokens
 *   7. Write resonance JSON snapshot to stdout for Aria hybrid integration
 *    (nikola_hello.py can extend its state prompt with the resonance vector)
 *
 * Usage:
 *   ./nikola_cognitive_echo [prompt] [num_steps]
 *
 *   prompt:    Text to inject (default: "Hello Nikola")
 *   num_steps: Physics propagation steps (default: 200)
 *
 * Requires NIKOLA_HAS_ORT=1 (ONNX Runtime linked).
 */

#ifndef NIKOLA_HAS_ORT
#  error "nikola_cognitive_echo requires ONNX Runtime. Build with -DNIKOLA_HAS_ORT=1."
#endif

#include <nikola/cognitive/cognitive_torus.hpp>
#include <nikola/cognitive/resonance_decoder.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace nikola::cognitive;

// ─────────────────────────────────────────────────────────────────────────────
// Vocabulary: terms the decoder can recognise after torus resonance.
// Extend this list for richer output.
// ─────────────────────────────────────────────────────────────────────────────
static const std::vector<std::string> VOCAB = {
    // Greetings / common
    "hello", "hi", "greetings", "welcome", "goodbye",
    // Cognitive / self-model
    "consciousness", "awareness", "mind", "thought", "memory",
    "resonance", "wave", "torus", "field", "energy",
    // Identity
    "nikola", "aria", "system", "network", "brain",
    // Temporal
    "time", "moment", "now", "past", "future",
    // Physics
    "quantum", "wave", "interference", "propagate", "frequency",
    // Affect
    "curious", "wonder", "interest", "explore", "understand",
    // Communication
    "language", "word", "speak", "listen", "respond",
    // Abstract
    "concept", "pattern", "structure", "form", "flow",
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static void print_separator(char c = '-', int width = 72) {
    for (int i = 0; i < width; ++i) std::cout << c;
    std::cout << '\n';
}

static void print_resonance_stats(const CognitiveTorus& ct, int top_k = 10) {
    auto hot = ct.hot_nodes(static_cast<size_t>(top_k));
    std::cout << "  Hot nodes (top-" << top_k << "):\n";
    for (size_t i = 0; i < hot.size(); ++i) {
        size_t idx = hot[i];
        float  I   = ct.intensity(idx);
        auto   p   = ct.psi(idx);
        std::cout << "    [" << std::setw(5) << idx << "] "
                  << "  |ψ|² = " << std::fixed << std::setprecision(6) << I
                  << "  ψ = (" << std::setprecision(4) << p.real()
                  << ", " << p.imag() << ")\n";
    }
    std::cout << "  Total probability P = "
              << std::scientific << std::setprecision(6)
              << ct.total_probability() << '\n';
}

static void write_resonance_json(const CognitiveTorus& ct,
                                 const std::string& prompt,
                                 const std::string& decoded,
                                 int steps) {
    auto snap = ct.resonance_snapshot();

    // Find top-20 hot nodes for compact JSON export
    auto hot = ct.hot_nodes(20);

    std::cout << "\n{\"path_b_resonance\":{\n"
              << "  \"prompt\": \"" << prompt << "\",\n"
              << "  \"steps\": " << steps << ",\n"
              << "  \"total_P\": " << ct.total_probability() << ",\n"
              << "  \"decoded\": \"" << decoded << "\",\n"
              << "  \"hot_nodes\": [";
    for (size_t i = 0; i < hot.size(); ++i) {
        if (i) std::cout << ',';
        size_t idx = hot[i];
        std::cout << "{\"idx\":" << idx
                  << ",\"I\":" << ct.intensity(idx) << '}';
    }
    std::cout << "]\n}}\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // ── Parse arguments ──────────────────────────────────────────────────────
    std::string prompt    = (argc > 1) ? argv[1] : "Hello Nikola";
    int         num_steps = (argc > 2) ? std::atoi(argv[2]) : 200;
    if (num_steps <= 0) num_steps = 200;

    print_separator('=');
    std::cout << "  Nikola Path B - Cognitive Echo  v0.0.4\n";
    print_separator('=');
    std::cout << "  Prompt    : \"" << prompt << "\"\n"
              << "  Steps     : " << num_steps << "\n"
              << "  Grid      : 3^9 = 19,683 nodes\n"
              << "  Vocab     : " << VOCAB.size() << " terms\n";
    print_separator();

    // ── Build CognitiveTorus ─────────────────────────────────────────────────
    std::cout << "[1/5] Initialising CognitiveTorus (n=3)...\n";
    auto t0 = std::chrono::steady_clock::now();

    CognitiveTorus ct(3);   // 3^9 = 19,683 nodes; ONNX paths from CMake defs

    auto t1 = std::chrono::steady_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "    Done in " << std::fixed << std::setprecision(1)
              << init_ms << " ms\n"
              << "    Nodes : " << ct.num_nodes()
              << "  |  max_dt : " << std::scientific << ct.max_dt()
              << "  |  safe_dt : " << ct.safe_dt() << "\n";
    print_separator();

    // ── Pre-populate decoder vocabulary ──────────────────────────────────────
    std::cout << "[2/5] Registering vocabulary (" << VOCAB.size() << " terms)...\n";
    auto t2 = std::chrono::steady_clock::now();

    ResonanceDecoder decoder;

    // CognitiveTorus holds the NonaryEmbedder; we need a separate instance for
    // vocabulary pre-population.  We construct one directly using the same paths.
    NonaryEmbedder vocab_embedder(NIKOLA_ORT_TOKENIZER_PATH, NIKOLA_ORT_MODEL_PATH);
    decoder.register_vocabulary(vocab_embedder, VOCAB);

    auto t3 = std::chrono::steady_clock::now();
    double vocab_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "    Registered " << decoder.vocab_size() << " tokens in "
              << std::fixed << std::setprecision(1) << vocab_ms << " ms\n";
    print_separator();

    // ── Inject text ───────────────────────────────────────────────────────────
    std::cout << "[3/5] Injecting text into torus...\n";
    double P_before = ct.total_probability();
    std::cout << "    P_before = " << std::scientific << P_before << "\n";

    ct.inject_text(prompt, 0.0);

    double P_after_inject = ct.total_probability();
    std::cout << "    P_after_inject = " << P_after_inject
              << "  (ΔP = " << (P_after_inject - P_before) << ")\n";
    print_separator();

    // ── Run physics ───────────────────────────────────────────────────────────
    std::cout << "[4/5] Running " << num_steps << " physics steps...\n";
    auto t4 = std::chrono::steady_clock::now();

    float dt = ct.safe_dt();
    ct.run(num_steps, dt);

    auto t5 = std::chrono::steady_clock::now();
    double run_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    std::cout << "    Done in " << std::fixed << std::setprecision(1)
              << run_ms << " ms  ("
              << std::setprecision(3) << run_ms / num_steps << " ms/step)\n"
              << "    Simulation time : " << ct.time() << "\n";

    std::cout << "\n  Resonance readout:\n";
    print_resonance_stats(ct, 10);
    print_separator();

    // ── Decode resonance ──────────────────────────────────────────────────────
    std::cout << "[5/5] Decoding resonance → tokens...\n";
    auto tokens = decoder.decode(ct, 30);
    std::string decoded_text = decoder.decode_text(ct, 30);

    if (tokens.empty()) {
        std::cout << "    No tokens matched (lexicon cosine threshold not met).\n"
                  << "    Hint: increase vocabulary size or run more steps.\n";
    } else {
        std::cout << "    Matched tokens (" << tokens.size() << "):";
        for (const auto& tok : tokens) std::cout << " [" << tok << "]";
        std::cout << "\n    Decoded text: \"" << decoded_text << "\"\n";
    }
    print_separator('=');

    // ── JSON output for hybrid Path A/B integration ───────────────────────────
    write_resonance_json(ct, prompt, decoded_text, num_steps);

    return 0;
}
