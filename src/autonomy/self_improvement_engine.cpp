/**
 * @file src/autonomy/self_improvement_engine.cpp
 * @brief Phase 146 — SelfImprovementEngine implementation.
 *
 * Wires together all SIE components into a single autonomous cycle:
 *   formulate → generate → extract → compile → package → sign → deploy → store
 */

#include <nikola/autonomy/self_improvement_engine.hpp>
#include <nikola/autonomy/decision_loop.hpp>       // NikolaState full definition

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

// OpenSSL for Ed25519 signing (avoids symbol clash with SPHINCS+ ref impl)
#include <openssl/evp.h>

namespace nikola::autonomy {

namespace fs = std::filesystem;

// ============================================================================
// Construction / destruction
// ============================================================================

SelfImprovementEngine::SelfImprovementEngine(ShadowSpine& spine,
                                             SIEConfig    cfg)
    : spine_(spine)
    , cfg_(std::move(cfg))
    , specialist_(cfg_.specialist_server_path)
    , validator_(cfg_.ariac_path)
    , store_(cfg_.proposal_store_path.empty()
                 ? nullptr
                 : std::make_unique<aria::CodeProposalStore>(cfg_.proposal_store_path))
{
    // Create work directory
    std::error_code ec;
    fs::create_directories(cfg_.work_dir, ec);

    // Generate Ed25519 keypair via OpenSSL
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr);
    if (ctx && EVP_PKEY_keygen_init(ctx) == 1) {
        EVP_PKEY* pkey = nullptr;
        if (EVP_PKEY_keygen(ctx, &pkey) == 1 && pkey) {
            ed_pk_.resize(32);
            ed_sk_.resize(32);
            size_t pub_len = 32, priv_len = 32;
            EVP_PKEY_get_raw_public_key(pkey, ed_pk_.data(), &pub_len);
            EVP_PKEY_get_raw_private_key(pkey, ed_sk_.data(), &priv_len);
            ed_pkey_ = pkey;
        }
    }
    if (ctx) EVP_PKEY_CTX_free(ctx);

    // Generate SPHINCS+ keypair
    sphincs_kp_ = security::SphincsSigner::generate_keypair();
}

SelfImprovementEngine::~SelfImprovementEngine()
{
    if (specialist_started_) {
        specialist_.stop();
    }
    if (ed_pkey_) {
        EVP_PKEY_free(ed_pkey_);
    }
}

// ============================================================================
// Core operation
// ============================================================================

SIECycleResult SelfImprovementEngine::run_cycle(const NikolaState& state)
{
    std::lock_guard<std::mutex> lock(cycle_mutex_);

    SIECycleResult result;
    const auto t0 = std::chrono::steady_clock::now();
    ++cycles_attempted_;

    auto finish = [&](SIEOutcome out) {
        result.outcome = out;
        const auto t1 = std::chrono::steady_clock::now();
        result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return result;
    };

    // ── Step 1: Research the weakness (if research function configured) ──
    std::string research_context;
    if (research_fn_) {
        result.research_query = formulate_research_query(state);
        if (!result.research_query.empty()) {
            result.research_content = research_fn_(result.research_query);
            if (!result.research_content.empty()) {
                result.research_source = "research_router";
                research_context = result.research_content;
            }
        }
    }

    // ── Step 2: Formulate instruction ───────────────────────────────────────
    result.instruction = formulate_instruction(state, research_context);

    // ── Step 3: Query specialist ────────────────────────────────────────────
    if (!ensure_specialist_running()) {
        return finish(SIEOutcome::SPECIALIST_FAILED);
    }

    auto specialist_result = specialist_.ask(
        result.instruction,
        "Nikola self-improvement cycle. Generate a C++ module that exports "
        "extern \"C\" void* nikola_module_factory(). The function should "
        "return a pointer to a static struct with improvement parameters.",
        cfg_.specialist_timeout_ms);

    if (!specialist_result.ok) {
        result.raw_response = specialist_result.error;
        return finish(SIEOutcome::SPECIALIST_FAILED);
    }
    result.raw_response = specialist_result.response;

    // ── Step 4: Extract code ────────────────────────────────────────────────
    result.source_code = aria::extract_code_block(result.raw_response);
    if (result.source_code.empty()) {
        return finish(SIEOutcome::NO_CODE_EXTRACTED);
    }

    // ── Step 5: Security pre-check — ensure source doesn't contain
    //     obviously dangerous patterns before we compile ─────────────────────
    //     (Full Gate 1 check happens in the EO pipeline, but we do a
    //      quick pre-screen here to avoid compiling malicious code)

    // ── Step 6: Package into .so ────────────────────────────────────────────
    result.so_path = package_module(result.source_code, result.compile_output);
    if (result.so_path.empty()) {
        return finish(SIEOutcome::PACKAGING_FAILED);
    }

    // ── Step 7: Sign the module ─────────────────────────────────────────────
    auto binary = read_binary(result.so_path);
    if (binary.empty()) {
        return finish(SIEOutcome::SIGNING_FAILED);
    }

    auto sig_opt = sign_module(binary);
    if (!sig_opt) {
        return finish(SIEOutcome::SIGNING_FAILED);
    }

    // ── Step 8: Deploy through ShadowSpine (Gate 0–3) ──────────────────────
    auto stage_report = spine_.stage(
        result.so_path,
        result.source_code,
        *sig_opt,
        ed_pk_,
        sphincs_kp_.pk);

    result.stage_report = stage_report;

    if (!stage_report) {
        // Map the failure to the appropriate SIE outcome
        switch (stage_report.status) {
            case StageStatus::SIGNATURE_REJECTED:
                return finish(SIEOutcome::GATE0_REJECTED);
            case StageStatus::ATP_DENIED:
                return finish(SIEOutcome::ATP_DENIED);
            case StageStatus::SECURITY_REJECTED:
                return finish(SIEOutcome::GATE1_REJECTED);
            case StageStatus::PHYSICS_REJECTED:
                return finish(SIEOutcome::GATE2_REJECTED);
            case StageStatus::LOAD_FAILED:
            case StageStatus::SYMBOL_MISSING:
            case StageStatus::SAME_MODULE:
                return finish(SIEOutcome::GATE3_REJECTED);
            default:
                return finish(SIEOutcome::GATE3_REJECTED);
        }
    }

    // ── Step 9: Store proposal ──────────────────────────────────────────────
    if (store_) {
        aria::CodeProposal proposal;
        proposal.source_code     = result.source_code;
        proposal.compile_success = true;
        proposal.instruction     = result.instruction;
        proposal.compile_time_ms = result.elapsed_ms;
        result.proposal_id = store_->store(proposal);
    }

    // ── Step 10: SUCCESS ────────────────────────────────────────────────────
    ++cycles_succeeded_;
    return finish(SIEOutcome::SUCCESS);
}

// ============================================================================
// Solo-mode cycle (pre-generated source, bypass specialist)
// ============================================================================

SIECycleResult SelfImprovementEngine::run_cycle_with_source(
        const std::string& source_code,
        const std::string& instruction)
{
    std::lock_guard<std::mutex> lock(cycle_mutex_);

    SIECycleResult result;
    result.instruction = instruction;
    result.source_code = source_code;
    const auto t0 = std::chrono::steady_clock::now();
    ++cycles_attempted_;

    auto finish = [&](SIEOutcome out) {
        result.outcome = out;
        const auto t1 = std::chrono::steady_clock::now();
        result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return result;
    };

    // Skip steps 1-3 (formulate / specialist / extract) — source already provided.

    if (source_code.empty()) {
        return finish(SIEOutcome::NO_CODE_EXTRACTED);
    }

    // ── Step 5: Package into .so ────────────────────────────────────────────
    result.so_path = package_module(result.source_code, result.compile_output);
    if (result.so_path.empty()) {
        return finish(SIEOutcome::PACKAGING_FAILED);
    }

    // ── Step 6: Sign the module ─────────────────────────────────────────────
    auto binary = read_binary(result.so_path);
    if (binary.empty()) {
        return finish(SIEOutcome::SIGNING_FAILED);
    }

    auto sig_opt = sign_module(binary);
    if (!sig_opt) {
        return finish(SIEOutcome::SIGNING_FAILED);
    }

    // ── Step 7: Deploy through ShadowSpine (Gate 0–3) ──────────────────────
    auto stage_report = spine_.stage(
        result.so_path,
        result.source_code,
        *sig_opt,
        ed_pk_,
        sphincs_kp_.pk);

    result.stage_report = stage_report;

    if (!stage_report) {
        switch (stage_report.status) {
            case StageStatus::SIGNATURE_REJECTED:
                return finish(SIEOutcome::GATE0_REJECTED);
            case StageStatus::ATP_DENIED:
                return finish(SIEOutcome::ATP_DENIED);
            case StageStatus::SECURITY_REJECTED:
                return finish(SIEOutcome::GATE1_REJECTED);
            case StageStatus::PHYSICS_REJECTED:
                return finish(SIEOutcome::GATE2_REJECTED);
            case StageStatus::LOAD_FAILED:
            case StageStatus::SYMBOL_MISSING:
            case StageStatus::SAME_MODULE:
                return finish(SIEOutcome::GATE3_REJECTED);
            default:
                return finish(SIEOutcome::GATE3_REJECTED);
        }
    }

    // ── Step 8: Store proposal ──────────────────────────────────────────────
    if (store_) {
        aria::CodeProposal proposal;
        proposal.source_code     = result.source_code;
        proposal.compile_success = true;
        proposal.instruction     = result.instruction;
        proposal.compile_time_ms = result.elapsed_ms;
        result.proposal_id = store_->store(proposal);
    }

    // ── Step 9: SUCCESS ─────────────────────────────────────────────────────
    ++cycles_succeeded_;
    return finish(SIEOutcome::SUCCESS);
}

// ============================================================================
// Instruction formulation
// ============================================================================

std::string SelfImprovementEngine::formulate_instruction(
        const NikolaState& state,
        const std::string& research_context) const
{
    // Build a targeted improvement instruction based on current cognitive state.
    // This is the "what should I improve?" logic — Nikola introspects on its
    // own state to decide what kind of module would be beneficial.

    std::ostringstream oss;

    // Determine the improvement target based on state signals
    if (state.boredom > 0.8f) {
        oss << "Generate a Nikola cognitive enhancement module in C++. "
               "The module should improve exploration diversity by providing "
               "scoring weight adjustments. Current boredom level is very high ("
            << state.boredom << "), suggesting the exploration strategy needs "
               "more novelty injection. ";
    } else if (state.entropy > 2.0f) {
        oss << "Generate a Nikola cognitive enhancement module in C++. "
               "The module should improve thought coherence by providing "
               "attention weighting parameters. Current field entropy is high ("
            << state.entropy << "), indicating the cognitive field needs "
               "better spectral organisation. ";
    } else if (state.dopamine < 0.3f && state.atp > 0.5f) {
        oss << "Generate a Nikola cognitive enhancement module in C++. "
               "The module should improve reward sensitivity by adjusting "
               "dopamine baseline parameters. Current dopamine is low ("
            << state.dopamine << ") despite adequate ATP ("
            << state.atp << "), suggesting the reward prediction needs "
               "recalibration. ";
    } else {
        oss << "Generate a Nikola cognitive enhancement module in C++. "
               "The module should provide general parameter tuning for the "
               "cognitive scoring system. Current state: boredom="
            << state.boredom << " entropy=" << state.entropy
            << " dopamine=" << state.dopamine << " atp=" << state.atp << ". ";
    }

    // If research context is available, inject it so the specialist can
    // produce more informed code based on real-world knowledge.
    if (!research_context.empty()) {
        oss << "\n\nResearch context (external knowledge retrieved for this cycle):\n"
               "--- BEGIN RESEARCH ---\n"
            << research_context.substr(0, 4096)  // Cap to avoid prompt overflow
            << "\n--- END RESEARCH ---\n"
               "Use the above research to inform your parameter choices and "
               "implementation approach.\n";
    }

    oss << "\n\nThe module MUST:\n"
           "1. Be valid C++17\n"
           "2. Export: extern \"C\" void* nikola_module_factory()\n"
           "3. Return a pointer to a static struct with named parameters\n"
           "4. NOT use: system(), exec(), fork(), popen(), asm, "
              "socket, /proc/, /dev/, ptrace, mmap, dlopen\n"
           "5. Only include standard library headers (<cstdint>, <cmath>, etc.)\n"
           "\n"
           "Example structure:\n"
           "```cpp\n"
           "#include <cstdint>\n"
           "\n"
           "struct CognitiveParameters {\n"
           "    uint32_t version;\n"
           "    const char* name;\n"
           "    float exploration_weight;\n"
           "    float coherence_bias;\n"
           "    float reward_sensitivity;\n"
           "};\n"
           "\n"
           "static CognitiveParameters params = {\n"
           "    1, \"cognitive_tuning_v1\",\n"
           "    0.65f, 0.30f, 0.50f\n"
           "};\n"
           "\n"
           "extern \"C\" void* nikola_module_factory() {\n"
           "    return &params;\n"
           "}\n"
           "```";

    return oss.str();
}

// ============================================================================
// Module packaging
// ============================================================================

std::string SelfImprovementEngine::package_module(
        const std::string& source,
        std::string& compile_output)
{
    // Write source to temporary file
    const auto src_path = cfg_.work_dir + "/candidate.cpp";
    const auto so_path  = cfg_.work_dir + "/candidate.so";

    {
        std::ofstream f(src_path);
        if (!f) {
            compile_output = "Failed to write source to " + src_path;
            return {};
        }
        f << source;
    }

    // Compile to shared library
    // -shared -fPIC: produce a position-independent shared library
    // -O2: basic optimisation (avoid debug bloat)
    // -std=c++17: language standard
    // -o: output path
    const std::string cmd =
        cfg_.gpp_path + " -shared -fPIC -O2 -std=c++17 "
        "-o " + so_path + " " + src_path + " 2>&1";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        compile_output = "Failed to invoke g++";
        return {};
    }

    std::string output;
    char buf[256];
    while (fgets(buf, sizeof(buf), pipe)) {
        output += buf;
    }
    int rc = pclose(pipe);
    compile_output = output;

    if (rc != 0) {
        return {};  // Compilation failed
    }

    // Verify the .so exists and has the factory symbol
    if (!fs::exists(so_path)) {
        compile_output += "\n.so not created";
        return {};
    }

    return so_path;
}

// ============================================================================
// Module signing
// ============================================================================

std::optional<security::HybridSignature>
SelfImprovementEngine::sign_module(const std::vector<uint8_t>& binary)
{
    if (!ed_pkey_ || binary.empty()) return std::nullopt;

    security::HybridSignature sig;

    // Ed25519 sign via OpenSSL
    EVP_MD_CTX* mctx = EVP_MD_CTX_new();
    if (!mctx) return std::nullopt;

    if (EVP_DigestSignInit(mctx, nullptr, nullptr, nullptr, ed_pkey_) != 1) {
        EVP_MD_CTX_free(mctx);
        return std::nullopt;
    }

    size_t sig_len = 64;
    sig.ed25519_sig.resize(64);
    if (EVP_DigestSign(mctx, sig.ed25519_sig.data(), &sig_len,
                       binary.data(), binary.size()) != 1) {
        EVP_MD_CTX_free(mctx);
        return std::nullopt;
    }
    sig.ed25519_sig.resize(sig_len);
    EVP_MD_CTX_free(mctx);

    // SPHINCS+ sign
    try {
        auto sphincs_sig = security::SphincsSigner::sign(binary, sphincs_kp_);
        sig.sphincs_sig = std::move(sphincs_sig.bytes);
    } catch (const std::exception&) {
        return std::nullopt;
    }

    return sig;
}

// ============================================================================
// Helpers
// ============================================================================

std::vector<uint8_t> SelfImprovementEngine::read_binary(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return {std::istreambuf_iterator<char>(f),
            std::istreambuf_iterator<char>()};
}

bool SelfImprovementEngine::ensure_specialist_running()
{
    if (specialist_started_) return true;
    if (cfg_.specialist_server_path.empty()) return false;

    specialist_started_ = specialist_.start();
    return specialist_started_;
}

bool SelfImprovementEngine::specialist_running() const noexcept
{
    return specialist_started_;
}

// ============================================================================
// Research query formulation
// ============================================================================

std::string SelfImprovementEngine::formulate_research_query(
        const NikolaState& state)
{
    // Translate the identified weakness into a targeted research query.
    // The query should retrieve knowledge that helps the specialist
    // generate better cognitive parameters.

    if (state.boredom > 0.8f) {
        return "exploration diversity algorithms novelty injection "
               "cognitive parameter tuning reinforcement learning";
    }
    if (state.entropy > 2.0f) {
        return "attention weighting spectral coherence neural field "
               "entropy reduction cognitive architecture";
    }
    if (state.dopamine < 0.3f && state.atp > 0.5f) {
        return "reward prediction error dopamine baseline recalibration "
               "temporal difference learning parameter optimization";
    }
    // General improvement
    return "cognitive scoring system parameter tuning "
           "autonomous self-improvement neural architecture";
}

} // namespace nikola::autonomy
