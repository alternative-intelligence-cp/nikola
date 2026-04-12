/**
 * @file phase157_prerelease_audit_test.cpp
 * @brief Phase 157 — v0.1.20 Pre-Release Audit integration test suite
 *
 * Comprehensive tests covering gaps identified in the v0.1.20 audit:
 *   §1–§6   Training pipeline (Mamba, Transformer, Bicameral)
 *   §7–§10  Persistence (DMC checkpoint save/load round-trip)
 *   §11–§16 NAP orchestrator lifecycle
 *   §17–§21 Security pipeline integration (CSVP + Anomaly + KVM)
 *   §22–§24 Cross-subsystem integration
 *
 * 24 test cases, no external dependencies (no live Gemini API / SIE).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Training
#include <nikola/trainers/mamba_trainer.hpp>
#include <nikola/trainers/transformer_trainer.hpp>
#include <nikola/trainers/bicameral_trainer.hpp>

// Persistence
#include <nikola/persistence/dmc_checkpoint.hpp>

// NAP
#include <nikola/autonomy/nap_orchestrator.hpp>
#include <nikola/autonomy/nap_controller.hpp>

// Security
#include <nikola/security/csvp.hpp>
#include <nikola/security/anomaly_detector.hpp>
#include <nikola/security/kvm_sandbox.hpp>

#include <cmath>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

using namespace nikola;

// ============================================================================
// Helper: generate deterministic SSM training data
// ============================================================================

static std::vector<trainers::TrainingSample> make_mamba_data(size_t n) {
    std::vector<trainers::TrainingSample> data(n);
    for (size_t i = 0; i < n; ++i) {
        for (int d = 0; d < 9; ++d) {
            double phase = static_cast<double>(i * 9 + d) * 0.1;
            data[i].state[d]      = std::sin(phase);
            data[i].input[d]      = std::cos(phase);
            data[i].next_state[d] = std::sin(phase + 0.1);
        }
    }
    return data;
}

static std::vector<trainers::AttentionSample> make_attn_data(size_t n) {
    std::vector<trainers::AttentionSample> data(n);
    for (size_t i = 0; i < n; ++i) {
        for (int d = 0; d < 9; ++d) {
            double phase = static_cast<double>(i * 9 + d) * 0.05;
            data[i].x1[d] = std::sin(phase);
            data[i].x2[d] = std::cos(phase);
            data[i].y1[d] = std::sin(phase + 0.1);
            data[i].y2[d] = std::cos(phase + 0.1);
        }
    }
    return data;
}

// ============================================================================
// §1  MambaTrainer — batch training reduces loss
// ============================================================================

TEST_CASE("§1 Mamba batch training reduces loss", "[phase157][training]") {
    trainers::MambaTrainer mt(0.01);
    auto data = make_mamba_data(32);

    auto stats0 = mt.train_batch(data);
    double loss0 = stats0.loss;
    REQUIRE(loss0 > 0.0);
    REQUIRE(stats0.samples == 32);

    // Train 10 more epochs
    double last_loss = loss0;
    for (int e = 0; e < 10; ++e) {
        auto s = mt.train_batch(data);
        last_loss = s.loss;
    }

    REQUIRE(last_loss < loss0);       // loss decreased
    REQUIRE(mt.epoch() == 11);         // 11 epochs total
    REQUIRE(mt.graph_size() == 539);   // graph topology unchanged
}

// ============================================================================
// §2  MambaTrainer — prediction after training
// ============================================================================

TEST_CASE("§2 Mamba predict after training", "[phase157][training]") {
    trainers::MambaTrainer mt(0.005);
    auto data = make_mamba_data(64);

    // Train for 20 epochs
    for (int e = 0; e < 20; ++e) mt.train_batch(data);

    // Predict on first sample
    auto pred = mt.predict(data[0].state, data[0].input);
    double error = 0.0;
    for (int d = 0; d < 9; ++d) {
        double diff = pred[d] - data[0].next_state[d];
        error += diff * diff;
    }
    error = std::sqrt(error);

    // Prediction error should be finite and reasonable
    REQUIRE(std::isfinite(error));
    REQUIRE(error < 10.0);  // not degenerate
}

// ============================================================================
// §3  TransformerTrainer — batch training reduces loss
// ============================================================================

TEST_CASE("§3 Transformer batch training reduces loss", "[phase157][training]") {
    trainers::TransformerTrainer tt(0.001);
    auto data = make_attn_data(16);

    auto stats0 = tt.train_batch(data);
    double loss0 = stats0.loss;
    REQUIRE(loss0 > 0.0);
    REQUIRE(stats0.samples == 16);

    double last_loss = loss0;
    for (int e = 0; e < 10; ++e) {
        auto s = tt.train_batch(data);
        last_loss = s.loss;
    }

    REQUIRE(last_loss < loss0);
    REQUIRE(tt.epoch() == 11);
    REQUIRE(tt.graph_size() == 1377);
}

// ============================================================================
// §4  TransformerTrainer — gradient evaluation
// ============================================================================

TEST_CASE("§4 Transformer gradient eval non-zero", "[phase157][training]") {
    trainers::TransformerTrainer tt(0.001);
    auto data = make_attn_data(1);

    auto gr = tt.eval_gradient(data[0]);
    REQUIRE(gr.loss > 0.0);

    // At least some gradients should be non-zero
    bool any_nonzero = false;
    for (int i = 0; i < trainers::ATTN_DIM_SQ; ++i) {
        if (std::abs(gr.grad_Q[i]) > 1e-12) any_nonzero = true;
        if (std::abs(gr.grad_K[i]) > 1e-12) any_nonzero = true;
        if (std::abs(gr.grad_V[i]) > 1e-12) any_nonzero = true;
    }
    REQUIRE(any_nonzero);
}

// ============================================================================
// §5  MambaTrainer — auto-training trigger
// ============================================================================

TEST_CASE("§5 Mamba auto-training trigger", "[phase157][training]") {
    trainers::MambaTrainer mt;
    mt.set_error_threshold(0.05);

    // Low errors: should NOT trigger
    for (int i = 0; i < 10; ++i) {
        REQUIRE_FALSE(mt.should_train(0.001));
    }

    // High errors: EMA rises, should trigger eventually
    bool triggered = false;
    for (int i = 0; i < 100; ++i) {
        if (mt.should_train(0.5)) { triggered = true; break; }
    }
    REQUIRE(triggered);
}

// ============================================================================
// §6  MambaTrainer — gradient correctness (numerical check)
// ============================================================================

TEST_CASE("§6 Mamba gradient numerical check", "[phase157][training]") {
    trainers::MambaTrainer mt(0.01);
    mt.reset_params(123);
    auto data = make_mamba_data(1);
    const auto& sample = data[0];

    auto gr = mt.eval_gradient(sample);
    REQUIRE(std::isfinite(gr.loss));

    // Numerical gradient for A[0][0] via finite differences
    double eps = 1e-5;
    double orig = mt.A()[0];

    mt.A()[0] = orig + eps;
    auto gr_plus = mt.eval_gradient(sample);
    mt.A()[0] = orig - eps;
    auto gr_minus = mt.eval_gradient(sample);
    mt.A()[0] = orig;

    double numerical_grad = (gr_plus.loss - gr_minus.loss) / (2.0 * eps);
    double analytical_grad = gr.grad_A[0];

    // Should agree within ~1% relative or absolute tolerance
    double abs_diff = std::abs(numerical_grad - analytical_grad);
    double scale = std::max(std::abs(numerical_grad), std::abs(analytical_grad));
    if (scale > 1e-8) {
        REQUIRE(abs_diff / scale < 0.05);
    } else {
        REQUIRE(abs_diff < 1e-6);
    }
}

// ============================================================================
// §7  Persistence — NikolaState pack/unpack round-trip
// ============================================================================

TEST_CASE("§7 NikolaState pack/unpack round-trip", "[phase157][persistence]") {
    using namespace persistence;

    autonomy::NikolaState original;
    original.time         = 42.5f;
    original.torus_energy = 0.87f;
    original.dopamine     = 0.6f;
    original.td_error     = -0.1f;
    original.atp          = 0.45f;
    original.boredom      = 0.3f;
    original.entropy      = 2.1f;
    original.last_action  = autonomy::ActionType::EMIT_THOUGHT;
    original.tokens       = {"hello", "world", "nikola"};

    auto packed = pack_nikola_state(original);
    REQUIRE(!packed.empty());

    autonomy::NikolaState restored;
    unpack_nikola_state(packed.data(), packed.size(), restored);

    REQUIRE_THAT(restored.time,         Catch::Matchers::WithinAbs(42.5, 1e-4));
    REQUIRE_THAT(restored.torus_energy, Catch::Matchers::WithinAbs(0.87, 1e-4));
    REQUIRE_THAT(restored.dopamine,     Catch::Matchers::WithinAbs(0.6, 1e-4));
    REQUIRE_THAT(restored.td_error,     Catch::Matchers::WithinAbs(-0.1, 1e-4));
    REQUIRE_THAT(restored.atp,          Catch::Matchers::WithinAbs(0.45, 1e-4));
    REQUIRE_THAT(restored.boredom,      Catch::Matchers::WithinAbs(0.3, 1e-4));
    REQUIRE_THAT(restored.entropy,      Catch::Matchers::WithinAbs(2.1, 1e-4));
    REQUIRE(restored.last_action == autonomy::ActionType::EMIT_THOUGHT);
    REQUIRE(restored.tokens.size() == 3);
    REQUIRE(restored.tokens[0] == "hello");
    REQUIRE(restored.tokens[1] == "world");
    REQUIRE(restored.tokens[2] == "nikola");
}

// ============================================================================
// §8  Persistence — .nik header constants
// ============================================================================

TEST_CASE("§8 .nik format constants valid", "[phase157][persistence]") {
    using namespace persistence;

    REQUIRE(NIK_MAGIC == 0x4E494B4Fu);
    REQUIRE(NIK_HEADER_SIZE == 64);
    REQUIRE(NIK_FOOTER_SIZE == 128);
    REQUIRE(sizeof(NikHeader) == 64);
    REQUIRE(sizeof(NikFooter) == 128);
    REQUIRE(sizeof(SectionHeader) == 24);
    REQUIRE(NIK_DIM_ENCODING == 0x09);
}

// ============================================================================
// §9  Persistence — WAL lifecycle
// ============================================================================

TEST_CASE("§9 WAL write and cleanup", "[phase157][persistence]") {
    using namespace persistence;

    std::string wal_path = "/tmp/nikola_test_phase157.wal";

    // Clean up any prior leftover
    std::filesystem::remove(wal_path);

    {
        WriteAheadLog wal(wal_path);
        REQUIRE(wal.open());

        std::vector<uint8_t> payload{0x01, 0x02, 0x03, 0x04};
        wal.append(WAL_INSERT, payload);
        wal.append(WAL_UPDATE, payload);
        wal.commit();

        REQUIRE(wal.exists());
    }

    // WAL file should exist
    REQUIRE(std::filesystem::exists(wal_path));

    // Close and remove
    {
        WriteAheadLog wal(wal_path);
        wal.close_and_remove();
        REQUIRE_FALSE(wal.exists());
    }
}

// ============================================================================
// §10  Persistence — CheckpointController triggers
// ============================================================================

TEST_CASE("§10 CheckpointController triggers", "[phase157][persistence]") {
    using namespace persistence;

    CheckpointController ctrl(300.f);  // 300s interval

    // NAP always triggers
    REQUIRE(ctrl.should_checkpoint(0.f, autonomy::ActionType::NAP));

    // Non-NAP: not enough time elapsed
    REQUIRE_FALSE(ctrl.should_checkpoint(10.f, autonomy::ActionType::SILENT));

    // Periodic trigger: 300s elapsed
    REQUIRE(ctrl.should_checkpoint(301.f, autonomy::ActionType::SILENT));

    // Record checkpoint, then verify reset
    ctrl.record_checkpoint(301.f);
    REQUIRE(ctrl.checkpoint_count() == 1);
    REQUIRE_FALSE(ctrl.should_checkpoint(350.f, autonomy::ActionType::SILENT));
    REQUIRE(ctrl.should_checkpoint(602.f, autonomy::ActionType::SILENT));
}

// ============================================================================
// §11  NAP Orchestrator — construction and config
// ============================================================================

TEST_CASE("§11 NapOrchestrator default config", "[phase157][nap]") {
    using namespace autonomy;

    NapOrchestrator orch;
    REQUIRE_FALSE(orch.is_napping());
    REQUIRE(orch.state() == NapState::AWAKE);
    REQUIRE(orch.nap_count() == 0);

    auto cfg = orch.config();
    REQUIRE_THAT(cfg.atp_recharge_rate, Catch::Matchers::WithinAbs(0.05, 1e-6));
    REQUIRE_THAT(cfg.norepinephrine_level, Catch::Matchers::WithinAbs(0.3, 1e-6));
    REQUIRE(cfg.checkpoint_on_entry == true);
    REQUIRE(cfg.run_dream_weave == true);
    REQUIRE(cfg.run_dream_engine == true);
    REQUIRE(cfg.run_consolidation == true);
}

// ============================================================================
// §12  NAP Orchestrator — ATP recharge during nap
// ============================================================================

TEST_CASE("§12 NAP ATP recharge mechanics", "[phase157][nap]") {
    using namespace autonomy;

    NapOrchestratorConfig cfg;
    cfg.atp_recharge_rate   = 0.1f;
    cfg.checkpoint_on_entry = false;
    cfg.run_dream_weave     = false;
    cfg.run_dream_engine    = false;
    cfg.run_consolidation   = false;
    NapOrchestrator orch(cfg);

    // Start with low ATP → should enter nap
    float atp = 0.1f;  // below entry threshold (0.15)
    float time = 0.f;

    orch.update(atp, time, 0.f);

    // Should now be napping (ATP < 0.15)
    REQUIRE(orch.is_napping());

    // Simulate ticks — ATP should recharge
    float prev_atp = atp;
    for (int i = 0; i < 10; ++i) {
        time += 1.f;
        orch.update(atp, time, 1.f);
        REQUIRE(atp >= prev_atp);  // ATP never decreases during nap
        prev_atp = atp;
    }

    // ATP should have increased substantially (0.1 recharge/s * 10s = +1.0)
    REQUIRE(atp > 0.5f);
}

// ============================================================================
// §13  NAP Orchestrator — nap exit at high ATP
// ============================================================================

TEST_CASE("§13 NAP exit at high ATP", "[phase157][nap]") {
    using namespace autonomy;

    NapOrchestratorConfig cfg;
    cfg.atp_recharge_rate   = 0.5f;  // fast recharge
    cfg.checkpoint_on_entry = false;
    cfg.run_dream_weave     = false;
    cfg.run_dream_engine    = false;
    cfg.run_consolidation   = false;
    NapOrchestrator orch(cfg);

    float atp = 0.05f;
    float time = 0.f;

    // Enter nap
    orch.update(atp, time, 0.f);
    REQUIRE(orch.is_napping());

    // Recharge until exit threshold (0.90)
    while (orch.is_napping() && time < 60.f) {
        time += 1.f;
        orch.update(atp, time, 1.f);
    }

    // Should have exited nap
    REQUIRE_FALSE(orch.is_napping());
    REQUIRE(atp >= 0.90f);
    REQUIRE(orch.nap_count() == 1);
}

// ============================================================================
// §14  NAP Orchestrator — z-normalize utility
// ============================================================================

TEST_CASE("§14 NAP z-normalize", "[phase157][nap]") {
    using namespace autonomy;

    std::vector<float> data{1.f, 2.f, 3.f, 4.f, 5.f};
    NapOrchestrator::z_normalize(data);

    // After z-normalize: mean ≈ 0, std ≈ 1
    float sum = 0.f;
    for (float v : data) sum += v;
    REQUIRE_THAT(sum / 5.f, Catch::Matchers::WithinAbs(0.0, 1e-5));

    float sq_sum = 0.f;
    for (float v : data) sq_sum += v * v;
    float var = sq_sum / 5.f;
    REQUIRE_THAT(var, Catch::Matchers::WithinAbs(1.0, 0.3));  // variance near 1

    // Empty data should not crash
    std::vector<float> empty;
    NapOrchestrator::z_normalize(empty);
    REQUIRE(empty.empty());
}

// ============================================================================
// §15  NAP Orchestrator — dream buffer isolation
// ============================================================================

TEST_CASE("§15 NAP dream buffer isolation", "[phase157][nap]") {
    using namespace autonomy;

    NapOrchestratorConfig cfg;
    cfg.checkpoint_on_entry = false;
    cfg.run_dream_weave     = false;
    cfg.run_dream_engine    = false;
    cfg.run_consolidation   = false;
    NapOrchestrator orch(cfg);

    // Set waking psi buffers
    std::vector<float> psi_r{1.f, 2.f, 3.f, 4.f};
    std::vector<float> psi_i{0.5f, 1.5f, 2.5f, 3.5f};
    orch.init_dream_buffers(psi_r, psi_i);

    // Enter nap — dream buffers should be copied and z-normalized
    float atp = 0.05f;
    orch.update(atp, 0.f, 0.f);
    REQUIRE(orch.is_napping());

    // Dream buffers should be sized correctly
    REQUIRE(orch.dream_psi_real().size() == 4);
    REQUIRE(orch.dream_psi_imag().size() == 4);

    // Dream buffers should be z-normalized (different from original)
    bool different = false;
    for (size_t i = 0; i < 4; ++i) {
        if (std::abs(orch.dream_psi_real()[i] - psi_r[i]) > 1e-6)
            different = true;
    }
    REQUIRE(different);  // z-normalization changed the values
}

// ============================================================================
// §16  NAP Orchestrator — consolidation callback
// ============================================================================

TEST_CASE("§16 NAP consolidation callback fires", "[phase157][nap]") {
    using namespace autonomy;

    NapOrchestratorConfig cfg;
    cfg.atp_recharge_rate   = 0.01f;
    cfg.checkpoint_on_entry = false;
    cfg.run_dream_weave     = false;
    cfg.run_dream_engine    = false;
    cfg.run_consolidation   = true;
    NapOrchestrator orch(cfg);

    bool consolidated = false;
    orch.set_consolidation_fn([&]() -> ConsolidationResult {
        consolidated = true;
        return {5, 10};  // pruned 5, replayed 10
    });

    float atp = 0.05f;
    orch.update(atp, 0.f, 0.f);  // enter nap
    REQUIRE(orch.is_napping());

    // Tick to trigger consolidation
    orch.update(atp, 1.f, 1.f);
    REQUIRE(consolidated);
}

// ============================================================================
// §17  CSVP — safe code passes all stages
// ============================================================================

TEST_CASE("§17 CSVP safe code passes", "[phase157][security]") {
    using namespace security;

    CodeSafetyVerifier csvp;

    std::string safe_code = R"(
        #include <cmath>
        double compute(double x) {
            return std::sin(x) * std::cos(x);
        }
    )";

    auto result = csvp.verify(safe_code);
    REQUIRE(result.approved);
    REQUIRE(result.stages_passed >= 1);
    REQUIRE(result.violations.empty());
}

// ============================================================================
// §18  CSVP — dangerous patterns rejected
// ============================================================================

TEST_CASE("§18 CSVP dangerous patterns rejected", "[phase157][security]") {
    using namespace security;

    CodeSafetyVerifier csvp;

    // system() call should be caught by pattern blacklist
    std::string dangerous = R"(
        #include <cstdlib>
        void run() {
            system("rm -rf /");
        }
    )";

    auto result = csvp.verify(dangerous);
    REQUIRE_FALSE(result.approved);
    REQUIRE(result.failed_stage == CSVPStage::PATTERN_BLACKLIST);
    REQUIRE(!result.violations.empty());
}

// ============================================================================
// §19  CSVP — oversized source rejected
// ============================================================================

TEST_CASE("§19 CSVP oversized source rejected", "[phase157][security]") {
    using namespace security;

    CodeSafetyVerifier csvp;

    // Source > 1MB
    std::string huge(CSVP_MAX_SOURCE_BYTES + 1, 'x');
    auto result = csvp.verify(huge);
    REQUIRE_FALSE(result.approved);
}

// ============================================================================
// §20  AnomalyDetector — baseline then spike
// ============================================================================

TEST_CASE("§20 Anomaly detector baseline then spike", "[phase157][security]") {
    using namespace security;

    AnomalyDetector::Config cfg;
    cfg.min_baseline_samples = 10;
    cfg.sigma_multiplier     = 3.0;
    AnomalyDetector ad(cfg);

    ad.register_module("test_mod");
    REQUIRE(ad.module_count() == 1);

    // Build baseline with low CPU usage (10+ samples)
    for (int i = 0; i < 20; ++i) {
        BehaviorObservation obs;
        obs.cpu_usage    = 0.1 + 0.01 * (i % 3);  // ~10%
        obs.memory_usage = 0.2;
        ad.record_observation("test_mod", obs);
    }

    // Inject a spike
    BehaviorObservation spike;
    spike.cpu_usage    = 0.95;  // 95% CPU — way above baseline
    spike.memory_usage = 0.2;
    ad.record_observation("test_mod", spike);

    auto threats = ad.analyze("test_mod");
    REQUIRE(!threats.empty());

    // Should detect a resource spike
    bool found_spike = false;
    for (const auto& t : threats) {
        if (t.type == ThreatType::RESOURCE_SPIKE) found_spike = true;
    }
    REQUIRE(found_spike);
}

// ============================================================================
// §21  AnomalyDetector — quarantine and release
// ============================================================================

TEST_CASE("§21 Anomaly quarantine and release", "[phase157][security]") {
    using namespace security;

    AnomalyDetector ad;
    ad.register_module("bad_mod");

    // Manual quarantine
    REQUIRE(ad.quarantine("bad_mod", QuarantineReason::MANUAL));
    REQUIRE(ad.is_quarantined("bad_mod"));
    REQUIRE(ad.quarantined_count() == 1);

    auto qmods = ad.quarantined_modules();
    REQUIRE(qmods.size() == 1);
    REQUIRE(qmods[0] == "bad_mod");

    // Release
    REQUIRE(ad.release("bad_mod"));
    REQUIRE_FALSE(ad.is_quarantined("bad_mod"));
    REQUIRE(ad.quarantined_count() == 0);

    // Profile should show correct quarantine reason (now NOT_QUARANTINED)
    auto* prof = ad.get_profile("bad_mod");
    REQUIRE(prof != nullptr);
    REQUIRE(prof->quarantine_reason == QuarantineReason::NOT_QUARANTINED);
}

// ============================================================================
// §22  Cross-subsystem: Training → CheckpointController trigger
// ============================================================================

TEST_CASE("§22 Training informs checkpoint timing", "[phase157][cross]") {
    trainers::MambaTrainer mt(0.01);
    persistence::CheckpointController ctrl(300.f);

    auto data = make_mamba_data(16);

    // Simulate training loop with checkpoint checks
    float sim_time = 0.f;
    int checkpoints = 0;

    for (int epoch = 0; epoch < 5; ++epoch) {
        mt.train_batch(data);
        sim_time += 100.f;

        // Check if we should checkpoint
        if (ctrl.should_checkpoint(sim_time, autonomy::ActionType::SILENT)) {
            ctrl.record_checkpoint(sim_time);
            ++checkpoints;
        }
    }

    // At 500s with 300s interval, should have 1 checkpoint (at 300+)
    REQUIRE(checkpoints >= 1);
    REQUIRE(ctrl.checkpoint_count() >= 1);
}

// ============================================================================
// §23  Cross-subsystem: Security gate → NikolaState validation
// ============================================================================

TEST_CASE("§23 Security verification on generated code", "[phase157][cross]") {
    using namespace security;

    // Simulate the SIE cycle's Gate 1: CSVP check on self-generated code
    CodeSafetyVerifier csvp;

    // Simulated code that the SIE might generate (safe version)
    std::string generated_safe = R"(
        double step(double x, double dt) {
            return x + dt * 0.1;
        }
    )";

    // Simulated malicious code (should be caught)
    std::string generated_bad = R"(
        void exploit() {
            execve("/bin/sh", nullptr, nullptr);
        }
    )";

    auto safe_result = csvp.verify(generated_safe);
    auto bad_result  = csvp.verify(generated_bad);

    REQUIRE(safe_result.approved);
    REQUIRE_FALSE(bad_result.approved);

    // Anomaly detector should track the module
    AnomalyDetector ad;
    ad.register_module("sie_cycle_1");

    BehaviorObservation obs;
    obs.cpu_usage    = 0.3;
    obs.memory_usage = 0.2;
    obs.syscall_count = 50;
    ad.record_observation("sie_cycle_1", obs);

    auto* prof = ad.get_profile("sie_cycle_1");
    REQUIRE(prof != nullptr);
    REQUIRE(prof->total_observations == 1);
}

// ============================================================================
// §24  Cross-subsystem: KVM isolation rules + CSVP constants
// ============================================================================

TEST_CASE("§24 KVM + CSVP security boundaries consistent", "[phase157][cross]") {
    using namespace security;

    // KVM sandbox limits should be stricter than CSVP resource limits
    // KVM memory (512 MB) < CSVP allocation limit (64 MB per allocation)
    // KVM provides the outer boundary, CSVP the inner per-allocation check
    REQUIRE(KVM_CGROUP_MEM_BYTES == 512 * 1024 * 1024);
    REQUIRE(CSVP_MAX_ALLOCATION_BYTES == 64 * 1024 * 1024);
    REQUIRE(CSVP_MAX_ALLOCATION_BYTES < KVM_CGROUP_MEM_BYTES);

    // CSVP source size limit should be reasonable
    REQUIRE(CSVP_MAX_SOURCE_BYTES == 1024 * 1024);
    REQUIRE(CSVP_MAX_FUNCTION_LINES == 500);
    REQUIRE(CSVP_MAX_NESTING_DEPTH == 10);

    // KVM isolation defaults should be strict
    IsolationRules rules;
    REQUIRE(KvmSandbox::validate_isolation(rules));

    // Relaxed rules must fail validation
    IsolationRules relaxed = rules;
    relaxed.network_disabled = false;
    REQUIRE_FALSE(KvmSandbox::validate_isolation(relaxed));
}
