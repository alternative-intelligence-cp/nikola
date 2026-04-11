// ============================================================
// tests/unit/phase154_persistence_test.cpp
// Phase 154 — DMC Checkpointing, NRLE, GGUF Export, Validation
// ============================================================
// v0.1.17 persistence test suite.
//
// §1  NRLE codec (compress/decompress fidelity)
// §2  NRLE float wrappers (quantise round-trip)
// §3  Q9_0 quantization (pack/unpack, block round-trip)
// §4  DMC checkpoint format (header/footer, section CRC)
// §5  NikolaState serialization round-trip
// §6  TorusGrid serialization round-trip (compressed + raw)
// §7  SSM weight serialization round-trip
// §8  Metric tensor serialization round-trip
// §9  Full cognitive checkpoint save/load
// §10 WAL crash safety
// §11 Checkpoint controller (periodic + NAP trigger)
// §12 File validation (corruption detection)
// §13 GGUF export (file structure + metadata)
// ============================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/persistence/nrle_codec.hpp>
#include <nikola/persistence/dmc_checkpoint.hpp>
#include <nikola/persistence/gguf_writer.hpp>
#include <nikola/persistence/checkpoint_validator.hpp>

#include <nikola/autonomy/decision_loop.hpp>
#include <nikola/foundation/toroidal_grid.hpp>
#include <nikola/cognitive/cognitive_core.hpp>
#include <nikola/cognitive/neuroplastic_transformer.hpp>
#include <nikola/physics/metric_tensor.hpp>
#include <nikola/physics/wave_function.hpp>
#include <nikola/system/crc32c.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using Catch::Approx;
namespace fs = std::filesystem;
using namespace nikola;
using namespace nikola::persistence;
using namespace nikola::autonomy;

// ============================================================================
// Test helpers
// ============================================================================

namespace {

/// Temporary directory with automatic cleanup.
struct TmpDir {
    std::string path;
    TmpDir() : path("/tmp/nikola_p154_" + std::to_string(getpid())) {
        fs::create_directories(path);
    }
    ~TmpDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string file(const std::string& name) const {
        return path + "/" + name;
    }
};

/// Create a small seeded TorusGrid (3^9 = 19683 nodes).
foundation::TorusGrid make_test_grid(int n = 3) {
    foundation::TorusGrid grid(foundation::GridConfig::uniform(n));
    // Populate grid with all coordinates in [0, n)^9
    const int total = static_cast<int>(std::pow(n, 9));
    for (int idx = 0; idx < total; ++idx) {
        std::array<int, 9> coords{};
        int tmp = idx;
        for (int d = 0; d < 9; ++d) {
            coords[d] = tmp % n;
            tmp /= n;
        }
        foundation::TorusNode node;
        // Sparse: most nodes near zero (vacuum)
        if (idx % 100 == 0) {
            node.psi = {0.1f * (idx % 7), 0.05f * (idx % 11)};
            node.vel = {0.01f, -0.01f};
            node.resonance = 0.7f;
            node.state_field = 0.3f;
        }
        grid.add_node(coords, node);
    }
    return grid;
}

/// Create a test NikolaState.
NikolaState make_test_state() {
    NikolaState s;
    s.time         = 42.5f;
    s.torus_energy = 3.14f;
    s.dopamine     = 0.65f;
    s.td_error     = -0.12f;
    s.atp          = 0.80f;
    s.boredom      = 0.25f;
    s.entropy      = 2.718f;
    s.last_action  = ActionType::EMIT_THOUGHT;
    s.tokens       = {"resonance", "pattern", "detected"};
    return s;
}

/// Create a small SSMLayer with known weights.
cognitive::SSMLayer make_test_ssm(int H = 16, int I = 9, int O = 32) {
    cognitive::SSMLayer ssm(H, I, O);
    ssm.randomise(42);
    return ssm;
}

/// Create a flat MetricTensorCache.
physics::MetricTensorCache make_test_metric() {
    return physics::MetricTensorCache::flat();
}

}  // anonymous namespace

// ============================================================================
// §1  NRLE codec — basic compress/decompress
// ============================================================================

TEST_CASE("Phase 154 §1 — NRLE codec basics",
          "[persistence][nrle][phase154]") {

    SECTION("empty input") {
        std::vector<int8_t> input;
        auto compressed = nrle_compress(input);
        auto decompressed = nrle_decompress(compressed);
        CHECK(decompressed.empty());
    }

    SECTION("all zeros (maximal sparsity)") {
        std::vector<int8_t> input(10000, 0);
        auto compressed = nrle_compress(input);
        auto decompressed = nrle_decompress(compressed);
        CHECK(decompressed.size() == input.size());
        CHECK(decompressed == input);
        // Compression ratio should be excellent (>100:1)
        CHECK(compressed.size() < 100);
    }

    SECTION("all non-zero (dense)") {
        std::vector<int8_t> input = {1, -2, 3, -4, 4, -3, 2, -1};
        auto compressed = nrle_compress(input);
        auto decompressed = nrle_decompress(compressed);
        CHECK(decompressed == input);
    }

    SECTION("mixed sparse + dense") {
        std::vector<int8_t> input(1000, 0);
        // Sprinkle some non-zero values
        input[0]   =  4;
        input[1]   = -4;
        input[500] =  2;
        input[501] = -2;
        input[502] =  1;
        input[999] =  3;

        auto compressed = nrle_compress(input);
        auto decompressed = nrle_decompress(compressed);
        CHECK(decompressed == input);
        CHECK(compressed.size() < input.size() / 10);  // >10:1
    }

    SECTION("all values [-4..+4] represented") {
        std::vector<int8_t> input;
        for (int v = -4; v <= 4; ++v) {
            input.push_back(static_cast<int8_t>(v));
        }
        auto compressed = nrle_compress(input);
        auto decompressed = nrle_decompress(compressed);
        CHECK(decompressed == input);
    }
}

// ============================================================================
// §2  NRLE float wrappers
// ============================================================================

TEST_CASE("Phase 154 §2 — NRLE float compress/decompress",
          "[persistence][nrle][phase154]") {

    SECTION("sparse float array (>95% zeros)") {
        std::vector<float> input(1000, 0.f);
        input[50]  = 1.5f;
        input[100] = -2.3f;
        input[500] = 0.8f;

        auto compressed = nrle_compress_floats(input.data(), input.size());
        CHECK(compressed.scale > 0.f);
        CHECK(compressed.data.size() < input.size());

        auto decompressed = nrle_decompress_floats(compressed);
        CHECK(decompressed.size() == input.size());

        // Quantisation error bounded by scale/2
        for (size_t i = 0; i < input.size(); ++i) {
            CHECK(decompressed[i] == Approx(input[i]).margin(compressed.scale));
        }
    }

    SECTION("all-zero array") {
        std::vector<float> input(500, 0.f);
        auto compressed = nrle_compress_floats(input.data(), input.size());
        auto decompressed = nrle_decompress_floats(compressed);
        for (size_t i = 0; i < input.size(); ++i) {
            CHECK(decompressed[i] == 0.f);
        }
    }
}

// ============================================================================
// §3  Q9_0 quantization
// ============================================================================

TEST_CASE("Phase 154 §3 — Q9_0 balanced nonary quantization",
          "[persistence][q9_0][phase154]") {

    SECTION("pack/unpack 5 trits identity") {
        int8_t trits[5] = {-4, -2, 0, 2, 4};
        uint16_t packed = pack_5_trits(trits);
        int8_t unpacked[5];
        unpack_5_trits(packed, unpacked);
        for (int i = 0; i < 5; ++i) {
            CHECK(unpacked[i] == trits[i]);
        }
    }

    SECTION("pack_5_trits max value < 65536") {
        int8_t max_trits[5] = {4, 4, 4, 4, 4};
        uint16_t packed = pack_5_trits(max_trits);
        CHECK(packed == 59048u);  // 8 + 8*9 + 8*81 + 8*729 + 8*6561
        CHECK(static_cast<uint32_t>(packed) < 65536u);
    }

    SECTION("Q9_0 block round-trip") {
        float input[32];
        for (int i = 0; i < 32; ++i) {
            input[i] = static_cast<float>(i - 16) * 0.1f;
        }

        BlockQ9_0 block;
        quantize_q9_0_block(input, 32, block);
        CHECK(block.scale > 0.f);

        float output[32] = {};
        dequantize_q9_0_block(block, output, 32);

        // Error bounded by scale / 2
        for (int i = 0; i < 32; ++i) {
            CHECK(output[i] == Approx(input[i]).margin(block.scale * 0.6f));
        }
    }

    SECTION("Q9_0 array round-trip") {
        const size_t N = 100;
        std::vector<float> input(N);
        for (size_t i = 0; i < N; ++i) {
            input[i] = std::sin(static_cast<float>(i) * 0.1f) * 2.f;
        }

        auto blocks = quantize_q9_0(input.data(), N);
        auto output = dequantize_q9_0(blocks.data(), blocks.size(), N);
        CHECK(output.size() == N);

        for (size_t i = 0; i < N; ++i) {
            CHECK(output[i] == Approx(input[i]).margin(1.0f));
        }
    }
}

// ============================================================================
// §4  DMC format structures
// ============================================================================

TEST_CASE("Phase 154 §4 — DMC format structure sizes",
          "[persistence][dmc][phase154]") {

    SECTION("NikHeader is 64 bytes") {
        CHECK(sizeof(NikHeader) == 64);
    }

    SECTION("SectionHeader is 24 bytes") {
        CHECK(sizeof(SectionHeader) == 24);
    }

    SECTION("NikFooter is 128 bytes") {
        CHECK(sizeof(NikFooter) == 128);
    }

    SECTION("magic constant is NIKO") {
        CHECK(NIK_MAGIC == 0x4E494B4Fu);
        const char* magic_str = reinterpret_cast<const char*>(&NIK_MAGIC);
        CHECK(magic_str[0] == 'O');  // little-endian: reversed
        CHECK(magic_str[1] == 'K');
        CHECK(magic_str[2] == 'I');
        CHECK(magic_str[3] == 'N');
    }
}

// ============================================================================
// §5  NikolaState serialization
// ============================================================================

TEST_CASE("Phase 154 §5 — NikolaState round-trip",
          "[persistence][dmc][phase154]") {

    SECTION("pack/unpack exact scalars") {
        auto state = make_test_state();
        auto packed = pack_nikola_state(state);
        CHECK(packed.size() > 28);  // 7 floats + action + tokens

        NikolaState loaded;
        unpack_nikola_state(packed.data(), packed.size(), loaded);

        CHECK(loaded.time         == state.time);
        CHECK(loaded.torus_energy == state.torus_energy);
        CHECK(loaded.dopamine     == state.dopamine);
        CHECK(loaded.td_error     == state.td_error);
        CHECK(loaded.atp          == state.atp);
        CHECK(loaded.boredom      == state.boredom);
        CHECK(loaded.entropy      == state.entropy);
        CHECK(loaded.last_action  == state.last_action);
    }

    SECTION("pack/unpack tokens") {
        auto state = make_test_state();
        auto packed = pack_nikola_state(state);

        NikolaState loaded;
        unpack_nikola_state(packed.data(), packed.size(), loaded);

        CHECK(loaded.tokens.size() == state.tokens.size());
        for (size_t i = 0; i < state.tokens.size(); ++i) {
            CHECK(loaded.tokens[i] == state.tokens[i]);
        }
    }

    SECTION("empty tokens") {
        NikolaState state;
        state.tokens.clear();
        auto packed = pack_nikola_state(state);

        NikolaState loaded;
        unpack_nikola_state(packed.data(), packed.size(), loaded);
        CHECK(loaded.tokens.empty());
    }
}

// ============================================================================
// §6  TorusGrid serialization
// ============================================================================

TEST_CASE("Phase 154 §6 — TorusGrid round-trip",
          "[persistence][dmc][phase154]") {

    SECTION("compressed round-trip (NRLE)") {
        auto grid = make_test_grid();
        const size_t N = grid.num_active_nodes();
        CHECK(N == 19683);  // 3^9

        auto packed = pack_torus_grid(grid, /*compress=*/true);
        CHECK(!packed.empty());

        // Create identical topology grid to unpack into
        auto loaded_grid = make_test_grid();
        // Zero out fields
        std::memset(loaded_grid.psi_real(), 0, N * sizeof(float));
        std::memset(loaded_grid.psi_imag(), 0, N * sizeof(float));

        unpack_torus_grid(packed.data(), packed.size(), loaded_grid);

        // Compressed → quantisation error exists but bounded
        float max_err = grids_max_error(grid, loaded_grid);
        CHECK(max_err < 1.0f);  // Within NRLE quantisation bounds
    }

    SECTION("uncompressed round-trip (exact)") {
        auto grid = make_test_grid();
        const size_t N = grid.num_active_nodes();

        auto packed = pack_torus_grid(grid, /*compress=*/false);
        CHECK(!packed.empty());

        auto loaded_grid = make_test_grid();
        unpack_torus_grid(packed.data(), packed.size(), loaded_grid);

        // Uncompressed → bitwise identical
        float max_err = grids_max_error(grid, loaded_grid);
        CHECK(max_err == 0.f);
    }

    SECTION("compression effective on sparse data") {
        auto grid = make_test_grid();  // Mostly zeros
        auto compressed = pack_torus_grid(grid, /*compress=*/true);
        auto raw = pack_torus_grid(grid, /*compress=*/false);
        CHECK(compressed.size() < raw.size());
    }
}

// ============================================================================
// §7  SSM weight serialization
// ============================================================================

TEST_CASE("Phase 154 §7 — SSM weight round-trip",
          "[persistence][dmc][phase154]") {

    SECTION("pack/unpack exact (FP32)") {
        auto ssm = make_test_ssm();
        auto packed = pack_ssm_weights(ssm);
        CHECK(!packed.empty());

        cognitive::SSMLayer loaded(ssm.hidden_dim(), ssm.input_dim(),
                                   ssm.output_dim());
        unpack_ssm_weights(packed.data(), packed.size(), loaded);

        CHECK(ssm_max_error(ssm, loaded) == 0.f);
    }

    SECTION("dimension mismatch throws") {
        auto ssm = make_test_ssm();
        auto packed = pack_ssm_weights(ssm);

        cognitive::SSMLayer wrong(32, 9, 32);  // Different H
        CHECK_THROWS(unpack_ssm_weights(packed.data(), packed.size(), wrong));
    }
}

// ============================================================================
// §8  Metric tensor serialization
// ============================================================================

TEST_CASE("Phase 154 §8 — Metric tensor round-trip",
          "[persistence][dmc][phase154]") {

    SECTION("flat metric exact round-trip") {
        auto mtc = make_test_metric();
        CHECK(mtc.is_valid());

        auto packed = pack_metric_tensor(mtc);

        physics::MetricTensorCache loaded;
        unpack_metric_tensor(packed.data(), packed.size(), loaded);

        CHECK(loaded.is_valid());
        CHECK(metric_max_error(mtc, loaded) == 0.0);
    }

    SECTION("invalid metric round-trip") {
        physics::MetricTensorCache mtc;  // invalid by default
        CHECK(!mtc.is_valid());

        auto packed = pack_metric_tensor(mtc);
        physics::MetricTensorCache loaded;
        unpack_metric_tensor(packed.data(), packed.size(), loaded);

        CHECK(!loaded.is_valid());
    }
}

// ============================================================================
// §9  Full cognitive checkpoint save/load
// ============================================================================

TEST_CASE("Phase 154 §9 — Full checkpoint save/load",
          "[persistence][dmc][phase154]") {

    TmpDir tmp;

    SECTION("save and load all sections") {
        auto state  = make_test_state();
        auto grid   = make_test_grid();
        auto ssm    = make_test_ssm();
        auto metric = make_test_metric();

        CognitiveSnapshot snap;
        snap.state  = state;
        snap.grid   = &grid;
        snap.ssm    = &ssm;
        snap.metric = &metric;
        snap.npt    = nullptr;  // NPT tested separately

        const std::string nik_path = tmp.file("test.nik");
        const size_t bytes = save_checkpoint(nik_path, snap);
        CHECK(bytes > NIK_HEADER_SIZE + NIK_FOOTER_SIZE);

        // Verify file exists
        CHECK(fs::exists(nik_path));
        CHECK(fs::file_size(nik_path) == bytes);

        // Load
        NikolaState loaded_state;
        auto loaded_grid = make_test_grid();
        cognitive::SSMLayer loaded_ssm(ssm.hidden_dim(), ssm.input_dim(),
                                        ssm.output_dim());
        auto loaded_metric = physics::MetricTensorCache::flat();

        CognitiveSnapshot loaded;
        loaded.state  = loaded_state;
        loaded.grid   = &loaded_grid;
        loaded.ssm    = &loaded_ssm;
        loaded.metric = &loaded_metric;

        load_checkpoint(nik_path, loaded);

        // Verify state
        CHECK(states_match(state, loaded.state));

        // SSM exact
        CHECK(ssm_max_error(ssm, loaded_ssm) == 0.f);

        // Metric exact
        CHECK(metric_max_error(metric, loaded_metric) == 0.0);
    }

    SECTION("WAL cleaned up after successful save") {
        auto state = make_test_state();
        CognitiveSnapshot snap;
        snap.state = state;

        const std::string nik_path = tmp.file("clean.nik");
        save_checkpoint(nik_path, snap);

        CHECK(fs::exists(nik_path));
        CHECK(!fs::exists(nik_path + ".wal"));
        CHECK(!fs::exists(nik_path + ".tmp"));
    }

    SECTION("invalid magic rejected") {
        auto state = make_test_state();
        CognitiveSnapshot snap;
        snap.state = state;

        const std::string nik_path = tmp.file("bad_magic.nik");
        save_checkpoint(nik_path, snap);

        // Corrupt the magic bytes
        {
            std::fstream f(nik_path, std::ios::binary | std::ios::in |
                           std::ios::out);
            uint32_t bad_magic = 0xDEADBEEF;
            f.write(reinterpret_cast<const char*>(&bad_magic), 4);
        }

        CognitiveSnapshot loaded;
        CHECK_THROWS(load_checkpoint(nik_path, loaded));
    }
}

// ============================================================================
// §10  WAL crash safety
// ============================================================================

TEST_CASE("Phase 154 §10 — WAL crash safety",
          "[persistence][wal][phase154]") {

    TmpDir tmp;

    SECTION("WAL write and commit") {
        const std::string wal_path = tmp.file("test.wal");
        WriteAheadLog wal(wal_path);
        CHECK(wal.open());

        std::vector<uint8_t> payload = {0x01, 0x02, 0x03, 0x04};
        wal.append(WAL_UPDATE, payload);
        wal.commit();

        CHECK(fs::exists(wal_path));
        CHECK(fs::file_size(wal_path) > 0);
    }

    SECTION("WAL close and remove") {
        const std::string wal_path = tmp.file("removable.wal");
        WriteAheadLog wal(wal_path);
        wal.open();
        wal.append(WAL_UPDATE, {0xFF});
        wal.close_and_remove();

        CHECK(!fs::exists(wal_path));
    }
}

// ============================================================================
// §11  Checkpoint controller (periodic + NAP trigger)
// ============================================================================

TEST_CASE("Phase 154 §11 — Checkpoint controller triggers",
          "[persistence][controller][phase154]") {

    SECTION("periodic trigger at 300s") {
        CheckpointController ctrl(300.f);

        // Not yet time
        CHECK(!ctrl.should_checkpoint(100.f, ActionType::SILENT));
        CHECK(!ctrl.should_checkpoint(200.f, ActionType::SILENT));

        // At 300s boundary
        CHECK(ctrl.should_checkpoint(300.f, ActionType::SILENT));
        ctrl.record_checkpoint(300.f);

        // Not again until 600s
        CHECK(!ctrl.should_checkpoint(400.f, ActionType::SILENT));
        CHECK(ctrl.should_checkpoint(600.f, ActionType::SILENT));
    }

    SECTION("NAP trigger always fires") {
        CheckpointController ctrl(300.f);

        // NAP at any time
        CHECK(ctrl.should_checkpoint(10.f, ActionType::NAP));
        CHECK(ctrl.should_checkpoint(50.f, ActionType::NAP));
    }

    SECTION("checkpoint count tracking") {
        CheckpointController ctrl(100.f);
        CHECK(ctrl.checkpoint_count() == 0);

        ctrl.record_checkpoint(100.f);
        CHECK(ctrl.checkpoint_count() == 1);

        ctrl.record_checkpoint(200.f);
        CHECK(ctrl.checkpoint_count() == 2);
    }
}

// ============================================================================
// §12  File validation (corruption detection)
// ============================================================================

TEST_CASE("Phase 154 §12 — Checkpoint file validation",
          "[persistence][validation][phase154]") {

    TmpDir tmp;

    SECTION("valid file passes validation") {
        auto state = make_test_state();
        auto grid  = make_test_grid();
        auto ssm   = make_test_ssm();
        auto metric = make_test_metric();

        CognitiveSnapshot snap;
        snap.state  = state;
        snap.grid   = &grid;
        snap.ssm    = &ssm;
        snap.metric = &metric;

        const std::string path = tmp.file("valid.nik");
        save_checkpoint(path, snap);

        auto result = validate_checkpoint_file(path);
        CHECK(result.valid);
        CHECK(result.merkle_ok);
        CHECK(result.sections_checked == 4);  // state + grid + ssm + metric
        CHECK(result.crc_ok == 4);
    }

    SECTION("corrupted payload detected") {
        auto state = make_test_state();
        CognitiveSnapshot snap;
        snap.state = state;

        const std::string path = tmp.file("corrupt.nik");
        save_checkpoint(path, snap);

        // Corrupt a byte in the first section payload
        {
            std::fstream f(path, std::ios::binary | std::ios::in |
                           std::ios::out);
            // Header (64) + SectionHeader (24) + 1 byte into payload
            f.seekp(64 + 24 + 1);
            char bad = 0xFF;
            f.write(&bad, 1);
        }

        auto result = validate_checkpoint_file(path);
        CHECK(!result.valid);
        CHECK(result.error.find("CRC") != std::string::npos);
    }

    SECTION("nonexistent file") {
        auto result = validate_checkpoint_file("/tmp/nonexistent_nik_file.nik");
        CHECK(!result.valid);
    }

    SECTION("truncated file") {
        const std::string path = tmp.file("truncated.nik");
        {
            std::ofstream f(path, std::ios::binary);
            f.write("NIKO", 4);  // Just magic, incomplete
        }
        auto result = validate_checkpoint_file(path);
        CHECK(!result.valid);
    }
}

// ============================================================================
// §13  GGUF export
// ============================================================================

TEST_CASE("Phase 154 §13 — GGUF export",
          "[persistence][gguf][phase154]") {

    TmpDir tmp;

    SECTION("export creates valid GGUF file") {
        auto grid   = make_test_grid();
        auto ssm    = make_test_ssm();
        auto metric = make_test_metric();

        const std::string path = tmp.file("nikola.gguf");
        const size_t bytes = export_gguf(path, grid, ssm, metric);
        CHECK(bytes > 0);
        CHECK(fs::exists(path));

        // GGUF v3 magic: 0x46554747 ("GGUF" little-endian)
        std::ifstream in(path, std::ios::binary);
        uint32_t magic = 0;
        in.read(reinterpret_cast<char*>(&magic), 4);
        CHECK(magic == 0x46554747u);
    }

    SECTION("Q9_0 compression ratio") {
        // 32 weights per block × 4 bytes = 128 bytes FP32
        // 1 Q9_0 block = 20 bytes
        // Ratio: 128/20 = 6.4:1
        const size_t n = 3200;
        std::vector<float> data(n);
        for (size_t i = 0; i < n; ++i) {
            data[i] = std::sin(static_cast<float>(i) * 0.01f);
        }
        auto blocks = quantize_q9_0(data.data(), n);
        const size_t q9_bytes = blocks.size() * sizeof(BlockQ9_0);
        const size_t fp32_bytes = n * sizeof(float);

        // Should be ~6x compression
        CHECK(q9_bytes < fp32_bytes / 5);
    }
}
