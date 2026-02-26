/// @file   phase102_gguf_model_streamer_test.cpp
/// @brief  Phase 102 — GAP-014: GGUF Model Streaming (GgufModelStreamer)
///
/// Tests ggml-base GGUF v3 streaming reader over tests/assets/nikola_test.gguf
/// Generated file layout:
///   KV  [0] general.name = "nikola_test"  (GGUF_TYPE_STRING)
///   [0] weight_a  F32  [4, 8]       32 elem   off=0
///   [1] bias_b    F32  [4]           4 elem   off=128
///   [2] gamma_c   F32  [2, 4, 4]    32 elem   off=160

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "nikola/persistence/gguf_model_streamer.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace nikola::persistence;

#ifndef NIKOLA_TEST_ASSETS_DIR
#  error "NIKOLA_TEST_ASSETS_DIR must be defined by CMakeLists.txt"
#endif

static const fs::path kGgufPath =
    fs::path(NIKOLA_TEST_ASSETS_DIR) / "nikola_test.gguf";

// ============================================================
// TEST CASE 1: Architecture / enum constants
// ============================================================
TEST_CASE("GgufModelStreamer — enum and constant checks",
          "[phase102][constants]")
{
    SECTION("GGML_MAX_DIMS is 4") {
        CHECK(GGML_MAX_DIMS == 4);
    }
    SECTION("GGML_TYPE_F32 is 0") {
        CHECK(static_cast<int>(GGML_TYPE_F32) == 0);
    }
    SECTION("GgmlType::F32 maps to 0") {
        CHECK(static_cast<uint32_t>(GgmlType::F32) == 0u);
    }
    SECTION("GgmlType::F16 maps to 1") {
        CHECK(static_cast<uint32_t>(GgmlType::F16) == 1u);
    }
    SECTION("to_ggml_type round-trips GGML_TYPE_F32") {
        CHECK(to_ggml_type(GGML_TYPE_F32) == GgmlType::F32);
    }
    SECTION("DEFAULT_GGML_MEM is at least 1 MB") {
        CHECK(GgufModelStreamer::DEFAULT_GGML_MEM >= 1'048'576u);
    }
}

// ============================================================
// TEST CASE 2: ggml_type_name helper
// ============================================================
TEST_CASE("GgufModelStreamer — ggml_type_name helper",
          "[phase102][helpers]")
{
    SECTION("F32 → \"F32\"") {
        CHECK(ggml_type_name(GgmlType::F32)  == "F32");
    }
    SECTION("F16 → \"F16\"") {
        CHECK(ggml_type_name(GgmlType::F16)  == "F16");
    }
    SECTION("Q4_0 → \"Q4_0\"") {
        CHECK(ggml_type_name(GgmlType::Q4_0) == "Q4_0");
    }
    SECTION("Q8_0 → \"Q8_0\"") {
        CHECK(ggml_type_name(GgmlType::Q8_0) == "Q8_0");
    }
    SECTION("UNKNOWN → \"UNKNOWN\"") {
        CHECK(ggml_type_name(GgmlType::UNKNOWN) == "UNKNOWN");
    }
}

// ============================================================
// TEST CASE 3: Bad-path throws
// ============================================================
TEST_CASE("GgufModelStreamer — bad path throws",
          "[phase102][open][error]")
{
    SECTION("completely missing file throws std::runtime_error") {
        CHECK_THROWS_AS(
            GgufModelStreamer{"/tmp/this_gguf_does_not_exist_xyz.gguf"},
            std::runtime_error);
    }
    SECTION("empty path throws") {
        CHECK_THROWS(GgufModelStreamer{""});
    }
    SECTION("directory path throws") {
        CHECK_THROWS(GgufModelStreamer{"/tmp"});
    }
}

// ============================================================
// TEST CASE 4: Successful open
// ============================================================
TEST_CASE("GgufModelStreamer — opens nikola_test.gguf",
          "[phase102][open]")
{
    REQUIRE(fs::exists(kGgufPath));

    SECTION("constructor does not throw") {
        CHECK_NOTHROW(GgufModelStreamer{kGgufPath});
    }
    SECTION("file has non-zero size on disk") {
        CHECK(fs::file_size(kGgufPath) > 0u);
    }
    SECTION("file has .gguf extension") {
        CHECK(kGgufPath.extension() == ".gguf");
    }
    SECTION("path() accessor returns the original path") {
        GgufModelStreamer s{kGgufPath};
        CHECK(s.path() == kGgufPath);
    }
}

// ============================================================
// TEST CASE 5: GgufMeta fields
// ============================================================
TEST_CASE("GgufModelStreamer — meta() fields", "[phase102][meta]")
{
    GgufModelStreamer s{kGgufPath};
    auto m = s.meta();

    SECTION("version is 3 (GGUF v3)") {
        CHECK(m.version == 3u);
    }
    SECTION("alignment is 32 (default)") {
        CHECK(m.alignment == 32u);
    }
    SECTION("n_kv is 1") {
        CHECK(m.n_kv == 1);
    }
    SECTION("n_tensors is 3") {
        CHECK(m.n_tensors == 3);
    }
    SECTION("data_offset is positive (non-zero header)") {
        CHECK(m.data_offset > 0u);
    }
    SECTION("n_tensors matches s.n_tensors()") {
        CHECK(m.n_tensors == s.n_tensors());
    }
}

// ============================================================
// TEST CASE 6: Tensor info – all three tensors
// ============================================================
TEST_CASE("GgufModelStreamer — tensor_info() shape and type",
          "[phase102][tensors]")
{
    GgufModelStreamer s{kGgufPath};

    SECTION("n_tensors() is 3") {
        CHECK(s.n_tensors() == 3);
    }

    SECTION("tensor 0: weight_a F32 [4,8] 32 elem") {
        const auto& t = s.tensor_info(0);
        CHECK(t.name        == "weight_a");
        CHECK(t.type        == GgmlType::F32);
        CHECK(t.n_dims      == 2u);
        CHECK(t.ne[0]       == 4);
        CHECK(t.ne[1]       == 8);
        CHECK(t.n_elements  == 32);
        CHECK(t.data_offset == 0u);
        CHECK(t.bytes_as_f32() == 32u * sizeof(float));
    }

    SECTION("tensor 1: bias_b F32 [4] 4 elem") {
        const auto& t = s.tensor_info(1);
        CHECK(t.name        == "bias_b");
        CHECK(t.type        == GgmlType::F32);
        CHECK(t.n_dims      == 1u);
        CHECK(t.ne[0]       == 4);
        CHECK(t.n_elements  == 4);
        CHECK(t.data_offset == 128u);
    }

    SECTION("tensor 2: gamma_c F32 [2,4,4] 32 elem") {
        const auto& t = s.tensor_info(2);
        CHECK(t.name        == "gamma_c");
        CHECK(t.type        == GgmlType::F32);
        CHECK(t.n_dims      == 3u);
        CHECK(t.ne[0]       == 2);
        CHECK(t.ne[1]       == 4);
        CHECK(t.ne[2]       == 4);
        CHECK(t.n_elements  == 32);
        CHECK(t.data_offset == 160u);
    }

    SECTION("out-of-range index throws std::out_of_range") {
        CHECK_THROWS_AS(s.tensor_info(3),  std::out_of_range);
        CHECK_THROWS_AS(s.tensor_info(-1), std::out_of_range);
    }
}

// ============================================================
// TEST CASE 7: find_tensor() lookup
// ============================================================
TEST_CASE("GgufModelStreamer — find_tensor()", "[phase102][find]")
{
    GgufModelStreamer s{kGgufPath};

    SECTION("find 'weight_a' returns index 0") {
        auto idx = s.find_tensor("weight_a");
        REQUIRE(idx.has_value());
        CHECK(*idx == 0);
    }
    SECTION("find 'bias_b' returns index 1") {
        auto idx = s.find_tensor("bias_b");
        REQUIRE(idx.has_value());
        CHECK(*idx == 1);
    }
    SECTION("find 'gamma_c' returns index 2") {
        auto idx = s.find_tensor("gamma_c");
        REQUIRE(idx.has_value());
        CHECK(*idx == 2);
    }
    SECTION("find non-existent tensor returns nullopt") {
        CHECK_FALSE(s.find_tensor("nonexistent_xyz").has_value());
    }
    SECTION("find_tensor result consistent with tensor_info name") {
        auto idx = s.find_tensor("bias_b");
        REQUIRE(idx.has_value());
        CHECK(s.tensor_info(*idx).name == "bias_b");
    }
}

// ============================================================
// TEST CASE 8: KV metadata
// ============================================================
TEST_CASE("GgufModelStreamer — KV metadata", "[phase102][kv]")
{
    GgufModelStreamer s{kGgufPath};

    SECTION("n_kv() is 1") {
        CHECK(s.n_kv() == 1);
    }
    SECTION("kv_key(0) is 'general.name'") {
        CHECK(s.kv_key(0) == "general.name");
    }
    SECTION("kv_type(0) is GGUF_TYPE_STRING") {
        CHECK(s.kv_type(0) == GGUF_TYPE_STRING);
    }
    SECTION("kv_string_value(0) is 'nikola_test'") {
        CHECK(s.kv_string_value(0) == "nikola_test");
    }
    SECTION("kv out-of-range throws std::out_of_range") {
        CHECK_THROWS_AS(s.kv_key(1),  std::out_of_range);
        CHECK_THROWS_AS(s.kv_key(-1), std::out_of_range);
    }
}

// ============================================================
// TEST CASE 9: for_each_tensor() streaming iterator
// ============================================================
TEST_CASE("GgufModelStreamer — for_each_tensor()",
          "[phase102][streaming]")
{
    GgufModelStreamer s{kGgufPath};

    SECTION("visits exactly 3 tensors") {
        int count = 0;
        s.for_each_tensor([&](int64_t /*idx*/, const GgufTensorInfo& /*t*/) {
            ++count;
        });
        CHECK(count == 3);
    }

    SECTION("indices are 0, 1, 2 in order") {
        std::vector<int64_t> seen_idx;
        s.for_each_tensor([&](int64_t idx, const GgufTensorInfo& /*t*/) {
            seen_idx.push_back(idx);
        });
        REQUIRE(seen_idx.size() == 3u);
        CHECK(seen_idx[0] == 0);
        CHECK(seen_idx[1] == 1);
        CHECK(seen_idx[2] == 2);
    }

    SECTION("tensor names from callback match direct access") {
        std::vector<std::string> names;
        s.for_each_tensor([&](int64_t /*idx*/, const GgufTensorInfo& t) {
            names.push_back(t.name);
        });
        REQUIRE(names.size() == 3u);
        CHECK(names[0] == "weight_a");
        CHECK(names[1] == "bias_b");
        CHECK(names[2] == "gamma_c");
    }

    SECTION("all tensors visited have F32 type") {
        s.for_each_tensor([&](int64_t /*idx*/, const GgufTensorInfo& t) {
            CHECK(t.type == GgmlType::F32);
        });
    }

    SECTION("total element count across all tensors is 68") {
        int64_t total = 0;
        s.for_each_tensor([&](int64_t, const GgufTensorInfo& t) {
            total += t.n_elements;
        });
        // 32 (weight_a) + 4 (bias_b) + 32 (gamma_c) = 68
        CHECK(total == 68);
    }
}
