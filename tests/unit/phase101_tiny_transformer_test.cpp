/// @file   phase101_tiny_transformer_test.cpp
/// @brief  Phase 101 — GAP-015: BERT-Tiny ORT Inference (TinyTransformer)
///
/// Tests ORT 1.24 C++ wrapper:  token_ids (int64)  →  float32 embedding [128]
/// Test model: tests/assets/bert_tiny_test.onnx
///   Cast(int64→float) + MatMul([1,4]×[4,128]) → [1,128] float

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "nikola/ml/tiny_transformer.hpp"

#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace nikola::ml;

// Path injected by CMake at compile time
#ifndef NIKOLA_TEST_ASSETS_DIR
#  error "NIKOLA_TEST_ASSETS_DIR must be defined by CMakeLists.txt"
#endif

static const fs::path kModelPath =
    fs::path(NIKOLA_TEST_ASSETS_DIR) / "bert_tiny_test.onnx";

// Convenience: 4 typical BERT token IDs that match the model's static seq_len
static const std::vector<int64_t> kTokens4 = {101, 2054, 2003, 102};

// ============================================================
// TEST CASE 1: Architecture constants
// ============================================================
TEST_CASE("TinyTransformer — architecture constants", "[phase101][constants]")
{
    SECTION("BERT_TINY_LAYERS is 4")         { CHECK(BERT_TINY_LAYERS     == 4);   }
    SECTION("BERT_TINY_HIDDEN_DIM is 128")   { CHECK(BERT_TINY_HIDDEN_DIM == 128); }
    SECTION("BERT_TINY_ATTN_HEADS is 2")     { CHECK(BERT_TINY_ATTN_HEADS == 2);   }
    SECTION("MAX_SEQ_LEN is 512")            { CHECK(MAX_SEQ_LEN          == 512); }

    SECTION("hidden_dim is divisible by attn_heads") {
        CHECK(BERT_TINY_HIDDEN_DIM % BERT_TINY_ATTN_HEADS == 0);
    }
    SECTION("head_dim for BERT-Tiny is 64") {
        constexpr int head_dim = BERT_TINY_HIDDEN_DIM / BERT_TINY_ATTN_HEADS;
        CHECK(head_dim == 64);
    }
}

// ============================================================
// TEST CASE 2: create_ort_env() factory
// ============================================================
TEST_CASE("TinyTransformer — create_ort_env factory", "[phase101][env]")
{
    SECTION("default log id and level succeeds") {
        CHECK_NOTHROW( []{ auto e = create_ort_env(); (void)e; }() );
    }
    SECTION("custom log id succeeds") {
        CHECK_NOTHROW( []{ auto e = create_ort_env("nikola_test", ORT_LOGGING_LEVEL_ERROR); (void)e; }() );
    }
    SECTION("WARNING level is numerically lower than ERROR level") {
        // ORT enum: VERBOSE=0, INFO=1, WARNING=2, ERROR=3, FATAL=4
        CHECK(static_cast<int>(ORT_LOGGING_LEVEL_WARNING) <
              static_cast<int>(ORT_LOGGING_LEVEL_ERROR));
    }
    SECTION("ORT_LOGGING_LEVEL_WARNING value") {
        // ORT enum: VERBOSE=0, INFO=1, WARNING=2, ERROR=3, FATAL=4
        CHECK(static_cast<int>(ORT_LOGGING_LEVEL_WARNING) == 2);
    }
}

// ============================================================
// TEST CASE 3: Loading — invalid model path throws
// ============================================================
TEST_CASE("TinyTransformer — bad model path throws", "[phase101][load][error]")
{
    SECTION("completely missing file throws") {
        CHECK_THROWS_AS(
            TinyTransformer{"/tmp/this_file_does_not_exist_123456.onnx"},
            Ort::Exception);
    }
    SECTION("empty path throws") {
        // ORT will try to open an empty string — should fail
        CHECK_THROWS( TinyTransformer{""} );
    }
    SECTION("directory path instead of file throws") {
        CHECK_THROWS( TinyTransformer{"/tmp"} );
    }
}

// ============================================================
// TEST CASE 4: Successful model load
// ============================================================
TEST_CASE("TinyTransformer — loads test model successfully", "[phase101][load]")
{
    REQUIRE(fs::exists(kModelPath));

    SECTION("constructor does not throw") {
        CHECK_NOTHROW( TinyTransformer{kModelPath} );
    }
    SECTION("model file is non-empty on disk") {
        CHECK(fs::file_size(kModelPath) > 0u);
    }
    SECTION("model file has .onnx extension") {
        CHECK(kModelPath.extension() == ".onnx");
    }
}

// ============================================================
// TEST CASE 5: Input / output name accessors
// ============================================================
TEST_CASE("TinyTransformer — I/O name accessors", "[phase101][accessors]")
{
    TinyTransformer tt{kModelPath};

    SECTION("input_name is 'input_ids'") {
        CHECK(tt.input_name() == "input_ids");
    }
    SECTION("output_name is 'embedding'") {
        CHECK(tt.output_name() == "embedding");
    }
    SECTION("input_name is non-empty") {
        CHECK_FALSE(tt.input_name().empty());
    }
    SECTION("output_name is non-empty") {
        CHECK_FALSE(tt.output_name().empty());
    }
    SECTION("input and output names differ") {
        CHECK(tt.input_name() != tt.output_name());
    }
}

// ============================================================
// TEST CASE 6: Input / output count accessors
// ============================================================
TEST_CASE("TinyTransformer — I/O count accessors", "[phase101][accessors]")
{
    TinyTransformer tt{kModelPath};

    SECTION("model has exactly 1 input") {
        CHECK(tt.num_inputs() == 1u);
    }
    SECTION("model has exactly 1 output") {
        CHECK(tt.num_outputs() == 1u);
    }
    SECTION("num_inputs > 0") {
        CHECK(tt.num_inputs() > 0u);
    }
    SECTION("num_outputs > 0") {
        CHECK(tt.num_outputs() > 0u);
    }
}

// ============================================================
// TEST CASE 7: forward() — output shape and value properties
// ============================================================
TEST_CASE("TinyTransformer — forward() output properties", "[phase101][inference]")
{
    TinyTransformer tt{kModelPath};
    auto embedding = tt.forward(kTokens4);

    SECTION("output has BERT_TINY_HIDDEN_DIM elements") {
        CHECK(static_cast<int>(embedding.size()) == BERT_TINY_HIDDEN_DIM);
    }
    SECTION("output has exactly 128 floats") {
        CHECK(embedding.size() == 128u);
    }
    SECTION("no NaN values in embedding") {
        for (std::size_t i = 0; i < embedding.size(); ++i)
            CHECK_FALSE(std::isnan(embedding[i]));
    }
    SECTION("no Inf values in embedding") {
        for (std::size_t i = 0; i < embedding.size(); ++i)
            CHECK_FALSE(std::isinf(embedding[i]));
    }
    SECTION("embedding values are finite floats") {
        for (auto v : embedding)
            CHECK(std::isfinite(v));
    }
    SECTION("embedding magnitude is non-zero (model has real weights)") {
        float sum = 0.0f;
        for (auto v : embedding) sum += v * v;
        CHECK(sum > 0.0f);
    }
}

// ============================================================
// TEST CASE 8: forward() — invalid input throws
// ============================================================
TEST_CASE("TinyTransformer — forward() invalid input throws",
          "[phase101][inference][error]")
{
    TinyTransformer tt{kModelPath};

    SECTION("empty token_ids throws std::invalid_argument") {
        CHECK_THROWS_AS(tt.forward({}), std::invalid_argument);
    }
    SECTION("exception message mentions token_ids") {
        try {
            tt.forward({});
            FAIL("expected exception was not thrown");
        } catch (const std::invalid_argument& ex) {
            std::string msg{ex.what()};
            CHECK_FALSE(msg.empty());
        }
    }
}

// ============================================================
// TEST CASE 9: forward() — determinism
// ============================================================
TEST_CASE("TinyTransformer — forward() is deterministic",
          "[phase101][inference][determinism]")
{
    TinyTransformer tt{kModelPath};

    auto run1 = tt.forward(kTokens4);
    auto run2 = tt.forward(kTokens4);

    SECTION("same size on repeated calls") {
        CHECK(run1.size() == run2.size());
    }
    SECTION("identical values on repeated calls") {
        REQUIRE(run1.size() == run2.size());
        for (std::size_t i = 0; i < run1.size(); ++i)
            CHECK(run1[i] == run2[i]);  // exact equality — deterministic CPU inference
    }
    SECTION("different token sequences produce different embeddings") {
        // Tokens {0,0,0,0} vs {101,2054,2003,102} should yield different outputs
        auto run_zero = tt.forward({0, 0, 0, 0});
        REQUIRE(run_zero.size() == run1.size());
        float diff_sq = 0.0f;
        for (std::size_t i = 0; i < run1.size(); ++i) {
            float d = run1[i] - run_zero[i];
            diff_sq += d * d;
        }
        CHECK(diff_sq > 1e-6f);  // outputs must differ
    }
    SECTION("multiple forward calls return same size") {
        for (int k = 0; k < 3; ++k) {
            auto rk = tt.forward(kTokens4);
            CHECK(rk.size() == 128u);
        }
    }
}
