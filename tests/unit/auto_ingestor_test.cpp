/**
 * @file tests/unit/auto_ingestor_test.cpp
 * @brief v0.2.2 Phase 2 — AutoIngestor unit tests.
 *
 * Tests chunking strategies and ingestion pipeline using mock callbacks.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <nikola/autonomy/auto_ingestor.hpp>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using namespace nikola::autonomy;
using namespace nikola::infrastructure;

// ============================================================================
// Helper: TempDir RAII
// ============================================================================

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() / ("nikola_ai_test_" +
               std::to_string(std::chrono::steady_clock::now()
                                  .time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

static void write_file(const fs::path& p, const std::string& content) {
    std::ofstream ofs(p);
    ofs << content;
    ofs.close();
}

// ============================================================================
// §A — Paragraph Chunking (TEXT / MARKDOWN)
// ============================================================================

TEST_CASE("§A-1 chunk_paragraphs splits on blank lines", "[auto_ingestor][chunk]") {
    auto chunks = AutoIngestor::chunk_paragraphs(
        "First paragraph\nstill first.\n\nSecond paragraph.\n\nThird.");
    REQUIRE(chunks.size() == 3);
    CHECK(chunks[0] == "First paragraph\nstill first.");
    CHECK(chunks[1] == "Second paragraph.");
    CHECK(chunks[2] == "Third.");
}

TEST_CASE("§A-2 chunk_paragraphs handles single paragraph", "[auto_ingestor][chunk]") {
    auto chunks = AutoIngestor::chunk_paragraphs("Single block of text.");
    REQUIRE(chunks.size() == 1);
    CHECK(chunks[0] == "Single block of text.");
}

TEST_CASE("§A-3 chunk_paragraphs ignores multiple blank lines", "[auto_ingestor][chunk]") {
    auto chunks = AutoIngestor::chunk_paragraphs("A\n\n\n\nB");
    REQUIRE(chunks.size() == 2);
    CHECK(chunks[0] == "A");
    CHECK(chunks[1] == "B");
}

TEST_CASE("§A-4 chunk_paragraphs empty input", "[auto_ingestor][chunk]") {
    auto chunks = AutoIngestor::chunk_paragraphs("");
    CHECK(chunks.empty());
}

TEST_CASE("§A-5 chunk_text dispatches TEXT to paragraphs", "[auto_ingestor][chunk]") {
    auto chunks = AutoIngestor::chunk_text("A\n\nB", FileType::TEXT);
    REQUIRE(chunks.size() == 2);
}

TEST_CASE("§A-6 chunk_text dispatches MARKDOWN to paragraphs", "[auto_ingestor][chunk]") {
    auto chunks = AutoIngestor::chunk_text("# Title\n\nBody text", FileType::MARKDOWN);
    REQUIRE(chunks.size() == 2);
    CHECK(chunks[0] == "# Title");
}

// ============================================================================
// §B — Code Chunking (CODE_CPP / CODE_ARIA)
// ============================================================================

TEST_CASE("§B-1 chunk_code splits on top-level boundaries", "[auto_ingestor][chunk]") {
    std::string code =
        "void foo() {\n"
        "    int x = 1;\n"
        "}\n"
        "\n"
        "void bar() {\n"
        "    int y = 2;\n"
        "}";
    auto chunks = AutoIngestor::chunk_code(code);
    REQUIRE(chunks.size() == 2);
    CHECK(chunks[0].find("foo") != std::string::npos);
    CHECK(chunks[1].find("bar") != std::string::npos);
}

TEST_CASE("§B-2 chunk_code keeps indented blocks together", "[auto_ingestor][chunk]") {
    std::string code =
        "class Foo {\n"
        "    void method() {\n"
        "        return;\n"
        "    }\n"
        "};";
    auto chunks = AutoIngestor::chunk_code(code);
    REQUIRE(chunks.size() == 1);
    CHECK(chunks[0].find("Foo") != std::string::npos);
    CHECK(chunks[0].find("method") != std::string::npos);
}

TEST_CASE("§B-3 chunk_code handles single function", "[auto_ingestor][chunk]") {
    auto chunks = AutoIngestor::chunk_code("int main() { return 0; }");
    REQUIRE(chunks.size() == 1);
}

TEST_CASE("§B-4 chunk_text dispatches CODE_CPP to chunk_code", "[auto_ingestor][chunk]") {
    auto chunks = AutoIngestor::chunk_text("void f() {}\n\nvoid g() {}",
                                            FileType::CODE_CPP);
    REQUIRE(chunks.size() == 2);
}

TEST_CASE("§B-5 chunk_text dispatches CODE_ARIA to chunk_code", "[auto_ingestor][chunk]") {
    auto chunks = AutoIngestor::chunk_text("fn hello() {}\n\nfn world() {}",
                                            FileType::CODE_ARIA);
    REQUIRE(chunks.size() == 2);
}

// ============================================================================
// §C — JSON Chunking
// ============================================================================

TEST_CASE("§C-1 chunk_json splits JSONL", "[auto_ingestor][chunk]") {
    std::string jsonl =
        R"({"text": "hello"})" "\n"
        R"({"text": "world"})" "\n";
    auto chunks = AutoIngestor::chunk_json(jsonl);
    REQUIRE(chunks.size() == 2);
    CHECK(chunks[0].find("hello") != std::string::npos);
    CHECK(chunks[1].find("world") != std::string::npos);
}

TEST_CASE("§C-2 chunk_json single object is one chunk", "[auto_ingestor][chunk]") {
    auto chunks = AutoIngestor::chunk_json(R"({"key": "value"})");
    REQUIRE(chunks.size() == 1);
}

TEST_CASE("§C-3 chunk_json skips blank lines in JSONL", "[auto_ingestor][chunk]") {
    std::string jsonl = R"({"a":1})" "\n\n" R"({"b":2})" "\n";
    auto chunks = AutoIngestor::chunk_json(jsonl);
    REQUIRE(chunks.size() == 2);
}

TEST_CASE("§C-4 chunk_text dispatches JSON", "[auto_ingestor][chunk]") {
    auto chunks = AutoIngestor::chunk_text(R"({"x":1})" "\n" R"({"y":2})",
                                            FileType::JSON);
    REQUIRE(chunks.size() == 2);
}

// ============================================================================
// §D — CSV Chunking
// ============================================================================

TEST_CASE("§D-1 chunk_csv creates header+row pairs", "[auto_ingestor][chunk]") {
    std::string csv = "name,age\nAlice,30\nBob,25\n";
    auto chunks = AutoIngestor::chunk_csv(csv);
    REQUIRE(chunks.size() == 2);
    CHECK(chunks[0] == "name,age\nAlice,30");
    CHECK(chunks[1] == "name,age\nBob,25");
}

TEST_CASE("§D-2 chunk_csv header only", "[auto_ingestor][chunk]") {
    auto chunks = AutoIngestor::chunk_csv("col1,col2\n");
    REQUIRE(chunks.size() == 1);
    CHECK(chunks[0] == "col1,col2");
}

TEST_CASE("§D-3 chunk_csv skips blank rows", "[auto_ingestor][chunk]") {
    std::string csv = "h1,h2\n\nval1,val2\n\nval3,val4\n";
    auto chunks = AutoIngestor::chunk_csv(csv);
    REQUIRE(chunks.size() == 2);
}

TEST_CASE("§D-4 chunk_text dispatches CSV", "[auto_ingestor][chunk]") {
    auto chunks = AutoIngestor::chunk_text("a,b\n1,2\n3,4", FileType::CSV);
    REQUIRE(chunks.size() == 2);
}

// ============================================================================
// §E — UNKNOWN type fallback
// ============================================================================

TEST_CASE("§E-1 chunk_text UNKNOWN falls back to paragraph chunking", "[auto_ingestor][chunk]") {
    auto chunks = AutoIngestor::chunk_text("A\n\nB", FileType::UNKNOWN);
    REQUIRE(chunks.size() == 2);
}

// ============================================================================
// §F — Ingestion Pipeline (with mock callbacks)
// ============================================================================

TEST_CASE("§F-1 ingest_file with text file", "[auto_ingestor][ingest]") {
    TempDir tmp;
    auto f = tmp.path / "test.txt";
    write_file(f, "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.");

    int inject_count = 0;
    int store_count  = 0;
    int tick_count   = 0;

    AutoIngestorConfig cfg;
    cfg.min_chunk_chars = 5;
    cfg.ticks_per_chunk = 3;
    AutoIngestor ai(cfg);
    ai.set_inject_fn([&](const std::string&) { inject_count++; });
    ai.set_store_fn([&]() { store_count++; });
    ai.set_tick_fn([&](int n) { tick_count += n; });

    auto result = ai.ingest_file(f.string());

    CHECK(result.success);
    CHECK(result.chunks_total == 3);
    CHECK(result.chunks_ingested == 3);
    CHECK(result.chunks_skipped == 0);
    CHECK(inject_count == 3);
    CHECK(store_count == 3);
    CHECK(tick_count == 9);  // 3 chunks × 3 ticks
    CHECK(result.elapsed_seconds > 0.0);
}

TEST_CASE("§F-2 ingest_file skips small chunks", "[auto_ingestor][ingest]") {
    TempDir tmp;
    auto f = tmp.path / "test.txt";
    write_file(f, "OK\n\nThis is a longer paragraph that should be kept.");

    int inject_count = 0;

    AutoIngestorConfig cfg;
    cfg.min_chunk_chars = 10;
    AutoIngestor ai(cfg);
    ai.set_inject_fn([&](const std::string&) { inject_count++; });
    ai.set_store_fn([]() {});
    ai.set_tick_fn([](int) {});

    auto result = ai.ingest_file(f.string());

    CHECK(result.success);
    CHECK(result.chunks_total == 2);
    CHECK(result.chunks_ingested == 1);
    CHECK(result.chunks_skipped == 1);
    CHECK(inject_count == 1);
}

TEST_CASE("§F-3 ingest_file with code file", "[auto_ingestor][ingest]") {
    TempDir tmp;
    auto f = tmp.path / "test.cpp";
    write_file(f, "void foo() {\n    return;\n}\n\nvoid bar() {\n    return;\n}");

    int inject_count = 0;

    AutoIngestorConfig cfg;
    cfg.min_chunk_chars = 5;
    AutoIngestor ai(cfg);
    ai.set_inject_fn([&](const std::string&) { inject_count++; });
    ai.set_store_fn([]() {});
    ai.set_tick_fn([](int) {});

    auto result = ai.ingest_file(f.string());

    CHECK(result.success);
    CHECK(result.file_type == FileType::CODE_CPP);
    CHECK(result.chunks_ingested >= 2);
    CHECK(inject_count >= 2);
}

TEST_CASE("§F-4 ingest_file with JSON file", "[auto_ingestor][ingest]") {
    TempDir tmp;
    auto f = tmp.path / "data.jsonl";
    write_file(f, R"({"text":"hello"})" "\n" R"({"text":"world"})" "\n");

    int inject_count = 0;

    AutoIngestorConfig cfg;
    cfg.min_chunk_chars = 5;
    AutoIngestor ai(cfg);
    ai.set_inject_fn([&](const std::string&) { inject_count++; });
    ai.set_store_fn([]() {});
    ai.set_tick_fn([](int) {});

    auto result = ai.ingest_file(f.string());

    CHECK(result.success);
    CHECK(result.file_type == FileType::JSON);
    CHECK(inject_count == 2);
}

TEST_CASE("§F-5 ingest_file with CSV file", "[auto_ingestor][ingest]") {
    TempDir tmp;
    auto f = tmp.path / "data.csv";
    write_file(f, "name,value\nfoo,42\nbar,99\n");

    int inject_count = 0;

    AutoIngestorConfig cfg;
    cfg.min_chunk_chars = 5;
    AutoIngestor ai(cfg);
    ai.set_inject_fn([&](const std::string&) { inject_count++; });
    ai.set_store_fn([]() {});
    ai.set_tick_fn([](int) {});

    auto result = ai.ingest_file(f.string());

    CHECK(result.success);
    CHECK(result.file_type == FileType::CSV);
    CHECK(inject_count == 2);
}

TEST_CASE("§F-6 ingest_file nonexistent file fails", "[auto_ingestor][ingest]") {
    AutoIngestor ai;
    auto result = ai.ingest_file("/nonexistent/file.txt");
    CHECK_FALSE(result.success);
    CHECK_FALSE(result.error.empty());
}

TEST_CASE("§F-7 ingest_file too large is skipped", "[auto_ingestor][ingest]") {
    TempDir tmp;
    auto f = tmp.path / "big.txt";
    // Write a file larger than max
    std::string big(200, 'x');
    write_file(f, big);

    AutoIngestorConfig cfg;
    cfg.max_file_bytes = 100;
    AutoIngestor ai(cfg);

    auto result = ai.ingest_file(f.string());
    CHECK_FALSE(result.success);
}

TEST_CASE("§F-8 ingest_file without callbacks still works", "[auto_ingestor][ingest]") {
    TempDir tmp;
    auto f = tmp.path / "test.txt";
    write_file(f, "Some content here.");

    AutoIngestor ai;  // no callbacks set
    auto result = ai.ingest_file(f.string());
    CHECK(result.success);
    CHECK(result.chunks_ingested == 1);
}

// ============================================================================
// §G — Statistics
// ============================================================================

TEST_CASE("§G-1 stats accumulate across files", "[auto_ingestor][stats]") {
    TempDir tmp;
    write_file(tmp.path / "a.txt", "Para one.\n\nPara two.");
    write_file(tmp.path / "b.txt", "Single para.");

    int inject_count = 0;
    AutoIngestorConfig cfg;
    cfg.min_chunk_chars = 5;
    AutoIngestor ai(cfg);
    ai.set_inject_fn([&](const std::string&) { inject_count++; });
    ai.set_store_fn([]() {});
    ai.set_tick_fn([](int) {});

    ai.ingest_file((tmp.path / "a.txt").string());
    ai.ingest_file((tmp.path / "b.txt").string());

    auto s = ai.stats();
    CHECK(s.files_processed == 2);
    CHECK(s.files_succeeded == 2);
    CHECK(s.total_ingested == 3);
}

TEST_CASE("§G-2 reset_stats clears everything", "[auto_ingestor][stats]") {
    TempDir tmp;
    write_file(tmp.path / "a.txt", "Content.");

    AutoIngestor ai;
    ai.ingest_file((tmp.path / "a.txt").string());
    CHECK(ai.stats().files_processed == 1);

    ai.reset_stats();
    CHECK(ai.stats().files_processed == 0);
}

// ============================================================================
// §H — Event processing
// ============================================================================

TEST_CASE("§H-1 ingest_event skips DELETED events", "[auto_ingestor][events]") {
    FileEvent ev;
    ev.kind = FileEvent::DELETED;
    ev.path = "/some/file.txt";
    ev.type = FileType::TEXT;

    AutoIngestor ai;
    auto result = ai.ingest_event(ev);
    CHECK_FALSE(result.success);
    CHECK(result.error.find("deleted") != std::string::npos);
}

TEST_CASE("§H-2 process_events handles batch", "[auto_ingestor][events]") {
    TempDir tmp;
    write_file(tmp.path / "a.txt", "Hello world.");
    write_file(tmp.path / "b.md", "# Title\n\nBody.");

    AutoIngestor ai;
    ai.set_inject_fn([](const std::string&) {});
    ai.set_store_fn([]() {});
    ai.set_tick_fn([](int) {});

    std::vector<FileEvent> events;
    {
        FileEvent ev;
        ev.kind = FileEvent::CREATED;
        ev.path = (tmp.path / "a.txt").string();
        ev.type = FileType::TEXT;
        events.push_back(ev);
    }
    {
        FileEvent ev;
        ev.kind = FileEvent::CREATED;
        ev.path = (tmp.path / "b.md").string();
        ev.type = FileType::MARKDOWN;
        events.push_back(ev);
    }

    auto results = ai.process_events(std::move(events));
    REQUIRE(results.size() == 2);
    CHECK(results[0].success);
    CHECK(results[1].success);
}

// ============================================================================
// §I — Chunk truncation
// ============================================================================

TEST_CASE("§I-1 large chunks are truncated to max_chunk_chars", "[auto_ingestor][ingest]") {
    TempDir tmp;
    auto f = tmp.path / "test.txt";
    std::string big(500, 'x');
    write_file(f, big);

    std::string injected;
    AutoIngestorConfig cfg;
    cfg.min_chunk_chars = 5;
    cfg.max_chunk_chars = 100;
    cfg.max_file_bytes = 10000;
    AutoIngestor ai(cfg);
    ai.set_inject_fn([&](const std::string& s) { injected = s; });
    ai.set_store_fn([]() {});
    ai.set_tick_fn([](int) {});

    ai.ingest_file(f.string());
    CHECK(injected.size() == 100);
}
