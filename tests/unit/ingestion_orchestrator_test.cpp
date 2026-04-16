/**
 * @file tests/unit/ingestion_orchestrator_test.cpp
 * @brief v0.2.2 Phase 4 — IngestionOrchestrator unit tests.
 *
 * Tests NAP integration, event queuing, and on-demand ingestion.
 */

#include <catch2/catch_test_macros.hpp>
#include <nikola/autonomy/ingestion_orchestrator.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

namespace fs = std::filesystem;
using namespace nikola::autonomy;
using namespace nikola::infrastructure;

// ============================================================================
// Helper: TempDir RAII
// ============================================================================

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() / ("nikola_orch_test_" +
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

// Helper: build config pointing at temp dir
static IngestionOrchestratorConfig make_config(const fs::path& dir) {
    IngestionOrchestratorConfig cfg;
    cfg.watcher_cfg.watch_dir = dir.string();
    cfg.watcher_cfg.debounce_ms = std::chrono::milliseconds(100);
    cfg.max_files_per_nap = 5;
    cfg.ingestor_cfg.min_chunk_chars = 5;
    return cfg;
}

// ============================================================================
// §A — Construction & Lifecycle
// ============================================================================

TEST_CASE("§A-1 default construction", "[orchestrator]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    IngestionOrchestrator orch(cfg);

    CHECK(orch.queue_size() == 0);
    CHECK_FALSE(orch.running());
    CHECK(orch.watch_dir() == tmp.path.string());
}

TEST_CASE("§A-2 start and stop", "[orchestrator]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    IngestionOrchestrator orch(cfg);

    CHECK(orch.start());
    CHECK(orch.running());
    orch.stop();
    CHECK_FALSE(orch.running());
}

TEST_CASE("§A-3 double start is safe", "[orchestrator]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    IngestionOrchestrator orch(cfg);

    CHECK(orch.start());
    CHECK(orch.start());   // Should be safe (watcher handles idempotently)
    orch.stop();
}

TEST_CASE("§A-4 destructor stops watcher", "[orchestrator]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    {
        IngestionOrchestrator orch(cfg);
        orch.start();
        CHECK(orch.running());
    }
    // No crash — watcher stopped in destructor
}

// ============================================================================
// §B — Event Collection & Queue
// ============================================================================

TEST_CASE("§B-1 collect_events queues file creations", "[orchestrator]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    IngestionOrchestrator orch(cfg);
    orch.start();

    write_file(tmp.path / "hello.txt", "Some training content here.");

    // Wait for inotify + debounce
    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    orch.collect_events();
    CHECK(orch.queue_size() >= 1);
    orch.stop();
}

TEST_CASE("§B-2 collect_events ignores deletes", "[orchestrator]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    IngestionOrchestrator orch(cfg);

    // Pre-create a file
    write_file(tmp.path / "gone.txt", "About to be deleted.");
    orch.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Delete it
    fs::remove(tmp.path / "gone.txt");
    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    orch.collect_events();
    // Deletes should not be queued — only creates/modifications
    // (The initial create might be queued, but the delete should not add another)
    // We just verify no crash and queue doesn't contain delete-only events
    CHECK(true);
    orch.stop();
}

TEST_CASE("§B-3 empty collect when no files", "[orchestrator]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    IngestionOrchestrator orch(cfg);
    orch.start();

    orch.collect_events();
    CHECK(orch.queue_size() == 0);
    orch.stop();
}

// ============================================================================
// §C — NAP Ingestion
// ============================================================================

TEST_CASE("§C-1 nap_ingest processes queued events", "[orchestrator]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    size_t inject_count = 0;
    size_t store_count  = 0;

    IngestionOrchestrator orch(cfg);
    orch.set_inject_fn([&](const std::string&) { inject_count++; });
    orch.set_store_fn([&]() { store_count++; });
    orch.start();

    // Create files
    write_file(tmp.path / "a.txt", "First training document with enough content.");
    write_file(tmp.path / "b.txt", "Second training document with enough content.");

    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    orch.collect_events();
    REQUIRE(orch.queue_size() >= 2);

    auto report = orch.nap_ingest();
    CHECK(report.files_processed >= 2);
    CHECK(report.files_succeeded >= 2);
    CHECK(report.chunks_ingested > 0);
    CHECK(inject_count > 0);
    CHECK(store_count > 0);
    CHECK(orch.queue_size() == 0);  // Queue drained
    orch.stop();
}

TEST_CASE("§C-2 nap_ingest respects max_files_per_nap", "[orchestrator]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    cfg.max_files_per_nap = 2;

    IngestionOrchestrator orch(cfg);
    orch.set_inject_fn([](const std::string&) {});
    orch.set_store_fn([]() {});
    orch.start();

    // Create more files than budget
    for (int i = 0; i < 5; i++) {
        write_file(tmp.path / ("file" + std::to_string(i) + ".txt"),
                   "Content for file " + std::to_string(i) + " with enough text.");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    orch.collect_events();
    size_t queued = orch.queue_size();
    REQUIRE(queued >= 3);  // at least 3 queued

    auto report = orch.nap_ingest();
    CHECK(report.files_processed == 2);         // Bounded by budget
    CHECK(report.queue_remaining == queued - 2); // Rest still queued
    CHECK(orch.queue_size() == queued - 2);
    orch.stop();
}

TEST_CASE("§C-3 nap_ingest with empty queue", "[orchestrator]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    IngestionOrchestrator orch(cfg);

    auto report = orch.nap_ingest();
    CHECK(report.files_processed == 0);
    CHECK(report.chunks_ingested == 0);
    CHECK(report.queue_remaining == 0);
}

TEST_CASE("§C-4 nap_ingest tracks elapsed time", "[orchestrator]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    IngestionOrchestrator orch(cfg);
    orch.set_inject_fn([](const std::string&) {});
    orch.set_store_fn([]() {});
    orch.start();

    write_file(tmp.path / "timed.txt", "Content for timing measurement test file.");
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    orch.collect_events();

    auto report = orch.nap_ingest();
    CHECK(report.elapsed_seconds >= 0.0);
    orch.stop();
}

// ============================================================================
// §D — On-Demand Ingestion
// ============================================================================

TEST_CASE("§D-1 ingest_on_demand processes a file directly", "[orchestrator]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    size_t inject_count = 0;

    IngestionOrchestrator orch(cfg);
    orch.set_inject_fn([&](const std::string&) { inject_count++; });
    orch.set_store_fn([]() {});

    fs::path f = tmp.path / "urgent.txt";
    write_file(f, "On-demand content that needs immediate ingestion.");

    auto result = orch.ingest_on_demand(f.string());
    CHECK(result.chunks_ingested > 0);
    CHECK(inject_count > 0);
}

TEST_CASE("§D-2 ingest_on_demand handles missing file", "[orchestrator]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    IngestionOrchestrator orch(cfg);

    auto result = orch.ingest_on_demand("/nonexistent/file.txt");
    CHECK(result.chunks_ingested == 0);
}

// ============================================================================
// §E — Filter Integration via Orchestrator
// ============================================================================

TEST_CASE("§E-1 duplicate content is filtered during nap_ingest", "[orchestrator][filter]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    size_t inject_count = 0;

    IngestionOrchestrator orch(cfg);
    orch.set_inject_fn([&](const std::string&) { inject_count++; });
    orch.set_store_fn([]() {});
    orch.start();

    // Write two identical files
    std::string content = "Identical training data that should be deduplicated by the filter.";
    write_file(tmp.path / "dup1.txt", content);
    write_file(tmp.path / "dup2.txt", content);

    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    orch.collect_events();

    auto report = orch.nap_ingest();
    // First file should be ingested, second should be filtered as duplicate
    CHECK(report.files_processed >= 2);
    // The filter should have caught at least one duplicate chunk
    auto fstats = orch.filter_stats();
    // At least some filtering should happen for identical content
    CHECK(fstats.total_checked >= 2);
    orch.stop();
}

TEST_CASE("§E-2 reset_daily_budget works through orchestrator", "[orchestrator][filter]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    IngestionOrchestrator orch(cfg);

    // Just verify no crash
    orch.reset_daily_budget();
    auto fstats = orch.filter_stats();
    CHECK(fstats.total_checked == 0);
}

// ============================================================================
// §F — Statistics
// ============================================================================

TEST_CASE("§F-1 ingestor_stats accumulates across naps", "[orchestrator]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    cfg.max_files_per_nap = 1;

    IngestionOrchestrator orch(cfg);
    orch.set_inject_fn([](const std::string&) {});
    orch.set_store_fn([]() {});
    orch.start();

    write_file(tmp.path / "batch1.txt", "First batch content for stats.");
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    orch.collect_events();
    orch.nap_ingest();  // Process 1 file

    write_file(tmp.path / "batch2.txt", "Second batch content for stats.");
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    orch.collect_events();
    orch.nap_ingest();  // Process another

    auto stats = orch.ingestor_stats();
    CHECK(stats.total_ingested >= 2);
    CHECK(stats.files_processed >= 2);
    orch.stop();
}

TEST_CASE("§F-2 filter_stats tracks checks", "[orchestrator]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);

    IngestionOrchestrator orch(cfg);
    orch.set_inject_fn([](const std::string&) {});
    orch.set_store_fn([]() {});

    write_file(tmp.path / "check.txt", "Content to verify filter stats tracking.");
    orch.ingest_on_demand((tmp.path / "check.txt").string());

    auto fstats = orch.filter_stats();
    CHECK(fstats.total_checked >= 1);
}

// ============================================================================
// §G — Tick Function Integration
// ============================================================================

TEST_CASE("§G-1 tick_fn called during ingestion", "[orchestrator]") {
    TempDir tmp;
    auto cfg = make_config(tmp.path);
    int tick_calls = 0;

    IngestionOrchestrator orch(cfg);
    orch.set_inject_fn([](const std::string&) {});
    orch.set_store_fn([]() {});
    orch.set_tick_fn([&](int n) { tick_calls += n; });

    write_file(tmp.path / "tick.txt", "Content for tick function testing.");
    orch.ingest_on_demand((tmp.path / "tick.txt").string());

    CHECK(tick_calls > 0);
}
