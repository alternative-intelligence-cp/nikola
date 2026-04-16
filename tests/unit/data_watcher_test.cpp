/**
 * @file tests/unit/data_watcher_test.cpp
 * @brief v0.2.2 Phase 1 — DataWatcher unit tests.
 *
 * Uses real inotify via temporary directories.
 */

#include <catch2/catch_test_macros.hpp>
#include <nikola/infrastructure/data_watcher.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

namespace fs = std::filesystem;
using namespace nikola::infrastructure;

// ============================================================================
// Helper: TempDir RAII
// ============================================================================

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() / ("nikola_dw_test_" +
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

// Wait for events with timeout
static std::vector<FileEvent> wait_for_events(DataWatcher& w, size_t min_count,
                                               int timeout_ms = 3000) {
    std::vector<FileEvent> all;
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline && all.size() < min_count) {
        auto batch = w.poll_events();
        all.insert(all.end(), batch.begin(), batch.end());
        if (all.size() < min_count) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    return all;
}

// ============================================================================
// §A — FileType Classification
// ============================================================================

TEST_CASE("§A-1 classify text files", "[data_watcher][classify]") {
    CHECK(DataWatcher::classify("notes.txt") == FileType::TEXT);
    CHECK(DataWatcher::classify("/some/path/readme.TXT") == FileType::TEXT);
}

TEST_CASE("§A-2 classify markdown files", "[data_watcher][classify]") {
    CHECK(DataWatcher::classify("doc.md") == FileType::MARKDOWN);
    CHECK(DataWatcher::classify("GUIDE.MD") == FileType::MARKDOWN);
}

TEST_CASE("§A-3 classify C++ files", "[data_watcher][classify]") {
    CHECK(DataWatcher::classify("main.cpp") == FileType::CODE_CPP);
    CHECK(DataWatcher::classify("header.hpp") == FileType::CODE_CPP);
    CHECK(DataWatcher::classify("legacy.h") == FileType::CODE_CPP);
    CHECK(DataWatcher::classify("alt.cc") == FileType::CODE_CPP);
    CHECK(DataWatcher::classify("weird.cxx") == FileType::CODE_CPP);
}

TEST_CASE("§A-4 classify Aria files", "[data_watcher][classify]") {
    CHECK(DataWatcher::classify("module.aria") == FileType::CODE_ARIA);
}

TEST_CASE("§A-5 classify JSON files", "[data_watcher][classify]") {
    CHECK(DataWatcher::classify("data.json") == FileType::JSON);
    CHECK(DataWatcher::classify("training.jsonl") == FileType::JSON);
}

TEST_CASE("§A-6 classify CSV files", "[data_watcher][classify]") {
    CHECK(DataWatcher::classify("table.csv") == FileType::CSV);
}

TEST_CASE("§A-7 classify unknown extensions", "[data_watcher][classify]") {
    CHECK(DataWatcher::classify("binary.exe") == FileType::UNKNOWN);
    CHECK(DataWatcher::classify("noext") == FileType::UNKNOWN);
    CHECK(DataWatcher::classify("archive.tar.gz") == FileType::UNKNOWN);
}

TEST_CASE("§A-8 file_type_name round-trips", "[data_watcher][classify]") {
    CHECK(std::string(file_type_name(FileType::TEXT)) == "TEXT");
    CHECK(std::string(file_type_name(FileType::MARKDOWN)) == "MARKDOWN");
    CHECK(std::string(file_type_name(FileType::CODE_CPP)) == "CODE_CPP");
    CHECK(std::string(file_type_name(FileType::CODE_ARIA)) == "CODE_ARIA");
    CHECK(std::string(file_type_name(FileType::JSON)) == "JSON");
    CHECK(std::string(file_type_name(FileType::CSV)) == "CSV");
    CHECK(std::string(file_type_name(FileType::UNKNOWN)) == "UNKNOWN");
}

// ============================================================================
// §B — Construction and Lifecycle
// ============================================================================

TEST_CASE("§B-1 default construction", "[data_watcher][lifecycle]") {
    DataWatcher w;
    CHECK_FALSE(w.running());
    CHECK(w.watch_dir() == "data/inbox");
    CHECK(w.pending_count() == 0);
}

TEST_CASE("§B-2 custom config", "[data_watcher][lifecycle]") {
    DataWatcherConfig cfg;
    cfg.watch_dir = "/tmp/custom_inbox";
    DataWatcher w(cfg);
    CHECK(w.watch_dir() == "/tmp/custom_inbox");
}

TEST_CASE("§B-3 start creates directory if missing", "[data_watcher][lifecycle]") {
    TempDir tmp;
    auto inbox = tmp.path / "sub" / "inbox";
    DataWatcherConfig cfg;
    cfg.watch_dir = inbox.string();
    cfg.create_dir_if_missing = true;
    DataWatcher w(cfg);
    CHECK(w.start());
    CHECK(fs::is_directory(inbox));
    w.stop();
}

TEST_CASE("§B-4 start fails if dir missing and create disabled", "[data_watcher][lifecycle]") {
    DataWatcherConfig cfg;
    cfg.watch_dir = "/tmp/nonexistent_nikola_dir_" +
                    std::to_string(std::chrono::steady_clock::now()
                                      .time_since_epoch().count());
    cfg.create_dir_if_missing = false;
    DataWatcher w(cfg);
    CHECK_FALSE(w.start());
    CHECK_FALSE(w.running());
}

TEST_CASE("§B-5 start and stop lifecycle", "[data_watcher][lifecycle]") {
    TempDir tmp;
    DataWatcherConfig cfg;
    cfg.watch_dir = tmp.path.string();
    DataWatcher w(cfg);
    CHECK(w.start());
    CHECK(w.running());
    w.stop();
    CHECK_FALSE(w.running());
}

TEST_CASE("§B-6 double start is idempotent", "[data_watcher][lifecycle]") {
    TempDir tmp;
    DataWatcherConfig cfg;
    cfg.watch_dir = tmp.path.string();
    DataWatcher w(cfg);
    CHECK(w.start());
    CHECK(w.start()); // second call returns true, no-op
    w.stop();
}

TEST_CASE("§B-7 double stop is safe", "[data_watcher][lifecycle]") {
    TempDir tmp;
    DataWatcherConfig cfg;
    cfg.watch_dir = tmp.path.string();
    DataWatcher w(cfg);
    w.start();
    w.stop();
    w.stop(); // no crash
}

TEST_CASE("§B-8 poll on stopped watcher returns empty", "[data_watcher][lifecycle]") {
    DataWatcher w;
    auto events = w.poll_events();
    CHECK(events.empty());
}

// ============================================================================
// §C — File Event Detection
// ============================================================================

TEST_CASE("§C-1 detect new text file", "[data_watcher][events]") {
    TempDir tmp;
    DataWatcherConfig cfg;
    cfg.watch_dir = tmp.path.string();
    cfg.debounce_ms = std::chrono::milliseconds(100);
    DataWatcher w(cfg);
    REQUIRE(w.start());

    write_file(tmp.path / "hello.txt", "world");
    auto events = wait_for_events(w, 1);
    w.stop();

    REQUIRE(events.size() >= 1);
    bool found = false;
    for (const auto& ev : events) {
        if (ev.path.find("hello.txt") != std::string::npos) {
            CHECK(ev.type == FileType::TEXT);
            found = true;
        }
    }
    CHECK(found);
}

TEST_CASE("§C-2 detect new markdown file", "[data_watcher][events]") {
    TempDir tmp;
    DataWatcherConfig cfg;
    cfg.watch_dir = tmp.path.string();
    cfg.debounce_ms = std::chrono::milliseconds(100);
    DataWatcher w(cfg);
    REQUIRE(w.start());

    write_file(tmp.path / "notes.md", "# Title");
    auto events = wait_for_events(w, 1);
    w.stop();

    REQUIRE(events.size() >= 1);
    CHECK(events[0].type == FileType::MARKDOWN);
}

TEST_CASE("§C-3 detect deleted file", "[data_watcher][events]") {
    TempDir tmp;
    auto f = tmp.path / "temp.txt";
    write_file(f, "data");

    DataWatcherConfig cfg;
    cfg.watch_dir = tmp.path.string();
    cfg.debounce_ms = std::chrono::milliseconds(100);
    DataWatcher w(cfg);
    REQUIRE(w.start());

    // Give inotify a moment to start
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    fs::remove(f);

    auto events = wait_for_events(w, 1);
    w.stop();

    REQUIRE(events.size() >= 1);
    bool found_delete = false;
    for (const auto& ev : events) {
        if (ev.kind == FileEvent::DELETED) found_delete = true;
    }
    CHECK(found_delete);
}

TEST_CASE("§C-4 detect multiple file types", "[data_watcher][events]") {
    TempDir tmp;
    DataWatcherConfig cfg;
    cfg.watch_dir = tmp.path.string();
    cfg.debounce_ms = std::chrono::milliseconds(100);
    DataWatcher w(cfg);
    REQUIRE(w.start());

    write_file(tmp.path / "a.txt",  "text");
    write_file(tmp.path / "b.cpp",  "code");
    write_file(tmp.path / "c.json", "{}");
    write_file(tmp.path / "d.csv",  "a,b,c");

    auto events = wait_for_events(w, 4);
    w.stop();

    REQUIRE(events.size() >= 4);
    bool has_txt = false, has_cpp = false, has_json = false, has_csv = false;
    for (const auto& ev : events) {
        if (ev.type == FileType::TEXT)     has_txt = true;
        if (ev.type == FileType::CODE_CPP) has_cpp = true;
        if (ev.type == FileType::JSON)     has_json = true;
        if (ev.type == FileType::CSV)      has_csv = true;
    }
    CHECK(has_txt);
    CHECK(has_cpp);
    CHECK(has_json);
    CHECK(has_csv);
}

TEST_CASE("§C-5 directories are ignored", "[data_watcher][events]") {
    TempDir tmp;
    DataWatcherConfig cfg;
    cfg.watch_dir = tmp.path.string();
    cfg.debounce_ms = std::chrono::milliseconds(100);
    DataWatcher w(cfg);
    REQUIRE(w.start());

    fs::create_directory(tmp.path / "subdir");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    auto events = w.poll_events();
    w.stop();

    // Should not have any events for directory creation
    for (const auto& ev : events) {
        CHECK(ev.path.find("subdir") == std::string::npos);
    }
}

TEST_CASE("§C-6 unknown extensions still produce events", "[data_watcher][events]") {
    TempDir tmp;
    DataWatcherConfig cfg;
    cfg.watch_dir = tmp.path.string();
    cfg.debounce_ms = std::chrono::milliseconds(100);
    DataWatcher w(cfg);
    REQUIRE(w.start());

    write_file(tmp.path / "binary.exe", "MZ");
    auto events = wait_for_events(w, 1);
    w.stop();

    REQUIRE(events.size() >= 1);
    CHECK(events[0].type == FileType::UNKNOWN);
}

// ============================================================================
// §D — Debounce Behavior
// ============================================================================

TEST_CASE("§D-1 rapid writes coalesce into single event", "[data_watcher][debounce]") {
    TempDir tmp;
    DataWatcherConfig cfg;
    cfg.watch_dir = tmp.path.string();
    cfg.debounce_ms = std::chrono::milliseconds(300);
    DataWatcher w(cfg);
    REQUIRE(w.start());

    auto f = tmp.path / "rapid.txt";
    // Write rapidly 5 times
    for (int i = 0; i < 5; ++i) {
        write_file(f, "version " + std::to_string(i));
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    // Wait for debounce + some margin
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    auto events = w.poll_events();
    w.stop();

    // Should have coalesced to 1 or 2 events (not 5)
    size_t rapid_events = 0;
    for (const auto& ev : events) {
        if (ev.path.find("rapid.txt") != std::string::npos) ++rapid_events;
    }
    CHECK(rapid_events <= 2);
}

// ============================================================================
// §E — Queue Limits
// ============================================================================

TEST_CASE("§E-1 queue respects max size", "[data_watcher][queue]") {
    TempDir tmp;
    DataWatcherConfig cfg;
    cfg.watch_dir = tmp.path.string();
    cfg.debounce_ms = std::chrono::milliseconds(50);
    cfg.max_queue_size = 5;
    DataWatcher w(cfg);
    REQUIRE(w.start());

    // Create more files than the queue can hold
    for (int i = 0; i < 10; ++i) {
        write_file(tmp.path / ("file" + std::to_string(i) + ".txt"), "data");
    }

    // Wait for all to process
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    auto events = w.poll_events();
    w.stop();

    CHECK(events.size() <= 5);
}

// ============================================================================
// §F — FileEvent accessors
// ============================================================================

TEST_CASE("§F-1 kind_name returns correct strings", "[data_watcher][event]") {
    FileEvent ev;
    ev.kind = FileEvent::CREATED;
    CHECK(std::string(ev.kind_name()) == "CREATED");
    ev.kind = FileEvent::MODIFIED;
    CHECK(std::string(ev.kind_name()) == "MODIFIED");
    ev.kind = FileEvent::DELETED;
    CHECK(std::string(ev.kind_name()) == "DELETED");
}
