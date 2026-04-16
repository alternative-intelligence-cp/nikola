#pragma once
/**
 * @file include/nikola/infrastructure/data_watcher.hpp
 * @brief v0.2.2 Phase 1 — DataWatcher: inotify-based directory monitor for
 *        automatic training data ingestion.
 *
 * Watches a configurable inbox directory for new or modified files, classifies
 * them by type, debounces rapid writes, and queues FileEvent records for
 * consumption by the AutoIngestor.
 *
 * Design:
 *   · One inotify instance per DataWatcher.
 *   · Worker thread reads inotify events and enqueues them.
 *   · Debounce: coalesces rapid writes into a single MODIFIED event
 *     by waiting for write quiescence (default 500ms).
 *   · Thread-safe: poll_events() can be called from any thread.
 *   · File type classification by extension.
 *
 * Usage:
 *   DataWatcher watcher("/path/to/inbox");
 *   watcher.start();
 *   // ... later ...
 *   auto events = watcher.poll_events();
 *   for (const auto& ev : events) {
 *       std::cout << ev.path << " [" << file_type_name(ev.type) << "]\n";
 *   }
 *   watcher.stop();
 *
 * Phase: NIK-INGEST-01 (DataWatcher, v0.2.2)
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace nikola::infrastructure {

// ============================================================================
// FileType
// ============================================================================

enum class FileType : uint8_t {
    TEXT,       ///< .txt
    MARKDOWN,   ///< .md
    CODE_CPP,   ///< .cpp, .hpp, .h, .cc, .cxx
    CODE_ARIA,  ///< .aria
    JSON,       ///< .json, .jsonl
    CSV,        ///< .csv
    UNKNOWN     ///< Unrecognized extension
};

/// Human-readable name for a FileType.
[[nodiscard]] inline const char* file_type_name(FileType ft) noexcept {
    switch (ft) {
        case FileType::TEXT:      return "TEXT";
        case FileType::MARKDOWN:  return "MARKDOWN";
        case FileType::CODE_CPP:  return "CODE_CPP";
        case FileType::CODE_ARIA: return "CODE_ARIA";
        case FileType::JSON:      return "JSON";
        case FileType::CSV:       return "CSV";
        default:                  return "UNKNOWN";
    }
}

// ============================================================================
// FileEvent
// ============================================================================

struct FileEvent {
    enum Kind : uint8_t { CREATED, MODIFIED, DELETED };

    Kind        kind;
    std::string path;       ///< Full path to the file
    FileType    type;       ///< Classified file type
    std::chrono::steady_clock::time_point timestamp;

    /// Human-readable kind string.
    [[nodiscard]] const char* kind_name() const noexcept {
        switch (kind) {
            case CREATED:  return "CREATED";
            case MODIFIED: return "MODIFIED";
            case DELETED:  return "DELETED";
            default:       return "UNKNOWN";
        }
    }
};

// ============================================================================
// DataWatcherConfig
// ============================================================================

struct DataWatcherConfig {
    /// Directory to watch for incoming training data.
    std::string watch_dir = "data/inbox";

    /// Debounce window: wait this long after last write before emitting event.
    std::chrono::milliseconds debounce_ms{500};

    /// Maximum number of queued events before oldest are dropped.
    size_t max_queue_size = 1024;

    /// Whether to create the watch directory if it doesn't exist.
    bool create_dir_if_missing = true;
};

// ============================================================================
// DataWatcher
// ============================================================================

class DataWatcher {
public:
    explicit DataWatcher(const DataWatcherConfig& cfg = {});
    ~DataWatcher();

    // Non-copyable, non-movable (owns thread + fd)
    DataWatcher(const DataWatcher&)            = delete;
    DataWatcher& operator=(const DataWatcher&) = delete;
    DataWatcher(DataWatcher&&)                 = delete;
    DataWatcher& operator=(DataWatcher&&)      = delete;

    /// Start watching.  Returns false if inotify setup fails.
    bool start();

    /// Stop watching and join the worker thread.
    void stop();

    /// True if the watcher is actively monitoring.
    [[nodiscard]] bool running() const noexcept { return running_.load(std::memory_order_relaxed); }

    /// Drain all pending events.  Thread-safe.
    [[nodiscard]] std::vector<FileEvent> poll_events();

    /// Number of events currently queued.
    [[nodiscard]] size_t pending_count() const noexcept;

    /// The directory being watched.
    [[nodiscard]] const std::string& watch_dir() const noexcept { return cfg_.watch_dir; }

    /// Classify a file path by extension.
    [[nodiscard]] static FileType classify(const std::string& path) noexcept;

private:
    DataWatcherConfig cfg_;

    int inotify_fd_ = -1;
    int watch_fd_   = -1;

    std::atomic<bool> running_{false};
    std::thread       worker_;

    mutable std::mutex        queue_mutex_;
    std::vector<FileEvent>    event_queue_;

    /// Debounce state: path → last-write timestamp
    struct PendingWrite {
        std::chrono::steady_clock::time_point last_write;
        FileEvent::Kind                       kind;
    };
    std::unordered_map<std::string, PendingWrite> pending_writes_;

    void worker_loop_();
    void flush_debounced_();
    void enqueue_(FileEvent ev);
};

} // namespace nikola::infrastructure
