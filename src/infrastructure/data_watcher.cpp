/**
 * @file src/infrastructure/data_watcher.cpp
 * @brief v0.2.2 Phase 1 — DataWatcher implementation using Linux inotify.
 */

#include <nikola/infrastructure/data_watcher.hpp>

#include <sys/inotify.h>
#include <unistd.h>
#include <poll.h>
#include <sys/stat.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>

namespace nikola::infrastructure {

namespace fs = std::filesystem;

// ============================================================================
// classify — extension → FileType
// ============================================================================

FileType DataWatcher::classify(const std::string& path) noexcept {
    auto dot = path.rfind('.');
    if (dot == std::string::npos) return FileType::UNKNOWN;

    // Extract extension (lowercase comparison)
    std::string ext = path.substr(dot);
    for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (ext == ".txt")                             return FileType::TEXT;
    if (ext == ".md")                              return FileType::MARKDOWN;
    if (ext == ".cpp" || ext == ".hpp" ||
        ext == ".h"   || ext == ".cc"  ||
        ext == ".cxx")                             return FileType::CODE_CPP;
    if (ext == ".aria")                            return FileType::CODE_ARIA;
    if (ext == ".json" || ext == ".jsonl")          return FileType::JSON;
    if (ext == ".csv")                             return FileType::CSV;
    return FileType::UNKNOWN;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

DataWatcher::DataWatcher(const DataWatcherConfig& cfg)
    : cfg_(cfg)
{}

DataWatcher::~DataWatcher() {
    stop();
}

// ============================================================================
// start / stop
// ============================================================================

bool DataWatcher::start() {
    if (running_.load(std::memory_order_relaxed)) return true;

    // Optionally create watch directory
    if (cfg_.create_dir_if_missing) {
        std::error_code ec;
        fs::create_directories(cfg_.watch_dir, ec);
        if (ec) return false;
    }

    // Verify directory exists
    if (!fs::is_directory(cfg_.watch_dir)) return false;

    // Init inotify
    inotify_fd_ = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
    if (inotify_fd_ < 0) return false;

    // Watch for creates, writes, moves-in, and deletes
    watch_fd_ = inotify_add_watch(inotify_fd_, cfg_.watch_dir.c_str(),
                                  IN_CLOSE_WRITE | IN_CREATE | IN_MOVED_TO | IN_DELETE);
    if (watch_fd_ < 0) {
        close(inotify_fd_);
        inotify_fd_ = -1;
        return false;
    }

    running_.store(true, std::memory_order_release);
    worker_ = std::thread([this]() { worker_loop_(); });
    return true;
}

void DataWatcher::stop() {
    if (!running_.load(std::memory_order_relaxed)) return;
    running_.store(false, std::memory_order_release);

    if (worker_.joinable()) worker_.join();

    if (watch_fd_ >= 0) {
        inotify_rm_watch(inotify_fd_, watch_fd_);
        watch_fd_ = -1;
    }
    if (inotify_fd_ >= 0) {
        close(inotify_fd_);
        inotify_fd_ = -1;
    }
}

// ============================================================================
// poll_events / pending_count
// ============================================================================

std::vector<FileEvent> DataWatcher::poll_events() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    std::vector<FileEvent> result;
    result.swap(event_queue_);
    return result;
}

size_t DataWatcher::pending_count() const noexcept {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return event_queue_.size();
}

// ============================================================================
// enqueue_
// ============================================================================

void DataWatcher::enqueue_(FileEvent ev) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (event_queue_.size() < cfg_.max_queue_size) {
        event_queue_.push_back(std::move(ev));
    }
    // else: silently drop (bounded queue)
}

// ============================================================================
// flush_debounced_ — emit events for writes that have quiesced
// ============================================================================

void DataWatcher::flush_debounced_() {
    const auto now = std::chrono::steady_clock::now();

    auto it = pending_writes_.begin();
    while (it != pending_writes_.end()) {
        auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now - it->second.last_write);
        if (age >= cfg_.debounce_ms) {
            FileEvent ev;
            ev.kind      = it->second.kind;
            ev.path      = it->first;
            ev.type      = classify(ev.path);
            ev.timestamp = it->second.last_write;
            enqueue_(std::move(ev));
            it = pending_writes_.erase(it);
        } else {
            ++it;
        }
    }
}

// ============================================================================
// worker_loop_ — inotify read + debounce
// ============================================================================

void DataWatcher::worker_loop_() {
    // Buffer for inotify events (each event = fixed header + variable name)
    alignas(struct inotify_event) char buf[4096];

    while (running_.load(std::memory_order_acquire)) {
        // Poll with 100ms timeout so we can flush debounced events regularly
        struct pollfd pfd;
        pfd.fd     = inotify_fd_;
        pfd.events = POLLIN;
        int ret = poll(&pfd, 1, 100);

        if (ret > 0 && (pfd.revents & POLLIN)) {
            ssize_t len = read(inotify_fd_, buf, sizeof(buf));
            if (len > 0) {
                const char* ptr = buf;
                while (ptr < buf + len) {
                    const auto* event = reinterpret_cast<const struct inotify_event*>(ptr);

                    if (event->len > 0 && !(event->mask & IN_ISDIR)) {
                        std::string name(event->name);
                        std::string full_path = cfg_.watch_dir + "/" + name;
                        auto now = std::chrono::steady_clock::now();

                        if (event->mask & IN_DELETE) {
                            // Deletes are immediate (no debounce)
                            FileEvent ev;
                            ev.kind      = FileEvent::DELETED;
                            ev.path      = full_path;
                            ev.type      = classify(full_path);
                            ev.timestamp = now;
                            enqueue_(std::move(ev));
                            // Remove from pending if present
                            pending_writes_.erase(full_path);
                        } else {
                            // CREATE / CLOSE_WRITE / MOVED_TO → debounce
                            FileEvent::Kind kind = (event->mask & IN_CREATE)
                                                       ? FileEvent::CREATED
                                                       : FileEvent::MODIFIED;
                            auto& pw = pending_writes_[full_path];
                            pw.last_write = now;
                            // Keep CREATED if first event, otherwise MODIFIED
                            if (pending_writes_.count(full_path) == 1 &&
                                kind == FileEvent::CREATED) {
                                pw.kind = FileEvent::CREATED;
                            } else {
                                // Already existed in pending → stays as whatever it was
                                // unless this is the first entry
                                if (pw.kind != FileEvent::CREATED)
                                    pw.kind = kind;
                            }
                        }
                    }

                    ptr += sizeof(struct inotify_event) + event->len;
                }
            }
        }

        // Flush events that have quiesced past the debounce window
        flush_debounced_();
    }

    // Final flush before exit
    flush_debounced_();
}

} // namespace nikola::infrastructure
