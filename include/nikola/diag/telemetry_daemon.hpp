/**
 * @file diag/telemetry_daemon.hpp
 * @brief Async metric exporter writing JSON Lines to stddbg (FD 3).
 *
 * Design requirements (ecosystem/07_TelemetryDaemon):
 *   - Export wave interference metrics via stddbg (file descriptor 3)
 *   - JSON Lines format, one metric record per line
 *   - 1kHz-capable: producer path never blocks — uses a lock-free-style
 *     try-push into a bounded circular queue; drops silently on overflow
 *   - Async drain background thread flushes queue at configurable rate
 *   - Metric types: gauge (instantaneous value), counter (cumulative delta),
 *     event (free-form JSON payload)
 *   - Never emits to stderr (wrong semantic channel per integration rules)
 *   - Silently no-ops if FD 3 is not open
 *
 * Wire format (JSON Lines, UTF-8):
 * @code
 *   {"ts":1740147600.001,"type":"gauge","metric":"wave.H","value":68890.5,"unit":"J"}
 *   {"ts":1740147600.002,"type":"counter","metric":"prop.steps","delta":1}
 *   {"ts":1740147600.003,"type":"event","metric":"bootstrap","payload":{"tier":3}}
 * @endcode
 *
 * Usage:
 * @code
 *   auto& td = nikola::diag::TelemetryDaemon::global();
 *   td.start();               // launch drain thread
 *   td.gauge("wave.H", 68890.5, "J");
 *   td.counter("prop.steps", 1);
 *   td.event("bootstrap", R"({"tier":3})");
 *   td.stop();                // flush + join drain thread
 * @endcode
 */

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>  // write(), close(), POSIX

namespace nikola::diag {

// ─────────────────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────────────────

inline constexpr int    STDDBG_FD          = 3;   ///< Standard debug file descriptor
inline constexpr size_t TELEMETRY_QUEUE_CAP = 4096; ///< Power-of-2 Ring buffer capacity

// ─────────────────────────────────────────────────────────────────────────────
//  MetricRecord — fixed-size wire-format buffer (zero heap alloc on hot path)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Single pre-serialised metric record ready for write().
 *
 * Serialised as JSON Lines in the producer thread so the drain thread
 * only needs to call write() — no formatting on the I/O path.
 */
struct MetricRecord {
    static constexpr size_t MAX_LEN = 512;
    char   data[MAX_LEN];
    size_t len = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
//  SPSC Ring Buffer
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Single-Producer Single-Consumer lock-free ring buffer.
 *
 * Capacity must be a power of 2.  try_push() never blocks; returns false
 * (drop) when full.  pop() never blocks; returns false when empty.
 */
template<typename T, size_t Cap>
class SpscRing {
    static_assert((Cap & (Cap - 1)) == 0, "Cap must be power of 2");
public:
    bool try_push(T&& item) noexcept {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next = (head + 1) & MASK;
        if (next == tail_.load(std::memory_order_acquire)) return false;  // full
        buf_[head] = std::move(item);
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& out) noexcept {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) return false;  // empty
        out = std::move(buf_[tail]);
        tail_.store((tail + 1) & MASK, std::memory_order_release);
        return true;
    }

    size_t size_approx() const noexcept {
        const size_t h = head_.load(std::memory_order_relaxed);
        const size_t t = tail_.load(std::memory_order_relaxed);
        return (h - t) & MASK;
    }

    bool empty() const noexcept { return size_approx() == 0; }

private:
    static constexpr size_t MASK = Cap - 1;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    std::array<T, Cap> buf_{};
};

// ─────────────────────────────────────────────────────────────────────────────
//  TelemetryDaemon
// ─────────────────────────────────────────────────────────────────────────────

class TelemetryDaemon {
public:
    // ── Singleton ────────────────────────────────────────────────────────────
    static TelemetryDaemon& global() noexcept {
        static TelemetryDaemon inst;
        return inst;
    }

    // ── Lifecycle ────────────────────────────────────────────────────────────

    /**
     * @brief Start the background drain thread.
     * @param output_fd  File descriptor to write to (default: 3 = stddbg).
     *                   Pass a different fd in tests.
     */
    void start(int output_fd = STDDBG_FD) {
        if (running_.load()) return;
        output_fd_ = output_fd;
        dropped_.store(0);
        running_.store(true);
        drain_thread_ = std::thread(&TelemetryDaemon::drain_loop, this);
    }

    /**
     * @brief Flush all queued records and stop the drain thread.
     * Safe to call multiple times.
     */
    void stop() {
        if (!running_.load()) return;
        running_.store(false);
        cv_.notify_all();
        if (drain_thread_.joinable()) drain_thread_.join();
    }

    ~TelemetryDaemon() { stop(); }

    // ── Metric producers (called from 1kHz hot path) ─────────────────────────

    /**
     * @brief Emit a gauge (instantaneous value).
     * @param metric  Dot-separated metric name e.g. "wave.hamiltonian"
     * @param value   Measurement value
     * @param unit    Optional unit string e.g. "J", "ms", "nodes"
     */
    void gauge(std::string_view metric, double value,
               std::string_view unit = "") noexcept
    {
        if (!running_.load(std::memory_order_relaxed)) return;

        MetricRecord rec;
        const double ts = timestamp_s();
        if (unit.empty()) {
            rec.len = static_cast<size_t>(std::snprintf(
                rec.data, MetricRecord::MAX_LEN,
                "{\"ts\":%.3f,\"type\":\"gauge\",\"metric\":\"%.*s\",\"value\":%.6g}\n",
                ts,
                static_cast<int>(metric.size()), metric.data(),
                value));
        } else {
            rec.len = static_cast<size_t>(std::snprintf(
                rec.data, MetricRecord::MAX_LEN,
                "{\"ts\":%.3f,\"type\":\"gauge\",\"metric\":\"%.*s\","
                "\"value\":%.6g,\"unit\":\"%.*s\"}\n",
                ts,
                static_cast<int>(metric.size()), metric.data(),
                value,
                static_cast<int>(unit.size()), unit.data()));
        }
        push(rec);
    }

    /**
     * @brief Emit a counter delta.
     * @param metric  Metric name
     * @param delta   Amount to add to the cumulative counter (default: 1)
     */
    void counter(std::string_view metric, int64_t delta = 1) noexcept {
        if (!running_.load(std::memory_order_relaxed)) return;

        MetricRecord rec;
        rec.len = static_cast<size_t>(std::snprintf(
            rec.data, MetricRecord::MAX_LEN,
            "{\"ts\":%.3f,\"type\":\"counter\",\"metric\":\"%.*s\",\"delta\":%lld}\n",
            timestamp_s(),
            static_cast<int>(metric.size()), metric.data(),
            static_cast<long long>(delta)));
        push(rec);
    }

    /**
     * @brief Emit a free-form event with a JSON payload.
     * @param metric   Event name / category
     * @param payload  Valid JSON object string (caller's responsibility)
     */
    void event(std::string_view metric, std::string_view payload) noexcept {
        if (!running_.load(std::memory_order_relaxed)) return;

        MetricRecord rec;
        rec.len = static_cast<size_t>(std::snprintf(
            rec.data, MetricRecord::MAX_LEN,
            "{\"ts\":%.3f,\"type\":\"event\",\"metric\":\"%.*s\",\"payload\":%.*s}\n",
            timestamp_s(),
            static_cast<int>(metric.size()), metric.data(),
            static_cast<int>(payload.size()), payload.data()));
        push(rec);
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    /** @brief Number of records dropped (queue full) since last start(). */
    uint64_t dropped() const noexcept { return dropped_.load(); }

    /** @brief Approximate number of records queued but not yet written. */
    size_t queue_depth() const noexcept { return queue_.size_approx(); }

    /** @brief True if the drain thread is running. */
    bool is_running() const noexcept { return running_.load(); }

    // ── Test helpers ─────────────────────────────────────────────────────────

    /**
     * @brief Synchronously flush all queued records to output_fd.
     * Useful in tests where the drain thread latency is undesirable.
     */
    void flush_sync() {
        std::lock_guard<std::mutex> lock(drain_mutex_);
        MetricRecord rec;
        while (queue_.pop(rec)) {
            write_record(rec);
        }
    }

    /** @brief Reset drop counter (test helper). */
    void reset_dropped() noexcept { dropped_.store(0); }

    /// Publicly constructible for test isolation; use global() for production.
    TelemetryDaemon() = default;

    TelemetryDaemon(const TelemetryDaemon&) = delete;
    TelemetryDaemon& operator=(const TelemetryDaemon&) = delete;

private:

    // ── Core internals ────────────────────────────────────────────────────────

    static double timestamp_s() noexcept {
        const auto now = std::chrono::system_clock::now();
        return std::chrono::duration<double>(now.time_since_epoch()).count();
    }

    void push(MetricRecord& rec) noexcept {
        if (!queue_.try_push(std::move(rec))) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
        } else {
            cv_.notify_one();
        }
    }

    void write_record(const MetricRecord& rec) noexcept {
        if (output_fd_ < 0 || rec.len == 0) return;
        // Non-blocking: if the fd is not writable, skip
        // write() is async-signal-safe; loop to handle partial writes
        size_t written = 0;
        while (written < rec.len) {
            const ssize_t n = ::write(output_fd_,
                                      rec.data + written,
                                      rec.len - written);
            if (n <= 0) break;  // fd closed or error — drop silently
            written += static_cast<size_t>(n);
        }
    }

    void drain_loop() {
        MetricRecord rec;
        while (running_.load(std::memory_order_acquire)) {
            // Drain all available
            {
                std::lock_guard<std::mutex> lock(drain_mutex_);
                while (queue_.pop(rec)) {
                    write_record(rec);
                }
            }
            // Wait for more with a short timeout so stop() is responsive
            std::unique_lock lk(cv_mutex_);
            cv_.wait_for(lk, std::chrono::milliseconds(5),
                         [this]{ return !queue_.empty() || !running_.load(); });
        }
        // Final drain after stop()
        std::lock_guard<std::mutex> lock(drain_mutex_);
        while (queue_.pop(rec)) write_record(rec);
    }

    // ── Members ───────────────────────────────────────────────────────────────

    std::atomic<bool>     running_{false};
    int                   output_fd_{STDDBG_FD};
    std::atomic<uint64_t> dropped_{0};

    SpscRing<MetricRecord, TELEMETRY_QUEUE_CAP> queue_;

    std::thread             drain_thread_;
    std::mutex              drain_mutex_;
    std::mutex              cv_mutex_;
    std::condition_variable cv_;
};

} // namespace nikola::diag
