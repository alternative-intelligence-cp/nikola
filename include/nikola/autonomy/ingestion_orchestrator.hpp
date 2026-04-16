#pragma once
/**
 * @file include/nikola/autonomy/ingestion_orchestrator.hpp
 * @brief v0.2.2 Phase 4 — IngestionOrchestrator: wires DataWatcher, AutoIngestor,
 *        and IngestionFilter into the NAP cycle for automatic training data ingestion.
 *
 * During ACTIVE state, the orchestrator watches the inbox directory and queues
 * file events.  During NAP, it processes the queue (low priority, bounded by
 * a per-nap budget).  On-demand ingestion is also supported for GoalSystem-
 * triggered TRAINING goals.
 *
 * Design:
 *   · Owns DataWatcher, AutoIngestor, IngestionFilter.
 *   · Callbacks (inject, store, tick) wired by the caller at setup time.
 *   · nap_ingest() called from NapOrchestrator's on_nap_tick callback.
 *   · Bounded: max files per nap, max bytes per day.
 *   · Thread-safe: internal mutex for queue access.
 *
 * Phase: NIK-INGEST-04 (IngestionOrchestrator, v0.2.2)
 */

#include <nikola/autonomy/auto_ingestor.hpp>
#include <nikola/autonomy/ingestion_filter.hpp>
#include <nikola/infrastructure/data_watcher.hpp>

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace nikola::autonomy {

// ============================================================================
// IngestionOrchestratorConfig
// ============================================================================

struct IngestionOrchestratorConfig {
    /// DataWatcher config (inbox directory, debounce, etc.)
    infrastructure::DataWatcherConfig watcher_cfg;

    /// AutoIngestor config (chunk sizes, ticks per chunk, etc.)
    AutoIngestorConfig ingestor_cfg;

    /// IngestionFilter config (dedup, safety, budget, etc.)
    IngestionFilterConfig filter_cfg;

    /// Maximum files to process per NAP cycle.
    size_t max_files_per_nap = 10;
};

// ============================================================================
// NapIngestionReport — stats for a single NAP ingestion pass
// ============================================================================

struct NapIngestionReport {
    size_t files_processed = 0;
    size_t files_succeeded = 0;
    size_t chunks_ingested = 0;
    size_t chunks_filtered = 0;
    double elapsed_seconds = 0.0;
    size_t queue_remaining = 0;   ///< Events still queued after this pass
};

// ============================================================================
// IngestionOrchestrator
// ============================================================================

class IngestionOrchestrator {
public:
    explicit IngestionOrchestrator(const IngestionOrchestratorConfig& cfg = {});
    ~IngestionOrchestrator();

    // Non-copyable
    IngestionOrchestrator(const IngestionOrchestrator&)            = delete;
    IngestionOrchestrator& operator=(const IngestionOrchestrator&) = delete;

    // ── Setup (call before start) ────────────────────────────────────────

    /// Wire the torus injection callback.
    void set_inject_fn(AutoIngestor::InjectFn fn);

    /// Wire the memory store callback.
    void set_store_fn(AutoIngestor::StoreFn fn);

    /// Wire the physics tick callback.
    void set_tick_fn(AutoIngestor::TickFn fn);

    /// Set optional relevance scoring callback for the filter.
    void set_relevance_fn(IngestionFilter::RelevanceFn fn);

    // ── Lifecycle ────────────────────────────────────────────────────────

    /// Start the file watcher.  Returns false if inotify setup fails.
    bool start();

    /// Stop the file watcher.
    void stop();

    /// True if watcher is running.
    [[nodiscard]] bool running() const noexcept;

    // ── NAP Integration ──────────────────────────────────────────────────

    /// Collect pending file events from the watcher into the internal queue.
    /// Call periodically (e.g., every tick) to drain the watcher.
    void collect_events();

    /// Process queued events during a NAP cycle.
    /// Processes up to max_files_per_nap files, returns a report.
    [[nodiscard]] NapIngestionReport nap_ingest();

    /// Process a single file on demand (e.g., GoalSystem TRAINING goal).
    [[nodiscard]] IngestionResult ingest_on_demand(const std::string& path);

    // ── Observers ────────────────────────────────────────────────────────

    /// Number of events queued for NAP processing.
    [[nodiscard]] size_t queue_size() const noexcept;

    /// Cumulative AutoIngestor statistics.
    [[nodiscard]] IngestionStats ingestor_stats() const noexcept;

    /// Cumulative IngestionFilter statistics.
    [[nodiscard]] IngestionFilterStats filter_stats() const noexcept;

    /// Reset the daily ingestion budget (call at midnight or on demand).
    void reset_daily_budget() noexcept;

    /// Access the watcher's watch directory.
    [[nodiscard]] const std::string& watch_dir() const noexcept;

private:
    IngestionOrchestratorConfig     cfg_;
    infrastructure::DataWatcher     watcher_;
    AutoIngestor                    ingestor_;
    std::shared_ptr<IngestionFilter> filter_;

    mutable std::mutex                          queue_mutex_;
    std::vector<infrastructure::FileEvent>      event_queue_;
};

} // namespace nikola::autonomy
