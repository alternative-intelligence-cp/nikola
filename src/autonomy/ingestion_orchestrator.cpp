/**
 * @file src/autonomy/ingestion_orchestrator.cpp
 * @brief v0.2.2 Phase 4 — IngestionOrchestrator implementation.
 */

#include <nikola/autonomy/ingestion_orchestrator.hpp>

#include <algorithm>
#include <chrono>

namespace nikola::autonomy {

// ── Constructor / Destructor ─────────────────────────────────────────────────

IngestionOrchestrator::IngestionOrchestrator(const IngestionOrchestratorConfig& cfg)
    : cfg_(cfg)
    , watcher_(cfg.watcher_cfg)
    , ingestor_(cfg.ingestor_cfg)
    , filter_(std::make_shared<IngestionFilter>(cfg.filter_cfg))
{
    ingestor_.set_filter(filter_);
}

IngestionOrchestrator::~IngestionOrchestrator() {
    stop();
}

// ── Setup ────────────────────────────────────────────────────────────────────

void IngestionOrchestrator::set_inject_fn(AutoIngestor::InjectFn fn) {
    ingestor_.set_inject_fn(std::move(fn));
}

void IngestionOrchestrator::set_store_fn(AutoIngestor::StoreFn fn) {
    ingestor_.set_store_fn(std::move(fn));
}

void IngestionOrchestrator::set_tick_fn(AutoIngestor::TickFn fn) {
    ingestor_.set_tick_fn(std::move(fn));
}

void IngestionOrchestrator::set_relevance_fn(IngestionFilter::RelevanceFn fn) {
    filter_->set_relevance_fn(std::move(fn));
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

bool IngestionOrchestrator::start() {
    return watcher_.start();
}

void IngestionOrchestrator::stop() {
    watcher_.stop();
}

bool IngestionOrchestrator::running() const noexcept {
    return watcher_.running();
}

// ── NAP Integration ──────────────────────────────────────────────────────────

void IngestionOrchestrator::collect_events() {
    auto events = watcher_.poll_events();
    if (events.empty()) return;

    std::lock_guard<std::mutex> lock(queue_mutex_);
    for (auto& ev : events) {
        // Only queue creates/modifications — ignore deletes
        if (ev.kind != infrastructure::FileEvent::Kind::DELETED) {
            event_queue_.push_back(std::move(ev));
        }
    }
}

NapIngestionReport IngestionOrchestrator::nap_ingest() {
    NapIngestionReport report{};

    // Grab batch from queue under lock
    std::vector<infrastructure::FileEvent> batch;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        size_t n = std::min(cfg_.max_files_per_nap, event_queue_.size());
        if (n == 0) {
            report.queue_remaining = 0;
            return report;
        }
        batch.assign(event_queue_.begin(),
                     event_queue_.begin() + static_cast<std::ptrdiff_t>(n));
        event_queue_.erase(event_queue_.begin(),
                           event_queue_.begin() + static_cast<std::ptrdiff_t>(n));
        report.queue_remaining = event_queue_.size();
    }

    auto start = std::chrono::steady_clock::now();

    // Process batch through AutoIngestor
    auto results = ingestor_.process_events(batch);
    for (const auto& r : results) {
        report.files_processed++;
        if (r.chunks_ingested > 0) {
            report.files_succeeded++;
        }
        report.chunks_ingested += r.chunks_ingested;
        report.chunks_filtered += r.chunks_filtered;
    }

    auto end = std::chrono::steady_clock::now();
    report.elapsed_seconds =
        std::chrono::duration<double>(end - start).count();

    return report;
}

IngestionResult IngestionOrchestrator::ingest_on_demand(const std::string& path) {
    return ingestor_.ingest_file(path);
}

// ── Observers ────────────────────────────────────────────────────────────────

size_t IngestionOrchestrator::queue_size() const noexcept {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return event_queue_.size();
}

IngestionStats IngestionOrchestrator::ingestor_stats() const noexcept {
    return ingestor_.stats();
}

IngestionFilterStats IngestionOrchestrator::filter_stats() const noexcept {
    return filter_->stats();
}

void IngestionOrchestrator::reset_daily_budget() noexcept {
    filter_->reset_daily_budget();
}

const std::string& IngestionOrchestrator::watch_dir() const noexcept {
    return cfg_.watcher_cfg.watch_dir;
}

} // namespace nikola::autonomy
