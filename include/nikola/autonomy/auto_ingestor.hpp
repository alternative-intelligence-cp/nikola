#pragma once
/**
 * @file include/nikola/autonomy/auto_ingestor.hpp
 * @brief v0.2.2 Phase 2 — AutoIngestor: automated training data ingestion
 *        pipeline for files dropped into the watched inbox.
 *
 * Reads files, chunks them by type (paragraphs for text/markdown, functions
 * for code, rows for CSV, objects for JSON), and feeds each chunk through the
 * cognitive torus embedding → semantic memory store path.
 *
 * Design:
 *   · Works with CognitiveTorus + SemanticMemory directly (no DecisionLoop).
 *   · Chunks are injected via torus.inject_text(), then wave-function is
 *     stored into semantic memory — same physics as nikola-train but without
 *     the autonomous scoring loop overhead.
 *   · Thread-safe: one ingestion at a time via internal mutex.
 *   · Tracks per-file and cumulative statistics.
 *
 * Phase: NIK-INGEST-02 (AutoIngestor, v0.2.2)
 */

#include <nikola/autonomy/ingestion_filter.hpp>
#include <nikola/infrastructure/data_watcher.hpp>

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace nikola::autonomy {

// ============================================================================
// ChunkType — how a file was chunked
// ============================================================================

enum class ChunkType : uint8_t {
    PARAGRAPH,   ///< Text/markdown: split on blank lines
    CODE_BLOCK,  ///< Code: split on function/class boundaries
    JSON_OBJECT, ///< JSON: one chunk per top-level object/array element
    CSV_ROW,     ///< CSV: header + each row as a chunk
    WHOLE_FILE   ///< Fallback: entire file as one chunk
};

// ============================================================================
// IngestionResult — stats for a single file ingestion
// ============================================================================

struct IngestionResult {
    std::string file_path;
    infrastructure::FileType file_type = infrastructure::FileType::UNKNOWN;
    size_t      chunks_total     = 0;   ///< Total chunks extracted
    size_t      chunks_ingested  = 0;   ///< Chunks successfully embedded + stored
    size_t      chunks_skipped   = 0;   ///< Chunks skipped (empty, too small, etc.)
    size_t      chunks_filtered  = 0;   ///< Chunks rejected by quality filter
    double      elapsed_seconds  = 0.0; ///< Wall-clock time for this file
    bool        success          = false;
    std::string error;                  ///< Error message if !success
};

// ============================================================================
// IngestionStats — cumulative statistics
// ============================================================================

struct IngestionStats {
    size_t files_processed  = 0;
    size_t files_succeeded  = 0;
    size_t files_failed     = 0;
    size_t total_chunks     = 0;
    size_t total_ingested   = 0;
    size_t total_skipped    = 0;
    size_t total_filtered   = 0;
    double total_elapsed_s  = 0.0;
};

// ============================================================================
// AutoIngestorConfig
// ============================================================================

struct AutoIngestorConfig {
    /// Minimum chunk size in characters to bother embedding.
    size_t min_chunk_chars = 10;

    /// Maximum chunk size in characters (longer chunks are split).
    size_t max_chunk_chars = 4096;

    /// Maximum file size in bytes (files larger than this are skipped).
    size_t max_file_bytes = 10 * 1024 * 1024;  // 10 MB

    /// Number of physics ticks per chunk to let the wave settle.
    int ticks_per_chunk = 5;

    /// Physics steps per tick.
    int steps_per_tick = 10;
};

// ============================================================================
// AutoIngestor
// ============================================================================

class AutoIngestor {
public:
    /// Callback type for torus text injection.
    /// Signature: inject(text) → injects text into torus and ticks.
    using InjectFn = std::function<void(const std::string& text)>;

    /// Callback type for storing current wave-function to memory.
    /// Signature: store() → snapshots current ψ-field.
    using StoreFn  = std::function<void()>;

    /// Callback type for running physics ticks.
    /// Signature: tick(n) → runs n physics ticks.
    using TickFn   = std::function<void(int n)>;

    explicit AutoIngestor(const AutoIngestorConfig& cfg = {});

    /// Wire the injection callback (calls torus.inject_text).
    void set_inject_fn(InjectFn fn) { inject_fn_ = std::move(fn); }

    /// Wire the store callback (calls memory.store + save).
    void set_store_fn(StoreFn fn) { store_fn_ = std::move(fn); }

    /// Wire the tick callback (runs physics ticks).
    void set_tick_fn(TickFn fn) { tick_fn_ = std::move(fn); }

    /// Set an optional quality filter.  Ownership is shared.
    void set_filter(std::shared_ptr<IngestionFilter> f) { filter_ = std::move(f); }

    /// Ingest a single file.  Thread-safe (serialized internally).
    [[nodiscard]] IngestionResult ingest_file(const std::string& path);

    /// Ingest a file from a FileEvent.
    [[nodiscard]] IngestionResult ingest_event(const infrastructure::FileEvent& ev);

    /// Process all pending events from a DataWatcher.
    [[nodiscard]] std::vector<IngestionResult> process_events(
        std::vector<infrastructure::FileEvent> events);

    /// Cumulative statistics.
    [[nodiscard]] IngestionStats stats() const noexcept;

    /// Reset statistics.
    void reset_stats() noexcept;

    // ── Chunking (public for testing) ────────────────────────────────────

    /// Chunk text content by file type.
    [[nodiscard]] static std::vector<std::string> chunk_text(
        const std::string& content, infrastructure::FileType type);

    /// Chunk paragraphs (split on blank lines).
    [[nodiscard]] static std::vector<std::string> chunk_paragraphs(
        const std::string& content);

    /// Chunk code (split on function/class boundaries).
    [[nodiscard]] static std::vector<std::string> chunk_code(
        const std::string& content);

    /// Chunk JSON (one chunk per top-level element).
    [[nodiscard]] static std::vector<std::string> chunk_json(
        const std::string& content);

    /// Chunk CSV (header + row pairs).
    [[nodiscard]] static std::vector<std::string> chunk_csv(
        const std::string& content);

private:
    AutoIngestorConfig cfg_;
    mutable std::mutex mutex_;
    IngestionStats     stats_;

    InjectFn inject_fn_;
    StoreFn  store_fn_;
    TickFn   tick_fn_;
    std::shared_ptr<IngestionFilter> filter_;

    /// Read file contents, respecting max_file_bytes.
    [[nodiscard]] static std::string read_file_content(
        const std::string& path, size_t max_bytes);
};

} // namespace nikola::autonomy
