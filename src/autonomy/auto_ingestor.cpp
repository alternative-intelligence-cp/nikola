/**
 * @file src/autonomy/auto_ingestor.cpp
 * @brief v0.2.2 Phase 2 — AutoIngestor implementation.
 */

#include <nikola/autonomy/auto_ingestor.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>

namespace nikola::autonomy {

using infrastructure::FileType;
using Clock = std::chrono::steady_clock;

// ============================================================================
// Constructor
// ============================================================================

AutoIngestor::AutoIngestor(const AutoIngestorConfig& cfg)
    : cfg_(cfg)
{}

// ============================================================================
// stats / reset
// ============================================================================

IngestionStats AutoIngestor::stats() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void AutoIngestor::reset_stats() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = {};
}

// ============================================================================
// read_file_content
// ============================================================================

std::string AutoIngestor::read_file_content(const std::string& path,
                                             size_t max_bytes) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) return {};

    auto size = static_cast<size_t>(ifs.tellg());
    if (size > max_bytes) return {};   // too large
    ifs.seekg(0, std::ios::beg);

    std::string content(size, '\0');
    ifs.read(content.data(), static_cast<std::streamsize>(size));
    return content;
}

// ============================================================================
// Chunking — paragraphs (text / markdown)
// ============================================================================

std::vector<std::string> AutoIngestor::chunk_paragraphs(const std::string& content) {
    std::vector<std::string> chunks;
    std::istringstream stream(content);
    std::string line;
    std::string current;

    while (std::getline(stream, line)) {
        // Blank line → paragraph boundary
        bool blank = true;
        for (char c : line) {
            if (c != ' ' && c != '\t' && c != '\r') { blank = false; break; }
        }

        if (blank) {
            if (!current.empty()) {
                chunks.push_back(std::move(current));
                current.clear();
            }
        } else {
            if (!current.empty()) current += '\n';
            current += line;
        }
    }
    if (!current.empty()) {
        chunks.push_back(std::move(current));
    }
    return chunks;
}

// ============================================================================
// Chunking — code (C++ / Aria)
// ============================================================================

std::vector<std::string> AutoIngestor::chunk_code(const std::string& content) {
    // Split on function/class boundaries:
    //   - Lines starting with a non-space character after a blank line
    //   - Lines containing '{' at the end (function/class opening)
    // Fallback: split on double-newlines like paragraphs
    std::vector<std::string> chunks;
    std::istringstream stream(content);
    std::string line;
    std::string current;
    bool prev_blank = false;

    while (std::getline(stream, line)) {
        bool blank = true;
        for (char c : line) {
            if (c != ' ' && c != '\t' && c != '\r') { blank = false; break; }
        }

        if (blank) {
            prev_blank = true;
            if (!current.empty()) current += '\n';
            continue;
        }

        // Start new chunk if:
        //   - Previous line was blank AND current line starts at column 0
        //     (top-level definition)
        bool is_top_level = !line.empty() && line[0] != ' ' && line[0] != '\t';
        if (prev_blank && is_top_level && !current.empty()) {
            // Trim trailing whitespace from current chunk
            while (!current.empty() &&
                   (current.back() == '\n' || current.back() == ' '))
                current.pop_back();
            if (!current.empty()) chunks.push_back(std::move(current));
            current.clear();
        }

        if (!current.empty()) current += '\n';
        current += line;
        prev_blank = false;
    }

    // Last chunk
    while (!current.empty() &&
           (current.back() == '\n' || current.back() == ' '))
        current.pop_back();
    if (!current.empty()) chunks.push_back(std::move(current));

    return chunks;
}

// ============================================================================
// Chunking — JSON
// ============================================================================

std::vector<std::string> AutoIngestor::chunk_json(const std::string& content) {
    std::vector<std::string> chunks;

    // JSONL: one object per line
    if (content.find('\n') != std::string::npos) {
        std::istringstream stream(content);
        std::string line;
        while (std::getline(stream, line)) {
            // Trim
            size_t start = line.find_first_not_of(" \t\r");
            if (start == std::string::npos) continue;
            line = line.substr(start);

            // Must start with { or [
            if (line.empty()) continue;
            if (line[0] == '{' || line[0] == '[') {
                chunks.push_back(std::move(line));
            }
        }
        if (!chunks.empty()) return chunks;
    }

    // Single JSON object/array: treat whole file as one chunk
    std::string trimmed = content;
    size_t start = trimmed.find_first_not_of(" \t\r\n");
    if (start != std::string::npos) {
        trimmed = trimmed.substr(start);
    }
    if (!trimmed.empty()) {
        chunks.push_back(std::move(trimmed));
    }
    return chunks;
}

// ============================================================================
// Chunking — CSV
// ============================================================================

std::vector<std::string> AutoIngestor::chunk_csv(const std::string& content) {
    std::vector<std::string> chunks;
    std::istringstream stream(content);
    std::string header;
    std::string line;

    // First non-empty line is the header
    while (std::getline(stream, header)) {
        if (!header.empty() && header.find_first_not_of(" \t\r") != std::string::npos)
            break;
    }

    if (header.empty()) return chunks;

    // Each subsequent row becomes "header\nrow"
    while (std::getline(stream, line)) {
        size_t start = line.find_first_not_of(" \t\r");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        if (line.empty()) continue;

        chunks.push_back(header + "\n" + line);
    }

    // If no data rows, at least return the header
    if (chunks.empty()) {
        chunks.push_back(header);
    }
    return chunks;
}

// ============================================================================
// chunk_text — dispatcher
// ============================================================================

std::vector<std::string> AutoIngestor::chunk_text(const std::string& content,
                                                   FileType type) {
    switch (type) {
        case FileType::TEXT:
        case FileType::MARKDOWN:
            return chunk_paragraphs(content);
        case FileType::CODE_CPP:
        case FileType::CODE_ARIA:
            return chunk_code(content);
        case FileType::JSON:
            return chunk_json(content);
        case FileType::CSV:
            return chunk_csv(content);
        default:
            // Unknown: treat as single paragraph
            return chunk_paragraphs(content);
    }
}

// ============================================================================
// ingest_file
// ============================================================================

IngestionResult AutoIngestor::ingest_file(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);

    IngestionResult result;
    result.file_path = path;
    result.file_type = infrastructure::DataWatcher::classify(path);

    auto t0 = Clock::now();

    // Read file
    auto content = read_file_content(path, cfg_.max_file_bytes);
    if (content.empty()) {
        result.error = "file empty, unreadable, or exceeds size limit";
        stats_.files_processed++;
        stats_.files_failed++;
        return result;
    }

    // Chunk
    auto chunks = chunk_text(content, result.file_type);
    result.chunks_total = chunks.size();

    // Inject + store each chunk
    for (const auto& chunk : chunks) {
        // Skip chunks that are too small
        if (chunk.size() < cfg_.min_chunk_chars) {
            result.chunks_skipped++;
            continue;
        }

        // Truncate chunks that are too large
        std::string to_ingest = chunk;
        if (to_ingest.size() > cfg_.max_chunk_chars) {
            to_ingest.resize(cfg_.max_chunk_chars);
        }

        // Quality filter check (if filter is set)
        if (filter_) {
            auto verdict = filter_->check(to_ingest);
            if (verdict != FilterVerdict::ACCEPT) {
                result.chunks_filtered++;
                continue;
            }
        }

        // Inject into torus
        if (inject_fn_) {
            inject_fn_(to_ingest);
        }

        // Let wave settle
        if (tick_fn_) {
            tick_fn_(cfg_.ticks_per_chunk);
        }

        // Store wave-function snapshot
        if (store_fn_) {
            store_fn_();
        }

        // Record in filter for dedup tracking
        if (filter_) {
            filter_->record_ingested(to_ingest);
        }

        result.chunks_ingested++;
    }

    auto t1 = Clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(t1 - t0).count();
    result.success = true;

    // Update stats
    stats_.files_processed++;
    stats_.files_succeeded++;
    stats_.total_chunks   += result.chunks_total;
    stats_.total_ingested += result.chunks_ingested;
    stats_.total_skipped  += result.chunks_skipped;
    stats_.total_filtered += result.chunks_filtered;
    stats_.total_elapsed_s += result.elapsed_seconds;

    return result;
}

// ============================================================================
// ingest_event
// ============================================================================

IngestionResult AutoIngestor::ingest_event(const infrastructure::FileEvent& ev) {
    if (ev.kind == infrastructure::FileEvent::DELETED) {
        IngestionResult result;
        result.file_path = ev.path;
        result.file_type = ev.type;
        result.error = "file deleted, nothing to ingest";
        return result;
    }
    return ingest_file(ev.path);
}

// ============================================================================
// process_events
// ============================================================================

std::vector<IngestionResult> AutoIngestor::process_events(
    std::vector<infrastructure::FileEvent> events) {
    std::vector<IngestionResult> results;
    results.reserve(events.size());
    for (const auto& ev : events) {
        results.push_back(ingest_event(ev));
    }
    return results;
}

} // namespace nikola::autonomy
