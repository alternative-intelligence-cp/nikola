/**
 * @file src/autonomy/auto_ingestor.cpp
 * @brief v0.2.2 Phase 2 — AutoIngestor implementation.
 */

#include <nikola/autonomy/auto_ingestor.hpp>
#include <nikola/infrastructure/mime_detection_policy.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <sstream>

namespace {

[[nodiscard]] bool is_printable_pdf_char(unsigned char c) noexcept {
    return c == '\t' || c == '\n' || c == '\r' || (c >= 0x20 && c <= 0x7e);
}

[[nodiscard]] std::string normalize_whitespace(std::string_view in) {
    std::string out;
    out.reserve(in.size());

    bool prev_space = true;
    for (char ch : in) {
        const bool ws = (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t' || ch == '\f' || ch == '\v');
        if (ws) {
            if (!prev_space) {
                out.push_back(' ');
                prev_space = true;
            }
            continue;
        }
        out.push_back(ch);
        prev_space = false;
    }

    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

// Lightweight PDF text extractor:
//  1) Prefer text operands inside BT...ET blocks: ( ... )
//  2) Fallback to long printable runs if no text objects were found.
[[nodiscard]] std::string extract_pdf_text(std::string_view bytes) {
    std::string extracted;
    extracted.reserve(bytes.size() / 8);

    // Pass 1: parse BT ... ET blocks and collect string literals ( ... ).
    size_t i = 0;
    while (i + 1 < bytes.size()) {
        if (!(bytes[i] == 'B' && bytes[i + 1] == 'T')) {
            ++i;
            continue;
        }

        i += 2;
        while (i + 1 < bytes.size() && !(bytes[i] == 'E' && bytes[i + 1] == 'T')) {
            if (bytes[i] != '(') {
                ++i;
                continue;
            }

            ++i; // consume '('
            int depth = 1;
            std::string token;
            token.reserve(64);

            while (i < bytes.size() && depth > 0) {
                char c = bytes[i++];
                if (c == '\\') {
                    if (i >= bytes.size()) break;
                    char esc = bytes[i++];
                    switch (esc) {
                        case 'n': token.push_back('\n'); break;
                        case 'r': token.push_back('\r'); break;
                        case 't': token.push_back('\t'); break;
                        case 'b': token.push_back('\b'); break;
                        case 'f': token.push_back('\f'); break;
                        case '(': token.push_back('('); break;
                        case ')': token.push_back(')'); break;
                        case '\\': token.push_back('\\'); break;
                        default:
                            if (is_printable_pdf_char(static_cast<unsigned char>(esc))) token.push_back(esc);
                            break;
                    }
                    continue;
                }

                if (c == '(') {
                    ++depth;
                    token.push_back(c);
                    continue;
                }
                if (c == ')') {
                    --depth;
                    if (depth == 0) break;
                    token.push_back(c);
                    continue;
                }

                if (is_printable_pdf_char(static_cast<unsigned char>(c))) token.push_back(c);
            }

            if (!token.empty()) {
                if (!extracted.empty()) extracted.push_back(' ');
                extracted += token;
            }
        }

        if (i + 1 < bytes.size()) i += 2; // consume ET
    }

    // Pass 2 fallback: gather printable runs if text objects were absent.
    if (extracted.empty()) {
        std::string run;
        for (char c : bytes) {
            const auto uc = static_cast<unsigned char>(c);
            if (is_printable_pdf_char(uc)) {
                run.push_back(c);
            } else {
                if (run.size() >= 8) {
                    if (!extracted.empty()) extracted.push_back(' ');
                    extracted += run;
                }
                run.clear();
            }
        }
        if (run.size() >= 8) {
            if (!extracted.empty()) extracted.push_back(' ');
            extracted += run;
        }
    }

    return normalize_whitespace(extracted);
}

} // namespace

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
    auto raw_content = read_file_content(path, cfg_.max_file_bytes);
    if (raw_content.empty()) {
        result.error = "file empty, unreadable, or exceeds size limit";
        stats_.files_processed++;
        stats_.files_failed++;
        return result;
    }

    auto content = raw_content;

    // v0.3.6 QoL: MIME-aware file-type fallback (supports content-based
    // detection when extension is absent or misleading).
    const auto mime = infrastructure::resolve_mime(path, raw_content);
    result.file_type = infrastructure::detect_file_type(path, raw_content);

    // v0.3.6 slice 4: lightweight PDF extraction path.
    if (mime == infrastructure::MimeType::APPLICATION_PDF) {
        content = extract_pdf_text(raw_content);
        if (content.empty()) {
            result.error = "pdf contains no extractable text";
            stats_.files_processed++;
            stats_.files_failed++;
            return result;
        }
        // Downstream chunking should treat extracted payload as text.
        result.file_type = FileType::TEXT;
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
