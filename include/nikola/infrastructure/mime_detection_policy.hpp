#pragma once
/**
 * @file include/nikola/infrastructure/mime_detection_policy.hpp
 * @brief v0.3.6 QoL: lightweight MIME detection policy.
 *
 * No external dependency (e.g., libmagic) is required.  Policy order:
 *   1) Trust strong magic signatures (currently PDF)
 *   2) Infer textual content class from bytes (JSON / CSV / plain text)
 *   3) Fall back to extension mapping
 */

#include <nikola/infrastructure/data_watcher.hpp>

#include <cctype>
#include <cstdint>
#include <string>
#include <string_view>

namespace nikola::infrastructure {

enum class MimeType : uint8_t {
    TEXT_PLAIN,
    TEXT_MARKDOWN,
    TEXT_X_CPP,
    TEXT_X_ARIA,
    APPLICATION_JSON,
    TEXT_CSV,
    APPLICATION_PDF,
    APPLICATION_OCTET_STREAM,
};

[[nodiscard]] inline const char* mime_type_name(MimeType mt) noexcept {
    switch (mt) {
        case MimeType::TEXT_PLAIN:               return "text/plain";
        case MimeType::TEXT_MARKDOWN:            return "text/markdown";
        case MimeType::TEXT_X_CPP:               return "text/x-c++src";
        case MimeType::TEXT_X_ARIA:              return "text/x-aria";
        case MimeType::APPLICATION_JSON:         return "application/json";
        case MimeType::TEXT_CSV:                 return "text/csv";
        case MimeType::APPLICATION_PDF:          return "application/pdf";
        default:                                 return "application/octet-stream";
    }
}

[[nodiscard]] inline std::string extension_of(std::string_view path) {
    const auto dot = path.rfind('.');
    if (dot == std::string_view::npos) return {};
    std::string ext(path.substr(dot));
    for (auto& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext;
}

[[nodiscard]] inline MimeType detect_mime_from_path(std::string_view path) {
    const std::string ext = extension_of(path);
    if (ext == ".md") return MimeType::TEXT_MARKDOWN;
    if (ext == ".cpp" || ext == ".hpp" || ext == ".h" || ext == ".cc" || ext == ".cxx") {
        return MimeType::TEXT_X_CPP;
    }
    if (ext == ".aria") return MimeType::TEXT_X_ARIA;
    if (ext == ".json" || ext == ".jsonl") return MimeType::APPLICATION_JSON;
    if (ext == ".csv") return MimeType::TEXT_CSV;
    if (ext == ".pdf") return MimeType::APPLICATION_PDF;
    if (ext == ".txt") return MimeType::TEXT_PLAIN;
    return MimeType::APPLICATION_OCTET_STREAM;
}

[[nodiscard]] inline bool has_pdf_magic(std::string_view bytes) noexcept {
    return bytes.size() >= 5 && bytes[0] == '%' && bytes[1] == 'P' &&
           bytes[2] == 'D' && bytes[3] == 'F' && bytes[4] == '-';
}

[[nodiscard]] inline bool is_likely_text(std::string_view bytes) noexcept {
    if (bytes.empty()) return true;

    std::size_t printable_or_ws = 0;
    for (unsigned char c : bytes) {
        if (c == 0) return false;
        if (std::isprint(c) || std::isspace(c)) printable_or_ws++;
    }
    const double ratio = static_cast<double>(printable_or_ws) /
                         static_cast<double>(bytes.size());
    return ratio >= 0.90;
}

[[nodiscard]] inline std::string_view trim_left_ws(std::string_view s) noexcept {
    std::size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) i++;
    return s.substr(i);
}

[[nodiscard]] inline MimeType detect_mime_from_bytes(std::string_view bytes) {
    if (bytes.empty()) return MimeType::APPLICATION_OCTET_STREAM;

    if (has_pdf_magic(bytes)) return MimeType::APPLICATION_PDF;

    if (!is_likely_text(bytes)) return MimeType::APPLICATION_OCTET_STREAM;

    const auto trimmed = trim_left_ws(bytes);
    if (!trimmed.empty() && (trimmed.front() == '{' || trimmed.front() == '[')) {
        return MimeType::APPLICATION_JSON;
    }

    const auto first_newline = bytes.find('\n');
    const auto first_line = first_newline == std::string_view::npos
        ? bytes
        : bytes.substr(0, first_newline);
    if (first_line.find(',') != std::string_view::npos) {
        return MimeType::TEXT_CSV;
    }

    return MimeType::TEXT_PLAIN;
}

[[nodiscard]] inline MimeType resolve_mime(std::string_view path,
                                           std::string_view bytes) {
    const MimeType by_bytes = detect_mime_from_bytes(bytes);
    if (by_bytes == MimeType::APPLICATION_PDF) {
        return by_bytes; // strong signature wins
    }

    const MimeType by_path = detect_mime_from_path(path);
    if (by_path != MimeType::APPLICATION_OCTET_STREAM) {
        return by_path;
    }

    return by_bytes;
}

[[nodiscard]] inline FileType mime_to_file_type(MimeType mime) noexcept {
    switch (mime) {
        case MimeType::TEXT_PLAIN:       return FileType::TEXT;
        case MimeType::TEXT_MARKDOWN:    return FileType::MARKDOWN;
        case MimeType::TEXT_X_CPP:       return FileType::CODE_CPP;
        case MimeType::TEXT_X_ARIA:      return FileType::CODE_ARIA;
        case MimeType::APPLICATION_JSON: return FileType::JSON;
        case MimeType::TEXT_CSV:         return FileType::CSV;
        default:                         return FileType::UNKNOWN;
    }
}

[[nodiscard]] inline FileType detect_file_type(std::string_view path,
                                               std::string_view bytes) {
    return mime_to_file_type(resolve_mime(path, bytes));
}

} // namespace nikola::infrastructure
