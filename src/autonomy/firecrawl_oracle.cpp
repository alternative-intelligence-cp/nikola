/**
 * @file src/autonomy/firecrawl_oracle.cpp
 * @brief Phase 32 — FirecrawlOracle implementation.
 *
 * Hand-rolled JSON serialization/parsing — consistent with the rest of the
 * nikola codebase (no external JSON library dependency).
 *
 * Firecrawl v2 scrape API:
 *   POST https://api.firecrawl.dev/v2/scrape
 *   Auth: Bearer <api_key> header
 *   Body: { "url": "...", "formats": ["markdown"], "onlyMainContent": true }
 *   Response: { "success": true, "data": { "markdown": "...", "metadata": {...} } }
 */

#include <nikola/autonomy/firecrawl_oracle.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace nikola::autonomy {

// ============================================================================
// Construction
// ============================================================================

FirecrawlOracle::FirecrawlOracle(const std::string& api_key)
    : FirecrawlOracle(FirecrawlConfig{.api_key = api_key})
{}

FirecrawlOracle::FirecrawlOracle(const FirecrawlConfig& config)
    : config_(config)
    , http_(config.http_config)
{
    if (config_.api_key.empty()) {
        throw std::invalid_argument("FirecrawlOracle: API key must not be empty");
    }
}

// ============================================================================
// Oracle interface — assess()
// ============================================================================

OracleVerdict FirecrawlOracle::assess(const std::string& query,
                                      const std::string& content) {
    if (query.empty() || content.empty()) {
        return { 0.0f, "empty query or content" };
    }

    // Extract URLs from the content to verify against
    auto urls = extract_urls(content);
    if (urls.empty()) {
        return { 0.5f, "firecrawl: no URLs found in content" };
    }

    // Scrape up to 3 URLs to keep API usage reasonable
    const size_t max_urls = std::min(urls.size(), size_t{3});
    std::string reference;
    size_t scraped_count = 0;

    for (size_t i = 0; i < max_urls; ++i) {
        auto resp = scrape(urls[i]);
        if (resp.ok() && !resp.result.markdown.empty()) {
            reference += resp.result.markdown;
            reference += " ";
            ++scraped_count;
        }
    }

    if (scraped_count == 0) {
        return { 0.5f, "firecrawl: could not scrape any referenced URLs" };
    }

    float similarity = content_similarity_(content, reference);
    float credibility = std::clamp(similarity, 0.0f, 1.0f);

    return { credibility, "firecrawl: scraped " + std::to_string(scraped_count)
                          + "/" + std::to_string(urls.size())
                          + " URLs, sim=" + std::to_string(similarity) };
}

// ============================================================================
// scrape() — direct API call
// ============================================================================

FirecrawlScrapeResponse FirecrawlOracle::scrape(const std::string& url) {
    if (url.empty()) {
        return { false, {}, "empty URL" };
    }

    std::string body = build_request_json(
        url, config_.only_main_content, config_.timeout_ms);

    // Firecrawl uses Bearer token auth in the header
    auto resp = http_.post_json(config_.endpoint, body,
        {{"Authorization", "Bearer " + config_.api_key}});
    ++api_calls_;

    if (!resp.ok()) {
        return { false, {}, "HTTP " + std::to_string(resp.status_code)
                            + ": " + resp.error };
    }

    return parse_response_json(resp.body);
}

// ============================================================================
// scrape_markdown() — convenience
// ============================================================================

std::string FirecrawlOracle::scrape_markdown(const std::string& url) {
    auto response = scrape(url);
    if (!response.ok()) {
        return "";
    }
    return response.result.markdown;
}

// ============================================================================
// JSON helpers (anonymous namespace — internal)
// ============================================================================

namespace {

/// JSON-escape a string (handles ", \, newlines, tabs).
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x",
                             static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

/// Extract a JSON string value for a given key.
/// Handles whitespace between : and opening ".
std::string extract_json_string(const std::string& json, const std::string& key) {
    const std::string pattern = "\"" + key + "\":";
    auto k = json.find(pattern);
    if (k == std::string::npos) return "";

    auto vs = k + pattern.size();
    // Skip whitespace between : and opening "
    while (vs < json.size() && (json[vs] == ' ' || json[vs] == '\t' ||
                                 json[vs] == '\n' || json[vs] == '\r')) ++vs;
    if (vs >= json.size() || json[vs] != '"') return "";
    ++vs; // skip opening quote

    std::string val;
    bool escaped = false;
    for (auto i = vs; i < json.size(); ++i) {
        char c = json[i];
        if (escaped) {
            switch (c) {
                case '"':  val += '"';  break;
                case '\\': val += '\\'; break;
                case 'n':  val += '\n'; break;
                case 'r':  val += '\r'; break;
                case 't':  val += '\t'; break;
                case '/':  val += '/';  break;
                case 'u':  val += "\\u"; break;
                default:   val += c;    break;
            }
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (c == '"') {
            break;
        } else {
            val += c;
        }
    }
    return val;
}

/// Extract a JSON number value for a given key.
double extract_json_number(const std::string& json, const std::string& key) {
    const std::string pattern = "\"" + key + "\":";
    auto k = json.find(pattern);
    if (k == std::string::npos) return 0.0;

    auto vs = k + pattern.size();
    auto i = vs;
    while (i < json.size() && (json[i] == ' ' || json[i] == '\t')) ++i;

    std::string num_str;
    while (i < json.size() && (std::isdigit(json[i]) || json[i] == '.' ||
                                json[i] == '-' || json[i] == '+' ||
                                json[i] == 'e' || json[i] == 'E')) {
        num_str += json[i++];
    }

    if (num_str.empty()) return 0.0;
    try {
        return std::stod(num_str);
    } catch (...) {
        return 0.0;
    }
}

/// Extract a JSON boolean value for a given key.
bool extract_json_bool(const std::string& json, const std::string& key) {
    const std::string pattern = "\"" + key + "\":";
    auto k = json.find(pattern);
    if (k == std::string::npos) return false;

    auto vs = k + pattern.size();
    while (vs < json.size() && (json[vs] == ' ' || json[vs] == '\t')) ++vs;

    if (vs + 4 <= json.size() && json.substr(vs, 4) == "true") return true;
    return false;
}

} // anonymous namespace

// ============================================================================
// build_request_json — static
// ============================================================================

std::string FirecrawlOracle::build_request_json(
    const std::string& url,
    bool only_main_content,
    int timeout_ms)
{
    return "{\"url\":\"" + json_escape(url)
         + "\",\"formats\":[\"markdown\"]"
         + ",\"onlyMainContent\":" + (only_main_content ? "true" : "false")
         + ",\"timeout\":" + std::to_string(timeout_ms)
         + "}";
}

// ============================================================================
// parse_response_json — static
// ============================================================================

FirecrawlScrapeResponse FirecrawlOracle::parse_response_json(const std::string& json) {
    FirecrawlScrapeResponse resp;

    if (json.empty()) {
        resp.error = "empty response";
        return resp;
    }

    // Check for success field
    resp.success = extract_json_bool(json, "success");

    // Check for error
    std::string error_msg = extract_json_string(json, "error");
    if (!error_msg.empty()) {
        resp.error = error_msg;
        resp.success = false;
        return resp;
    }

    if (!resp.success) {
        resp.error = "API returned success=false";
        return resp;
    }

    // The response nests content under "data": { ... }
    // Find the "data" object boundary
    const std::string data_pattern = "\"data\":";
    auto data_pos = json.find(data_pattern);
    if (data_pos == std::string::npos) {
        resp.error = "no data field in response";
        resp.success = false;
        return resp;
    }

    // Extract from the data object onwards — our helpers will find
    // the first occurrence of each key, which will be inside data
    auto data_start = data_pos + data_pattern.size();
    // Skip whitespace to find the opening {
    while (data_start < json.size() && json[data_start] != '{') ++data_start;
    if (data_start >= json.size()) {
        resp.error = "malformed data field";
        resp.success = false;
        return resp;
    }

    // Find matching closing brace for the data object
    int depth = 0;
    size_t data_end = data_start;
    bool in_string = false;
    bool escaped = false;
    for (auto i = data_start; i < json.size(); ++i) {
        char c = json[i];
        if (escaped) { escaped = false; continue; }
        if (c == '\\') { escaped = true; continue; }
        if (c == '"') { in_string = !in_string; continue; }
        if (in_string) continue;
        if (c == '{') ++depth;
        else if (c == '}') {
            --depth;
            if (depth == 0) { data_end = i + 1; break; }
        }
    }

    std::string data_json = json.substr(data_start, data_end - data_start);

    // Extract markdown content
    resp.result.markdown = extract_json_string(data_json, "markdown");

    // Extract metadata (nested inside data)
    // metadata fields: title, description, sourceURL, url, statusCode
    auto meta_pos = data_json.find("\"metadata\":");
    if (meta_pos != std::string::npos) {
        // Find the metadata object
        auto ms = meta_pos + 11; // len("\"metadata\":")
        while (ms < data_json.size() && data_json[ms] != '{') ++ms;

        int mdepth = 0;
        size_t me = ms;
        bool ms_in_str = false;
        bool ms_esc = false;
        for (auto i = ms; i < data_json.size(); ++i) {
            char c = data_json[i];
            if (ms_esc) { ms_esc = false; continue; }
            if (c == '\\') { ms_esc = true; continue; }
            if (c == '"') { ms_in_str = !ms_in_str; continue; }
            if (ms_in_str) continue;
            if (c == '{') ++mdepth;
            else if (c == '}') {
                --mdepth;
                if (mdepth == 0) { me = i + 1; break; }
            }
        }

        std::string meta_json = data_json.substr(ms, me - ms);
        resp.result.title = extract_json_string(meta_json, "title");
        resp.result.description = extract_json_string(meta_json, "description");
        resp.result.url = extract_json_string(meta_json, "sourceURL");
        if (resp.result.url.empty()) {
            resp.result.url = extract_json_string(meta_json, "url");
        }
        resp.result.status_code = static_cast<int>(
            extract_json_number(meta_json, "statusCode"));
    }

    return resp;
}

// ============================================================================
// extract_urls — find http/https URLs in text
// ============================================================================

std::vector<std::string> FirecrawlOracle::extract_urls(const std::string& text) {
    std::vector<std::string> urls;
    const std::string http_prefix = "http";

    size_t pos = 0;
    while (pos < text.size()) {
        auto found = text.find(http_prefix, pos);
        if (found == std::string::npos) break;

        // Verify it's http:// or https://
        std::string url_start;
        if (text.substr(found, 8) == "https://") {
            url_start = "https://";
        } else if (text.substr(found, 7) == "http://") {
            url_start = "http://";
        } else {
            pos = found + 1;
            continue;
        }

        // Extract URL until whitespace or certain delimiters
        auto url_begin = found;
        auto i = found + url_start.size();
        while (i < text.size()) {
            char c = text[i];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                c == '"' || c == '\'' || c == '>' || c == '<' ||
                c == ')' || c == ']' || c == '}') {
                break;
            }
            ++i;
        }

        std::string url = text.substr(url_begin, i - url_begin);

        // Strip trailing punctuation that's likely not part of the URL
        while (!url.empty() && (url.back() == '.' || url.back() == ',' ||
                                 url.back() == ';' || url.back() == ':' ||
                                 url.back() == '!')) {
            url.pop_back();
        }

        // Must have something after the scheme
        if (url.size() > url_start.size() + 2) {
            // Deduplicate
            if (std::find(urls.begin(), urls.end(), url) == urls.end()) {
                urls.push_back(std::move(url));
            }
        }

        pos = i;
    }

    return urls;
}

// ============================================================================
// content_similarity_ — word overlap metric
// ============================================================================

float FirecrawlOracle::content_similarity_(const std::string& content,
                                           const std::string& reference) {
    if (content.empty() || reference.empty()) return 0.0f;

    auto tokenize = [](const std::string& text) -> std::unordered_set<std::string> {
        std::unordered_set<std::string> words;
        std::string word;
        for (char c : text) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                word += static_cast<char>(
                    std::tolower(static_cast<unsigned char>(c)));
            } else {
                if (word.size() >= 3) {
                    words.insert(word);
                }
                word.clear();
            }
        }
        if (word.size() >= 3) words.insert(word);
        return words;
    };

    auto content_words = tokenize(content);
    auto reference_words = tokenize(reference);

    if (content_words.empty()) return 0.0f;

    size_t overlap = 0;
    for (const auto& w : content_words) {
        if (reference_words.count(w)) ++overlap;
    }

    return static_cast<float>(overlap) / static_cast<float>(content_words.size());
}

// ============================================================================
// load_firecrawl_api_key — credential loader
// ============================================================================

std::string load_firecrawl_api_key(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return "";

    std::string line;
    while (std::getline(file, line)) {
        auto start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        auto trimmed = line.substr(start);

        if (trimmed.rfind("fc-", 0) == 0) {
            auto end = trimmed.find_last_not_of(" \t\r\n");
            if (end != std::string::npos) {
                return trimmed.substr(0, end + 1);
            }
            return trimmed;
        }
    }

    return "";
}

} // namespace nikola::autonomy
