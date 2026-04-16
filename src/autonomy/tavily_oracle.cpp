/**
 * @file src/autonomy/tavily_oracle.cpp
 * @brief Phase 32 — TavilyOracle implementation.
 *
 * Hand-rolled JSON serialization/parsing — consistent with the rest of the
 * nikola codebase (no external JSON library dependency).
 */

#include <nikola/autonomy/tavily_oracle.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace nikola::autonomy {

// ============================================================================
// Construction
// ============================================================================

TavilyOracle::TavilyOracle(const std::string& api_key)
    : TavilyOracle(TavilyConfig{.api_key = api_key})
{}

TavilyOracle::TavilyOracle(const TavilyConfig& config)
    : config_(config)
    , http_(config.http_config)
{
    if (config_.api_key.empty()) {
        throw std::invalid_argument("TavilyOracle: API key must not be empty");
    }
}

// ============================================================================
// Oracle interface — assess()
// ============================================================================

OracleVerdict TavilyOracle::assess(const std::string& query,
                                   const std::string& content) {
    if (query.empty() || content.empty()) {
        return { 0.0f, "empty query or content" };
    }

    auto response = search(query);
    if (!response.ok() || response.results.empty()) {
        // Can't assess — return neutral
        return { 0.5f, "tavily search failed or empty: " + response.error };
    }

    // Aggregate all result content into a reference corpus
    std::string reference;
    for (const auto& r : response.results) {
        reference += r.content;
        reference += " ";
    }

    float similarity = content_similarity_(content, reference);

    // Blend Tavily's relevance scores with our similarity measure
    float avg_tavily_score = 0.0f;
    for (const auto& r : response.results) {
        avg_tavily_score += r.score;
    }
    avg_tavily_score /= static_cast<float>(response.results.size());

    // Final credibility: 60% content similarity + 40% Tavily relevance
    float credibility = 0.6f * similarity + 0.4f * avg_tavily_score;
    credibility = std::clamp(credibility, 0.0f, 1.0f);

    return { credibility, "tavily: " + std::to_string(response.results.size())
                          + " results, sim=" + std::to_string(similarity) };
}

// ============================================================================
// search() — direct API call
// ============================================================================

TavilySearchResponse TavilyOracle::search(const std::string& query) {
    if (query.empty()) {
        return { query, {}, 0.0, "empty query" };
    }

    std::string body = build_request_json(
        config_.api_key, query, config_.max_results, config_.search_depth);

    auto resp = http_.post_json(config_.endpoint, body);
    ++api_calls_;

    if (!resp.ok()) {
        return { query, {}, resp.elapsed_seconds,
                 "HTTP " + std::to_string(resp.status_code) + ": " + resp.error };
    }

    auto parsed = parse_response_json(resp.body);
    parsed.query = query;
    return parsed;
}

// ============================================================================
// search_text() — convenience
// ============================================================================

std::string TavilyOracle::search_text(const std::string& query) {
    auto response = search(query);
    if (!response.ok() || response.results.empty()) {
        return "";
    }

    std::string text;
    for (const auto& r : response.results) {
        text += "## " + r.title + "\n";
        text += r.url + "\n";
        text += r.content + "\n\n";
    }
    return text;
}

// ============================================================================
// JSON helpers
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
                    // Control characters
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

/// Extract a JSON string value for a given key.
/// Returns empty string if key not found.
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
                case 'u':  val += "\\u"; break;  // Pass through unicode escapes
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
/// Returns 0.0 if key not found.
double extract_json_number(const std::string& json, const std::string& key) {
    const std::string pattern = "\"" + key + "\":";
    auto k = json.find(pattern);
    if (k == std::string::npos) return 0.0;

    const auto vs = k + pattern.size();
    // Skip whitespace
    auto i = vs;
    while (i < json.size() && (json[i] == ' ' || json[i] == '\t')) ++i;

    // Extract the number string
    std::string num_str;
    while (i < json.size() && (std::isdigit(json[i]) || json[i] == '.' ||
                                json[i] == '-' || json[i] == '+' || json[i] == 'e' || json[i] == 'E')) {
        num_str += json[i++];
    }

    if (num_str.empty()) return 0.0;
    try {
        return std::stod(num_str);
    } catch (...) {
        return 0.0;
    }
}

/// Find all objects in a JSON array that follows the given key.
/// Returns start/end positions of each {...} object.
std::vector<std::pair<size_t, size_t>>
find_json_array_objects(const std::string& json, const std::string& key) {
    std::vector<std::pair<size_t, size_t>> objects;

    const std::string pattern = "\"" + key + "\":[";
    auto arr_start = json.find(pattern);
    if (arr_start == std::string::npos) {
        // Try without quotes around brackets
        const std::string pattern2 = "\"" + key + "\": [";
        arr_start = json.find(pattern2);
        if (arr_start == std::string::npos) return objects;
        arr_start += pattern2.size();
    } else {
        arr_start += pattern.size();
    }

    // Find each { ... } in the array
    int depth = 0;
    size_t obj_start = 0;
    bool in_string = false;
    bool escaped = false;

    for (auto i = arr_start; i < json.size(); ++i) {
        char c = json[i];

        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) continue;

        if (c == '{') {
            if (depth == 0) obj_start = i;
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0) {
                objects.emplace_back(obj_start, i + 1);
            }
        } else if (c == ']' && depth == 0) {
            break;  // End of array
        }
    }

    return objects;
}

} // anonymous namespace

// ============================================================================
// build_request_json — static
// ============================================================================

std::string TavilyOracle::build_request_json(
    const std::string& api_key,
    const std::string& query,
    int max_results,
    const std::string& search_depth)
{
    return "{\"api_key\":\"" + json_escape(api_key)
         + "\",\"query\":\"" + json_escape(query)
         + "\",\"max_results\":" + std::to_string(max_results)
         + ",\"search_depth\":\"" + json_escape(search_depth)
         + "\"}";
}

// ============================================================================
// parse_response_json — static
// ============================================================================

TavilySearchResponse TavilyOracle::parse_response_json(const std::string& json) {
    TavilySearchResponse resp;

    if (json.empty()) {
        resp.error = "empty response";
        return resp;
    }

    // Check for error response
    std::string error_msg = extract_json_string(json, "error");
    if (!error_msg.empty()) {
        resp.error = error_msg;
        return resp;
    }
    // Also check "detail" (Tavily error format)
    std::string detail = extract_json_string(json, "detail");
    if (!detail.empty()) {
        resp.error = detail;
        return resp;
    }

    resp.query = extract_json_string(json, "query");
    resp.response_time = extract_json_number(json, "response_time");

    // Parse results array
    auto objects = find_json_array_objects(json, "results");
    for (const auto& [start, end] : objects) {
        std::string obj = json.substr(start, end - start);

        TavilyResult result;
        result.url     = extract_json_string(obj, "url");
        result.title   = extract_json_string(obj, "title");
        result.content = extract_json_string(obj, "content");
        result.score   = static_cast<float>(extract_json_number(obj, "score"));

        // Skip results with no content
        if (!result.content.empty()) {
            resp.results.push_back(std::move(result));
        }
    }

    return resp;
}

// ============================================================================
// content_similarity_ — word overlap metric
// ============================================================================

float TavilyOracle::content_similarity_(const std::string& content,
                                        const std::string& reference) {
    if (content.empty() || reference.empty()) return 0.0f;

    // Tokenize into lowercase word sets
    auto tokenize = [](const std::string& text) -> std::unordered_set<std::string> {
        std::unordered_set<std::string> words;
        std::string word;
        for (char c : text) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                word += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            } else {
                if (word.size() >= 3) {  // Skip short words (a, an, the, is, ...)
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

    // Jaccard-like: |intersection| / |content_words|
    size_t overlap = 0;
    for (const auto& w : content_words) {
        if (reference_words.count(w)) ++overlap;
    }

    return static_cast<float>(overlap) / static_cast<float>(content_words.size());
}

// ============================================================================
// load_tavily_api_key — credential loader
// ============================================================================

std::string load_tavily_api_key(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return "";

    std::string line;
    // Skip lines until we find one starting with "tvly-"
    while (std::getline(file, line)) {
        // Trim whitespace
        auto start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        auto trimmed = line.substr(start);

        if (trimmed.rfind("tvly-", 0) == 0) {
            // Trim trailing whitespace
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
