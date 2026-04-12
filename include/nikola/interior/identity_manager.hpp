/**
 * @file interior/identity_manager.hpp
 * @brief v0.1.18 Phase 4 — Identity persistence: JSON-backed agent profile.
 *
 * Stores an IdentityProfile (name, preferences, memories, topic counts)
 * to disk as JSON and reloads it across NAP cycles and restarts.
 *
 * Preference learning:  ±PREFERENCE_LEARN_RATE per feedback event.
 * Memory recording:     FIFO capped at MAX_MEMORIES.
 *
 * The JSON format is intentionally simple — no external library required.
 * A minimal inline reader/writer handles the four supported types:
 *   string, map<string,double>, vector<string>, map<string,int>.
 *
 * Reference:
 *   Integration Report §21.1–21.3 (IdentityProfile, IdentityManager)
 *   RELEASE_0.1.18.md Phase 4 — Identity Persistence
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace nikola::interior {

// ============================================================================
// Constants
// ============================================================================

/// Maximum stored memories (FIFO eviction).
inline constexpr size_t IDENTITY_MAX_MEMORIES      = 1000;

/// Additive delta per preference update.
inline constexpr double IDENTITY_PREFERENCE_LEARN   = 0.1;

/// Filename within the identity directory.
inline constexpr const char* IDENTITY_FILENAME      = "identity.json";

// ============================================================================
// IdentityProfile
// ============================================================================

struct IdentityProfile {
    std::string                    name = "Nikola";
    std::map<std::string, double>  preferences;     ///< topic → affinity ∈ ℝ
    std::vector<std::string>       memories;         ///< significant events
    std::map<std::string, int>     topic_counts;     ///< topic → query count
};

// ============================================================================
// IdentityManager
// ============================================================================

class IdentityManager {
public:
    /**
     * @brief Construct a manager backed by a directory on disk.
     * @param directory  Path to identity directory (e.g., /var/lib/nikola/identity).
     *                   Must already exist.
     */
    explicit IdentityManager(std::string directory)
        : directory_(std::move(directory))
    {}

    // ── persistence ──────────────────────────────────────────────────────

    /**
     * @brief Load profile from IDENTITY_FILENAME inside the directory.
     * @return true on success, false if file missing or parse error.
     */
    bool load() {
        std::ifstream in(path_());
        if (!in.good()) return false;
        std::string content((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
        return parse_json_(content);
    }

    /**
     * @brief Save profile to IDENTITY_FILENAME inside the directory.
     * @return true on success.
     */
    bool save() const {
        std::ofstream out(path_());
        if (!out.good()) return false;
        out << write_json_();
        return out.good();
    }

    // ── preference learning ──────────────────────────────────────────────

    /**
     * @brief Adjust preference for a topic.
     *
     * preference[topic] += delta * IDENTITY_PREFERENCE_LEARN.
     * Creates the entry if it doesn't exist.
     */
    void update_preference(const std::string& topic, double delta) {
        profile_.preferences[topic] += delta * IDENTITY_PREFERENCE_LEARN;
    }

    // ── memory recording ─────────────────────────────────────────────────

    /**
     * @brief Record a significant event.  FIFO eviction at MAX_MEMORIES.
     */
    void record_memory(const std::string& event) {
        if (profile_.memories.size() >= IDENTITY_MAX_MEMORIES) {
            profile_.memories.erase(profile_.memories.begin());
        }
        profile_.memories.push_back(event);
    }

    // ── topic counting ───────────────────────────────────────────────────

    void increment_topic(const std::string& topic) {
        ++profile_.topic_counts[topic];
    }

    // ── accessors ────────────────────────────────────────────────────────

    [[nodiscard]] const IdentityProfile& profile()   const noexcept { return profile_; }
    [[nodiscard]] IdentityProfile&       profile()         noexcept { return profile_; }
    [[nodiscard]] std::string            identity_path() const      { return path_(); }

private:
    std::string     directory_;
    IdentityProfile profile_;

    [[nodiscard]] std::string path_() const {
        if (directory_.empty()) return IDENTITY_FILENAME;
        if (directory_.back() == '/') return directory_ + IDENTITY_FILENAME;
        return directory_ + "/" + IDENTITY_FILENAME;
    }

    // ── JSON escape / unescape ───────────────────────────────────────────

    static std::string json_escape(const std::string& s) {
        std::string out;
        out.reserve(s.size() + 8);
        for (char c : s) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:   out += c;      break;
            }
        }
        return out;
    }

    static std::string json_unescape(const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '\\' && i + 1 < s.size()) {
                ++i;
                switch (s[i]) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case 'n':  out += '\n'; break;
                    case 'r':  out += '\r'; break;
                    case 't':  out += '\t'; break;
                    default:   out += s[i]; break;
                }
            } else {
                out += s[i];
            }
        }
        return out;
    }

    // ── JSON writer ──────────────────────────────────────────────────────

    [[nodiscard]] std::string write_json_() const {
        std::ostringstream o;
        o << "{\n";
        o << "  \"name\": \"" << json_escape(profile_.name) << "\",\n";

        // preferences
        o << "  \"preferences\": {";
        {
            bool first = true;
            for (const auto& [k, v] : profile_.preferences) {
                if (!first) o << ",";
                o << "\n    \"" << json_escape(k) << "\": " << v;
                first = false;
            }
        }
        o << "\n  },\n";

        // memories
        o << "  \"memories\": [";
        {
            bool first = true;
            for (const auto& m : profile_.memories) {
                if (!first) o << ",";
                o << "\n    \"" << json_escape(m) << "\"";
                first = false;
            }
        }
        o << "\n  ],\n";

        // topic_counts
        o << "  \"topic_counts\": {";
        {
            bool first = true;
            for (const auto& [k, v] : profile_.topic_counts) {
                if (!first) o << ",";
                o << "\n    \"" << json_escape(k) << "\": " << v;
                first = false;
            }
        }
        o << "\n  }\n";

        o << "}\n";
        return o.str();
    }

    // ── JSON reader (minimal, handles our exact format) ──────────────────

    struct JsonReader {
        const std::string& s;
        size_t pos = 0;

        void skip_ws() {
            while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t'
                   || s[pos] == '\n' || s[pos] == '\r'))
                ++pos;
        }

        char peek() { skip_ws(); return pos < s.size() ? s[pos] : '\0'; }

        void expect(char c) { skip_ws(); if (pos < s.size() && s[pos] == c) ++pos; }

        std::string read_string() {
            skip_ws();
            if (pos >= s.size() || s[pos] != '"') return "";
            ++pos;
            std::string result;
            while (pos < s.size() && s[pos] != '"') {
                if (s[pos] == '\\' && pos + 1 < s.size()) {
                    ++pos;
                    switch (s[pos]) {
                        case '"':  result += '"';  break;
                        case '\\': result += '\\'; break;
                        case 'n':  result += '\n'; break;
                        case 'r':  result += '\r'; break;
                        case 't':  result += '\t'; break;
                        default:   result += s[pos]; break;
                    }
                } else {
                    result += s[pos];
                }
                ++pos;
            }
            if (pos < s.size()) ++pos;  // skip closing "
            return result;
        }

        double read_number() {
            skip_ws();
            size_t start = pos;
            if (pos < s.size() && (s[pos] == '-' || s[pos] == '+')) ++pos;
            while (pos < s.size() && (s[pos] >= '0' && s[pos] <= '9'))
                ++pos;
            if (pos < s.size() && s[pos] == '.') {
                ++pos;
                while (pos < s.size() && (s[pos] >= '0' && s[pos] <= '9'))
                    ++pos;
            }
            if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
                ++pos;
                if (pos < s.size() && (s[pos] == '+' || s[pos] == '-'))
                    ++pos;
                while (pos < s.size() && (s[pos] >= '0' && s[pos] <= '9'))
                    ++pos;
            }
            if (pos == start) return 0.0;
            return std::stod(s.substr(start, pos - start));
        }

        int read_int() { return static_cast<int>(read_number()); }
    };

    bool parse_json_(const std::string& content) {
        JsonReader r{content};
        r.expect('{');

        while (r.peek() != '}' && r.peek() != '\0') {
            std::string key = r.read_string();
            r.expect(':');

            if (key == "name") {
                profile_.name = r.read_string();
            } else if (key == "preferences") {
                profile_.preferences.clear();
                r.expect('{');
                while (r.peek() != '}' && r.peek() != '\0') {
                    std::string k = r.read_string();
                    r.expect(':');
                    double v = r.read_number();
                    profile_.preferences[k] = v;
                    if (r.peek() == ',') r.expect(',');
                }
                r.expect('}');
            } else if (key == "memories") {
                profile_.memories.clear();
                r.expect('[');
                while (r.peek() != ']' && r.peek() != '\0') {
                    profile_.memories.push_back(r.read_string());
                    if (r.peek() == ',') r.expect(',');
                }
                r.expect(']');
            } else if (key == "topic_counts") {
                profile_.topic_counts.clear();
                r.expect('{');
                while (r.peek() != '}' && r.peek() != '\0') {
                    std::string k = r.read_string();
                    r.expect(':');
                    int v = r.read_int();
                    profile_.topic_counts[k] = v;
                    if (r.peek() == ',') r.expect(',');
                }
                r.expect('}');
            }

            if (r.peek() == ',') r.expect(',');
        }
        r.expect('}');
        return true;
    }
};

} // namespace nikola::interior
