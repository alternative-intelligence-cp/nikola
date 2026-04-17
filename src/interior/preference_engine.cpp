/**
 * @file src/interior/preference_engine.cpp
 * @brief v0.2.3 Phase 1 — PreferenceEngine implementation.
 */

#include <nikola/interior/preference_engine.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace nikola::interior {

// ── Action type → key name mapping ──────────────────────────────────────────

const char* PreferenceEngine::action_key(int action_type) {
    // Matches ActionType enum values from decision_loop.hpp
    switch (action_type) {
        case  0: return "silent";
        case  1: return "emit_thought";
        case  2: return "store_memory";
        case  3: return "request_lookup";
        case  4: return "explore";
        case  5: return "nap";
        case  6: return "refuse";
        case  7: return "escalate";
        case  8: return "recall_memory";
        case  9: return "reason";
        case 10: return "generate_code";
        case 11: return "pursue_goal";
        default: return "unknown";
    }
}

// ── Learning ────────────────────────────────────────────────────────────────

void PreferenceEngine::learn(PreferenceDomain domain, const std::string& key,
                             double delta, uint64_t tick) {
    auto idx = static_cast<size_t>(domain);
    if (idx >= static_cast<size_t>(PreferenceDomain::COUNT)) return;

    auto& pref = domains_[idx][key];
    pref.value += delta * cfg_.learn_rate;
    pref.value = std::clamp(pref.value, -cfg_.max_value, cfg_.max_value);
    pref.strength += std::abs(delta) * cfg_.learn_rate;
    pref.last_tick = tick;
    pref.update_count++;
}

void PreferenceEngine::learn_from_action(int action_type, uint64_t tick) {
    const char* akey = action_key(action_type);

    // Always learn in ACTIONS domain
    learn(PreferenceDomain::ACTIONS, akey, 1.0, tick);

    // Cross-domain implicit learning
    switch (action_type) {
        case 4:  // EXPLORE
            learn(PreferenceDomain::TOPICS, "novelty", 1.0, tick);
            break;
        case 10: // GENERATE_CODE
            learn(PreferenceDomain::CODE_PATTERNS, "active", 1.0, tick);
            break;
        case 1:  // EMIT_THOUGHT
            learn(PreferenceDomain::INTERACTION_STYLES, "expressive", 1.0, tick);
            break;
        case 9:  // REASON
            learn(PreferenceDomain::TOPICS, "analysis", 1.0, tick);
            break;
        case 3:  // REQUEST_LOOKUP
            learn(PreferenceDomain::DATA_SOURCES, "external", 1.0, tick);
            break;
        default:
            break;
    }
}

// ── Decay ───────────────────────────────────────────────────────────────────

void PreferenceEngine::decay(double dt) {
    const double factor = 1.0 - cfg_.decay_rate * dt;
    if (factor <= 0.0) return;  // Sanity: don't invert

    for (auto& domain : domains_) {
        auto it = domain.begin();
        while (it != domain.end()) {
            it->second.value *= factor;
            // Prune near-zero, weak preferences to prevent unbounded growth
            if (std::abs(it->second.value) < 1e-6 && it->second.strength < 0.01) {
                it = domain.erase(it);
            } else {
                ++it;
            }
        }
    }
}

// ── Query ───────────────────────────────────────────────────────────────────

double PreferenceEngine::query(PreferenceDomain domain,
                               const std::string& key) const {
    auto idx = static_cast<size_t>(domain);
    if (idx >= static_cast<size_t>(PreferenceDomain::COUNT)) return 0.0;

    auto it = domains_[idx].find(key);
    if (it == domains_[idx].end()) return 0.0;
    return it->second.value;
}

const Preference* PreferenceEngine::get(PreferenceDomain domain,
                                        const std::string& key) const {
    auto idx = static_cast<size_t>(domain);
    if (idx >= static_cast<size_t>(PreferenceDomain::COUNT)) return nullptr;

    auto it = domains_[idx].find(key);
    if (it == domains_[idx].end()) return nullptr;
    return &it->second;
}

// ── Action scoring bias ─────────────────────────────────────────────────────

double PreferenceEngine::action_bias(int action_type) const {
    const char* akey = action_key(action_type);
    auto idx = static_cast<size_t>(PreferenceDomain::ACTIONS);

    auto it = domains_[idx].find(akey);
    if (it == domains_[idx].end()) return 0.0;

    const auto& pref = it->second;
    // Only influence scoring if preference is strong enough
    if (pref.strength < cfg_.min_influence_strength) return 0.0;

    // Scale value by strength (saturating), clamped to max_bias
    double raw = pref.value * std::min(pref.strength, 2.0) / 2.0;
    return std::clamp(raw * cfg_.max_bias, -cfg_.max_bias, cfg_.max_bias);
}

// ── Listing ─────────────────────────────────────────────────────────────────

std::vector<std::pair<std::string, Preference>>
PreferenceEngine::list_domain(PreferenceDomain domain) const {
    auto idx = static_cast<size_t>(domain);
    if (idx >= static_cast<size_t>(PreferenceDomain::COUNT)) return {};

    std::vector<std::pair<std::string, Preference>> result;
    result.reserve(domains_[idx].size());
    for (const auto& [k, v] : domains_[idx]) {
        result.emplace_back(k, v);
    }
    return result;
}

std::vector<std::pair<std::string, Preference>>
PreferenceEngine::top_preferences(size_t n) const {
    std::vector<std::pair<std::string, Preference>> all;
    for (size_t d = 0; d < static_cast<size_t>(PreferenceDomain::COUNT); ++d) {
        for (const auto& [k, v] : domains_[d]) {
            all.emplace_back(std::string(domain_name(static_cast<PreferenceDomain>(d)))
                             + ":" + k, v);
        }
    }
    // Sort by absolute value (strongest preferences first)
    std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) {
        return std::abs(a.second.value) > std::abs(b.second.value);
    });
    if (all.size() > n) all.resize(n);
    return all;
}

// ── Stats ───────────────────────────────────────────────────────────────────

PreferenceStats PreferenceEngine::stats() const {
    PreferenceStats s;
    for (size_t d = 0; d < static_cast<size_t>(PreferenceDomain::COUNT); ++d) {
        s.total_preferences += domains_[d].size();
        if (!domains_[d].empty()) s.domains_active++;
        for (const auto& [_, p] : domains_[d]) {
            s.total_updates += p.update_count;
        }
    }
    return s;
}

// ── Persistence ─────────────────────────────────────────────────────────────

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

std::string PreferenceEngine::to_json() const {
    std::ostringstream o;
    o << "{\n";
    bool first_domain = true;
    for (size_t d = 0; d < static_cast<size_t>(PreferenceDomain::COUNT); ++d) {
        if (!first_domain) o << ",\n";
        o << "  \"" << domain_name(static_cast<PreferenceDomain>(d)) << "\": {";
        bool first_entry = true;
        for (const auto& [k, p] : domains_[d]) {
            if (!first_entry) o << ",";
            o << "\n    \"" << json_escape(k) << "\": {"
              << "\"v\":" << p.value
              << ",\"s\":" << p.strength
              << ",\"t\":" << p.last_tick
              << ",\"n\":" << p.update_count
              << "}";
            first_entry = false;
        }
        o << "\n  }";
        first_domain = false;
    }
    o << "\n}\n";
    return o.str();
}

bool PreferenceEngine::from_json(const std::string& json) {
    // Minimal JSON parser — matches the format produced by to_json()
    // Looks for domain names, then key-value objects within each

    reset();

    for (size_t d = 0; d < static_cast<size_t>(PreferenceDomain::COUNT); ++d) {
        const char* dname = domain_name(static_cast<PreferenceDomain>(d));
        std::string marker = std::string("\"") + dname + "\"";
        auto dpos = json.find(marker);
        if (dpos == std::string::npos) continue;

        // Find the opening { for this domain
        auto brace_start = json.find('{', dpos + marker.size());
        if (brace_start == std::string::npos) continue;

        // Find matching closing }
        int depth = 1;
        size_t brace_end = brace_start + 1;
        while (brace_end < json.size() && depth > 0) {
            if (json[brace_end] == '{') depth++;
            else if (json[brace_end] == '}') depth--;
            brace_end++;
        }
        std::string domain_block = json.substr(brace_start + 1, brace_end - brace_start - 2);

        // Parse key: {v:..., s:..., t:..., n:...} entries
        size_t pos = 0;
        while (pos < domain_block.size()) {
            // Find key
            auto kstart = domain_block.find('"', pos);
            if (kstart == std::string::npos) break;
            auto kend = domain_block.find('"', kstart + 1);
            if (kend == std::string::npos) break;
            std::string key = domain_block.substr(kstart + 1, kend - kstart - 1);

            // Find value object
            auto vstart = domain_block.find('{', kend);
            if (vstart == std::string::npos) break;
            auto vend = domain_block.find('}', vstart);
            if (vend == std::string::npos) break;
            std::string vblock = domain_block.substr(vstart + 1, vend - vstart - 1);

            Preference pref;
            // Parse "v":..., "s":..., "t":..., "n":...
            auto parse_field = [&](const char* field) -> double {
                std::string pat = std::string("\"") + field + "\":";
                auto fpos = vblock.find(pat);
                if (fpos == std::string::npos) return 0.0;
                fpos += pat.size();
                auto fend = vblock.find_first_of(",}", fpos);
                if (fend == std::string::npos) fend = vblock.size();
                try {
                    return std::stod(vblock.substr(fpos, fend - fpos));
                } catch (...) {
                    return 0.0;
                }
            };

            pref.value = parse_field("v");
            pref.strength = parse_field("s");
            pref.last_tick = static_cast<uint64_t>(parse_field("t"));
            pref.update_count = static_cast<uint32_t>(parse_field("n"));

            domains_[d][key] = pref;
            pos = vend + 1;
        }
    }
    return true;
}

// ── Reset ───────────────────────────────────────────────────────────────────

void PreferenceEngine::reset() {
    for (auto& domain : domains_) {
        domain.clear();
    }
}

} // namespace nikola::interior
