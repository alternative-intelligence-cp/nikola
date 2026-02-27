/**
 * @file autobiography.cpp
 * @brief Phase 124 — AutobiographicalMemory implementation
 */

#include <nikola/interior/autobiography.hpp>

#include <algorithm>
#include <sstream>
#include <numeric>
#include <cctype>
#include <map>

namespace nikola::interior {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

std::string to_lower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    std::string cur;
    for (char c : text) {
        if (std::isalpha(static_cast<unsigned char>(c))) {
            cur += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else {
            if (!cur.empty()) { words.push_back(cur); cur.clear(); }
        }
    }
    if (!cur.empty()) words.push_back(cur);
    return words;
}

double clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }

} // anonymous namespace

// ---------------------------------------------------------------------------
// Pure-static
// ---------------------------------------------------------------------------

bool AutobiographicalMemory::text_matches(const std::string& text,
                                           const std::string& query) {
    if (query.empty()) return false;
    std::string ltext = to_lower(text);
    auto keywords = split_words(query);
    for (const auto& kw : keywords)
        if (ltext.find(kw) != std::string::npos) return true;
    return false;
}

std::string AutobiographicalMemory::affect_label(Affect a) {
    switch (a) {
        case Affect::CURIOSITY:    return "curious";
        case Affect::FRUSTRATION:  return "frustrated";
        case Affect::SATISFACTION: return "satisfied";
        case Affect::CONCERN:      return "concerned";
        case Affect::BOREDOM:      return "bored";
        case Affect::INTEREST:     return "interested";
        case Affect::CONFUSION:    return "confused";
        case Affect::CONFIDENCE:   return "confident";
        case Affect::ANXIETY:      return "anxious";
        case Affect::EXCITEMENT:   return "excited";
        case Affect::NEUTRAL:
        default:                   return "neutral";
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

SkillLevel* AutobiographicalMemory::find_skill(const std::string& name) {
    for (auto& s : skills_) if (s.skill_name == name) return &s;
    return nullptr;
}

ValueEntry* AutobiographicalMemory::find_value(const std::string& name) {
    for (auto& v : values_) if (v.value_name == name) return &v;
    return nullptr;
}

// ---------------------------------------------------------------------------
// record_event
// ---------------------------------------------------------------------------

void AutobiographicalMemory::record_event(
        const std::string&              description,
        const NikolaState&               state,
        Affect                           dominant_affect,
        double                           significance,
        const std::vector<std::string>&  tags) {

    if (events_.size() >= AUTOBIOGRAPHY_MAX_EVENTS)
        events_.erase(events_.begin());

    LifeEvent e;
    e.tick            = events_.empty() ? 0 : events_.back().tick + 1;
    e.description     = description;
    e.state           = state;
    e.dominant_affect = dominant_affect;
    e.significance    = clamp01(significance);
    e.tags            = tags;
    events_.push_back(std::move(e));

    if (event_cb_) event_cb_(events_.back());
}

// ---------------------------------------------------------------------------
// Recall
// ---------------------------------------------------------------------------

std::vector<const LifeEvent*>
AutobiographicalMemory::recall_period(const TickRange& range) const {
    std::vector<const LifeEvent*> result;
    for (const auto& e : events_)
        if (range.contains(e.tick)) result.push_back(&e);
    return result;
}

std::vector<const LifeEvent*>
AutobiographicalMemory::recall_by_query(const std::string& query,
                                         size_t max) const {
    if (query.empty()) return {};
    std::vector<const LifeEvent*> result;
    for (const auto& e : events_) {
        bool match = text_matches(e.description, query);
        if (!match)
            for (const auto& t : e.tags)
                if (text_matches(t, query)) { match = true; break; }
        if (match) result.push_back(&e);
        if (result.size() >= max) break;
    }
    return result;
}

std::vector<const LifeEvent*>
AutobiographicalMemory::get_most_significant(size_t count) const {
    std::vector<const LifeEvent*> candidates;
    for (const auto& e : events_)
        if (e.significance >= AUTOBIOGRAPHY_SIGNIFICANCE_MIN)
            candidates.push_back(&e);
    std::sort(candidates.begin(), candidates.end(),
              [](const LifeEvent* a, const LifeEvent* b){
                  return a->significance > b->significance;
              });
    if (candidates.size() > count) candidates.resize(count);
    return candidates;
}

std::vector<const LifeEvent*>
AutobiographicalMemory::find_by_tag(const std::string& tag) const {
    std::vector<const LifeEvent*> result;
    for (const auto& e : events_)
        for (const auto& t : e.tags)
            if (t == tag) { result.push_back(&e); break; }
    return result;
}

// ---------------------------------------------------------------------------
// Narrative
// ---------------------------------------------------------------------------

std::string AutobiographicalMemory::generate_narrative(
        const TickRange* range) const {

    std::vector<const LifeEvent*> evts;
    if (range) {
        auto p = recall_period(*range);
        evts = std::move(p);
    } else {
        for (const auto& e : events_) evts.push_back(&e);
    }

    if (evts.empty())
        return "No events recorded in this period.";

    // Sort by significance descending, pick up to 5 highlights
    std::vector<const LifeEvent*> highlights = evts;
    std::sort(highlights.begin(), highlights.end(),
              [](const LifeEvent* a, const LifeEvent* b){
                  return a->significance > b->significance;
              });
    if (highlights.size() > 5) highlights.resize(5);

    std::ostringstream ss;
    ss << "I have experienced " << evts.size() << " event"
       << (evts.size() == 1 ? "" : "s") << ". ";

    ss << "Among the most significant: ";
    for (size_t i = 0; i < highlights.size(); ++i) {
        const auto* e = highlights[i];
        ss << "[tick " << e->tick << "] "
           << e->description
           << " (" << affect_label(e->dominant_affect) << ", "
           << "significance=" << static_cast<int>(e->significance * 100) << "%)";
        if (i + 1 < highlights.size()) ss << "; ";
    }
    ss << ".";

    return ss.str();
}

std::string AutobiographicalMemory::get_identity() const {
    std::ostringstream ss;

    // Most common affect across all events
    std::map<Affect, size_t> affect_counts;
    for (const auto& e : events_) affect_counts[e.dominant_affect]++;
    Affect top_affect = Affect::NEUTRAL;
    size_t top_count  = 0;
    for (const auto& [af, cnt] : affect_counts)
        if (cnt > top_count) { top_count = cnt; top_affect = af; }

    ss << "I am a " << affect_label(top_affect) << " entity. ";

    // Top 2 values
    if (!values_.empty()) {
        auto sorted_values = values_;
        std::sort(sorted_values.begin(), sorted_values.end(),
                  [](const ValueEntry& a, const ValueEntry& b){
                      return a.importance > b.importance;
                  });
        ss << "I value ";
        for (size_t i = 0; i < std::min(size_t{2}, sorted_values.size()); ++i) {
            if (i > 0) ss << " and ";
            ss << sorted_values[i].value_name;
        }
        ss << ". ";
    }

    // Top 2 skills
    if (!skills_.empty()) {
        auto sorted_skills = skills_;
        std::sort(sorted_skills.begin(), sorted_skills.end(),
                  [](const SkillLevel& a, const SkillLevel& b){
                      return a.proficiency > b.proficiency;
                  });
        ss << "My strongest skill"
           << (sorted_skills.size() > 1 ? "s are " : " is ");
        for (size_t i = 0; i < std::min(size_t{2}, sorted_skills.size()); ++i) {
            if (i > 0) ss << " and ";
            ss << sorted_skills[i].skill_name;
        }
        ss << ". ";
    }

    ss << "I have recorded " << events_.size() << " event"
       << (events_.size() == 1 ? "" : "s") << ".";

    return ss.str();
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

std::map<std::string, double> AutobiographicalMemory::get_values() const {
    std::map<std::string, double> result;
    for (const auto& v : values_) result[v.value_name] = v.importance;
    return result;
}

void AutobiographicalMemory::update_value(const std::string& value_name,
                                           double delta) {
    auto* v = find_value(value_name);
    if (!v) {
        ValueEntry ve;
        ve.value_name   = value_name;
        ve.importance   = 0.5;
        ve.update_count = 0;
        values_.push_back(std::move(ve));
        v = &values_.back();
    }
    v->importance = clamp01(v->importance +
                            delta * AUTOBIOGRAPHY_VALUE_LEARN_RATE);
    ++v->update_count;
}

std::string AutobiographicalMemory::dominant_value() const {
    if (values_.empty()) return "";
    const ValueEntry* best = &values_[0];
    for (const auto& v : values_)
        if (v.importance > best->importance) best = &v;
    return best->value_name;
}

// ---------------------------------------------------------------------------
// Skills
// ---------------------------------------------------------------------------

void AutobiographicalMemory::update_skill(const std::string& skill_name,
                                           bool success,
                                           uint64_t tick) {
    auto* s = find_skill(skill_name);
    if (!s) {
        SkillLevel sl;
        sl.skill_name = skill_name;
        sl.proficiency = 0.0;
        skills_.push_back(std::move(sl));
        s = &skills_.back();
    }
    if (success) {
        s->proficiency = clamp01(s->proficiency + AUTOBIOGRAPHY_SKILL_LEARN_RATE);
        ++s->success_count;
    } else {
        s->proficiency = clamp01(s->proficiency - AUTOBIOGRAPHY_SKILL_DECAY);
    }
    ++s->practice_count;
    s->last_tick = tick;
}

std::string AutobiographicalMemory::best_skill() const {
    if (skills_.empty()) return "";
    const SkillLevel* best = &skills_[0];
    for (const auto& s : skills_)
        if (s.proficiency > best->proficiency) best = &s;
    return best->skill_name;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

AutobiographicalMemory::Stats AutobiographicalMemory::stats() const {
    Stats s;
    s.total_events = events_.size();
    s.total_skills = skills_.size();
    s.total_values = values_.size();

    if (!events_.empty()) {
        double sum = 0.0;
        for (const auto& e : events_) sum += e.significance;
        s.mean_significance = sum / static_cast<double>(events_.size());

        std::map<Affect, size_t> counts;
        for (const auto& e : events_) counts[e.dominant_affect]++;
        Affect top = Affect::NEUTRAL;
        size_t top_n = 0;
        for (const auto& [af, cnt] : counts)
            if (cnt > top_n) { top_n = cnt; top = af; }
        s.most_common_affect = top;
    }
    return s;
}

} // namespace nikola::interior
