/**
 * @file src/interior/personality_drift.cpp
 * @brief v0.2.3 Phase 2 — PersonalityDrift implementation.
 */

#include <nikola/interior/personality_drift.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace nikola::interior {

// ── Internal: bounded drift ─────────────────────────────────────────────────

void PersonalityDrift::drift_trait(TraitAxis axis, float direction) {
    auto idx = static_cast<size_t>(axis);
    float sign = (direction > 0.0f) ? 1.0f : -1.0f;
    float amount = std::min(std::abs(direction), cfg_.drift_per_event);

    // Check epoch budget
    float remaining = cfg_.drift_per_epoch - std::abs(epoch_drift_[idx]);
    if (remaining <= 0.0f) return;
    amount = std::min(amount, remaining);

    float delta = sign * amount;
    traits_[idx] = std::clamp(traits_[idx] + delta, -1.0f, 1.0f);
    epoch_drift_[idx] += delta;
}

// ── Apply outcome ───────────────────────────────────────────────────────────

void PersonalityDrift::apply_outcome(const ExperienceOutcome& outcome) {
    ++total_events_;
    float s = outcome.success;  // [-1, +1]

    // CAUTIOUS_BOLD: success with risk → bolder, failure with risk → more cautious
    if (outcome.risk_taken > 0.3f) {
        drift_trait(TraitAxis::CAUTIOUS_BOLD, s * outcome.risk_taken);
    }

    // CURIOUS_FOCUSED: EXPLORE(4) success → more curious; non-EXPLORE success → focused
    if (outcome.action_type == 4) {  // EXPLORE
        drift_trait(TraitAxis::CURIOUS_FOCUSED, -std::abs(s) * 0.5f);  // toward curious
    } else if (s > 0.3f && outcome.action_type != 0) {  // success on non-EXPLORE
        drift_trait(TraitAxis::CURIOUS_FOCUSED, s * 0.3f);  // toward focused
    }

    // ANALYTICAL_INTUITIVE: REASON(9) or GENERATE_CODE(10) success → analytical
    if (outcome.action_type == 9 || outcome.action_type == 10) {
        drift_trait(TraitAxis::ANALYTICAL_INTUITIVE, -s * 0.4f);  // toward analytical
    }

    // PATIENT_URGENT: complex task success → patient; simple success → urgent
    if (outcome.complexity > 0.5f && s > 0.0f) {
        drift_trait(TraitAxis::PATIENT_URGENT, -s * outcome.complexity * 0.5f);  // patient
    } else if (outcome.complexity < 0.3f && s > 0.0f) {
        drift_trait(TraitAxis::PATIENT_URGENT, s * 0.3f);  // urgent
    }

    // VERBOSE_TERSE: EMIT_THOUGHT(1) → verbose
    if (outcome.action_type == 1) {
        drift_trait(TraitAxis::VERBOSE_TERSE, -0.3f);  // toward verbose
    }
}

// ── Homeostatic decay ───────────────────────────────────────────────────────

void PersonalityDrift::decay(float dt) {
    float rate = cfg_.homeostatic_decay_rate * dt;
    for (size_t i = 0; i < N_TRAITS; ++i) {
        if (std::abs(traits_[i]) > 0.01f) {
            float regression = -traits_[i] * rate;
            traits_[i] += regression;
            // Snap to zero if very small
            if (std::abs(traits_[i]) < 1e-5f) traits_[i] = 0.0f;
        }
    }
}

// ── Epoch reset ─────────────────────────────────────────────────────────────

void PersonalityDrift::reset_epoch() {
    epoch_drift_.fill(0.0f);
}

// ── Action multiplier ───────────────────────────────────────────────────────

float PersonalityDrift::action_multiplier(int action_type) const {
    float mult = 1.0f;

    float curious = traits_[static_cast<size_t>(TraitAxis::CURIOUS_FOCUSED)];
    float bold    = traits_[static_cast<size_t>(TraitAxis::CAUTIOUS_BOLD)];
    float verbose = traits_[static_cast<size_t>(TraitAxis::VERBOSE_TERSE)];
    float patient = traits_[static_cast<size_t>(TraitAxis::PATIENT_URGENT)];
    float analytical = traits_[static_cast<size_t>(TraitAxis::ANALYTICAL_INTUITIVE)];

    switch (action_type) {
        case 4:  // EXPLORE — curious personalities explore more
            mult += -curious * 0.15f;  // curious (negative) → boost
            mult += bold * 0.1f;       // bold → slightly more exploration
            break;

        case 9:  // REASON — analytical personalities reason more
            mult += -analytical * 0.15f;  // analytical (negative) → boost
            mult += -patient * 0.05f;     // patient → slight boost
            break;

        case 10: // GENERATE_CODE — bold + analytical → more code gen
            mult += bold * 0.1f;
            mult += -analytical * 0.1f;
            break;

        case 1:  // EMIT_THOUGHT — verbose personalities emit more
            mult += -verbose * 0.15f;  // verbose (negative) → boost
            break;

        case 2:  // STORE_MEMORY — patient personalities store more
            mult += -patient * 0.1f;  // patient (negative) → boost
            break;

        case 11: // PURSUE_GOAL — urgent + bold → more goal pursuit
            mult += patient * 0.1f;   // urgent (positive) → boost
            mult += bold * 0.1f;
            break;

        case 6:  // REFUSE — cautious personalities refuse more readily
            mult += -bold * 0.1f;  // cautious (negative bold) → boost
            break;

        case 3:  // REQUEST_LOOKUP — curious + analytical
            mult += -curious * 0.1f;
            mult += -analytical * 0.05f;
            break;

        default:
            break;
    }

    return std::clamp(mult, 0.7f, 1.3f);
}

// ── Description ─────────────────────────────────────────────────────────────

std::string PersonalityDrift::describe() const {
    std::ostringstream o;
    o << "Personality (";
    bool first = true;
    for (size_t i = 0; i < N_TRAITS; ++i) {
        auto axis = static_cast<TraitAxis>(i);
        if (std::abs(traits_[i]) >= cfg_.balanced_threshold) {
            if (!first) o << ", ";
            o << trait_description(axis, traits_[i]);
            first = false;
        }
    }
    if (first) o << "balanced";
    o << ")";
    return o.str();
}

// ── Persistence ─────────────────────────────────────────────────────────────

std::string PersonalityDrift::to_json() const {
    std::ostringstream o;
    o << "{\n  \"traits\": {";
    for (size_t i = 0; i < N_TRAITS; ++i) {
        if (i > 0) o << ",";
        o << "\n    \"" << trait_axis_name(static_cast<TraitAxis>(i))
          << "\": " << traits_[i];
    }
    o << "\n  },\n  \"total_events\": " << total_events_ << "\n}\n";
    return o.str();
}

bool PersonalityDrift::from_json(const std::string& json) {
    for (size_t i = 0; i < N_TRAITS; ++i) {
        auto axis = static_cast<TraitAxis>(i);
        std::string key = std::string("\"") + trait_axis_name(axis) + "\":";
        auto pos = json.find(key);
        if (pos == std::string::npos) continue;
        pos += key.size();
        // Skip whitespace
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
        auto end = json.find_first_of(",}\n", pos);
        if (end == std::string::npos) end = json.size();
        try {
            traits_[i] = std::stof(json.substr(pos, end - pos));
        } catch (...) {
            traits_[i] = 0.0f;
        }
    }
    // Parse total_events
    auto epos = json.find("\"total_events\":");
    if (epos != std::string::npos) {
        epos += 16;  // length of "total_events":
        while (epos < json.size() && (json[epos] == ' ' || json[epos] == '\t')) epos++;
        auto eend = json.find_first_of(",}\n", epos);
        if (eend == std::string::npos) eend = json.size();
        try {
            total_events_ = static_cast<uint64_t>(std::stoull(json.substr(epos, eend - epos)));
        } catch (...) {
            total_events_ = 0;
        }
    }
    return true;
}

} // namespace nikola::interior
