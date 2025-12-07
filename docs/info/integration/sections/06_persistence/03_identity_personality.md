# IDENTITY AND PERSONALITY

## 21.1 Identity Subsystem

**Purpose:** Develop persistent identity and preferences over time.

**Storage:**

```cpp
struct IdentityProfile {
    std::string name = "Nikola";
    std::map<std::string, double> preferences;  // Topic → affinity score
    std::vector<std::string> memories;          // Significant events
    std::map<std::string, int> topic_counts;    // Topic → query count
};
```

**Implementation:**

```cpp
#include "nikola/core/config.hpp"  // DESIGN NOTE (Finding 2.1)

class IdentityManager {
    IdentityProfile profile;
    // DESIGN NOTE (Finding 2.1): Use centralized configuration
    std::string profile_path = nikola::core::Config::get().identity_directory() + "/identity.json";

public:
    void load() {
        std::ifstream file(profile_path);
        if (file.is_open()) {
            nlohmann::json j;
            file >> j;

            profile.name = j["name"];
            profile.preferences = j["preferences"];
            profile.memories = j["memories"];
            profile.topic_counts = j["topic_counts"];
        }
    }

    void save() {
        nlohmann::json j;
        j["name"] = profile.name;
        j["preferences"] = profile.preferences;
        j["memories"] = profile.memories;
        j["topic_counts"] = profile.topic_counts;

        std::ofstream file(profile_path);
        file << j.dump(2);
    }

    void update_preference(const std::string& topic, double delta) {
        profile.preferences[topic] += delta;
    }

    void record_memory(const std::string& event) {
        profile.memories.push_back(event);

        // Keep only recent 1000 memories
        if (profile.memories.size() > 1000) {
            profile.memories.erase(profile.memories.begin());
        }
    }
};
```

## 21.2 Preference Learning

**Update Rule:**

After each interaction:
- If user provides positive feedback → $\text{preference}[\text{topic}] += 0.1$
- If user provides negative feedback → $\text{preference}[\text{topic}] -= 0.1$
- Track query topics to learn interests

## 21.3 Implementation

**Integration:**

```cpp
class PersonalizedOrchestrator : public Orchestrator {
    IdentityManager identity;

public:
    std::string process_query(const std::string& query) override {
        // Extract topic
        std::string topic = extract_topic(query);

        // Update topic count
        identity.profile.topic_counts[topic]++;

        // Process normally
        auto response = Orchestrator::process_query(query);

        // Record memory
        identity.record_memory("Query: " + query);

        // Save periodically
        if (identity.profile.memories.size() % 10 == 0) {
            identity.save();
        }

        return response;
    }
};
```

---

**Cross-References:**
- See Section 11 for Orchestrator base class
- See Section 14 for Dopamine-based reward integration
- See Section 19 for Persistence integration
- See Section 22 for Memory consolidation during Nap
