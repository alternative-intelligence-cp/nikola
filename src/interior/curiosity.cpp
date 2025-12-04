#include "nikola/interior/curiosity.hpp"

namespace nikola::interior {

double CuriosityEngine::measure_information_gain(const std::string& query,
                                                const TorusManifold& torus) {
    // STUB: Returns neutral gain - implementation deferred to Phase 6
    return 0.5;
}

std::vector<Question> CuriosityEngine::generate_questions(const TorusManifold& torus, int count) {
    // STUB: Returns empty vector - implementation deferred to Phase 6
    return {};
}

bool CuriosityEngine::pursue_interest(const std::string& topic, TorusManifold& torus) {
    // STUB: Returns false - implementation deferred to Phase 6
    return false;
}

double CuriosityEngine::get_exploration_rate() const {
    return exploration_rate_;
}

void CuriosityEngine::set_exploration_rate(double rate) {
    exploration_rate_ = std::clamp(rate, 0.0, 1.0);
}

std::vector<KnowledgeGap> CuriosityEngine::identify_knowledge_gaps(const TorusManifold& torus) {
    // STUB: Returns empty vector - implementation deferred to Phase 6
    return {};
}

double CuriosityEngine::measure_interestingness(const std::string& topic,
                                               const TorusManifold& torus) {
    // STUB: Returns neutral interest - implementation deferred to Phase 6
    return 0.5;
}

void CuriosityEngine::start_autonomous_learning(TorusManifold& torus, uint64_t interval_ms) {
    // STUB: No-op - implementation deferred to Phase 6
    learning_active_ = true;
}

void CuriosityEngine::stop_autonomous_learning() {
    // STUB: Minimal implementation - deferred to Phase 6
    learning_active_ = false;
}

void CuriosityEngine::set_curiosity_callback(std::function<void(const Question&)> callback) {
    curiosity_callback_ = callback;
}

std::map<std::string, uint64_t> CuriosityEngine::get_stats() const {
    // STUB: Returns zero stats - implementation deferred to Phase 6
    return {
        {"questions_generated", questions_generated_},
        {"topics_pursued", topics_pursued_}
    };
}

} // namespace nikola::interior
