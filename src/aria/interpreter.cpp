#include "nikola/aria/interpreter.hpp"
#include <stdexcept>

namespace nikola::aria {

NitpickInterpreter::NitpickInterpreter() {
    // Phase 0c: Stub only
}

NitpickInterpreter::~NitpickInterpreter() {
    // Phase 0c: Stub only
}

NitpickInterpreter::Value NitpickInterpreter::eval(const std::string& expression) {
    // Phase 0c: Stub implementation
    // Phase 8: Full expression evaluation
    throw std::runtime_error("NitpickInterpreter::eval() not implemented (Phase 0c stub)");
}

void NitpickInterpreter::execute(const std::string& statement) {
    // Phase 0c: Stub implementation
    // Phase 8: Full statement execution
    throw std::runtime_error("NitpickInterpreter::execute() not implemented (Phase 0c stub)");
}

void NitpickInterpreter::load_script(const std::string& script_path) {
    // Phase 0c: Stub implementation
    throw std::runtime_error("NitpickInterpreter::load_script() not implemented (Phase 0c stub)");
}

void NitpickInterpreter::set_variable(const std::string& name, const Value& value) {
    // Phase 0c: Store in map for stub purposes
    global_variables_[name] = value;
}

NitpickInterpreter::Value NitpickInterpreter::get_variable(const std::string& name) {
    // Phase 0c: Retrieve from map
    auto it = global_variables_.find(name);
    if (it != global_variables_.end()) {
        return it->second;
    }
    throw std::runtime_error("Variable not found: " + name);
}

void NitpickInterpreter::register_function(
    const std::string& name,
    std::function<Value(const std::vector<Value>&)> fn)
{
    // Phase 0c: Store in map for stub purposes
    functions_[name] = fn;
}

void NitpickInterpreter::reset() {
    // Phase 0c: Clear state
    global_variables_.clear();
    functions_.clear();
}

} // namespace nikola::aria
