#include "nikola/aria/metaprogramming.hpp"
#include <stdexcept>

namespace nikola::aria {

NitpickMetaprogramming::NitpickMetaprogramming() {
    // Phase 0c: Stub only
    // Initialize with default context
    push_context("global");
}

NitpickMetaprogramming::~NitpickMetaprogramming() {
    // Phase 0c: Stub only
}

std::string NitpickMetaprogramming::expand_macros(const std::string& source_code) {
    // Phase 0c: Stub implementation
    // Phase 8: Full NASM-style macro expansion
    return "// NitpickMetaprogramming::expand_macros() not implemented (Phase 0c stub)\n" +
           source_code;
}

NitpickInterpreter::Value NitpickMetaprogramming::evaluate_comptime(
    const std::string& expression)
{
    // Phase 0c: Stub implementation
    throw std::runtime_error(
        "NitpickMetaprogramming::evaluate_comptime() not implemented (Phase 0c stub)"
    );
}

std::string NitpickMetaprogramming::invoke_macro(
    const std::string& macro_name,
    const std::vector<std::string>& args)
{
    // Phase 0c: Stub implementation
    return "// Macro " + macro_name + " not expanded (Phase 0c stub)\n";
}

void NitpickMetaprogramming::define_macro(
    const std::string& name,
    const std::string& parameters,
    const std::string& body)
{
    // Phase 0c: Store in map for stub purposes
    macro_definitions_[name] = body;
    macro_parameters_[name] = parameters;
}

std::string NitpickMetaprogramming::generate_specialized_version(
    const std::string& generic_function,
    const std::vector<NitpickInterpreter::Value>& concrete_types)
{
    // Phase 0c: Stub implementation
    return "// NitpickMetaprogramming::generate_specialized_version() not implemented (Phase 0c stub)\n" +
           generic_function;
}

void NitpickMetaprogramming::push_context(const std::string& context_name) {
    // Phase 0c: Basic context stack management
    MacroContext ctx;
    ctx.name = context_name;
    ctx.depth = context_stack_.size();
    context_stack_.push(ctx);
}

void NitpickMetaprogramming::pop_context() {
    // Phase 0c: Basic context stack management
    if (!context_stack_.empty()) {
        context_stack_.pop();
    }
}

std::string NitpickMetaprogramming::current_context() const {
    // Phase 0c: Return current context name
    if (!context_stack_.empty()) {
        return context_stack_.top().name;
    }
    return "";
}

void NitpickMetaprogramming::set_local_define(
    const std::string& name,
    const std::string& value)
{
    // Phase 0c: Store in current context
    if (!context_stack_.empty()) {
        context_stack_.top().local_defines[name] = value;
    }
}

std::string NitpickMetaprogramming::get_local_define(const std::string& name) const {
    // Phase 0c: Retrieve from current context
    if (!context_stack_.empty()) {
        const auto& defines = context_stack_.top().local_defines;
        auto it = defines.find(name);
        if (it != defines.end()) {
            return it->second;
        }
    }
    return "";
}

} // namespace nikola::aria
