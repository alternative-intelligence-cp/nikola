#include "nikola/aria/metaprogramming.hpp"
#include <stdexcept>

namespace nikola::aria {

AriaMetaprogramming::AriaMetaprogramming() {
    // Phase 0c: Stub only
    // Initialize with default context
    push_context("global");
}

AriaMetaprogramming::~AriaMetaprogramming() {
    // Phase 0c: Stub only
}

std::string AriaMetaprogramming::expand_macros(const std::string& source_code) {
    // Phase 0c: Stub implementation
    // Phase 8: Full NASM-style macro expansion
    return "// AriaMetaprogramming::expand_macros() not implemented (Phase 0c stub)\n" +
           source_code;
}

AriaInterpreter::Value AriaMetaprogramming::evaluate_comptime(
    const std::string& expression)
{
    // Phase 0c: Stub implementation
    throw std::runtime_error(
        "AriaMetaprogramming::evaluate_comptime() not implemented (Phase 0c stub)"
    );
}

std::string AriaMetaprogramming::invoke_macro(
    const std::string& macro_name,
    const std::vector<std::string>& args)
{
    // Phase 0c: Stub implementation
    return "// Macro " + macro_name + " not expanded (Phase 0c stub)\n";
}

void AriaMetaprogramming::define_macro(
    const std::string& name,
    const std::string& parameters,
    const std::string& body)
{
    // Phase 0c: Store in map for stub purposes
    macro_definitions_[name] = body;
    macro_parameters_[name] = parameters;
}

std::string AriaMetaprogramming::generate_specialized_version(
    const std::string& generic_function,
    const std::vector<AriaInterpreter::Value>& concrete_types)
{
    // Phase 0c: Stub implementation
    return "// AriaMetaprogramming::generate_specialized_version() not implemented (Phase 0c stub)\n" +
           generic_function;
}

void AriaMetaprogramming::push_context(const std::string& context_name) {
    // Phase 0c: Basic context stack management
    MacroContext ctx;
    ctx.name = context_name;
    ctx.depth = context_stack_.size();
    context_stack_.push(ctx);
}

void AriaMetaprogramming::pop_context() {
    // Phase 0c: Basic context stack management
    if (!context_stack_.empty()) {
        context_stack_.pop();
    }
}

std::string AriaMetaprogramming::current_context() const {
    // Phase 0c: Return current context name
    if (!context_stack_.empty()) {
        return context_stack_.top().name;
    }
    return "";
}

void AriaMetaprogramming::set_local_define(
    const std::string& name,
    const std::string& value)
{
    // Phase 0c: Store in current context
    if (!context_stack_.empty()) {
        context_stack_.top().local_defines[name] = value;
    }
}

std::string AriaMetaprogramming::get_local_define(const std::string& name) const {
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
