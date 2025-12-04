#pragma once

#include <string>
#include <vector>
#include <map>
#include <stack>

#include "interpreter.hpp"  // For Value type

namespace nikola::aria {

/**
 * @brief AriaMetaprogramming - Handle Aria's macro system and comptime evaluation
 *
 * Phase 0c: Stub interface only
 * Implementation: Phase 8 (58-82 hours estimated)
 *
 * Purpose:
 * - Expand NASM-style macros
 * - Evaluate comptime expressions
 * - Generate code via macro invocation
 * - Define custom macros
 * - Comptime specialization for optimization
 *
 * Integration Points:
 * - AriaCompiler: Macro expansion before code generation
 * - WaveProcessor: Generate specialized wave propagation code at compile time
 * - Performance-critical paths: Eliminate runtime branching
 */
class AriaMetaprogramming {
public:
    AriaMetaprogramming();
    ~AriaMetaprogramming();

    // Expand NASM-style macros
    std::string expand_macros(const std::string& source_code);

    // Evaluate comptime expressions
    AriaInterpreter::Value evaluate_comptime(const std::string& expression);

    // Generate code via macro invocation
    std::string invoke_macro(const std::string& macro_name,
                            const std::vector<std::string>& args);

    // Define custom macro
    void define_macro(const std::string& name,
                     const std::string& parameters,
                     const std::string& body);

    // Comptime code generation for optimization
    std::string generate_specialized_version(
        const std::string& generic_function,
        const std::vector<AriaInterpreter::Value>& concrete_types
    );

    // Push/pop context stack (NASM-style %push, %pop)
    void push_context(const std::string& context_name);
    void pop_context();
    std::string current_context() const;

    // Set/get local defines in current context
    void set_local_define(const std::string& name, const std::string& value);
    std::string get_local_define(const std::string& name) const;

private:
    struct MacroContext {
        std::string name;
        std::map<std::string, std::string> local_defines;
        int depth;
    };

    std::stack<MacroContext> context_stack_;
    std::map<std::string, std::string> macro_definitions_;
    std::map<std::string, std::string> macro_parameters_;
};

} // namespace nikola::aria
