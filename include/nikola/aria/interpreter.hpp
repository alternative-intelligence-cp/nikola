#pragma once

#include <string>
#include <vector>
#include <map>
#include <variant>
#include <functional>
#include <cstdint>

namespace nikola::aria {

/**
 * @brief AriaInterpreter - Runtime evaluation and REPL for Aria code
 *
 * Phase 0c: Stub interface only
 * Implementation: Phase 8 (58-82 hours estimated)
 *
 * Purpose:
 * - Evaluate Aria expressions dynamically
 * - Execute Aria statements
 * - Provide REPL for interactive exploration
 * - Bind C++ functions to Aria namespace
 *
 * Integration Points:
 * - CuriosityEngine: Execute exploratory code snippets
 * - InternalDialogue: Reason about code behavior
 * - DreamEngine: Test hypothetical implementations
 */
class AriaInterpreter {
public:
    struct Value {
        enum class Type {
            NIT,       // Balanced nonary digit {-4..4}
            NYTE,      // 5 nits packed in uint16
            TRIT,      // Balanced ternary digit {-1,0,1}
            TRYTE,     // 10 trits packed in uint16
            INT64,
            FLOAT64,
            STRING,
            FUNCTION,
            VOID
        };

        Type type;
        std::variant<int8_t, uint16_t, int64_t, double, std::string, void*> data;
    };

    AriaInterpreter();
    ~AriaInterpreter();

    // Evaluate Aria expression
    Value eval(const std::string& expression);

    // Execute Aria statement
    void execute(const std::string& statement);

    // Load Aria script
    void load_script(const std::string& script_path);

    // Get/set variables in interpreter context
    void set_variable(const std::string& name, const Value& value);
    Value get_variable(const std::string& name);

    // Bind C++ functions to Aria namespace
    void register_function(const std::string& name,
                          std::function<Value(const std::vector<Value>&)> fn);

    // Clear interpreter state
    void reset();

private:
    std::map<std::string, Value> global_variables_;
    std::map<std::string, std::function<Value(const std::vector<Value>&)>> functions_;
};

} // namespace nikola::aria
