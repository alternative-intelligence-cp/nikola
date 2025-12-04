#pragma once

#include <string>
#include <vector>
#include <utility>

// Forward declarations
namespace nikola {
    class TorusManifold;
    struct Coord9D;
    struct ThoughtTrace;
}

namespace nikola::aria {

/**
 * @brief AriaCodeGenerator - Generate Aria code from Nikola's internal representations
 *
 * Phase 0c: Stub interface only
 * Implementation: Phase 8 (58-82 hours estimated)
 *
 * Purpose:
 * - Generate Aria functions from wave patterns
 * - Translate reasoning chains to Aria code
 * - Generate test cases from examples
 * - Optimize existing Aria code
 * - Convert C++ functions to Aria
 *
 * Integration Points:
 * - CuriosityEngine: Generate exploratory functions
 * - InternalDialogue: Translate reasoning to verifiable code
 * - Self-improvement system: Write optimized versions of components
 */
class AriaCodeGenerator {
public:
    AriaCodeGenerator();
    ~AriaCodeGenerator();

    // Generate Aria function from wave pattern
    std::string generate_from_wave_pattern(
        const TorusManifold& torus,
        const Coord9D& center,
        double radius,
        const std::string& function_name
    );

    // Generate Aria code from reasoning chain
    std::string generate_from_reasoning(
        const std::vector<ThoughtTrace>& reasoning_chain,
        const std::string& function_name
    );

    // Generate Aria test cases from input/output examples
    std::string generate_tests(
        const std::string& function_name,
        const std::vector<std::pair<std::string, std::string>>& input_output_pairs
    );

    // Optimize existing Aria code
    std::string optimize(const std::string& source_code);

    // Convert C++ function to Aria
    std::string translate_from_cpp(const std::string& cpp_code);

    // Generate function signature
    std::string emit_function_signature(
        const std::string& name,
        const std::vector<std::string>& params
    );

private:
    std::string emit_function_body(const std::vector<std::string>& statements);
    std::string encode_wave_pattern_as_nyte(const std::vector<double>& pattern);
};

} // namespace nikola::aria
