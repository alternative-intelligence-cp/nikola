#pragma once

#include <string>
#include <map>
#include <functional>
#include <memory>

#include "interpreter.hpp"
#include "compiler.hpp"

// Forward declarations
namespace nikola {
    class TorusManifold;
    class WaveProcessor;
    enum class BalancedNonary : int8_t;
}

namespace nikola::aria {

/**
 * @brief AriaNativeInterface - Bidirectional bridge between C++ Nikola and Aria code
 *
 * Phase 0c: Stub interface only
 * Implementation: Phase 8 (58-82 hours estimated)
 *
 * Purpose:
 * - Call Aria functions from C++
 * - Register C++ functions for Aria to call
 * - Share memory between C++ and Aria (zero-copy)
 * - Convert between C++ and Aria types
 * - Expose Nikola's cognitive API to Aria
 *
 * Integration Points:
 * - All Nikola modules: Can call Aria functions
 * - Aria code: Can directly manipulate wave patterns
 * - Performance: Zero-copy sharing of large data structures
 */
class AriaNativeInterface {
public:
    AriaNativeInterface();
    ~AriaNativeInterface();

    // Call Aria function from C++ (template for type safety)
    template<typename ReturnType, typename... Args>
    ReturnType call_aria_function(const std::string& function_name, Args... args);

    // Generic version returning Value
    AriaInterpreter::Value call_aria_function_generic(
        const std::string& function_name,
        const std::vector<AriaInterpreter::Value>& args
    );

    // Register C++ function for Aria to call
    template<typename ReturnType, typename... Args>
    void register_cpp_function(const std::string& name,
                              std::function<ReturnType(Args...)> fn);

    // Share memory between C++ and Aria
    void share_torus_manifold(TorusManifold* torus);
    void share_wave_processor(WaveProcessor* processor);
    void* get_shared_memory(const std::string& name);

    // Convert between C++ and Aria types
    AriaInterpreter::Value to_aria_value(const BalancedNonary& value);
    AriaInterpreter::Value to_aria_value(int8_t nit);
    AriaInterpreter::Value to_aria_value(uint16_t nyte_or_tryte);
    AriaInterpreter::Value to_aria_value(int64_t value);
    AriaInterpreter::Value to_aria_value(double value);
    AriaInterpreter::Value to_aria_value(const std::string& value);

    BalancedNonary from_aria_nit(const AriaInterpreter::Value& value);
    uint16_t from_aria_nyte(const AriaInterpreter::Value& value);
    int64_t from_aria_int64(const AriaInterpreter::Value& value);
    double from_aria_float64(const AriaInterpreter::Value& value);
    std::string from_aria_string(const AriaInterpreter::Value& value);

    // Expose Nikola's cognitive API to Aria
    void expose_cognitive_api();
    void expose_wave_api();
    void expose_interior_life_api();

    // Load and cache compiled Aria modules
    void load_aria_module(const std::string& module_path);
    bool is_module_loaded(const std::string& module_name) const;

private:
    std::map<std::string, void*> shared_memory_regions_;
    std::map<std::string, void*> loaded_functions_;
    std::unique_ptr<AriaInterpreter> interpreter_;
    std::unique_ptr<AriaCompiler> compiler_;
};

// Template implementations
template<typename ReturnType, typename... Args>
ReturnType AriaNativeInterface::call_aria_function(
    const std::string& function_name,
    Args... args)
{
    // Phase 8 implementation
    // For now, return default value
    return ReturnType{};
}

template<typename ReturnType, typename... Args>
void AriaNativeInterface::register_cpp_function(
    const std::string& name,
    std::function<ReturnType(Args...)> fn)
{
    // Phase 8 implementation
    // For now, do nothing
}

} // namespace nikola::aria
