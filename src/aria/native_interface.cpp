#include "nikola/aria/native_interface.hpp"
#include <stdexcept>

namespace nikola::aria {

AriaNativeInterface::AriaNativeInterface()
    : interpreter_(new AriaInterpreter()),
      compiler_(new AriaCompiler())
{
    // Phase 0c: Stub only
}

AriaNativeInterface::~AriaNativeInterface() {
    // Phase 0c: Stub only
}

AriaInterpreter::Value AriaNativeInterface::call_aria_function_generic(
    const std::string& function_name,
    const std::vector<AriaInterpreter::Value>& args)
{
    // Phase 0c: Stub implementation
    throw std::runtime_error(
        "AriaNativeInterface::call_aria_function_generic() not implemented (Phase 0c stub)"
    );
}

void AriaNativeInterface::share_torus_manifold(TorusManifold* torus) {
    // Phase 0c: Store pointer for stub purposes
    shared_memory_regions_["torus_manifold"] = static_cast<void*>(torus);
}

void AriaNativeInterface::share_wave_processor(WaveProcessor* processor) {
    // Phase 0c: Store pointer for stub purposes
    shared_memory_regions_["wave_processor"] = static_cast<void*>(processor);
}

void* AriaNativeInterface::get_shared_memory(const std::string& name) {
    // Phase 0c: Retrieve pointer
    auto it = shared_memory_regions_.find(name);
    if (it != shared_memory_regions_.end()) {
        return it->second;
    }
    return nullptr;
}

AriaInterpreter::Value AriaNativeInterface::to_aria_value(
    const BalancedNonary& value)
{
    // Phase 0c: Stub implementation
    AriaInterpreter::Value v;
    v.type = AriaInterpreter::Value::Type::NIT;
    v.data = static_cast<int8_t>(value);
    return v;
}

AriaInterpreter::Value AriaNativeInterface::to_aria_value(int8_t nit) {
    // Phase 0c: Stub implementation
    AriaInterpreter::Value v;
    v.type = AriaInterpreter::Value::Type::NIT;
    v.data = nit;
    return v;
}

AriaInterpreter::Value AriaNativeInterface::to_aria_value(uint16_t nyte_or_tryte) {
    // Phase 0c: Stub implementation
    AriaInterpreter::Value v;
    v.type = AriaInterpreter::Value::Type::NYTE;
    v.data = nyte_or_tryte;
    return v;
}

AriaInterpreter::Value AriaNativeInterface::to_aria_value(int64_t value) {
    // Phase 0c: Stub implementation
    AriaInterpreter::Value v;
    v.type = AriaInterpreter::Value::Type::INT64;
    v.data = value;
    return v;
}

AriaInterpreter::Value AriaNativeInterface::to_aria_value(double value) {
    // Phase 0c: Stub implementation
    AriaInterpreter::Value v;
    v.type = AriaInterpreter::Value::Type::FLOAT64;
    v.data = value;
    return v;
}

AriaInterpreter::Value AriaNativeInterface::to_aria_value(const std::string& value) {
    // Phase 0c: Stub implementation
    AriaInterpreter::Value v;
    v.type = AriaInterpreter::Value::Type::STRING;
    v.data = value;
    return v;
}

BalancedNonary AriaNativeInterface::from_aria_nit(const AriaInterpreter::Value& value) {
    // Phase 0c: Stub implementation
    if (value.type == AriaInterpreter::Value::Type::NIT) {
        return static_cast<BalancedNonary>(std::get<int8_t>(value.data));
    }
    throw std::runtime_error("Value is not NIT type");
}

uint16_t AriaNativeInterface::from_aria_nyte(const AriaInterpreter::Value& value) {
    // Phase 0c: Stub implementation
    if (value.type == AriaInterpreter::Value::Type::NYTE) {
        return std::get<uint16_t>(value.data);
    }
    throw std::runtime_error("Value is not NYTE type");
}

int64_t AriaNativeInterface::from_aria_int64(const AriaInterpreter::Value& value) {
    // Phase 0c: Stub implementation
    if (value.type == AriaInterpreter::Value::Type::INT64) {
        return std::get<int64_t>(value.data);
    }
    throw std::runtime_error("Value is not INT64 type");
}

double AriaNativeInterface::from_aria_float64(const AriaInterpreter::Value& value) {
    // Phase 0c: Stub implementation
    if (value.type == AriaInterpreter::Value::Type::FLOAT64) {
        return std::get<double>(value.data);
    }
    throw std::runtime_error("Value is not FLOAT64 type");
}

std::string AriaNativeInterface::from_aria_string(const AriaInterpreter::Value& value) {
    // Phase 0c: Stub implementation
    if (value.type == AriaInterpreter::Value::Type::STRING) {
        return std::get<std::string>(value.data);
    }
    throw std::runtime_error("Value is not STRING type");
}

void AriaNativeInterface::expose_cognitive_api() {
    // Phase 0c: Stub implementation
    // Phase 8: Register C++ functions for WaveMirror, AffectiveState, etc.
}

void AriaNativeInterface::expose_wave_api() {
    // Phase 0c: Stub implementation
    // Phase 8: Register wave propagation functions
}

void AriaNativeInterface::expose_interior_life_api() {
    // Phase 0c: Stub implementation
    // Phase 8: Register interior life components
}

void AriaNativeInterface::load_aria_module(const std::string& module_path) {
    // Phase 0c: Stub implementation
    throw std::runtime_error(
        "AriaNativeInterface::load_aria_module() not implemented (Phase 0c stub)"
    );
}

bool AriaNativeInterface::is_module_loaded(const std::string& module_name) const {
    // Phase 0c: Stub implementation
    return loaded_functions_.find(module_name) != loaded_functions_.end();
}

} // namespace nikola::aria
