#include "nikola/aria/compiler.hpp"

namespace nikola::aria {

// Pimpl struct (empty for Phase 0c)
struct NitpickCompiler::Impl {
    // LLVM context, module, etc. will go here in Phase 8
};

NitpickCompiler::NitpickCompiler()
    : pimpl_(new Impl())
{
    // Phase 0c: Stub only
}

NitpickCompiler::~NitpickCompiler() {
    delete pimpl_;
}

NitpickCompiler::CompilationResult NitpickCompiler::compile(
    const std::string& source_code,
    bool optimize,
    bool debug_info)
{
    // Phase 0c: Stub implementation
    // Phase 8: Full LLVM-based compilation
    return CompilationResult{
        .success = false,
        .machine_code = {},
        .error_message = "NitpickCompiler::compile() not implemented (Phase 0c stub)",
        .warnings = {}
    };
}

std::string NitpickCompiler::compile_to_ir(const std::string& source_code) {
    // Phase 0c: Stub implementation
    return "; NitpickCompiler::compile_to_ir() not implemented (Phase 0c stub)\n";
}

NitpickCompiler::CompilationResult NitpickCompiler::compile_incremental(
    const std::string& code_fragment)
{
    // Phase 0c: Stub implementation
    return CompilationResult{
        .success = false,
        .machine_code = {},
        .error_message = "NitpickCompiler::compile_incremental() not implemented (Phase 0c stub)",
        .warnings = {}
    };
}

NitpickCompiler::CompilationResult NitpickCompiler::link(
    const std::vector<std::string>& module_paths)
{
    // Phase 0c: Stub implementation
    return CompilationResult{
        .success = false,
        .machine_code = {},
        .error_message = "NitpickCompiler::link() not implemented (Phase 0c stub)",
        .warnings = {}
    };
}

std::string NitpickCompiler::get_version() const {
    return "0.0.1-phase0c-stub";
}

} // namespace nikola::aria
