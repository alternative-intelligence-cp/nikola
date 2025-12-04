#include "nikola/aria/compiler.hpp"

namespace nikola::aria {

// Pimpl struct (empty for Phase 0c)
struct AriaCompiler::Impl {
    // LLVM context, module, etc. will go here in Phase 8
};

AriaCompiler::AriaCompiler()
    : pimpl_(new Impl())
{
    // Phase 0c: Stub only
}

AriaCompiler::~AriaCompiler() {
    delete pimpl_;
}

AriaCompiler::CompilationResult AriaCompiler::compile(
    const std::string& source_code,
    bool optimize,
    bool debug_info)
{
    // Phase 0c: Stub implementation
    // Phase 8: Full LLVM-based compilation
    return CompilationResult{
        .success = false,
        .machine_code = {},
        .error_message = "AriaCompiler::compile() not implemented (Phase 0c stub)",
        .warnings = {}
    };
}

std::string AriaCompiler::compile_to_ir(const std::string& source_code) {
    // Phase 0c: Stub implementation
    return "; AriaCompiler::compile_to_ir() not implemented (Phase 0c stub)\n";
}

AriaCompiler::CompilationResult AriaCompiler::compile_incremental(
    const std::string& code_fragment)
{
    // Phase 0c: Stub implementation
    return CompilationResult{
        .success = false,
        .machine_code = {},
        .error_message = "AriaCompiler::compile_incremental() not implemented (Phase 0c stub)",
        .warnings = {}
    };
}

AriaCompiler::CompilationResult AriaCompiler::link(
    const std::vector<std::string>& module_paths)
{
    // Phase 0c: Stub implementation
    return CompilationResult{
        .success = false,
        .machine_code = {},
        .error_message = "AriaCompiler::link() not implemented (Phase 0c stub)",
        .warnings = {}
    };
}

std::string AriaCompiler::get_version() const {
    return "0.0.1-phase0c-stub";
}

} // namespace nikola::aria
