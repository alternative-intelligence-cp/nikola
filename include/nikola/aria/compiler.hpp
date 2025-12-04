#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace nikola::aria {

/**
 * @brief AriaCompiler - Compile Aria source code to native machine code via LLVM
 *
 * Phase 0c: Stub interface only
 * Implementation: Phase 8 (58-82 hours estimated)
 *
 * Purpose:
 * - Compile Aria source to executable code
 * - Generate LLVM IR for inspection
 * - Incremental compilation for REPL
 * - Link multiple Aria modules
 *
 * Dependencies:
 * - LLVM 18+ (code generation backend)
 * - Nikola's balanced nonary libraries
 */
class AriaCompiler {
public:
    struct CompilationResult {
        bool success;
        std::vector<uint8_t> machine_code;
        std::string error_message;
        std::vector<std::string> warnings;
    };

    AriaCompiler();
    ~AriaCompiler();

    // Compile Aria source to executable code
    CompilationResult compile(const std::string& source_code,
                             bool optimize = true,
                             bool debug_info = false);

    // Compile to LLVM IR for inspection
    std::string compile_to_ir(const std::string& source_code);

    // Incremental compilation for REPL
    CompilationResult compile_incremental(const std::string& code_fragment);

    // Link multiple Aria modules
    CompilationResult link(const std::vector<std::string>& module_paths);

    // Get compiler version
    std::string get_version() const;

private:
    // Implementation details hidden in Phase 8
    struct Impl;
    Impl* pimpl_;
};

} // namespace nikola::aria
