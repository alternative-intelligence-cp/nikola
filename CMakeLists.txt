cmake_minimum_required(VERSION 3.20)
project(Nikola VERSION 0.1.0 LANGUAGES CXX)

# C++20 required for concepts, ranges, and modules
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build configuration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Warning flags
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Werror=return-type
)

# Find dependencies
find_package(Eigen3 3.4 REQUIRED)
find_package(Protobuf REQUIRED)
find_package(cppzmq REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated protobuf headers
)

# Protocol Buffer compilation
file(GLOB_RECURSE PROTO_FILES "${CMAKE_CURRENT_SOURCE_DIR}/proto/*.proto")
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# Core Infrastructure (Audit 1 - Finding 2.1)
set(CORE_SOURCES
    src/core/config.cpp
)

# Genesis Module 1: Cognitive Resonance
set(COGNITIVE_SOURCES
    src/cognitive/spectral_filter.cpp
    src/cognitive/attention_primer.cpp
    src/cognitive/scratchpad.cpp
)

# Genesis Module 2: Social Layer
set(SOCIAL_SOURCES
    src/social/membrane.cpp
    src/social/peer_registry.cpp
)

# Genesis Module 3: Economic Layer
set(ECONOMY_SOURCES
    src/economy/wallet.cpp
    src/economy/marketplace.cpp
)

# Genesis Module 4: Security Layer
set(SECURITY_SOURCES
    src/security/homeostasis.cpp
    src/security/polymorphic_defense.cpp
)

# Genesis Module 5: Interior Life & Metacognition
set(INTERIOR_SOURCES
    src/interior/wave_mirror.cpp
    src/interior/affective_state.cpp
    src/interior/dream_engine.cpp
    src/interior/autobiography.cpp
    src/interior/curiosity.cpp
    src/interior/internal_dialogue.cpp
)

# Genesis Module 7: Code Generation & Self-Improvement (Aria Language)
set(ARIA_SOURCES
    src/aria/compiler.cpp
    src/aria/interpreter.cpp
    src/aria/code_generator.cpp
    src/aria/metaprogramming.cpp
    src/aria/native_interface.cpp
)

# Nikola Genesis Library (stub implementations + core infrastructure)
add_library(nikola_genesis STATIC
    ${CORE_SOURCES}
    ${COGNITIVE_SOURCES}
    ${SOCIAL_SOURCES}
    ${ECONOMY_SOURCES}
    ${SECURITY_SOURCES}
    ${INTERIOR_SOURCES}
    ${ARIA_SOURCES}
    ${PROTO_SRCS}
)

target_include_directories(nikola_genesis PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(nikola_genesis PUBLIC
    Eigen3::Eigen
    protobuf::libprotobuf
    cppzmq
    Threads::Threads
)

# Test targets
add_executable(genesis_stub_test
    tests/genesis_stub_test.cpp
)

target_link_libraries(genesis_stub_test PRIVATE
    nikola_genesis
)

add_executable(config_test
    tests/config_test.cpp
)

target_link_libraries(config_test PRIVATE
    nikola_genesis
)

# Installation
install(TARGETS nikola_genesis
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/nikola
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Status message
message(STATUS "Nikola Genesis Integration - Phase 0c (Stubs)")
message(STATUS "  Module 1: Cognitive Resonance - STUB")
message(STATUS "  Module 2: Social Layer (IRSP) - STUB")
message(STATUS "  Module 3: Economic Layer (NES) - STUB")
message(STATUS "  Module 4: Security Layer (HSK) - STUB")
message(STATUS "  Module 5: Interior Life & Metacognition - STUB")
message(STATUS "  Module 7: Code Generation & Self-Improvement (Aria) - STUB")
message(STATUS "  Module 6: Subjective Experience - PENDING (documentation only)")
message(STATUS "Implementation deferred until Phase 1 (Audit 1 & 2) completion")
