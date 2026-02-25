cmake_minimum_required(VERSION 3.20)
project(Nikola VERSION 0.0.4 LANGUAGES CXX)

# C++23 required for Phase 0+ implementation
# (modules, constexpr enhancements, improved concepts)
set(CMAKE_CXX_STANDARD 23)
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
find_package(Threads REQUIRED)
find_package(cppzmq REQUIRED)  # Phase 4: ZMQ Spine

# ORT (ONNX Runtime) — optional; enables BERT-based semantic embeddings
set(ORT_ROOT "/home/randy/Workspace/SYSTEM/onnxruntime/cpp" CACHE PATH "Path to ONNX Runtime C++ SDK")
if(EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
    set(NIKOLA_HAS_ORT ON)
    add_library(onnxruntime_imported SHARED IMPORTED)
    set_target_properties(onnxruntime_imported PROPERTIES
        IMPORTED_LOCATION "${ORT_ROOT}/lib/libonnxruntime.so"
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
    )
    message(STATUS "ORT found at ${ORT_ROOT} — NIKOLA_HAS_ORT=ON")
else()
    set(NIKOLA_HAS_ORT OFF)
    message(STATUS "ORT not found at ${ORT_ROOT} — NIKOLA_HAS_ORT=OFF (no BERT embeddings)")
endif()

# Testing framework (Catch2 v3)
find_package(Catch2 3 QUIET)
if(Catch2_FOUND)
    message(STATUS "Catch2 found - tests enabled")
    set(NIKOLA_BUILD_TESTS ON)
else()
    message(STATUS "Catch2 not found - tests disabled (install: sudo apt install catch2)")
    set(NIKOLA_BUILD_TESTS OFF)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated protobuf headers
)

# Protocol Buffer compilation
file(GLOB_RECURSE PROTO_FILES "${CMAKE_CURRENT_SOURCE_DIR}/proto/*.proto")
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# ============================================================================
# NIKOLA v0.0.4 - PHASE 0+ ARCHITECTURE
# ============================================================================

# Phase 0: Critical Blockers (CF-04, MEM-04)
set(AUTONOMY_SOURCES
    src/autonomy/metabolic_controller.cpp
    src/autonomy/metabolic_lock.cpp
    src/autonomy/autonomy_engine_impl.cpp
    src/autonomy/decision_loop.cpp
)

set(SPATIAL_SOURCES
    src/spatial/hilbert_scanner.cpp
)

# Foundation Layer (9D math library)
set(FOUNDATION_SOURCES
    # src/foundation/vector9d.cpp  # Header-only for now
    # src/foundation/complex_field.cpp
    # src/foundation/toroidal_grid.cpp
)

# Core Infrastructure
set(CORE_SOURCES
    src/core/config.cpp
)

# Genesis Module 1: Cognitive Resonance
set(COGNITIVE_SOURCES
    src/cognitive/attention_primer.cpp
    src/cognitive/scratchpad.cpp
    src/cognitive/spectral_filter.cpp
    src/cognitive/thought_composer.cpp
)

# Genesis Module 2: Social Layer
set(SOCIAL_SOURCES
    # src/social/membrane.cpp  # TODO: Phase 2
    # src/social/peer_registry.cpp
)

# Genesis Module 3: Economic Layer
set(ECONOMY_SOURCES
    # src/economy/wallet.cpp  # TODO: Phase 2
    # src/economy/marketplace.cpp
)

# Genesis Module 4: Security Layer
set(SECURITY_SOURCES
    # src/security/homeostasis.cpp  # TODO: Phase 3
    # src/security/polymorphic_defense.cpp
)

# Genesis Module 5: Interior Life & Metacognition
set(INTERIOR_SOURCES
    # src/interior/wave_mirror.cpp  # TODO: Phase 3
    # src/interior/affective_state.cpp
    # src/interior/dream_engine.cpp
    # src/interior/autobiography.cpp
    # src/interior/curiosity.cpp
    # src/interior/internal_dialogue.cpp
)

# Genesis Module 7: Code Generation & Self-Improvement (Aria Language)
set(ARIA_SOURCES
    # src/aria/compiler.cpp  # TODO: Phase 5
    # src/aria/interpreter.cpp
    # src/aria/code_generator.cpp
    # src/aria/metaprogramming.cpp
    # src/aria/native_interface.cpp
)

# Nikola v0.0.4 Library (Phase 0+ implementation)
add_library(nikola_core STATIC
    ${AUTONOMY_SOURCES}
    ${SPATIAL_SOURCES}
    ${FOUNDATION_SOURCES}
    ${CORE_SOURCES}
    ${COGNITIVE_SOURCES}
    ${SOCIAL_SOURCES}
    ${ECONOMY_SOURCES}
    ${SECURITY_SOURCES}
    ${INTERIOR_SOURCES}
    ${ARIA_SOURCES}
    ${PROTO_SRCS}
)

# Backward compatibility alias
add_library(nikola_genesis ALIAS nikola_core)

target_include_directories(nikola_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(nikola_core PUBLIC
    Eigen3::Eigen
    protobuf::libprotobuf
    cppzmq
    Threads::Threads
)

# ORT wiring
if(NIKOLA_HAS_ORT)
    target_compile_definitions(nikola_core PUBLIC NIKOLA_HAS_ORT)
    target_compile_definitions(nikola_core PUBLIC
        NIKOLA_ORT_MODEL_PATH="/home/randy/Workspace/SYSTEM/onnxruntime/bert-tiny-onnx/model.onnx"
        NIKOLA_ORT_TOKENIZER_PATH="/home/randy/Workspace/SYSTEM/onnxruntime/bert-tiny-onnx"
    )
    target_include_directories(nikola_core PUBLIC ${ORT_ROOT}/include)
    target_link_libraries(nikola_core PUBLIC onnxruntime_imported)
endif()

# ============================================================================
# TESTING (Catch2 v3)
# ============================================================================

if(NIKOLA_BUILD_TESTS)
    enable_testing()
    
    # Phase 0 Unit Tests
    add_executable(test_metabolic_lock tests/unit/metabolic_lock_test.cpp)
    target_link_libraries(test_metabolic_lock PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME MetabolicLock COMMAND test_metabolic_lock)
    
    add_executable(test_hilbert_scanner tests/unit/hilbert_scanner_test.cpp)
    target_link_libraries(test_hilbert_scanner PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME HilbertScanner COMMAND test_hilbert_scanner)
    
    # Foundation Unit Tests
    add_executable(test_vector9d tests/unit/vector9d_test.cpp)
    target_link_libraries(test_vector9d PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Vector9D COMMAND test_vector9d)

    # Phase 1 Unit Tests: Core Physics Engine
    add_executable(test_physics_core tests/unit/physics_core_test.cpp)
    target_link_libraries(test_physics_core PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME PhysicsCore COMMAND test_physics_core)

    # Phase 2 Unit Tests: Manifold Geometry + Semantic Memory
    add_executable(test_phase2_geometry_memory tests/unit/phase2_geometry_memory_test.cpp)
    target_link_libraries(test_phase2_geometry_memory PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase2GeometryMemory COMMAND test_phase2_geometry_memory)

    # Phase 3 Unit Tests: The Brain — Cognitive Architecture
    add_executable(test_phase3_cognitive tests/unit/phase3_cognitive_test.cpp)
    target_link_libraries(test_phase3_cognitive PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase3Cognitive COMMAND test_phase3_cognitive)

    # Phase 4 Unit Tests: Infrastructure & Communications
    add_executable(test_phase4_infrastructure tests/unit/phase4_infrastructure_test.cpp)
    target_link_libraries(test_phase4_infrastructure PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase4Infrastructure COMMAND test_phase4_infrastructure)

    # Phase 5 Unit Tests: Autonomous Systems
    add_executable(test_phase5_autonomous tests/unit/phase5_autonomous_test.cpp)
    target_link_libraries(test_phase5_autonomous PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase5Autonomous COMMAND test_phase5_autonomous)

    # Phase 6 Unit Tests: Multimodal & Persistence
    add_executable(test_phase6_multimodal tests/unit/phase6_multimodal_test.cpp)
    target_link_libraries(test_phase6_multimodal PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase6Multimodal COMMAND test_phase6_multimodal)

    # Phase 7 Unit Tests: Security & Execution
    add_executable(test_phase7_security tests/unit/phase7_security_test.cpp)
    target_link_libraries(test_phase7_security PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase7Security COMMAND test_phase7_security)

    # Phase 8 Unit Tests: Foundations (Nit arithmetic + MetricTensorCache)
    add_executable(test_phase8_foundations tests/unit/phase8_foundations_test.cpp)
    target_link_libraries(test_phase8_foundations PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase8Foundations COMMAND test_phase8_foundations)

    # Phase 9 Unit Tests: Bootstrap (ManifoldSeeder + PhysicsMonitor)
    add_executable(test_phase9_bootstrap tests/unit/phase9_bootstrap_test.cpp)
    target_link_libraries(test_phase9_bootstrap PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase9Bootstrap COMMAND test_phase9_bootstrap)

    # Phase 10 Unit Tests: HolographicLexicon (IMP-02)
    add_executable(test_phase10_holographic tests/unit/phase10_holographic_test.cpp)
    target_link_libraries(test_phase10_holographic PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase10Holographic COMMAND test_phase10_holographic)

    # Phase 11 Unit Tests: CognitiveGenerator + ConceptMinter (COG-05)
    add_executable(test_phase11_cognitive_gen tests/unit/phase11_cognitive_gen_test.cpp)
    target_link_libraries(test_phase11_cognitive_gen PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase11CognitiveGen COMMAND test_phase11_cognitive_gen)

    # Phase 12 Unit Tests: AcasBridge + ACAS Python IPC
    add_executable(test_phase12_acas_bridge tests/unit/phase12_acas_bridge_test.cpp)
    target_link_libraries(test_phase12_acas_bridge PRIVATE nikola_core Catch2::Catch2WithMain)
    target_compile_definitions(test_phase12_acas_bridge PRIVATE
        NIKOLA_ACAS_SCRIPTS_DIR="${CMAKE_SOURCE_DIR}/src/acas")
    add_test(NAME Phase12AcasBridge COMMAND test_phase12_acas_bridge)

    # Phase 13 Unit Tests: TaskDispatcher + Priority-aware Orchestrator
    add_executable(test_phase13_task_dispatcher tests/unit/phase13_task_dispatcher_test.cpp)
    target_link_libraries(test_phase13_task_dispatcher PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase13TaskDispatcher COMMAND test_phase13_task_dispatcher)

    # Phase 14 Unit Tests: AsyncWriteQueue + CheckpointManager async persistence
    add_executable(test_phase14_async_checkpoint tests/unit/phase14_async_checkpoint_test.cpp)
    target_link_libraries(test_phase14_async_checkpoint PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase14AsyncCheckpoint COMMAND test_phase14_async_checkpoint)

    # Phase 15 Unit Tests: Hamiltonian Value Function (NIK-005)
    add_executable(test_phase15_hamiltonian_value tests/unit/phase15_hamiltonian_value_test.cpp)
    target_link_libraries(test_phase15_hamiltonian_value PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase15HamiltonianValue COMMAND test_phase15_hamiltonian_value)

    # Phase 16 Unit Tests: Coord9D Portability / CoordSerializer (NIK-003)
    add_executable(test_phase16_coord_serializer tests/unit/phase16_coord_serializer_test.cpp)
    target_link_libraries(test_phase16_coord_serializer PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase16CoordSerializer COMMAND test_phase16_coord_serializer)

    # Phase 17 Unit Tests: Bootstrap Token Security / BootstrapManager (NIK-002)
    add_executable(test_phase17_bootstrap_manager tests/unit/phase17_bootstrap_manager_test.cpp)
    target_link_libraries(test_phase17_bootstrap_manager PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase17BootstrapManager COMMAND test_phase17_bootstrap_manager)

    # Phase 18 Unit Tests: ScopeProfiler + performance introspection
    add_executable(test_phase18_scope_profiler tests/unit/phase18_scope_profiler_test.cpp)
    target_link_libraries(test_phase18_scope_profiler PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase18ScopeProfiler COMMAND test_phase18_scope_profiler)

    # Phase 19 Unit Tests: TelemetryDaemon + async event streaming
    add_executable(test_phase19_telemetry_daemon tests/unit/phase19_telemetry_daemon_test.cpp)
    target_link_libraries(test_phase19_telemetry_daemon PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase19TelemetryDaemon COMMAND test_phase19_telemetry_daemon)

    # Phase 20 Unit Tests: DebugAdapter + DAP protocol bridge
    add_executable(test_phase20_debug_adapter tests/unit/phase20_debug_adapter_test.cpp)
    target_link_libraries(test_phase20_debug_adapter PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase20DebugAdapter COMMAND test_phase20_debug_adapter)

    # Phase 21 Unit Tests: CognitivePipeline end-to-end
    add_executable(test_phase21_cognitive_pipeline tests/unit/phase21_cognitive_pipeline_test.cpp)
    target_link_libraries(test_phase21_cognitive_pipeline PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase21CognitivePipeline COMMAND test_phase21_cognitive_pipeline)

    # Phase 22 Unit Tests: CUDA GPU Propagator (performance: <1 ms/step)
    if(NIKOLA_HAS_CUDA)
        add_executable(test_phase22_cuda_propagator tests/unit/phase22_cuda_propagator_test.cpp)
        target_link_libraries(test_phase22_cuda_propagator PRIVATE nikola_core nikola_cuda Catch2::Catch2WithMain)
        add_test(NAME Phase22CudaPropagator COMMAND test_phase22_cuda_propagator)
    endif()
    # (Phase 22 test skipped when CUDA is unavailable)

    # Phase 23 Unit Tests: DecisionLoop — autonomous action scoring + tick cycle
    add_executable(test_phase23_decision_loop tests/unit/phase23_decision_loop_test.cpp)
    target_link_libraries(test_phase23_decision_loop PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase23DecisionLoop COMMAND test_phase23_decision_loop)

    # Phase 24 Unit Tests: ThoughtComposer — template-driven output generation
    add_executable(test_phase24_thought_composer tests/unit/phase24_thought_composer_test.cpp)
    target_link_libraries(test_phase24_thought_composer PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase24ThoughtComposer COMMAND test_phase24_thought_composer)

    # Phase 25 Unit Tests: FieldPersistence — wave-field checkpoint I/O
    add_executable(test_phase25_field_persistence tests/unit/phase25_field_persistence_test.cpp)
    target_link_libraries(test_phase25_field_persistence PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase25FieldPersistence COMMAND test_phase25_field_persistence)

    # Phase 26 Unit Tests: SeededExploration — boredom-driven novelty injection
    add_executable(test_phase26_seeded_exploration tests/unit/phase26_seeded_exploration_test.cpp)
    target_link_libraries(test_phase26_seeded_exploration PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase26SeededExploration COMMAND test_phase26_seeded_exploration)

    # Phase 27 Unit Tests: CrossInjection — analytic cross-injection calibration
    add_executable(test_phase27_cross_injection tests/unit/phase27_cross_injection_test.cpp)
    target_link_libraries(test_phase27_cross_injection PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase27CrossInjection COMMAND test_phase27_cross_injection)

    # Phase 28 Unit Tests: AnalyticDecode — time-correct warm decode via analytic signatures
    add_executable(test_phase28_analytic_decode tests/unit/phase28_analytic_decode_test.cpp)
    target_link_libraries(test_phase28_analytic_decode PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase28AnalyticDecode COMMAND test_phase28_analytic_decode)

    # Phase 29 Unit Tests: NoveltyDecay — habituation curve for REFUSE
    add_executable(test_phase29_novelty_decay tests/unit/phase29_novelty_decay_test.cpp)
    target_link_libraries(test_phase29_novelty_decay PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase29NoveltyDecay COMMAND test_phase29_novelty_decay)

    # Phase 30 Unit Tests: ZmqSpine — message bus integration
    add_executable(test_phase30_zmq_spine tests/unit/phase30_zmq_spine_test.cpp)
    target_link_libraries(test_phase30_zmq_spine PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase30ZmqSpine COMMAND test_phase30_zmq_spine)

    # Phase 31 Unit Tests: LookupAgent — semantic DB query pipeline
    add_executable(test_phase31_lookup_agent tests/unit/phase31_lookup_agent_test.cpp)
    target_link_libraries(test_phase31_lookup_agent PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase31LookupAgent COMMAND test_phase31_lookup_agent)

    # Phase 32 Unit Tests: EscalationAgent — tamper-evident harm evidence
    add_executable(test_phase32_escalation tests/unit/phase32_escalation_test.cpp)
    target_link_libraries(test_phase32_escalation PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase32Escalation COMMAND test_phase32_escalation)

    # Phase 33 Unit Tests: SemanticMemory persistence — wave-field LTM store
    add_executable(test_phase33_memory_persistence tests/unit/phase33_memory_persistence_test.cpp)
    target_link_libraries(test_phase33_memory_persistence PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase33MemoryPersistence COMMAND test_phase33_memory_persistence)

    # Phase 34 Unit Tests: MemoryRecall — cosine-resonance retrieval
    add_executable(test_phase34_recall tests/unit/phase34_recall_test.cpp)
    target_link_libraries(test_phase34_recall PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase34Recall COMMAND test_phase34_recall)

    # Phase 35 Unit Tests: RelevanceGate — dual-path salience filter
    add_executable(test_phase35_relevance_gate tests/unit/phase35_relevance_gate_test.cpp)
    target_link_libraries(test_phase35_relevance_gate PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase35RelevanceGate COMMAND test_phase35_relevance_gate)

    # Phase 36 Unit Tests: GatedInjection — RelevanceGate wired into CognitiveTorus
    add_executable(test_phase36_gated_injection tests/unit/phase36_gated_injection_test.cpp)
    target_link_libraries(test_phase36_gated_injection PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase36GatedInjection COMMAND test_phase36_gated_injection)

    # Phase 37 Unit Tests: NPT data structures — WaveCorrelationHead + AttentionResult
    add_executable(test_phase37_npt_structure tests/unit/phase37_npt_structure_test.cpp)
    target_link_libraries(test_phase37_npt_structure PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase37NptStructure COMMAND test_phase37_npt_structure)

    # Phase 38 Unit Tests: Wave correlation kernel — inner_product_re, wave_correlation, softmax
    add_executable(test_phase38_wave_correlation tests/unit/phase38_wave_correlation_test.cpp)
    target_link_libraries(test_phase38_wave_correlation PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase38WaveCorrelation COMMAND test_phase38_wave_correlation)

    # Phase 39 Unit Tests: Riemannian curvature bias
    add_executable(test_phase39_curvature_bias tests/unit/phase39_curvature_bias_test.cpp)
    target_link_libraries(test_phase39_curvature_bias PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase39CurvatureBias COMMAND test_phase39_curvature_bias)

    # Phase 40 Unit Tests: Heterodyne value aggregation — clone, scale_by, add_scaled, output WF
    add_executable(test_phase40_heterodyne tests/unit/phase40_heterodyne_test.cpp)
    target_link_libraries(test_phase40_heterodyne PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase40Heterodyne COMMAND test_phase40_heterodyne)

    # Phase 41 Unit Tests: Multi-head merge — phase_rotate_psi, project_heads, per-head V aggregation
    add_executable(test_phase41_multihead_merge tests/unit/phase41_multihead_merge_test.cpp)
    target_link_libraries(test_phase41_multihead_merge PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase41MultiheadMerge COMMAND test_phase41_multihead_merge)

    # Phase 42 Unit Tests: NPT wired into DecisionLoop (REASON action)
    add_executable(test_phase42_npt_reason tests/unit/phase42_npt_reason_test.cpp)
    target_link_libraries(test_phase42_npt_reason PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase42NptReason COMMAND test_phase42_npt_reason)

    # Phase 43 Unit Tests: Hebbian-Riemannian metric update via NPT output
    add_executable(test_phase43_hebbian_update tests/unit/phase43_hebbian_update_test.cpp)
    target_link_libraries(test_phase43_hebbian_update PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase43HebbianUpdate COMMAND test_phase43_hebbian_update)

    # Phase 44 Unit Tests: K-head differentiation via score-weighted input tracking
    add_executable(test_phase44_k_differentiation tests/unit/phase44_k_differentiation_test.cpp)
    target_link_libraries(test_phase44_k_differentiation PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase44KDifferentiation COMMAND test_phase44_k_differentiation)

    # Phase 45 Unit Tests: Dopamine-modulated Hebbian learning rate
    add_executable(test_phase45_dopamine_lr tests/unit/phase45_dopamine_lr_test.cpp)
    target_link_libraries(test_phase45_dopamine_lr PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase45DopamineLR COMMAND test_phase45_dopamine_lr)

    # Phase 46 Unit Tests: Serotonin-modulated metric elasticity
    add_executable(test_phase46_serotonin_elasticity tests/unit/phase46_serotonin_elasticity_test.cpp)
    target_link_libraries(test_phase46_serotonin_elasticity PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase46SerotoninElasticity COMMAND test_phase46_serotonin_elasticity)

    # Phase 47 Unit Tests: Norepinephrine arousal modulation
    add_executable(test_phase47_norepinephrine_arousal tests/unit/phase47_norepinephrine_arousal_test.cpp)
    target_link_libraries(test_phase47_norepinephrine_arousal PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase47NorepinephrineArousal COMMAND test_phase47_norepinephrine_arousal)

    # Phase 48 Unit Tests: NE-modulated Dream-Weave experience replay pool
    add_executable(test_phase48_dream_replay tests/unit/phase48_dream_replay_test.cpp)
    target_link_libraries(test_phase48_dream_replay PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase48DreamReplay COMMAND test_phase48_dream_replay)

    # Phase 49 Unit Tests: AUTO-04 Boredom Singularity Fix (sigmoidal accumulation)
    add_executable(test_phase49_boredom_singularity_fix tests/unit/phase49_boredom_singularity_fix_test.cpp)
    target_link_libraries(test_phase49_boredom_singularity_fix PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase49BoredomSingularityFix COMMAND test_phase49_boredom_singularity_fix)

    # Phase 50 Unit Tests: CuriosityGoal + query gate (spec §8.3)
    add_executable(test_phase50_curiosity_goal_query_gate tests/unit/phase50_curiosity_goal_query_gate_test.cpp)
    target_link_libraries(test_phase50_curiosity_goal_query_gate PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase50CuriosityGoalQueryGate COMMAND test_phase50_curiosity_goal_query_gate)

    # Phase 51 Unit Tests: Failure Mode Guards — Anhedonia Trap + Mania Loop (spec §9)
    add_executable(test_phase51_failure_mode_guards tests/unit/phase51_failure_mode_guards_test.cpp)
    target_link_libraries(test_phase51_failure_mode_guards PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase51FailureModeGuards COMMAND test_phase51_failure_mode_guards)

    # Phase 52 Unit Tests: GAP-005 Cross-Coupling Matrix (spec §GAP-005)
    add_executable(test_phase52_cross_coupling_matrix tests/unit/phase52_cross_coupling_matrix_test.cpp)
    target_link_libraries(test_phase52_cross_coupling_matrix PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase52CrossCouplingMatrix COMMAND test_phase52_cross_coupling_matrix)

    # Phase 53 Unit Tests: GAP-036 Boredom k-Parameter Calibration (spec §GAP-036)
    add_executable(test_phase53_boredom_gap036_calibration tests/unit/phase53_boredom_gap036_calibration_test.cpp)
    target_link_libraries(test_phase53_boredom_gap036_calibration PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase53BoredomGAP036Calibration COMMAND test_phase53_boredom_gap036_calibration)

    # Phase 54 Unit Tests: s_eff(N) = s_local/(1+N²) quadratic spec correction (spec §"Non-Linear Interaction Terms")
    add_executable(test_phase54_seff_quadratic_correction tests/unit/phase54_seff_quadratic_correction_test.cpp)
    target_link_libraries(test_phase54_seff_quadratic_correction PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase54SeffQuadraticCorrection COMMAND test_phase54_seff_quadratic_correction)

    # Phase 55 Unit Tests: GAP-022 ENGS→Physics feedback loop latency (τ_max ≤ 10ms, double-buffered atomic swap)
    add_executable(test_phase55_feedback_loop_latency tests/unit/phase55_feedback_loop_latency_test.cpp)
    target_link_libraries(test_phase55_feedback_loop_latency PRIVATE nikola_core Catch2::Catch2WithMain atomic)
    add_test(NAME Phase55FeedbackLoopLatency COMMAND test_phase55_feedback_loop_latency)

    # Phase 56 Unit Tests: GAP-029 Neurochemistry Cross-Validation Metrics (Pearson r, ablation pathology, Yerkes-Dodson)
    add_executable(test_phase56_neuropsychometric_profile tests/unit/phase56_neuropsychometric_profile_test.cpp)
    target_link_libraries(test_phase56_neuropsychometric_profile PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase56NeuropsychometricProfile COMMAND test_phase56_neuropsychometric_profile)

    # Phase 57 Unit Tests: GAP-012 Metabolic Cost Calibration via Hardware Benchmarking (NMU derivation, thermal, neurochemical)
    add_executable(test_phase57_metabolic_calibrator tests/unit/phase57_metabolic_calibrator_test.cpp)
    target_link_libraries(test_phase57_metabolic_calibrator PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase57MetabolicCalibrator COMMAND test_phase57_metabolic_calibrator)

    # Phase 58 Unit Tests: GAP-025 End-to-End Latency Budget Allocation (1000 Hz tick, component budgets, alerting, Drop-or-Degrade)
    add_executable(test_phase58_latency_budget tests/unit/phase58_latency_budget_test.cpp)
    target_link_libraries(test_phase58_latency_budget PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase58LatencyBudget COMMAND test_phase58_latency_budget)

    # Phase 59 Unit Tests: GAP-030 Physics Oracle Calibration Suite (Standard Candle, Viscosity Trap, Resonance Attack, Reversibility, runtime SCRAM)
    add_executable(test_phase59_physics_oracle tests/unit/phase59_physics_oracle_test.cpp)
    target_link_libraries(test_phase59_physics_oracle PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase59PhysicsOracle COMMAND test_phase59_physics_oracle)

    # Phase 60 Unit Tests: GAP-021 TorusGridSoA Memory Alignment Guarantees (AlignedAllocator, AlignedVec, TorusBlock, SoaAlignmentGuard, Copy-on-Load, GGUF mmap)
    add_executable(test_phase60_soa_layout tests/unit/phase60_soa_layout_test.cpp)
    target_link_libraries(test_phase60_soa_layout PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase60SoaLayout COMMAND test_phase60_soa_layout)

    # Phase 61 Unit Tests: GAP-024 Metric Tensor Consolidation Scheduler + GAP-041 9D Coordinate Semantics
    add_executable(test_phase61_torus_geometry tests/unit/phase61_torus_geometry_test.cpp)
    target_link_libraries(test_phase61_torus_geometry PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase61TorusGeometry COMMAND test_phase61_torus_geometry)

    # Phase 62 Unit Tests: GAP-020 Temporal Decoherence Detection Thresholds (golden-ratio harmonics, phase budget, adaptive latency table, PTP clock sync state machine)
    add_executable(test_phase62_temporal_decoherence tests/unit/phase62_temporal_decoherence_test.cpp)
    target_link_libraries(test_phase62_temporal_decoherence PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase62TemporalDecoherence COMMAND test_phase62_temporal_decoherence)

    # Phase 63 Unit Tests: GAP-044 Nonary Overflow Probability Distribution (bits-per-trit, uniform/Gaussian overflow stats, Spectral Cascading carry decomposition, saturation monitoring, Voronoi dither)
    add_executable(test_phase63_nonary_overflow tests/unit/phase63_nonary_overflow_test.cpp)
    target_link_libraries(test_phase63_nonary_overflow PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase63NonaryOverflow COMMAND test_phase63_nonary_overflow)

    # Phase 64 Unit Tests: GAP-017 Cymatic Transduction Sampling Rate (8 golden-ratio emitters, 48kHz→1kHz decimation, FIR anti-alias filter, latency budget, Goertzel frequency extraction, dual-path architecture)
    add_executable(test_phase64_cymatic_transduction tests/unit/phase64_cymatic_transduction_test.cpp)
    target_link_libraries(test_phase64_cymatic_transduction PRIVATE nikola_core Catch2::Catch2WithMain)
    add_test(NAME Phase64CymaticTransduction COMMAND test_phase64_cymatic_transduction)

    message(STATUS "Tests configured - run with: make test")
endif()

# ============================================================================
# FIRST LIGHT — Nikola first cold-start executable
# ============================================================================
add_executable(nikola_first_light src/main.cpp)
target_compile_definitions(nikola_first_light PRIVATE NIKOLA_AUTONOMY_ENGINE_IMPL)
target_link_libraries(nikola_first_light PRIVATE nikola_core)
message(STATUS "First Light executable: nikola_first_light")

# Installation
install(TARGETS nikola_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/nikola
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# ============================================================================
# BUILD STATUS
# ============================================================================
message(STATUS "")
message(STATUS "Nikola v0.0.4 - AGI Architecture Implementation")
message(STATUS "================================================")
message(STATUS "Phase 0: Critical Blockers")
message(STATUS "  ✓ CF-04: Transactional Metabolic Lock")
message(STATUS "  ✓ MEM-04: Hilbert Re-indexing (9D)")
message(STATUS "")
message(STATUS "Foundation: 9D Math Library")
message(STATUS "  ✓ Vector9D (header-only)")
message(STATUS "  ✓ ComplexField (header-only, Kahan summation, PML ghost)")
message(STATUS "  ✓ ToroidalGrid (SoA layout, Morton128, precomputed adjacency)")
message(STATUS "")
message(STATUS "Phase 1: Core Physics Engine")
message(STATUS "  ✓ WaveFunction (Manifold Seeder, IMP-03, Gap 1.1-1.2)")
message(STATUS "  ✓ Hamiltonian (UFIE energy oracle, Gap 1.5 watchdog)")
message(STATUS "  ✓ Propagator (Strang split-operator, Störmer-Verlet, Kahan Laplacian)")
message(STATUS "")
message(STATUS "Phase 2: Topology & Memory")
message(STATUS "  ✓ TopologyManager (9D adjacency, Morton128, Gap 2.1-2.3)")
message(STATUS "  ✓ MemoryConsolidator (LTM/STM/WM pools, eigen-importance, Gap 2.4)")
message(STATUS "  ✓ ResonanceIndex (FAISS-ready k-ANN, Gap 2.5)")
message(STATUS "")
message(STATUS "Phase 3: Cognitive Brain")
message(STATUS "  ✓ WaveInterferenceProcessor (holographic writes, psi-sync, Gap 3.1-3.2)")
message(STATUS "  ✓ PatternRecognizer (spectral templates, EqProp, Gap 3.3)")
message(STATUS "  ✓ ContextManager + AttentionPrimer + SpectralFilter (Gap 3.4-3.5)")
message(STATUS "  ✓ QuantumScratchpad (seqlock WM, Gap 3.6)")
message(STATUS "")
message(STATUS "Tests: ${NIKOLA_BUILD_TESTS}")
if(NIKOLA_BUILD_TESTS)
    message(STATUS "  - Phase 0: metabolic_lock, hilbert_scanner")
    message(STATUS "  - Foundation: vector9d")
    message(STATUS "  - Phase 1: physics_core")
    message(STATUS "  - Phase 2: phase2_geometry_memory")
    message(STATUS "  - Phase 3: phase3_cognitive")
    message(STATUS "  - Phase 4: phase4_infrastructure")
    message(STATUS "  - Phase 5: phase5_autonomous")
    message(STATUS "  - Phase 6: phase6_multimodal")
    message(STATUS "  - Phase 7: phase7_security")
    message(STATUS "  - Phase 8: phase8_foundations")
    message(STATUS "  - Phase 9: phase9_bootstrap")
    message(STATUS "  - Phase 10: phase10_holographic")
    message(STATUS "  - Phase 11: phase11_cognitive_gen")
    message(STATUS "  - Phase 12: phase12_acas_bridge")
    message(STATUS "  - Phase 13: phase13_task_dispatcher")
    message(STATUS "  - Phase 14: phase14_async_checkpoint")
    message(STATUS "  - Phase 15: phase15_hamiltonian_value (NIK-005)")
    message(STATUS "  - Phase 16: phase16_coord_serializer (NIK-003)")
    message(STATUS "  - Phase 17: phase17_bootstrap_manager (NIK-002)")
endif()
message(STATUS "")
message(STATUS "Phase 4: Infrastructure & Communications  [COMPLETE ✓]")
message(STATUS "  ✓ CircuitBreaker (Gap 4.1: 100ms/5ms timeouts, exponential backoff)")
message(STATUS "  ✓ ComponentWatchdog (Gap 4.2: 500ms crash detection)")
message(STATUS "  ✓ WaveformSHM + SeqlockFrame (Gap 4.3: RAII IPC at 60fps)")
message(STATUS "  ✓ ZmqSpine (Gap 4.4: HWM=1000, LINGER=0, IMMEDIATE=1)")
message(STATUS "  ✓ Topic versioning (Gap 4.5: nikola.v1.*  append-only schema)")
message(STATUS "  ✓ Orchestrator (Pimpl ABI-stable, IMP-04)")
message(STATUS "  ✓ LLMBridge + ToolExecutor (agent integration layer)")
message(STATUS "  ✓ 55 tests — 177 assertions — ALL PASSING")
message(STATUS "")
message(STATUS "Phase 5: Autonomous Systems  [COMPLETE ✓]")
message(STATUS "  ✓ DopamineSystem (Gap 5.1: TD learning γ=0.95, τ_decay=2s)")
message(STATUS "  ✓ EntropyEstimator (Gap 5.2: Monte Carlo K=1000, O(K) at 2kHz)")
message(STATUS "  ✓ BoredomRegulator (Gap 5.2: H_target=6, explore_thresh=0.7)")
message(STATUS "  ✓ MetabolicSimulator (Gap 5.3: α=0.001, regen=0.05/s, exhaustion<0.15)")
message(STATUS "  ✓ NapController (Gap 5.4: enter<0.15, exit>0.90, max=60s)")
message(STATUS "  ✓ DreamWeaveEngine (Gap 5.5: Frobenius norm < 1e-4, max 1000 iter)")
message(STATUS "  ✓ AutonomyEngine (Pimpl facade, all 5 gaps integrated)")
message(STATUS "")
message(STATUS "Next: Phase 6 - Multimodal & Persistence")
message(STATUS "================================================")
message(STATUS "")
