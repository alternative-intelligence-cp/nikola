/**
 * @file src/autonomy/autonomy_engine_impl.cpp
 * @brief Compiles the AutonomyEngine implementation into nikola_core.
 *
 * The AutonomyEngine methods are defined inside autonomy_engine.hpp behind
 * the NIKOLA_AUTONOMY_ENGINE_IMPL guard.  This file acts as the single
 * translation unit that pulls those definitions into libnikola_core.a,
 * making them available to any code that links nikola_core.
 *
 * Before this file existed, each binary had to define the macro itself.
 * Centralising it here removes that per-binary/per-test boilerplate.
 */
#define NIKOLA_AUTONOMY_ENGINE_IMPL
#include <nikola/autonomy/autonomy_engine.hpp>
