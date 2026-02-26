// SPDX-License-Identifier: MIT
// GAP-023: Protocol Buffer Schema Evolution Strategy
// Phase 91 — nikola::system
//
// Encodes naming conventions, versioning rules and compatibility policy for
// all .proto schemas used on the Nikola ZeroMQ spine.  The values here are
// the machine-checkable distillation of the evolution rules specified in the
// integration document; actual schema files are in the proto/ directory.
//
// Core principle: Field IDs are immutable once assigned.  Breaking changes
// require a MAJOR version bump and a translation shim.
//
// Source: 01_zeromq_spine.md §"Schema Evolution and Lifecycle Management"

#pragma once

#include <cstdint>
#include <string_view>

namespace nikola::system {

// ─── Semantic versioning components ──────────────────────────────────────────

/// Current MAJOR version of the NeuralSpike wire schema.
/// A bump here requires a translation layer shim (v→v+1).
inline constexpr uint32_t SCHEMA_VERSION_MAJOR  = 2;

/// Current MINOR version (backward-compatible additions only).
inline constexpr uint32_t SCHEMA_VERSION_MINOR  = 0;

/// Current PATCH version (doc / comment changes; no wire impact).
inline constexpr uint32_t SCHEMA_VERSION_PATCH  = 0;

// ─── Package namespace prefix ────────────────────────────────────────────────

/// Package namespace root used in all .proto files.
/// Full package for MAJOR version v: "nikola.spine.vN"
inline constexpr std::string_view SCHEMA_PACKAGE_ROOT = "nikola.spine";

// ─── Compatibility categories ─────────────────────────────────────────────────

/// Classification of a schema change based on its wire-level impact.
enum class SchemaChangeClass : uint8_t {
    BREAKING          = 0,  ///< MAJOR bump required; needs shim
    BACKWARD_COMPAT   = 1,  ///< MINOR bump; new optional fields
    NON_FUNCTIONAL    = 2   ///< PATCH; comments / whitespace only
};

// ─── Field lifecycle states ──────────────────────────────────────────────────

/// Lifecycle phase of a field within a proto message.
enum class FieldLifecycle : uint8_t {
    ACTIVE       = 0,  ///< In use; included in all new messages
    DEPRECATED   = 1,  ///< Marked deprecated = true; rename to OBSOLETE_<name>
    TOMBSTONED   = 2   ///< Removed from generated code; ID reserved forever
};

// ─── Compatibility expectations (used in matrix assertions) ──────────────────

/// Result expected when a producer of version P sends to a consumer at version C.
enum class CompatibilityResult : uint8_t {
    SUCCESS_FULL        = 0,  ///< All fields accessible; perfect fidelity
    SUCCESS_FORWARD     = 1,  ///< New fields missing → default values used
    SUCCESS_BACKWARD    = 2,  ///< Unknown new fields silently dropped / passed through
    FAILURE_MAJOR       = 3   ///< Incompatible MAJOR; translation required
};

// ─── Required header fields ───────────────────────────────────────────────────

/// Number of fields in the NeuralSpike envelope that are logically required
/// (enforced by SecureChannel, not by proto3 schema).
inline constexpr uint8_t NEURAL_SPIKE_REQUIRED_HEADER_FIELDS = 2;
/// Field name: unique per-message identifier.
inline constexpr std::string_view REQUIRED_FIELD_REQUEST_ID  = "request_id";
/// Field name: nanosecond-precision send timestamp.
inline constexpr std::string_view REQUIRED_FIELD_TIMESTAMP   = "timestamp";

// ─── Morton key migration constants (INT-06 case study) ──────────────────────

/// Bytes in each Morton key entry in the new schema (128 bits, Big Endian).
inline constexpr uint8_t MORTON_KEY_BYTES_V2                 = 16;

/// Deprecated proto field tag for the old split int32 coordinate array.
inline constexpr uint32_t OBSOLETE_COORDINATES_FIELD_TAG     = 1;

/// Active proto field tag for the new morton_indices bytes array.
inline constexpr uint32_t MORTON_INDICES_FIELD_TAG           = 5;

// ─── Policy predicates ───────────────────────────────────────────────────────

/// True when two MAJOR versions are directly compatible without a shim.
[[nodiscard]] constexpr bool compatible_without_shim(uint32_t major_a, uint32_t major_b) noexcept {
    return major_a == major_b;
}

/// True when the given change class requires a MAJOR version bump.
[[nodiscard]] constexpr bool requires_major_bump(SchemaChangeClass c) noexcept {
    return c == SchemaChangeClass::BREAKING;
}

/// True when a field ID should be added to the "reserved" list after tombstoning.
/// (Always true — field IDs must never be reused.)
[[nodiscard]] constexpr bool must_reserve_tombstoned_id(uint32_t /*field_id*/) noexcept {
    return true;
}

/// Determine the compatibility result for a (producer_major, consumer_major) pair.
[[nodiscard]] constexpr CompatibilityResult
compatibility(uint32_t producer_major, uint32_t consumer_major) noexcept {
    if (producer_major == consumer_major) return CompatibilityResult::SUCCESS_FULL;
    if (producer_major  < consumer_major) return CompatibilityResult::SUCCESS_FORWARD;   // legacy prod → new cons
    if (producer_major == consumer_major + 1) return CompatibilityResult::SUCCESS_BACKWARD; // newer prod → older cons (unknown fields buffered)
    return CompatibilityResult::FAILURE_MAJOR;
}

// ─── Label helpers ───────────────────────────────────────────────────────────

[[nodiscard]] constexpr std::string_view change_class_label(SchemaChangeClass c) noexcept {
    switch (c) {
        case SchemaChangeClass::BREAKING:        return "BREAKING (MAJOR bump required)";
        case SchemaChangeClass::BACKWARD_COMPAT: return "BACKWARD_COMPAT (MINOR bump)";
        case SchemaChangeClass::NON_FUNCTIONAL:  return "NON_FUNCTIONAL (PATCH)";
        default:                                 return "UNKNOWN";
    }
}

[[nodiscard]] constexpr std::string_view lifecycle_label(FieldLifecycle l) noexcept {
    switch (l) {
        case FieldLifecycle::ACTIVE:     return "active";
        case FieldLifecycle::DEPRECATED: return "deprecated";
        case FieldLifecycle::TOMBSTONED: return "tombstoned";
        default:                         return "unknown";
    }
}

[[nodiscard]] constexpr std::string_view compat_result_label(CompatibilityResult r) noexcept {
    switch (r) {
        case CompatibilityResult::SUCCESS_FULL:      return "SUCCESS_FULL";
        case CompatibilityResult::SUCCESS_FORWARD:   return "SUCCESS_FORWARD";
        case CompatibilityResult::SUCCESS_BACKWARD:  return "SUCCESS_BACKWARD";
        case CompatibilityResult::FAILURE_MAJOR:     return "FAILURE_MAJOR";
        default:                                     return "UNKNOWN";
    }
}

} // namespace nikola::system
