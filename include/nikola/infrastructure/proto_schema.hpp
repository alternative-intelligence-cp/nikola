#pragma once
// ============================================================
// nikola/infrastructure/proto_schema.hpp
// GAP-023 — Protocol Buffer Schema Evolution Strategy
//
// Models and enforces the schema evolution rules for the
// NeuralSpike ZeroMQ spine protocol:
//   §2.6.1 SemVer classification and package namespacing
//   §2.6.2 Field lifecycle (immutability, deprecation, tombstoning)
//   §2.6.3 Required vs optional field guidelines
//   §2.6.4 Automated compatibility matrix M[producer, consumer]
//   §2.6.5 Documentation and artifact requirements
//
// Namespace: nikola::infrastructure
// ============================================================

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace nikola::infrastructure {

// ── § SemVer Types ────────────────────────────────────────────────────────────

/// Semantic version triple.
struct SchemaVersion {
    uint32_t major;  ///< Breaking wire-format change (field renumber / type change)
    uint32_t minor;  ///< Backward-compatible addition (new optional field)
    uint32_t patch;  ///< Non-functional (docs, comments)

    [[nodiscard]] constexpr bool operator==(const SchemaVersion&) const noexcept = default;
    [[nodiscard]] constexpr auto operator<=>(const SchemaVersion&) const noexcept = default;
};

/// Classification of a version-to-version transition (§2.6.1).
enum class VersionChangeKind : uint8_t {
    NO_CHANGE,       ///< Identical versions.
    PATCH,           ///< Comment / documentation only.
    MINOR_ADDITIVE,  ///< New optional field — backward-compatible.
    MAJOR_BREAKING,  ///< Field renumber / type change / removal — requires migration.
};

/// True when the change kind mandates a synchronous migration or translation shim.
[[nodiscard]] constexpr bool is_breaking_change(VersionChangeKind k) noexcept
{
    return k == VersionChangeKind::MAJOR_BREAKING;
}

/// Classify the transition from @p old_ver to @p new_ver.
/// Spec rule: major increment → MAJOR_BREAKING;
///            minor increment only → MINOR_ADDITIVE;
///            patch increment only → PATCH;
///            identical → NO_CHANGE;
///            downgrade → MAJOR_BREAKING (treat as breaking).
[[nodiscard]] constexpr VersionChangeKind classify_version_change(
    SchemaVersion old_ver,
    SchemaVersion new_ver) noexcept
{
    if (old_ver == new_ver)                        return VersionChangeKind::NO_CHANGE;
    if (new_ver.major != old_ver.major)            return VersionChangeKind::MAJOR_BREAKING;
    if (new_ver < old_ver)                         return VersionChangeKind::MAJOR_BREAKING; // downgrade
    if (new_ver.minor != old_ver.minor)            return VersionChangeKind::MINOR_ADDITIVE;
    return VersionChangeKind::PATCH;
}

// ── § Package Namespacing ─────────────────────────────────────────────────────

/// Maximum supported MAJOR version for pre-computed namespace strings.
constexpr uint32_t MAX_SUPPORTED_MAJOR = 9u;

/// Spec §2.6.1: package namespace pattern is "nikola.spine.vN".
/// Compile-time helper — returns the package string for a given major version.
///
/// @throws std::invalid_argument for major > MAX_SUPPORTED_MAJOR.
[[nodiscard]] inline std::string make_package_name(uint32_t major)
{
    if (major > MAX_SUPPORTED_MAJOR)
        throw std::invalid_argument("make_package_name: major version out of supported range");
    return "nikola.spine.v" + std::to_string(major);
}

/// True when two packages are in the same major-version namespace.
[[nodiscard]] constexpr bool same_namespace(uint32_t major_a, uint32_t major_b) noexcept
{
    return major_a == major_b;
}

// ── § Field Lifecycle ─────────────────────────────────────────────────────────

/// Field lifecycle stages per §2.6.2 Tombstone protocol.
enum class FieldStatus : uint8_t {
    ACTIVE,      ///< Field is in normal use.
    DEPRECATED,  ///< Marked `deprecated=true`; old readers still work, new code avoids it.
    TOMBSTONED,  ///< Added to `reserved` list; ID must NEVER be reused.
};

/// Spec rule: a tombstoned field ID is permanently off-limits.
/// Any other status never implies the ID is safe to reuse for a *different* semantic.
[[nodiscard]] constexpr bool is_id_safe_to_reuse(FieldStatus status) noexcept
{
    // Spec §2.6.2: "Once a field ID is assigned, it must never be reused or re-purposed."
    // ACTIVE and DEPRECATED IDs are still in circulation; TOMBSTONED are explicitly reserved.
    // All three are unsafe to reuse. Return false for all states.
    (void)status;
    return false;
}

/// Step 3 of the Tombstone protocol: field must be renamed with OBSOLETE_ prefix.
[[nodiscard]] inline std::string make_obsolete_name(std::string_view field_name)
{
    return std::string("OBSOLETE_") + std::string(field_name);
}

/// True when a field name has been correctly tombstone-renamed per spec §2.6.2.
[[nodiscard]] inline bool is_obsolete_name(std::string_view name) noexcept
{
    return name.starts_with("OBSOLETE_");
}

/// True when all three Tombstone steps have been applied:
///   1. FieldStatus::DEPRECATED or TOMBSTONED
///   2. ID reserved (modelled as status == TOMBSTONED)
///   3. Name bears OBSOLETE_ prefix
[[nodiscard]] inline bool tombstone_protocol_complete(
    FieldStatus status,
    std::string_view name) noexcept
{
    return status == FieldStatus::TOMBSTONED && is_obsolete_name(name);
}

// ── § Optional vs Required Field Guidelines ───────────────────────────────────

/// Spec §2.6.3: `request_id` and `timestamp` are logically required in every
/// NeuralSpike envelope (validated by SecureChannel before deserialization).
[[nodiscard]] constexpr bool is_logically_required_header(std::string_view field_name) noexcept
{
    return field_name == "request_id" || field_name == "timestamp";
}

/// True when a zero value is a physically meaningful quantity that must be
/// distinguishable from "field absent" — spec §2.6.3 Guideline 1.
/// Applies to all coordinate fields and energy values.
[[nodiscard]] constexpr bool requires_explicit_optional(std::string_view field_name) noexcept
{
    // Fields where zero is a valid physical value: coordinates and energies.
    return field_name == "coordinate"
        || field_name == "energy"
        || field_name == "amplitude"
        || field_name == "resonance";
}

// ── § Compatibility Matrix M[producer, consumer] ──────────────────────────────

/// Spec §2.6.4 table: four defined compatibility classes for (producer, consumer) pairs.
enum class CompatibilityClass : uint8_t {
    FULL_FIDELITY,   ///< Same version → all fields accessible.
    FORWARD_COMPAT,  ///< Legacy producer → modern consumer: defaults for new fields.
    BACKWARD_COMPAT, ///< Modern producer → legacy consumer: new fields ignored safely.
    FUTURE_COMPAT,   ///< Future producer → current consumer: unknown fields buffered.
    INCOMPATIBLE,    ///< More-than-one-major gap; no safe handling guaranteed.
};

/// Classify compatibility between @p producer_major and @p consumer_major.
///
/// Spec matrix:
///   producer == consumer (vX,vX) → FULL_FIDELITY
///   producer < consumer  (v1,v2) → FORWARD_COMPAT
///   producer > consumer  (v2,v1) → BACKWARD_COMPAT
///   producer == consumer+1+      → FUTURE_COMPAT (one step ahead)
///   delta > 1                    → INCOMPATIBLE
///
/// Note: spec defines max safe "future gap" as 1 major step,
/// since beyond that unknown-field pass-through cannot be guaranteed.
[[nodiscard]] constexpr CompatibilityClass compatibility_class(
    uint32_t producer_major,
    uint32_t consumer_major) noexcept
{
    if (producer_major == consumer_major)
        return CompatibilityClass::FULL_FIDELITY;

    if (producer_major < consumer_major) {
        // Legacy producer, modern consumer — forward compat (default values)
        return (consumer_major - producer_major == 1u)
               ? CompatibilityClass::FORWARD_COMPAT
               : CompatibilityClass::INCOMPATIBLE;
    }

    // producer_major > consumer_major: new producer, old consumer
    if (producer_major - consumer_major == 1u)
        return CompatibilityClass::BACKWARD_COMPAT;

    // "Future" from producer's perspective (spec row 4 in table)
    // In Nikola spec this maps to "Future (v3) → Current (v2)" = FUTURE_COMPAT
    // but only for 1-step gap; larger gaps are INCOMPATIBLE.
    return CompatibilityClass::INCOMPATIBLE;
}

/// True when the compatibility class allows safe processing without a translation shim.
[[nodiscard]] constexpr bool is_safely_processable(CompatibilityClass c) noexcept
{
    return c != CompatibilityClass::INCOMPATIBLE;
}

/// True when a translation shim is mandatory (breaking major gap).
[[nodiscard]] constexpr bool requires_translation_shim(
    uint32_t producer_major,
    uint32_t consumer_major) noexcept
{
    return compatibility_class(producer_major, consumer_major)
           == CompatibilityClass::INCOMPATIBLE;
}

// ── § Sparse Waveform Significance Threshold ─────────────────────────────────

/// Spec §2 (NET-02 fix): Only nodes with |Ψ| > θ × RMS are serialised.
/// Default θ = 0.1.
constexpr float SPARSE_WAVEFORM_THETA = 0.10f;

/// True when a node's amplitude exceeds the sparse-waveform significance threshold.
/// @param amplitude  |Ψ| of the node.
/// @param rms        Root-mean-square energy of the grid (must be > 0).
/// @throws std::invalid_argument if rms ≤ 0.
[[nodiscard]] inline bool is_above_significance_threshold(
    float amplitude,
    float rms,
    float theta = SPARSE_WAVEFORM_THETA)
{
    if (rms <= 0.0f)
        throw std::invalid_argument(
            "is_above_significance_threshold: rms must be positive");
    return amplitude > theta * rms;
}

// ── § Morton Key Validation ───────────────────────────────────────────────────

/// Spec §2.6.2 case study: every `morton_indices` entry must be exactly 16 bytes
/// (128-bit, Big Endian).
constexpr size_t MORTON_KEY_SIZE_BYTES = 16u;

[[nodiscard]] constexpr bool is_valid_morton_key_size(size_t byte_count) noexcept
{
    return byte_count == MORTON_KEY_SIZE_BYTES;
}

// ── § Artifact Naming Convention ─────────────────────────────────────────────

/// Spec §2.6.5: artifact name follows pattern `lib<name>-proto-v<M>.<m>.<p>.so`.
[[nodiscard]] inline std::string make_artifact_name(
    std::string_view component,
    SchemaVersion ver)
{
    return "lib" + std::string(component)
         + "-proto-v" + std::to_string(ver.major)
         + "." + std::to_string(ver.minor)
         + "." + std::to_string(ver.patch)
         + ".so";
}

// ── § Diagnostic Names ────────────────────────────────────────────────────────

[[nodiscard]] constexpr std::string_view version_change_kind_name(
    VersionChangeKind k) noexcept
{
    switch (k) {
        case VersionChangeKind::NO_CHANGE:       return "NO_CHANGE";
        case VersionChangeKind::PATCH:           return "PATCH";
        case VersionChangeKind::MINOR_ADDITIVE:  return "MINOR_ADDITIVE";
        case VersionChangeKind::MAJOR_BREAKING:  return "MAJOR_BREAKING";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr std::string_view compatibility_class_name(
    CompatibilityClass c) noexcept
{
    switch (c) {
        case CompatibilityClass::FULL_FIDELITY:   return "FULL_FIDELITY";
        case CompatibilityClass::FORWARD_COMPAT:  return "FORWARD_COMPAT";
        case CompatibilityClass::BACKWARD_COMPAT: return "BACKWARD_COMPAT";
        case CompatibilityClass::FUTURE_COMPAT:   return "FUTURE_COMPAT";
        case CompatibilityClass::INCOMPATIBLE:    return "INCOMPATIBLE";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr std::string_view field_status_name(FieldStatus s) noexcept
{
    switch (s) {
        case FieldStatus::ACTIVE:    return "ACTIVE";
        case FieldStatus::DEPRECATED: return "DEPRECATED";
        case FieldStatus::TOMBSTONED: return "TOMBSTONED";
    }
    return "UNKNOWN";
}

} // namespace nikola::infrastructure
