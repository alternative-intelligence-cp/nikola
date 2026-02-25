// ============================================================
// Phase 68 — GAP-023 Protocol Buffer Schema Evolution Strategy
// tests/unit/phase68_proto_schema_test.cpp
//
// Test domains:
//  §1  SchemaVersion struct, comparison
//  §2  classify_version_change + is_breaking_change
//  §3  Package namespacing
//  §4  Field lifecycle (FieldStatus, immutability rule)
//  §5  Tombstone protocol helpers
//  §6  Required vs optional field guidelines
//  §7  Compatibility matrix M[producer, consumer]
//  §8  is_safely_processable + requires_translation_shim
//  §9  Sparse waveform significance threshold
//  §10 Morton key size validation
//  §11 Artifact naming convention
//  §12 Diagnostic names
//  §13 Invariants
//  §14 Integration: rolling upgrade scenarios
// ============================================================
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <nikola/infrastructure/proto_schema.hpp>

using namespace nikola::infrastructure;
using Catch::Approx;

// ── §1 SchemaVersion ──────────────────────────────────────────────────────────

TEST_CASE("§1.1 SchemaVersion equality", "[semver][GAP-023]") {
    SchemaVersion a{2, 1, 0};
    SchemaVersion b{2, 1, 0};
    SchemaVersion c{2, 2, 0};
    REQUIRE(a == b);
    REQUIRE(a != c);
}

TEST_CASE("§1.2 SchemaVersion ordering (MAJOR precedence)", "[semver][GAP-023]") {
    REQUIRE(SchemaVersion{1,9,9} < SchemaVersion{2,0,0});
    REQUIRE(SchemaVersion{2,0,0} > SchemaVersion{1,9,9});
}

TEST_CASE("§1.3 SchemaVersion ordering (MINOR precedence)", "[semver][GAP-023]") {
    REQUIRE(SchemaVersion{2,0,9} < SchemaVersion{2,1,0});
}

TEST_CASE("§1.4 SchemaVersion ordering (PATCH precedence)", "[semver][GAP-023]") {
    REQUIRE(SchemaVersion{2,1,0} < SchemaVersion{2,1,1});
}

TEST_CASE("§1.5 SchemaVersion v1.0.0 < v2.0.0 (epoch boundary)", "[semver][GAP-023]") {
    REQUIRE(SchemaVersion{1,0,0} < SchemaVersion{2,0,0});
    REQUIRE_FALSE(SchemaVersion{2,0,0} < SchemaVersion{1,0,0});
}

// ── §2 classify_version_change ────────────────────────────────────────────────

TEST_CASE("§2.1 Identical versions → NO_CHANGE", "[version_change][GAP-023]") {
    REQUIRE(classify_version_change({2,1,3}, {2,1,3}) == VersionChangeKind::NO_CHANGE);
    REQUIRE(classify_version_change({1,0,0}, {1,0,0}) == VersionChangeKind::NO_CHANGE);
}

TEST_CASE("§2.2 PATCH increment only → PATCH", "[version_change][GAP-023]") {
    REQUIRE(classify_version_change({2,1,0}, {2,1,1}) == VersionChangeKind::PATCH);
    REQUIRE(classify_version_change({1,0,5}, {1,0,6}) == VersionChangeKind::PATCH);
}

TEST_CASE("§2.3 MINOR increment → MINOR_ADDITIVE", "[version_change][GAP-023]") {
    REQUIRE(classify_version_change({2,0,0}, {2,1,0}) == VersionChangeKind::MINOR_ADDITIVE);
    REQUIRE(classify_version_change({2,1,3}, {2,2,0}) == VersionChangeKind::MINOR_ADDITIVE);
}

TEST_CASE("§2.4 MAJOR increment → MAJOR_BREAKING", "[version_change][GAP-023]") {
    REQUIRE(classify_version_change({1,0,0}, {2,0,0}) == VersionChangeKind::MAJOR_BREAKING);
    REQUIRE(classify_version_change({2,5,3}, {3,0,0}) == VersionChangeKind::MAJOR_BREAKING);
}

TEST_CASE("§2.5 Downgrade → MAJOR_BREAKING (treated as breaking)", "[version_change][GAP-023]") {
    REQUIRE(classify_version_change({2,0,0}, {1,0,0}) == VersionChangeKind::MAJOR_BREAKING);
    REQUIRE(classify_version_change({2,1,0}, {2,0,0}) == VersionChangeKind::MAJOR_BREAKING);
}

TEST_CASE("§2.6 is_breaking_change is true only for MAJOR_BREAKING", "[version_change][GAP-023]") {
    REQUIRE_FALSE(is_breaking_change(VersionChangeKind::NO_CHANGE));
    REQUIRE_FALSE(is_breaking_change(VersionChangeKind::PATCH));
    REQUIRE_FALSE(is_breaking_change(VersionChangeKind::MINOR_ADDITIVE));
    REQUIRE(is_breaking_change(VersionChangeKind::MAJOR_BREAKING));
}

TEST_CASE("§2.7 INT-06 case: v1.0.0 → v2.0.0 (int32→bytes coordinate migration)", "[version_change][GAP-023]") {
    // Spec case study §2.6.2: coordinate format migration required MAJOR bump
    const auto kind = classify_version_change({1,0,0}, {2,0,0});
    REQUIRE(kind == VersionChangeKind::MAJOR_BREAKING);
    REQUIRE(is_breaking_change(kind));
}

// ── §3 Package Namespacing ────────────────────────────────────────────────────

TEST_CASE("§3.1 make_package_name generates correct namespace strings", "[namespace][GAP-023]") {
    REQUIRE(make_package_name(1) == "nikola.spine.v1");
    REQUIRE(make_package_name(2) == "nikola.spine.v2");
    REQUIRE(make_package_name(3) == "nikola.spine.v3");
}

TEST_CASE("§3.2 v1 and v2 namespaces are distinct", "[namespace][GAP-023]") {
    REQUIRE(make_package_name(1) != make_package_name(2));
}

TEST_CASE("§3.3 same_namespace: same major → true", "[namespace][GAP-023]") {
    REQUIRE(same_namespace(2, 2));
    REQUIRE_FALSE(same_namespace(1, 2));
    REQUIRE_FALSE(same_namespace(2, 1));
}

TEST_CASE("§3.4 make_package_name out-of-range throws", "[namespace][error][GAP-023]") {
    REQUIRE_THROWS_AS(make_package_name(MAX_SUPPORTED_MAJOR + 1),
                      std::invalid_argument);
}

TEST_CASE("§3.5 make_package_name v0 (initial schema)", "[namespace][GAP-023]") {
    REQUIRE(make_package_name(0) == "nikola.spine.v0");
}

// ── §4 Field Lifecycle ────────────────────────────────────────────────────────

TEST_CASE("§4.1 ACTIVE field ID is not safe to reuse", "[field_lifecycle][GAP-023]") {
    REQUIRE_FALSE(is_id_safe_to_reuse(FieldStatus::ACTIVE));
}

TEST_CASE("§4.2 DEPRECATED field ID is not safe to reuse", "[field_lifecycle][GAP-023]") {
    REQUIRE_FALSE(is_id_safe_to_reuse(FieldStatus::DEPRECATED));
}

TEST_CASE("§4.3 TOMBSTONED field ID is not safe to reuse (permanently reserved)", "[field_lifecycle][GAP-023]") {
    REQUIRE_FALSE(is_id_safe_to_reuse(FieldStatus::TOMBSTONED));
}

TEST_CASE("§4.4 No field status permits ID reuse (universal law)", "[field_lifecycle][GAP-023]") {
    for (auto s : {FieldStatus::ACTIVE, FieldStatus::DEPRECATED, FieldStatus::TOMBSTONED})
        REQUIRE_FALSE(is_id_safe_to_reuse(s));
}

// ── §5 Tombstone Protocol ─────────────────────────────────────────────────────

TEST_CASE("§5.1 make_obsolete_name prepends OBSOLETE_ prefix", "[tombstone][GAP-023]") {
    REQUIRE(make_obsolete_name("coordinates") == "OBSOLETE_coordinates");
    REQUIRE(make_obsolete_name("x_pos") == "OBSOLETE_x_pos");
}

TEST_CASE("§5.2 is_obsolete_name detects OBSOLETE_ prefix", "[tombstone][GAP-023]") {
    REQUIRE(is_obsolete_name("OBSOLETE_coordinates"));
    REQUIRE(is_obsolete_name("OBSOLETE_x_pos"));
    REQUIRE_FALSE(is_obsolete_name("coordinates"));
    REQUIRE_FALSE(is_obsolete_name("coordinates_OBSOLETE"));
}

TEST_CASE("§5.3 tombstone_protocol_complete: all 3 steps required", "[tombstone][GAP-023]") {
    // Step 1+2: TOMBSTONED status; Step 3: OBSOLETE_ name
    REQUIRE(tombstone_protocol_complete(FieldStatus::TOMBSTONED, "OBSOLETE_coordinates"));

    // Missing name rename
    REQUIRE_FALSE(tombstone_protocol_complete(FieldStatus::TOMBSTONED, "coordinates"));

    // Only deprecated, not tombstoned
    REQUIRE_FALSE(tombstone_protocol_complete(FieldStatus::DEPRECATED, "OBSOLETE_coordinates"));

    // Active field — both wrong
    REQUIRE_FALSE(tombstone_protocol_complete(FieldStatus::ACTIVE, "coordinates"));
}

TEST_CASE("§5.4 INT-06 tombstone: OBSOLETE_coordinates field", "[tombstone][GAP-023]") {
    // Spec §2.6.2 case study — old int32 coordinate field
    const std::string old_name = "coordinates";
    const auto obsolete = make_obsolete_name(old_name);
    REQUIRE(is_obsolete_name(obsolete));
    REQUIRE(tombstone_protocol_complete(FieldStatus::TOMBSTONED, obsolete));
}

// ── §6 Required vs Optional ───────────────────────────────────────────────────

TEST_CASE("§6.1 request_id and timestamp are logically required", "[optional][GAP-023]") {
    REQUIRE(is_logically_required_header("request_id"));
    REQUIRE(is_logically_required_header("timestamp"));
}

TEST_CASE("§6.2 Other fields are not logically required headers", "[optional][GAP-023]") {
    REQUIRE_FALSE(is_logically_required_header("amplitude"));
    REQUIRE_FALSE(is_logically_required_header("dopamine_level"));
    REQUIRE_FALSE(is_logically_required_header("morton_indices"));
}

TEST_CASE("§6.3 Fields where zero is valid physical value need explicit optional", "[optional][GAP-023]") {
    REQUIRE(requires_explicit_optional("coordinate"));
    REQUIRE(requires_explicit_optional("energy"));
    REQUIRE(requires_explicit_optional("amplitude"));
    REQUIRE(requires_explicit_optional("resonance"));
}

TEST_CASE("§6.4 Non-physical fields do not require explicit optional", "[optional][GAP-023]") {
    REQUIRE_FALSE(requires_explicit_optional("request_id"));
    REQUIRE_FALSE(requires_explicit_optional("sender_rank"));
    REQUIRE_FALSE(requires_explicit_optional("deprecated_flag"));
}

// ── §7 Compatibility Matrix ───────────────────────────────────────────────────

TEST_CASE("§7.1 Same major → FULL_FIDELITY (spec row 1)", "[compat_matrix][GAP-023]") {
    REQUIRE(compatibility_class(2, 2) == CompatibilityClass::FULL_FIDELITY);
    REQUIRE(compatibility_class(1, 1) == CompatibilityClass::FULL_FIDELITY);
    REQUIRE(compatibility_class(0, 0) == CompatibilityClass::FULL_FIDELITY);
}

TEST_CASE("§7.2 Legacy producer → modern consumer (v1→v2) → FORWARD_COMPAT (spec row 2)", "[compat_matrix][GAP-023]") {
    REQUIRE(compatibility_class(1, 2) == CompatibilityClass::FORWARD_COMPAT);
}

TEST_CASE("§7.3 Modern producer → legacy consumer (v2→v1) → BACKWARD_COMPAT (spec row 3)", "[compat_matrix][GAP-023]") {
    REQUIRE(compatibility_class(2, 1) == CompatibilityClass::BACKWARD_COMPAT);
}

TEST_CASE("§7.4 Next-gen producer → current consumer (v3→v2) → BACKWARD_COMPAT (spec row 4)", "[compat_matrix][GAP-023]") {
    // Producer=3, consumer=2: 1-step gap. Unknown fields buffered transparently.
    REQUIRE(compatibility_class(3, 2) == CompatibilityClass::BACKWARD_COMPAT);
}

TEST_CASE("§7.5 Two-step gap (v3→v1) → INCOMPATIBLE", "[compat_matrix][GAP-023]") {
    REQUIRE(compatibility_class(3, 1) == CompatibilityClass::INCOMPATIBLE);
    REQUIRE(compatibility_class(1, 3) == CompatibilityClass::INCOMPATIBLE);
}

TEST_CASE("§7.6 Large forward gap → INCOMPATIBLE", "[compat_matrix][GAP-023]") {
    REQUIRE(compatibility_class(1, 5) == CompatibilityClass::INCOMPATIBLE);
    REQUIRE(compatibility_class(5, 1) == CompatibilityClass::INCOMPATIBLE);
}

TEST_CASE("§7.7 is_safely_processable: only INCOMPATIBLE is unsafe", "[compat_matrix][GAP-023]") {
    REQUIRE(is_safely_processable(CompatibilityClass::FULL_FIDELITY));
    REQUIRE(is_safely_processable(CompatibilityClass::FORWARD_COMPAT));
    REQUIRE(is_safely_processable(CompatibilityClass::BACKWARD_COMPAT));
    REQUIRE_FALSE(is_safely_processable(CompatibilityClass::INCOMPATIBLE));
}

TEST_CASE("§7.8 FORWARD_COMPAT is safely processable (default-value semantics)", "[compat_matrix][GAP-023]") {
    // v1 producer, v2 consumer — new v2 fields get defaults
    REQUIRE(is_safely_processable(compatibility_class(1, 2)));
}

// ── §8 Translation Shim Requirement ──────────────────────────────────────────

TEST_CASE("§8.1 requires_translation_shim: 1-step gap does NOT require shim", "[shim][GAP-023]") {
    REQUIRE_FALSE(requires_translation_shim(1, 2));
    REQUIRE_FALSE(requires_translation_shim(2, 1));
    REQUIRE_FALSE(requires_translation_shim(2, 2));
}

TEST_CASE("§8.2 requires_translation_shim: 2+ step gap requires shim", "[shim][GAP-023]") {
    REQUIRE(requires_translation_shim(3, 1));
    REQUIRE(requires_translation_shim(1, 3));
    REQUIRE(requires_translation_shim(1, 5));
}

TEST_CASE("§8.3 INT-06 migration (v1→v2) does not require shim (1-step)", "[shim][GAP-023]") {
    // The spec provides a shim by choice; the protocol doesn't mandate one for 1-step
    REQUIRE_FALSE(requires_translation_shim(1, 2));
}

// ── §9 Sparse Waveform Threshold ──────────────────────────────────────────────

TEST_CASE("§9.1 SPARSE_WAVEFORM_THETA constant is 0.1", "[sparse][GAP-023]") {
    REQUIRE(SPARSE_WAVEFORM_THETA == Approx(0.10f));
}

TEST_CASE("§9.2 Node above threshold (|Ψ| > 0.1 × RMS) → serialise", "[sparse][GAP-023]") {
    REQUIRE(is_above_significance_threshold(0.5f, 1.0f));   // 0.5 > 0.1
    REQUIRE(is_above_significance_threshold(1.0f, 1.0f));   // 1.0 > 0.1
}

TEST_CASE("§9.3 Node exactly at threshold → NOT serialised (strict >)", "[sparse][GAP-023]") {
    REQUIRE_FALSE(is_above_significance_threshold(0.1f, 1.0f));  // 0.1 > 0.1 is false
}

TEST_CASE("§9.4 Node below threshold → filter out", "[sparse][GAP-023]") {
    REQUIRE_FALSE(is_above_significance_threshold(0.05f, 1.0f));
    REQUIRE_FALSE(is_above_significance_threshold(0.0f,  1.0f));
}

TEST_CASE("§9.5 Custom theta applied correctly", "[sparse][GAP-023]") {
    REQUIRE(is_above_significance_threshold(0.3f, 1.0f, 0.2f));   // 0.3 > 0.2
    REQUIRE_FALSE(is_above_significance_threshold(0.1f, 1.0f, 0.2f)); // 0.1 <= 0.2
}

TEST_CASE("§9.6 RMS ≤ 0 throws invalid_argument", "[sparse][error][GAP-023]") {
    REQUIRE_THROWS_AS(is_above_significance_threshold(1.0f, 0.0f), std::invalid_argument);
    REQUIRE_THROWS_AS(is_above_significance_threshold(1.0f, -1.0f), std::invalid_argument);
}

// ── §10 Morton Key Validation ─────────────────────────────────────────────────

TEST_CASE("§10.1 MORTON_KEY_SIZE_BYTES is 16", "[morton][GAP-023]") {
    REQUIRE(MORTON_KEY_SIZE_BYTES == 16u);
}

TEST_CASE("§10.2 Exactly 16 bytes → valid", "[morton][GAP-023]") {
    REQUIRE(is_valid_morton_key_size(16u));
}

TEST_CASE("§10.3 Wrong sizes → invalid", "[morton][GAP-023]") {
    REQUIRE_FALSE(is_valid_morton_key_size(0u));
    REQUIRE_FALSE(is_valid_morton_key_size(8u));
    REQUIRE_FALSE(is_valid_morton_key_size(15u));
    REQUIRE_FALSE(is_valid_morton_key_size(17u));
    REQUIRE_FALSE(is_valid_morton_key_size(32u));
}

// ── §11 Artifact Naming ───────────────────────────────────────────────────────

TEST_CASE("§11.1 make_artifact_name generates spec-compliant string", "[artifact][GAP-023]") {
    REQUIRE(make_artifact_name("nikola", {2,1,0}) == "libnikola-proto-v2.1.0.so");
    REQUIRE(make_artifact_name("nikola", {1,0,0}) == "libnikola-proto-v1.0.0.so");
}

TEST_CASE("§11.2 Artifact names for different patch versions differ", "[artifact][GAP-023]") {
    REQUIRE(make_artifact_name("nikola", {2,1,0}) != make_artifact_name("nikola", {2,1,1}));
}

TEST_CASE("§11.3 Artifact names for different major versions differ", "[artifact][GAP-023]") {
    REQUIRE(make_artifact_name("nikola", {1,0,0}) != make_artifact_name("nikola", {2,0,0}));
}

TEST_CASE("§11.4 Spec example: libnikola-proto-v2.1.so naming pattern", "[artifact][GAP-023]") {
    // Spec §2.6.5 references e.g. libnikola-proto-v2.1.so
    const auto name = make_artifact_name("nikola", {2,1,0});
    REQUIRE(name.starts_with("libnikola-proto-v2.1.0"));
    REQUIRE(name.ends_with(".so"));
}

// ── §12 Diagnostic Names ──────────────────────────────────────────────────────

TEST_CASE("§12.1 version_change_kind_name covers all values", "[names][GAP-023]") {
    REQUIRE(version_change_kind_name(VersionChangeKind::NO_CHANGE)      == "NO_CHANGE");
    REQUIRE(version_change_kind_name(VersionChangeKind::PATCH)          == "PATCH");
    REQUIRE(version_change_kind_name(VersionChangeKind::MINOR_ADDITIVE) == "MINOR_ADDITIVE");
    REQUIRE(version_change_kind_name(VersionChangeKind::MAJOR_BREAKING) == "MAJOR_BREAKING");
}

TEST_CASE("§12.2 compatibility_class_name covers all values", "[names][GAP-023]") {
    REQUIRE(compatibility_class_name(CompatibilityClass::FULL_FIDELITY)   == "FULL_FIDELITY");
    REQUIRE(compatibility_class_name(CompatibilityClass::FORWARD_COMPAT)  == "FORWARD_COMPAT");
    REQUIRE(compatibility_class_name(CompatibilityClass::BACKWARD_COMPAT) == "BACKWARD_COMPAT");
    REQUIRE(compatibility_class_name(CompatibilityClass::FUTURE_COMPAT)   == "FUTURE_COMPAT");
    REQUIRE(compatibility_class_name(CompatibilityClass::INCOMPATIBLE)    == "INCOMPATIBLE");
}

TEST_CASE("§12.3 field_status_name covers all values", "[names][GAP-023]") {
    REQUIRE(field_status_name(FieldStatus::ACTIVE)     == "ACTIVE");
    REQUIRE(field_status_name(FieldStatus::DEPRECATED) == "DEPRECATED");
    REQUIRE(field_status_name(FieldStatus::TOMBSTONED) == "TOMBSTONED");
}

// ── §13 Invariants ────────────────────────────────────────────────────────────

TEST_CASE("§13.1 classify_version_change is reflexive: v → v = NO_CHANGE", "[invariants][GAP-023]") {
    for (uint32_t major : {0u, 1u, 2u, 3u}) {
        REQUIRE(classify_version_change({major,0,0}, {major,0,0}) == VersionChangeKind::NO_CHANGE);
    }
}

TEST_CASE("§13.2 MAJOR bump always dominates MINOR/PATCH changes", "[invariants][GAP-023]") {
    // Even if minor/patch also change, a major increment is MAJOR_BREAKING
    REQUIRE(classify_version_change({1,5,3}, {2,0,0}) == VersionChangeKind::MAJOR_BREAKING);
    REQUIRE(classify_version_change({1,0,0}, {2,3,5}) == VersionChangeKind::MAJOR_BREAKING);
}

TEST_CASE("§13.3 No field status permits ID reuse (universality)", "[invariants][GAP-023]") {
    for (auto s : {FieldStatus::ACTIVE, FieldStatus::DEPRECATED, FieldStatus::TOMBSTONED})
        REQUIRE_FALSE(is_id_safe_to_reuse(s));
}

TEST_CASE("§13.4 FULL_FIDELITY iff producer == consumer (always safe)", "[invariants][GAP-023]") {
    for (uint32_t v : {0u, 1u, 2u, 5u}) {
        REQUIRE(compatibility_class(v, v) == CompatibilityClass::FULL_FIDELITY);
        REQUIRE(is_safely_processable(compatibility_class(v, v)));
    }
}

TEST_CASE("§13.5 make_obsolete_name is idempotent-safe (just prepends once)", "[invariants][GAP-023]") {
    const auto once = make_obsolete_name("field");
    REQUIRE(is_obsolete_name(once));
    REQUIRE(once == "OBSOLETE_field");
}

TEST_CASE("§13.6 Sparse threshold linearly scales with RMS", "[invariants][GAP-023]") {
    // Doubling RMS doubles the effective threshold
    const float amp = 0.2f;
    const bool at_rms1 = is_above_significance_threshold(amp, 1.0f);  // threshold = 0.1
    const bool at_rms2 = is_above_significance_threshold(amp, 2.0f);  // threshold = 0.2
    // amp=0.2 > 0.1 → true; amp=0.2 not > 0.2 → false
    REQUIRE(at_rms1);
    REQUIRE_FALSE(at_rms2);
}

// ── §14  Integration: Rolling Upgrade Scenarios ───────────────────────────────

TEST_CASE("§14.1 Ship of Theseus: coexist v1 and v2 on same bus", "[integration][GAP-023]") {
    // During rolling upgrade, both v1 and v2 producers coexist
    // v2 consumer receiving v1 message: forward compat (defaults for new fields)
    const auto v1_to_v2 = compatibility_class(1, 2);
    REQUIRE(v1_to_v2 == CompatibilityClass::FORWARD_COMPAT);
    REQUIRE(is_safely_processable(v1_to_v2));

    // v1 consumer receiving v2 message: backward compat (new fields silently ignored)
    const auto v2_to_v1 = compatibility_class(2, 1);
    REQUIRE(v2_to_v1 == CompatibilityClass::BACKWARD_COMPAT);
    REQUIRE(is_safely_processable(v2_to_v1));
}

TEST_CASE("§14.2 INT-06 complete migration lifecycle", "[integration][GAP-023]") {
    // Step 1: classify the change from v1 to v2
    const auto change = classify_version_change({1,0,0}, {2,0,0});
    REQUIRE(is_breaking_change(change));

    // Step 2: old field gets tombstoned
    const auto old_field = make_obsolete_name("coordinates");
    REQUIRE(tombstone_protocol_complete(FieldStatus::TOMBSTONED, old_field));

    // Step 3: new field must be exactly 16 bytes
    REQUIRE(is_valid_morton_key_size(16u));
    REQUIRE_FALSE(is_valid_morton_key_size(4u));  // old int32 was 4 bytes

    // Step 4: artifact published for v2
    const auto artifact = make_artifact_name("nikola", {2,0,0});
    REQUIRE(artifact == "libnikola-proto-v2.0.0.so");
}

TEST_CASE("§14.3 Sparse waveform filtering: physics grid scenario", "[integration][GAP-023]") {
    // Simulate 5 nodes with varying amplitudes, RMS = 1.0
    const float rms = 1.0f;
    const std::array<float,5> amplitudes{0.05f, 0.08f, 0.10f, 0.25f, 1.50f};
    int serialised = 0;
    for (float amp : amplitudes)
        if (is_above_significance_threshold(amp, rms))
            ++serialised;
    // Only 0.25 and 1.50 exceed 0.1 × 1.0 = 0.1 (strict >)
    REQUIRE(serialised == 2);
}

TEST_CASE("§14.4 Version negotiation: multi-rank cluster", "[integration][GAP-023]") {
    // Cluster has ranks at v1, v2, v2 — compute all pairwise compat
    const std::array<uint32_t,3> ranks{1, 2, 2};
    int safe_pairs  = 0;
    int unsafe_pairs = 0;
    for (auto prod : ranks) {
        for (auto cons : ranks) {
            const auto c = compatibility_class(prod, cons);
            if (is_safely_processable(c))
                ++safe_pairs;
            else
                ++unsafe_pairs;
        }
    }
    // All 3×3 = 9 pairs should be safe (max 1-step gap)
    REQUIRE(safe_pairs == 9);
    REQUIRE(unsafe_pairs == 0);
}

TEST_CASE("§14.5 Schema document lifecycle: PATCH → MINOR → MAJOR", "[integration][GAP-023]") {
    SchemaVersion v{2, 0, 0};

    // Documentation fix → PATCH
    SchemaVersion v_p = {v.major, v.minor, v.patch + 1};
    REQUIRE(classify_version_change(v, v_p) == VersionChangeKind::PATCH);
    REQUIRE_FALSE(is_breaking_change(classify_version_change(v, v_p)));

    // New field (dopamine_level) → MINOR
    SchemaVersion v_m = {v.major, v.minor + 1, 0};
    REQUIRE(classify_version_change(v, v_m) == VersionChangeKind::MINOR_ADDITIVE);
    REQUIRE_FALSE(is_breaking_change(classify_version_change(v, v_m)));

    // Coordinate migration (int32→bytes) → MAJOR
    SchemaVersion v_M = {v.major + 1, 0, 0};
    REQUIRE(classify_version_change(v, v_M) == VersionChangeKind::MAJOR_BREAKING);
    REQUIRE(is_breaking_change(classify_version_change(v, v_M)));
}
