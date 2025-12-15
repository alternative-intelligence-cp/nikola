#!/bin/bash

################################################################################
# Nikola AGI v0.0.4 Specification Compilation Script
# Purpose: Compile specification files into 9 parts for Gemini Deep Research
# Output: 9 part files (400-500KB each) in current directory
################################################################################

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BASE_DIR="/home/randy/._____RANDY_____/REPOS/nikola/docs/info/integration/sections"
OUTPUT_DIR="/home/randy/._____RANDY_____/REPOS/nikola/docs/info/integration/sections/gemini/compilation"
MAX_SIZE=$((500 * 1024))  # 500KB in bytes

# Function to add a section separator and file content
add_section() {
    local output_file="$1"
    local source_file="$2"
    local section_title="$3"

    # Skip backup files
    if [[ "$source_file" == *".backup"* ]]; then
        return
    fi

    # Check if file exists
    if [[ ! -f "$source_file" ]]; then
        echo -e "${YELLOW}Warning: File not found: $source_file${NC}" >&2
        return
    fi

    # Get relative path for SOURCE comment
    local rel_path="${source_file#$BASE_DIR/}"

    # Add section header
    echo "" >> "$output_file"
    echo "================================================================================" >> "$output_file"
    echo "SECTION: $section_title" >> "$output_file"
    echo "================================================================================" >> "$output_file"
    echo "" >> "$output_file"
    echo "<!-- SOURCE: $rel_path -->" >> "$output_file"
    echo "" >> "$output_file"

    # Add file content
    cat "$source_file" >> "$output_file"

    echo "" >> "$output_file"
}

# Function to create part file header
create_part_header() {
    local part_num="$1"
    local output_file="$2"
    local title="$3"

    cat > "$output_file" << EOF
################################################################################
# NIKOLA AGI v0.0.4 SPECIFICATION - PART $part_num OF 9
# $title
################################################################################
#
# Compiled: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
# Source: Nikola AGI Integration Repository
# Purpose: Gemini Deep Research Analysis
#
# This document contains integrated specifications including latest bug sweep
# updates and critical remediations.
#
################################################################################

EOF
}

# Function to report file size
report_size() {
    local file="$1"
    local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
    local size_kb=$((size / 1024))
    echo -e "${BLUE}Generated: $(basename "$file") - ${size_kb}KB${NC}"
}

################################################################################
# PART 1: Front Matter + Core Foundations
################################################################################
echo -e "${GREEN}Creating Part 1: Front Matter + Core Foundations${NC}"
PART1="$OUTPUT_DIR/part_1_of_9.txt"
create_part_header "1" "$PART1" "Front Matter + Core Foundations"

# Front Matter
add_section "$PART1" "$BASE_DIR/00_front_matter/00_title_page.md" "0.1 Title Page"
add_section "$PART1" "$BASE_DIR/00_front_matter/01_table_of_contents.md" "0.2 Table of Contents"
add_section "$PART1" "$BASE_DIR/00_front_matter/02_document_provenance.md" "0.3 Document Provenance"

# Core Foundations (Part 1)
add_section "$PART1" "$BASE_DIR/02_foundations/01_9d_toroidal_geometry.md" "2.1 9D Toroidal Geometry"
add_section "$PART1" "$BASE_DIR/02_foundations/02_wave_interference_physics.md" "2.2 Wave Interference Physics [UPDATED]"

report_size "$PART1"

################################################################################
# PART 2: Executive + Foundations + Cognitive Systems
################################################################################
echo -e "${GREEN}Creating Part 2: Executive + Foundations + Cognitive Systems${NC}"
PART2="$OUTPUT_DIR/part_2_of_9.txt"
create_part_header "2" "$PART2" "Executive + Foundations + Cognitive Systems"

# Executive
add_section "$PART2" "$BASE_DIR/01_executive/01_executive_summary.md" "1.1 Executive Summary"

# Foundations (Part 2)
add_section "$PART2" "$BASE_DIR/02_foundations/03_balanced_nonary_logic.md" "2.3 Balanced Nonary Logic"
add_section "$PART2" "$BASE_DIR/02_foundations/04_energy_conservation.md" "2.4 Energy Conservation"

# Cognitive Systems
add_section "$PART2" "$BASE_DIR/03_cognitive_systems/01_wave_interference_processor.md" "3.1 Wave Interference Processor"
add_section "$PART2" "$BASE_DIR/03_cognitive_systems/02_mamba_9d_ssm.md" "3.2 Mamba 9D SSM [UPDATED - Bug Sweep 013]"
add_section "$PART2" "$BASE_DIR/03_cognitive_systems/03_neuroplastic_transformer.md" "3.3 Neuroplastic Transformer [UPDATED - Bug Sweep 015]"
add_section "$PART2" "$BASE_DIR/03_cognitive_systems/04_memory_data_systems.md" "3.4 Memory & Data Systems"

report_size "$PART2"

################################################################################
# PART 3: Infrastructure
################################################################################
echo -e "${GREEN}Creating Part 3: Infrastructure${NC}"
PART3="$OUTPUT_DIR/part_3_of_9.txt"
create_part_header "3" "$PART3" "Infrastructure"

add_section "$PART3" "$BASE_DIR/04_infrastructure/01_zeromq_spine.md" "4.1 ZeroMQ Spine"
add_section "$PART3" "$BASE_DIR/04_infrastructure/02_orchestrator_router.md" "4.2 Orchestrator Router"
add_section "$PART3" "$BASE_DIR/04_infrastructure/03_external_tool_agents.md" "4.3 External Tool Agents"
add_section "$PART3" "$BASE_DIR/04_infrastructure/04_executor_kvm.md" "4.4 Executor KVM"
add_section "$PART3" "$BASE_DIR/04_infrastructure/05_security_subsystem.md" "4.5 Security Subsystem"
add_section "$PART3" "$BASE_DIR/04_infrastructure/06_database_persistence.md" "4.6 Database Persistence"

report_size "$PART3"

################################################################################
# PART 4: Autonomous Systems
################################################################################
echo -e "${GREEN}Creating Part 4: Autonomous Systems${NC}"
PART4="$OUTPUT_DIR/part_4_of_9.txt"
create_part_header "4" "$PART4" "Autonomous Systems"

add_section "$PART4" "$BASE_DIR/05_autonomous_systems/01_computational_neurochemistry.md" "5.1 Computational Neurochemistry"
add_section "$PART4" "$BASE_DIR/05_autonomous_systems/02_training_systems.md" "5.2 Training Systems"
add_section "$PART4" "$BASE_DIR/05_autonomous_systems/03_ingestion_pipeline.md" "5.3 Ingestion Pipeline"
add_section "$PART4" "$BASE_DIR/05_autonomous_systems/04_self_improvement.md" "5.4 Self-Improvement"
add_section "$PART4" "$BASE_DIR/05_autonomous_systems/05_security_systems.md" "5.5 Security Systems"

report_size "$PART4"

################################################################################
# PART 5: Implementation Specifications (Part 1)
################################################################################
echo -e "${GREEN}Creating Part 5: Implementation Specifications (Part 1)${NC}"
PART5="$OUTPUT_DIR/part_5_of_9.txt"
create_part_header "5" "$PART5" "Implementation Specifications (Part 1)"

add_section "$PART5" "$BASE_DIR/06_implementation_specifications/README.md" "6.0 Implementation README"
add_section "$PART5" "$BASE_DIR/06_implementation_specifications/00_implementation_roadmap.md" "6.0 Implementation Roadmap"
add_section "$PART5" "$BASE_DIR/06_implementation_specifications/01_core_physics_implementation.md" "6.1 Core Physics Implementation [UPDATED - Bug Sweep 012]"
add_section "$PART5" "$BASE_DIR/06_implementation_specifications/02_geometry_spatial_implementation.md" "6.2 Geometry & Spatial Implementation"
add_section "$PART5" "$BASE_DIR/06_implementation_specifications/03_cognitive_architecture_implementation.md" "6.3 Cognitive Architecture Implementation"

report_size "$PART5"

################################################################################
# PART 6: Implementation Specifications (Part 2) + Persistence
################################################################################
echo -e "${GREEN}Creating Part 6: Implementation Specifications (Part 2) + Persistence${NC}"
PART6="$OUTPUT_DIR/part_6_of_9.txt"
create_part_header "6" "$PART6" "Implementation Specifications (Part 2) + Persistence"

add_section "$PART6" "$BASE_DIR/06_implementation_specifications/04_infrastructure_comms_implementation.md" "6.4 Infrastructure & Comms Implementation"
add_section "$PART6" "$BASE_DIR/06_implementation_specifications/05_autonomous_systems_implementation.md" "6.5 Autonomous Systems Implementation"
add_section "$PART6" "$BASE_DIR/06_implementation_specifications/06_multimodal_persistence_implementation.md" "6.6 Multimodal Persistence Implementation"
add_section "$PART6" "$BASE_DIR/06_implementation_specifications/07_security_execution_implementation.md" "6.7 Security & Execution Implementation"
add_section "$PART6" "$BASE_DIR/06_implementation_specifications/08_critical_remediations.md" "6.8 Critical Remediations [UPDATED - Bug Sweep 014]"
add_section "$PART6" "$BASE_DIR/06_persistence/01_dmc_persistence.md" "6.P1 DMC Persistence"

report_size "$PART6"

################################################################################
# PART 7: Persistence + Multimodal
################################################################################
echo -e "${GREEN}Creating Part 7: Persistence + Multimodal${NC}"
PART7="$OUTPUT_DIR/part_7_of_9.txt"
create_part_header "7" "$PART7" "Persistence + Multimodal"

add_section "$PART7" "$BASE_DIR/06_persistence/02_gguf_interoperability.md" "6.P2 GGUF Interoperability"
add_section "$PART7" "$BASE_DIR/06_persistence/03_identity_personality.md" "6.P3 Identity & Personality"
add_section "$PART7" "$BASE_DIR/06_persistence/04_nap_system.md" "6.P4 NAP System"
add_section "$PART7" "$BASE_DIR/07_multimodal/01_cymatic_transduction.md" "7.1 Cymatic Transduction"
add_section "$PART7" "$BASE_DIR/07_multimodal/02_audio_resonance.md" "7.2 Audio Resonance"
add_section "$PART7" "$BASE_DIR/07_multimodal/03_visual_cymatics.md" "7.3 Visual Cymatics"

report_size "$PART7"

################################################################################
# PART 8: Implementation Details + Protocols
################################################################################
echo -e "${GREEN}Creating Part 8: Implementation Details + Protocols${NC}"
PART8="$OUTPUT_DIR/part_8_of_9.txt"
create_part_header "8" "$PART8" "Implementation Details + Protocols"

# Phase 0 Requirements
if [[ -d "$BASE_DIR/08_phase_0_requirements" ]]; then
    for file in "$BASE_DIR/08_phase_0_requirements"/*.md; do
        if [[ -f "$file" && "$file" != *".backup"* ]]; then
            filename=$(basename "$file" .md)
            add_section "$PART8" "$file" "8.P0 Phase 0: $filename"
        fi
    done
fi

# Implementation
if [[ -d "$BASE_DIR/09_implementation" ]]; then
    for file in "$BASE_DIR/09_implementation"/*.md; do
        if [[ -f "$file" && "$file" != *".backup"* ]]; then
            filename=$(basename "$file" .md)
            add_section "$PART8" "$file" "9.I $filename"
        fi
    done
fi

# Protocols
if [[ -d "$BASE_DIR/10_protocols" ]]; then
    for file in "$BASE_DIR/10_protocols"/*.md; do
        if [[ -f "$file" && "$file" != *".backup"* ]]; then
            filename=$(basename "$file" .md)
            add_section "$PART8" "$file" "10.P $filename"
        fi
    done
fi

report_size "$PART8"

################################################################################
# PART 9: Appendices
################################################################################
echo -e "${GREEN}Creating Part 9: Appendices${NC}"
PART9="$OUTPUT_DIR/part_9_of_9.txt"
create_part_header "9" "$PART9" "Appendices"

if [[ -d "$BASE_DIR/11_appendices" ]]; then
    for file in "$BASE_DIR/11_appendices"/*.md; do
        if [[ -f "$file" && "$file" != *".backup"* ]]; then
            filename=$(basename "$file" .md)
            add_section "$PART9" "$file" "11.A $filename"
        fi
    done
fi

report_size "$PART9"

################################################################################
# Summary Report
################################################################################
echo ""
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}Compilation Complete!${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
echo "Output Directory: $OUTPUT_DIR"
echo ""
echo "File Sizes:"
for i in {1..9}; do
    file="$OUTPUT_DIR/part_${i}_of_9.txt"
    if [[ -f "$file" ]]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        size_kb=$((size / 1024))
        if [[ $size -gt $MAX_SIZE ]]; then
            echo -e "  Part $i: ${YELLOW}${size_kb}KB (WARNING: Exceeds 500KB)${NC}"
        else
            echo -e "  Part $i: ${GREEN}${size_kb}KB${NC}"
        fi
    fi
done
echo ""
echo "Total Files Created: 9"
echo ""
echo -e "${BLUE}Ready for Gemini Deep Research Analysis${NC}"
echo ""
