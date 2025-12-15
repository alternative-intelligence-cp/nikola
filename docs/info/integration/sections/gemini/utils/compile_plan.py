#!/usr/bin/env python3
"""
Helper script to compile the Nikola plan into 9 files for Gemini research.
Automatically distributes sections across files and updates section numbers.
"""

import os
from pathlib import Path
from typing import List, Tuple

# Base paths
SECTIONS_DIR = Path(__file__).parent.parent.parent
COMPILATION_DIR = Path(__file__).parent.parent / "compilation"
SPECS_FILE = Path(__file__).parent.parent.parent.parent.parent / "specs.txt"

# Section directories to include (in order)
SECTION_DIRS = [
    "00_front_matter",
    "01_executive",
    "02_foundations",
    "03_cognitive_systems",
    "04_infrastructure",
    "05_autonomous_systems",
    "06_implementation_specifications",
    "06_persistence",
    "07_multimodal",
    "08_phase_0_requirements",
    "09_implementation",
    "10_protocols",
    "11_appendices",
]


def get_all_markdown_files() -> List[Tuple[str, Path]]:
    """Get all markdown files from sections in order."""
    files = []
    for section_dir in SECTION_DIRS:
        section_path = SECTIONS_DIR / section_dir
        if section_path.exists():
            md_files = sorted(section_path.glob("*.md"))
            for md_file in md_files:
                files.append((section_dir, md_file))
    return files


def read_file_content(file_path: Path) -> str:
    """Read the content of a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def compile_plan():
    """Main compilation function."""
    print("Starting plan compilation...")

    # Create compilation directory if it doesn't exist
    COMPILATION_DIR.mkdir(parents=True, exist_ok=True)

    # Get all files
    all_files = get_all_markdown_files()
    total_files = len(all_files)

    print(f"Found {total_files} markdown files")

    # Calculate files per compilation (approximately)
    files_per_comp = (total_files + 8) // 9  # Ceiling division for 9 files

    # Read table of contents and specs
    toc_path = SECTIONS_DIR / "00_front_matter" / "01_table_of_contents.md"
    toc_content = read_file_content(toc_path)
    specs_content = read_file_content(SPECS_FILE)

    # Create compilation files
    current_comp = 1
    current_file_count = 0
    current_content = []

    # Start first file with TOC and specs
    current_content.append("# NIKOLA MODEL IMPLEMENTATION PLAN - PART 1 OF 9\n\n")
    current_content.append("---\n\n")
    current_content.append(toc_content)
    current_content.append("\n\n---\n\n# SPECIFICATIONS\n\n")
    current_content.append(specs_content)
    current_content.append("\n\n---\n\n")

    # Section mapping for TOC reference
    section_map = {
        "01_executive/01_executive_summary.md": "SECTION 1.1",
        "02_foundations/01_9d_toroidal_geometry.md": "SECTION 2.1",
        "02_foundations/02_wave_interference_physics.md": "SECTION 2.2",
        "02_foundations/03_balanced_nonary_logic.md": "SECTION 2.3",
        "03_cognitive_systems/01_wave_interference_processor.md": "SECTION 3.1",
        "03_cognitive_systems/02_mamba_9d_ssm.md": "SECTION 3.2",
        "03_cognitive_systems/03_neuroplastic_transformer.md": "SECTION 3.3",
        "03_cognitive_systems/04_memory_data_systems.md": "SECTION 3.4",
        "04_infrastructure/01_zeromq_spine.md": "SECTION 4.1",
        "04_infrastructure/02_orchestrator_router.md": "SECTION 4.2",
        "04_infrastructure/03_external_tool_agents.md": "SECTION 4.3",
        "04_infrastructure/04_executor_kvm.md": "SECTION 4.4",
        "05_autonomous_systems/01_computational_neurochemistry.md": "SECTION 5.1",
        "05_autonomous_systems/02_training_systems.md": "SECTION 5.2",
        "05_autonomous_systems/03_ingestion_pipeline.md": "SECTION 5.3",
        "05_autonomous_systems/04_self_improvement.md": "SECTION 5.4",
        "05_autonomous_systems/05_security_systems.md": "SECTION 5.5",
        "06_persistence/01_dmc_persistence.md": "SECTION 6.1",
        "06_persistence/02_gguf_interoperability.md": "SECTION 6.2",
        "06_persistence/03_identity_personality.md": "SECTION 6.3",
        "06_persistence/04_nap_system.md": "SECTION 6.4",
        "07_multimodal/01_cymatic_transduction.md": "SECTION 7.1",
        "07_multimodal/02_audio_resonance.md": "SECTION 7.2",
        "07_multimodal/03_visual_cymatics.md": "SECTION 7.3",
        "09_implementation/01_file_structure.md": "SECTION 8.1",
        "09_implementation/02_development_roadmap.md": "SECTION 8.2",
        "09_implementation/03_implementation_checklist.md": "SECTION 8.3",
        "09_implementation/04_build_deployment.md": "SECTION 8.4",
        "10_protocols/01_rcis_specification.md": "SECTION 9.1",
        "10_protocols/02_cli_controller.md": "SECTION 9.2",
    }

    for section_dir, file_path in all_files:
        # Skip TOC as we already added it
        if "table_of_contents" in file_path.name:
            continue

        content = read_file_content(file_path)

        # Build file reference key
        file_key = f"{section_dir}/{file_path.name}"

        # Add prominent section header with TOC reference
        section_ref = section_map.get(file_key, "")
        if section_ref:
            current_content.append(f"\n\n{'='*80}\n")
            current_content.append(
                f"{section_ref}: {file_path.stem.replace('_', ' ').title()}\n"
            )
            current_content.append(f"{'='*80}\n")

        # Add section separator and file info
        current_content.append(f"\n<!-- SOURCE: {section_dir}/{file_path.name} -->\n\n")
        current_content.append(content)

        current_file_count += 1

        # Check if we should move to next compilation file
        if current_file_count >= files_per_comp and current_comp < 9:
            # Write current compilation
            output_file = COMPILATION_DIR / f"part_{current_comp}_of_9.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("".join(current_content))
            print(f"Created {output_file.name}")

            # Start next compilation
            current_comp += 1
            current_file_count = 0
            current_content = [
                f"# NIKOLA MODEL IMPLEMENTATION PLAN - PART {current_comp} OF 9\n\n"
            ]
            current_content.append("---\n\n")

    # Write final compilation file
    if current_content:
        output_file = COMPILATION_DIR / f"part_{current_comp}_of_9.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("".join(current_content))
        print(f"Created {output_file.name}")

    print(f"\nCompilation complete! Created {current_comp} files in {COMPILATION_DIR}")
    print("\nGenerated files:")
    for i in range(1, current_comp + 1):
        file_path = COMPILATION_DIR / f"part_{i}_of_9.txt"
        size_kb = file_path.stat().st_size / 1024
        print(f"  - part_{i}_of_9.txt ({size_kb:.1f} KB)")


if __name__ == "__main__":
    compile_plan()
