#!/bin/bash
wdir="$(dirname "$(readlink -f "$0")")"
OUTPUT="$wdir/nikola_plan_compiled.txt"

# Clear output file
> "$OUTPUT"

# Add header
cat >> "$OUTPUT" << 'HEADER'
================================================================================
NIKOLA MODEL v0.0.4 - COMPLETE TECHNICAL SPECIFICATION
================================================================================

Compiled Documentation - All 58 Files
Generated: December 3, 2025
Total Sections: 12 (Front Matter + Sections 1-11)

This file contains the complete technical specification for the Nikola Model
v0.0.4 - a 9D toroidal wave interference AGI system based on balanced nonary
logic.

NAVIGATION:
- Use /* path */ comments to locate specific files
- Search for section numbers (e.g., "Section 2.1") to jump to topics
- Total documentation: ~20,531 lines across 58 files

================================================================================

HEADER

# Function to append a file with header
append_file() {
    local file="$1"
    echo "" >> "$OUTPUT"
    echo "/* $file */" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
    cat "$file" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
    echo "=================================================================================" >> "$OUTPUT"
}

# Change to working directory to ensure relative paths work
cd "$wdir"

# Process all sections in order
for section_dir in sections/*/; do
    # Sort files within each section
    for file in $(ls "$section_dir"*.md 2>/dev/null | sort); do
        if [ -f "$file" ]; then
            append_file "$file"
        fi
    done
done

echo "Compilation complete: $OUTPUT"
echo "Files processed: $(grep -c '^/\* sections/' "$OUTPUT")"
