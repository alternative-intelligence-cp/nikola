#!/usr/bin/env bash

wdir="$(dirname "$(readlink -f "$0")")"
out="$wdir/nikola_research_full.txt"

# Clear the output file if it exists
if [[ -f "$out" ]]; then
    rm "$out"
fi

# Enable nullglob: if *.txt matches nothing, it won't be passed as a literal string
shopt -s nullglob

# 1. Add files from the root directory first (r0)
# We store matches in an array to check if any exist before catting
root_files=("$wdir"/*.txt)
if [ ${#root_files[@]} -gt 0 ]; then
    cat "${root_files[@]}" >> "$out"
fi

# 2. Find and loop through 'round' directories dynamically
# -maxdepth 1: prevents looking inside sub-sub-folders
# -type d: looks only for directories
# sort -V: ensures "round10" comes after "round2" (Natural/Version sort)
find "$wdir" -maxdepth 1 -type d -name 'round[0-9]*' | sort -V | while read -r round_dir; do
    
    # Check for .txt files in this specific round folder
    round_files=("$round_dir"/*.txt)
    
    if [ ${#round_files[@]} -gt 0 ]; then
        echo "Processing: $round_dir" # Optional: useful for feedback
        cat "${round_files[@]}" >> "$out"
    fi
done

echo "Done! Combined research saved to: $out"
