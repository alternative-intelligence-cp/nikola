#!/usr/bin/env bash
set -e

#directories
wdir="$(dirname "$(readlink -f "$0" )" )" #the gemini directory this script is in
specs="$wdir/specs.txt" #nikola specs
res="$wdir/responses" #gemini responses/audits
dat="$wdir/.." #engineering plan files (sections directory)

#output files - 9 partials + 1 response compilation
resFull="$res/RES_COMPILED.txt"

#split compilation output files (0-8 = 9 files total)
dat0="$wdir/0.txt" #specs + index
dat1="$wdir/1.txt"
dat2="$wdir/2.txt"
dat3="$wdir/3.txt"
dat4="$wdir/4.txt"
dat5="$wdir/5.txt"
dat6="$wdir/6.txt"
dat7="$wdir/7.txt"
dat8="$wdir/8.txt"

# Compile all response files into resFull
echo "=== Compiling response files ==="
> "$resFull"  # clear file
find "$res" -type f \( -name "*.md" -o -name "*.txt" \) -not -name "RES_COMPILED.txt" | sort | while read -r file; do
    echo -e "\n\n### FILE: ${file#$res/} ###\n" >> "$resFull"
    cat "$file" >> "$resFull"
done

# Split section files logically into 9 files
echo "=== Splitting into 9 partial files ==="
> "$dat0"
> "$dat1"
> "$dat2"
> "$dat3"
> "$dat4"
> "$dat5"
> "$dat6"
> "$dat7"
> "$dat8"

# dat0: specs + index
cat "$specs" > "$dat0"
echo -e "\n\n=== DOCUMENT INDEX ===\n" >> "$dat0"
find "$dat/00_front_matter" -type f -name "*.md" | sort | while read -r file; do
    echo -e "\n### ${file#$dat/} ###\n" >> "$dat0"
    cat "$file" >> "$dat0"
done

# dat1: Executive + Foundations (00-02)
find "$dat/01_executive" "$dat/02_foundations" -type f -name "*.md" 2>/dev/null | sort | while read -r file; do
    echo -e "\n### ${file#$dat/} ###\n" >> "$dat1"
    cat "$file" >> "$dat1"
done

# dat2: Cognitive Systems (03)
find "$dat/03_cognitive_systems" -type f -name "*.md" 2>/dev/null | sort | while read -r file; do
    echo -e "\n### ${file#$dat/} ###\n" >> "$dat2"
    cat "$file" >> "$dat2"
done

# dat3: Infrastructure (04)
find "$dat/04_infrastructure" -type f -name "*.md" 2>/dev/null | sort | while read -r file; do
    echo -e "\n### ${file#$dat/} ###\n" >> "$dat3"
    cat "$file" >> "$dat3"
done

# dat4: Autonomous Systems (05)
find "$dat/05_autonomous_systems" -type f -name "*.md" 2>/dev/null | sort | while read -r file; do
    echo -e "\n### ${file#$dat/} ###\n" >> "$dat4"
    cat "$file" >> "$dat4"
done

# dat5: Persistence + Multimodal (06-07)
find "$dat/06_persistence" "$dat/07_multimodal" -type f -name "*.md" 2>/dev/null | sort | while read -r file; do
    echo -e "\n### ${file#$dat/} ###\n" >> "$dat5"
    cat "$file" >> "$dat5"
done

# dat6: Phase 0 Requirements (08)
find "$dat/08_phase_0_requirements" -type f -name "*.md" 2>/dev/null | sort | while read -r file; do
    echo -e "\n### ${file#$dat/} ###\n" >> "$dat6"
    cat "$file" >> "$dat6"
done

# dat7: Implementation (09)
find "$dat/09_implementation" -type f -name "*.md" 2>/dev/null | sort | while read -r file; do
    echo -e "\n### ${file#$dat/} ###\n" >> "$dat7"
    cat "$file" >> "$dat7"
done

# dat8: Protocols + Appendices (10-11)
find "$dat/10_protocols" "$dat/11_appendices" -type f -name "*.md" 2>/dev/null | sort | while read -r file; do
    echo -e "\n### ${file#$dat/} ###\n" >> "$dat8"
    cat "$file" >> "$dat8"
done

# Verify line counts
echo -e "\n=== Verification ==="

srcLines=$(cd "$dat" && find . -type f \( -name "*.md" -o -name "*.txt" \) ! -path "./gemini/*" -exec cat {} + | wc -l)
specsLines=$(wc -l < "$specs")
totalSrc=$((srcLines + specsLines))
splitLines=$(($(wc -l < "$dat0") + $(wc -l < "$dat1") + $(wc -l < "$dat2") + $(wc -l < "$dat3") + $(wc -l < "$dat4") + $(wc -l < "$dat5") + $(wc -l < "$dat6") + $(wc -l < "$dat7") + $(wc -l < "$dat8")))

resLines=$(find "$res" -type f \( -name "*.md" -o -name "*.txt" \) -not -name "COMPILED.txt" -exec cat {} + 2>/dev/null | wc -l || echo "0")
resCompiledLines=$(wc -l < "$resFull" 2>/dev/null || echo "0")

echo "Section sources: $totalSrc lines (specs: $specsLines + sections: $srcLines)"
echo "Split files (0-8): $splitLines lines total"
echo "  0.txt (specs+index): $(wc -l < "$dat0") lines"
echo "  1.txt (exec+found): $(wc -l < "$dat1") lines"
echo "  2.txt (cognitive): $(wc -l < "$dat2") lines"
echo "  3.txt (infra): $(wc -l < "$dat3") lines"
echo "  4.txt (autonomous): $(wc -l < "$dat4") lines"
echo "  5.txt (persist+multi): $(wc -l < "$dat5") lines"
echo "  6.txt (phase0): $(wc -l < "$dat6") lines"
echo "  7.txt (implementation): $(wc -l < "$dat7") lines"
echo "  8.txt (proto+app): $(wc -l < "$dat8") lines"
echo ""
echo "Response sources: $resLines lines"
echo "responses/RES_COMPILED.txt: $resCompiledLines lines"

echo -e "\n✅ Compilation complete!"
echo "Output files (9 partials + 1 response):"
echo "  - $dat0 through $dat8 (9 partial files)"
echo "  - $resFull (response compilation)"
