#!/usr/bin/env bash
set -e

#directories
wdir="$(dirname "$(readlink -f "$0" )" )" #the gemini directory this script is in
specs="$wdir/specs.txt" #nikola specs
res="$wdir/responses" #gemini responses/audits
dat="$wdir/.." #engineering plan files (sections directory)

#full compilation output files
datFull="$wdir/DAT_COMPILED.txt"
resFull="$res/RES_COMPILED.txt"

#split compilation output files
dat0="$wdir/0.txt" #index
dat1="$wdir/1.txt"
dat2="$wdir/2.txt"
dat3="$wdir/3.txt"
dat4="$wdir/4.txt"
dat5="$wdir/5.txt"
dat6="$wdir/6.txt"
dat7="$wdir/7.txt"

# Compile all section files into datFull, including specs
echo "=== Compiling section files ==="
> "$datFull"  # clear file
cat "$specs" >> "$datFull"
echo -e "\n\n=== SECTIONS ===\n\n" >> "$datFull"
cd "$dat" && find . -type f \( -name "*.md" -o -name "*.txt" \) ! -path "./gemini/*" | sort | while read -r file; do
    echo -e "\n\n### FILE: ${file#./} ###\n" >> "$datFull"
    cat "$file" >> "$datFull"
done
cd "$wdir"

# Compile all response files into resFull
echo "=== Compiling response files ==="
> "$resFull"  # clear file
find "$res" -type f \( -name "*.md" -o -name "*.txt" \) -not -name "RES_COMPILED.txt" | sort | while read -r file; do
    echo -e "\n\n### FILE: ${file#$res/} ###\n" >> "$resFull"
    cat "$file" >> "$resFull"
done

# Split section files logically into 8 files (index + 7 sections)
echo "=== Splitting into logical sections ==="
> "$dat0"  # index
> "$dat1"
> "$dat2"
> "$dat3"
> "$dat4"
> "$dat5"
> "$dat6"
> "$dat7"

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

# dat7: Implementation + Protocols + Appendices (09-11)
find "$dat/09_implementation" "$dat/10_protocols" "$dat/11_appendices" -type f -name "*.md" 2>/dev/null | sort | while read -r file; do
    echo -e "\n### ${file#$dat/} ###\n" >> "$dat7"
    cat "$file" >> "$dat7"
done

# Verify line counts
echo -e "\n=== Verification ==="

srcLines=$(cd "$dat" && find . -type f \( -name "*.md" -o -name "*.txt" \) ! -path "./gemini/*" -exec cat {} + | wc -l)
specsLines=$(wc -l < "$specs")
totalSrc=$((srcLines + specsLines))
compiledLines=$(wc -l < "$datFull")
splitLines=$(($(wc -l < "$dat0") + $(wc -l < "$dat1") + $(wc -l < "$dat2") + $(wc -l < "$dat3") + $(wc -l < "$dat4") + $(wc -l < "$dat5") + $(wc -l < "$dat6") + $(wc -l < "$dat7")))

resLines=$(find "$res" -type f \( -name "*.md" -o -name "*.txt" \) -not -name "COMPILED.txt" -exec cat {} + 2>/dev/null | wc -l || echo "0")
resCompiledLines=$(wc -l < "$resFull" 2>/dev/null || echo "0")

echo "Section sources: $totalSrc lines (specs: $specsLines + sections: $srcLines)"
echo "COMPILED.txt: $compiledLines lines"
echo "Split files (0-7): $splitLines lines total"
echo "  0.txt (index): $(wc -l < "$dat0") lines"
echo "  1.txt (exec+found): $(wc -l < "$dat1") lines"
echo "  2.txt (cognitive): $(wc -l < "$dat2") lines"
echo "  3.txt (infra): $(wc -l < "$dat3") lines"
echo "  4.txt (autonomous): $(wc -l < "$dat4") lines"
echo "  5.txt (persist+multi): $(wc -l < "$dat5") lines"
echo "  6.txt (phase0): $(wc -l < "$dat6") lines"
echo "  7.txt (impl+proto+app): $(wc -l < "$dat7") lines"
echo ""
echo "Response sources: $resLines lines"
echo "responses/COMPILED.txt: $resCompiledLines lines"

# Summary or error
if [ $((compiledLines - splitLines)) -gt 100 ] || [ $((splitLines - compiledLines)) -gt 100 ]; then
    echo -e "\n⚠️  WARNING: Line count mismatch between COMPILED and split files (diff: $((compiledLines - splitLines)))"
    echo "This is expected due to section headers added during splitting"
fi

echo -e "\n✅ Compilation complete!"
echo "Output files:"
echo "  - $datFull"
echo "  - $dat0 through $dat7"
echo "  - $resFull"
