
#!/bin/bash
set -euo pipefail

BASE_DIR="docs/info/integration/sections"

RED='\03333[0;31m'
GREEN='\03333[0;32m'
YELLOW='\03333[1;33m'
BLUE='\03333[0;34m'
NC='\03333[0m'

echo -e "${BLUE}✅ Cross-Reference Verification Script${NC}"
echo "======================================="
echo ""

if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}❌ Error: Directory not found: $BASE_DIR${NC}"
    echo "Please run from nikola repo root"
    exit 1
fi

ERRORS=0

echo "Checking for old references that should have been updated..."
echo ""

# Check for old folder references
echo "[1/4] Checking for 09_implementation references..."
OLD_09=$(grep -r "09_implementation" "$BASE_DIR" --include="\*.md" 2>/dev/null || true)
if [ -n "$OLD_09" ]; then
    echo -e "${RED}❌ Found old 09_implementation references:${NC}"
    echo "$OLD_09"
    ((ERRORS++))
else
    echo -e "${GREEN}✓ No old 09_implementation references${NC}"
fi
echo ""

echo "[2/4] Checking for 10_protocols references..."
OLD_10=$(grep -r "10_protocols" "$BASE_DIR" --include="\*.md" 2>/dev/null || true)
if [ -n "$OLD_10" ]; then
    echo -e "${RED}❌ Found old 10_protocols references:${NC}"
    echo "$OLD_10"
    ((ERRORS++))
else
    echo -e "${GREEN}✓ No old 10_protocols references${NC}"
fi
echo ""

echo "[3/4] Checking for 11_appendices references..."
OLD_11=$(grep -r "11_appendices" "$BASE_DIR" --include="\*.md" 2>/dev/null || true)
if [ -n "$OLD_11" ]; then
    echo -e "${RED}❌ Found old 11_appendices references:${NC}"
    echo "$OLD_11"
    ((ERRORS++))
else
    echo -e "${GREEN}✓ No old 11_appendices references${NC}"
fi
echo ""

echo "[4/4] Checking folder structure..."
FOLDERS=$(ls -d "$BASE_DIR"/[0-9][0-9]_\* 2>/dev/null | sort)
echo "Found section folders:"
echo "$FOLDERS" | while read folder; do
    basename "$folder"
done
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✅ VERIFICATION PASSED!${NC}"
    echo -e "${GREEN}All cross-references are correct.${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}❌ VERIFICATION FAILED!${NC}"
    echo -e "${RED}Found $ERRORS issue(s) that need fixing.${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 1
fi
