
#!/bin/bash
set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🔍 DRY RUN MODE - No files will be deleted"
fi

BASE_DIR="docs/info/integration/sections"

echo "🧹 Cleanup Backup Files Script"
echo "================================"
echo ""

if [ ! -d "$BASE_DIR" ]; then
    echo "❌ Error: Directory not found: $BASE_DIR"
    echo "Please run from nikola repo root"
    exit 1
fi

echo "Searching for .backup files in: $BASE_DIR"
echo ""

# Find all backup files
BACKUP_FILES=$(find "$BASE_DIR" -type f -name "\*.backup\*" 2>/dev/null || true)

if [ -z "$BACKUP_FILES" ]; then
    echo "✅ No backup files found! Directory is clean."
    exit 0
fi

COUNT=$(echo "$BACKUP_FILES" | wc -l)
echo "Found $COUNT backup file(s):"
echo ""
echo "$BACKUP_FILES"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would delete $COUNT files"
    exit 0
fi

read -p "Delete these $COUNT files? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Deleting backup files..."

DELETED=0
while IFS= read -r file; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "  ✓ Deleted: $file"
        ((DELETED++))
    fi
done <<< "$BACKUP_FILES"

echo ""
echo "✅ Cleanup complete! Deleted $DELETED files."
echo ""
echo "Run 'git status' to see changes"
