

#!/bin/bash
set -euo pipefail

# Nikola Documentation Renumbering Script
# Fixes the section numbering chaos in docs/info/integration/sections/
# Created: 2025-01-08

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🔍 DRY RUN MODE - No changes will be made"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="renumber_log_${TIMESTAMP}.txt"
BASE_DIR="docs/info/integration/sections"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo "$1" | tee -a "$LOG_FILE"
}

log_color() {
    echo -e "${2}${1}${NC}" | tee -a "$LOG_FILE"
}


create_backup() {
    log_color "\n📦 Phase 1: Creating Safety Backup" "$BLUE"
    
    if [ "$DRY_RUN" = true ]; then
        log "[DRY RUN] Would create backup: sections_backup_${TIMESTAMP}.tar.gz"
        return
    fi
    
    log "Creating backup of entire sections directory..."
    tar -czf "sections_backup_${TIMESTAMP}.tar.gz" "$BASE_DIR"
    
    if [ $? -eq 0 ]; then
        log_color "✅ Backup created: sections_backup_${TIMESTAMP}.tar.gz" "$GREEN"
    else
        log_color "❌ Backup failed! Aborting." "$RED"
        exit 1
    fi
}


archive_backups() {
    log_color "\n🗄️  Phase 2: Archiving .backup Files" "$BLUE"
    
    ARCHIVE_DIR="${BASE_DIR}/archive_${TIMESTAMP}"
    
    if [ "$DRY_RUN" = true ]; then
        local count=$(find "$BASE_DIR" -type f -name "\*.backup\*" | wc -l)
        log "[DRY RUN] Would move $count backup files to: $ARCHIVE_DIR"
        return
    fi
    
    mkdir -p "$ARCHIVE_DIR"
    
    local moved=0
    while IFS= read -r -d '' file; do
        mv "$file" "$ARCHIVE_DIR/"
        ((moved++))
    done < <(find "$BASE_DIR" -type f -name "\*.backup\*" -print0)
    
    log_color "✅ Moved $moved backup files to archive" "$GREEN"
}


rename_folders() {
    log_color "\n🔄 Phase 3: Renaming Folders" "$BLUE"
    
    cd "$BASE_DIR"
    
    # Rename in reverse order to avoid conflicts
    declare -a renames=(
        "11_appendices:10_appendices"
        "10_protocols:09_protocols"
        "09_implementation:08_implementation"
    )
    
    for rename in "${renames[@]}"; do
        IFS=':' read -r old new <<< "$rename"
        
        if [ "$DRY_RUN" = true ]; then
            if [ -d "$old" ]; then
                log "[DRY RUN] Would rename: $old → $new"
            else
                log_color "⚠️  Directory not found: $old" "$YELLOW"
            fi
        else
            if [ -d "$old" ]; then
                mv "$old" "$new"
                log_color "✅ Renamed: $old → $new" "$GREEN"
            else
                log_color "⚠️  Directory not found: $old" "$YELLOW"
            fi
        fi
    done
    
    cd - > /dev/null
}


fix_references() {
    log_color "\n🔗 Phase 4: Fixing Cross-References" "$BLUE"
    
    declare -a replacements=(
        "09_implementation:08_implementation"
        "10_protocols:09_protocols"
        "11_appendices:10_appendices"
    )
    
    local total_files=0
    
    for replacement in "${replacements[@]}"; do
        IFS=':' read -r old new <<< "$replacement"
        
        log "\nSearching for references to: $old"
        
        while IFS= read -r -d '' file; do
            if grep -q "$old" "$file"; then
                if [ "$DRY_RUN" = true ]; then
                    log "[DRY RUN] Would update: $file"
                else
                    # Use sed for in-place replacement
                    if [[ "$OSTYPE" == "darwin"\* ]]; then
                        # macOS
                        sed -i '' "s|$old|$new|g" "$file"
                    else
                        # Linux
                        sed -i "s|$old|$new|g" "$file"
                    fi
                    log "  ✓ Updated: $file"
                fi
                ((total_files++))
            fi
        done < <(find "$BASE_DIR" -name "\*.md" -type f -print0)
    done
    
    if [ "$DRY_RUN" = true ]; then
        log_color "\n[DRY RUN] Would update $total_files files" "$YELLOW"
    else
        log_color "\n✅ Updated references in $total_files files" "$GREEN"
    fi
}


fix_toc() {
    log_color "\n📑 Phase 5: Fixing Table of Contents" "$BLUE"
    
    TOC_FILE="${BASE_DIR}/00_front_matter/01_table_of_contents.md"
    
    if [ ! -f "$TOC_FILE" ]; then
        log_color "⚠️  TOC file not found: $TOC_FILE" "$YELLOW"
        return
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log "[DRY RUN] Would update TOC: $TOC_FILE"
        return
    fi
    
    # Fix the TOC references
    if [[ "$OSTYPE" == "darwin"\* ]]; then
        sed -i '' 's|09_implementation|08_implementation|g' "$TOC_FILE"
        sed -i '' 's|10_protocols|09_protocols|g' "$TOC_FILE"
        sed -i '' 's|11_appendices|10_appendices|g' "$TOC_FILE"
    else
        sed -i 's|09_implementation|08_implementation|g' "$TOC_FILE"
        sed -i 's|10_protocols|09_protocols|g' "$TOC_FILE"
        sed -i 's|11_appendices|10_appendices|g' "$TOC_FILE"
    fi
    
    log_color "✅ Updated Table of Contents" "$GREEN"
}


main() {
    log_color "\n╔════════════════════════════════════════════╗" "$BLUE"
    log_color "║  Nikola Documentation Renumbering Script  ║" "$BLUE"
    log_color "╚════════════════════════════════════════════╝" "$BLUE"
    log ""
    log "Log file: $LOG_FILE"
    log "Base directory: $BASE_DIR"
    log ""
    
    # Check if base directory exists
    if [ ! -d "$BASE_DIR" ]; then
        log_color "❌ Error: Directory not found: $BASE_DIR" "$RED"
        log "Please run this script from the nikola repo root"
        exit 1
    fi
    
    # Execute phases
    create_backup
    archive_backups
    rename_folders
    fix_references
    fix_toc
    
    # Summary
    log_color "\n╔════════════════════════════════════════════╗" "$GREEN"
    if [ "$DRY_RUN" = true ]; then
        log_color "║         DRY RUN COMPLETE                   ║" "$GREEN"
        log_color "║  Review output and run without --dry-run  ║" "$GREEN"
    else
        log_color "║      RENUMBERING COMPLETE! ✨              ║" "$GREEN"
        log_color "║                                            ║" "$GREEN"
        log_color "║  Backup: sections_backup_${TIMESTAMP}.tar.gz" "$GREEN"
        log_color "║  Rollback: tar -xzf sections_backup_\*.gz   ║" "$GREEN"
    fi
    log_color "╚════════════════════════════════════════════╝" "$GREEN"
    log ""
    log "Full log saved to: $LOG_FILE"
}

# Run main function
main "$@"
