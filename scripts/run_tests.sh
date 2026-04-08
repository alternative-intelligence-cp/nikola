#!/usr/bin/env bash
# ============================================================
# scripts/run_tests.sh — Nikola test runner
#
# Runs all safe tests (unit, integration, end-to-end) and
# produces a PASS/FAIL summary with timing.
#
# Usage:
#   ./scripts/run_tests.sh            # All safe tests
#   ./scripts/run_tests.sh --quick    # Skip long-session tests
#   ./scripts/run_tests.sh --integ    # Integration only
#   ./scripts/run_tests.sh --e2e      # End-to-end only
#   ./scripts/run_tests.sh --unit     # Unit only
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Parse args
MODE="all"
SKIP_LONG=""
for arg in "$@"; do
    case "$arg" in
        --quick)      SKIP_LONG="~[longsession]" ;;
        --integ)      MODE="integ" ;;
        --e2e)        MODE="e2e" ;;
        --unit)       MODE="unit" ;;
        --help|-h)
            echo "Usage: $0 [--quick|--integ|--e2e|--unit]"
            exit 0
            ;;
    esac
done

# Ensure build dir exists
if [[ ! -d "$BUILD_DIR" ]]; then
    echo -e "${RED}ERROR: Build directory not found at $BUILD_DIR${NC}"
    echo "Run: cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

cd "$BUILD_DIR"

# ── Collect test names by category ──────────────────────────────────────────

# Known blocking tests (ZMQ socket tests that hang — see GOTCHAS.md)
EXCLUDE_PATTERN="Phase4[0-9]|Phase5[0-9]|Phase6[0-9]"

echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║       Nikola Test Suite Runner v0.0.17       ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_SKIP=0
RESULTS=()
START_TIME=$(date +%s)

run_ctest_filter() {
    local label="$1"
    local include="$2"
    local extra="${3:-}"

    echo -e "${CYAN}── ${label} ──${NC}"

    local cmd="ctest --output-on-failure --timeout 120"
    cmd+=" -R '${include}'"
    cmd+=" -E '${EXCLUDE_PATTERN}'"

    if [[ -n "$SKIP_LONG" ]]; then
        # For quick mode, exclude long-session tests
        cmd+=" -E '${EXCLUDE_PATTERN}|LongSession'"
    fi

    local t_start=$(date +%s%N)
    local output
    local rc=0
    output=$(eval "$cmd" 2>&1) || rc=$?
    local t_end=$(date +%s%N)
    local elapsed_ms=$(( (t_end - t_start) / 1000000 ))

    # Parse ctest summary line: "X% tests passed, N tests failed out of M"
    local passed=0
    local failed=0
    if echo "$output" | grep -qP '\d+% tests passed'; then
        local total_tests=$(echo "$output" | grep -oP '\d+ tests failed out of \K\d+' || echo "0")
        failed=$(echo "$output" | grep -oP '(\d+) tests? failed' | grep -oP '^\d+' || echo "0")
        passed=$((total_tests - failed))
    fi
    if [[ "$failed" -eq 0 && $rc -ne 0 ]]; then
        failed=1
    fi

    TOTAL_PASS=$((TOTAL_PASS + passed))
    TOTAL_FAIL=$((TOTAL_FAIL + failed))

    local status_color="${GREEN}"
    local status_icon="✓"
    if [[ "$failed" -gt 0 ]]; then
        status_color="${RED}"
        status_icon="✗"
    fi

    RESULTS+=("$(printf "${status_color}${status_icon}${NC} %-35s %s passed  %s  ${YELLOW}%dms${NC}" \
        "$label" "$passed" \
        "$(if [[ "$failed" -gt 0 ]]; then echo -e "${RED}${failed} failed${NC}"; else echo ""; fi)" \
        "$elapsed_ms")")

    if [[ "$failed" -gt 0 ]]; then
        echo "$output" | grep -A 5 "FAILED" || true
    fi
    echo ""
}

# ── Run Tests ───────────────────────────────────────────────────────────────

if [[ "$MODE" == "all" || "$MODE" == "unit" ]]; then
    run_ctest_filter "Unit Tests (safe phases)" "Phase[0-9]"
fi

if [[ "$MODE" == "all" || "$MODE" == "integ" ]]; then
    run_ctest_filter "Integration: Physics"  "IntegrationPhysics"
    run_ctest_filter "Integration: Cognitive" "IntegrationCognitive"
    run_ctest_filter "Integration: Autonomy"  "IntegrationAutonomy"
fi

if [[ "$MODE" == "all" || "$MODE" == "e2e" ]]; then
    run_ctest_filter "End-to-End: Pipeline" "E2EPipeline"
fi

# ── Summary ─────────────────────────────────────────────────────────────────

END_TIME=$(date +%s)
TOTAL_SECS=$((END_TIME - START_TIME))

echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${NC}"
echo -e "${BOLD}RESULTS${NC}"
echo -e "${CYAN}──────────────────────────────────────────────${NC}"

for line in "${RESULTS[@]}"; do
    echo -e "  $line"
done

echo -e "${CYAN}──────────────────────────────────────────────${NC}"

if [[ "$TOTAL_FAIL" -eq 0 ]]; then
    echo -e "  ${GREEN}${BOLD}ALL PASSED${NC}  (${TOTAL_PASS} tests, ${TOTAL_SECS}s)"
else
    echo -e "  ${RED}${BOLD}FAILURES${NC}   (${TOTAL_PASS} passed, ${TOTAL_FAIL} failed, ${TOTAL_SECS}s)"
fi

echo -e "${CYAN}══════════════════════════════════════════════${NC}"

exit $TOTAL_FAIL
