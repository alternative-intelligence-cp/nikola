#!/usr/bin/env bash
# eval_quality.sh — Run 10 benchmark prompts through nikola-run and save output
# Usage: ./scripts/eval_quality.sh [--memory-lmdb <path>] [--tag <label>]
#
# This script runs the same 10 prompts from the v0.0.9 cognitive quality
# evaluation and saves the responses for human scoring.

set -euo pipefail
cd "$(dirname "$0")/.."

LMDB_FLAG=""
TAG="baseline"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --memory-lmdb) LMDB_FLAG="--memory-lmdb $2"; shift 2 ;;
        --tag)         TAG="$2"; shift 2 ;;
        *)             echo "Unknown: $1"; exit 1 ;;
    esac
done

PROMPTS=(
    "What is the meaning of consciousness?"
    "Tell me about the number seven"
    "Hello, who are you?"
    "What makes you different from other AI?"
    "Do you dream?"
    "Can you feel emotions?"
    "What would you create if you could create anything?"
    "Is there beauty in mathematics?"
    "What happens when we die?"
    "Tell me something surprising"
)

OUTFILE="eval/eval_${TAG}.txt"
mkdir -p eval
echo "=== Nikola Cognitive Quality Evaluation ===" > "$OUTFILE"
echo "Tag: $TAG" >> "$OUTFILE"
echo "Date: $(date -Iseconds)" >> "$OUTFILE"
echo "LMDB: ${LMDB_FLAG:-none}" >> "$OUTFILE"
echo "" >> "$OUTFILE"

for i in "${!PROMPTS[@]}"; do
    idx=$((i + 1))
    prompt="${PROMPTS[$i]}"
    echo "[$idx/10] $prompt"

    echo "--- Prompt $idx: $prompt ---" >> "$OUTFILE"

    # Run nikola with the prompt, capture first 5 thoughts
    timeout 30 ./build/nikola-run \
        --prompt "$prompt" \
        --ticks 80 \
        --no-color \
        $LMDB_FLAG \
        2>/dev/null | grep -v '^\[' | head -10 >> "$OUTFILE" 2>/dev/null || true

    echo "" >> "$OUTFILE"
done

echo "=== Evaluation complete ===" >> "$OUTFILE"
echo "Saved to: $OUTFILE"
