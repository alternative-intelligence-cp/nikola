#!/bin/bash
# cognitive_baseline.sh — Run 10 diverse prompts and capture responses + metabolic state
# Usage: cd build && bash ../scripts/cognitive_baseline.sh > ../baseline_results.txt 2>&1

NIKOLA="./nikola-run"
COMMON="--ticks 200 --no-color --emit-all --stream"

prompts=(
    "What is the meaning of consciousness?"
    "Tell me about the number seven"
    "How do waves propagate through space?"
    "What are you thinking about right now?"
    "Explain the relationship between energy and entropy"
    "Do you dream?"
    "What is 2 + 2?"
    "Describe a beautiful pattern you can imagine"
    "Why does mathematics work?"
    "Hello, who are you?"
)

echo "========================================"
echo " Nikola Cognitive Baseline — v0.0.9"
echo " $(date)"
echo "========================================"
echo ""

for i in "${!prompts[@]}"; do
    prompt="${prompts[$i]}"
    num=$((i + 1))
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "PROMPT $num: $prompt"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "$prompt" | $NIKOLA $COMMON --profile 2>&1
    echo ""
done

echo "========================================"
echo " Baseline complete."
echo "========================================"
