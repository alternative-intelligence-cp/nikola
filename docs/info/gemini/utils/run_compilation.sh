#!/bin/bash
# Helper script to run the compilation process

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=================================="
echo "Nikola Plan Compilation Helper"
echo "=================================="
echo ""

# Make Python scripts executable
chmod +x "$SCRIPT_DIR/compile_plan.py"
chmod +x "$SCRIPT_DIR/create_task.py"

# Run compilation
echo "Running plan compilation..."
python3 "$SCRIPT_DIR/compile_plan.py"

echo ""
echo "=================================="
echo "Compilation complete!"
echo ""
echo "Next steps:"
echo "  1. Review compiled files in ../compilation/"
echo "  2. Use create_task.py to generate research tasks"
echo "  3. Place Gemini responses in ../responses/"
echo "=================================="
