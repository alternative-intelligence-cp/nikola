# Gemini Research System for Nikola Bug Sweep

This folder contains tools and workflows for preparing comprehensive research tasks for Gemini to fill gaps in the Nikola implementation plan.

## Directory Structure

```
gemini/
├── compilation/         # 9 compiled plan files (auto-generated)
│   ├── part_1_of_9.txt
│   ├── part_2_of_9.txt
│   └── ... (through part_9_of_9.txt)
├── tasks/              # Research task files for Gemini
│   ├── bug_sweep_001_wave_interference.txt
│   ├── bug_sweep_001_wave_interference.json
│   └── ... (15 tasks total)
├── responses/          # Gemini's research responses (place here)
├── utils/              # Helper scripts
│   ├── compile_plan.py
│   ├── create_task.py
│   ├── analyze_gaps.py
│   ├── generate_bug_sweep_tasks.py
│   └── run_compilation.sh
├── gap_analysis_report.txt  # Automated gap analysis (auto-generated)
└── README.txt          # This file
```

## Quick Start

### 1. Compile the Plan

The plan is automatically split into 9 files for Gemini, each with the table of contents and specifications:

```bash
cd utils
./run_compilation.sh
```

Or directly:
```bash
python3 utils/compile_plan.py
```

This creates 9 files in `compilation/` ranging from ~70KB to ~500KB each.

### 2. Analyze Gaps

Run the gap analysis to identify missing implementations:

```bash
python3 utils/analyze_gaps.py
```

This generates `gap_analysis_report.txt` with:
- Total flagged lines (TODOs, missing items, placeholders)
- Missing implementation areas
- Detailed findings per file

### 3. Generate Research Tasks

Create structured research tasks for Gemini:

```bash
python3 utils/generate_bug_sweep_tasks.py
```

This creates 15 research tasks in `tasks/`:
- **5 Critical priority** (Wave interference, 9D geometry, nonary encoding, energy conservation, bootstrap)
- **7 High priority** (Mamba, transformer, ZeroMQ, database, security, working memory, wave decoding)
- **3 Medium priority** (ENGS, executor, ABI stability)

## Workflow

### For Randy:

1. **Run compilation** when the plan is updated
2. **Review gap analysis** to understand current state
3. **Select tasks** to send to Gemini (start with critical priority)
4. **Send to Gemini** with the compiled plan parts referenced in each task
5. **Place responses** in `responses/` folder with matching task ID name
6. **Integrate findings** back into the plan

### Task File Format

Each task includes:
- **Problem Statement**: Clear description of what's missing
- **Deliverables**: 3 or fewer specific outputs (Gemini quality degrades above 3)
- **Relevant Files**: Specific plan parts and sections to reference
- **Instructions**: Guidance to avoid ineffective searches

Example:
```
tasks/bug_sweep_001_wave_interference.txt  (Human readable)
tasks/bug_sweep_001_wave_interference.json (Machine readable)
```

### Response Workflow

When Gemini completes a task:
1. Save response as `responses/bug_sweep_001_wave_interference.txt`
2. Review for completeness and accuracy
3. Integrate findings into the appropriate plan sections
4. Mark task complete: `python3 utils/create_task.py complete bug_sweep_001_wave_interference`
5. Re-run compilation if plan changed

## Current Status

### Compilation
✅ 9 files generated successfully
✅ Table of contents included in part 1
✅ Specifications included in part 1
✅ All 54 plan sections distributed

### Gap Analysis
✅ 83 flagged lines identified
✅ 21 missing implementation areas found
✅ Detailed report generated

### Tasks Generated
✅ 15 research tasks created
✅ Organized by priority
✅ References to specific plan locations included

## Helper Scripts

### compile_plan.py
- Reads all plan sections from `../sections/`
- Distributes evenly across 9 files
- Includes TOC and specs in part 1
- Adds source file markers

### create_task.py
- Template for creating new research tasks
- Usage: Import and call `create_task()` function
- Creates both .txt and .json formats
- Includes task management: list, complete

### analyze_gaps.py
- Scans compiled files for gaps
- Identifies TODO, FIXME, placeholder patterns
- Checks implementation completeness
- Generates comprehensive report

### generate_bug_sweep_tasks.py
- Uses gap analysis to create tasks
- Organizes by priority and area
- Groups related gaps into single tasks
- Limits deliverables to 3 per task

## Tips for Working with Gemini

1. **Always reference specific files** - Don't let Gemini waste time searching
2. **Limit to 3 deliverables** - Quality degrades above this
3. **Provide context** - The compilation includes specs and TOC for reference
4. **Be specific** - The more precise the problem statement, the better the response
5. **Iterate** - If response is incomplete, create follow-up task with narrower scope

## Maintenance

### When Plan Updates:
```bash
cd utils
./run_compilation.sh
python3 analyze_gaps.py
# Review gap_analysis_report.txt
# Create new tasks as needed with create_task.py
```

### To Create Custom Task:
```python
from utils.create_task import create_task

create_task(
    task_id="custom_001",
    title="Your Task Title",
    problem_statement="What needs solving...",
    deliverables=[
        "Deliverable 1",
        "Deliverable 2", 
        "Deliverable 3"
    ],
    relevant_files=[
        {"file": "part_X_of_9.txt", "description": "Section Y.Z - Component Name"}
    ],
    priority="high"  # critical/high/medium/low
)
```

## Notes

- All helper scripts are designed to be re-runnable without modification
- Compilation preserves source file references for traceability
- JSON files enable programmatic task management
- The system is optimized for Gemini's strengths and limitations

---

**Last Updated:** 2025-12-11
**Total Tasks:** 15 (5 critical, 7 high, 3 medium)
**Status:** Ready for Gemini research
