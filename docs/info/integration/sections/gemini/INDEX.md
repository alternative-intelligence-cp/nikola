# Gemini Research System - Quick Index

**Purpose:** Bug sweep preparation for Nikola implementation plan  
**Status:** ✅ READY FOR RESEARCH  
**Date:** December 11, 2025

---

## Quick Links

| Document | Purpose |
|----------|---------|
| [SUMMARY.txt](SUMMARY.txt) | Executive summary of what was done and next steps |
| [README.txt](README.txt) | Complete documentation and workflow guide |
| [gap_analysis_report.txt](gap_analysis_report.txt) | Detailed gap analysis results |

---

## Folders

### 📁 compilation/
9 compiled plan files ready for Gemini (488KB to 504KB each)
- part_1_of_9.txt through part_9_of_9.txt
- Includes TOC, specs, and all plan sections

### 📁 tasks/
15 research tasks (30 files: .txt + .json pairs)

**Critical Priority:**
- bug_sweep_001_wave_interference
- bug_sweep_002_9d_geometry  
- bug_sweep_003_nonary_encoding
- bug_sweep_011_energy_conservation
- bug_sweep_012_bootstrap

**High Priority:**
- bug_sweep_004_mamba_integration
- bug_sweep_005_transformer
- bug_sweep_006_zeromq
- bug_sweep_007_database
- bug_sweep_010_security
- bug_sweep_013_wave_text_decode
- bug_sweep_015_working_memory

**Medium Priority:**
- bug_sweep_008_engs
- bug_sweep_009_executor
- bug_sweep_014_abi_stability

### 📁 responses/
(Empty - place Gemini responses here)

### 📁 utils/
Helper scripts:
- compile_plan.py - Compile plan into 9 files
- analyze_gaps.py - Identify missing implementations
- create_task.py - Task management
- generate_bug_sweep_tasks.py - Generate research tasks
- run_compilation.sh - Quick compilation

---

## Workflow

```
1. Pick task from tasks/
2. Send to Gemini with referenced plan parts
3. Place response in responses/
4. Integrate into plan
5. Re-compile if needed
```

---

## Quick Commands

```bash
# Re-compile plan
./utils/run_compilation.sh

# Analyze gaps
python3 utils/analyze_gaps.py

# List tasks
python3 utils/create_task.py list

# Mark task complete
python3 utils/create_task.py complete bug_sweep_001_wave_interference
```

---

**Start here:** [SUMMARY.txt](SUMMARY.txt) for overview or [README.txt](README.txt) for details
