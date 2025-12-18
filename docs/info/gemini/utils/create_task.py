#!/usr/bin/env python3
"""
Helper script to create research task files for Gemini.
Tasks are structured to include problem statement, deliverables, and file references.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict


TASKS_DIR = Path(__file__).parent.parent / "tasks"


def create_task(
    task_id: str,
    title: str,
    problem_statement: str,
    deliverables: List[str],
    relevant_files: List[Dict[str, str]],
    context: str = "",
    priority: str = "medium",
):
    """
    Create a research task file for Gemini.

    Args:
        task_id: Unique identifier for the task (e.g., "bug_sweep_001")
        title: Brief title of the task
        problem_statement: Clear description of what needs to be solved/researched
        deliverables: List of expected outputs (max 3 recommended)
        relevant_files: List of dicts with 'file' and 'description' keys
        context: Additional context or notes
        priority: Priority level (low/medium/high/critical)
    """
    TASKS_DIR.mkdir(parents=True, exist_ok=True)

    task_data = {
        "task_id": task_id,
        "title": title,
        "created": datetime.now().isoformat(),
        "priority": priority,
        "status": "pending",
        "problem_statement": problem_statement,
        "deliverables": deliverables,
        "relevant_files": relevant_files,
        "context": context,
        "notes": [],
    }

    # Create markdown format for Gemini
    task_content = f"""# RESEARCH TASK: {title}

**Task ID:** {task_id}
**Priority:** {priority.upper()}
**Status:** PENDING
**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## PROBLEM STATEMENT

{problem_statement}

---

## DELIVERABLES

"""

    for i, deliverable in enumerate(deliverables, 1):
        task_content += f"{i}. {deliverable}\n"

    task_content += "\n---\n\n## RELEVANT PLAN SECTIONS\n\n"

    for file_ref in relevant_files:
        task_content += f"- **{file_ref['file']}**: {file_ref['description']}\n"

    if context:
        task_content += f"\n---\n\n## ADDITIONAL CONTEXT\n\n{context}\n"

    task_content += """
---

## INSTRUCTIONS FOR GEMINI

Please provide detailed research and analysis for each deliverable listed above. 
Reference the relevant plan sections provided to ensure consistency with the 
overall architecture. If you find gaps or inconsistencies in the plan sections, 
please note them in your response.

**IMPORTANT - HOW TO FIND SECTIONS:** 
- Each compilation file has clear section markers like:
  ================================================================================
  SECTION X.Y: Component Name
  ================================================================================
- Use text search (Ctrl+F) for keywords instead of section numbers
- The files have internal numbering that differs from the TOC - IGNORE internal numbers
- Search for concepts like "Wave Interference", "Mamba", "ZeroMQ", etc.
- All relevant content is in the files listed above

**DELIVERABLES:**
- Focus on providing concrete, implementable solutions
- Include code snippets, formulas, or specifications where applicable
- Flag any assumptions you need to make
- If you can't find something, note what's missing rather than making assumptions

"""

    # Save as text file
    task_file = TASKS_DIR / f"{task_id}.txt"
    with open(task_file, "w", encoding="utf-8") as f:
        f.write(task_content)

    # Also save JSON for programmatic access
    json_file = TASKS_DIR / f"{task_id}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(task_data, f, indent=2)

    print(f"Created task: {task_file.name}")
    return task_file


def list_tasks():
    """List all pending tasks."""
    if not TASKS_DIR.exists():
        print("No tasks directory found.")
        return

    task_files = sorted(TASKS_DIR.glob("*.json"))
    if not task_files:
        print("No tasks found.")
        return

    print("\nCurrent Tasks:")
    print("-" * 80)
    for task_file in task_files:
        with open(task_file, "r", encoding="utf-8") as f:
            task = json.load(f)
        status_icon = "⏳" if task["status"] == "pending" else "✓"
        print(
            f"{status_icon} [{task['priority'].upper():8}] {task['task_id']}: {task['title']}"
        )
    print("-" * 80)


def mark_complete(task_id: str):
    """Mark a task as complete."""
    json_file = TASKS_DIR / f"{task_id}.json"
    if not json_file.exists():
        print(f"Task {task_id} not found.")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        task = json.load(f)

    task["status"] = "completed"
    task["completed"] = datetime.now().isoformat()

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2)

    print(f"Task {task_id} marked as complete.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            list_tasks()
        elif sys.argv[1] == "complete" and len(sys.argv) > 2:
            mark_complete(sys.argv[2])
        else:
            print("Usage:")
            print("  python create_task.py list                 - List all tasks")
            print("  python create_task.py complete <task_id>   - Mark task complete")
    else:
        print("Task creation template loaded. Import and use create_task() function.")
        print("\nExample:")
        print(
            """
create_task(
    task_id="bug_sweep_001",
    title="Verify Wave Interference Math",
    problem_statement="Need to verify the mathematical correctness of wave interference calculations",
    deliverables=[
        "Mathematical proof of wave interference formulas",
        "Edge case analysis",
        "Pseudo-code for implementation"
    ],
    relevant_files=[
        {"file": "part_1_of_9.md", "description": "Section 2.2 - Wave Interference Physics"},
        {"file": "part_2_of_9.md", "description": "Section 3.1 - Wave Interference Processor"}
    ],
    priority="high"
)
"""
        )
