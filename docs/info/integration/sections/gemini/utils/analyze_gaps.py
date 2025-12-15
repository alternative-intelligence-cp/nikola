#!/usr/bin/env python3
"""
Script to analyze the Nikola plan and identify missing implementation details.
This script reads through all plan sections and flags potential gaps.
"""

import re
from pathlib import Path
from typing import List, Dict, Set


COMPILATION_DIR = Path(__file__).parent.parent / "compilation"


# Patterns that indicate missing information
GAP_PATTERNS = {
    "todo": r"(?i)(TODO|FIXME|TBD|TO BE DETERMINED|PLACEHOLDER)",
    "incomplete": r"(?i)(incomplete|not yet|needs? (?:to be |)(?:defined|implemented|specified|detailed))",
    "missing": r"(?i)(missing|lacking|undefined|unspecified|TBC)",
    "question": r"(?i)(\?\?\?|QUESTION:|How (?:should|do) we)",
    "stub": r"(?i)(stub|skeleton|outline only|coming soon)",
    "vague": r"(?i)(somehow|someway|details later|specifics pending)",
}


# Key implementation areas to check
IMPLEMENTATION_CHECKS = {
    "wave_interference": [
        "wave equation formulas",
        "interference calculation",
        "phase synchronization",
        "frequency modulation",
    ],
    "9d_geometry": [
        "dimension mapping",
        "toroidal coordinates",
        "spatial transformations",
        "dimension traversal",
    ],
    "balanced_nonary": [
        "encoding algorithm",
        "decoding algorithm",
        "conversion functions",
        "validation logic",
    ],
    "mamba_integration": [
        "state space equations",
        "layer implementation",
        "forward pass logic",
        "backward pass logic",
    ],
    "transformer": [
        "attention mechanism",
        "neuroplasticity update rules",
        "weight initialization",
        "training loop",
    ],
    "zeromq_spine": [
        "message protocol",
        "connection management",
        "routing logic",
        "error handling",
    ],
    "database": [
        "schema definition",
        "query interface",
        "embedding storage",
        "index structure",
    ],
    "engs": [
        "neurochemical formulas",
        "reward calculation",
        "curiosity metric",
        "boredom detection",
    ],
    "executor": [
        "sandbox implementation",
        "permission model",
        "task queue",
        "result callback",
    ],
    "security": [
        "threat detection",
        "input validation",
        "permission checks",
        "audit logging",
    ],
}


def analyze_file(file_path: Path) -> Dict:
    """Analyze a single compilation file for gaps."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    gaps = {
        "file": file_path.name,
        "flagged_lines": [],
        "missing_implementations": [],
    }

    # Check for explicit gap markers
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        for gap_type, pattern in GAP_PATTERNS.items():
            if re.search(pattern, line):
                context_start = max(0, i - 2)
                context_end = min(len(lines), i + 2)
                context = "\n".join(lines[context_start:context_end])
                gaps["flagged_lines"].append(
                    {
                        "line": i,
                        "type": gap_type,
                        "context": context,
                    }
                )

    # Check for missing implementation details
    for area, keywords in IMPLEMENTATION_CHECKS.items():
        area_mentioned = any(keyword.lower() in content.lower() for keyword in keywords)

        if area_mentioned:
            # Check if there are actual implementation details (code, formulas, etc.)
            has_code = bool(re.search(r"```[\s\S]*?```", content))
            has_formulas = bool(re.search(r"\$[\s\S]*?\$", content))
            has_algorithms = bool(
                re.search(r"(?i)(algorithm|pseudo-?code|step \d+)", content)
            )

            # Count how many keywords are actually discussed in detail
            detailed_keywords = sum(
                1
                for kw in keywords
                if len(re.findall(rf"\b{re.escape(kw)}\b", content, re.IGNORECASE)) > 2
            )

            if detailed_keywords < len(keywords) / 2:
                gaps["missing_implementations"].append(
                    {
                        "area": area,
                        "keywords": keywords,
                        "detailed_count": detailed_keywords,
                        "has_code": has_code,
                        "has_formulas": has_formulas,
                        "has_algorithms": has_algorithms,
                    }
                )

    return gaps


def generate_report(all_gaps: List[Dict]) -> str:
    """Generate a comprehensive gap report."""
    report = ["# NIKOLA PLAN GAP ANALYSIS REPORT\n"]
    report.append(f"Generated: {Path(__file__).parent}\n")
    report.append("=" * 80 + "\n\n")

    # Summary
    total_flagged = sum(len(g["flagged_lines"]) for g in all_gaps)
    total_missing = sum(len(g["missing_implementations"]) for g in all_gaps)

    report.append("## SUMMARY\n\n")
    report.append(f"- Total Flagged Lines: {total_flagged}\n")
    report.append(f"- Total Missing Implementation Areas: {total_missing}\n")
    report.append(f"- Files Analyzed: {len(all_gaps)}\n\n")

    # Detailed findings
    report.append("## DETAILED FINDINGS\n\n")

    for gaps in all_gaps:
        if gaps["flagged_lines"] or gaps["missing_implementations"]:
            report.append(f"### {gaps['file']}\n\n")

            if gaps["flagged_lines"]:
                report.append(f"**Flagged Lines: {len(gaps['flagged_lines'])}**\n\n")
                for flag in gaps["flagged_lines"][:10]:  # Limit to first 10
                    report.append(f"- Line {flag['line']} ({flag['type']}):\n")
                    report.append(f"  ```\n  {flag['context']}\n  ```\n\n")

            if gaps["missing_implementations"]:
                report.append(
                    f"**Missing Implementations: {len(gaps['missing_implementations'])}**\n\n"
                )
                for missing in gaps["missing_implementations"]:
                    report.append(
                        f"- **{missing['area'].replace('_', ' ').title()}**\n"
                    )
                    report.append(f"  - Keywords: {', '.join(missing['keywords'])}\n")
                    report.append(
                        f"  - Details found: {missing['detailed_count']}/{len(missing['keywords'])}\n"
                    )
                    report.append(
                        f"  - Has code: {'✓' if missing['has_code'] else '✗'}\n"
                    )
                    report.append(
                        f"  - Has formulas: {'✓' if missing['has_formulas'] else '✗'}\n"
                    )
                    report.append(
                        f"  - Has algorithms: {'✓' if missing['has_algorithms'] else '✗'}\n\n"
                    )

    return "".join(report)


def main():
    """Main analysis function."""
    print("Starting Nikola Plan Gap Analysis...")
    print("=" * 80)

    compilation_files = sorted(COMPILATION_DIR.glob("part_*.txt"))

    if not compilation_files:
        print("ERROR: No compilation files found!")
        print(f"Expected location: {COMPILATION_DIR}")
        return

    print(f"Found {len(compilation_files)} files to analyze\n")

    all_gaps = []

    for file_path in compilation_files:
        print(f"Analyzing {file_path.name}...", end=" ")
        gaps = analyze_file(file_path)
        all_gaps.append(gaps)
        flag_count = len(gaps["flagged_lines"])
        miss_count = len(gaps["missing_implementations"])
        print(f"({flag_count} flags, {miss_count} missing areas)")

    print("\nGenerating report...")
    report = generate_report(all_gaps)

    # Save report
    report_file = Path(__file__).parent.parent / "gap_analysis_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport saved to: {report_file}")
    print("\n" + "=" * 80)
    print("Analysis complete!")

    return all_gaps


if __name__ == "__main__":
    main()
