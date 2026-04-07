#!/usr/bin/env python3
"""plot_metrics.py — Visualize Nikola telemetry sessions.

Reads JSON Lines from a stddbg capture file and plots gauge time series.
Requires: matplotlib (pip install matplotlib)

Usage:
    ./nikola-run --prompt "think" --telemetry --ticks 500 3>session.jsonl
    python3 scripts/plot_metrics.py session.jsonl
    python3 scripts/plot_metrics.py session.jsonl --output session.png
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_metrics(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping line {lineno}: {e}", file=sys.stderr)
    return records


def main():
    parser = argparse.ArgumentParser(description="Plot Nikola telemetry metrics")
    parser.add_argument("input", help="Path to JSON Lines metrics file")
    parser.add_argument("--output", "-o", help="Save plot to file instead of showing")
    parser.add_argument("--no-show", action="store_true", help="Don't display plot window")
    args = parser.parse_args()

    records = load_metrics(args.input)
    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    # Separate gauges by metric name
    gauges: dict[str, list[tuple[float, float]]] = defaultdict(list)
    counters: dict[str, int] = defaultdict(int)
    t0 = records[0].get("ts", 0)

    for rec in records:
        rtype = rec.get("type")
        metric = rec.get("metric", "")
        ts = rec.get("ts", 0) - t0

        if rtype == "gauge":
            gauges[metric].append((ts, rec.get("value", 0)))
        elif rtype == "counter":
            counters[metric] += rec.get("delta", 1)

    if not gauges:
        print("No gauge metrics found.", file=sys.stderr)
        sys.exit(1)

    # Plot layout: metabolic state (top), energy+entropy (middle), timing (bottom)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    metabolic = ["tick.dopamine", "tick.atp", "tick.boredom"]
    field = ["tick.energy", "tick.entropy"]
    timing = ["tick.duration"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Nikola Telemetry — {Path(args.input).name}  "
                 f"({counters.get('tick.count', len(gauges.get('tick.energy', [])))} ticks)",
                 fontsize=13)

    # Panel 1: Metabolic state
    ax = axes[0]
    for name in metabolic:
        if name in gauges:
            ts, vals = zip(*gauges[name])
            label = name.split(".")[-1]
            ax.plot(ts, vals, label=label, linewidth=0.8)
    ax.set_ylabel("Level [0–1]")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Metabolic State", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Field energy & entropy
    ax = axes[1]
    if "tick.energy" in gauges:
        ts, vals = zip(*gauges["tick.energy"])
        ax.plot(ts, vals, label="energy (J)", color="tab:red", linewidth=0.8)
    ax.set_ylabel("Energy (J)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    if "tick.entropy" in gauges:
        ax2 = ax.twinx()
        ts, vals = zip(*gauges["tick.entropy"])
        ax2.plot(ts, vals, label="entropy (nat)", color="tab:purple",
                 linewidth=0.8, linestyle="--")
        ax2.set_ylabel("Entropy (nat)")
        ax2.legend(loc="upper right", fontsize=9)
    ax.set_title("Field Energy & Entropy", fontsize=10)

    # Panel 3: Tick duration
    ax = axes[2]
    if "tick.duration" in gauges:
        ts, vals = zip(*gauges["tick.duration"])
        ax.plot(ts, vals, label="tick duration", color="tab:green", linewidth=0.8)
        ax.axhline(y=1000, color="red", linestyle=":", alpha=0.5, label="1kHz budget (1000µs)")
    ax.set_ylabel("Duration (µs)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Tick Duration", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.output}")
    if not args.no_show:
        plt.show()

    # Print summary stats
    print(f"\n--- Telemetry Summary ---")
    print(f"Total ticks: {counters.get('tick.count', 'N/A')}")
    for name in sorted(gauges.keys()):
        vals = [v for _, v in gauges[name]]
        print(f"  {name:20s}  min={min(vals):10.4f}  "
              f"max={max(vals):10.4f}  mean={sum(vals)/len(vals):10.4f}")


if __name__ == "__main__":
    main()
