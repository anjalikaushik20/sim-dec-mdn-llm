"""
Plot sample efficiency curves from a single run directory.

Usage:
    python3 plot_sample_efficiency.py <run_dir>

Example:
    python3 plot_sample_efficiency.py output/latest_output/sample_efficiency/qwen3_0.6B/20260520_171159

Expects log files named {dataset}_frac{frac}.log inside <run_dir>.
Saves plots as sample_efficiency_*.png in the same directory.
"""

import sys
import os
import re
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_log(path):
    """Extract best_profit, best_on_time, best_dm_accuracy_true from a log file.
    Returns dict with those keys, or None if the file is incomplete."""
    metrics = {}
    patterns = {
        "best_profit":           re.compile(r"best_profit=([\d.]+)"),
        "best_on_time":          re.compile(r"best_on_time=([\d.]+)"),
        "best_dm_accuracy_true": re.compile(r"best_dm_accuracy_true=([\d.]+)"),
    }
    with open(path) as f:
        for line in f:
            for key, pat in patterns.items():
                m = pat.search(line)
                if m:
                    metrics[key] = float(m.group(1))
    return metrics if len(metrics) == 3 else None


def collect_results(run_dir):
    """
    Returns dict: dataset → sorted list of (frac, profit, on_time, acc_true)
    Skips log files that are incomplete (still running or crashed).
    """
    log_pat = re.compile(r"([a-z]+)_frac([\d.]+)\.log$")
    results = defaultdict(list)

    for path in glob.glob(os.path.join(run_dir, "*.log")):
        fname = os.path.basename(path)
        m = log_pat.match(fname)
        if not m:
            continue
        dataset, frac_str = m.group(1), m.group(2)
        frac = float(frac_str)
        metrics = parse_log(path)
        if metrics is None:
            print(f"  [skip] {fname} — incomplete or still running")
            continue
        results[dataset].append((
            frac,
            metrics["best_profit"],
            metrics["best_on_time"],
            metrics["best_dm_accuracy_true"],
        ))

    for dataset in results:
        results[dataset].sort(key=lambda x: x[0])

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

DATASET_LABELS = {
    "dataco":      "DataCo",
    "globalstore": "GlobalStore",
    "oas":         "OAS",
}

COLORS = {
    "dataco":      "#2196F3",
    "globalstore": "#FF5722",
    "oas":         "#4CAF50",
}

METRICS = [
    ("best_profit",  "Profit"),
    ("best_on_time", "On-Time Ratio"),
    ("sum",          "Profit + On-Time"),
    ("acc_true",     "Accuracy (vs Optimal)"),
]


def frac_to_label(frac):
    if frac == 0.0:
        return "0\n(zero-shot)"
    pct = int(round(frac * 100))
    return f"{pct}%"


def plot_all(results, run_dir, title_suffix=""):
    if not results:
        print("No completed logs found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    metric_keys = ["best_profit", "best_on_time", "sum", "acc_true"]
    metric_labels = ["Profit", "On-Time Ratio", "Profit + On-Time", "Accuracy (vs Optimal)"]

    for ax, key, label in zip(axes, metric_keys, metric_labels):
        for dataset, rows in sorted(results.items()):
            fracs = [r[0] for r in rows]
            profit = [r[1] for r in rows]
            on_time = [r[2] for r in rows]
            acc = [r[3] for r in rows]

            if key == "best_profit":
                values = profit
            elif key == "best_on_time":
                values = on_time
            elif key == "sum":
                values = [p + o for p, o in zip(profit, on_time)]
            else:
                values = acc

            color = COLORS.get(dataset, None)
            dset_label = DATASET_LABELS.get(dataset, dataset)
            ax.plot(fracs, values, marker="o", label=dset_label, color=color, linewidth=2, markersize=6)

            # Annotate zero-shot point
            if fracs and fracs[0] == 0.0:
                ax.annotate("zero-shot", xy=(fracs[0], values[0]),
                            xytext=(8, 4), textcoords="offset points",
                            fontsize=7, color=color, alpha=0.8)

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Fraction", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # X-axis: show percentage labels
        all_fracs = sorted({r[0] for rows in results.values() for r in rows})
        ax.set_xticks(all_fracs)
        ax.set_xticklabels([frac_to_label(f) for f in all_fracs], fontsize=8)

    model_tag = os.path.basename(os.path.dirname(run_dir))
    fig.suptitle(f"Sample Efficiency — {model_tag}{title_suffix}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(run_dir, "sample_efficiency.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_dataset(results, run_dir):
    """One figure per dataset with all 4 metrics as subplots."""
    model_tag = os.path.basename(os.path.dirname(run_dir))

    for dataset, rows in sorted(results.items()):
        fracs  = [r[0] for r in rows]
        profit = [r[1] for r in rows]
        on_time = [r[2] for r in rows]
        acc    = [r[3] for r in rows]
        total  = [p + o for p, o in zip(profit, on_time)]
        color  = COLORS.get(dataset, "#555555")
        dset_label = DATASET_LABELS.get(dataset, dataset)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
        for ax, values, label in zip(axes,
                                     [profit, on_time, total, acc],
                                     ["Profit", "On-Time Ratio", "Profit + On-Time", "Accuracy (vs Optimal)"]):
            ax.plot(fracs, values, marker="o", color=color, linewidth=2, markersize=7)
            if fracs and fracs[0] == 0.0:
                ax.axvline(x=0.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.set_xticks(fracs)
            ax.set_xticklabels([frac_to_label(f) for f in fracs], fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{dset_label} — {model_tag}", fontsize=12, fontweight="bold")
        plt.tight_layout()

        out_path = os.path.join(run_dir, f"sample_efficiency_{dataset}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    run_dir = sys.argv[1].rstrip("/")
    if not os.path.isdir(run_dir):
        print(f"Error: directory not found: {run_dir}")
        sys.exit(1)

    print(f"Reading logs from: {run_dir}")
    results = collect_results(run_dir)

    if not results:
        print("No completed log files found.")
        sys.exit(1)

    datasets_found = sorted(results.keys())
    fracs_found = sorted({r[0] for rows in results.values() for r in rows})
    print(f"Datasets: {[DATASET_LABELS.get(d, d) for d in datasets_found]}")
    print(f"Fractions: {fracs_found}")

    # Combined plot (all datasets on one figure)
    plot_all(results, run_dir)

    # Per-dataset plots (one figure per dataset, 4 metric subplots)
    plot_per_dataset(results, run_dir)

    print("Done.")
