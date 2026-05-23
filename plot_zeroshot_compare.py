"""
Compare zero-shot results across models from one or more run directories.

Usage:
    python3 plot_zeroshot_compare.py <dir1> [<dir2> ...]

Example:
    python3 plot_zeroshot_compare.py \\
        output/latest_output/zero_shot/20260521_130031 \\
        output/latest_output/zero_shot/20260521_140045

Expects log files named {dataset}_{model_tag}.log inside each directory.
Saves plots and a summary table in the first directory provided.
"""

import sys
import os
import re
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_log(path):
    metrics = {}
    patterns = {
        "profit":   re.compile(r"best_profit=([\d.]+)"),
        "on_time":  re.compile(r"best_on_time=([\d.]+)"),
        "accuracy": re.compile(r"best_dm_accuracy_true=([\d.]+)"),
    }
    with open(path) as f:
        for line in f:
            for key, pat in patterns.items():
                m = pat.search(line)
                if m:
                    metrics[key] = float(m.group(1))
    return metrics if len(metrics) == 3 else None


LOG_PAT = re.compile(r"([a-z]+)_([a-zA-Z0-9.\-]+)\.log$")

def collect_results(dirs):
    """Returns dict: dataset → model_tag → {profit, on_time, accuracy}"""
    results = defaultdict(dict)
    for d in dirs:
        for path in glob.glob(os.path.join(d, "*.log")):
            fname = os.path.basename(path)
            m = LOG_PAT.match(fname)
            if not m:
                continue
            dataset, model_tag = m.group(1), m.group(2)
            metrics = parse_log(path)
            if metrics is None:
                print(f"  [skip] {fname} — incomplete")
                continue
            results[dataset][model_tag] = metrics
    return results


# ── Layout ────────────────────────────────────────────────────────────────────

DATASET_LABELS = {
    "dataco":      "DataCo",
    "globalstore": "GlobalStore",
    "oas":         "OAS",
}

MODEL_ORDER = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "qwen3-0.6B",
    "qwen3-1.7B",
    "qwen3-4B",
]

MODEL_LABELS = {
    "gpt2":        "GPT-2",
    "gpt2-medium": "GPT-2 Med",
    "gpt2-large":  "GPT-2 Lg",
    "qwen3-0.6B":  "Qwen3-0.6B",
    "qwen3-1.7B":  "Qwen3-1.7B",
    "qwen3-4B":    "Qwen3-4B",
}

COLORS = [
    "#90A4AE", "#546E7A", "#263238",   # GPT-2 shades
    "#81D4FA", "#0288D1", "#01579B",   # Qwen3 shades
]

METRICS = [
    ("profit",   "Profit"),
    ("on_time",  "On-Time Ratio"),
    ("sum",      "Profit + On-Time"),
    ("accuracy", "Accuracy (vs Optimal)"),
]


def sorted_models(model_tags):
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(model_tags, key=lambda m: order.get(m.lower(), 99))


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_grouped_bar(results, out_dir):
    """4 subplots (one per metric), grouped bars by dataset, one bar per model."""
    datasets = sorted(results.keys())
    all_models = sorted_models({m for d in results.values() for m in d})
    n_models = len(all_models)
    x = np.arange(len(datasets))
    width = 0.8 / n_models

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (key, label) in zip(axes, METRICS):
        for i, model in enumerate(all_models):
            values = []
            for ds in datasets:
                m = results[ds].get(model)
                if m is None:
                    values.append(0.0)
                elif key == "sum":
                    values.append(m["profit"] + m["on_time"])
                else:
                    values.append(m[key])

            color = COLORS[i % len(COLORS)]
            bars = ax.bar(x + i * width - (n_models - 1) * width / 2,
                          values, width * 0.9,
                          label=MODEL_LABELS.get(model, model),
                          color=color)

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=10)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Zero-Shot Performance — All Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "zeroshot_compare.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_dataset(results, out_dir):
    """One figure per dataset: 4 metric subplots, one bar per model."""
    all_models = sorted_models({m for d in results.values() for m in d})
    x = np.arange(len(all_models))

    for ds, model_map in sorted(results.items()):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
        for ax, (key, label) in zip(axes, METRICS):
            values = []
            for model in all_models:
                m = model_map.get(model)
                if m is None:
                    values.append(0.0)
                elif key == "sum":
                    values.append(m["profit"] + m["on_time"])
                else:
                    values.append(m[key])

            bars = ax.bar(x, values, color=COLORS[:len(all_models)], edgecolor="white")
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in all_models],
                               fontsize=7, rotation=30, ha="right")
            ax.grid(axis="y", alpha=0.3)

        dset_label = DATASET_LABELS.get(ds, ds)
        fig.suptitle(f"Zero-Shot — {dset_label}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"zeroshot_compare_{ds}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def print_table(results):
    all_models = sorted_models({m for d in results.values() for m in d})
    datasets = sorted(results.keys())

    header = f"{'Model':<16}" + "".join(
        f"  {DATASET_LABELS.get(d, d):^28}" for d in datasets
    )
    subheader = f"{'':16}" + "".join(
        f"  {'Profit':>8} {'OnTime':>8} {'Acc':>8}" for _ in datasets
    )
    print("\n" + header)
    print(subheader)
    print("-" * len(subheader))

    for model in all_models:
        row = f"{MODEL_LABELS.get(model, model):<16}"
        for ds in datasets:
            m = results[ds].get(model)
            if m:
                row += f"  {m['profit']:>8.4f} {m['on_time']:>8.4f} {m['accuracy']:>8.4f}"
            else:
                row += f"  {'—':>8} {'—':>8} {'—':>8}"
        print(row)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dirs = [d.rstrip("/") for d in sys.argv[1:]]
    for d in dirs:
        if not os.path.isdir(d):
            print(f"Error: directory not found: {d}")
            sys.exit(1)

    print(f"Reading logs from: {dirs}")
    results = collect_results(dirs)

    if not results:
        print("No completed log files found.")
        sys.exit(1)

    out_dir = dirs[0]
    print_table(results)
    plot_grouped_bar(results, out_dir)
    plot_per_dataset(results, out_dir)
    print("Done.")
