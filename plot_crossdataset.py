"""
Plot cross-dataset transfer results: trained on DataCo (frac=1.00),
evaluated zero-shot on GlobalStore and OAS.

Reads eval_{dataset}.log files from each model subdir.
Produces per-model plots and an all-models summary plot.
Saves plots and a txt table to OUT_DIR.

Usage: python3 plot_crossdataset.py [--GPU 0]
"""

import os
import re
import glob
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--GPU", type=int, default=0)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

# ── Hardcoded paths ───────────────────────────────────────────────────────────

RUN_DIR = "output/decision_maker/cross_dataset/20260601_115907"
OUT_DIR = "output/decision_maker/comparisons/cross_dataset"

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "qwen3-0.6B", "qwen3-1.7B", "qwen3-4B"]
MODEL_LABELS = {
    "gpt2":        "GPT-2",
    "gpt2-medium": "GPT-2 Med",
    "gpt2-large":  "GPT-2 Lg",
    "qwen3-0.6B":  "Qwen3-0.6B",
    "qwen3-1.7B":  "Qwen3-1.7B",
    "qwen3-4B":    "Qwen3-4B",
}
MODEL_COLORS = {
    "gpt2":        "#90A4AE",
    "gpt2-medium": "#546E7A",
    "gpt2-large":  "#263238",
    "qwen3-0.6B":  "#81D4FA",
    "qwen3-1.7B":  "#0288D1",
    "qwen3-4B":    "#01579B",
}

EVAL_DATASETS = ["globalstore", "oas"]
DATASET_LABELS = {"globalstore": "GlobalStore", "oas": "OAS"}
DATASET_COLORS = {"globalstore": "#FF5722", "oas": "#4CAF50"}

METRICS = [
    ("profit",  "Profit"),
    ("on_time", "On-Time Ratio"),
    ("sum",     "Profit + On-Time"),
]

# ── Parsing ───────────────────────────────────────────────────────────────────

LOG_PAT = re.compile(r"eval_([a-z]+)\.log$")

def parse_log(path):
    metrics = {}
    for key, pat in [
        ("profit",  re.compile(r"best_profit[=\s]+([\d.]+)")),
        ("on_time", re.compile(r"best_on_time[=\s]+([\d.]+)")),
    ]:
        with open(path) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    metrics[key] = float(m.group(1))
    return metrics if len(metrics) == 2 else None


def load_results(run_dir):
    """Returns dict: model → dataset → {profit, on_time, sum}"""
    results = {}
    for model in MODEL_ORDER:
        model_dir = os.path.join(run_dir, model)
        if not os.path.isdir(model_dir):
            continue
        model_results = {}
        for path in glob.glob(os.path.join(model_dir, "eval_*.log")):
            m = LOG_PAT.match(os.path.basename(path))
            if not m:
                continue
            ds = m.group(1)
            metrics = parse_log(path)
            if metrics is None:
                print(f"  [skip] {model}/{os.path.basename(path)} — incomplete")
                continue
            metrics["sum"] = metrics["profit"] + metrics["on_time"]
            model_results[ds] = metrics
        if model_results:
            results[model] = model_results
    return results


def sorted_models(model_keys):
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(model_keys, key=lambda m: order.get(m, 99))

# ── Text table ────────────────────────────────────────────────────────────────

def save_table(results):
    col_w = 10
    active_models = sorted_models(results.keys())
    lines = []

    for ds in EVAL_DATASETS:
        lines.append(f"\n{'='*70}")
        lines.append(f"Eval Dataset: {DATASET_LABELS[ds]}  (trained on DataCo frac=1.00)")
        lines.append("=" * 70)
        header = f"{'Model':<16}" + \
                 f"  {'Profit':>{col_w}} {'OnTime':>{col_w}} {'Sum':>{col_w}}"
        lines.append(header)
        lines.append("-" * len(header))
        for model in active_models:
            r = results[model].get(ds)
            label = MODEL_LABELS.get(model, model)
            if r:
                lines.append(f"{label:<16}  {r['profit']:>{col_w}.4f} "
                              f"{r['on_time']:>{col_w}.4f} {r['sum']:>{col_w}.4f}")
            else:
                lines.append(f"{label:<16}  {'—':>{col_w}} {'—':>{col_w}} {'—':>{col_w}}")

    table = "\n".join(lines)
    print(table)
    path = os.path.join(OUT_DIR, "cross_dataset_results.txt")
    with open(path, "w") as f:
        f.write(table + "\n")
    print(f"\nSaved: {path}")

# ── Plot 1: Per model — 3 metric subplots, bars per eval dataset ──────────────

def plot_per_model(results):
    """One figure per model: 3 metric subplots, one bar per eval dataset."""
    active_models = sorted_models(results.keys())
    x = np.arange(len(EVAL_DATASETS))

    for model in active_models:
        model_label = MODEL_LABELS.get(model, model)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for ax, (metric, metric_label) in zip(axes, METRICS):
            vals   = [results[model].get(ds, {}).get(metric, 0.0) for ds in EVAL_DATASETS]
            colors = [DATASET_COLORS[ds] for ds in EVAL_DATASETS]
            bars   = ax.bar(x, vals, color=colors, edgecolor="white", width=0.5)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{v:.4f}", ha="center", va="bottom", fontsize=8)
            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([DATASET_LABELS[ds] for ds in EVAL_DATASETS], fontsize=10)
            ax.set_ylabel(metric_label if metric == "profit" else "")
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(bottom=0)

        fig.suptitle(f"{model_label} — Cross-Dataset Transfer (trained on DataCo)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        safe = re.sub(r"[^a-z0-9]+", "_", model.lower())
        path = os.path.join(OUT_DIR, f"cross_dataset_{safe}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

# ── Plot 2: All models — grouped bars per eval dataset, 3 metric subplots ─────

def plot_all_models(results):
    """One figure: 3 metric subplots, grouped bars = models, one group per eval dataset."""
    active_models = sorted_models(results.keys())
    n_models = len(active_models)
    n_datasets = len(EVAL_DATASETS)
    x = np.arange(n_datasets)
    width = 0.8 / n_models

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (metric, metric_label) in zip(axes, METRICS):
        for i, model in enumerate(active_models):
            vals = [results[model].get(ds, {}).get(metric, 0.0) for ds in EVAL_DATASETS]
            offset = (i - (n_models - 1) / 2) * width
            ax.bar(x + offset, vals, width * 0.9,
                   label=MODEL_LABELS.get(model, model),
                   color=MODEL_COLORS.get(model, "#555"),
                   alpha=0.85, edgecolor="white")

        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[ds] for ds in EVAL_DATASETS], fontsize=10)
        ax.set_ylabel(metric_label if metric == "profit" else "")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("Cross-Dataset Transfer — All Models (trained on DataCo frac=1.00)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cross_dataset_all_models.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.isdir(RUN_DIR):
        print(f"Error: run dir not found: {RUN_DIR}")
        raise SystemExit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Run dir : {RUN_DIR}")
    print(f"Out dir : {OUT_DIR}")

    results = load_results(RUN_DIR)
    if not results:
        print("No completed eval logs found.")
        raise SystemExit(1)

    print(f"Models with results: {sorted_models(results.keys())}")

    save_table(results)
    plot_per_model(results)
    plot_all_models(results)
    print("\nDone.")
