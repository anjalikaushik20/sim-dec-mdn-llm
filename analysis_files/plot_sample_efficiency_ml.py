"""
Plot sample efficiency curves for ML baselines (random, historical, rf, xgb).
Auto-discovers the latest run under ML_BASE, or set ML_RUN_ID to pin a specific run.

Usage: python3 plot_sample_efficiency_ml.py [--GPU 0]
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

ML_BASE   = "output/decision_maker/ml"
ML_RUN_ID = None   # set to e.g. "20260531_120000" to pin a run; None = latest
OUT_DIR   = "output/decision_maker/comparisons/ml"

# ── Constants ─────────────────────────────────────────────────────────────────

BASELINES = ["random", "historical", "rf", "xgb"]
BASELINE_LABELS = {
    "random":    "Random",
    "historical":"Historical",
    "rf":        "Random Forest",
    "xgb":       "XGBoost",
}
BASELINE_COLORS = {
    "random":    "#B0BEC5",
    "historical":"#FF8A65",
    "rf":        "#66BB6A",
    "xgb":       "#42A5F5",
}

DATASETS = ["dataco", "globalstore", "oas"]
DATASET_LABELS = {"dataco": "DataCo", "globalstore": "GlobalStore", "oas": "OAS"}

FRACS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

METRICS = [
    ("profit",  "Profit"),
    ("on_time", "On-Time Ratio"),
    ("sum",     "Profit + On-Time"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

LOG_PAT = re.compile(r"([a-z]+)_frac([\d.]+)\.log$")

def latest_subdir(base):
    subdirs = [d for d in glob.glob(os.path.join(base, "2*")) if os.path.isdir(d)]
    return max(subdirs, key=os.path.getmtime) if subdirs else None


def parse_log(path):
    if not os.path.isfile(path):
        return None
    text = open(path).read()
    def _find(pat):
        m = re.search(pat, text)
        return float(m.group(1)) if m else None
    profit  = _find(r"best_profit[=\s]+([\d.]+)")
    on_time = _find(r"best_on_time[=\s]+([\d.]+)")
    if profit is None or on_time is None:
        return None
    return {"profit": profit, "on_time": on_time, "sum": profit + on_time}


# ── Load results ──────────────────────────────────────────────────────────────

if ML_RUN_ID:
    ml_run = os.path.join(ML_BASE, ML_RUN_ID)
else:
    ml_run = latest_subdir(ML_BASE)

if ml_run is None or not os.path.isdir(ml_run):
    print(f"No ML run found in {ML_BASE}. Run run_sample_efficiency_ml_server.sh first.")
    raise SystemExit(1)

print(f"ML run : {ml_run}")
os.makedirs(OUT_DIR, exist_ok=True)

# results[baseline][dataset] = sorted list of (frac, profit, on_time, sum)
results = defaultdict(lambda: defaultdict(list))
missing = []

for baseline in BASELINES:
    bl_dir = os.path.join(ml_run, baseline)
    if not os.path.isdir(bl_dir):
        print(f"  WARNING: baseline '{baseline}' not found — skipping")
        continue
    for path in glob.glob(os.path.join(bl_dir, "*.log")):
        m = LOG_PAT.match(os.path.basename(path))
        if not m:
            continue
        ds, frac = m.group(1), float(m.group(2))
        r = parse_log(path)
        if r:
            results[baseline][ds].append((frac, r["profit"], r["on_time"], r["sum"]))
        else:
            missing.append(f"{baseline}/{ds}_frac{frac}")

for bl in results:
    for ds in results[bl]:
        results[bl][ds].sort(key=lambda x: x[0])

print(f"Baselines found: {[b for b in BASELINES if b in results]}")
if missing:
    print(f"Missing: {len(missing)} log(s)")

# ── Text table ────────────────────────────────────────────────────────────────

def save_table():
    col_w = 10
    lines = []
    for ds in DATASETS:
        lines.append(f"\n{'='*80}")
        lines.append(f"Dataset: {DATASET_LABELS[ds]}")
        lines.append("=" * 80)
        header = f"{'Frac':<8}" + "".join(
            f"  {BASELINE_LABELS[b]:^{col_w*3+2}}" for b in BASELINES if b in results
        )
        subheader = f"{'':8}" + "".join(
            f"  {'Profit':>{col_w}} {'OnTime':>{col_w}} {'Sum':>{col_w}}"
            for b in BASELINES if b in results
        )
        lines.append(header)
        lines.append(subheader)
        lines.append("-" * len(subheader))
        for frac in FRACS:
            row = f"{int(round(frac*100))}%{'':<5}"
            for bl in BASELINES:
                if bl not in results:
                    continue
                entry = next((r for r in results[bl][ds] if r[0] == frac), None)
                if entry:
                    row += f"  {entry[1]:>{col_w}.4f} {entry[2]:>{col_w}.4f} {entry[3]:>{col_w}.4f}"
                else:
                    row += f"  {'—':>{col_w}} {'—':>{col_w}} {'—':>{col_w}}"
            lines.append(row)

    table = "\n".join(lines)
    print(table)
    path = os.path.join(OUT_DIR, "ml_results.txt")
    with open(path, "w") as f:
        f.write(table + "\n")
    print(f"\nSaved: {path}")

save_table()

# ── Plot 1: Line plots — per metric, subplots = datasets, lines = baselines ───

def plot_lines_per_metric():
    """One figure per metric: 3 dataset subplots, one line per baseline."""
    for metric, metric_label in METRICS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, ds in zip(axes, DATASETS):
            for bl in BASELINES:
                if bl not in results or ds not in results[bl]:
                    continue
                rows = results[bl][ds]
                fracs = [r[0] for r in rows]
                idx   = {"profit": 1, "on_time": 2, "sum": 3}[metric]
                vals  = [r[idx] for r in rows]
                ax.plot(fracs, vals, marker="o", linewidth=2, markersize=6,
                        color=BASELINE_COLORS[bl],
                        label=BASELINE_LABELS[bl])

            ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.set_ylabel(metric_label if ds == DATASETS[0] else "")
            ax.set_xticks(FRACS)
            ax.set_xticklabels([f"{int(round(f*100))}%" for f in FRACS], fontsize=7)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"ML Baselines — {metric_label}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"ml_lines_{metric}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

plot_lines_per_metric()

# ── Plot 2: Bar charts — per dataset, x = fracs, bars = baselines ─────────────

def plot_bars_per_dataset():
    """One figure per dataset: 3 metric subplots, grouped bars per frac."""
    active = [b for b in BASELINES if b in results]
    n_bl   = len(active)
    x      = np.arange(len(FRACS))
    width  = 0.8 / n_bl

    for ds in DATASETS:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax, (metric, metric_label) in zip(axes, METRICS):
            for j, bl in enumerate(active):
                vals = []
                for frac in FRACS:
                    entry = next((r for r in results[bl][ds] if r[0] == frac), None)
                    idx   = {"profit": 1, "on_time": 2, "sum": 3}[metric]
                    vals.append(entry[idx] if entry else 0.0)
                offset = (j - (n_bl - 1) / 2) * width
                ax.bar(x + offset, vals, width * 0.9,
                       label=BASELINE_LABELS[bl],
                       color=BASELINE_COLORS[bl],
                       alpha=0.85, edgecolor="white")

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(round(f*100))}%" for f in FRACS], fontsize=8)
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"{DATASET_LABELS[ds]} — ML Baselines Sample Efficiency",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"ml_bars_{ds}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

plot_bars_per_dataset()

# ── Plot 3: Line plots — per baseline, subplots = metrics, lines = datasets ───

DS_COLORS = {
    "dataco":      "#2196F3",
    "globalstore": "#FF5722",
    "oas":         "#4CAF50",
}

def plot_lines_per_baseline():
    """One figure per baseline: 3 metric subplots, one line per dataset."""
    for bl in BASELINES:
        if bl not in results:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, (metric, metric_label) in zip(axes, METRICS):
            idx = {"profit": 1, "on_time": 2, "sum": 3}[metric]
            for ds in DATASETS:
                if ds not in results[bl]:
                    continue
                rows  = results[bl][ds]
                fracs = [r[0] for r in rows]
                vals  = [r[idx] for r in rows]
                ax.plot(fracs, vals, marker="o", linewidth=2, markersize=6,
                        color=DS_COLORS[ds],
                        label=DATASET_LABELS[ds])

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.set_ylabel(metric_label if metric == "profit" else "")
            ax.set_xticks(FRACS)
            ax.set_xticklabels([f"{int(round(f*100))}%" for f in FRACS], fontsize=7)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{BASELINE_LABELS[bl]} — Sample Efficiency",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"ml_lines_{bl}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

plot_lines_per_baseline()

print(f"\nAll ML baseline outputs saved to: {OUT_DIR}")
