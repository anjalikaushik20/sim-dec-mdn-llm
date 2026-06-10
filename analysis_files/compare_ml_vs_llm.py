"""
Compare ML baselines (random, historical, rf, xgb) vs LLM models (GPT-2 and Qwen3 families)
across all datasets, fracs, and metrics.

ML results  : output/decision_maker/ml/<latest_run>/{baseline}/{dataset}_frac{frac}.log
LLM results : output/decision_maker/all_fracs/{model_tag}/{dataset}_frac{frac}.log

Usage: python3 compare_ml_vs_llm.py [--GPU 0]
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

ML_BASE     = "output/decision_maker/ml"
ML_RUN_ID   = None   # None = auto-discover latest
LLM_DIR     = "output/decision_maker/all_fracs/run_01"
OUT_DIR     = "output/decision_maker/comparisons/ml_vs_llm"

# ── Constants ─────────────────────────────────────────────────────────────────

BASELINES = ["random", "historical", "rf", "xgb"]
BASELINE_LABELS = {
    "random":     "Random",
    "historical": "Historical",
    "rf":         "Random Forest",
    "xgb":        "XGBoost",
}
BASELINE_COLORS = {
    "random":     "#B0BEC5",
    "historical": "#FF8A65",
    "rf":         "#66BB6A",
    "xgb":        "#42A5F5",
}

GPT2_MODELS  = ["gpt2", "gpt2-medium", "gpt2-large"]
QWEN3_MODELS = ["qwen3-0.6B", "qwen3-1.7B", "qwen3-4B"]
ALL_MODELS   = GPT2_MODELS + QWEN3_MODELS

MODEL_LABELS = {
    "gpt2":        "GPT-2",
    "gpt2-medium": "GPT-2 Med",
    "gpt2-large":  "GPT-2 Lg",
    "qwen3-0.6B":  "Qwen3-0.6B",
    "qwen3-1.7B":  "Qwen3-1.7B",
    "qwen3-4B":    "Qwen3-4B",
}
GPT2_COLORS = {
    "gpt2":        "#90A4AE",
    "gpt2-medium": "#546E7A",
    "gpt2-large":  "#263238",
}
QWEN3_COLORS = {
    "qwen3-0.6B":  "#81D4FA",
    "qwen3-1.7B":  "#0288D1",
    "qwen3-4B":    "#01579B",
}
MODEL_COLORS = {**GPT2_COLORS, **QWEN3_COLORS}

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


def load_ml(ml_run):
    """Returns dict: baseline → dataset → sorted [(frac, profit, on_time, sum)]"""
    data = defaultdict(lambda: defaultdict(list))
    for bl in BASELINES:
        bl_dir = os.path.join(ml_run, bl)
        if not os.path.isdir(bl_dir):
            continue
        for path in glob.glob(os.path.join(bl_dir, "*.log")):
            m = LOG_PAT.match(os.path.basename(path))
            if not m:
                continue
            ds, frac = m.group(1), float(m.group(2))
            r = parse_log(path)
            if r:
                data[bl][ds].append((frac, r["profit"], r["on_time"], r["sum"]))
    for bl in data:
        for ds in data[bl]:
            data[bl][ds].sort(key=lambda x: x[0])
    return data


def load_llm(llm_dir):
    """Returns dict: model → dataset → sorted [(frac, profit, on_time, sum)]"""
    data = defaultdict(lambda: defaultdict(list))
    for model in ALL_MODELS:
        model_dir = os.path.join(llm_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for path in glob.glob(os.path.join(model_dir, "*.log")):
            m = LOG_PAT.match(os.path.basename(path))
            if not m:
                continue
            ds, frac = m.group(1), float(m.group(2))
            r = parse_log(path)
            if r:
                data[model][ds].append((frac, r["profit"], r["on_time"], r["sum"]))
    for model in data:
        for ds in data[model]:
            data[model][ds].sort(key=lambda x: x[0])
    return data


def get_val(row, metric):
    return {"profit": row[1], "on_time": row[2], "sum": row[3]}[metric]

# ── Load ──────────────────────────────────────────────────────────────────────

ml_run = os.path.join(ML_BASE, ML_RUN_ID) if ML_RUN_ID else latest_subdir(ML_BASE)
if not ml_run or not os.path.isdir(ml_run):
    print(f"No ML run found in {ML_BASE}.")
    raise SystemExit(1)

print(f"ML run  : {ml_run}")
print(f"LLM dir : {LLM_DIR}")
os.makedirs(OUT_DIR, exist_ok=True)

ml_data  = load_ml(ml_run)
llm_data = load_llm(LLM_DIR)

print(f"ML baselines : {[b for b in BASELINES if b in ml_data]}")
print(f"LLM models   : {[m for m in ALL_MODELS if m in llm_data]}")

# ── Text table ────────────────────────────────────────────────────────────────

def save_table():
    col_w = 9
    active_bl  = [b for b in BASELINES   if b in ml_data]
    active_llm = [m for m in ALL_MODELS   if m in llm_data]
    all_methods = active_bl + active_llm
    all_labels  = {**{b: BASELINE_LABELS[b] for b in active_bl},
                   **{m: MODEL_LABELS[m]     for m in active_llm}}

    lines = []
    for ds in DATASETS:
        lines.append(f"\n{'='*100}")
        lines.append(f"Dataset: {DATASET_LABELS[ds]}")
        lines.append("=" * 100)
        header = f"{'Frac':<7}" + "".join(
            f"  {all_labels[k]:^{col_w*3+2}}" for k in all_methods
        )
        subheader = f"{'':7}" + "".join(
            f"  {'P':>{col_w}} {'O':>{col_w}} {'P+O':>{col_w}}" for _ in all_methods
        )
        lines.append(header)
        lines.append(subheader)
        lines.append("-" * len(subheader))

        for frac in FRACS:
            row = f"{int(round(frac*100))}%{'':<4}"
            for k in all_methods:
                src = ml_data[k] if k in BASELINES else llm_data[k]
                entry = next((r for r in src.get(ds, []) if r[0] == frac), None)
                if entry:
                    row += f"  {entry[1]:>{col_w}.4f} {entry[2]:>{col_w}.4f} {entry[3]:>{col_w}.4f}"
                else:
                    row += f"  {'—':>{col_w}} {'—':>{col_w}} {'—':>{col_w}}"
            lines.append(row)

    table = "\n".join(lines)
    print(table)
    path = os.path.join(OUT_DIR, "ml_vs_llm_table.txt")
    with open(path, "w") as f:
        f.write(table + "\n")
    print(f"\nSaved: {path}")

save_table()

def plot_all_datasets(title, baselines, models, model_colors, filename):
    """3×3 grid: rows = datasets, cols = metrics. ML solid, LLM dashed."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))

    for row, ds in enumerate(DATASETS):
        for col, (metric, metric_label) in enumerate(METRICS):
            ax = axes[row][col]

            for bl in baselines:
                if bl not in ml_data or ds not in ml_data[bl]:
                    continue
                rows_  = ml_data[bl][ds]
                fracs  = [r[0] for r in rows_]
                vals   = [get_val(r, metric) for r in rows_]
                ax.plot(fracs, vals, linestyle="-", marker="s", linewidth=1.8,
                        markersize=5, color=BASELINE_COLORS[bl],
                        label=BASELINE_LABELS[bl])

            for model in models:
                if model not in llm_data or ds not in llm_data[model]:
                    continue
                rows_  = llm_data[model][ds]
                fracs  = [r[0] for r in rows_]
                vals   = [get_val(r, metric) for r in rows_]
                ax.plot(fracs, vals, linestyle="--", marker="o", linewidth=2,
                        markersize=5, color=model_colors[model],
                        label=MODEL_LABELS[model])

            if row == 0:
                ax.set_title(metric_label, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(DATASET_LABELS[ds], fontsize=10, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=8)
            ax.set_xticks(FRACS)
            ax.set_xticklabels([f"{int(round(f*100))}%" for f in FRACS], fontsize=6)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# Figure 1: All 4 ML baselines vs GPT-2 family, all 3 datasets
plot_all_datasets(
    title        = "ML Baselines vs GPT-2 Family — All Datasets",
    baselines    = BASELINES,
    models       = GPT2_MODELS,
    model_colors = GPT2_COLORS,
    filename     = "ml_vs_gpt2.png",
)

# Figure 2: All 4 ML baselines vs Qwen3 family, all 3 datasets
plot_all_datasets(
    title        = "ML Baselines vs Qwen3 Family — All Datasets",
    baselines    = BASELINES,
    models       = QWEN3_MODELS,
    model_colors = QWEN3_COLORS,
    filename     = "ml_vs_qwen.png",
)

print(f"\nAll outputs saved to: {OUT_DIR}")
