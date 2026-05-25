"""
Unified comparison: Random, Historical, RF/XGBoost, RL, and all VocabAlign models.

Auto-discovers the latest run directories for each method.
Produces:
  - Per-dataset profit/on-time tables (all methods × all fracs)
  - Bar charts at selected fracs (0.01, 0.10, 1.00) per dataset
  - Heatmap of Total (profit + on_time) across methods and fracs

Usage: conda run -n simenv python3 compare_all_methods.py [--out_dir <dir>]
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str,
                    default="/data/akaush39/sim-to-dec/output/latest_output/comparisons/all_methods")
parser.add_argument("--rl_dir", type=str,
                    default="/data/akaush39/sim-to-dec/output/latest_output/sample_efficiency/rl_baseline")
parser.add_argument("--llm_dir", type=str,
                    default="/data/akaush39/sim-to-dec/output/latest_output/sample_efficiency/all_vocabalign")
parser.add_argument("--ml_dir", type=str,
                    default="/data/akaush39/sim-to-dec/output/latest_output/sample_efficiency/ml_baseline")
parser.add_argument("--zero_dir", type=str,
                    default="/data/akaush39/sim-to-dec/output/latest_output/zero_shot/vocabalign")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

DATASETS = ["DataCo", "GlobalStore", "OAS"]
FRACS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
MODELS = ["qwen3-0.6B", "qwen3-1.7B", "qwen3-4B", "gpt2", "gpt2-medium", "gpt2-large"]
ML_BASELINES = ["random", "historical", "rf", "xgb"]
BAR_FRACS = [0.01, 0.10, 1.00]

# ── Helpers ────────────────────────────────────────────────────────────────────

def latest_subdir(base):
    """Return most-recently-modified timestamped subdirectory of base."""
    if not os.path.isdir(base):
        return None
    subdirs = [d for d in glob.glob(os.path.join(base, "2*")) if os.path.isdir(d)]
    return max(subdirs, key=os.path.getmtime) if subdirs else None


def parse_log(path):
    """Return dict {profit, on_time, pmp1, pmp2, pmp3} from a log file.

    Handles three formats:
      VocabAlign:  'best_profit=0.5322'
      RL:          'best_profit 0.5252'
      ML (same as RL or VocabAlign — both written in main/cb_main_ml.py)
    """
    if not os.path.isfile(path):
        return None
    text = open(path).read()

    def _find(key):
        m = re.search(rf'\b{key}[= ]([0-9.\-]+)', text)
        return float(m.group(1)) if m else None

    profit  = _find("best_profit")
    on_time = _find("best_on_time")
    pmp1    = _find("best_pmp_1")
    pmp2    = _find("best_pmp_2")
    pmp3    = _find("best_pmp_3")
    if profit is None:
        return None
    return {"profit": profit, "on_time": on_time or 0.0,
            "pmp1": pmp1 or 0.0, "pmp2": pmp2 or 0.0, "pmp3": pmp3 or 0.0,
            "total": (profit or 0.0) + (on_time or 0.0)}


def ds_key(dataset):
    return dataset.lower().replace("_ood", "ood")


# ── Load data ──────────────────────────────────────────────────────────────────
records = []   # list of dicts: method, model, dataset, frac, profit, on_time, total, ...

# 1) RL baseline
rl_run = latest_subdir(args.rl_dir)
if rl_run:
    print(f"RL dir: {rl_run}")
    for ds in DATASETS:
        for frac in FRACS:
            fn = os.path.join(rl_run, f"{ds_key(ds)}_frac{frac}.log")
            if not os.path.isfile(fn):
                fn = os.path.join(rl_run, f"{ds_key(ds)}_frac{frac:.1f}.log")
            r = parse_log(fn)
            if r:
                records.append({"method": "RL", "model": "RL", "dataset": ds,
                                 "frac": frac, **r})
else:
    print("WARNING: no RL run directory found")

# 2) VocabAlign (sample efficiency)
llm_run_dirs = sorted(glob.glob(os.path.join(args.llm_dir, "2*")), key=os.path.getmtime)
if llm_run_dirs:
    for run_dir in llm_run_dirs:
        for model in MODELS:
            model_dir = os.path.join(run_dir, model)
            if not os.path.isdir(model_dir):
                continue
            for ds in DATASETS:
                for frac in FRACS:
                    fn = os.path.join(model_dir, f"{ds_key(ds)}_frac{frac:.2f}.log")
                    if not os.path.isfile(fn):
                        fn = os.path.join(model_dir, f"{ds_key(ds)}_frac{frac}.log")
                    r = parse_log(fn)
                    if r:
                        records.append({"method": f"VocabAlign-{model}", "model": model,
                                         "dataset": ds, "frac": frac, **r})
    print(f"Loaded VocabAlign from {len(llm_run_dirs)} run dir(s)")
else:
    print("WARNING: no VocabAlign run directories found")

# 3) Zero-shot VocabAlign (frac=0)
zs_run = latest_subdir(args.zero_dir)
if zs_run:
    print(f"Zero-shot dir: {zs_run}")
    for model in MODELS:
        for ds in DATASETS:
            fn = os.path.join(zs_run, f"{ds_key(ds)}_{model}.log")
            r = parse_log(fn)
            if r:
                records.append({"method": f"VocabAlign-{model}", "model": model,
                                 "dataset": ds, "frac": 0.0, **r})

# 4) ML baselines
ml_run = latest_subdir(args.ml_dir)
if ml_run:
    print(f"ML baseline dir: {ml_run}")
    for bl in ML_BASELINES:
        bl_dir = os.path.join(ml_run, bl)
        if not os.path.isdir(bl_dir):
            continue
        for ds in DATASETS:
            for frac in FRACS:
                fn = os.path.join(bl_dir, f"{ds_key(ds)}_frac{frac:.2f}.log")
                if not os.path.isfile(fn):
                    fn = os.path.join(bl_dir, f"{ds_key(ds)}_frac{frac}.log")
                r = parse_log(fn)
                if r:
                    records.append({"method": bl.upper(), "model": bl, "dataset": ds,
                                     "frac": frac, **r})
else:
    print("WARNING: no ML baseline directory found (run run_sample_efficiency_ml_server.sh first)")

df = pd.DataFrame(records)
print(f"\nLoaded {len(df)} result records total")
if df.empty:
    print("No data found. Exiting.")
    raise SystemExit(1)

# ── Print per-dataset summary tables ──────────────────────────────────────────
for ds in DATASETS:
    sub = df[df["dataset"] == ds]
    if sub.empty:
        continue
    pivot = sub.pivot_table(index=["method", "model"], columns="frac",
                            values="total", aggfunc="max")
    print(f"\n{'='*70}")
    print(f"DATASET: {ds}  — Total (profit + on_time) by method × frac")
    print('='*70)
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

# ── Save full CSV ──────────────────────────────────────────────────────────────
csv_path = os.path.join(args.out_dir, "all_methods_results.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved full results → {csv_path}")

# ── Bar charts at selected fracs ───────────────────────────────────────────────
METHOD_ORDER = (
    ["RANDOM", "HISTORICAL", "RF", "XGB", "RL"]
    + [f"VocabAlign-{m}" for m in MODELS]
)
COLORS = {
    "RANDOM":    "#aaaaaa",
    "HISTORICAL":"#888888",
    "RF":        "#55a868",
    "XGB":       "#2ca02c",
    "RL":        "#d62728",
}
LLM_COLORS = plt.cm.tab10(np.linspace(0.0, 0.9, len(MODELS)))
for i, m in enumerate(MODELS):
    COLORS[f"VocabAlign-{m}"] = LLM_COLORS[i]

for ds in DATASETS:
    sub = df[df["dataset"] == ds]
    if sub.empty:
        continue

    fig, axes = plt.subplots(1, len(BAR_FRACS), figsize=(5 * len(BAR_FRACS), 5), sharey=False)
    fig.suptitle(f"{ds} — Profit + On-Time by Method", fontsize=13)

    for ax, frac in zip(axes, BAR_FRACS):
        frac_sub = sub[sub["frac"] == frac]
        # aggregate: take best total per method
        agg = frac_sub.groupby("method")["total"].max().reset_index()
        # sort by METHOD_ORDER
        agg["_order"] = agg["method"].apply(
            lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 999)
        agg = agg.sort_values("_order")

        bar_colors = [COLORS.get(m, "#4c72b0") for m in agg["method"]]
        ax.bar(range(len(agg)), agg["total"], color=bar_colors, edgecolor="white")
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg["method"], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"frac={frac}")
        ax.set_ylabel("Total (profit + on_time)")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, f"all_methods_{ds_key(ds)}_bar.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {fig_path}")

# ── Line plots: profit vs frac per dataset (all methods) ───────────────────────
for ds in DATASETS:
    sub = df[df["dataset"] == ds]
    if sub.empty:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{ds} — All Methods vs Training Fraction", fontsize=13)

    for metric, ax in zip(["profit", "on_time"], axes):
        for method in sorted(sub["method"].unique()):
            msub = sub[sub["method"] == method].sort_values("frac")
            agg = msub.groupby("frac")[metric].max().reset_index()
            color = COLORS.get(method, None)
            ls = "--" if "VocabAlign" not in method else "-"
            ax.plot(agg["frac"], agg[metric], marker="o", label=method,
                    color=color, linestyle=ls, linewidth=1.5, markersize=4)
        ax.set_xlabel("Training fraction")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, f"all_methods_{ds_key(ds)}_lines.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {fig_path}")

print(f"\nAll outputs saved to: {args.out_dir}")
