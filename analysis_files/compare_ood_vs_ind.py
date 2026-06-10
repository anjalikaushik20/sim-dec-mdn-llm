"""
OOD vs In-Distribution degradation comparison.

Compares DataCo (in-distribution) vs DataCo_OOD across:
  - All VocabAlign models at all training fractions
  - RL at all training fractions
  - Zero-shot VocabAlign (frac=0)

Shows how each method degrades under the distribution shift found in DataCo_OOD
(Subset_2 has only Standard/Second Class shipping modes; late delivery risk ~85%).

Auto-discovers the latest run directories for in-dist and OOD experiments.

Usage: conda run -n simenv python3 compare_ood_vs_ind.py
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

parser = argparse.ArgumentParser()
parser.add_argument("--ind_llm_dir", type=str,
                    default="/data/akaush39/sim-to-dec/output/latest_output/sample_efficiency/all_vocabalign",
                    help="In-distribution VocabAlign sample efficiency runs")
parser.add_argument("--ood_llm_dir", type=str,
                    default="/data/akaush39/sim-to-dec/output/latest_output/sample_efficiency/ood_vocabalign",
                    help="OOD VocabAlign sample efficiency runs")
parser.add_argument("--ind_rl_dir", type=str,
                    default="/data/akaush39/sim-to-dec/output/latest_output/sample_efficiency/rl_baseline")
parser.add_argument("--ood_rl_dir", type=str,
                    default="/data/akaush39/sim-to-dec/output/latest_output/sample_efficiency/ood_rl_baseline")
parser.add_argument("--ind_zs_dir", type=str,
                    default="/data/akaush39/sim-to-dec/output/latest_output/zero_shot/vocabalign")
parser.add_argument("--ood_zs_dir", type=str,
                    default="/data/akaush39/sim-to-dec/output/latest_output/zero_shot/ood_vocabalign")
parser.add_argument("--out_dir", type=str,
                    default="/data/akaush39/sim-to-dec/output/latest_output/comparisons/ood_vs_ind")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

MODELS = ["qwen3-0.6B", "qwen3-1.7B", "qwen3-4B", "gpt2", "gpt2-medium", "gpt2-large"]
FRACS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

MODEL_COLORS = {}
_cmap = plt.cm.tab10(np.linspace(0, 0.9, len(MODELS)))
for i, m in enumerate(MODELS):
    MODEL_COLORS[m] = _cmap[i]
MODEL_COLORS["RL"] = "#d62728"


def latest_subdir(base):
    if not os.path.isdir(base):
        return None
    subdirs = [d for d in glob.glob(os.path.join(base, "2*")) if os.path.isdir(d)]
    return max(subdirs, key=os.path.getmtime) if subdirs else None


def parse_log(path):
    if not os.path.isfile(path):
        return None
    text = open(path).read()
    def _find(key):
        m = re.search(rf'\b{key}[= ]([0-9.\-]+)', text)
        return float(m.group(1)) if m else None
    profit  = _find("best_profit")
    on_time = _find("best_on_time")
    if profit is None:
        return None
    return {"profit": profit, "on_time": on_time or 0.0,
            "total": (profit or 0.0) + (on_time or 0.0)}


def ds_key(ds):
    return ds.lower().replace("_ood", "ood")


records = []  # {"split", "method", "model", "frac", "profit", "on_time", "total"}

# ── Load in-distribution VocabAlign ──────────────────────────────────────────
for run_dir in sorted(glob.glob(os.path.join(args.ind_llm_dir, "2*")),
                      key=os.path.getmtime):
    for model in MODELS:
        model_dir = os.path.join(run_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for frac in FRACS:
            fn = os.path.join(model_dir, f"dataco_frac{frac:.2f}.log")
            if not os.path.isfile(fn):
                fn = os.path.join(model_dir, f"dataco_frac{frac}.log")
            r = parse_log(fn)
            if r:
                records.append({"split": "IND", "method": f"VocabAlign-{model}",
                                 "model": model, "frac": frac, **r})

# ── Load OOD VocabAlign ───────────────────────────────────────────────────────
ood_llm_run = latest_subdir(args.ood_llm_dir)
if ood_llm_run:
    print(f"OOD VocabAlign dir: {ood_llm_run}")
    for model in MODELS:
        model_dir = os.path.join(ood_llm_run, model)
        if not os.path.isdir(model_dir):
            continue
        for frac in FRACS:
            fn = os.path.join(model_dir, f"dataco_ood_frac{frac:.2f}.log")
            if not os.path.isfile(fn):
                fn = os.path.join(model_dir, f"dataco_ood_frac{frac}.log")
            r = parse_log(fn)
            if r:
                records.append({"split": "OOD", "method": f"VocabAlign-{model}",
                                 "model": model, "frac": frac, **r})
else:
    print("WARNING: no OOD VocabAlign run found — run run_ood_vocabalign_server.sh first")

# ── Load in-distribution zero-shot ────────────────────────────────────────────
ind_zs_run = latest_subdir(args.ind_zs_dir)
if ind_zs_run:
    for model in MODELS:
        fn = os.path.join(ind_zs_run, f"dataco_{model}.log")
        r = parse_log(fn)
        if r:
            records.append({"split": "IND", "method": f"VocabAlign-{model}",
                             "model": model, "frac": 0.0, **r})

# ── Load OOD zero-shot ────────────────────────────────────────────────────────
ood_zs_run = latest_subdir(args.ood_zs_dir)
if ood_zs_run:
    for model in MODELS:
        fn = os.path.join(ood_zs_run, f"dataco_ood_{model}.log")
        r = parse_log(fn)
        if r:
            records.append({"split": "OOD", "method": f"VocabAlign-{model}",
                             "model": model, "frac": 0.0, **r})

# ── Load in-distribution RL ───────────────────────────────────────────────────
ind_rl_run = latest_subdir(args.ind_rl_dir)
if ind_rl_run:
    for frac in FRACS:
        fn = os.path.join(ind_rl_run, f"dataco_frac{frac}.log")
        r = parse_log(fn)
        if r:
            records.append({"split": "IND", "method": "RL", "model": "RL",
                             "frac": frac, **r})

# ── Load OOD RL ───────────────────────────────────────────────────────────────
ood_rl_run = latest_subdir(args.ood_rl_dir)
if ood_rl_run:
    for frac in FRACS:
        fn = os.path.join(ood_rl_run, f"dataco_ood_frac{frac}.log")
        r = parse_log(fn)
        if r:
            records.append({"split": "OOD", "method": "RL", "model": "RL",
                             "frac": frac, **r})

df = pd.DataFrame(records)
print(f"\nLoaded {len(df)} records  (IND={len(df[df['split']=='IND'])}, OOD={len(df[df['split']=='OOD'])})")
if df.empty:
    print("No data found. Exiting.")
    raise SystemExit(1)

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(args.out_dir, "ood_vs_ind_results.csv")
df.to_csv(csv_path, index=False)
print(f"Saved → {csv_path}")

# ── Compute degradation (IND total − OOD total per method × frac) ─────────────
methods = df["method"].unique()
deg_rows = []
for method in methods:
    for frac in sorted(df["frac"].unique()):
        ind_rows = df[(df["method"] == method) & (df["split"] == "IND") & (df["frac"] == frac)]
        ood_rows = df[(df["method"] == method) & (df["split"] == "OOD") & (df["frac"] == frac)]
        if ind_rows.empty or ood_rows.empty:
            continue
        ind_val = ind_rows["total"].max()
        ood_val = ood_rows["total"].max()
        deg_rows.append({"method": method, "frac": frac,
                         "ind_total": ind_val, "ood_total": ood_val,
                         "degradation": ind_val - ood_val,
                         "pct_drop": (ind_val - ood_val) / (abs(ind_val) + 1e-9) * 100})

deg_df = pd.DataFrame(deg_rows)
if not deg_df.empty:
    deg_csv = os.path.join(args.out_dir, "degradation_summary.csv")
    deg_df.to_csv(deg_csv, index=False)
    print(f"Saved degradation summary → {deg_csv}")
    print("\nDegradation (IND total - OOD total) — lower is more robust:")
    pivot = deg_df.pivot_table(index="method", columns="frac",
                               values="degradation", aggfunc="mean")
    print(pivot.to_string(float_format=lambda x: f"{x:+.4f}"))

# ── Plot: IND vs OOD side by side per model ────────────────────────────────────
for metric in ["profit", "on_time", "total"]:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    fig.suptitle(f"DataCo In-Distribution vs OOD — {metric.replace('_',' ').title()}",
                 fontsize=13)

    for ax, model in zip(axes.flatten(), MODELS):
        method = f"VocabAlign-{model}"
        ind_sub = df[(df["method"] == method) & (df["split"] == "IND")].sort_values("frac")
        ood_sub = df[(df["method"] == method) & (df["split"] == "OOD")].sort_values("frac")
        color = MODEL_COLORS[model]

        if not ind_sub.empty:
            ind_agg = ind_sub.groupby("frac")[metric].max().reset_index()
            ax.plot(ind_agg["frac"], ind_agg[metric], "o-", color=color,
                    label="In-dist", linewidth=2, markersize=5)
        if not ood_sub.empty:
            ood_agg = ood_sub.groupby("frac")[metric].max().reset_index()
            ax.plot(ood_agg["frac"], ood_agg[metric], "s--", color=color,
                    label="OOD", linewidth=2, markersize=5, alpha=0.75)

        ax.set_title(model, fontsize=9)
        ax.set_xlabel("Training frac", fontsize=8)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, f"ood_vs_ind_{metric}_by_model.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {fig_path}")

# ── Plot: All methods degradation bar chart ────────────────────────────────────
if not deg_df.empty:
    for frac in BAR_FRACS if (BAR_FRACS := [0.01, 0.10, 1.00]) else []:
        sub = deg_df[deg_df["frac"] == frac].sort_values("degradation")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ["#d62728" if m == "RL" else "#4C72B0" for m in sub["method"]]
        ax.barh(sub["method"], sub["degradation"], color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Degradation (IND total − OOD total)")
        ax.set_title(f"Robustness to OOD shift at frac={frac}  (lower = more robust)")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(args.out_dir, f"ood_degradation_frac{frac}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {fig_path}")

print(f"\nAll OOD vs In-Dist outputs saved to: {args.out_dir}")
