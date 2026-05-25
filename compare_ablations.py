"""
Compare VocabAlign ablations: full vs no_vocab_init vs mean_pool vs hard_labels_only.

Auto-discovers the latest ablation run directory.
Produces:
  - Per-dataset grouped bar charts at fracs {0.01, 0.10, 1.00}
  - Summary table CSV

Usage: conda run -n simenv python3 compare_ablations.py [--ablation_dir <dir>]
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
parser.add_argument("--ablation_dir", type=str,
                    default="output/latest_output/sample_efficiency/ablation")
parser.add_argument("--out_dir", type=str,
                    default="output/latest_output/comparisons/ablations")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

DATASETS = ["DataCo", "GlobalStore", "OAS"]
FRACS = [0.01, 0.10, 1.00]
VARIANTS = ["full", "no_vocab_init", "mean_pool", "hard_labels_only"]
VARIANT_LABELS = {
    "full":            "Full VocabAlign",
    "no_vocab_init":   "No Vocab Init",
    "mean_pool":       "Mean Pool",
    "hard_labels_only":"Hard Labels Only",
}
VARIANT_COLORS = {
    "full":            "#4C72B0",
    "no_vocab_init":   "#DD8452",
    "mean_pool":       "#55A868",
    "hard_labels_only":"#C44E52",
}


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
    pmp1    = _find("best_pmp_1")
    if profit is None:
        return None
    return {"profit": profit, "on_time": on_time or 0.0,
            "pmp1": pmp1 or 0.0,
            "total": (profit or 0.0) + (on_time or 0.0)}


def ds_key(ds):
    return ds.lower().replace("_ood", "ood")


abl_run = latest_subdir(args.ablation_dir)
if abl_run is None:
    print(f"No ablation run found in {args.ablation_dir}.")
    print("Run run_ablation_vocabalign_server.sh first, then re-run this script.")
    raise SystemExit(0)

print(f"Ablation run: {abl_run}")

records = []
for variant in VARIANTS:
    var_dir = os.path.join(abl_run, variant)
    if not os.path.isdir(var_dir):
        print(f"  WARNING: variant '{variant}' directory not found — skipping")
        continue
    for ds in DATASETS:
        for frac in FRACS:
            fn = os.path.join(var_dir, f"{ds_key(ds)}_frac{frac:.2f}.log")
            if not os.path.isfile(fn):
                fn = os.path.join(var_dir, f"{ds_key(ds)}_frac{frac}.log")
            r = parse_log(fn)
            if r:
                records.append({"variant": variant, "dataset": ds, "frac": frac, **r})
            else:
                print(f"  Missing: {fn}")

df = pd.DataFrame(records)
if df.empty:
    print("No ablation results loaded. Run the ablation experiments first.")
    raise SystemExit(0)

print(f"Loaded {len(df)} records")

# ── Summary table ──────────────────────────────────────────────────────────────
csv_path = os.path.join(args.out_dir, "ablation_results.csv")
df.to_csv(csv_path, index=False)
print(f"Saved → {csv_path}")

for ds in DATASETS:
    sub = df[df["dataset"] == ds]
    if sub.empty:
        continue
    pivot = sub.pivot_table(index="variant", columns="frac",
                            values="total", aggfunc="max")
    print(f"\n{'='*60}")
    print(f"DATASET: {ds}  — Total (profit + on_time)")
    print('='*60)
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

# ── Grouped bar charts ─────────────────────────────────────────────────────────
for ds in DATASETS:
    sub = df[df["dataset"] == ds]
    if sub.empty:
        continue

    fig, axes = plt.subplots(1, len(FRACS), figsize=(5 * len(FRACS), 5), sharey=False)
    fig.suptitle(f"{ds} — Ablation Study (Qwen3-1.7B)", fontsize=13)

    for ax, frac in zip(axes, FRACS):
        frac_sub = sub[sub["frac"] == frac]
        vals, colors, labels = [], [], []
        for v in VARIANTS:
            row = frac_sub[frac_sub["variant"] == v]
            vals.append(float(row["total"].iloc[0]) if not row.empty else 0.0)
            colors.append(VARIANT_COLORS[v])
            labels.append(VARIANT_LABELS[v])
        x = np.arange(len(VARIANTS))
        bars = ax.bar(x, vals, color=colors, edgecolor="white", width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"frac={frac}")
        ax.set_ylabel("Total (profit + on_time)")
        ax.grid(axis="y", alpha=0.3)
        # annotate bars
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, f"ablation_{ds_key(ds)}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {fig_path}")

# ── Line plots across fracs ───────────────────────────────────────────────────
for ds in DATASETS:
    sub = df[df["dataset"] == ds]
    if sub.empty:
        continue
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{ds} — Ablation Variants vs Training Fraction", fontsize=12)
    for metric, ax in zip(["profit", "on_time"], axes):
        for v in VARIANTS:
            vsub = sub[sub["variant"] == v].sort_values("frac")
            if vsub.empty:
                continue
            ax.plot(vsub["frac"], vsub[metric], marker="o",
                    color=VARIANT_COLORS[v], label=VARIANT_LABELS[v],
                    linewidth=2, markersize=5)
        ax.set_xlabel("Training fraction")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, f"ablation_{ds_key(ds)}_lines.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {fig_path}")

print(f"\nAll ablation outputs saved to: {args.out_dir}")
