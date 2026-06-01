"""
Compare VocabAlign ablations: no_vocab_init vs mean_pool vs hard_labels_only.
All 6 models × 6 fracs × 3 datasets.

Auto-discovers the latest run under ABLATION_BASE, or set ABLATION_RUN_ID
to pin a specific run.

Usage: python3 compare_ablations.py [--GPU 0]
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

ABLATION_BASE  = "output/decision_maker/ablation"
ABLATION_RUN_ID = None   # set to e.g. "20260531_120000" to pin a run; None = latest
OUT_DIR        = "output/decision_maker/comparisons/ablations"

# ── Constants ─────────────────────────────────────────────────────────────────

VARIANTS = ["no_vocab_init", "mean_pool", "hard_labels_only"]
VARIANT_LABELS = {
    "no_vocab_init":    "No Vocab Init",
    "mean_pool":        "Mean Pool",
    "hard_labels_only": "Hard Labels Only",
}
VARIANT_COLORS = {
    "no_vocab_init":    "#DD8452",   # orange
    "mean_pool":        "#55A868",   # green
    "hard_labels_only": "#8E24AA",   # purple
}

MODEL_ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "qwen3-0.6B", "qwen3-1.7B", "qwen3-4B"]
MODEL_LABELS = {
    "gpt2":        "GPT-2",
    "gpt2-medium": "GPT-2 Med",
    "gpt2-large":  "GPT-2 Lg",
    "qwen3-0.6B":  "Qwen3-0.6B",
    "qwen3-1.7B":  "Qwen3-1.7B",
    "qwen3-4B":    "Qwen3-4B",
}

DATASETS = ["dataco", "globalstore", "oas"]
DATASET_LABELS = {"dataco": "DataCo", "globalstore": "GlobalStore", "oas": "OAS"}

FRACS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

# ── Helpers ───────────────────────────────────────────────────────────────────

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


def sorted_models(tags):
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(tags, key=lambda m: order.get(m, 99))

# ── Load results ──────────────────────────────────────────────────────────────

if ABLATION_RUN_ID:
    abl_run = os.path.join(ABLATION_BASE, ABLATION_RUN_ID)
else:
    abl_run = latest_subdir(ABLATION_BASE)

if abl_run is None or not os.path.isdir(abl_run):
    print(f"No ablation run found in {ABLATION_BASE}. Run run_ablation_vocabalign_server.sh first.")
    raise SystemExit(1)

print(f"Ablation run : {abl_run}")
os.makedirs(OUT_DIR, exist_ok=True)

# results[variant][model][dataset][frac] = {profit, on_time, sum}
results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
missing = []

for variant in VARIANTS:
    var_dir = os.path.join(abl_run, variant)
    if not os.path.isdir(var_dir):
        print(f"  WARNING: variant '{variant}' not found — skipping")
        continue
    for model in sorted(os.listdir(var_dir)):
        model_dir = os.path.join(var_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for ds in DATASETS:
            for frac in FRACS:
                path = os.path.join(model_dir, f"{ds}_frac{frac:.2f}.log")
                if not os.path.isfile(path):
                    path = os.path.join(model_dir, f"{ds}_frac{frac}.log")
                r = parse_log(path)
                if r:
                    results[variant][model][ds][frac] = r
                else:
                    missing.append(f"{variant}/{model}/{ds}_frac{frac}")

all_models = sorted_models({m for v in results.values() for m in v})
print(f"Models found : {all_models}")
print(f"Missing      : {len(missing)} log(s)")
if missing:
    for m in missing[:10]:
        print(f"  {m}")
    if len(missing) > 10:
        print(f"  ... and {len(missing)-10} more")

# ── Text table ────────────────────────────────────────────────────────────────

def save_table():
    col_w = 10
    lines = []
    for ds in DATASETS:
        for model in all_models:
            lines.append(f"\n{'='*90}")
            lines.append(f"Dataset: {DATASET_LABELS[ds]}  |  Model: {MODEL_LABELS.get(model, model)}")
            lines.append("=" * 90)
            header = f"{'Frac':<8}" + "".join(
                f"  {VARIANT_LABELS[v]:^{col_w*3+2}}" for v in VARIANTS
            )
            subheader = f"{'':8}" + "".join(
                f"  {'Profit':>{col_w}} {'OnTime':>{col_w}} {'Sum':>{col_w}}" for _ in VARIANTS
            )
            lines.append(header)
            lines.append(subheader)
            lines.append("-" * len(subheader))
            for frac in FRACS:
                row = f"{int(round(frac*100))}%{'':<5}"
                for v in VARIANTS:
                    r = results[v][model][ds].get(frac)
                    if r:
                        row += f"  {r['profit']:>{col_w}.4f} {r['on_time']:>{col_w}.4f} {r['sum']:>{col_w}.4f}"
                    else:
                        row += f"  {'—':>{col_w}} {'—':>{col_w}} {'—':>{col_w}}"
                lines.append(row)

    table = "\n".join(lines)
    print(table)
    path = os.path.join(OUT_DIR, "ablation_results.txt")
    with open(path, "w") as f:
        f.write(table + "\n")
    print(f"\nSaved: {path}")

save_table()

# ── Plot 1: Bar charts — per dataset × frac, bars = variants, groups = models ──

def plot_bars_per_dataset():
    """One figure per dataset: subplots = fracs, grouped bars = models × variants."""
    for ds in DATASETS:
        dset_label = DATASET_LABELS[ds]
        n_fracs = len(FRACS)
        fig, axes = plt.subplots(1, n_fracs, figsize=(4 * n_fracs, 5), sharey=False)

        for ax, frac in zip(axes, FRACS):
            n_models = len(all_models)
            n_variants = len(VARIANTS)
            group_w = 0.8
            bar_w = group_w / (n_models * n_variants)
            x = np.arange(n_variants)

            for mi, model in enumerate(all_models):
                vals = []
                for v in VARIANTS:
                    r = results[v][model][ds].get(frac)
                    vals.append(r["sum"] if r else 0.0)
                offset = (mi - (n_models - 1) / 2) * bar_w * n_variants
                ax.bar(x + offset, vals, bar_w * 0.9,
                       label=MODEL_LABELS.get(model, model),
                       alpha=0.85, edgecolor="white")

            ax.set_xticks(x)
            ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS],
                               rotation=25, ha="right", fontsize=7)
            ax.set_title(f"frac={int(round(frac*100))}%", fontsize=10)
            ax.set_ylabel("Profit + On-Time" if ax == axes[0] else "")
            ax.grid(axis="y", alpha=0.3)
            if ax == axes[-1]:
                ax.legend(fontsize=6, loc="upper right")

        fig.suptitle(f"{dset_label} — Ablation Study (All Models)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"ablation_bars_{ds}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

plot_bars_per_dataset()

# ── Plot 2: Line plots — per dataset × model, lines = variants across fracs ───

def plot_lines_per_model():
    """One figure per model: subplots = datasets, lines = variants across fracs."""
    for model in all_models:
        model_label = MODEL_LABELS.get(model, model)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, ds in zip(axes, DATASETS):
            dset_label = DATASET_LABELS[ds]
            for v in VARIANTS:
                pts = [(frac, results[v][model][ds][frac])
                       for frac in FRACS if frac in results[v][model][ds]]
                if not pts:
                    continue
                fracs_v = [p[0] for p in pts]
                sums    = [p[1]["sum"] for p in pts]
                ax.plot(fracs_v, sums, marker="o", linewidth=2, markersize=5,
                        color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])

            ax.set_title(dset_label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.set_ylabel("Profit + On-Time" if ds == DATASETS[0] else "")
            ax.set_xticks(FRACS)
            ax.set_xticklabels([f"{int(round(f*100))}%" for f in FRACS], fontsize=7)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{model_label} — Ablation Variants Across Fractions",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        safe = re.sub(r"[^a-z0-9]+", "_", model.lower())
        path = os.path.join(OUT_DIR, f"ablation_lines_{safe}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

plot_lines_per_model()

# ── Plot 3: Line plots — per dataset, subplots = metrics, lines = variant×model

def plot_lines_per_dataset():
    """One figure per dataset: 3 metric subplots, one line per variant×model combo."""
    for ds in DATASETS:
        dset_label = DATASET_LABELS[ds]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, metric in zip(axes, ["profit", "on_time", "sum"]):
            metric_label = {"profit": "Profit", "on_time": "On-Time", "sum": "Profit + On-Time"}[metric]
            for v in VARIANTS:
                for model in all_models:
                    pts = [(frac, results[v][model][ds][frac])
                           for frac in FRACS if frac in results[v][model][ds]]
                    if not pts:
                        continue
                    fracs_v = [p[0] for p in pts]
                    vals    = [p[1][metric] for p in pts]
                    ax.plot(fracs_v, vals, marker="o", linewidth=1.5, markersize=4,
                            color=VARIANT_COLORS[v], alpha=0.6)

            # Add one legend entry per variant
            for v in VARIANTS:
                ax.plot([], [], color=VARIANT_COLORS[v], linewidth=2,
                        label=VARIANT_LABELS[v])

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.set_xticks(FRACS)
            ax.set_xticklabels([f"{int(round(f*100))}%" for f in FRACS], fontsize=7)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{dset_label} — All Models × Ablation Variants",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"ablation_lines_{ds}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

plot_lines_per_dataset()

# ── Plot 4: Line plots — per variant, subplots = datasets, lines = models ─────

MODEL_COLORS = {
    "gpt2":        "#90A4AE",
    "gpt2-medium": "#546E7A",
    "gpt2-large":  "#263238",
    "qwen3-0.6B":  "#81D4FA",
    "qwen3-1.7B":  "#0288D1",
    "qwen3-4B":    "#01579B",
}

METRIC_KEYS   = ["profit", "on_time", "sum"]
METRIC_LABELS = {"profit": "Profit", "on_time": "On-Time Ratio", "sum": "Profit + On-Time"}

def plot_lines_per_variant():
    """Per variant × dataset: 3 metric subplots, one line per model — same style
    as VocabAlign sample efficiency plots but without RL reference."""
    for v in VARIANTS:
        for ds in DATASETS:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for ax, metric in zip(axes, METRIC_KEYS):
                for model in all_models:
                    pts = [(frac, results[v][model][ds][frac])
                           for frac in FRACS if frac in results[v][model][ds]]
                    if not pts:
                        continue
                    fracs_v = [p[0] for p in pts]
                    vals    = [p[1][metric] for p in pts]
                    ax.plot(fracs_v, vals, marker="o", linewidth=2, markersize=5,
                            color=MODEL_COLORS.get(model, "#555"),
                            label=MODEL_LABELS.get(model, model))

                ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
                ax.set_xlabel("Training Fraction", fontsize=9)
                ax.set_ylabel(METRIC_LABELS[metric] if metric == "profit" else "")
                ax.set_xticks(FRACS)
                ax.set_xticklabels([f"{int(round(f*100))}%" for f in FRACS], fontsize=7)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            dset_label = DATASET_LABELS[ds]
            fig.suptitle(f"{dset_label} — {VARIANT_LABELS[v]}",
                         fontsize=12, fontweight="bold")
            plt.tight_layout()
            safe_v = re.sub(r"[^a-z0-9]+", "_", v.lower())
            path = os.path.join(OUT_DIR, f"ablation_variant_{safe_v}_{ds}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {path}")

plot_lines_per_variant()

# ── Plot 5: Summary grid — per variant, 3×3 (datasets × metrics), lines = models

def plot_variant_summary():
    """Per variant × metric: one figure with 3 dataset subplots, lines per model.
    Produces 9 figures total (3 variants × 3 metrics)."""
    for v in VARIANTS:
        for metric in METRIC_KEYS:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for ax, ds in zip(axes, DATASETS):
                for model in all_models:
                    pts = [(frac, results[v][model][ds][frac])
                           for frac in FRACS if frac in results[v][model][ds]]
                    if not pts:
                        continue
                    fracs_v = [p[0] for p in pts]
                    vals    = [p[1][metric] for p in pts]
                    ax.plot(fracs_v, vals, marker="o", linewidth=2, markersize=5,
                            color=MODEL_COLORS.get(model, "#555"),
                            label=MODEL_LABELS.get(model, model))

                ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold")
                ax.set_xlabel("Training Fraction", fontsize=9)
                ax.set_ylabel(METRIC_LABELS[metric] if ds == DATASETS[0] else "")
                ax.set_xticks(FRACS)
                ax.set_xticklabels([f"{int(round(f*100))}%" for f in FRACS], fontsize=7)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            fig.suptitle(f"{VARIANT_LABELS[v]} — {METRIC_LABELS[metric]} — All Models",
                         fontsize=12, fontweight="bold")
            plt.tight_layout()
            safe_v = re.sub(r"[^a-z0-9]+", "_", v.lower())
            path = os.path.join(OUT_DIR, f"ablation_variant_{safe_v}_{metric}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {path}")

plot_variant_summary()

print(f"\nAll ablation outputs saved to: {OUT_DIR}")
