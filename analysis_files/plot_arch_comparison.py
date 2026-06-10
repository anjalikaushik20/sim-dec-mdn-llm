#!/usr/bin/env python3
"""
Architecture comparison: BERT, Serialized MLP, and selected LLMs.

Produces 3 images, one per dataset (DataCo, GlobalStore, OAS).
Each image has 3 subplots: Profit | On-Time Rate | Profit + On-Time.
Lines: GPT-2, Qwen3-1.7B, Phi-4 Mini, BERT, Serialized MLP.
Mean across seeds shown as a line; style follows compare_fullrl_vs_llm_fracs.py.

Output: {out}/{dataset}_arch_comparison.png

Usage:
  conda run -n simenv python3 plot_arch_comparison.py \\
      --bert_dir output/exp2_arch_ablation/20260605_020523_bert \\
      --smlp_dir output/exp2_arch_ablation/20260605_121824_sMLP \\
      --va_dir   output/decision_maker/all_fracs/models \\
      --out      output/figures/arch_comparison
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

DATASETS   = ["dataco", "globalstore", "oas"]
DS_DISPLAY = {"dataco": "DataCo", "globalstore": "GlobalStore", "oas": "OAS"}
DS_ALIASES = {"supplychainshipmentpricing": "scsp"}
FRACS_ORDER = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

MODELS = ["gpt2", "qwen3-1.7B", "phi4-mini", "bert", "serialized_mlp"]
MODEL_LABELS = {
    "gpt2":           "GPT-2",
    "qwen3-1.7B":     "Qwen3-1.7B",
    "phi4-mini":      "Phi4-mini",
    "bert":           "BERT",
    "serialized_mlp": "Serialized MLP",
}
MODEL_COLORS = {
    "gpt2":           "#90A4AE",
    "qwen3-1.7B":     "#0288D1",
    "phi4-mini":      "#01579B",
    "bert":           "#E53935",
    "serialized_mlp": "#43A047",
}

METRICS = [("profit", "Profit"), ("on_time", "On-Time Ratio"), ("total", "Profit + On-Time")]


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_metric(path, key):
    try:
        with open(path) as f:
            for line in f:
                m = re.search(
                    rf"{re.escape(key)}[=\s]+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", line)
                if m:
                    return float(m.group(1))
    except OSError:
        pass
    return None


def norm_ds(s):
    s = s.lower()
    return DS_ALIASES.get(s, s)


# ── Loaders ───────────────────────────────────────────────────────────────────

def collect_ablation_dir(base_dir, model_name):
    """bert/sMLP: {model_name}/seed{N}/{dataset}/frac{X}.log"""
    raw = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(base_dir, model_name, "seed*", "*", "frac*.log")):
        parts = path.replace("\\", "/").split("/")
        frac  = float(re.sub(r"^frac|\.log$", "", parts[-1]))
        ds    = norm_ds(parts[-2])
        p  = parse_metric(path, "best_profit")
        o  = parse_metric(path, "best_on_time")
        if p is None or o is None:
            continue
        raw[ds][frac].append((p, o))
    # average across seeds
    result = {}
    for ds, frac_map in raw.items():
        rows = []
        for frac in sorted(frac_map):
            vals = frac_map[frac]
            rows.append((frac,
                         float(np.mean([v[0] for v in vals])),
                         float(np.mean([v[1] for v in vals]))))
        result[ds] = rows
    return result


def collect_va_dir(va_dir):
    """VocabAlign: {model_tag}/seed{N}/{model_tag}/{dataset}_frac{X}.log"""
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in glob.glob(os.path.join(va_dir, "*", "seed*", "*", "*_frac*.log")):
        parts = path.replace("\\", "/").split("/")
        m = re.match(r"(.+)_frac([\d.]+)\.log$", parts[-1])
        if m is None:
            continue
        ds        = norm_ds(m.group(1))
        frac      = float(m.group(2))
        model_tag = parts[-2]
        p  = parse_metric(path, "best_profit")
        o  = parse_metric(path, "best_on_time")
        if p is None or o is None:
            continue
        raw[model_tag][ds][frac].append((p, o))
    # average across seeds
    result = {}
    for model_tag, ds_map in raw.items():
        result[model_tag] = {}
        for ds, frac_map in ds_map.items():
            rows = []
            for frac in sorted(frac_map):
                vals = frac_map[frac]
                rows.append((frac,
                             float(np.mean([v[0] for v in vals])),
                             float(np.mean([v[1] for v in vals]))))
            result[model_tag][ds] = rows
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_val(row, key):
    if key == "profit":  return row[1]
    if key == "on_time": return row[2]
    return row[1] + row[2]


def frac_label(f):
    return f"{int(round(f * 100))}%"


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_dataset(all_data, ds, out_path):
    display = DS_DISPLAY[ds]
    all_fracs = FRACS_ORDER

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (key, metric_label) in zip(axes, METRICS):
        for model in MODELS:
            rows = all_data.get(model, {}).get(ds, [])
            if not rows:
                print(f"  [skip] {model} — no data for {display}")
                continue
            fracs = [r[0] for r in rows]
            vals  = [get_val(r, key) for r in rows]
            ax.plot(fracs, vals,
                    color=MODEL_COLORS.get(model, "#555"),
                    linewidth=2, marker="o", markersize=5,
                    label=MODEL_LABELS.get(model, model))

        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Training Fraction", fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(all_fracs)
        ax.set_xticklabels([frac_label(f) for f in all_fracs], fontsize=6)

    fig.suptitle(f"{display} — Architecture Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert_dir", default="output/exp2_arch_ablation/20260605_020523_bert")
    ap.add_argument("--smlp_dir", default="output/exp2_arch_ablation/20260605_121824_sMLP")
    ap.add_argument("--va_dir",   default="output/decision_maker/all_fracs/models")
    ap.add_argument("--out",      default="output/figures/arch_comparison")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[bert]  Scanning: {args.bert_dir}")
    bert_data = collect_ablation_dir(args.bert_dir, "bert")
    print(f"        datasets: {sorted(bert_data.keys())}")

    print(f"[smlp]  Scanning: {args.smlp_dir}")
    smlp_data = collect_ablation_dir(args.smlp_dir, "serialized_mlp")
    print(f"        datasets: {sorted(smlp_data.keys())}")

    print(f"[va]    Scanning: {args.va_dir}")
    va_data = collect_va_dir(args.va_dir)
    print(f"        models: {sorted(va_data.keys())}")

    # merge into a single {model -> {ds -> rows}} dict
    all_data = {**va_data, "bert": bert_data, "serialized_mlp": smlp_data}

    for ds in DATASETS:
        out_path = os.path.join(args.out, f"{ds}_arch_comparison.png")
        plot_dataset(all_data, ds, out_path)

    print(f"\nAll figures saved to: {args.out}")


if __name__ == "__main__":
    main()
