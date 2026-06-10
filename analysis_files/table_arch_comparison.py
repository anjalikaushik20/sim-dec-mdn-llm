#!/usr/bin/env python3
"""
Table: Architecture comparison — BERT, Serialized MLP, GPT-2, Qwen3-1.7B, Phi4-mini.
frac=1.00 only. Mean ± std across seeds.

Same log paths as plot_arch_comparison.py:
  BERT/smlp : {base_dir}/{model_name}/seed{N}/{dataset}/frac1.00.log
  VocabAlign: {va_dir}/{model_dir}/seed{N}/{model_tag}/{dataset}_frac1.00.log

Output: output/tables/arch_comparison_frac100.txt

Usage:
  python3 table_arch_comparison.py
  python3 table_arch_comparison.py \\
      --bert_dir output/exp2_arch_ablation/20260605_020523_bert \\
      --smlp_dir output/exp2_arch_ablation/20260605_121824_sMLP \\
      --va_dir   output/decision_maker/all_fracs/models
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

FRAC = 1.00

DATASETS   = ["dataco", "globalstore", "oas"]
DS_DISPLAY = {"dataco": "DataCo", "globalstore": "GlobalStore", "oas": "OAS"}
DS_ALIASES = {"supplychainshipmentpricing": "scsp"}

MODELS = ["gpt2", "qwen3-1.7B", "phi4-mini", "bert", "serialized_mlp"]
MODEL_LABELS = {
    "gpt2":           "GPT-2",
    "qwen3-1.7B":     "Qwen3-1.7B",
    "phi4-mini":      "Phi4-mini",
    "bert":           "BERT",
    "serialized_mlp": "Serialized MLP",
}


# ── Parsing ───────────────────────────────────────────────────────────────────

def norm_ds(s):
    s = s.lower()
    return DS_ALIASES.get(s, s)


def parse_log(path):
    profit, on_time = None, None
    try:
        with open(path) as f:
            for line in f:
                m = re.search(r"best_profit[=\s]+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if m:
                    profit = float(m.group(1))
                m = re.search(r"best_on_time[=\s]+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if m:
                    on_time = float(m.group(1))
    except OSError:
        pass
    return (profit, on_time) if profit is not None and on_time is not None else None


# ── Loaders ───────────────────────────────────────────────────────────────────

def collect_ablation(base_dir, model_name, frac):
    """BERT / Serialized MLP: {base_dir}/{model_name}/seed{N}/{dataset}/frac{X}.log
    Returns {dataset: [(profit, on_time), ...]} across seeds."""
    raw = defaultdict(list)
    pattern = os.path.join(base_dir, model_name, "seed*", "*", f"frac{frac:.2f}.log")
    for path in glob.glob(pattern):
        parts = path.replace("\\", "/").split("/")
        ds = norm_ds(parts[-2])
        r = parse_log(path)
        if r:
            raw[ds].append(r)
    return dict(raw)


def collect_va(va_dir, frac):
    """VocabAlign: {va_dir}/{model_dir}/seed{N}/{model_tag}/{ds}_frac{X}.log
    Returns {model_tag: {dataset: [(profit, on_time), ...]}} across seeds."""
    raw = defaultdict(lambda: defaultdict(list))
    pattern = os.path.join(va_dir, "*", "seed*", "*", f"*_frac{frac:.2f}.log")
    for path in glob.glob(pattern):
        parts = path.replace("\\", "/").split("/")
        m = re.match(r"(.+)_frac([\d.]+)\.log$", parts[-1])
        if m is None:
            continue
        ds        = norm_ds(m.group(1))
        model_tag = parts[-2]
        r = parse_log(path)
        if r:
            raw[model_tag][ds].append(r)
    return {mt: dict(ds_map) for mt, ds_map in raw.items()}


# ── Formatting ────────────────────────────────────────────────────────────────

def mean_std(vals):
    if not vals:
        return None
    profits = [v[0] for v in vals]
    ontimes = [v[1] for v in vals]
    n  = len(vals)
    mp = float(np.mean(profits))
    sp = float(np.std(profits, ddof=1) if n > 1 else 0.0)
    mo = float(np.mean(ontimes))
    so = float(np.std(ontimes, ddof=1) if n > 1 else 0.0)
    return mp, sp, mo, so, mp + mo, sp + so, n


def fmt_cell(vals):
    r = mean_std(vals)
    if r is None:
        return "—", "—", "—", 0
    mp, sp, mo, so, ms, ss, n = r
    return (f"{mp:.4f}±{sp:.4f}",
            f"{mo:.4f}±{so:.4f}",
            f"{ms:.4f}±{ss:.4f}",
            n)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert_dir", default="output/exp2_arch_ablation/20260605_020523_bert")
    ap.add_argument("--smlp_dir", default="output/exp2_arch_ablation/20260605_121824_sMLP")
    ap.add_argument("--va_dir",   default="output/decision_maker/all_fracs/models")
    ap.add_argument("--out",      default="output/tables")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[bert]  {args.bert_dir}")
    bert_data = collect_ablation(args.bert_dir, "bert", FRAC)
    print(f"        datasets: {sorted(bert_data.keys())}")

    print(f"[smlp]  {args.smlp_dir}")
    smlp_data = collect_ablation(args.smlp_dir, "serialized_mlp", FRAC)
    print(f"        datasets: {sorted(smlp_data.keys())}")

    print(f"[va]    {args.va_dir}")
    va_data = collect_va(args.va_dir, FRAC)
    print(f"        model tags: {sorted(va_data.keys())}")

    def get_vals(model, ds):
        if model == "bert":
            return bert_data.get(ds, [])
        if model == "serialized_mlp":
            return smlp_data.get(ds, [])
        return va_data.get(model, {}).get(ds, [])

    W = 16
    col_block = W * 3 + 8
    total_w = 22 + len(DATASETS) * col_block
    sep  = "=" * total_w
    dash = "-" * total_w

    lines = [
        sep,
        "  Architecture Comparison — frac = 1.00  (mean ± std across seeds)",
        sep,
    ]

    hdr = f"  {'Method':<20}"
    for ds in DATASETS:
        hdr += f"  {DS_DISPLAY[ds]:^{col_block}}"
    lines.append(hdr)

    sub = f"  {'':20}"
    for _ in DATASETS:
        sub += f"  {'Profit':>{W}}  {'On-Time':>{W}}  {'Sum':>{W}}"
    lines.append(sub)
    lines.append(dash)

    for model in MODELS:
        row = f"  {MODEL_LABELS[model]:<20}"
        n_shown = []
        for ds in DATASETS:
            vals = get_vals(model, ds)
            p, o, s, n = fmt_cell(vals)
            row += f"  {p:>{W}}  {o:>{W}}  {s:>{W}}"
            n_shown.append(n)
        if any(n_shown):
            row += f"  (n={n_shown[0]})"
        lines.append(row)

    lines.append(sep)
    lines.append("  Values: mean ± std at frac=1.00")
    lines.append(sep)

    table = "\n".join(lines)
    print("\n" + table)

    out_path = os.path.join(args.out, "arch_comparison_frac100.txt")
    with open(out_path, "w") as f:
        f.write(table + "\n")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
