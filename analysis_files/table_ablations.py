#!/usr/bin/env python3
"""
Table: Method ablations (no_vocab_init, mean_pool, hard_labels_only).
3 backbones × 3 variants × 4 datasets × 5 seeds, frac=1.00.

One CSV per dataset: rows = (backbone, variant), columns = profit / on_time / sum.
Full VocabAlign baseline (frac=1.00 from --va_dir) included as variant "Full".
All metrics are mean ± std (n=seeds).  Sum = profit + on_time per seed.

Output filenames: ablations_{dataset}.csv

Log structures:
  ablation_3b: {base_dir}/{RUN_ID}/{variant}/{model_tag}/seed{N}/{dataset}.log
  va baseline: {va_dir}/**/seed{N}/{model_tag}/{dataset}_frac1.00.log

Usage:
  conda run -n simenv python3 table_ablations.py \\
      --base_dir output/decision_maker/ablation_3b \\
      --va_dir   output/decision_maker/all_fracs/models \\
      --out      output/tables/ablations
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

DS_ALIASES = {"supplychainshipmentpricing": "scsp"}
DATASETS   = ["dataco", "globalstore", "oas", "scsp"]
DS_DISPLAY = {"dataco": "DataCo", "globalstore": "GlobalStore",
              "oas": "OAS", "scsp": "SCSP"}
VARIANTS   = ["full", "no_vocab_init", "mean_pool", "hard_labels_only"]
VAR_LABEL  = {
    "full":             "Full VocabAlign",
    "no_vocab_init":    "No VocabInit",
    "mean_pool":        "Mean Pool",
    "hard_labels_only": "Hard Labels Only",
}


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


def collect_ablations(base_dir):
    """(variant, model, dataset) → [(profit, on_time), ...]"""
    data = defaultdict(list)
    for path in glob.glob(os.path.join(base_dir, "**", "seed*", "*.log"),
                          recursive=True):
        parts = path.replace("\\", "/").split("/")
        seed_seg = next((p for p in parts if re.match(r"seed\d+$", p)), None)
        if seed_seg is None:
            continue
        si = parts.index(seed_seg)
        if si < 2:
            continue
        model_tag = parts[si - 1]
        variant   = parts[si - 2]
        ds        = norm_ds(os.path.splitext(parts[-1])[0])
        profit    = parse_metric(path, "best_profit")
        on_time   = parse_metric(path, "best_on_time")
        if profit is None or on_time is None:
            continue
        data[(variant, model_tag, ds)].append((profit, on_time))
    return data



# Normalise model tags from all_fracs/models to match the ablation dir naming
VA_TAG_REMAP = {
    "phi4-mini": "phi4-mini-reasoning",
}

def collect_va_baseline(va_dir, frac="1.00"):
    """Full VocabAlign at frac=1.00, keyed as variant='full'."""
    data = defaultdict(list)
    for path in glob.glob(
            os.path.join(va_dir, "**", "seed*", "*", f"*_frac{frac}.log"),
            recursive=True):
        parts = path.replace("\\", "/").split("/")
        seed_seg = next((p for p in parts if re.match(r"seed\d+$", p)), None)
        if seed_seg is None:
            continue
        si = parts.index(seed_seg)
        if si + 2 >= len(parts):
            continue
        model_tag = parts[si + 1]
        model_tag = VA_TAG_REMAP.get(model_tag, model_tag)
        m = re.match(r"(.+)_frac[\d.]+\.log$", parts[-1])
        if m is None:
            continue
        ds      = norm_ds(m.group(1))
        profit  = parse_metric(path, "best_profit")
        on_time = parse_metric(path, "best_on_time")
        if profit is None or on_time is None:
            continue
        data[("full", model_tag, ds)].append((profit, on_time))
    return data


def fmt(values, idx):
    vals = [v[idx] for v in values]
    if not vals:
        return "—"
    mean = np.mean(vals)
    std  = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f} (n={len(vals)})"


def fmt_sum(values):
    sums = [v[0] + v[1] for v in values]
    if not sums:
        return "—"
    mean = np.mean(sums)
    std  = np.std(sums, ddof=1) if len(sums) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f} (n={len(sums)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="output/decision_maker/ablation_3b/20260605_160341")
    ap.add_argument("--va_dir",   default="output/decision_maker/all_fracs/models")
    ap.add_argument("--out",      default="output/tables/ablations")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[ablation] Scanning: {args.base_dir}")
    data = collect_ablations(args.base_dir)
    print(f"[ablation] {len(data)} (variant, model, dataset) groups")

    if args.va_dir:
        baseline = collect_va_baseline(args.va_dir)
        data.update(baseline)
        print(f"[ablation] {len(baseline)} Full VocabAlign baseline entries added")

    MODEL_FILTER = ["gpt2", "qwen3-1.7B"]  # "phi4-mini-reasoning" still running
    all_models = [m for m in MODEL_FILTER if any(m == mod for (_, mod, _) in data)]

    for ds in DATASETS:
        display = DS_DISPLAY.get(ds, ds)
        rows = []
        for model in all_models:
            for variant in VARIANTS:
                vals = data.get((variant, model, ds), [])
                rows.append({
                    "backbone": model,
                    "variant":  VAR_LABEL.get(variant, variant),
                    "profit":   fmt(vals, 0),
                    "on_time":  fmt(vals, 1),
                    "sum":      fmt_sum(vals),
                })
        if not rows:
            print(f"\n  [SKIP] {display}: no data")
            continue
        df = pd.DataFrame(rows)
        print(f"\n{'='*100}")
        print(f"  {display}  (frac=1.00, mean ± std across seeds)")
        print(f"{'='*100}")
        print(df.to_string(index=False))
        out_path = os.path.join(args.out, f"ablations_{ds}.csv")
        df.to_csv(out_path, index=False)
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
