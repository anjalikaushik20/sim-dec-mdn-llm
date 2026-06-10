#!/usr/bin/env python3
"""
Table: Prompt ablations (natural, numeric, shuffled_names, names_only).
4 variants × 2 backbones × 2 datasets × 5 seeds, frac=1.00.

One CSV per dataset: rows = (model, prompt_variant), columns = profit / on_time / sum.
All metrics are mean ± std (n=seeds).  Sum = profit + on_time per seed.

Output filenames: prompt_ablation_{dataset}.csv

Log structure:
  {base_dir}/{RUN_ID}/{variant}/{model_tag}/seed{N}/{dataset}.log

Usage:
  conda run -n simenv python3 table_prompt_ablation.py \\
      --base_dir output/decision_maker/prompt_ablation \\
      --out      output/tables/prompt_ablation
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

DS_ALIASES = {"supplychainshipmentpricing": "scsp"}
VARIANTS   = ["natural", "numeric", "shuffled_names", "names_only"]
VAR_LABEL  = {
    "natural":        "Natural",
    "numeric":        "Numeric Only",
    "shuffled_names": "Shuffled Names",
    "names_only":     "Names Only",
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


def collect(base_dir):
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
    ap.add_argument("--base_dir", default="output/decision_maker/prompt_ablation/results")
    ap.add_argument("--out",      default="output/tables/prompt_ablation")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[prompt] Scanning: {args.base_dir}")
    data = collect(args.base_dir)
    print(f"[prompt] {len(data)} (variant, model, dataset) groups found")

    all_models   = sorted({m for (_, m, _) in data})
    all_datasets = [d for d in sorted({d for (_, _, d) in data}) if d != "scsp"]

    for ds in all_datasets:
        rows = []
        for model in all_models:
            for variant in VARIANTS:
                vals = data.get((variant, model, ds), [])
                rows.append({
                    "model":          model,
                    "prompt_variant": VAR_LABEL.get(variant, variant),
                    "profit":         fmt(vals, 0),
                    "on_time":        fmt(vals, 1),
                    "sum":            fmt_sum(vals),
                })
        if not rows:
            continue
        df = pd.DataFrame(rows)
        print(f"\n{'='*100}")
        print(f"  {ds}  (prompt ablation, mean ± std across seeds)")
        print(f"{'='*100}")
        print(df.to_string(index=False))
        out_path = os.path.join(args.out, f"prompt_ablation_{ds}.csv")
        df.to_csv(out_path, index=False)
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
