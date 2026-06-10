#!/usr/bin/env python3
"""
Table: VocabAlign sample efficiency + BERT + SerializedMLP + RL comparison.

One CSV per dataset: rows = frac, columns = {model}_profit / {model}_on_time /
{model}_sum for every model including RL.  All metrics are mean ± std (n=seeds).
Sum = profit + on_time computed per seed before averaging.

Output filenames: sample_efficiency_{dataset}.csv

Log structures:
  VocabAlign: {va_dir}/**/seed{N}/{model_tag}/{dataset}_frac{frac}.log
  BERT/sMLP:  {arch_dir}/{model}/seed{N}/{dataset}/frac{frac}.log
  RL:         {rl_dir}/**/seed{N}/{dataset}_frac{frac}.log

Usage:
  conda run -n simenv python3 table_sample_efficiency.py \\
      --va_dir    output/decision_maker/all_fracs/models \\
      --rl_dir    output/decision_maker/rl \\
      --arch_dirs output/exp2_arch_ablation/20260605_020523_bert \\
                  output/exp2_arch_ablation/20260605_121824_sMLP \\
      --out       output/tables/sample_efficiency
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


def collect_va(base_dir):
    """(model_tag, dataset, frac) → [(profit, on_time), ...]"""
    data = defaultdict(list)
    for path in glob.glob(os.path.join(base_dir, "**", "seed*", "*", "*.log"),
                          recursive=True):
        parts = path.replace("\\", "/").split("/")
        seed_seg = next((p for p in parts if re.match(r"seed\d+$", p)), None)
        if seed_seg is None:
            continue
        si = parts.index(seed_seg)
        if si + 2 >= len(parts):
            continue
        model_tag = parts[si + 1]
        m = re.match(r"(.+)_frac([\d.]+)\.log$", parts[-1])
        if m is None:
            continue
        ds, frac = norm_ds(m.group(1)), m.group(2)
        profit  = parse_metric(path, "best_profit")
        on_time = parse_metric(path, "best_on_time")
        if profit is None or on_time is None:
            continue
        data[(model_tag, ds, frac)].append((profit, on_time))
    return data


def collect_arch(base_dirs):
    """(model_tag, dataset, frac) → [(profit, on_time), ...]
    Structure: {model}/seed{N}/{dataset}/frac{frac}.log
    """
    data = defaultdict(list)
    for base_dir in base_dirs:
        for path in glob.glob(os.path.join(base_dir, "**", "*.log"), recursive=True):
            parts = path.replace("\\", "/").split("/")
            seed_seg = next((p for p in parts if re.match(r"seed\d+$", p)), None)
            if seed_seg is None:
                continue
            si = parts.index(seed_seg)
            if si < 1 or si + 2 >= len(parts):
                continue
            model_tag = parts[si - 1]
            ds        = norm_ds(parts[si + 1])
            m = re.match(r"frac([\d.]+)\.log$", parts[-1])
            if m is None:
                continue
            frac    = m.group(1)
            profit  = parse_metric(path, "best_profit")
            on_time = parse_metric(path, "best_on_time")
            if profit is None or on_time is None:
                continue
            data[(model_tag, ds, frac)].append((profit, on_time))
    return data


def collect_rl(base_dir):
    """(dataset, frac) → [(profit, on_time), ...]  — seed-labelled runs only."""
    data = defaultdict(list)
    for path in glob.glob(os.path.join(base_dir, "**", "seed*", "*.log"),
                          recursive=True):
        parts = path.replace("\\", "/").split("/")
        if not any(re.match(r"seed\d+$", p) for p in parts):
            continue
        m = re.match(r"(.+)_frac([\d.]+)\.log$", parts[-1])
        if m is None:
            continue
        ds, frac = norm_ds(m.group(1)), m.group(2)
        profit  = parse_metric(path, "best_profit")
        on_time = parse_metric(path, "best_on_time")
        if profit is None or on_time is None:
            continue
        data[(ds, frac)].append((profit, on_time))
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


def build_table(combined, rl, dataset):
    fracs  = sorted({frac for (_, ds, frac) in combined if ds == dataset} |
                    {frac for (ds, frac) in rl if ds == dataset},
                    key=lambda x: float(x))
    models = sorted({model for (model, ds, _) in combined if ds == dataset})
    rows = []
    for frac in fracs:
        row = {"frac": frac}
        for model in models:
            vals = combined.get((model, dataset, frac), [])
            row[f"{model}_profit"]  = fmt(vals, 0)
            row[f"{model}_on_time"] = fmt(vals, 1)
            row[f"{model}_sum"]     = fmt_sum(vals)
        rl_vals = rl.get((dataset, frac), [])
        row["RL_profit"]  = fmt(rl_vals, 0)
        row["RL_on_time"] = fmt(rl_vals, 1)
        row["RL_sum"]     = fmt_sum(rl_vals)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--va_dir",    default="output/decision_maker/all_fracs/models")
    ap.add_argument("--rl_dir",    default="output/decision_maker/rl")
    ap.add_argument("--arch_dirs", nargs="+", default=[],
                    help="BERT/sMLP log dirs (structure: {model}/seed{N}/{dataset}/frac{frac}.log)")
    ap.add_argument("--out",       default="output/tables/sample_efficiency")
    ap.add_argument("--datasets",  nargs="+", default=DATASETS)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[table] Scanning VocabAlign: {args.va_dir}")
    combined = collect_va(args.va_dir)
    print(f"[table] {len(combined)} VocabAlign groups found")

    if args.arch_dirs:
        print(f"[table] Scanning arch dirs: {args.arch_dirs}")
        arch = collect_arch(args.arch_dirs)
        print(f"[table] {len(arch)} BERT/sMLP groups found")
        combined.update(arch)

    print(f"[table] Scanning RL: {args.rl_dir}")
    rl = collect_rl(args.rl_dir)
    print(f"[table] {len(rl)} RL groups found")

    for ds in args.datasets:
        display = DS_DISPLAY.get(ds, ds)
        df = build_table(combined, rl, ds)
        if df.empty:
            print(f"\n  [SKIP] {display}: no data")
            continue
        print(f"\n{'='*100}")
        print(f"  {display}  (profit / on_time / sum, mean ± std across seeds)")
        print(f"{'='*100}")
        print(df.to_string(index=False))
        out_path = os.path.join(args.out, f"sample_efficiency_{ds}.csv")
        df.to_csv(out_path, index=False)
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
