#!/usr/bin/env python3
"""
Table: Cross-dataset vs in-domain evaluation (Profit + On-Time sum).

Columns per eval dataset: Cross-dataset (DataCo adapter) | In-domain (native adapter, mean±std)
Rows: models

Output: {cross_dir}/cross_dataset_table.txt

Usage:
  conda run -n simenv python3 table_cross_dataset.py \\
      --cross_dir output/decision_maker/cross_dataset/20260606_203853 \\
      --indomain_dir output/decision_maker/all_fracs/models
"""

import argparse
import glob
import os
import re

import numpy as np

DATASETS   = ["globalstore", "oas"]
DS_DISPLAY = {"globalstore": "GlobalStore", "oas": "OAS"}

MODEL_ORDER  = ["gpt2", "gpt2-large", "qwen3-0.6B", "qwen3-1.7B", "phi4-mini"]
MODEL_LABELS = {
    "gpt2":       "GPT-2",
    "gpt2-large": "GPT-2 Large",
    "qwen3-0.6B": "Qwen3-0.6B",
    "qwen3-1.7B": "Qwen3-1.7B",
    "phi4-mini":  "Phi-4 Mini",
}
# directory name → model tag used in log subdirs
INDOMAIN_DIR_TO_TAG = {
    "gpt2":      "gpt2",
    "gpt2_large": "gpt2-large",
    "qwen3_0.6": "qwen3-0.6B",
    "qwen3_1.7": "qwen3-1.7B",
    "phi4_mini": "phi4-mini",
}


def parse_log(path):
    profit, on_time = None, None
    try:
        with open(path) as f:
            for line in f:
                m = re.search(r"best_profit[=\s]+([\d.]+)", line)
                if m:
                    profit = float(m.group(1))
                m = re.search(r"best_on_time[=\s]+([\d.]+)", line)
                if m:
                    on_time = float(m.group(1))
    except OSError:
        pass
    return profit, on_time


def load_cross(cross_dir):
    """Returns {model_tag: {dataset: sum_value}}"""
    results = {}
    for entry in os.listdir(cross_dir):
        if not os.path.isdir(os.path.join(cross_dir, entry)):
            continue
        row = {}
        for ds in DATASETS:
            log = os.path.join(cross_dir, entry, f"eval_{ds}.log")
            p, o = parse_log(log)
            row[ds] = (p + o) if p is not None and o is not None else None
        results[entry] = row
    return results


def load_indomain(indomain_dir):
    """Returns {model_tag: {dataset: (mean_sum, std_sum, n)}}"""
    results = {}
    for dir_name, tag in INDOMAIN_DIR_TO_TAG.items():
        sums_by_ds = {ds: [] for ds in DATASETS}
        pattern = os.path.join(indomain_dir, dir_name, "seed*", tag, "*_frac1.00.log")
        for path in glob.glob(pattern):
            fname = os.path.basename(path)
            ds = re.match(r"([a-z]+)_frac", fname)
            if ds is None or ds.group(1) not in DATASETS:
                continue
            ds = ds.group(1)
            p, o = parse_log(path)
            if p is not None and o is not None:
                sums_by_ds[ds].append(p + o)
        row = {}
        for ds in DATASETS:
            vals = sums_by_ds[ds]
            if vals:
                mean = float(np.mean(vals))
                std  = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                row[ds] = (mean, std, len(vals))
            else:
                row[ds] = None
        results[tag] = row
    return results


def fmt_cross(val):
    if val is None:
        return "—"
    return f"{val:.4f}"


def fmt_indomain(entry):
    if entry is None:
        return "—"
    mean, std, n = entry
    return f"{mean:.4f} ± {std:.4f} (n={n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross_dir",
                    default="output/decision_maker/cross_dataset/20260606_203853")
    ap.add_argument("--indomain_dir",
                    default="output/decision_maker/all_fracs/models")
    args = ap.parse_args()

    cross    = load_cross(args.cross_dir)
    indomain = load_indomain(args.indomain_dir)

    models = sorted(set(cross.keys()) | set(indomain.keys()),
                    key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)

    lines = []
    lines.append("=" * 110)
    lines.append("  Cross-Dataset Generalisation  (metric: Profit + On-Time sum)")
    lines.append(f"  Cross-dataset run : {os.path.basename(args.cross_dir)}")
    lines.append(f"  In-domain source  : {args.indomain_dir}")
    lines.append("=" * 110)

    # Header
    hdr  = f"  {'Model':<18}"
    sub  = f"  {'':18}"
    sep  = "  " + "-" * 106
    for ds in DATASETS:
        hdr += f"  {DS_DISPLAY[ds]:^52}"
        sub += f"  {'Cross-dataset (DataCo→'+DS_DISPLAY[ds]+')':^24}  {'In-domain ('+DS_DISPLAY[ds]+'→'+DS_DISPLAY[ds]+')':^26}"
    lines.append(hdr)
    lines.append(sub)
    lines.append(sep)

    for model in models:
        label = MODEL_LABELS.get(model, model)
        row = f"  {label:<18}"
        for ds in DATASETS:
            cross_val    = cross.get(model, {}).get(ds)
            indomain_val = indomain.get(model, {}).get(ds)
            row += f"  {fmt_cross(cross_val):^24}  {fmt_indomain(indomain_val):^26}"
        lines.append(row)

    lines.append("=" * 110)

    table = "\n".join(lines)
    print(table)

    out_path = os.path.join(args.cross_dir, "cross_dataset_table.txt")
    with open(out_path, "w") as f:
        f.write(table + "\n")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
