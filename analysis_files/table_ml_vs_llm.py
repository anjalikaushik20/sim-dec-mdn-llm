#!/usr/bin/env python3
"""
Table: ML baselines vs 3 LLMs at frac=1.00.

Rows: HISTORICAL, RANDOM, RF, XGB, GPT-2, Qwen3-1.7B, Phi4-mini
Cols: DataCo / GlobalStore / OAS — profit, on_time, sum

LLM values are mean ± std across 5 seeds.
ML values are single-run (no seeds).

Output: {out}/ml_vs_llm_frac100.txt

Usage:
  conda run -n simenv python3 table_ml_vs_llm.py
"""

import argparse
import glob
import os
import re

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

DATASETS   = ["dataco", "globalstore", "oas"]
DS_DISPLAY = {"dataco": "DataCo", "globalstore": "GlobalStore", "oas": "OAS"}

ML_BASELINES = ["historical", "random", "rf", "xgb"]
ML_LABELS    = {"historical": "Historical", "random": "Random",
                "rf": "Random Forest", "xgb": "XGBoost"}

LLM_MODELS = ["gpt2", "qwen3-1.7B", "phi4-mini"]
LLM_DIR_MAP = {"gpt2": "gpt2", "qwen3-1.7B": "qwen3_1.7", "phi4-mini": "phi4_mini"}
LLM_LABELS  = {"gpt2": "GPT-2", "qwen3-1.7B": "Qwen3-1.7B", "phi4-mini": "Phi4-mini"}

# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_log(path):
    profit, on_time = None, None
    try:
        with open(path) as f:
            for line in f:
                m = re.search(r"best_profit[=\s]+([\d.eE+\-]+)", line)
                if m:
                    profit = float(m.group(1))
                m = re.search(r"best_on_time[=\s]+([\d.eE+\-]+)", line)
                if m:
                    on_time = float(m.group(1))
    except OSError:
        pass
    return (profit, on_time) if profit is not None and on_time is not None else None


def load_ml(ml_dir):
    """Returns {baseline: {dataset: (profit, on_time)}}"""
    results = {}
    for bl in ML_BASELINES:
        results[bl] = {}
        for ds in DATASETS:
            path = os.path.join(ml_dir, bl, f"{ds}_frac1.00.log")
            r = parse_log(path)
            if r:
                results[bl][ds] = r
    return results


def load_llm(llm_dir):
    """Returns {model: {dataset: (mean_profit, std_profit, mean_on_time, std_on_time, n)}}"""
    results = {}
    for model in LLM_MODELS:
        dir_name = LLM_DIR_MAP[model]
        model_base = os.path.join(llm_dir, dir_name)
        results[model] = {}
        for ds in DATASETS:
            vals = []
            for sd in sorted(glob.glob(os.path.join(model_base, "seed*"))):
                path = os.path.join(sd, model, f"{ds}_frac1.00.log")
                r = parse_log(path)
                if r:
                    vals.append(r)
            if vals:
                profits  = [v[0] for v in vals]
                ontimes  = [v[1] for v in vals]
                n = len(vals)
                results[model][ds] = (
                    float(np.mean(profits)),
                    float(np.std(profits, ddof=1) if n > 1 else 0.0),
                    float(np.mean(ontimes)),
                    float(np.std(ontimes, ddof=1) if n > 1 else 0.0),
                    n,
                )
    return results


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt_ml(val):
    if val is None:
        return f"{'—':>8}", f"{'—':>8}", f"{'—':>8}"
    p, o = val
    return f"{p:8.4f}", f"{o:8.4f}", f"{p+o:8.4f}"


def fmt_llm(val):
    if val is None:
        return f"{'—':>14}", f"{'—':>14}", f"{'—':>14}"
    mp, sp, mo, so, n = val
    ps = f"{mp:.4f}±{sp:.4f}"
    os_ = f"{mo:.4f}±{so:.4f}"
    ss = f"{mp+mo:.4f}±{sp+so:.4f}"
    return f"{ps:>16}", f"{os_:>16}", f"{ss:>16}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml_dir",  default="output/decision_maker/ml/20260529_202125")
    ap.add_argument("--llm_dir", default="output/decision_maker/all_fracs/models")
    ap.add_argument("--out",     default="output/tables")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ml_data  = load_ml(args.ml_dir)
    llm_data = load_llm(args.llm_dir)

    lines = []
    lines.append("=" * 110)
    lines.append("  ML Baselines vs LLMs — frac = 1.00 (100% training data)")
    lines.append("=" * 110)

    col_w = 18
    # Header row 1: dataset names
    hdr = f"  {'Method':<20}"
    for ds in DATASETS:
        hdr += f"  {DS_DISPLAY[ds]:^52}"
    lines.append(hdr)

    # Header row 2: metric names per dataset
    sub = f"  {'':20}"
    for _ in DATASETS:
        sub += f"  {'Profit':>16}  {'On-Time':>16}  {'Sum':>16}"
    lines.append(sub)
    lines.append("  " + "-" * 106)

    # ML baseline rows (single values, no std)
    for bl in ML_BASELINES:
        row = f"  {ML_LABELS[bl]:<20}"
        for ds in DATASETS:
            val = ml_data.get(bl, {}).get(ds)
            p, o, s = fmt_ml(val)
            row += f"  {p:>16}  {o:>16}  {s:>16}"
        lines.append(row)

    lines.append("  " + "-" * 106)

    # LLM rows (mean ± std across seeds)
    for model in LLM_MODELS:
        row = f"  {LLM_LABELS[model]:<20}"
        for ds in DATASETS:
            val = llm_data.get(model, {}).get(ds)
            p, o, s = fmt_llm(val)
            row += f"  {p}  {o}  {s}"
        lines.append(row)

    lines.append("=" * 110)
    lines.append("  ML: single run.  LLM: mean ± std across seeds.")
    lines.append("=" * 110)

    table = "\n".join(lines)
    print(table)

    out_path = os.path.join(args.out, "ml_vs_llm_frac100.txt")
    with open(out_path, "w") as f:
        f.write(table + "\n")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
