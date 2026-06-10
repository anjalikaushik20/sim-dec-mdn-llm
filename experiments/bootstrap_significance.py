#!/usr/bin/env python3
"""Experiment 2 — Statistical significance overlay on sample-efficiency results.

Reads per-seed log files from Experiment 1 (VocabAlign and RL) and computes,
for each (backbone, dataset, frac) pair:
  - Paired bootstrap CI (1000 resamples) on the profit difference
  - Wilcoxon signed-rank test (paired) across the 5 seeds

Output: CSV table + console-formatted LaTeX fragment.

Usage:
    conda run -n simenv python3 experiments/bootstrap_significance.py \\
        --vocabalign_dir output/decision_maker/all_fracs \\
        --rl_dir output/decision_maker/rl \\
        --out output/significance_table.csv

Log format expected:
    output/decision_maker/all_fracs/{RUN_ID}/seed{N}/{model_tag}/{dataset}_frac{frac}.log
    output/decision_maker/rl/{RUN_ID}/seed{N}/{dataset}_frac{frac}.log

    Each log must contain a line: best_profit={value}
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Log parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_best_profit(log_path: str):
    """Return the best_profit float from a log file, or None on failure."""
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(r"best_profit[=\s]+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", line)
                if m:
                    return float(m.group(1))
    except OSError:
        pass
    return None


_DS_NORM = {"supplychainshipmentpricing": "scsp"}


def _norm_ds(s: str) -> str:
    return _DS_NORM.get(s.lower(), s.lower())


def collect_vocabalign(base_dir: str):
    """Return dict: (model_tag, dataset, frac) → {seed: profit}."""
    results = {}
    # Pattern: {base_dir}/**/seed{N}/{model_tag}/{dataset}_frac{frac}.log
    for log_path in glob.glob(os.path.join(base_dir, "**", "seed*", "*", "*.log"),
                               recursive=True):
        parts = log_path.split(os.sep)
        # Find "seed{N}" segment
        seed_seg = next((p for p in parts if re.match(r"seed\d+$", p)), None)
        if seed_seg is None:
            continue
        seed = int(seed_seg.replace("seed", ""))
        seed_idx = parts.index(seed_seg)
        if seed_idx + 2 >= len(parts):
            continue
        model_tag = parts[seed_idx + 1]
        fname = parts[-1]  # e.g. dataco_frac0.10.log
        m = re.match(r"(.+)_frac([\d.]+)\.log$", fname)
        if m is None:
            continue
        dataset_lower = _norm_ds(m.group(1))
        frac = m.group(2)
        profit = parse_best_profit(log_path)
        if profit is None:
            continue
        key = (model_tag, dataset_lower, frac)
        results.setdefault(key, {})[seed] = profit
    return results


def collect_rl(base_dir: str):
    """Return dict: (dataset, frac) → {seed: profit}."""
    results = {}
    for log_path in glob.glob(os.path.join(base_dir, "**", "seed*", "*.log"),
                               recursive=True):
        parts = log_path.split(os.sep)
        seed_seg = next((p for p in parts if re.match(r"seed\d+$", p)), None)
        if seed_seg is None:
            continue
        seed = int(seed_seg.replace("seed", ""))
        fname = parts[-1]
        m = re.match(r"(.+)_frac([\d.]+)\.log$", fname)
        if m is None:
            continue
        dataset_lower = _norm_ds(m.group(1))
        frac = m.group(2)
        profit = parse_best_profit(log_path)
        if profit is None:
            continue
        key = (dataset_lower, frac)
        results.setdefault(key, {})[seed] = profit
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_paired_ci(a, b, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Paired bootstrap CI on mean(a - b)."""
    rng = np.random.default_rng(seed)
    diffs = np.array(a) - np.array(b)
    means = [rng.choice(diffs, size=len(diffs), replace=True).mean()
             for _ in range(n_boot)]
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(diffs.mean()), lo, hi


def wilcoxon_pvalue(a, b):
    """Wilcoxon signed-rank test on paired differences. Returns p-value."""
    diffs = np.array(a) - np.array(b)
    if np.all(diffs == 0):
        return 1.0
    try:
        _, p = stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        return float(p)
    except ValueError:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap + Wilcoxon significance tests over multi-seed Exp 1 logs"
    )
    parser.add_argument("--vocabalign_dir", type=str,
                        default="output/decision_maker/all_fracs",
                        help="Base dir for VocabAlign multi-seed logs")
    parser.add_argument("--rl_dir", type=str,
                        default="output/decision_maker/rl",
                        help="Base dir for RL multi-seed logs")
    parser.add_argument("--out", type=str,
                        default="output/significance_table.csv",
                        help="Output CSV path")
    parser.add_argument("--min_seeds", type=int, default=2,
                        help="Skip comparisons with fewer seeds than this")
    args = parser.parse_args()

    print(f"[significance] Scanning VocabAlign logs: {args.vocabalign_dir}")
    va = collect_vocabalign(args.vocabalign_dir)
    print(f"[significance] Found {len(va)} (model, dataset, frac) groups")

    print(f"[significance] Scanning RL logs: {args.rl_dir}")
    rl = collect_rl(args.rl_dir)
    print(f"[significance] Found {len(rl)} (dataset, frac) RL groups")

    rows = []
    for (model_tag, dataset_lower, frac), va_seeds in sorted(va.items()):
        rl_key = (dataset_lower, frac)
        if rl_key not in rl:
            continue
        rl_seeds = rl[rl_key]

        # Only use seeds present in both
        common_seeds = sorted(set(va_seeds) & set(rl_seeds))
        if len(common_seeds) < args.min_seeds:
            continue

        va_profits = [va_seeds[s] for s in common_seeds]
        rl_profits = [rl_seeds[s] for s in common_seeds]

        delta, ci_lo, ci_hi = bootstrap_paired_ci(va_profits, rl_profits)
        p_value = wilcoxon_pvalue(va_profits, rl_profits)
        sig = "*" if p_value < 0.05 else ("†" if p_value < 0.10 else "")

        rows.append({
            "backbone": model_tag,
            "dataset": dataset_lower,
            "frac": frac,
            "n_seeds": len(common_seeds),
            "mean_vocabalign": float(np.mean(va_profits)),
            "mean_rl": float(np.mean(rl_profits)),
            "delta": delta,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_value": p_value,
            "sig": sig,
        })

    if not rows:
        print("[significance] No matching (VocabAlign, RL) pairs found. "
              "Check that seed subdirectories exist in both log dirs.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n[significance] Saved {len(df)} rows → {args.out}")

    # ── LaTeX fragment ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("LaTeX table fragment (VocabAlign vs RL, best backbone per dataset/frac):")
    print("="*70)
    best = (
        df.sort_values("delta", ascending=False)
          .groupby(["dataset", "frac"])
          .first()
          .reset_index()
    )
    print(r"\begin{tabular}{llrrrr}")
    print(r"Dataset & Frac & $\Delta$ profit & 95\% CI & $p$ & \\")
    print(r"\hline")
    for _, row in best.iterrows():
        ci_str = f"[{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]"
        p_str = f"{row['p_value']:.3f}" if not np.isnan(row['p_value']) else "n/a"
        print(f"{row['dataset']} & {row['frac']} & {row['delta']:+.4f} & "
              f"{ci_str} & {p_str} & {row['sig']} \\\\")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
