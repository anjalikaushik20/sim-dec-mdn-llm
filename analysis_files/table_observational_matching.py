#!/usr/bin/env python3
"""
Table: Observational matching ATE results for DataCo, GlobalStore, OAS.

Reads {run_dir}/seed*/{dataset}_matching.log, aggregates ATE across seeds,
and writes a formatted text table.

Output: {run_dir}/observational_matching_table.txt

Usage:
  conda run -n simenv python3 table_observational_matching.py \\
      --run_dir output/exp1_observational_matching/20260606_113028
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np

DATASETS   = ["dataco", "globalstore", "oas"]
DS_DISPLAY = {"dataco": "DataCo", "globalstore": "GlobalStore", "oas": "OAS"}
OUTCOMES   = ["on_time_rate", "days_shipping"]
OUT_DISPLAY = {"on_time_rate": "On-Time Rate ATE", "days_shipping": "Days Shipping ATE"}

# Shipping class labels as they appear in logs
CLASS_PAT = re.compile(
    r"^\s+(Standard Class|Second Class|First Class|Same Day)\s+n=\s*(\d+)\s+"
    r"ATE=([+-]?\d+\.\d+)\s+\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]"
)
OUTCOME_PAT = re.compile(r"^(on_time_rate|days_shipping)\s+\[")
ATE_PAT     = re.compile(
    r"^\s+ATE\s*=\s*([+-]?\d+\.\d+)\s+95% CI \[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]"
)


def parse_matching_log(path):
    """Returns {outcome: {ate, ci_lo, ci_hi, classes: {name: (n, ate, lo, hi)}}}"""
    results = {}
    current_outcome = None
    with open(path) as f:
        for line in f:
            m = OUTCOME_PAT.match(line)
            if m:
                current_outcome = m.group(1)
                results[current_outcome] = {"classes": {}}
                continue
            if current_outcome is None:
                continue
            m = ATE_PAT.match(line)
            if m and "ate" not in results[current_outcome]:
                results[current_outcome]["ate"]   = float(m.group(1))
                results[current_outcome]["ci_lo"] = float(m.group(2))
                results[current_outcome]["ci_hi"] = float(m.group(3))
                continue
            m = CLASS_PAT.match(line)
            if m:
                cls = m.group(1)
                results[current_outcome]["classes"][cls] = {
                    "n":    int(m.group(2)),
                    "ate":  float(m.group(3)),
                    "lo":   float(m.group(4)),
                    "hi":   float(m.group(5)),
                }
    return results


def collect(run_dir):
    """Returns {dataset: {outcome: [(ate, ci_lo, ci_hi), ...], classes: {cls: [ate, ...]}}}"""
    data = {ds: {o: {"ates": [], "classes": defaultdict(list)}
                 for o in OUTCOMES}
            for ds in DATASETS}

    seed_dirs = sorted(glob.glob(os.path.join(run_dir, "seed*")))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed* directories found in {run_dir}")

    for seed_dir in seed_dirs:
        for ds in DATASETS:
            log = os.path.join(seed_dir, f"{ds}_matching.log")
            if not os.path.exists(log):
                print(f"  [skip] {log}")
                continue
            parsed = parse_matching_log(log)
            for outcome in OUTCOMES:
                if outcome not in parsed:
                    continue
                r = parsed[outcome]
                if "ate" not in r:
                    continue
                data[ds][outcome]["ates"].append(r["ate"])
                for cls, vals in r["classes"].items():
                    data[ds][outcome]["classes"][cls].append(vals["ate"])

    return data, len(seed_dirs)


def fmt_ate(ates):
    if not ates:
        return "—"
    mean = np.mean(ates)
    std  = np.std(ates, ddof=1) if len(ates) > 1 else 0.0
    sign = "+" if mean >= 0 else ""
    return f"{sign}{mean:.4f} ± {std:.4f} (n={len(ates)})"


def significance(ates):
    if not ates:
        return ""
    mean = np.mean(ates)
    std  = np.std(ates, ddof=1) if len(ates) > 1 else 0.0
    se   = std / np.sqrt(len(ates))
    lo   = mean - 1.96 * se
    hi   = mean + 1.96 * se
    if lo > 0:
        return "✓ better"
    if hi < 0:
        return "✗ adverse" if mean < 0 else "✓ better"
    return "~ n.s."


def build_table(run_dir):
    data, n_seeds = collect(run_dir)
    lines = []

    lines.append("=" * 100)
    lines.append("  Observational Matching — Average Treatment Effect (VocabAlign vs Historical Default)")
    lines.append(f"  Seeds: {n_seeds}   Run: {os.path.basename(run_dir)}")
    lines.append("=" * 100)

    for ds in DATASETS:
        display = DS_DISPLAY[ds]
        lines.append("")
        lines.append(f"── {display} " + "─" * (95 - len(display)))

        for outcome in OUTCOMES:
            ates = data[ds][outcome]["ates"]
            sig  = significance(ates)
            lines.append(f"  {OUT_DISPLAY[outcome]:30s}  {fmt_ate(ates):40s}  {sig}")

            # Per-class breakdown
            classes_data = data[ds][outcome]["classes"]
            if classes_data:
                for cls in ["Standard Class", "Second Class", "First Class", "Same Day"]:
                    cls_ates = classes_data.get(cls, [])
                    if not cls_ates:
                        continue
                    sig_cls = significance(cls_ates)
                    lines.append(f"    {cls:20s}  {fmt_ate(cls_ates):40s}  {sig_cls}")

        lines.append("")

    lines.append("=" * 100)
    lines.append("  ✓ better = 95% CI excludes 0 in favourable direction")
    lines.append("  ✗ adverse = 95% CI excludes 0 in adverse direction")
    lines.append("  ~ n.s.  = CI crosses 0, not statistically significant")
    lines.append("=" * 100)

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir",
                    default="output/exp1_observational_matching/20260606_113028")
    args = ap.parse_args()

    table = build_table(args.run_dir)
    print(table)

    out_path = os.path.join(args.run_dir, "observational_matching_table.txt")
    with open(out_path, "w") as f:
        f.write(table + "\n")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
