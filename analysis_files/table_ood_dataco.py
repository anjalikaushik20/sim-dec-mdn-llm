#!/usr/bin/env python3
"""
Table: DataCo OOD at frac=1.00 for GPT-2, Qwen3-1.7B, Phi4-mini + RL baseline.

Reads single-run logs (no seeds):
  output/decision_maker/all_fracs/models_ood/{dir_name}/dataco_ood_frac1.00.log
  output/decision_maker/rl/results_ood/dataco_ood_frac1.00.log

Output: output/tables/ood_dataco_frac100.txt
"""

import argparse
import os
import re

NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

MODELS = [
    ("GPT-2",      "gpt2"),
    ("Qwen3-1.7B", "qwen3-1.7"),
    ("Phi4-mini",  "phi4_mini"),
]


def parse_log(path):
    profit, on_time = None, None
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            m = re.search(r"best_profit[=\s]+" + NUM, line)
            if m:
                profit = float(m.group(1))
            m = re.search(r"best_on_time[=\s]+" + NUM, line)
            if m:
                on_time = float(m.group(1))
    if profit is None or on_time is None:
        return None
    return profit, on_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood_dir", default="output/decision_maker/all_fracs/models_ood")
    ap.add_argument("--rl_dir",  default="output/decision_maker/rl/results_ood")
    ap.add_argument("--out",     default="output/tables")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    rows = []

    # RL baseline
    rl_log = os.path.join(args.rl_dir, "dataco_ood_frac1.00.log")
    r = parse_log(rl_log)
    if r:
        rows.append(("RL", r[0], r[1]))
    else:
        print(f"  [warn] RL log not found or incomplete: {rl_log}")

    # VocabAlign models
    for label, dir_name in MODELS:
        log = os.path.join(args.ood_dir, dir_name, "dataco_ood_frac1.00.log")
        r = parse_log(log)
        if r:
            rows.append((label, r[0], r[1]))
        else:
            print(f"  [warn] log not found or incomplete: {log}")

    # Format table
    sep  = "=" * 54
    dash = "-" * 54
    lines = [
        sep,
        "  DataCo OOD — frac = 1.00  (single run, no seeds)",
        sep,
        f"  {'Method':<16}  {'Profit':>10}  {'On-Time':>10}  {'Sum':>10}",
        dash,
    ]
    for label, profit, on_time in rows:
        lines.append(
            f"  {label:<16}  {profit:>10.4f}  {on_time:>10.4f}  {profit+on_time:>10.4f}"
        )
    lines.append(sep)

    table = "\n".join(lines)
    print(table)

    out_path = os.path.join(args.out, "ood_dataco_frac100.txt")
    with open(out_path, "w") as f:
        f.write(table + "\n")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
