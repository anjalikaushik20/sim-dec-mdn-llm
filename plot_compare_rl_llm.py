"""
Compare RL baseline vs one or more LLM variants across training fractions.

Produces:
  1. Per-dataset plots   — one figure per dataset, metric subplots, RL vs LLM lines
  2. Per-model plots     — one figure per LLM model, all datasets, RL vs LLM lines
  3. Frac-to-frac plots  — grouped bars at each frac, all datasets × methods
  4. Text summary table  — printed and saved as compare_table.txt

Usage:
    python3 plot_compare_rl_llm.py \\
        --rl  <rl_run_dir> \\
        --llm <label>:<llm_run_dir> [--llm <label2>:<dir2> ...] \\
        --out <output_dir>

Example:
    python3 plot_compare_rl_llm.py \\
        --rl  /data/akaush39/sim-to-dec/output/latest_output/sample_efficiency/rl_baseline/20260520_182054 \\
        --llm "Qwen3-1.7B attnpool":/data/akaush39/sim-to-dec/output/latest_output/sample_efficiency/qwen3-1.7B/20260520_200221 \\
        --llm "Qwen3-1.7B vocabalign":/data/akaush39/sim-to-dec/output/latest_output/sample_efficiency/all_vocabalign/20260521_220458/qwen3-1.7B \\
        --out /data/akaush39/sim-to-dec/output/latest_output/comparisons/rl_vs_llm
"""

import sys
import os
import re
import glob
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Parsing ───────────────────────────────────────────────────────────────────

LOG_PAT = re.compile(r"([a-z]+)_frac([\d.]+)\.log$")

def parse_llm_log(path):
    """LLM logs: key=value format."""
    m = {}
    for key, pat in [
        ("profit",   re.compile(r"best_profit=([\d.]+)")),
        ("on_time",  re.compile(r"best_on_time=([\d.]+)")),
        ("accuracy", re.compile(r"best_dm_accuracy_true=([\d.]+)")),
    ]:
        with open(path) as f:
            for line in f:
                hit = pat.search(line)
                if hit:
                    m[key] = float(hit.group(1))
    return m if len(m) == 3 else None


def parse_rl_log(path):
    """RL logs: key<space>value format."""
    m = {}
    for key, pat in [
        ("profit",   re.compile(r"best_profit\s+([\d.]+)")),
        ("on_time",  re.compile(r"best_on_time\s+([\d.]+)")),
        ("accuracy", re.compile(r"best_dm_accuracy\s+([\d.]+)")),
    ]:
        with open(path) as f:
            for line in f:
                hit = pat.search(line)
                if hit:
                    m[key] = float(hit.group(1))
    return m if len(m) == 3 else None


def load_dir(run_dir, parser_fn):
    """Returns dict: dataset -> sorted list of (frac, profit, on_time, accuracy)."""
    results = defaultdict(list)
    for path in glob.glob(os.path.join(run_dir, "*.log")):
        fname = os.path.basename(path)
        m = LOG_PAT.match(fname)
        if not m:
            continue
        dataset, frac = m.group(1), float(m.group(2))
        metrics = parser_fn(path)
        if metrics is None:
            print(f"  [skip] {fname} — incomplete")
            continue
        results[dataset].append((frac, metrics["profit"], metrics["on_time"], metrics["accuracy"]))
    for ds in results:
        results[ds].sort(key=lambda x: x[0])
    return results


# ── Style ─────────────────────────────────────────────────────────────────────

DATASET_LABELS = {"dataco": "DataCo", "globalstore": "GlobalStore", "oas": "OAS"}

METRICS = [
    ("profit",   "Profit"),
    ("on_time",  "On-Time Ratio"),
    ("sum",      "Profit + On-Time"),
    ("accuracy", "Accuracy"),
]

RL_STYLE   = dict(color="#E53935", linewidth=2.5, linestyle="--", marker="s", markersize=6)
LLM_COLORS = ["#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1", "#F4511E"]

def frac_label(f):
    return f"{int(round(f*100))}%" if f > 0 else "0\n(zero-shot)"

def get_values(rows, key):
    fracs = [r[0] for r in rows]
    if key == "profit":   vals = [r[1] for r in rows]
    elif key == "on_time": vals = [r[2] for r in rows]
    elif key == "sum":     vals = [r[1]+r[2] for r in rows]
    else:                  vals = [r[3] for r in rows]
    return fracs, vals


# ── Plot 1: Per-dataset ───────────────────────────────────────────────────────

def plot_per_dataset(rl_data, llm_data_list, out_dir):
    """One figure per dataset: 4 metric subplots, RL dashed vs each LLM solid."""
    all_datasets = sorted({ds for data, _ in llm_data_list for ds in data} | set(rl_data))

    for ds in all_datasets:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
        dset_label = DATASET_LABELS.get(ds, ds)

        for ax, (key, metric_label) in zip(axes, METRICS):
            # RL line
            if ds in rl_data:
                fracs, vals = get_values(rl_data[ds], key)
                ax.plot(fracs, vals, label="RL baseline", **RL_STYLE)

            # LLM lines
            for i, (llm_data, llm_label) in enumerate(llm_data_list):
                if ds not in llm_data:
                    continue
                fracs, vals = get_values(llm_data[ds], key)
                color = LLM_COLORS[i % len(LLM_COLORS)]
                ax.plot(fracs, vals, label=llm_label, color=color,
                        linewidth=2, marker="o", markersize=6)

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            all_fracs = sorted({r[0] for rows in ([rl_data.get(ds, [])]
                                + [d.get(ds, []) for d, _ in llm_data_list])
                                for r in rows})
            ax.set_xticks(all_fracs)
            ax.set_xticklabels([frac_label(f) for f in all_fracs], fontsize=7)

        fig.suptitle(f"{dset_label} — RL vs LLM", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(out_dir, f"compare_dataset_{ds}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ── Plot 2: Per-model ─────────────────────────────────────────────────────────

def plot_per_model(rl_data, llm_data_list, out_dir):
    """One figure per LLM model: 4 metric subplots, all datasets, RL vs this LLM."""
    ds_colors = {"dataco": "#2196F3", "globalstore": "#FF5722", "oas": "#4CAF50"}

    for i, (llm_data, llm_label) in enumerate(llm_data_list):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
        safe_label = re.sub(r"[^a-z0-9]+", "_", llm_label.lower())

        for ax, (key, metric_label) in zip(axes, METRICS):
            for ds in sorted(llm_data):
                dset_label = DATASET_LABELS.get(ds, ds)
                color = ds_colors.get(ds, "#555")

                # RL dashed
                if ds in rl_data:
                    fracs, vals = get_values(rl_data[ds], key)
                    ax.plot(fracs, vals, color=color, linewidth=2, linestyle="--",
                            marker="s", markersize=5, alpha=0.7,
                            label=f"{dset_label} RL")

                # LLM solid
                fracs, vals = get_values(llm_data[ds], key)
                ax.plot(fracs, vals, color=color, linewidth=2.5, linestyle="-",
                        marker="o", markersize=6,
                        label=f"{dset_label} LLM")

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            all_fracs = sorted({r[0] for ds in llm_data for r in llm_data[ds]}
                               | {r[0] for ds in rl_data for r in rl_data[ds]})
            ax.set_xticks(all_fracs)
            ax.set_xticklabels([frac_label(f) for f in all_fracs], fontsize=7)

        fig.suptitle(f"{llm_label} — RL (dashed) vs LLM (solid)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(out_dir, f"compare_model_{safe_label}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ── Plot 3: Frac-to-frac ─────────────────────────────────────────────────────

def plot_frac_to_frac(rl_data, llm_data_list, out_dir):
    """For each frac: grouped bar chart, x=dataset, groups=methods, 4 metric subplots."""
    all_fracs = sorted({r[0] for ds in rl_data for r in rl_data[ds]}
                       | {r[0] for data, _ in llm_data_list for ds in data for r in data[ds]})
    all_datasets = sorted({ds for ds in rl_data} | {ds for data, _ in llm_data_list for ds in data})

    methods = [("RL baseline", rl_data, "#E53935")] + \
              [(label, data, LLM_COLORS[i % len(LLM_COLORS)])
               for i, (data, label) in enumerate(llm_data_list)]

    n_methods = len(methods)
    x = np.arange(len(all_datasets))
    width = 0.8 / n_methods

    for frac in all_fracs:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)

        for ax, (key, metric_label) in zip(axes, METRICS):
            for j, (method_label, data, color) in enumerate(methods):
                vals = []
                for ds in all_datasets:
                    row = next((r for r in data.get(ds, []) if r[0] == frac), None)
                    if row is None:
                        vals.append(0.0)
                    elif key == "profit":   vals.append(row[1])
                    elif key == "on_time":  vals.append(row[2])
                    elif key == "sum":      vals.append(row[1] + row[2])
                    else:                   vals.append(row[3])

                offset = (j - (n_methods-1)/2) * width
                ax.bar(x + offset, vals, width * 0.9, label=method_label,
                       color=color, alpha=0.85, edgecolor="white")

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in all_datasets], fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

        pct = frac_label(frac)
        fig.suptitle(f"Frac = {pct} — RL vs LLM", fontsize=13, fontweight="bold")
        plt.tight_layout()
        frac_str = f"{int(round(frac*100)):03d}"
        path = os.path.join(out_dir, f"compare_frac{frac_str}pct.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ── Text table ────────────────────────────────────────────────────────────────

def print_and_save_table(rl_data, llm_data_list, out_dir):
    all_datasets = sorted({ds for ds in rl_data}
                          | {ds for data, _ in llm_data_list for ds in data})
    all_fracs = sorted({r[0] for ds in rl_data for r in rl_data[ds]}
                       | {r[0] for data, _ in llm_data_list for ds in data for r in data[ds]})

    methods = [("RL", rl_data)] + [(lbl, d) for d, lbl in llm_data_list]

    lines = []
    for ds in all_datasets:
        dset_label = DATASET_LABELS.get(ds, ds)
        header = f"\n{'='*80}\n{dset_label}\n{'='*80}"
        col_w = 12
        sub = f"{'Frac':<8}" + "".join(
            f"{'Profit':>{col_w}}{'OnTime':>{col_w}}{'P+O':>{col_w}}{'Acc':>{col_w}}"
            for _ in methods
        )
        method_header = f"{'':8}" + "".join(
            f"{lbl:>{col_w*4}}" for lbl, _ in methods
        )
        lines.append(header)
        lines.append(method_header)
        lines.append(sub)
        lines.append("-" * len(sub))

        for frac in all_fracs:
            row_str = f"{frac_label(frac).replace(chr(10),' '):<8}"
            for _, data in methods:
                entry = next((r for r in data.get(ds, []) if r[0] == frac), None)
                if entry:
                    row_str += f"{entry[1]:>{col_w}.4f}{entry[2]:>{col_w}.4f}{entry[1]+entry[2]:>{col_w}.4f}{entry[3]:>{col_w}.4f}"
                else:
                    row_str += f"{'—':>{col_w}}{'—':>{col_w}}{'—':>{col_w}}{'—':>{col_w}}"
            lines.append(row_str)

    table_str = "\n".join(lines)
    print(table_str)
    path = os.path.join(out_dir, "compare_table.txt")
    with open(path, "w") as f:
        f.write(table_str + "\n")
    print(f"\nSaved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rl",  required=True, help="RL baseline run directory")
    parser.add_argument("--llm", action="append", required=True,
                        metavar="LABEL:DIR",
                        help="LLM run dir with label, e.g. 'Qwen3-1.7B:path/to/dir'")
    parser.add_argument("--out", required=True, help="Output directory for plots and tables")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading RL data from: {args.rl}")
    rl_data = load_dir(args.rl, parse_rl_log)

    llm_data_list = []
    for entry in args.llm:
        if ":" not in entry:
            print(f"Error: --llm must be LABEL:DIR, got: {entry}")
            sys.exit(1)
        label, path = entry.split(":", 1)
        print(f"Loading LLM '{label}' from: {path}")
        data = load_dir(path, parse_llm_log)
        llm_data_list.append((data, label))

    print(f"\nRL datasets:  {sorted(rl_data.keys())}")
    for data, label in llm_data_list:
        print(f"LLM '{label}': {sorted(data.keys())}")

    print("\n--- Generating plots ---")
    plot_per_dataset(rl_data, llm_data_list, args.out)
    plot_per_model(rl_data, llm_data_list, args.out)
    plot_frac_to_frac(rl_data, llm_data_list, args.out)

    print("\n--- Summary Table ---")
    print_and_save_table(rl_data, llm_data_list, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
