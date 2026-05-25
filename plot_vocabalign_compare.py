"""
Compare vocabalign sample efficiency results across all models and datasets.

Usage:
    python3 plot_vocabalign_compare.py <vocabalign_run_dir> [--out <output_dir>]

Example:
    python3 plot_vocabalign_compare.py \\
        /data/akaush39/sim-to-dec/output/latest_output/sample_efficiency/all_vocabalign/20260521_220458

Expects subdirectories named by model tag, each containing
{dataset}_frac{frac}.log files.
Saves plots and compare_table.txt in --out (defaults to <vocabalign_run_dir>).
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

def parse_log(path):
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


def load_all_models(run_dir):
    """Returns dict: model_tag -> dataset -> sorted list of (frac, profit, on_time, acc)"""
    all_data = {}
    for model_dir in sorted(glob.glob(os.path.join(run_dir, "*"))):
        if not os.path.isdir(model_dir):
            continue
        model_tag = os.path.basename(model_dir)
        if model_tag in ("ckpts", "job_queue.txt", "queue.lock"):
            continue
        ds_data = defaultdict(list)
        for path in glob.glob(os.path.join(model_dir, "*.log")):
            fname = os.path.basename(path)
            m = LOG_PAT.match(fname)
            if not m:
                continue
            dataset, frac = m.group(1), float(m.group(2))
            metrics = parse_log(path)
            if metrics is None:
                print(f"  [skip] {model_tag}/{fname} — incomplete")
                continue
            ds_data[dataset].append((frac, metrics["profit"], metrics["on_time"], metrics["accuracy"]))
        for ds in ds_data:
            ds_data[ds].sort(key=lambda x: x[0])
        if ds_data:
            all_data[model_tag] = dict(ds_data)
    return all_data


# ── Style ─────────────────────────────────────────────────────────────────────

DATASET_LABELS = {"dataco": "DataCo", "globalstore": "GlobalStore", "oas": "OAS"}

MODEL_ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "qwen3-0.6B", "qwen3-1.7B", "qwen3-4B"]
MODEL_LABELS = {
    "gpt2":        "GPT-2",
    "gpt2-medium": "GPT-2 Med",
    "gpt2-large":  "GPT-2 Lg",
    "qwen3-0.6B":  "Qwen3-0.6B",
    "qwen3-1.7B":  "Qwen3-1.7B",
    "qwen3-4B":    "Qwen3-4B",
}
MODEL_COLORS = {
    "gpt2":        "#90A4AE",
    "gpt2-medium": "#546E7A",
    "gpt2-large":  "#263238",
    "qwen3-0.6B":  "#81D4FA",
    "qwen3-1.7B":  "#0288D1",
    "qwen3-4B":    "#01579B",
}
DATASET_COLORS = {"dataco": "#2196F3", "globalstore": "#FF5722", "oas": "#4CAF50"}

METRICS = [
    ("profit",   "Profit"),
    ("on_time",  "On-Time Ratio"),
    ("sum",      "Profit + On-Time"),
    ("accuracy", "Accuracy (vs Optimal)"),
]

def sorted_models(tags):
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(tags, key=lambda m: order.get(m, 99))

def frac_label(f):
    return f"{int(round(f*100))}%" if f > 0 else "0\n(zero-shot)"

def get_values(rows, key):
    fracs = [r[0] for r in rows]
    if   key == "profit":  vals = [r[1] for r in rows]
    elif key == "on_time": vals = [r[2] for r in rows]
    elif key == "sum":     vals = [r[1]+r[2] for r in rows]
    else:                  vals = [r[3] for r in rows]
    return fracs, vals


# ── Plot 1: Per-dataset — all models as lines ─────────────────────────────────

def plot_per_dataset(all_data, out_dir):
    all_datasets = sorted({ds for m in all_data.values() for ds in m})
    models = sorted_models(all_data.keys())

    for ds in all_datasets:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
        dset_label = DATASET_LABELS.get(ds, ds)

        for ax, (key, metric_label) in zip(axes, METRICS):
            for model in models:
                if ds not in all_data[model]:
                    continue
                rows = all_data[model][ds]
                fracs, vals = get_values(rows, key)
                color = MODEL_COLORS.get(model, "#555")
                label = MODEL_LABELS.get(model, model)
                ax.plot(fracs, vals, marker="o", color=color, linewidth=2,
                        markersize=6, label=label)

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            all_fracs = sorted({r[0] for m in all_data
                                 for r in all_data[m].get(ds, [])})
            ax.set_xticks(all_fracs)
            ax.set_xticklabels([frac_label(f) for f in all_fracs], fontsize=7)

        fig.suptitle(f"{dset_label} — Vocabalign: All Models", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(out_dir, f"vocabalign_dataset_{ds}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ── Plot 2: Per-model — all datasets as lines ─────────────────────────────────

def plot_per_model(all_data, out_dir):
    models = sorted_models(all_data.keys())

    for model in models:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
        model_label = MODEL_LABELS.get(model, model)

        for ax, (key, metric_label) in zip(axes, METRICS):
            for ds, rows in sorted(all_data[model].items()):
                fracs, vals = get_values(rows, key)
                color = DATASET_COLORS.get(ds, "#555")
                ax.plot(fracs, vals, marker="o", color=color, linewidth=2,
                        markersize=6, label=DATASET_LABELS.get(ds, ds))

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            all_fracs = sorted({r[0] for rows in all_data[model].values() for r in rows})
            ax.set_xticks(all_fracs)
            ax.set_xticklabels([frac_label(f) for f in all_fracs], fontsize=7)

        fig.suptitle(f"{model_label} (Vocabalign) — All Datasets", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(out_dir, f"vocabalign_model_{model}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ── Plot 3: Frac-to-frac — grouped bars, x=model, color=dataset ──────────────

def plot_frac_to_frac(all_data, out_dir):
    models = sorted_models(all_data.keys())
    all_datasets = sorted({ds for m in all_data.values() for ds in m})
    all_fracs = sorted({r[0] for m in all_data
                        for ds in all_data[m].values() for r in ds})

    x = np.arange(len(models))
    n_ds = len(all_datasets)
    width = 0.8 / n_ds

    for frac in all_fracs:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        for ax, (key, metric_label) in zip(axes, METRICS):
            for j, ds in enumerate(all_datasets):
                vals = []
                for model in models:
                    row = next((r for r in all_data[model].get(ds, []) if r[0] == frac), None)
                    if row is None:
                        vals.append(0.0)
                    elif key == "profit":  vals.append(row[1])
                    elif key == "on_time": vals.append(row[2])
                    elif key == "sum":     vals.append(row[1]+row[2])
                    else:                  vals.append(row[3])

                offset = (j - (n_ds-1)/2) * width
                ax.bar(x + offset, vals, width * 0.9,
                       label=DATASET_LABELS.get(ds, ds),
                       color=DATASET_COLORS.get(ds, "#555"),
                       alpha=0.85, edgecolor="white")

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                               fontsize=7, rotation=20, ha="right")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"Frac = {frac_label(frac)} — Vocabalign All Models",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        frac_str = f"{int(round(frac*100)):03d}"
        path = os.path.join(out_dir, f"vocabalign_frac{frac_str}pct.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ── Text table ────────────────────────────────────────────────────────────────

def print_and_save_table(all_data, out_dir):
    models = sorted_models(all_data.keys())
    all_datasets = sorted({ds for m in all_data.values() for ds in m})
    all_fracs = sorted({r[0] for m in all_data
                        for ds in all_data[m].values() for r in ds})

    col_w = 10
    lines = []

    for ds in all_datasets:
        dset_label = DATASET_LABELS.get(ds, ds)
        lines.append(f"\n{'='*80}")
        lines.append(dset_label)
        lines.append("="*80)

        # Header: model names spanning 4 columns each
        model_header = f"{'':8}"
        for model in models:
            label = MODEL_LABELS.get(model, model)
            model_header += f"{label:>{col_w*4}}"
        lines.append(model_header)

        # Sub-header: metric names per model
        sub = f"{'Frac':<8}"
        for _ in models:
            sub += f"{'Profit':>{col_w}}{'OnTime':>{col_w}}{'P+O':>{col_w}}{'Acc':>{col_w}}"
        lines.append(sub)
        lines.append("-" * len(sub))

        # Rows
        for frac in all_fracs:
            row = f"{frac_label(frac).replace(chr(10), ' '):<8}"
            for model in models:
                entry = next((r for r in all_data[model].get(ds, []) if r[0] == frac), None)
                if entry:
                    row += f"{entry[1]:>{col_w}.4f}{entry[2]:>{col_w}.4f}{entry[1]+entry[2]:>{col_w}.4f}{entry[3]:>{col_w}.4f}"
                else:
                    row += f"{'—':>{col_w}}{'—':>{col_w}}{'—':>{col_w}}{'—':>{col_w}}"
            lines.append(row)

    table_str = "\n".join(lines)
    print(table_str)
    path = os.path.join(out_dir, "compare_table.txt")
    with open(path, "w") as f:
        f.write(table_str + "\n")
    print(f"\nSaved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Vocabalign run directory with per-model subdirs")
    parser.add_argument("--out", help="Output directory (defaults to run_dir)")
    args = parser.parse_args()

    run_dir = args.run_dir.rstrip("/")
    out_dir = args.out or run_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading from: {run_dir}")
    all_data = load_all_models(run_dir)

    if not all_data:
        print("No completed logs found.")
        sys.exit(1)

    print(f"Models: {sorted_models(all_data.keys())}")
    for model, ds_map in all_data.items():
        fracs = sorted({r[0] for rows in ds_map.values() for r in rows})
        print(f"  {MODEL_LABELS.get(model, model)}: datasets={sorted(ds_map)}, fracs={fracs}")

    print("\n--- Generating plots ---")
    plot_per_dataset(all_data, out_dir)
    plot_per_model(all_data, out_dir)
    plot_frac_to_frac(all_data, out_dir)

    print("\n--- Summary Table ---")
    print_and_save_table(all_data, out_dir)

    print("\nDone.")
