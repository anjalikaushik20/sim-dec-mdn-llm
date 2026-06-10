"""
Compare VocabAlign OOD sample efficiency results across all models on DataCo_OOD.

Mirrors plot_vocabalign_compare.py but adapted for the OOD directory structure:
  {run_dir}/{model_dir}/dataco_ood_frac{frac}.log   (flat, no seed subdirs)

Produces:
  - vocabalign_ood_dataset_dataco_ood.png   (all models, 3 metrics, RL dashed line)
  - vocabalign_ood_model_{model}.png        (one per model, RL dashed line)
  - vocabalign_ood_frac{N}pct.png           (bar chart per frac)
  - compare_table.txt

Usage:
    conda run -n simenv python3 plot_vocabalign_ood_compare.py \\
        --run_dir output/decision_maker/all_fracs/models_ood \\
        --rl_dir  output/decision_maker/rl/results_ood
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Parsing ───────────────────────────────────────────────────────────────────

LOG_PAT = re.compile(r"dataco_ood_frac([\d.]+)\.log$")
DATASET = "dataco_ood"
DATASET_LABEL = "DataCo OOD"

# directory name in models_ood → canonical model tag
DIR_TO_TAG = {
    "gpt2":      "gpt2",
    "gpt2_large": "gpt2-large",
    "phi4_mini": "phi4-mini",
    "qwen3-0.6": "qwen3-0.6B",
    "qwen3-1.7": "qwen3-1.7B",
}

MODEL_ORDER = ["gpt2", "gpt2-large", "qwen3-0.6B", "qwen3-1.7B", "phi4-mini"]
MODEL_LABELS = {
    "gpt2":       "GPT-2",
    "gpt2-large": "GPT-2 Large",
    "qwen3-0.6B": "Qwen3-0.6B",
    "qwen3-1.7B": "Qwen3-1.7B",
    "phi4-mini":  "Phi4-mini",
}
MODEL_COLORS = {
    "gpt2":       "#90A4AE",
    "gpt2-large": "#263238",
    "qwen3-0.6B": "#81D4FA",
    "qwen3-1.7B": "#0288D1",
    "phi4-mini":  "#01579B",
}
RL_COLOR  = "#E53935"
RL_STYLE  = dict(color=RL_COLOR, linewidth=2, linestyle="--")

METRICS = [
    ("profit",  "Profit"),
    ("on_time", "On-Time Ratio"),
    ("sum",     "Profit + On-Time"),
]


def parse_log(path):
    m = {}
    pats = {
        "profit":  re.compile(r"best_profit[=\s]+([\d.eE+\-]+)"),
        "on_time": re.compile(r"best_on_time[=\s]+([\d.eE+\-]+)"),
    }
    try:
        with open(path) as f:
            for line in f:
                for key, pat in pats.items():
                    hit = pat.search(line)
                    if hit:
                        m[key] = float(hit.group(1))
    except OSError:
        pass
    return m if len(m) == 2 else None


def load_models(run_dir):
    """Returns {model_tag: sorted list of (frac, profit, on_time)}"""
    all_data = {}
    for entry in sorted(os.listdir(run_dir)):
        model_dir = os.path.join(run_dir, entry)
        if not os.path.isdir(model_dir):
            continue
        tag = DIR_TO_TAG.get(entry, entry)
        rows = []
        for path in glob.glob(os.path.join(model_dir, "dataco_ood_frac*.log")):
            m = LOG_PAT.match(os.path.basename(path))
            if not m:
                continue
            frac    = float(m.group(1))
            metrics = parse_log(path)
            if metrics is None:
                print(f"  [skip] {tag}/dataco_ood_frac{frac:.2f}.log — incomplete")
                continue
            rows.append((frac, metrics["profit"], metrics["on_time"]))
        rows.sort(key=lambda x: x[0])
        if rows:
            all_data[tag] = rows
    return all_data


def load_rl(rl_dir):
    """Returns sorted list of (frac, profit, on_time) from rl_dir."""
    rows = []
    for path in glob.glob(os.path.join(rl_dir, "dataco_ood_frac*.log")):
        m = re.match(r"dataco_ood_frac([\d.]+)\.log$", os.path.basename(path))
        if not m:
            continue
        frac    = float(m.group(1))
        metrics = parse_log(path)
        if metrics is None:
            print(f"  [skip] RL dataco_ood_frac{frac:.2f}.log — incomplete")
            continue
        rows.append((frac, metrics["profit"], metrics["on_time"]))
    rows.sort(key=lambda x: x[0])
    return rows


# ── Helpers ───────────────────────────────────────────────────────────────────

def sorted_models(tags):
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(tags, key=lambda m: order.get(m, 99))


def frac_label(f):
    return f"{int(round(f * 100))}%"


def get_values(rows, key):
    fracs = [r[0] for r in rows]
    if   key == "profit":  vals = [r[1] for r in rows]
    elif key == "on_time": vals = [r[2] for r in rows]
    else:                  vals = [r[1] + r[2] for r in rows]
    return fracs, vals


# ── Plot 1: Per-dataset — all models as lines ─────────────────────────────────

def plot_per_dataset(all_data, rl_rows, out_dir):
    models    = sorted_models(all_data.keys())
    all_fracs = sorted({r[0] for rows in all_data.values() for r in rows})

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=False)

    for ax, (key, metric_label) in zip(axes, METRICS):
        for model in models:
            fracs, vals = get_values(all_data[model], key)
            ax.plot(fracs, vals, marker="o",
                    color=MODEL_COLORS.get(model, "#555"),
                    linewidth=2, markersize=6,
                    label=MODEL_LABELS.get(model, model))

        if rl_rows:
            fracs_rl, vals_rl = get_values(rl_rows, key)
            ax.plot(fracs_rl, vals_rl, marker="o", **RL_STYLE,
                    markersize=6, label="RL")

        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Training Fraction", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(all_fracs)
        ax.set_xticklabels([frac_label(f) for f in all_fracs], fontsize=7)

    fig.suptitle(f"{DATASET_LABEL} — VocabAlign: All Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"vocabalign_ood_dataset_{DATASET}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 2: Per-model — with RL dashed reference ──────────────────────────────

def plot_per_model(all_data, rl_rows, out_dir):
    models    = sorted_models(all_data.keys())
    all_fracs = sorted({r[0] for rows in all_data.values() for r in rows})

    for model in models:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=False)
        model_label = MODEL_LABELS.get(model, model)

        for ax, (key, metric_label) in zip(axes, METRICS):
            fracs, vals = get_values(all_data[model], key)
            ax.plot(fracs, vals, marker="o",
                    color=MODEL_COLORS.get(model, "#555"),
                    linewidth=2, markersize=6,
                    label=model_label)

            if rl_rows:
                fracs_rl, vals_rl = get_values(rl_rows, key)
                ax.plot(fracs_rl, vals_rl, marker="o", **RL_STYLE,
                        markersize=6, label="RL")

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(all_fracs)
            ax.set_xticklabels([frac_label(f) for f in all_fracs], fontsize=7)

        fig.suptitle(f"{model_label} (VocabAlign OOD) — {DATASET_LABEL}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        safe  = re.sub(r"[^a-z0-9]+", "_", model.lower())
        path  = os.path.join(out_dir, f"vocabalign_ood_model_{safe}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ── Plot 3: Frac-to-frac — grouped bars per model ────────────────────────────

def plot_frac_to_frac(all_data, rl_rows, out_dir):
    models    = sorted_models(all_data.keys())
    all_fracs = sorted({r[0] for rows in all_data.values() for r in rows})
    x         = np.arange(len(models))

    for frac in all_fracs:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, (key, metric_label) in zip(axes, METRICS):
            vals = []
            for model in models:
                row = next((r for r in all_data[model] if r[0] == frac), None)
                if row is None:
                    vals.append(0.0)
                elif key == "profit":  vals.append(row[1])
                elif key == "on_time": vals.append(row[2])
                else:                  vals.append(row[1] + row[2])

            colors = [MODEL_COLORS.get(m, "#555") for m in models]
            ax.bar(x, vals, 0.6, color=colors, alpha=0.85, edgecolor="white")

            if rl_rows:
                rl_row = next((r for r in rl_rows if r[0] == frac), None)
                if rl_row:
                    rl_val = rl_row[1] if key == "profit" else (rl_row[2] if key == "on_time" else rl_row[1] + rl_row[2])
                    ax.axhline(rl_val, **RL_STYLE, label=f"RL: {rl_val:.4f}")
                    ax.legend(fontsize=8)

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                               fontsize=7, rotation=20, ha="right")
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"Frac = {frac_label(frac)} — VocabAlign OOD All Models",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        frac_str = f"{int(round(frac * 100)):03d}"
        path = os.path.join(out_dir, f"vocabalign_ood_frac{frac_str}pct.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ── Text table ────────────────────────────────────────────────────────────────

def print_and_save_table(all_data, rl_rows, out_dir):
    models    = sorted_models(all_data.keys())
    all_fracs = sorted({r[0] for rows in all_data.values() for r in rows})
    col_w     = 10

    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(DATASET_LABEL)
    lines.append("=" * 80)

    header = f"{'':8}"
    for model in models:
        header += f"{MODEL_LABELS.get(model, model):>{col_w * 3}}"
    if rl_rows:
        header += f"{'RL':>{col_w * 3}}"
    lines.append(header)

    sub = f"{'Frac':<8}"
    for _ in range(len(models) + (1 if rl_rows else 0)):
        sub += f"{'Profit':>{col_w}}{'OnTime':>{col_w}}{'P+O':>{col_w}}"
    lines.append(sub)
    lines.append("-" * len(sub))

    for frac in all_fracs:
        row = f"{frac_label(frac):<8}"
        for model in models:
            entry = next((r for r in all_data[model] if r[0] == frac), None)
            if entry:
                row += f"{entry[1]:>{col_w}.4f}{entry[2]:>{col_w}.4f}{entry[1]+entry[2]:>{col_w}.4f}"
            else:
                row += f"{'—':>{col_w}}{'—':>{col_w}}{'—':>{col_w}}"
        if rl_rows:
            rl_entry = next((r for r in rl_rows if r[0] == frac), None)
            if rl_entry:
                row += f"{rl_entry[1]:>{col_w}.4f}{rl_entry[2]:>{col_w}.4f}{rl_entry[1]+rl_entry[2]:>{col_w}.4f}"
            else:
                row += f"{'—':>{col_w}}{'—':>{col_w}}{'—':>{col_w}}"
        lines.append(row)

    table_str = "\n".join(lines)
    print(table_str)
    path = os.path.join(out_dir, "compare_table.txt")
    with open(path, "w") as f:
        f.write(table_str + "\n")
    print(f"\nSaved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default="output/decision_maker/all_fracs/models_ood")
    ap.add_argument("--rl_dir",  default="output/decision_maker/rl/results_ood")
    ap.add_argument("--out",     help="Output dir (defaults to --run_dir)")
    args = ap.parse_args()

    out_dir = args.out or args.run_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading VocabAlign OOD from: {args.run_dir}")
    all_data = load_models(args.run_dir)
    if not all_data:
        print("No completed logs found.")
        raise SystemExit(1)

    print(f"Models: {sorted_models(all_data.keys())}")
    for tag, rows in all_data.items():
        fracs = [r[0] for r in rows]
        print(f"  {MODEL_LABELS.get(tag, tag)}: fracs={fracs}")

    print(f"\nLoading RL OOD from: {args.rl_dir}")
    rl_rows = load_rl(args.rl_dir)
    if rl_rows:
        print(f"  RL fracs: {[r[0] for r in rl_rows]}")
    else:
        print("  No RL results found.")

    print("\n--- Generating plots ---")
    plot_per_dataset(all_data, rl_rows, out_dir)
    plot_per_model(all_data, rl_rows, out_dir)
    plot_frac_to_frac(all_data, rl_rows, out_dir)

    print("\n--- Summary Table ---")
    print_and_save_table(all_data, rl_rows, out_dir)

    print("\nDone.")
