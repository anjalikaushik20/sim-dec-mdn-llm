"""
Compare RL baseline vs all VocabAlign LLM models across training fractions.

Produces:
  1. Per-dataset plots   — one figure per dataset, metric subplots, RL vs LLM lines
  2. Per-model plots     — one figure per LLM model, all datasets, RL vs LLM lines
  3. Frac-to-frac plots  — grouped bars at each frac, all datasets × methods
  4. Text summary table  — printed and saved as compare_table.txt
"""

import os
import re
import glob
import argparse
from collections import defaultdict

# ── Hardcoded paths ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--GPU", type=int, default=0)
_args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.GPU)

RL_DIR      = "output/decision_maker/rl/20260529_232419"
SAMPEFF_DIR = "output/decision_maker/all_fracs"
OUT_DIR     = "output/decision_maker/comparisons/rl_vs_llm"

MODEL_ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "qwen3-0.6B", "qwen3-1.7B", "qwen3-4B"]
MODEL_LABELS_MAP = {
    "gpt2":        "GPT-2",
    "gpt2-medium": "GPT-2-Med",
    "gpt2-large":  "GPT-2-Lg",
    "qwen3-0.6B":  "Qwen3-0.6B",
    "qwen3-1.7B":  "Qwen3-1.7B",
    "qwen3-4B":    "Qwen3-4B",
}

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
        ("profit",  re.compile(r"best_profit[=\s]+([\d.]+)")),
        ("on_time", re.compile(r"best_on_time[=\s]+([\d.]+)")),
    ]:
        with open(path) as f:
            for line in f:
                hit = pat.search(line)
                if hit:
                    m[key] = float(hit.group(1))
    return m if len(m) == 2 else None


def parse_rl_log(path):
    """RL logs: key<space>value format."""
    m = {}
    for key, pat in [
        ("profit",  re.compile(r"best_profit[=\s]+([\d.]+)")),
        ("on_time", re.compile(r"best_on_time[=\s]+([\d.]+)")),
    ]:
        with open(path) as f:
            for line in f:
                hit = pat.search(line)
                if hit:
                    m[key] = float(hit.group(1))
    return m if len(m) == 2 else None


def load_dir(run_dir, parser_fn):
    """Returns dict: dataset -> sorted list of (frac, profit, on_time)."""
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
        results[dataset].append((frac, metrics["profit"], metrics["on_time"]))
    for ds in results:
        results[ds].sort(key=lambda x: x[0])
    return results


# ── Style ─────────────────────────────────────────────────────────────────────

DATASET_LABELS = {"dataco": "DataCo", "globalstore": "GlobalStore", "oas": "OAS"}

METRICS = [
    ("profit",  "Profit"),
    ("on_time", "On-Time Ratio"),
    ("sum",     "Profit + On-Time"),
]

RL_COLOR   = "#E53935"
RL_STYLE   = dict(color=RL_COLOR, linewidth=2, linestyle="--")
LLM_COLORS = ["#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1", "#F4511E"]


def _rl_ref(rl_rows, key):
    """Return the frac=1.00 RL value (or highest available frac)."""
    rows = sorted(rl_rows, key=lambda r: r[0], reverse=True)
    if not rows:
        return None
    row = rows[0]
    if key == "profit":    return row[1]
    elif key == "on_time": return row[2]
    else:                  return row[1] + row[2]

def frac_label(f):
    return f"{int(round(f*100))}%" if f > 0 else "0\n(zero-shot)"

def get_values(rows, key):
    fracs = [r[0] for r in rows]
    if key == "profit":    vals = [r[1] for r in rows]
    elif key == "on_time": vals = [r[2] for r in rows]
    else:                  vals = [r[1]+r[2] for r in rows]
    return fracs, vals


# ── Plot 1: Per-dataset ───────────────────────────────────────────────────────

def plot_per_dataset(rl_data, llm_data_list, out_dir):
    """One figure per dataset: 4 metric subplots, RL dashed vs each LLM solid."""
    all_datasets = sorted({ds for data, _ in llm_data_list for ds in data} | set(rl_data))

    for ds in all_datasets:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
        dset_label = DATASET_LABELS.get(ds, ds)

        for ax, (key, metric_label) in zip(axes, METRICS):
            # RL full-training horizontal reference
            if ds in rl_data:
                val = _rl_ref(rl_data[ds], key)
                if val is not None:
                    ax.axhline(val, label=f"RL full: {val:.4f}", **RL_STYLE)

            # LLM curves
            all_llm_fracs = set()
            for i, (llm_data, llm_label) in enumerate(llm_data_list):
                if ds not in llm_data:
                    continue
                fracs, vals = get_values(llm_data[ds], key)
                all_llm_fracs.update(fracs)
                color = LLM_COLORS[i % len(LLM_COLORS)]
                ax.plot(fracs, vals, label=llm_label, color=color,
                        linewidth=2, marker="o", markersize=6)

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            all_fracs = sorted(all_llm_fracs)
            ax.set_xticks(all_fracs)
            ax.set_xticklabels([frac_label(f) for f in all_fracs], fontsize=7)

        fig.suptitle(f"{dset_label} — RL Full Training (dashed) vs LLM", fontsize=13, fontweight="bold")
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

                # RL full-training horizontal reference
                if ds in rl_data:
                    val = _rl_ref(rl_data[ds], key)
                    if val is not None:
                        ax.axhline(val, color=color, linewidth=1.8, linestyle="--", alpha=0.75,
                                   label=f"{dset_label} RL full: {val:.4f}")

                # LLM solid curve
                fracs, vals = get_values(llm_data[ds], key)
                ax.plot(fracs, vals, color=color, linewidth=2.5, linestyle="-",
                        marker="o", markersize=6,
                        label=f"{dset_label} LLM")

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            all_fracs = sorted({r[0] for ds in llm_data for r in llm_data[ds]})
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
    """For each LLM frac: grouped bar chart of LLM models + RL full as dashed reference."""
    all_fracs    = sorted({r[0] for data, _ in llm_data_list for ds in data for r in data[ds]})
    all_datasets = sorted({ds for data, _ in llm_data_list for ds in data})

    n_methods = len(llm_data_list)
    x = np.arange(len(all_datasets))
    width = 0.8 / n_methods

    for frac in all_fracs:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

        for ax, (key, metric_label) in zip(axes, METRICS):
            # RL full-training dashed reference (per dataset)
            for ds_idx, ds in enumerate(all_datasets):
                val = _rl_ref(rl_data.get(ds, []), key)
                if val is not None:
                    dset_label = DATASET_LABELS.get(ds, ds)
                    ax.axhline(val, color=RL_COLOR, linewidth=1.5, linestyle="--", alpha=0.8,
                               label=f"RL full {dset_label}: {val:.4f}" if ds_idx == 0 else f"RL full {dset_label}: {val:.4f}")

            # LLM bars
            for j, (llm_data, llm_label) in enumerate(llm_data_list):
                vals = []
                for ds in all_datasets:
                    row = next((r for r in llm_data.get(ds, []) if r[0] == frac), None)
                    if row is None:
                        vals.append(0.0)
                    elif key == "profit":    vals.append(row[1])
                    elif key == "on_time":   vals.append(row[2])
                    else:                    vals.append(row[1] + row[2])
                offset = (j - (n_methods - 1) / 2) * width
                ax.bar(x + offset, vals, width * 0.9, label=llm_label,
                       color=LLM_COLORS[j % len(LLM_COLORS)], alpha=0.85, edgecolor="white")

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in all_datasets], fontsize=9)
            ax.legend(fontsize=6, ncol=2)
            ax.grid(axis="y", alpha=0.3)

        pct = frac_label(frac)
        fig.suptitle(f"LLM frac={pct} — All Models vs RL Full Training (dashed)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        frac_str = f"{int(round(frac * 100)):03d}"
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
        col_w = 12
        sub = f"{'Frac':<8}" + "".join(
            f"{'Profit':>{col_w}}{'OnTime':>{col_w}}{'P+O':>{col_w}}"
            for _ in methods
        )
        method_header = f"{'':8}" + "".join(
            f"{lbl:>{col_w*3}}" for lbl, _ in methods
        )
        lines.append(f"\n{'='*80}\n{dset_label}\n{'='*80}")
        lines.append(method_header)
        lines.append(sub)
        lines.append("-" * len(sub))

        for frac in all_fracs:
            row_str = f"{frac_label(frac).replace(chr(10),' '):<8}"
            for _, data in methods:
                entry = next((r for r in data.get(ds, []) if r[0] == frac), None)
                if entry:
                    row_str += f"{entry[1]:>{col_w}.4f}{entry[2]:>{col_w}.4f}{entry[1]+entry[2]:>{col_w}.4f}"
                else:
                    row_str += f"{'—':>{col_w}}{'—':>{col_w}}{'—':>{col_w}}"
            lines.append(row_str)

    table_str = "\n".join(lines)
    print(table_str)
    path = os.path.join(out_dir, "compare_table.txt")
    with open(path, "w") as f:
        f.write(table_str + "\n")
    print(f"\nSaved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for d in [RL_DIR, SAMPEFF_DIR]:
        if not os.path.isdir(d):
            print(f"Error: directory not found: {d}")
            raise SystemExit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"RL dir      : {RL_DIR}")
    print(f"Sampeff dir : {SAMPEFF_DIR}")
    print(f"Output dir  : {OUT_DIR}")

    print(f"\nLoading RL data...")
    rl_data = load_dir(RL_DIR, parse_rl_log)
    print(f"  Datasets: {sorted(rl_data.keys())}")

    # Build LLM list from model subdirs in SAMPEFF_DIR
    llm_data_list = []
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    model_dirs = sorted(
        [d for d in os.listdir(SAMPEFF_DIR)
         if os.path.isdir(os.path.join(SAMPEFF_DIR, d))],
        key=lambda m: order.get(m, 99)
    )
    for model in model_dirs:
        label = MODEL_LABELS_MAP.get(model, model)
        path  = os.path.join(SAMPEFF_DIR, model)
        print(f"Loading LLM '{label}' from: {path}")
        data = load_dir(path, parse_llm_log)
        llm_data_list.append((data, label))
        print(f"  Datasets: {sorted(data.keys())}")

    print("\n--- Generating plots ---")
    plot_per_dataset(rl_data, llm_data_list, OUT_DIR)
    plot_per_model(rl_data, llm_data_list, OUT_DIR)
    plot_frac_to_frac(rl_data, llm_data_list, OUT_DIR)

    print("\n--- Summary Table ---")
    print_and_save_table(rl_data, llm_data_list, OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
