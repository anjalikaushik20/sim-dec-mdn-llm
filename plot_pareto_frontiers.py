#!/usr/bin/env python3
"""Experiment 7 — RF/XGBoost Pareto analysis.

Reads best_profit and best_on_time from ML logs (RF, XGBoost, Random, Historical)
and VocabAlign logs (frac=1.00), then plots the profit-vs-on_time Pareto frontier
for each dataset.

Hypothesis: RF/XGBoost achieve high profit but poor on-time (single-objective
optimization); VocabAlign achieves a better trade-off via the soft-label KL term.

Usage:
    conda run -n simenv python3 plot_pareto_frontiers.py \\
        --ml_dir output/decision_maker/ml \\
        --llm_dir output/decision_maker/all_fracs \\
        --rl_dir output/decision_maker/rl \\
        --frac 1.00 \\
        --out output/decision_maker/comparisons/pareto
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Log parsing helpers (reused from compare_all_methods.py pattern)
# ─────────────────────────────────────────────────────────────────────────────

def parse_metric(log_path: str, key: str):
    """Return float value for 'key=value' or 'key value' pattern in log."""
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(rf"{re.escape(key)}[=\s]+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", line)
                if m:
                    return float(m.group(1))
    except OSError:
        pass
    return None


def collect_ml(ml_dir: str, frac: str):
    """Collect (dataset, baseline, profit, on_time) from ML logs."""
    records = []
    for log_path in glob.glob(os.path.join(ml_dir, "**", "*.log"), recursive=True):
        profit = parse_metric(log_path, "best_profit")
        on_time = parse_metric(log_path, "best_on_time")
        if profit is None or on_time is None:
            continue
        fname = os.path.basename(log_path)
        # Expect {dataset}_{baseline}_frac{frac}.log or similar
        for baseline in ("rf", "xgb", "random", "historical"):
            if baseline in fname.lower():
                for ds in ("dataco", "globalstore", "oas", "supplychainshipmentpricing"):
                    if ds in fname.lower():
                        if frac in fname:
                            records.append({
                                "dataset": ds, "method": baseline,
                                "profit": profit, "on_time": on_time,
                            })
                            break
                break
    return records


def collect_llm(llm_dir: str, frac: str):
    """Collect (dataset, model_tag, profit, on_time) from VocabAlign logs at given frac."""
    records = []
    pattern = os.path.join(llm_dir, "**", "seed*", "*", f"*_frac{frac}.log")
    for log_path in glob.glob(pattern, recursive=True):
        profit = parse_metric(log_path, "best_profit")
        on_time = parse_metric(log_path, "best_on_time")
        if profit is None or on_time is None:
            continue
        parts = log_path.split(os.sep)
        seed_seg = next((p for p in parts if re.match(r"seed\d+$", p)), None)
        if seed_seg is None:
            continue
        seed_idx = parts.index(seed_seg)
        if seed_idx + 2 >= len(parts):
            continue
        model_tag = parts[seed_idx + 1]
        fname = parts[-1]
        m = re.match(r"(.+)_frac", fname)
        if m:
            records.append({
                "dataset": m.group(1).lower(), "method": f"vocabalign_{model_tag}",
                "profit": profit, "on_time": on_time,
            })
    return records


def collect_rl(rl_dir: str, frac: str):
    """Collect (dataset, rl, profit, on_time) from RL logs."""
    records = []
    for log_path in glob.glob(os.path.join(rl_dir, "**", f"*_frac{frac}.log"),
                               recursive=True):
        profit = parse_metric(log_path, "best_profit")
        on_time = parse_metric(log_path, "best_on_time")
        if profit is None or on_time is None:
            continue
        fname = os.path.basename(log_path)
        m = re.match(r"(.+)_frac", fname)
        if m:
            records.append({
                "dataset": m.group(1).lower(), "method": "rl",
                "profit": profit, "on_time": on_time,
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Pareto frontier
# ─────────────────────────────────────────────────────────────────────────────

def pareto_front(points):
    """Return indices of non-dominated points (maximise both axes)."""
    dominated = np.zeros(len(points), dtype=bool)
    pts = np.array(points)
    for i, pi in enumerate(pts):
        for j, pj in enumerate(pts):
            if i == j:
                continue
            if np.all(pj >= pi) and np.any(pj > pi):
                dominated[i] = True
                break
    return np.where(~dominated)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

METHOD_STYLE = {
    "rf":         {"color": "#e74c3c", "marker": "s", "label": "Random Forest", "zorder": 3},
    "xgb":        {"color": "#e67e22", "marker": "D", "label": "XGBoost",       "zorder": 3},
    "random":     {"color": "#95a5a6", "marker": "x", "label": "Random",        "zorder": 2},
    "historical": {"color": "#7f8c8d", "marker": "+", "label": "Historical",    "zorder": 2},
    "rl":         {"color": "#27ae60", "marker": "^", "label": "RL (MLP)",      "zorder": 4},
}
VOCABALIGN_COLORS = ["#2980b9", "#1a6fa0", "#3498db", "#5dade2", "#85c1e9", "#a9cce3"]


def plot_dataset(ax, records, dataset_label):
    plotted = []
    va_methods = sorted({r["method"] for r in records if r["method"].startswith("vocabalign")})
    for i, method in enumerate(va_methods):
        pts = [(r["profit"], r["on_time"]) for r in records
               if r["method"] == method]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys,
                   color=VOCABALIGN_COLORS[i % len(VOCABALIGN_COLORS)],
                   marker="o", s=40, alpha=0.7, zorder=5,
                   label=method.replace("vocabalign_", "VocabAlign-"))
        plotted.extend(list(zip(xs, ys)))

    for method, style in METHOD_STYLE.items():
        pts = [(r["profit"], r["on_time"]) for r in records if r["method"] == method]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, color=style["color"], marker=style["marker"],
                   s=80, zorder=style["zorder"], label=style["label"])
        plotted.extend(list(zip(xs, ys)))

    if len(plotted) >= 2:
        front_idx = pareto_front(plotted)
        front_pts = sorted([plotted[i] for i in front_idx], key=lambda p: p[0])
        fx, fy = zip(*front_pts)
        ax.step(fx, fy, where="post", color="#2c3e50", linewidth=1.5,
                linestyle="--", alpha=0.5, label="Pareto frontier")

    ax.set_title(dataset_label)
    ax.set_xlabel("Profit")
    ax.set_ylabel("On-time rate")
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Profit-vs-on_time Pareto plot")
    parser.add_argument("--ml_dir",  type=str, default="output/decision_maker/ml")
    parser.add_argument("--llm_dir", type=str, default="output/decision_maker/all_fracs")
    parser.add_argument("--rl_dir",  type=str, default="output/decision_maker/rl")
    parser.add_argument("--frac",    type=str, default="1.00")
    parser.add_argument("--out",     type=str, default="output/decision_maker/comparisons/pareto")
    args = parser.parse_args()

    all_records = (
        collect_ml(args.ml_dir, args.frac)
        + collect_llm(args.llm_dir, args.frac)
        + collect_rl(args.rl_dir, args.frac)
    )
    if not all_records:
        print("[pareto] No records found — check log directories.")
        return

    df = pd.DataFrame(all_records)
    datasets = sorted(df["dataset"].unique())
    print(f"[pareto] {len(all_records)} records across datasets: {datasets}")

    os.makedirs(args.out, exist_ok=True)

    # One figure per dataset
    for ds in datasets:
        subset = df[df["dataset"] == ds].to_dict("records")
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_dataset(ax, subset, ds.title())
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), fontsize=7, loc="best")
        fig.tight_layout()
        out_path = os.path.join(args.out, f"{ds}_pareto.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[pareto] Saved → {out_path}")

    # Combined figure
    ncols = min(2, len(datasets))
    nrows = (len(datasets) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[i // ncols][i % ncols]
        subset = df[df["dataset"] == ds].to_dict("records")
        plot_dataset(ax, subset, ds.title())
    for j in range(len(datasets), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    fig.legend(seen.values(), seen.keys(), loc="lower center",
               ncol=min(len(seen), 4), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Profit vs On-time Rate (frac={args.frac})", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_combined = os.path.join(args.out, "pareto_all_datasets.png")
    fig.savefig(out_combined, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[pareto] Combined figure saved → {out_combined}")


if __name__ == "__main__":
    main()
