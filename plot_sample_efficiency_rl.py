"""
Plot sample efficiency curves for the RL baseline.
Saves plots and a summary table to the run directory.
"""

import os
import re
import glob
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt

# ── Hardcoded path ────────────────────────────────────────────────────────────

RUN_DIR = "output/decision_maker/rl/20260529_232419"

# ── Parsing ───────────────────────────────────────────────────────────────────

LOG_PAT = re.compile(r"([a-z]+)_frac([\d.]+)\.log$")

def parse_log(path):
    metrics = {}
    for key, pat in [
        ("profit",  re.compile(r"best_profit[=\s]+([\d.]+)")),
        ("on_time", re.compile(r"best_on_time[=\s]+([\d.]+)")),
    ]:
        with open(path) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    metrics[key] = float(m.group(1))
    return metrics if len(metrics) == 2 else None


def collect_results(run_dir):
    """Returns dict: dataset → sorted list of (frac, profit, on_time)."""
    results = defaultdict(list)
    for path in glob.glob(os.path.join(run_dir, "*.log")):
        m = LOG_PAT.match(os.path.basename(path))
        if not m:
            continue
        dataset, frac = m.group(1), float(m.group(2))
        metrics = parse_log(path)
        if metrics is None:
            print(f"  [skip] {os.path.basename(path)} — incomplete")
            continue
        results[dataset].append((frac, metrics["profit"], metrics["on_time"]))
    for ds in results:
        results[ds].sort(key=lambda x: x[0])
    return results


# ── Style ─────────────────────────────────────────────────────────────────────

DATASET_LABELS = {
    "dataco":      "DataCo",
    "globalstore": "GlobalStore",
    "oas":         "OAS",
}
COLORS = {
    "dataco":      "#2196F3",
    "globalstore": "#FF5722",
    "oas":         "#4CAF50",
}
METRICS = [
    ("profit",  "Profit"),
    ("on_time", "On-Time Ratio"),
    ("sum",     "Profit + On-Time"),
]

def frac_label(f):
    return f"{int(round(f * 100))}%"


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_all(results, run_dir):
    """One figure: all datasets × 3 metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    all_fracs = sorted({r[0] for rows in results.values() for r in rows})

    for ax, (key, label) in zip(axes, METRICS):
        for dataset, rows in sorted(results.items()):
            vals = [r[1]+r[2] if key == "sum" else (r[1] if key == "profit" else r[2])
                    for r in rows]
            ax.plot([r[0] for r in rows], vals,
                    marker="o", linewidth=2, markersize=6,
                    color=COLORS.get(dataset),
                    label=DATASET_LABELS.get(dataset, dataset))
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Fraction", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(all_fracs)
        ax.set_xticklabels([frac_label(f) for f in all_fracs], fontsize=8)

    fig.suptitle(f"Sample Efficiency — RL Baseline\n{os.path.basename(run_dir)}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(run_dir, "sample_efficiency_rl.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_dataset(results, run_dir):
    """One figure per dataset: 3 metric subplots."""
    for dataset, rows in sorted(results.items()):
        fracs = [r[0] for r in rows]
        color = COLORS.get(dataset, "#555")

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (key, label) in zip(axes, METRICS):
            vals = [r[1]+r[2] if key == "sum" else (r[1] if key == "profit" else r[2])
                    for r in rows]
            ax.plot(fracs, vals, marker="o", color=color, linewidth=2, markersize=7)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.set_xticks(fracs)
            ax.set_xticklabels([frac_label(f) for f in fracs], fontsize=8)
            ax.grid(True, alpha=0.3)

        dset_label = DATASET_LABELS.get(dataset, dataset)
        fig.suptitle(f"{dset_label} — RL Baseline Sample Efficiency", fontsize=12, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(run_dir, f"sample_efficiency_rl_{dataset}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ── Table ─────────────────────────────────────────────────────────────────────

def save_table(results, run_dir):
    datasets = sorted(results.keys())
    all_fracs = sorted({r[0] for rows in results.values() for r in rows})
    col_w = 10

    header = f"{'Frac':<8}" + "".join(
        f"  {DATASET_LABELS.get(d, d):^{col_w*3+2}}" for d in datasets
    )
    subheader = f"{'':8}" + "".join(
        f"  {'Profit':>{col_w}} {'OnTime':>{col_w}} {'Sum':>{col_w}}" for _ in datasets
    )
    sep = "-" * len(subheader)

    lines = [header, subheader, sep]
    for frac in all_fracs:
        row = f"{frac_label(frac):<8}"
        for ds in datasets:
            entry = next((r for r in results[ds] if r[0] == frac), None)
            if entry:
                s = entry[1] + entry[2]
                row += f"  {entry[1]:>{col_w}.4f} {entry[2]:>{col_w}.4f} {s:>{col_w}.4f}"
            else:
                row += f"  {'—':>{col_w}} {'—':>{col_w}} {'—':>{col_w}}"
        lines.append(row)

    table = "\n".join(lines)
    print("\n" + table)
    out_path = os.path.join(run_dir, "sample_efficiency_rl.txt")
    with open(out_path, "w") as f:
        f.write(table + "\n")
    print(f"Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU", type=int, default=0, help="GPU device ID to use")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    if not os.path.isdir(RUN_DIR):
        print(f"Error: directory not found: {RUN_DIR}")
        raise SystemExit(1)

    print(f"Run dir: {RUN_DIR}")
    print(f"GPU    : {args.GPU}")
    results = collect_results(RUN_DIR)

    if not results:
        print("No completed log files found.")
        raise SystemExit(1)

    print(f"Datasets : {[DATASET_LABELS.get(d, d) for d in sorted(results)]}")
    print(f"Fractions: {sorted({r[0] for rows in results.values() for r in rows})}")

    plot_all(results, RUN_DIR)
    plot_per_dataset(results, RUN_DIR)
    save_table(results, RUN_DIR)
    print("Done.")
