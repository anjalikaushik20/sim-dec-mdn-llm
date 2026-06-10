"""
Plot sample efficiency curves for the RL baseline.
Saves plots and a summary table to the run directory.
"""

import os
import re
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ── Hardcoded path ────────────────────────────────────────────────────────────

RUN_DIR = "output/decision_maker/rl/results"

# ── Parsing ───────────────────────────────────────────────────────────────────

LOG_PAT = re.compile(r"([a-z]+)_frac([\d.]+)\.log$")

def parse_log(path):
    metrics = {}
    for key, pat in [
        ("profit",  re.compile(r"best_profit[=\s]+([\d.]+)")),
        ("on_time", re.compile(r"best_on_time[=\s]+([\d.]+)")),
    ]:
        try:
            with open(path) as f:
                for line in f:
                    m = pat.search(line)
                    if m:
                        metrics[key] = float(m.group(1))
        except OSError:
            pass
    return metrics if len(metrics) == 2 else None


def collect_results(run_dir):
    """Returns dict: dataset → sorted list of (frac, mean_profit, std_profit,
                                                      mean_on_time, std_on_time, n).
    Averages across seed subdirectories.
    """
    seed_dirs = sorted(glob.glob(os.path.join(run_dir, "seed*")))
    if not seed_dirs:
        # fallback: flat structure (no seeds)
        seed_dirs = [run_dir]

    # (dataset, frac) → [(profit, on_time), ...]
    raw = defaultdict(list)
    for sd in seed_dirs:
        for path in glob.glob(os.path.join(sd, "*.log")):
            m = LOG_PAT.match(os.path.basename(path))
            if not m:
                continue
            dataset, frac = m.group(1), float(m.group(2))
            metrics = parse_log(path)
            if metrics is None:
                print(f"  [skip] {sd}/{os.path.basename(path)} — incomplete")
                continue
            raw[(dataset, frac)].append((metrics["profit"], metrics["on_time"]))

    results = defaultdict(list)
    for (dataset, frac), vals in raw.items():
        profits  = [v[0] for v in vals]
        ontimes  = [v[1] for v in vals]
        n = len(vals)
        results[dataset].append((
            frac,
            float(np.mean(profits)),
            float(np.std(profits,  ddof=1) if n > 1 else 0.0),
            float(np.mean(ontimes)),
            float(np.std(ontimes,  ddof=1) if n > 1 else 0.0),
            n,
        ))
    for ds in results:
        results[ds].sort(key=lambda x: x[0])
    return results, len(seed_dirs)


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

def _get_vals(rows, key):
    """Extract (means, stds) for a given metric key from seed-averaged rows."""
    fracs = [r[0] for r in rows]
    if key == "profit":
        means = [r[1] for r in rows]; stds = [r[2] for r in rows]
    elif key == "on_time":
        means = [r[3] for r in rows]; stds = [r[4] for r in rows]
    else:  # sum
        means = [r[1]+r[3] for r in rows]; stds = [r[2]+r[4] for r in rows]
    return fracs, means, stds


def plot_all(results, run_dir, n_seeds):
    """One figure: all datasets × 3 metrics, with std error bars."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    all_fracs = sorted({r[0] for rows in results.values() for r in rows})

    for ax, (key, label) in zip(axes, METRICS):
        for dataset, rows in sorted(results.items()):
            fracs, means, stds = _get_vals(rows, key)
            color = COLORS.get(dataset)
            ax.plot(fracs, means, marker="o", linewidth=2, markersize=6,
                    color=color, label=DATASET_LABELS.get(dataset, dataset))
            ax.fill_between(fracs,
                            [m-s for m, s in zip(means, stds)],
                            [m+s for m, s in zip(means, stds)],
                            color=color, alpha=0.15)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Fraction", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(all_fracs)
        ax.set_xticklabels([frac_label(f) for f in all_fracs], fontsize=8)

    fig.suptitle(f"Sample Efficiency — RL Baseline  (mean ± std, n={n_seeds} seeds)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(run_dir, "sample_efficiency_rl.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_dataset(results, run_dir, n_seeds):
    """One figure per dataset: 3 metric subplots, with std shading."""
    for dataset, rows in sorted(results.items()):
        color = COLORS.get(dataset, "#555")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (key, label) in zip(axes, METRICS):
            fracs, means, stds = _get_vals(rows, key)
            ax.plot(fracs, means, marker="o", color=color, linewidth=2, markersize=7)
            ax.fill_between(fracs,
                            [m-s for m, s in zip(means, stds)],
                            [m+s for m, s in zip(means, stds)],
                            color=color, alpha=0.15)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.set_xticks(fracs)
            ax.set_xticklabels([frac_label(f) for f in fracs], fontsize=8)
            ax.grid(True, alpha=0.3)

        dset_label = DATASET_LABELS.get(dataset, dataset)
        fig.suptitle(f"{dset_label} — RL Baseline  (mean ± std, n={n_seeds} seeds)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(run_dir, f"sample_efficiency_rl_{dataset}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ── Table ─────────────────────────────────────────────────────────────────────

def save_table(results, run_dir, n_seeds):
    datasets  = sorted(results.keys())
    all_fracs = sorted({r[0] for rows in results.values() for r in rows})
    col_w = 18  # wide enough for "0.5321±0.0012"

    header = f"{'Frac':<8}" + "".join(
        f"  {DATASET_LABELS.get(d, d):^{col_w*3+4}}" for d in datasets
    )
    subheader = f"{'':8}" + "".join(
        f"  {'Profit (mean±std)':>{col_w}} {'OnTime (mean±std)':>{col_w}} {'Sum (mean±std)':>{col_w}}"
        for _ in datasets
    )
    sep = "-" * len(subheader)

    lines = [f"RL Baseline — mean ± std across {n_seeds} seeds", header, subheader, sep]
    for frac in all_fracs:
        row = f"{frac_label(frac):<8}"
        for ds in datasets:
            entry = next((r for r in results[ds] if r[0] == frac), None)
            if entry:
                _, mp, sp, mo, so, _ = entry
                row += (f"  {f'{mp:.4f}±{sp:.4f}':>{col_w}}"
                        f"  {f'{mo:.4f}±{so:.4f}':>{col_w}}"
                        f"  {f'{mp+mo:.4f}±{sp+so:.4f}':>{col_w}}")
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
    results, n_seeds = collect_results(RUN_DIR)

    if not results:
        print("No completed log files found.")
        raise SystemExit(1)

    print(f"Seeds    : {n_seeds}")
    print(f"Datasets : {[DATASET_LABELS.get(d, d) for d in sorted(results)]}")
    print(f"Fractions: {sorted({r[0] for rows in results.values() for r in rows})}")

    plot_all(results, RUN_DIR, n_seeds)
    plot_per_dataset(results, RUN_DIR, n_seeds)
    save_table(results, RUN_DIR, n_seeds)
    print("Done.")
