"""
Compare zero-shot VocabAlign results against RL baseline with mean/std.

Reads zero-shot logs from model-specific seed/run dirs:
  output/decision_maker/zeroshot/models/{model}/{seed}/{dataset}.log
  output/decision_maker/zeroshot/models/{model}/{seed}/{dataset}_frac1.00.log

Reads RL logs from:
  output/decision_maker/rl/results/{seed}/{dataset}_frac1.00.log

Saves:
  zeroshot_compare_mean_std.txt
  zeroshot_compare_mean_std.csv
  zeroshot_compare_{dataset}.png
to OUT_DIR.
"""

import os
import re
import csv
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
})


# ── Paths ─────────────────────────────────────────────────────────────────────

ZEROSHOT_RUN_DIRS_GPT2 = [
    "output/decision_maker/zeroshot/models/gpt2/01",
    "output/decision_maker/zeroshot/models/gpt2/02",
    "output/decision_maker/zeroshot/models/gpt2/03",
    "output/decision_maker/zeroshot/models/gpt2/04",
    "output/decision_maker/zeroshot/models/gpt2/05",
]

ZEROSHOT_RUN_DIRS_GPT2_LARGE = [
    "output/decision_maker/zeroshot/models/gpt2_large/01",
    "output/decision_maker/zeroshot/models/gpt2_large/02",
    "output/decision_maker/zeroshot/models/gpt2_large/03",
    "output/decision_maker/zeroshot/models/gpt2_large/04",
    "output/decision_maker/zeroshot/models/gpt2_large/05",
]

ZEROSHOT_RUN_DIRS_PHI4_MINI = [
    "output/decision_maker/zeroshot/models/phi4_mini/01",
    "output/decision_maker/zeroshot/models/phi4_mini/02",
    "output/decision_maker/zeroshot/models/phi4_mini/03",
    "output/decision_maker/zeroshot/models/phi4_mini/04",
    "output/decision_maker/zeroshot/models/phi4_mini/05",
]

ZEROSHOT_RUN_DIRS_QWEN_0_6 = [
    "output/decision_maker/zeroshot/models/qwen_0.6/01",
    "output/decision_maker/zeroshot/models/qwen_0.6/02",
    "output/decision_maker/zeroshot/models/qwen_0.6/03",
    "output/decision_maker/zeroshot/models/qwen_0.6/04",
    "output/decision_maker/zeroshot/models/qwen_0.6/05",
]

ZEROSHOT_RUN_DIRS_QWEN_1_7 = [
    "output/decision_maker/zeroshot/models/qwen_1.7/01",
    "output/decision_maker/zeroshot/models/qwen_1.7/02",
    "output/decision_maker/zeroshot/models/qwen_1.7/03",
    "output/decision_maker/zeroshot/models/qwen_1.7/04",
    "output/decision_maker/zeroshot/models/qwen_1.7/05",
]

RL_RUN_DIRS = [
    "output/decision_maker/rl/results/seed42",
    "output/decision_maker/rl/results/seed131",
    "output/decision_maker/rl/results/seed521",
    "output/decision_maker/rl/results/seed1009",
    "output/decision_maker/rl/results/seed2027",
]

ZEROSHOT_MODEL_RUN_DIRS = {
    "gpt2": ZEROSHOT_RUN_DIRS_GPT2,
    "gpt2-large": ZEROSHOT_RUN_DIRS_GPT2_LARGE,
    "phi4-mini": ZEROSHOT_RUN_DIRS_PHI4_MINI,
    "qwen3-0.6B": ZEROSHOT_RUN_DIRS_QWEN_0_6,
    "qwen3-1.7B": ZEROSHOT_RUN_DIRS_QWEN_1_7,
}

FRAC = "1.00"

OUT_DIR = "output/decision_maker/zeroshot/models/compare_rl_mean_std"


# ── Parsing ───────────────────────────────────────────────────────────────────

LOG_PAT = re.compile(r"([a-zA-Z0-9_]+)(?:_frac([\d.]+))?\.log$")
NUM_PAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

def parse_log(path):
    metrics = {}

    patterns = {
        "profit": re.compile(rf"best_profit[=\s:]+{NUM_PAT}"),
        "on_time": re.compile(rf"best_on_time[=\s:]+{NUM_PAT}"),
        # Matches best_dm_accuracy and best_dm_accuracy_true.
        # Last match wins, so _true overwrites plain value when both appear.
        "accuracy": re.compile(rf"best_dm_accuracy(?:_true)?[=\s:]+{NUM_PAT}"),
    }

    with open(path) as f:
        for line in f:
            for key, pat in patterns.items():
                m = pat.search(line)
                if m:
                    metrics[key] = float(m.group(1))

    return metrics if len(metrics) == 3 else None


def expand_dirs(paths):
    expanded = []
    for p in paths:
        matches = sorted(glob.glob(p))
        expanded.extend(matches if matches else [p])
    return expanded


# ── Labels / ordering ─────────────────────────────────────────────────────────

DATASET_LABELS = {
    "dataco": "DataCo",
    "globalstore": "GlobalStore",
    "oas": "OAS",
}

MODEL_ORDER = [
    "rl",
    "gpt2",
    "gpt2-large",
    "phi4-mini",
    "qwen3-0.6B",
    "qwen3-1.7B",
]

MODEL_LABELS = {
    "rl": "RL",
    "gpt2": "GPT-2",
    "gpt2-large": "GPT-2 Large",
    "phi4-mini": "Phi-4 Mini",
    "qwen3-0.6B": "Qwen3-0.6B",
    "qwen3-1.7B": "Qwen3-1.7B",
}


def sorted_models(model_tags):
    order = {m.lower(): i for i, m in enumerate(MODEL_ORDER)}
    return sorted(model_tags, key=lambda m: order.get(m.lower(), 99))


# ── Collection ────────────────────────────────────────────────────────────────

def collect_zeroshot(model_run_dirs, frac):
    """
    Returns:
      dataset -> model_tag -> list of metric dicts
    """
    results = defaultdict(lambda: defaultdict(list))

    for model_tag, run_dirs in model_run_dirs.items():
        for run_dir in expand_dirs(run_dirs):
            if not os.path.isdir(run_dir):
                print(f"[skip] zero-shot directory not found: {run_dir}")
                continue

            for path in glob.glob(os.path.join(run_dir, "**", "*.log"), recursive=True):
                base = os.path.basename(path)
                m = LOG_PAT.match(base)

                if not m:
                    continue

                dataset, found_frac = m.group(1).lower(), m.group(2)

                # Keep files with no frac suffix, or matching frac suffix.
                if found_frac is not None and found_frac != frac:
                    continue

                metrics = parse_log(path)

                if metrics is None:
                    print(f"[skip] incomplete zero-shot log: {path}")
                    continue

                results[dataset][model_tag].append(metrics)

    return results


def collect_rl(run_dirs, frac, model_tag="rl"):
    """
    Returns:
      dataset -> model_tag -> list of metric dicts
    """
    results = defaultdict(lambda: defaultdict(list))

    for run_dir in expand_dirs(run_dirs):
        if not os.path.isdir(run_dir):
            print(f"[skip] RL directory not found: {run_dir}")
            continue

        pattern = os.path.join(run_dir, "**", f"*_frac{frac}.log")

        for path in glob.glob(pattern, recursive=True):
            base = os.path.basename(path)
            m = LOG_PAT.match(base)

            if not m:
                continue

            dataset, found_frac = m.group(1).lower(), m.group(2)

            if found_frac != frac:
                continue

            metrics = parse_log(path)

            if metrics is None:
                print(f"[skip] incomplete RL log: {path}")
                continue

            results[dataset][model_tag].append(metrics)

    return results


def merge_results(*result_dicts):
    merged = defaultdict(lambda: defaultdict(list))

    for result in result_dicts:
        for dataset, model_map in result.items():
            for model, runs in model_map.items():
                merged[dataset][model].extend(runs)

    return merged


# ── Summary stats ─────────────────────────────────────────────────────────────

def mean_std(values):
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n == 0:
        return np.nan, np.nan, 0

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0

    return mean, std, n


def summarize_runs(runs):
    profit_values = [r["profit"] for r in runs]
    on_time_values = [r["on_time"] for r in runs]
    accuracy_values = [r["accuracy"] for r in runs]
    sum_values = [r["profit"] + r["on_time"] for r in runs]

    return {
        "profit": mean_std(profit_values),
        "on_time": mean_std(on_time_values),
        "accuracy": mean_std(accuracy_values),
        "sum": mean_std(sum_values),
    }


def summarize_results(results):
    """
    Returns:
      dataset -> model -> metric -> (mean, std, n)
    """
    summary = defaultdict(dict)

    for dataset, model_map in results.items():
        for model, runs in model_map.items():
            summary[dataset][model] = summarize_runs(runs)

    return summary


# ── Output table ──────────────────────────────────────────────────────────────

def fmt(metric_tuple):
    mean, std, n = metric_tuple

    if n == 0 or np.isnan(mean):
        return "—"

    return f"{mean:.4f}±{std:.4f} ({n})"


def format_table(summary):
    all_models = sorted_models({m for d in summary.values() for m in d})
    datasets = sorted(summary.keys())

    lines = []
    lines.append("")
    lines.append("Mean ± std, with n in parentheses")
    lines.append("")

    for ds in datasets:
        dset_label = DATASET_LABELS.get(ds, ds)
        lines.append(f"{dset_label}")
        lines.append("-" * len(dset_label))
        lines.append(
            f"{'Model':<16} "
            f"{'Profit':>20} "
            f"{'OnTime':>20} "
            f"{'Accuracy':>20} "
            f"{'Sum':>20}"
        )

        for model in all_models:
            if model not in summary[ds]:
                continue

            s = summary[ds][model]
            lines.append(
                f"{MODEL_LABELS.get(model, model):<16} "
                f"{fmt(s['profit']):>20} "
                f"{fmt(s['on_time']):>20} "
                f"{fmt(s['accuracy']):>20} "
                f"{fmt(s['sum']):>20}"
            )

        lines.append("")

    return "\n".join(lines)


def save_table_and_csv(summary, out_dir):
    table = format_table(summary)
    print(table)

    txt_path = os.path.join(out_dir, "zeroshot_compare_mean_std.txt")
    with open(txt_path, "w") as f:
        f.write(table)

    csv_path = os.path.join(out_dir, "zeroshot_compare_mean_std.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "model",
            "metric",
            "mean",
            "std",
            "n",
        ])

        for dataset in sorted(summary.keys()):
            for model in sorted_models(summary[dataset].keys()):
                for metric in ["profit", "on_time", "accuracy", "sum"]:
                    mean, std, n = summary[dataset][model][metric]
                    writer.writerow([
                        dataset,
                        model,
                        metric,
                        mean,
                        std,
                        n,
                    ])

    print(f"Saved: {txt_path}")
    print(f"Saved: {csv_path}")


# ── Plots ─────────────────────────────────────────────────────────────────────

BAR_COLORS = {
    "profit": "#4C72B0",
    "on_time": "#DD8452",
    "sum": "#55A868",
}

RL_COLORS = {
    "profit": "#4C72B0",
    "on_time": "#DD8452",
    "sum": "#55A868",
}


def get_metric(summary, dataset, model, metric):
    if dataset not in summary:
        return np.nan, np.nan, 0

    if model not in summary[dataset]:
        return np.nan, np.nan, 0

    return summary[dataset][model][metric]


def plot_per_dataset(summary, out_dir):
    """
    One figure per dataset:
      - grouped zero-shot bars with std error bars
      - RL dashed mean lines
      - RL shaded ±std bands
    """
    all_models = sorted_models({m for d in summary.values() for m in d})
    llm_models = [m for m in all_models if m != "rl"]

    x = np.arange(len(llm_models))
    width = 0.22

    for ds in sorted(summary.keys()):
        fig, ax = plt.subplots(figsize=(14, 6))

        plot_specs = [
            ("profit", -width, "Zero-Shot Profit"),
            ("on_time", 0.0, "Zero-Shot On-Time"),
            ("sum", width, "Zero-Shot Combined"),
        ]

        for metric, offset, label in plot_specs:
            means = []
            stds = []

            for model in llm_models:
                mean, std, _ = get_metric(summary, ds, model, metric)
                means.append(mean)
                stds.append(std)

            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=3,
                label=label,
                color=BAR_COLORS[metric],
                alpha=0.9,
            )

        # RL reference lines and ±std bands
        if "rl" in summary[ds]:
            for metric in ["profit", "on_time", "sum"]:
                mean, std, n = summary[ds]["rl"][metric]

                if np.isnan(mean):
                    continue

                ax.axhline(
                    mean,
                    color=RL_COLORS[metric],
                    linestyle="--",
                    linewidth=1.6,
                    label=f"RL {metric.replace('_', ' ').title()} {mean:.4f}±{std:.4f} ({n})",
                )

                if std > 0:
                    ax.axhspan(
                        mean - std,
                        mean + std,
                        color=RL_COLORS[metric],
                        alpha=0.10,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in llm_models], fontsize=16)
        ax.set_ylabel("Score")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(ncol=2, loc="upper right")

        dset_label = DATASET_LABELS.get(ds, ds)
        ax.set_title(
            f"{dset_label} — Zero-Shot vs RL Full Training, Mean ± Std",
            fontweight="bold",
        )

        plt.tight_layout()

        out_path = os.path.join(out_dir, f"zeroshot_compare_{ds}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Zero-shot dirs:")
    for model_tag, run_dirs in ZEROSHOT_MODEL_RUN_DIRS.items():
        print(f"  {model_tag}:")
        for d in expand_dirs(run_dirs):
            print(f"    {d}")

    print("RL dirs:")
    for d in expand_dirs(RL_RUN_DIRS):
        print(f"  {d}")

    print(f"Frac filter: {FRAC}")
    print(f"Output dir : {OUT_DIR}")

    zeroshot_results = collect_zeroshot(ZEROSHOT_MODEL_RUN_DIRS, FRAC)
    rl_results = collect_rl(RL_RUN_DIRS, FRAC)

    results = merge_results(zeroshot_results, rl_results)

    if not results:
        print("No completed log files found.")
        raise SystemExit(1)

    summary = summarize_results(results)

    save_table_and_csv(summary, OUT_DIR)
    plot_per_dataset(summary, OUT_DIR)

    print("Done.")