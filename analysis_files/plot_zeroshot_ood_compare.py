#!/usr/bin/env python3
"""
Zero-shot OOD comparison — grouped bar chart for DataCo_OOD.

Reads {results_dir}/dataco_ood_{model_tag}.log and the RL frac=1.00 reference.
Produces two plots (style matches plot_zeroshot_compare.py):
  - zeroshot_ood_compare_dataco_ood.png          (no RL lines)
  - zeroshot_ood_compare_dataco_ood_with_rl.png  (RL dashed lines + shaded band)
  - zeroshot_ood_compare_mean_std.txt

Usage:
  conda run -n simenv python3 plot_zeroshot_ood_compare.py \\
      --results_dir output/decision_maker/zeroshot/ood_vocabalign/results \\
      --rl_log      output/decision_maker/rl/results_ood/dataco_ood_frac1.00.log
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

MODEL_ORDER  = ["gpt2", "gpt2-large", "qwen3-0.6B", "qwen3-1.7B", "phi4-mini"]
MODEL_LABELS = {
    "gpt2":       "GPT-2",
    "gpt2-large": "GPT-2 Large",
    "qwen3-0.6B": "Qwen3-0.6B",
    "qwen3-1.7B": "Qwen3-1.7B",
    "phi4-mini":  "Phi4-mini",
}

BAR_COLORS = {
    "profit":  "#4C72B0",
    "on_time": "#DD8452",
    "sum":     "#55A868",
}
RL_COLORS = BAR_COLORS


def parse_log(path):
    profit, on_time = None, None
    try:
        with open(path) as f:
            for line in f:
                m = re.search(r"best_profit[=\s]+([\d.eE+\-]+)", line)
                if m:
                    profit = float(m.group(1))
                m = re.search(r"best_on_time[=\s]+([\d.eE+\-]+)", line)
                if m:
                    on_time = float(m.group(1))
    except OSError:
        pass
    return profit, on_time


def load_results(results_dir):
    """Returns {model_tag: {profit, on_time, sum}}"""
    results = {}
    for path in glob.glob(os.path.join(results_dir, "dataco_ood_*.log")):
        fname     = os.path.basename(path)
        model_tag = re.sub(r"^dataco_ood_|\.log$", "", fname)
        profit, on_time = parse_log(path)
        if profit is None or on_time is None:
            print(f"  [skip] incomplete: {fname}")
            continue
        results[model_tag] = {"profit": profit, "on_time": on_time, "sum": profit + on_time}
    return results


def load_rl(rl_log):
    """Returns {profit, on_time, sum} or None."""
    if not rl_log or not os.path.exists(rl_log):
        print(f"  [skip] RL log not found: {rl_log}")
        return None
    profit, on_time = parse_log(rl_log)
    if profit is None or on_time is None:
        print(f"  [skip] RL log incomplete: {rl_log}")
        return None
    return {"profit": profit, "on_time": on_time, "sum": profit + on_time}


def sorted_models(tags):
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(tags, key=lambda m: order.get(m, 99))


def save_table(results, rl, out_dir):
    models = sorted_models(results.keys())
    lines  = []
    lines.append("")
    lines.append("DataCo_OOD — Zero-Shot VocabAlign")
    lines.append("-" * 50)
    lines.append(f"{'Model':<16} {'Profit':>10} {'On-Time':>10} {'Sum':>10}")
    lines.append("-" * 50)
    for model in models:
        r     = results[model]
        label = MODEL_LABELS.get(model, model)
        lines.append(f"{label:<16} {r['profit']:>10.4f} {r['on_time']:>10.4f} {r['sum']:>10.4f}")
    if rl:
        lines.append("-" * 50)
        lines.append(f"{'RL (frac=1.00)':<16} {rl['profit']:>10.4f} {rl['on_time']:>10.4f} {rl['sum']:>10.4f}")
    lines.append("")
    table = "\n".join(lines)
    print(table)
    out_path = os.path.join(out_dir, "zeroshot_ood_compare_mean_std.txt")
    with open(out_path, "w") as f:
        f.write(table)
    print(f"Saved: {out_path}")


def make_plot(results, rl, out_path, show_rl):
    models = sorted_models(results.keys())
    x      = np.arange(len(models))
    width  = 0.22

    fig, ax = plt.subplots(figsize=(12, 5))

    plot_specs = [
        ("profit",  -width, "Zero-Shot Profit"),
        ("on_time",  0.0,   "Zero-Shot On-Time"),
        ("sum",      width, "Zero-Shot Combined"),
    ]

    for metric, offset, label in plot_specs:
        vals = [results[m][metric] for m in models]
        ax.bar(x + offset, vals, width,
               label=label, color=BAR_COLORS[metric], alpha=0.9)

    if show_rl and rl:
        for metric in ["profit", "on_time", "sum"]:
            val = rl[metric]
            ax.axhline(val, color=RL_COLORS[metric], linestyle="--", linewidth=1.6,
                       label=f"RL {metric.replace('_', ' ').title()} {val:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.set_title("DataCo OOD — Zero-Shot VocabAlign",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir",
                    default="output/decision_maker/zeroshot/ood_vocabalign/results")
    ap.add_argument("--rl_log",
                    default="output/decision_maker/rl/results_ood/dataco_ood_frac1.00.log")
    args = ap.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print("No completed log files found.")
        raise SystemExit(1)

    rl = load_rl(args.rl_log)
    print(f"Models found: {sorted_models(results.keys())}")
    print(f"RL reference: {rl}")

    save_table(results, rl, args.results_dir)

    make_plot(results, rl, out_path=os.path.join(
        args.results_dir, "zeroshot_ood_compare_dataco_ood.png"), show_rl=False)

    make_plot(results, rl, out_path=os.path.join(
        args.results_dir, "zeroshot_ood_compare_dataco_ood_with_rl.png"), show_rl=True)

    print("Done.")


if __name__ == "__main__":
    main()
