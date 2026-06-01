"""
Compare zero-shot VocabAlign results (all models) against the RL baseline.

Reads:
  ZEROSHOT_DIR/{model_tag}/{dataset}_frac1.00.log
  RL_DIR/{dataset}_frac1.00.log

Saves plots and summary table to ZEROSHOT_DIR.
"""

import os
import re
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# ── Hardcoded paths ───────────────────────────────────────────────────────────

ZEROSHOT_DIR = "output/decision_maker/zeroshot/20260529_172517"
RL_DIR       = "output/decision_maker/rl/20260529_232419"
FRAC         = "1.00"
OUT_DIR      = ZEROSHOT_DIR

# ── Parsing ───────────────────────────────────────────────────────────────────

ZEROSHOT_LOG_PAT = re.compile(r"([a-z]+)\.log$")
RL_LOG_PAT       = re.compile(r"([a-z]+)_frac([\d.]+)\.log$")

def parse_log(path):
    metrics = {}
    patterns = {
        "profit":   re.compile(r"best_profit[=\s]+([\d.]+)"),
        "on_time":  re.compile(r"best_on_time[=\s]+([\d.]+)"),
        # match best_dm_accuracy and best_dm_accuracy_true; last match wins
        # so _true overwrites the plain value when both appear (LLM logs)
        "accuracy": re.compile(r"best_dm_accuracy(?:_true)?[=:\s]+([\d.]+)"),
    }
    with open(path) as f:
        for line in f:
            for key, pat in patterns.items():
                m = pat.search(line)
                if m:
                    metrics[key] = float(m.group(1))
    return metrics if len(metrics) == 3 else None


def collect_zeroshot(run_dir):
    """Walk model subdirs, pick {dataset}.log files.
    Returns: dataset → model_tag → {profit, on_time, accuracy}"""
    results = defaultdict(dict)
    for model_tag in sorted(os.listdir(run_dir)):
        model_dir = os.path.join(run_dir, model_tag)
        if not os.path.isdir(model_dir):
            continue
        for path in glob.glob(os.path.join(model_dir, "*.log")):
            m = ZEROSHOT_LOG_PAT.match(os.path.basename(path))
            if not m:
                continue
            dataset = m.group(1)
            metrics = parse_log(path)
            if metrics is None:
                print(f"  [skip] {model_tag}/{os.path.basename(path)} — incomplete")
                continue
            results[dataset][model_tag] = metrics
    return results


def collect_rl(rl_dir, frac, model_tag="rl"):
    """Pick frac-filtered logs directly from rl_dir.
    Returns: dataset → model_tag → {profit, on_time, accuracy}"""
    results = defaultdict(dict)
    for path in glob.glob(os.path.join(rl_dir, f"*_frac{frac}.log")):
        m = RL_LOG_PAT.match(os.path.basename(path))
        if not m:
            continue
        dataset = m.group(1)
        metrics = parse_log(path)
        if metrics is None:
            print(f"  [skip] {os.path.basename(path)} — incomplete")
            continue
        results[dataset][model_tag] = metrics
    return results


# ── Layout ────────────────────────────────────────────────────────────────────

DATASET_LABELS = {
    "dataco":      "DataCo",
    "globalstore": "GlobalStore",
    "oas":         "OAS",
}

MODEL_ORDER = [
    "rl",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "qwen3-0.6B",
    "qwen3-1.7B",
    "qwen3-4B",
]

MODEL_LABELS = {
    "rl":          "RL",
    "gpt2":        "GPT-2",
    "gpt2-medium": "GPT-2 Med",
    "gpt2-large":  "GPT-2 Lg",
    "qwen3-0.6B":  "Qwen3-0.6B",
    "qwen3-1.7B":  "Qwen3-1.7B",
    "qwen3-4B":    "Qwen3-4B",
}


def sorted_models(model_tags):
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(model_tags, key=lambda m: order.get(m.lower(), 99))


# ── Plots ─────────────────────────────────────────────────────────────────────

BAR_COLORS = {
    "profit":  "#4C72B0",   # blue
    "on_time": "#DD8452",   # orange
    "sum":     "#55A868",   # green
}

RL_COLORS = {
    "profit":  "#4C72B0",
    "on_time": "#DD8452",
    "sum":     "#55A868",
}


def plot_per_dataset(results, out_dir):
    """One figure per dataset: grouped bars (profit, on_time, sum) per LLM model,
    with RL values as dashed horizontal reference lines."""
    llm_models = sorted_models({m for d in results.values() for m in d if m != "rl"})
    x = np.arange(len(llm_models))
    n_bars = 3
    width = 0.22

    for ds, model_map in sorted(results.items()):
        rl = model_map.get("rl", {})

        fig, ax = plt.subplots(figsize=(12, 5))

        # ── Grouped bars for LLM models ──────────────────────────────────────
        profits  = [model_map.get(m, {}).get("profit",  0.0) for m in llm_models]
        ontimes  = [model_map.get(m, {}).get("on_time", 0.0) for m in llm_models]
        sums     = [p + o for p, o in zip(profits, ontimes)]

        ax.bar(x - width, profits, width, label="Zero-Shot Profit",   color=BAR_COLORS["profit"])
        ax.bar(x,         ontimes, width, label="Zero-Shot On-Time",  color=BAR_COLORS["on_time"])
        ax.bar(x + width, sums,    width, label="Zero-Shot Combined", color=BAR_COLORS["sum"])

        # ── RL dashed reference lines ─────────────────────────────────────────
        if rl:
            rl_profit  = rl.get("profit",  0.0)
            rl_ontime  = rl.get("on_time", 0.0)
            rl_sum     = rl_profit + rl_ontime
            ax.axhline(rl_profit,  color=RL_COLORS["profit"],  linestyle="--", linewidth=1.5,
                       label=f"RL Profit ({rl_profit:.4f})")
            ax.axhline(rl_ontime,  color=RL_COLORS["on_time"], linestyle="--", linewidth=1.5,
                       label=f"RL On-Time ({rl_ontime:.4f})")
            ax.axhline(rl_sum,     color=RL_COLORS["sum"],     linestyle="--", linewidth=1.5,
                       label=f"RL Combined ({rl_sum:.4f})")

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in llm_models], fontsize=10)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, ncol=2, loc="upper right")

        dset_label = DATASET_LABELS.get(ds, ds)
        ax.set_title(f"{dset_label} — Zero-Shot (All Models) vs RL Full Training",
                     fontsize=12, fontweight="bold")

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"zeroshot_compare_{ds}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def format_table(results):
    all_models = sorted_models({m for d in results.values() for m in d})
    datasets = sorted(results.keys())

    header = f"{'Model':<16}" + "".join(
        f"  {DATASET_LABELS.get(d, d):^28}" for d in datasets
    )
    subheader = f"{'':16}" + "".join(
        f"  {'Profit':>8} {'OnTime':>8} {'Sum':>8}" for _ in datasets
    )
    sep = "-" * len(subheader)

    lines = ["\n" + header, subheader, sep]
    for model in all_models:
        row = f"{MODEL_LABELS.get(model, model):<16}"
        for ds in datasets:
            m = results[ds].get(model)
            if m:
                s = m["profit"] + m["on_time"]
                row += f"  {m['profit']:>8.4f} {m['on_time']:>8.4f} {s:>8.4f}"
            else:
                row += f"  {'—':>8} {'—':>8} {'—':>8}"
        lines.append(row)
    lines.append("")
    return "\n".join(lines)


def print_table(results, out_dir):
    table = format_table(results)
    print(table)
    out_path = os.path.join(out_dir, "zeroshot_compare.txt")
    with open(out_path, "w") as f:
        f.write(table)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for d in [ZEROSHOT_DIR, RL_DIR]:
        if not os.path.isdir(d):
            print(f"Error: directory not found: {d}")
            raise SystemExit(1)

    print(f"Zero-shot dir : {ZEROSHOT_DIR}")
    print(f"RL dir        : {RL_DIR}")
    print(f"Frac filter   : {FRAC}")

    results = collect_zeroshot(ZEROSHOT_DIR)
    rl_results = collect_rl(RL_DIR, FRAC)

    for dataset, model_map in rl_results.items():
        results[dataset].update(model_map)

    if not results:
        print("No completed log files found.")
        raise SystemExit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    print_table(results, OUT_DIR)
    plot_per_dataset(results, OUT_DIR)
    print("Done.")
