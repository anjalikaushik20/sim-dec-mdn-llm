"""
Compare zero-shot vocabalign + sample efficiency vocabalign vs RL baseline.

Merges:
  - Zero-shot logs  : {dataset}_{model}.log  (treated as frac=0)
  - Sample efficiency: per-model subdirs/{dataset}_frac{frac}.log
  - RL baseline      : {dataset}_frac{frac}.log  (space-separated metrics)

Usage:
    python3 compare_zeroshot_sampeff_rl.py \\
        --zeroshot  <zero_shot_dir> \\
        --sampeff   <sampeff_dir> [--sampeff <sampeff_dir2> ...] \\
        --rl        <rl_run_dir> \\
        --out       <output_dir>

Later --sampeff dirs take priority over earlier ones for the same model.

Example:
    python3 compare_zeroshot_sampeff_rl.py \\
        --zeroshot output/latest_output/zero_shot/vocabalign/20260523_030817 \\
        --sampeff  output/latest_output/sample_efficiency/all_vocabalign/20260521_234202 \\
        --sampeff  output/latest_output/sample_efficiency/all_vocabalign/20260523_045430 \\
        --rl       output/latest_output/sample_efficiency/rl_baseline/20260520_182054 \\
        --out      output/latest_output/comparisons/zeroshot_sampeff_vs_rl
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

# ── Constants ─────────────────────────────────────────────────────────────────

DATASET_LABELS = {"dataco": "DataCo", "globalstore": "GlobalStore", "oas": "OAS"}
DATASETS = ["dataco", "globalstore", "oas"]

MODEL_ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "qwen3-0.6B", "qwen3-1.7B", "qwen3-4B"]
MODEL_LABELS = {
    "gpt2":        "GPT-2",
    "gpt2-medium": "GPT-2-Med",
    "gpt2-large":  "GPT-2-Lg",
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
RL_COLOR = "#E53935"

# ── Parsers ───────────────────────────────────────────────────────────────────

_ZEROSHOT_PAT = re.compile(r"([a-z]+)_([a-zA-Z0-9.\-]+)\.log$")
_SAMPEFF_PAT  = re.compile(r"([a-z]+)_frac([\d.]+)\.log$")


def _parse_llm(path):
    m = {}
    for key, pat in [
        ("profit",  re.compile(r"best_profit=([\d.]+)")),
        ("on_time", re.compile(r"best_on_time=([\d.]+)")),
    ]:
        with open(path) as f:
            for line in f:
                hit = pat.search(line)
                if hit:
                    m[key] = float(hit.group(1))
    return m if len(m) == 2 else None


def _parse_rl(path):
    m = {}
    for key, pat in [
        ("profit",  re.compile(r"best_profit\s+([\d.]+)")),
        ("on_time", re.compile(r"best_on_time\s+([\d.]+)")),
        ("pmp_1",   re.compile(r"best_pmp_1\s+([\d.]+)")),
        ("pmp_2",   re.compile(r"best_pmp_2\s+([\d.]+)")),
        ("pmp_3",   re.compile(r"best_pmp_3\s+([\d.]+)")),
    ]:
        with open(path) as f:
            for line in f:
                hit = pat.search(line)
                if hit:
                    m[key] = float(hit.group(1))
    return m if "profit" in m and "on_time" in m else None


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_zeroshot(zs_dir):
    """Returns dict: model -> dataset -> {profit, on_time}  (frac=0 implicit)"""
    data = defaultdict(dict)
    for path in glob.glob(os.path.join(zs_dir, "*.log")):
        fname = os.path.basename(path)
        m = _ZEROSHOT_PAT.match(fname)
        if not m:
            continue
        dataset, model = m.group(1), m.group(2)
        metrics = _parse_llm(path)
        if metrics is None:
            print(f"  [zeroshot skip] {fname} — incomplete")
            continue
        data[model][dataset] = metrics
    return dict(data)


def load_sampeff_dir(se_dir):
    """Returns dict: model -> dataset -> list of (frac, profit, on_time)"""
    data = defaultdict(lambda: defaultdict(list))
    for model_dir in sorted(glob.glob(os.path.join(se_dir, "*"))):
        if not os.path.isdir(model_dir):
            continue
        model = os.path.basename(model_dir)
        if model in ("ckpts", "job_queue.txt", "queue.lock"):
            continue
        for path in glob.glob(os.path.join(model_dir, "*.log")):
            fname = os.path.basename(path)
            mp = _SAMPEFF_PAT.match(fname)
            if not mp:
                continue
            dataset, frac = mp.group(1), float(mp.group(2))
            metrics = _parse_llm(path)
            if metrics is None:
                print(f"  [sampeff skip] {model}/{fname} — incomplete")
                continue
            data[model][dataset].append((frac, metrics["profit"], metrics["on_time"]))
    # sort by frac
    result = {}
    for model, ds_map in data.items():
        result[model] = {}
        for ds, rows in ds_map.items():
            result[model][ds] = sorted(rows, key=lambda r: r[0])
    return result


def merge_sampeff(dirs):
    """Merge multiple sampeff dirs. Later dirs override earlier for same model."""
    merged = {}
    for d in dirs:
        new = load_sampeff_dir(d)
        for model, ds_map in new.items():
            if model not in merged:
                merged[model] = ds_map
            else:
                for ds, rows in ds_map.items():
                    existing = {r[0]: r for r in merged[model].get(ds, [])}
                    for row in rows:
                        existing[row[0]] = row
                    merged[model][ds] = sorted(existing.values(), key=lambda r: r[0])
    return merged


def load_rl(rl_dir):
    """Returns dict: dataset -> list of (frac, profit, on_time, pmp_1, pmp_2, pmp_3)"""
    data = defaultdict(list)
    for path in glob.glob(os.path.join(rl_dir, "*.log")):
        fname = os.path.basename(path)
        mp = _SAMPEFF_PAT.match(fname)
        if not mp:
            continue
        dataset, frac = mp.group(1), float(mp.group(2))
        metrics = _parse_rl(path)
        if metrics is None:
            print(f"  [rl skip] {fname} — incomplete")
            continue
        data[dataset].append((
            frac,
            metrics["profit"],
            metrics["on_time"],
            metrics.get("pmp_1", float("nan")),
            metrics.get("pmp_2", float("nan")),
            metrics.get("pmp_3", float("nan")),
        ))
    for ds in data:
        data[ds].sort(key=lambda r: r[0])
    return dict(data)


# ── Helpers ───────────────────────────────────────────────────────────────────

def sorted_models(tags):
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(tags, key=lambda m: order.get(m, 99))


def frac_label(f):
    if f == 0.0:
        return "0% (zero-shot)"
    return f"{int(round(f * 100))}%"


# ── Text table ────────────────────────────────────────────────────────────────

def build_and_save_table(zs_data, se_data, rl_data, out_dir):
    all_models = sorted_models(
        set(zs_data.keys()) | set(se_data.keys())
    )
    all_fracs = sorted(
        {0.0}
        | {r[0] for ds_map in se_data.values() for rows in ds_map.values() for r in rows}
        | {r[0] for rows in rl_data.values() for r in rows}
    )

    col_w = 8
    lines = []

    for ds in DATASETS:
        dset_label = DATASET_LABELS.get(ds, ds)
        lines.append(f"\n{'='*100}")
        lines.append(f"Dataset: {dset_label}")
        lines.append("=" * 100)

        # Build model column headers
        col_labels = ["RL"] + [MODEL_LABELS.get(m, m) for m in all_models]
        hdr = f"{'Frac':<16}"
        for lbl in col_labels:
            hdr += f"  {lbl:^{col_w*3+2}}"
        lines.append(hdr)

        sub = f"{'':16}"
        for _ in col_labels:
            sub += f"  {'Profit':>{col_w}} {'OnTime':>{col_w}} {'Total':>{col_w}}"
        lines.append(sub)
        lines.append("-" * len(sub))

        for frac in all_fracs:
            row = f"{frac_label(frac):<16}"

            # RL column
            if ds in rl_data:
                rl_row = next((r for r in rl_data[ds] if r[0] == frac), None)
                if rl_row:
                    p, o = rl_row[1], rl_row[2]
                    row += f"  {p:>{col_w}.4f} {o:>{col_w}.4f} {p+o:>{col_w}.4f}"
                else:
                    row += f"  {'—':>{col_w}} {'—':>{col_w}} {'—':>{col_w}}"
            else:
                row += f"  {'—':>{col_w}} {'—':>{col_w}} {'—':>{col_w}}"

            # Model columns
            for model in all_models:
                if frac == 0.0:
                    # zero-shot
                    entry = zs_data.get(model, {}).get(ds)
                    if entry:
                        p, o = entry["profit"], entry["on_time"]
                        row += f"  {p:>{col_w}.4f} {o:>{col_w}.4f} {p+o:>{col_w}.4f}"
                    else:
                        row += f"  {'—':>{col_w}} {'—':>{col_w}} {'—':>{col_w}}"
                else:
                    rows = se_data.get(model, {}).get(ds, [])
                    entry = next((r for r in rows if r[0] == frac), None)
                    if entry:
                        p, o = entry[1], entry[2]
                        row += f"  {p:>{col_w}.4f} {o:>{col_w}.4f} {p+o:>{col_w}.4f}"
                    else:
                        row += f"  {'—':>{col_w}} {'—':>{col_w}} {'—':>{col_w}}"
            lines.append(row)

    table_str = "\n".join(lines)
    print(table_str)
    path = os.path.join(out_dir, "compare_table.txt")
    with open(path, "w") as f:
        f.write(table_str + "\n")
    print(f"\nSaved: {path}")


# ── RL check vs user-provided values ─────────────────────────────────────────

USER_RL = {
    "dataco":      {"profit": 0.5268, "on_time": 0.2464, "pmp_1": 0.361, "pmp_2": 0.450, "pmp_3": 0.450},
    "globalstore": {"profit": 0.3445, "on_time": 0.7582, "pmp_1": 0.199, "pmp_2": 0.356, "pmp_3": 0.356},
    "oas":         {"profit": 0.4846, "on_time": 0.0828, "pmp_1": 0.234, "pmp_2": 0.373, "pmp_3": 0.373},
}


def check_rl_vs_user(rl_data):
    print("\n" + "=" * 80)
    print("RL CHECK: disk values vs. your recorded numbers")
    print("=" * 80)
    fmt = "  {:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}"
    print(fmt.format("Dataset/Frac", "Profit", "OnTime", "Total", "PMP-10%", "PMP-20%", "PMP-30%"))
    print("-" * 80)

    for ds in DATASETS:
        user = USER_RL[ds]
        print(f"\n{DATASET_LABELS[ds]}:")
        print(f"  {'[Your values]':<14} {user['profit']:>8.4f} {user['on_time']:>8.4f} "
              f"{user['profit']+user['on_time']:>8.4f} "
              f"{user['pmp_1']:>8.3f} {user['pmp_2']:>8.3f} {user['pmp_3']:>8.3f}")
        if ds not in rl_data:
            print("  [no RL data found on disk]")
            continue
        for r in rl_data[ds]:
            frac, p, o, p1, p2, p3 = r
            print(f"  frac={frac:.2f}          {p:>8.4f} {o:>8.4f} {p+o:>8.4f} "
                  f"{p1:>8.4f} {p2:>8.4f} {p3:>8.4f}")


# ── Plots ─────────────────────────────────────────────────────────────────────

METRICS = [
    ("profit",  "Profit"),
    ("on_time", "On-Time Ratio"),
    ("total",   "Profit + On-Time"),
]


def plot_per_dataset(zs_data, se_data, rl_data, out_dir):
    all_models = sorted_models(set(zs_data.keys()) | set(se_data.keys()))
    all_fracs = sorted(
        {0.0}
        | {r[0] for ds_map in se_data.values() for rows in ds_map.values() for r in rows}
        | {r[0] for rows in rl_data.values() for r in rows}
    )

    for ds in DATASETS:
        dset_label = DATASET_LABELS.get(ds, ds)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

        for ax, (key, metric_label) in zip(axes, METRICS):
            # RL line
            if ds in rl_data:
                rl_rows = rl_data[ds]
                fracs = [r[0] for r in rl_rows]
                if key == "profit":   vals = [r[1] for r in rl_rows]
                elif key == "on_time": vals = [r[2] for r in rl_rows]
                else:                  vals = [r[1]+r[2] for r in rl_rows]
                ax.plot(fracs, vals, color=RL_COLOR, linewidth=2.5,
                        linestyle="--", marker="s", markersize=6, label="RL baseline")

            # Model lines
            for model in all_models:
                pts = []
                # zero-shot point
                zs = zs_data.get(model, {}).get(ds)
                if zs:
                    pts.append((0.0, zs["profit"], zs["on_time"]))
                # sample efficiency points
                for r in se_data.get(model, {}).get(ds, []):
                    pts.append(r)
                if not pts:
                    continue
                pts.sort(key=lambda x: x[0])
                fracs = [p[0] for p in pts]
                if key == "profit":   vals = [p[1] for p in pts]
                elif key == "on_time": vals = [p[2] for p in pts]
                else:                  vals = [p[1]+p[2] for p in pts]
                color = MODEL_COLORS.get(model, "#555")
                ax.plot(fracs, vals, color=color, linewidth=2,
                        marker="o", markersize=5,
                        label=MODEL_LABELS.get(model, model))

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_xticks(all_fracs)
            ax.set_xticklabels([frac_label(f).replace("% (zero-shot)", "%\n(zero-shot)")
                                 for f in all_fracs], fontsize=6)

        fig.suptitle(f"{dset_label} — Zero-shot + Sample Efficiency vs RL",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(out_dir, f"compare_{ds}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def plot_per_model(zs_data, se_data, rl_data, out_dir):
    all_models = sorted_models(set(zs_data.keys()) | set(se_data.keys()))
    ds_colors = {"dataco": "#2196F3", "globalstore": "#FF5722", "oas": "#4CAF50"}

    for model in all_models:
        model_label = MODEL_LABELS.get(model, model)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

        for ax, (key, metric_label) in zip(axes, METRICS):
            for ds in DATASETS:
                color = ds_colors.get(ds, "#555")
                dset_label = DATASET_LABELS.get(ds, ds)

                # RL dashed
                if ds in rl_data:
                    rl_rows = rl_data[ds]
                    fracs = [r[0] for r in rl_rows]
                    if key == "profit":   vals = [r[1] for r in rl_rows]
                    elif key == "on_time": vals = [r[2] for r in rl_rows]
                    else:                  vals = [r[1]+r[2] for r in rl_rows]
                    ax.plot(fracs, vals, color=color, linewidth=2,
                            linestyle="--", marker="s", markersize=5, alpha=0.7,
                            label=f"{dset_label} RL")

                # Model solid (zero-shot + sampeff)
                pts = []
                zs = zs_data.get(model, {}).get(ds)
                if zs:
                    pts.append((0.0, zs["profit"], zs["on_time"]))
                for r in se_data.get(model, {}).get(ds, []):
                    pts.append(r)
                if pts:
                    pts.sort(key=lambda x: x[0])
                    fracs = [p[0] for p in pts]
                    if key == "profit":   vals = [p[1] for p in pts]
                    elif key == "on_time": vals = [p[2] for p in pts]
                    else:                  vals = [p[1]+p[2] for p in pts]
                    ax.plot(fracs, vals, color=color, linewidth=2.5,
                            marker="o", markersize=6,
                            label=f"{dset_label} VocabAlign")

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{model_label} — VocabAlign (solid) vs RL (dashed)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        safe = re.sub(r"[^a-z0-9]+", "_", model.lower())
        path = os.path.join(out_dir, f"compare_model_{safe}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zeroshot", required=True,
                        help="Zero-shot dir with {dataset}_{model}.log files")
    parser.add_argument("--sampeff", action="append", required=True,
                        help="Sample efficiency dir (with per-model subdirs); repeat for multiple")
    parser.add_argument("--rl", required=True,
                        help="RL baseline dir with {dataset}_frac{frac}.log files")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"\nLoading zero-shot from:      {args.zeroshot}")
    zs_data = load_zeroshot(args.zeroshot)
    print(f"  Models found: {sorted_models(zs_data.keys())}")

    print(f"\nLoading sample efficiency from: {args.sampeff}")
    se_data = merge_sampeff(args.sampeff)
    print(f"  Models found: {sorted_models(se_data.keys())}")
    for m, ds_map in se_data.items():
        fracs = sorted({r[0] for rows in ds_map.values() for r in rows})
        print(f"    {MODEL_LABELS.get(m, m)}: datasets={sorted(ds_map)}, fracs={fracs}")

    print(f"\nLoading RL from:             {args.rl}")
    rl_data = load_rl(args.rl)
    print(f"  Datasets found: {sorted(rl_data.keys())}")
    for ds, rows in rl_data.items():
        fracs = [r[0] for r in rows]
        print(f"    {DATASET_LABELS.get(ds, ds)}: fracs={fracs}")

    print("\n" + "="*80)
    check_rl_vs_user(rl_data)

    print("\n" + "="*80)
    print("COMPARISON TABLE")
    build_and_save_table(zs_data, se_data, rl_data, args.out)

    print("\n--- Generating plots ---")
    plot_per_dataset(zs_data, se_data, rl_data, args.out)
    plot_per_model(zs_data, se_data, rl_data, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
