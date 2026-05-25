"""
Compare RL full-training (single reference) vs VocabAlign LLM at all training fractions.

RL is taken at frac=1.0 (or highest available frac if 1.0 is missing).
LLM includes zero-shot (frac=0) + all sample-efficiency fractions.

Usage:
    python3 compare_fullrl_vs_llm_fracs.py \\
        --zeroshot <zero_shot_dir> \\
        --sampeff  <sampeff_dir> [--sampeff <dir2> ...] \\
        --rl       <rl_run_dir> \\
        --out      <output_dir>

Example:
    python3 compare_fullrl_vs_llm_fracs.py \\
        --zeroshot /data/akaush39/sim-to-dec/output/latest_output/zero_shot/vocabalign/20260523_030817 \\
        --sampeff  /data/akaush39/sim-to-dec/output/latest_output/sample_efficiency/all_vocabalign/20260521_234202 \\
        --sampeff  /data/akaush39/sim-to-dec/output/latest_output/sample_efficiency/all_vocabalign/20260523_045430 \\
        --rl       /data/akaush39/sim-to-dec/output/latest_output/sample_efficiency/rl_baseline/20260520_182054 \\
        --out      /data/akaush39/sim-to-dec/output/latest_output/comparisons/fullrl_vs_llm_fracs
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
METRICS = [("profit", "Profit"), ("on_time", "On-Time Ratio"), ("total", "Profit + On-Time")]

# ── Parsers ───────────────────────────────────────────────────────────────────

_ZS_PAT = re.compile(r"([a-z]+)_([a-zA-Z0-9.\-]+)\.log$")
_SE_PAT  = re.compile(r"([a-z]+)_frac([\d.]+)\.log$")


def _parse_llm(path):
    m = {}
    for key, pat in [("profit",  re.compile(r"best_profit=([\d.]+)")),
                     ("on_time", re.compile(r"best_on_time=([\d.]+)"))]:
        with open(path) as f:
            for line in f:
                hit = pat.search(line)
                if hit:
                    m[key] = float(hit.group(1))
    return m if len(m) == 2 else None


def _parse_rl(path):
    m = {}
    for key, pat in [("profit",  re.compile(r"best_profit\s+([\d.]+)")),
                     ("on_time", re.compile(r"best_on_time\s+([\d.]+)"))]:
        with open(path) as f:
            for line in f:
                hit = pat.search(line)
                if hit:
                    m[key] = float(hit.group(1))
    return m if len(m) == 2 else None


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_zeroshot(zs_dir):
    data = defaultdict(dict)
    for path in glob.glob(os.path.join(zs_dir, "*.log")):
        m = _ZS_PAT.match(os.path.basename(path))
        if not m:
            continue
        ds, model = m.group(1), m.group(2)
        metrics = _parse_llm(path)
        if metrics is None:
            print(f"  [zeroshot skip] {os.path.basename(path)}")
            continue
        data[model][ds] = metrics
    return dict(data)


def load_sampeff_dirs(dirs):
    merged = {}
    for d in dirs:
        for model_dir in sorted(glob.glob(os.path.join(d, "*"))):
            if not os.path.isdir(model_dir):
                continue
            model = os.path.basename(model_dir)
            if model in ("ckpts", "job_queue.txt", "queue.lock"):
                continue
            ds_map = defaultdict(list)
            for path in glob.glob(os.path.join(model_dir, "*.log")):
                mp = _SE_PAT.match(os.path.basename(path))
                if not mp:
                    continue
                ds, frac = mp.group(1), float(mp.group(2))
                metrics = _parse_llm(path)
                if metrics is None:
                    print(f"  [sampeff skip] {model}/{os.path.basename(path)}")
                    continue
                ds_map[ds].append((frac, metrics["profit"], metrics["on_time"]))
            if model not in merged:
                merged[model] = {}
            for ds, rows in ds_map.items():
                existing = {r[0]: r for r in merged[model].get(ds, [])}
                for r in rows:
                    existing[r[0]] = r
                merged[model][ds] = sorted(existing.values(), key=lambda r: r[0])
    return merged


def load_rl_full(rl_dir):
    """Returns dict: dataset -> (frac_used, profit, on_time) — highest complete frac."""
    all_rows = defaultdict(list)
    for path in glob.glob(os.path.join(rl_dir, "*.log")):
        mp = _SE_PAT.match(os.path.basename(path))
        if not mp:
            continue
        ds, frac = mp.group(1), float(mp.group(2))
        metrics = _parse_rl(path)
        if metrics is None:
            print(f"  [rl skip] {os.path.basename(path)} — incomplete")
            continue
        all_rows[ds].append((frac, metrics["profit"], metrics["on_time"]))
    result = {}
    for ds, rows in all_rows.items():
        rows.sort(key=lambda r: r[0], reverse=True)
        best = rows[0]
        result[ds] = best
        if best[0] < 1.0:
            print(f"  [rl warn] {DATASET_LABELS.get(ds, ds)}: frac1.0 missing, "
                  f"using frac={best[0]}")
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def sorted_models(tags):
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(tags, key=lambda m: order.get(m, 99))


def frac_label(f):
    return "0%\n(zero-shot)" if f == 0.0 else f"{int(round(f * 100))}%"


def get_val(row, key):
    if key == "profit":  return row[1]
    if key == "on_time": return row[2]
    return row[1] + row[2]


# ── Text table ────────────────────────────────────────────────────────────────

def build_table(zs_data, se_data, rl_full, out_dir):
    all_models = sorted_models(set(zs_data.keys()) | set(se_data.keys()))
    all_fracs  = sorted(
        {0.0} | {r[0] for ds_map in se_data.values() for rows in ds_map.values() for r in rows}
    )
    col_w = 8

    lines = []
    for ds in DATASETS:
        dset_label = DATASET_LABELS.get(ds, ds)
        rl_row = rl_full.get(ds)
        rl_frac_str = f"(frac={rl_row[0]:.2f})" if rl_row else "(N/A)"

        lines.append(f"\n{'='*100}")
        lines.append(f"Dataset: {dset_label}  |  RL reference {rl_frac_str}")
        lines.append("=" * 100)

        # Header
        col_labels = [f"RL-full{rl_frac_str}"] + [MODEL_LABELS.get(m, m) for m in all_models]
        hdr = f"{'Frac':<18}"
        for lbl in col_labels:
            hdr += f"  {lbl:^{col_w*3+2}}"
        lines.append(hdr)
        sub = f"{'':18}"
        for _ in col_labels:
            sub += f"  {'Profit':>{col_w}} {'OnTime':>{col_w}} {'Total':>{col_w}}"
        lines.append(sub)
        lines.append("-" * len(sub))

        for frac in all_fracs:
            row = f"{frac_label(frac).replace(chr(10),' '):<18}"

            # RL column — always the same full-training value
            if rl_row:
                p, o = rl_row[1], rl_row[2]
                row += f"  {p:>{col_w}.4f} {o:>{col_w}.4f} {p+o:>{col_w}.4f}"
            else:
                row += f"  {'—':>{col_w}} {'—':>{col_w}} {'—':>{col_w}}"

            # Model columns
            for model in all_models:
                if frac == 0.0:
                    entry = zs_data.get(model, {}).get(ds)
                    if entry:
                        p, o = entry["profit"], entry["on_time"]
                        row += f"  {p:>{col_w}.4f} {o:>{col_w}.4f} {p+o:>{col_w}.4f}"
                    else:
                        row += f"  {'—':>{col_w}} {'—':>{col_w}} {'—':>{col_w}}"
                else:
                    rows_ds = se_data.get(model, {}).get(ds, [])
                    entry = next((r for r in rows_ds if r[0] == frac), None)
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


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_per_dataset(zs_data, se_data, rl_full, out_dir):
    """One figure per dataset: 3 metric subplots, RL as horizontal dashed line."""
    all_models = sorted_models(set(zs_data.keys()) | set(se_data.keys()))
    all_fracs  = sorted(
        {0.0} | {r[0] for ds_map in se_data.values() for rows in ds_map.values() for r in rows}
    )

    for ds in DATASETS:
        dset_label = DATASET_LABELS.get(ds, ds)
        rl_row = rl_full.get(ds)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax, (key, metric_label) in zip(axes, METRICS):
            # RL horizontal reference line
            if rl_row:
                rl_val = get_val(rl_row, key)
                rl_frac = rl_row[0]
                ax.axhline(rl_val, color=RL_COLOR, linewidth=2.5, linestyle="--",
                           label=f"RL full (frac={int(rl_frac*100)}%): {rl_val:.4f}")

            # LLM lines
            for model in all_models:
                pts = []
                zs = zs_data.get(model, {}).get(ds)
                if zs:
                    pts.append((0.0, zs["profit"], zs["on_time"]))
                for r in se_data.get(model, {}).get(ds, []):
                    pts.append(r)
                if not pts:
                    continue
                pts.sort(key=lambda x: x[0])
                fracs = [p[0] for p in pts]
                vals  = [get_val(p, key) for p in pts]
                ax.plot(fracs, vals, color=MODEL_COLORS.get(model, "#555"),
                        linewidth=2, marker="o", markersize=5,
                        label=MODEL_LABELS.get(model, model))

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_xticks(all_fracs)
            ax.set_xticklabels([frac_label(f).replace("\n", "\n") for f in all_fracs], fontsize=6)

        fig.suptitle(f"{dset_label} — RL Full Training vs VocabAlign at All Fractions",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(out_dir, f"compare_{ds}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def plot_per_model(zs_data, se_data, rl_full, out_dir):
    """One figure per model: 3 metric subplots, all datasets, RL as horizontal line."""
    all_models = sorted_models(set(zs_data.keys()) | set(se_data.keys()))
    ds_colors = {"dataco": "#2196F3", "globalstore": "#FF5722", "oas": "#4CAF50"}
    all_fracs  = sorted(
        {0.0} | {r[0] for ds_map in se_data.values() for rows in ds_map.values() for r in rows}
    )

    for model in all_models:
        model_label = MODEL_LABELS.get(model, model)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax, (key, metric_label) in zip(axes, METRICS):
            for ds in DATASETS:
                color = ds_colors.get(ds, "#555")
                dset_label = DATASET_LABELS.get(ds, ds)

                # RL horizontal reference
                rl_row = rl_full.get(ds)
                if rl_row:
                    rl_val = get_val(rl_row, key)
                    ax.axhline(rl_val, color=color, linewidth=1.8, linestyle="--", alpha=0.75,
                               label=f"{dset_label} RL-full: {rl_val:.4f}")

                # LLM line
                pts = []
                zs = zs_data.get(model, {}).get(ds)
                if zs:
                    pts.append((0.0, zs["profit"], zs["on_time"]))
                for r in se_data.get(model, {}).get(ds, []):
                    pts.append(r)
                if pts:
                    pts.sort(key=lambda x: x[0])
                    fracs = [p[0] for p in pts]
                    vals  = [get_val(p, key) for p in pts]
                    ax.plot(fracs, vals, color=color, linewidth=2.5,
                            marker="o", markersize=6,
                            label=f"{dset_label} VocabAlign")

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training Fraction", fontsize=9)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_xticks(all_fracs)
            ax.set_xticklabels([frac_label(f).replace("\n", "\n") for f in all_fracs], fontsize=6)

        fig.suptitle(f"{model_label} — VocabAlign vs RL Full Training (dashed)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        safe = re.sub(r"[^a-z0-9]+", "_", model.lower())
        path = os.path.join(out_dir, f"compare_model_{safe}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def plot_grouped_fracs(zs_data, se_data, rl_full, out_dir):
    """
    For each dataset: one figure with 3 metric subplots.
    x-axis = training fraction; one bar group per fraction; bars = models + RL.
    RL is shown as a single bar repeated across all fractions (or as a hline overlay).
    """
    all_models = sorted_models(set(zs_data.keys()) | set(se_data.keys()))
    all_fracs  = sorted(
        {0.0} | {r[0] for ds_map in se_data.values() for rows in ds_map.values() for r in rows}
    )
    n_models = len(all_models)
    x = np.arange(len(all_fracs))
    width = 0.8 / n_models

    for ds in DATASETS:
        dset_label = DATASET_LABELS.get(ds, ds)
        rl_row = rl_full.get(ds)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, (key, metric_label) in zip(axes, METRICS):
            # RL horizontal reference line
            if rl_row:
                rl_val = get_val(rl_row, key)
                ax.axhline(rl_val, color=RL_COLOR, linewidth=2, linestyle="--", zorder=5,
                           label=f"RL full: {rl_val:.4f}")

            # Model bars
            for j, model in enumerate(all_models):
                vals = []
                for frac in all_fracs:
                    if frac == 0.0:
                        entry = zs_data.get(model, {}).get(ds)
                        vals.append(get_val((0, entry["profit"], entry["on_time"]), key)
                                    if entry else 0.0)
                    else:
                        rows_ds = se_data.get(model, {}).get(ds, [])
                        r = next((r for r in rows_ds if r[0] == frac), None)
                        vals.append(get_val(r, key) if r else 0.0)
                offset = (j - (n_models - 1) / 2) * width
                ax.bar(x + offset, vals, width * 0.9,
                       color=MODEL_COLORS.get(model, "#555"),
                       label=MODEL_LABELS.get(model, model),
                       alpha=0.85, edgecolor="white")

            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([frac_label(f).replace("\n", "\n") for f in all_fracs], fontsize=7)
            ax.set_xlabel("LLM Training Fraction", fontsize=9)
            ax.legend(fontsize=6, ncol=2)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"{dset_label} — All Models at Each Training Fraction vs RL Full",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(out_dir, f"grouped_{ds}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zeroshot", required=True)
    parser.add_argument("--sampeff",  action="append", required=True)
    parser.add_argument("--rl",       required=True)
    parser.add_argument("--out",      required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading zero-shot:      {args.zeroshot}")
    zs_data = load_zeroshot(args.zeroshot)
    print(f"  Models: {sorted_models(zs_data.keys())}")

    print(f"Loading sample efficiency: {args.sampeff}")
    se_data = load_sampeff_dirs(args.sampeff)
    print(f"  Models: {sorted_models(se_data.keys())}")
    for m, ds_map in se_data.items():
        fracs = sorted({r[0] for rows in ds_map.values() for r in rows})
        print(f"    {MODEL_LABELS.get(m, m)}: {fracs}")

    print(f"Loading RL (full training): {args.rl}")
    rl_full = load_rl_full(args.rl)
    print("  RL full-training reference:")
    for ds, row in sorted(rl_full.items()):
        p, o = row[1], row[2]
        print(f"    {DATASET_LABELS.get(ds,ds)}: frac={row[0]:.2f}  "
              f"profit={p:.4f}  on_time={o:.4f}  total={p+o:.4f}")

    print("\n--- Summary Table ---")
    build_table(zs_data, se_data, rl_full, args.out)

    print("\n--- Generating plots ---")
    plot_per_dataset(zs_data, se_data, rl_full, args.out)
    plot_per_model(zs_data, se_data, rl_full, args.out)
    plot_grouped_fracs(zs_data, se_data, rl_full, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
