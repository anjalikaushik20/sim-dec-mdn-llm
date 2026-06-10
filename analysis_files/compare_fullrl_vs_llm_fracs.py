"""
Compare RL full-training (single reference) vs VocabAlign LLM at all training fractions.

RL is taken at frac=1.0 (or highest available frac if 1.0 is missing), averaged across seeds.
LLM includes zero-shot (frac=0) + all sample-efficiency fractions, averaged across runs/seeds.
"""

import os
import re
import glob
import argparse
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--GPU", type=int, default=0)
_args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.GPU)

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
    "gpt2":       ZEROSHOT_RUN_DIRS_GPT2,
    "gpt2-large": ZEROSHOT_RUN_DIRS_GPT2_LARGE,
    "phi4-mini":  ZEROSHOT_RUN_DIRS_PHI4_MINI,
    "qwen3-0.6B": ZEROSHOT_RUN_DIRS_QWEN_0_6,
    "qwen3-1.7B": ZEROSHOT_RUN_DIRS_QWEN_1_7,
}

SAMPEFF_DIRS = ["output/decision_maker/all_fracs/models"]
OUT_DIR      = "output/decision_maker/comparisons/fullrl_vs_llm_fracs"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

DATASET_LABELS = {
    "dataco":      "DataCo",
    "globalstore": "GlobalStore",
    "oas":         "OAS",
    "scsp":        "SCSP",
}
DATASETS = ["dataco", "globalstore", "oas", "scsp"]

MODEL_ORDER = ["gpt2", "gpt2-large", "qwen3-0.6B", "qwen3-1.7B", "phi4-mini"]
MODEL_LABELS = {
    "gpt2":        "GPT-2",
    "gpt2-large":  "GPT-2-Large",
    "qwen3-0.6B":  "Qwen3-0.6B",
    "qwen3-1.7B":  "Qwen3-1.7B",
    "phi4-mini":   "Phi4-mini",
}
MODEL_COLORS = {
    "gpt2":        "#90A4AE",
    "gpt2-large":  "#263238",
    "qwen3-0.6B":  "#81D4FA",
    "qwen3-1.7B":  "#0288D1",
    "phi4-mini":   "#01579B",
}
RL_COLOR = "#E53935"
METRICS = [("profit", "Profit"), ("on_time", "On-Time Ratio"), ("total", "Profit + On-Time")]

# ── Dataset name normalisation ────────────────────────────────────────────────

_DS_NORM = {"supplychainshipmentpricing": "scsp"}

def _norm_ds(s):
    s = s.lower()
    return _DS_NORM.get(s, s)

# ── Parsers ───────────────────────────────────────────────────────────────────

_ZS_DS_PAT = re.compile(r"([a-z]+)\.log$")
_SE_PAT    = re.compile(r"([a-z]+)_frac([\d.]+)\.log$")


def _parse_llm(path):
    m = {}
    for key, pat in [("profit",  re.compile(r"best_profit=([\d.eE+\-]+)")),
                     ("on_time", re.compile(r"best_on_time=([\d.eE+\-]+)"))]:
        try:
            with open(path) as f:
                for line in f:
                    hit = pat.search(line)
                    if hit:
                        m[key] = float(hit.group(1))
        except OSError:
            pass
    return m if len(m) == 2 else None


def _parse_rl(path):
    m = {}
    for key, pat in [("profit",  re.compile(r"best_profit[=\s]+([\d.eE+\-]+)")),
                     ("on_time", re.compile(r"best_on_time[=\s]+([\d.eE+\-]+)"))]:
        try:
            with open(path) as f:
                for line in f:
                    hit = pat.search(line)
                    if hit:
                        m[key] = float(hit.group(1))
        except OSError:
            pass
    return m if len(m) == 2 else None


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_zeroshot(model_run_dirs):
    """Load zero-shot from per-model run dirs (01–05), average across runs.
    model_run_dirs: {model_tag: [dir1, dir2, ...]}
    Returns: {model_tag: {dataset: {"profit": float, "on_time": float}}}
    """
    data = {}
    for model, run_dirs in model_run_dirs.items():
        ds_vals = defaultdict(list)
        for run_dir in run_dirs:
            if not os.path.isdir(run_dir):
                continue
            for path in glob.glob(os.path.join(run_dir, "*.log")):
                m = _ZS_DS_PAT.match(os.path.basename(path))
                if not m:
                    continue
                ds = _norm_ds(m.group(1))
                metrics = _parse_llm(path)
                if metrics is None:
                    print(f"  [zeroshot skip] {model}/{os.path.basename(path)}")
                    continue
                ds_vals[ds].append((metrics["profit"], metrics["on_time"]))
        if not ds_vals:
            continue
        data[model] = {
            ds: {
                "profit":  float(np.mean([v[0] for v in vals])),
                "on_time": float(np.mean([v[1] for v in vals])),
            }
            for ds, vals in ds_vals.items()
        }
    return data


def load_sampeff_dirs(dirs):
    """Load sample efficiency logs, averaging across seeds.
    Handles both structures:
      seed-based: {dir}/**/seed{N}/{model}/{dataset}_frac{f}.log
      flat:       {dir}/{model}/{dataset}_frac{f}.log
    """
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for d in dirs:
        for path in glob.glob(os.path.join(d, "**", "*.log"), recursive=True):
            parts = path.replace("\\", "/").split("/")
            fname = parts[-1]
            mp = _SE_PAT.match(fname)
            if not mp:
                continue

            seed_seg = next((p for p in parts if re.match(r"seed\d+$", p)), None)
            if seed_seg:
                si = parts.index(seed_seg)
                if si + 2 >= len(parts):
                    continue
                model = parts[si + 1]
            else:
                model = parts[-2]

            if model in ("ckpts", "job_queue.txt", "queue.lock"):
                continue

            ds   = _norm_ds(mp.group(1))
            frac = float(mp.group(2))
            metrics = _parse_llm(path)
            if metrics is None:
                continue
            raw[model][ds][frac].append((metrics["profit"], metrics["on_time"]))

    result = {}
    for model, ds_map in raw.items():
        result[model] = {}
        for ds, frac_map in ds_map.items():
            avg_rows = []
            for frac in sorted(frac_map):
                vals = frac_map[frac]
                avg_rows.append((
                    frac,
                    float(np.mean([v[0] for v in vals])),
                    float(np.mean([v[1] for v in vals])),
                ))
            result[model][ds] = avg_rows
    return result


def load_rl_full(rl_run_dirs):
    """Load RL from seed dirs at frac=1.00, average across seeds.
    Returns: {dataset: (frac, mean_profit, mean_on_time)}
    """
    rows_by_ds_frac = defaultdict(list)
    for seed_dir in rl_run_dirs:
        if not os.path.isdir(seed_dir):
            print(f"  [rl warn] missing: {seed_dir}")
            continue
        for path in glob.glob(os.path.join(seed_dir, "*.log")):
            mp = _SE_PAT.match(os.path.basename(path))
            if not mp:
                continue
            ds   = _norm_ds(mp.group(1))
            frac = float(mp.group(2))
            metrics = _parse_rl(path)
            if metrics is None:
                continue
            rows_by_ds_frac[(ds, frac)].append((metrics["profit"], metrics["on_time"]))

    ds_frac_map = defaultdict(dict)
    for (ds, frac), vals in rows_by_ds_frac.items():
        ds_frac_map[ds][frac] = vals

    result = {}
    for ds, frac_map in ds_frac_map.items():
        best_frac = max(frac_map.keys())
        vals      = frac_map[best_frac]
        profits   = [v[0] for v in vals]
        ontimes   = [v[1] for v in vals]
        result[ds] = (best_frac, float(np.mean(profits)), float(np.mean(ontimes)))
        if best_frac < 1.0:
            print(f"  [rl warn] {DATASET_LABELS.get(ds, ds)}: frac1.0 missing, "
                  f"using frac={best_frac}")
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

            if rl_row:
                p, o = rl_row[1], rl_row[2]
                row += f"  {p:>{col_w}.4f} {o:>{col_w}.4f} {p+o:>{col_w}.4f}"
            else:
                row += f"  {'—':>{col_w}} {'—':>{col_w}} {'—':>{col_w}}"

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

        vertical = (ds == "globalstore")
        if vertical:
            fig, axes = plt.subplots(3, 1, figsize=(8, 13))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for i, (ax, (key, metric_label)) in enumerate(zip(axes, METRICS)):
            if rl_row:
                rl_val  = get_val(rl_row, key)
                rl_frac = rl_row[0]
                ax.axhline(rl_val, color=RL_COLOR, linewidth=2.5, linestyle="--",
                           label=f"RL full (frac={int(rl_frac*100)}%): {rl_val:.4f}")

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
            # only label x-axis on the bottom panel for vertical layout
            if not vertical or i == len(METRICS) - 1:
                ax.set_xlabel("Training Fraction", fontsize=9)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_xticks(all_fracs)
            ax.set_xticklabels([frac_label(f) for f in all_fracs], fontsize=6)

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
    ds_colors  = {"dataco": "#2196F3", "globalstore": "#FF5722",
                  "oas": "#4CAF50", "scsp": "#9C27B0"}
    all_fracs  = sorted(
        {0.0} | {r[0] for ds_map in se_data.values() for rows in ds_map.values() for r in rows}
    )

    for model in all_models:
        model_label = MODEL_LABELS.get(model, model)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax, (key, metric_label) in zip(axes, METRICS):
            for ds in DATASETS:
                color      = ds_colors.get(ds, "#555")
                dset_label = DATASET_LABELS.get(ds, ds)

                rl_row = rl_full.get(ds)
                if rl_row:
                    rl_val = get_val(rl_row, key)
                    ax.axhline(rl_val, color=color, linewidth=1.8, linestyle="--", alpha=0.75,
                               label=f"{dset_label} RL-full: {rl_val:.4f}")

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
            ax.set_xticklabels([frac_label(f) for f in all_fracs], fontsize=6)

        fig.suptitle(f"{model_label} — VocabAlign vs RL Full Training (dashed)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        safe = re.sub(r"[^a-z0-9]+", "_", model.lower())
        path = os.path.join(out_dir, f"compare_model_{safe}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def plot_grouped_fracs(zs_data, se_data, rl_full, out_dir):
    """For each dataset: bar chart per fraction, models side by side, RL as hline."""
    all_models = sorted_models(set(zs_data.keys()) | set(se_data.keys()))
    all_fracs  = sorted(
        {0.0} | {r[0] for ds_map in se_data.values() for rows in ds_map.values() for r in rows}
    )
    n_models = len(all_models)
    x        = np.arange(len(all_fracs))
    width    = 0.8 / n_models

    for ds in DATASETS:
        dset_label = DATASET_LABELS.get(ds, ds)
        rl_row     = rl_full.get(ds)
        fig, axes  = plt.subplots(1, 3, figsize=(18, 5))

        for ax, (key, metric_label) in zip(axes, METRICS):
            if rl_row:
                rl_val = get_val(rl_row, key)
                ax.axhline(rl_val, color=RL_COLOR, linewidth=2, linestyle="--", zorder=5,
                           label=f"RL full: {rl_val:.4f}")

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
            ax.set_xticklabels([frac_label(f) for f in all_fracs], fontsize=7)
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
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Sample-eff dirs: {SAMPEFF_DIRS}")
    print(f"RL seed dirs   : {RL_RUN_DIRS}")
    print(f"Output dir     : {OUT_DIR}")

    print(f"\nLoading zero-shot (averaging across runs)...")
    zs_data = load_zeroshot(ZEROSHOT_MODEL_RUN_DIRS)
    print(f"  Models: {sorted_models(zs_data.keys())}")
    for m, ds_map in zs_data.items():
        print(f"    {MODEL_LABELS.get(m, m)}: {sorted(ds_map.keys())}")

    print(f"Loading sample efficiency (averaging across seeds)...")
    se_data = load_sampeff_dirs(SAMPEFF_DIRS)
    print(f"  Models: {sorted_models(se_data.keys())}")
    for m, ds_map in se_data.items():
        fracs = sorted({r[0] for rows in ds_map.values() for r in rows})
        print(f"    {MODEL_LABELS.get(m, m)}: {fracs}")

    print(f"Loading RL (averaging across seeds)...")
    rl_full = load_rl_full(RL_RUN_DIRS)
    print("  RL full-training reference:")
    for ds, row in sorted(rl_full.items()):
        p, o = row[1], row[2]
        print(f"    {DATASET_LABELS.get(ds, ds)}: frac={row[0]:.2f}  "
              f"profit={p:.4f}  on_time={o:.4f}  total={p+o:.4f}")

    print("\n--- Summary Table ---")
    build_table(zs_data, se_data, rl_full, OUT_DIR)

    print("\n--- Generating plots ---")
    plot_per_dataset(zs_data, se_data, rl_full, OUT_DIR)
    plot_per_model(zs_data, se_data, rl_full, OUT_DIR)
    plot_grouped_fracs(zs_data, se_data, rl_full, OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
