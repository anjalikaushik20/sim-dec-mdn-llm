#!/usr/bin/env python3
"""
Generate all sample-efficiency and zero-shot analysis plots and tables.

Outputs:
  SAMPLE_OUT/
    {model}/  sample_efficiency_{model}.png     (Item 1 — per-model profit/on_time/combined)
    {model}/  llm_vs_rl_{model}.png             (Item 3 — LLM vs RL per-dataset)
    sample_efficiency_table.csv / .md            (Item 2)
    llm_vs_rl_table.csv / .md                   (Item 4)
  ZEROSHOT_OUT/
    rl_vs_zeroshot_table.csv / .md              (Item 5)
"""

import os, re, csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def df_to_markdown(df):
    """Render a DataFrame as a GitHub-flavoured markdown table (no tabulate needed)."""
    cols = list(df.columns)
    # compute column widths
    widths = [max(len(c), df[c].astype(str).map(len).max()) for c in cols]
    sep  = "| " + " | ".join("-" * w for w in widths) + " |"
    hdr  = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
    rows = ["| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([hdr, sep] + rows)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

SAMPLE_RUN_LLM = os.path.join(BASE, "output/latest_output/sample_efficiency/all_vocabalign/20260521_220458")
SAMPLE_RUN_RL  = os.path.join(BASE, "output/latest_output/sample_efficiency/rl_baseline/20260520_182054")
ZEROSHOT_RUN   = os.path.join(BASE, "output/latest_output/zero_shot/vocabalign/20260523_030817")
SAMPLE_OUT     = os.path.join(BASE, "output/latest_output/sample_efficiency/all_vocabalign/20260523_171937")
ZEROSHOT_OUT   = ZEROSHOT_RUN

os.makedirs(SAMPLE_OUT, exist_ok=True)

# ── RL full-training baseline (user-provided) ─────────────────────────────────
RL_FULL = {
    "DataCo":      {"profit": 0.5268, "on_time": 0.2464, "combined": 0.7732,
                    "pmp1": 0.361, "pmp2": 0.450, "pmp3": 0.450},
    "GlobalStore": {"profit": 0.3445, "on_time": 0.7582, "combined": 1.1020,
                    "pmp1": 0.199, "pmp2": 0.356, "pmp3": 0.356},
    "OAS":         {"profit": 0.4846, "on_time": 0.0828, "combined": 0.5674,
                    "pmp1": 0.234, "pmp2": 0.373, "pmp3": 0.373},
}

MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "qwen3-0.6B", "qwen3-1.7B", "qwen3-4B"]
MODEL_LABELS = {
    "gpt2":        "GPT-2",
    "gpt2-medium": "GPT-2 Medium",
    "gpt2-large":  "GPT-2 Large",
    "qwen3-0.6B":  "Qwen3-0.6B",
    "qwen3-1.7B":  "Qwen3-1.7B",
    "qwen3-4B":    "Qwen3-4B",
}
DATASETS  = ["DataCo", "GlobalStore", "OAS"]
FRACS     = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
FRAC_PCT  = ["1%",  "5%", "10%", "25%", "50%", "100%"]
FRAC_X    = np.array([1, 5, 10, 25, 50, 100])  # for plotting on a log-like x-axis

# ── Colours ───────────────────────────────────────────────────────────────────
C_PROFIT   = "#1f77b4"
C_ONTIME   = "#ff7f0e"
C_COMBINED = "#2ca02c"

MODEL_COLORS = {
    "gpt2":        "#1f77b4",
    "gpt2-medium": "#ff7f0e",
    "gpt2-large":  "#2ca02c",
    "qwen3-0.6B":  "#d62728",
    "qwen3-1.7B":  "#9467bd",
    "qwen3-4B":    "#8c564b",
}

# ── Parsing ───────────────────────────────────────────────────────────────────
def _parse_log(path, eq_sep=True):
    """
    Parse best_profit, best_on_time, best_pmp_1/2/3 from a log file.
    eq_sep=True  → matches   best_profit=0.52
    eq_sep=False → matches   best_profit 0.52  (RL logs)
    """
    sep  = r"=" if eq_sep else r"[\s=]+"
    pats = {
        "profit":  re.compile(rf"best_profit{sep}([\d.]+)"),
        "on_time": re.compile(rf"best_on_time{sep}([\d.]+)"),
        "pmp1":    re.compile(rf"best_pmp_1{sep}([\d.]+)"),
        "pmp2":    re.compile(rf"best_pmp_2{sep}([\d.]+)"),
        "pmp3":    re.compile(rf"best_pmp_3{sep}([\d.]+)"),
    }
    metrics = {}
    try:
        with open(path) as f:
            for line in f:
                for key, pat in pats.items():
                    if key not in metrics:
                        m = pat.search(line)
                        if m:
                            metrics[key] = float(m.group(1))
    except OSError:
        return None
    if "profit" not in metrics or "on_time" not in metrics:
        return None
    metrics["combined"] = round(metrics["profit"] + metrics["on_time"], 6)
    return metrics


def load_llm_data():
    """model → dataset → frac(float) → metrics dict"""
    data = defaultdict(lambda: defaultdict(dict))
    for model in MODELS:
        mdir = os.path.join(SAMPLE_RUN_LLM, model)
        for frac in FRACS:
            for ds in DATASETS:
                fname = f"{ds.lower()}_frac{frac:.2f}.log"
                m = _parse_log(os.path.join(mdir, fname), eq_sep=True)
                if m:
                    data[model][ds][frac] = m
    return data


def load_rl_sample_data():
    """dataset → frac(float) → metrics dict (RL sample efficiency)"""
    frac_strs = {0.01: "0.01", 0.05: "0.05", 0.10: "0.10",
                 0.25: "0.25", 0.50: "0.50", 1.00: "1.0"}
    data = defaultdict(dict)
    for ds in DATASETS:
        for frac, fs in frac_strs.items():
            fname = f"{ds.lower()}_frac{fs}.log"
            m = _parse_log(os.path.join(SAMPLE_RUN_RL, fname), eq_sep=False)
            if m:
                data[ds][frac] = m
    return data


def load_zeroshot_data():
    """model → dataset → metrics dict"""
    data = defaultdict(dict)
    for model in MODELS:
        for ds in DATASETS:
            fname = f"{ds.lower()}_{model}.log"
            m = _parse_log(os.path.join(ZEROSHOT_RUN, fname), eq_sep=True)
            if m:
                data[model][ds] = m
    return data


# ── Plot helpers ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

METRIC_YLIM = {
    "DataCo":      (0.0, 1.05),
    "GlobalStore": (0.0, 1.30),
    "OAS":         (0.0, 0.70),
}


def _x_from_fracs(fracs):
    return [f * 100 for f in fracs]


def _setup_ax(ax, ds, title_extra=""):
    ax.set_title(f"{ds}{title_extra}", fontweight="bold")
    ax.set_xlabel("Training data (%)")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 130)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v)}%" if v >= 1 else f"{v:.1f}%"))
    ax.set_xticks([1, 5, 10, 25, 50, 100])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ylo, yhi = METRIC_YLIM.get(ds, (0.0, 1.2))
    ax.set_ylim(ylo, yhi)
    ax.grid(True, alpha=0.3, linestyle="--")


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 1 — Per-model: profit / on_time / combined vs % data, per dataset
# ─────────────────────────────────────────────────────────────────────────────
def plot_item1(llm_data):
    print("\n[Item 1] Per-model metric curves with RL full-training baseline…")
    for model in MODELS:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
        fig.suptitle(f"{MODEL_LABELS[model]} — Sample Efficiency (VocabAlign)",
                     fontweight="bold", fontsize=11)

        for ax, ds in zip(axes, DATASETS):
            _setup_ax(ax, ds)
            ax.set_ylabel("Score")

            # LLM curves
            m_data = llm_data[model][ds]
            if m_data:
                fracs_avail = sorted(m_data.keys())
                xs = _x_from_fracs(fracs_avail)
                ax.plot(xs, [m_data[f]["profit"]   for f in fracs_avail],
                        color=C_PROFIT,   marker="o", lw=1.8, ms=4, label="Profit (LLM)")
                ax.plot(xs, [m_data[f]["on_time"]  for f in fracs_avail],
                        color=C_ONTIME,   marker="s", lw=1.8, ms=4, label="On-Time (LLM)")
                ax.plot(xs, [m_data[f]["combined"] for f in fracs_avail],
                        color=C_COMBINED, marker="^", lw=1.8, ms=4, label="Combined (LLM)")

            # RL full-training baselines (dashed)
            rl = RL_FULL[ds]
            ax.axhline(rl["profit"],   color=C_PROFIT,   lw=1.4, ls="--", alpha=0.8,
                       label=f"RL Profit ({rl['profit']:.4f})")
            ax.axhline(rl["on_time"],  color=C_ONTIME,   lw=1.4, ls="--", alpha=0.8,
                       label=f"RL On-Time ({rl['on_time']:.4f})")
            ax.axhline(rl["combined"], color=C_COMBINED, lw=1.4, ls="--", alpha=0.8,
                       label=f"RL Combined ({rl['combined']:.4f})")

            if ax is axes[0]:
                ax.legend(loc="lower right", fontsize=7, ncol=1)

        fig.tight_layout()
        out_dir = os.path.join(SAMPLE_OUT, model)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"sample_efficiency_{model}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {os.path.relpath(out_path, BASE)}")


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 2 — Table: LLM sample efficiency results
# ─────────────────────────────────────────────────────────────────────────────
def table_item2(llm_data):
    print("\n[Item 2] Sample efficiency table…")
    rows = []
    for model in MODELS:
        for ds in DATASETS:
            for frac in FRACS:
                m = llm_data[model][ds].get(frac)
                rows.append({
                    "Model":    MODEL_LABELS[model],
                    "Dataset":  ds,
                    "Frac (%)": f"{int(frac*100)}%",
                    "Profit":   f"{m['profit']:.4f}"   if m else "—",
                    "On-Time":  f"{m['on_time']:.4f}"  if m else "—",
                    "Combined": f"{m['combined']:.4f}" if m else "—",
                    "PMP@0.1":  f"{m['pmp1']:.4f}"     if m and "pmp1" in m else "—",
                    "PMP@0.2":  f"{m['pmp2']:.4f}"     if m and "pmp2" in m else "—",
                    "PMP@0.3":  f"{m['pmp3']:.4f}"     if m and "pmp3" in m else "—",
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(SAMPLE_OUT, "sample_efficiency_table.csv")
    md_path  = os.path.join(SAMPLE_OUT, "sample_efficiency_table.md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write("# Sample Efficiency Results — All VocabAlign Models\n\n")
        f.write(df_to_markdown(df))
        f.write("\n")
    print(f"  saved {os.path.relpath(csv_path, BASE)}")
    print(f"  saved {os.path.relpath(md_path, BASE)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 3 — Per-model: LLM combined vs RL combined, per dataset
# ─────────────────────────────────────────────────────────────────────────────
def plot_item3(llm_data, rl_data):
    print("\n[Item 3] LLM vs RL comparison curves…")
    for model in MODELS:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
        fig.suptitle(
            f"{MODEL_LABELS[model]} — LLM (VocabAlign) vs RL: Combined Score vs Training Data",
            fontweight="bold", fontsize=11)

        for ax, ds in zip(axes, DATASETS):
            _setup_ax(ax, ds)
            ax.set_ylabel("Combined Score (Profit + On-Time)")

            # LLM line
            m_data = llm_data[model][ds]
            if m_data:
                fracs_avail = sorted(m_data.keys())
                xs = _x_from_fracs(fracs_avail)
                ax.plot(xs, [m_data[f]["combined"] for f in fracs_avail],
                        color=MODEL_COLORS[model], marker="o", lw=2.0, ms=5,
                        label=f"LLM ({MODEL_LABELS[model]})")

            # RL sample efficiency line
            rl_ds = rl_data.get(ds, {})
            if rl_ds:
                rl_fracs = sorted(rl_ds.keys())
                xs_rl = _x_from_fracs(rl_fracs)
                ax.plot(xs_rl, [rl_ds[f]["combined"] for f in rl_fracs],
                        color="#e31a1c", marker="D", lw=1.8, ms=4, ls="--",
                        label="RL (sample eff.)")

            # RL full-training horizontal baseline
            rl_full_val = RL_FULL[ds]["combined"]
            ax.axhline(rl_full_val, color="black", lw=1.4, ls=":",
                       label=f"RL Full ({rl_full_val:.4f})")

            ax.legend(loc="lower right", fontsize=7)

        fig.tight_layout()
        out_dir = os.path.join(SAMPLE_OUT, model)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"llm_vs_rl_{model}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {os.path.relpath(out_path, BASE)}")


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 4 — Table: LLM vs RL per frac
# ─────────────────────────────────────────────────────────────────────────────
def table_item4(llm_data, rl_data):
    print("\n[Item 4] LLM vs RL comparison table…")
    rows = []
    for model in MODELS:
        for ds in DATASETS:
            for frac in FRACS:
                llm_m = llm_data[model][ds].get(frac)
                rl_m  = rl_data.get(ds, {}).get(frac)
                rows.append({
                    "Model":          MODEL_LABELS[model],
                    "Dataset":        ds,
                    "Frac (%)":       f"{int(frac*100)}%",
                    "LLM Profit":     f"{llm_m['profit']:.4f}"   if llm_m else "—",
                    "LLM On-Time":    f"{llm_m['on_time']:.4f}"  if llm_m else "—",
                    "LLM Combined":   f"{llm_m['combined']:.4f}" if llm_m else "—",
                    "RL Profit":      f"{rl_m['profit']:.4f}"    if rl_m  else "—",
                    "RL On-Time":     f"{rl_m['on_time']:.4f}"   if rl_m  else "—",
                    "RL Combined":    f"{rl_m['combined']:.4f}"  if rl_m  else "—",
                    "Δ Combined":     (f"{llm_m['combined']-rl_m['combined']:+.4f}"
                                       if llm_m and rl_m else "—"),
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(SAMPLE_OUT, "llm_vs_rl_table.csv")
    md_path  = os.path.join(SAMPLE_OUT, "llm_vs_rl_table.md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write("# LLM vs RL Sample Efficiency Comparison\n\n")
        f.write(df_to_markdown(df))
        f.write("\n")
    print(f"  saved {os.path.relpath(csv_path, BASE)}")
    print(f"  saved {os.path.relpath(md_path, BASE)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 5 — Table: RL full training vs zero-shot, per model and dataset
# ─────────────────────────────────────────────────────────────────────────────
def table_item5(zs_data):
    print("\n[Item 5] RL full training vs zero-shot table…")
    rows = []
    for ds in DATASETS:
        rl = RL_FULL[ds]
        for model in MODELS:
            zs = zs_data[model].get(ds)
            rows.append({
                "Dataset":         ds,
                "Model":           MODEL_LABELS[model],
                "ZS Profit":       f"{zs['profit']:.4f}"   if zs else "—",
                "ZS On-Time":      f"{zs['on_time']:.4f}"  if zs else "—",
                "ZS Combined":     f"{zs['combined']:.4f}" if zs else "—",
                "ZS PMP@0.1":      f"{zs['pmp1']:.4f}"     if zs and "pmp1" in zs else "—",
                "ZS PMP@0.2":      f"{zs['pmp2']:.4f}"     if zs and "pmp2" in zs else "—",
                "ZS PMP@0.3":      f"{zs['pmp3']:.4f}"     if zs and "pmp3" in zs else "—",
                "RL Profit":       f"{rl['profit']:.4f}",
                "RL On-Time":      f"{rl['on_time']:.4f}",
                "RL Combined":     f"{rl['combined']:.4f}",
                "RL PMP@0.1":      f"{rl['pmp1']:.4f}",
                "RL PMP@0.2":      f"{rl['pmp2']:.4f}",
                "RL PMP@0.3":      f"{rl['pmp3']:.4f}",
                "Δ Combined":      (f"{zs['combined'] - rl['combined']:+.4f}"
                                    if zs else "—"),
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(ZEROSHOT_OUT, "rl_vs_zeroshot_table.csv")
    md_path  = os.path.join(ZEROSHOT_OUT, "rl_vs_zeroshot_table.md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write("# RL Full Training vs Zero-Shot (VocabAlign)\n\n")
        f.write(df_to_markdown(df))
        f.write("\n")
    print(f"  saved {os.path.relpath(csv_path, BASE)}")
    print(f"  saved {os.path.relpath(md_path, BASE)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# BONUS — Regenerate RL baseline plots without run-ID in title
# ─────────────────────────────────────────────────────────────────────────────
DS_COLORS = {"DataCo": "#1f77b4", "GlobalStore": "#ff7f0e", "OAS": "#2ca02c"}

def regen_rl_baseline_plots(rl_data):
    """Overwrite the existing RL baseline PNGs with clean titles (no timestamp)."""
    print("\n[RL] Regenerating RL baseline plots without run-ID…")
    rl_out = SAMPLE_RUN_RL  # write back into the source directory

    # ── 1. Combined overview (all datasets, 4 metrics) ───────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    metrics = [
        ("profit",   "Profit"),
        ("on_time",  "On-Time"),
        ("combined", "Profit + On-Time"),
    ]
    for ax, (key, label) in zip(axes[:3], metrics):
        for ds in DATASETS:
            ds_data = rl_data.get(ds, {})
            if not ds_data:
                continue
            fracs_s = sorted(ds_data.keys())
            xs = _x_from_fracs(fracs_s)
            ax.plot(xs, [ds_data[f][key] for f in fracs_s],
                    marker="o", color=DS_COLORS[ds], lw=2, ms=6, label=ds)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Training data (%)")
        ax.set_xscale("log")
        ax.set_xlim(0.8, 130)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{int(v)}%" if v >= 1 else f"{v:.1f}%"))
        ax.set_xticks([1, 5, 10, 25, 50, 100])
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, ls="--")
    axes[3].axis("off")  # unused 4th panel
    fig.suptitle("Sample Efficiency — RL Baseline", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(rl_out, "sample_efficiency_rl.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {os.path.relpath(p, BASE)}")

    # ── 2. Per-dataset plots ──────────────────────────────────────────────────
    for ds in DATASETS:
        ds_data = rl_data.get(ds, {})
        if not ds_data:
            continue
        fracs_s = sorted(ds_data.keys())
        xs = _x_from_fracs(fracs_s)
        color = DS_COLORS[ds]

        fig, axes2 = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (key, label) in zip(axes2, metrics):
            ax.plot(xs, [ds_data[f][key] for f in fracs_s],
                    marker="o", color=color, lw=2, ms=6)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Training data (%)")
            ax.set_xscale("log")
            ax.set_xlim(0.8, 130)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v, _: f"{int(v)}%" if v >= 1 else f"{v:.1f}%"))
            ax.set_xticks([1, 5, 10, 25, 50, 100])
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.grid(True, alpha=0.3, ls="--")
        fig.suptitle(f"{ds} — RL Baseline", fontsize=12, fontweight="bold")
        fig.tight_layout()
        p = os.path.join(rl_out, f"sample_efficiency_rl_{ds.lower()}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {os.path.relpath(p, BASE)}")


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 6 — Zero-shot (all models) vs RL full training, per dataset
# ─────────────────────────────────────────────────────────────────────────────
def plot_zeroshot_vs_rl_full(zs_data):
    """
    Grouped bar chart per dataset showing zero-shot profit / on-time / combined
    for every model, with RL full-training dashed lines overlaid.
    Saved to both SAMPLE_OUT and ZEROSHOT_OUT.
    """
    print("\n[Item 6] Zero-shot all models vs RL full training…")

    short_labels = [MODEL_LABELS[m].replace(" ", "\n") for m in MODELS]
    n_models  = len(MODELS)
    bar_w     = 0.22
    x         = np.arange(n_models)
    offsets   = np.array([-bar_w, 0, bar_w])   # profit | on_time | combined

    for ds in DATASETS:
        rl = RL_FULL[ds]
        ymax = max(rl["combined"] * 1.25,
                   max((zs_data[m][ds]["combined"]
                        for m in MODELS if ds in zs_data[m]), default=0) * 1.15)

        fig, ax = plt.subplots(figsize=(10, 5))

        for i, model in enumerate(MODELS):
            zs = zs_data[model].get(ds)
            if zs is None:
                continue
            ax.bar(x[i] + offsets[0], zs["profit"],   width=bar_w,
                   color=C_PROFIT,   alpha=0.85, zorder=3)
            ax.bar(x[i] + offsets[1], zs["on_time"],  width=bar_w,
                   color=C_ONTIME,   alpha=0.85, zorder=3)
            ax.bar(x[i] + offsets[2], zs["combined"], width=bar_w,
                   color=C_COMBINED, alpha=0.85, zorder=3)

        # RL full-training horizontal dashed lines + legend proxies
        ax.axhline(rl["profit"],   color=C_PROFIT,   lw=1.6, ls="--", zorder=4,
                   label=f"RL Profit ({rl['profit']:.4f})")
        ax.axhline(rl["on_time"],  color=C_ONTIME,   lw=1.6, ls="--", zorder=4,
                   label=f"RL On-Time ({rl['on_time']:.4f})")
        ax.axhline(rl["combined"], color=C_COMBINED, lw=1.6, ls="--", zorder=4,
                   label=f"RL Combined ({rl['combined']:.4f})")

        # Dummy patches for bar legend entries
        import matplotlib.patches as mpatches
        bar_legend = [
            mpatches.Patch(color=C_PROFIT,   alpha=0.85, label="Zero-Shot Profit"),
            mpatches.Patch(color=C_ONTIME,   alpha=0.85, label="Zero-Shot On-Time"),
            mpatches.Patch(color=C_COMBINED, alpha=0.85, label="Zero-Shot Combined"),
        ]
        handles, labels_ = ax.get_legend_handles_labels()
        ax.legend(handles=bar_legend + handles, fontsize=8,
                  loc="upper right", ncol=2, framealpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=8.5)
        ax.set_ylabel("Score")
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.5, n_models - 0.5)
        ax.grid(True, axis="y", alpha=0.3, ls="--", zorder=0)
        ax.set_title(f"{ds} — Zero-Shot (All Models) vs RL Full Training",
                     fontweight="bold", fontsize=11)
        fig.tight_layout()

        fname = f"zero_shot_vs_rl_full_{ds.lower()}.png"
        for out_dir in [SAMPLE_OUT, ZEROSHOT_OUT]:
            p = os.path.join(out_dir, fname)
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"  saved {os.path.relpath(p, BASE)}")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data…")
    llm_data = load_llm_data()
    rl_data  = load_rl_sample_data()
    zs_data  = load_zeroshot_data()

    # Coverage report
    print("\nLLM data coverage:")
    for model in MODELS:
        for ds in DATASETS:
            count = len(llm_data[model][ds])
            flag = "" if count == 6 else f"  ⚠ only {count}/6"
            print(f"  {MODEL_LABELS[model]:16s} {ds:12s}: {count}/6{flag}")

    print("\nRL sample efficiency coverage:")
    for ds in DATASETS:
        count = len(rl_data[ds])
        print(f"  {ds:12s}: {count}/6")

    print("\nZero-shot coverage:")
    for model in MODELS:
        for ds in DATASETS:
            ok = ds in zs_data[model]
            print(f"  {MODEL_LABELS[model]:16s} {ds:12s}: {'✓' if ok else '✗'}")

    plot_item1(llm_data)
    table_item2(llm_data)
    plot_item3(llm_data, rl_data)
    table_item4(llm_data, rl_data)
    table_item5(zs_data)
    plot_zeroshot_vs_rl_full(zs_data)
    regen_rl_baseline_plots(rl_data)

    print("\nDone. All outputs written.")
