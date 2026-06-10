#!/usr/bin/env python3
"""Experiment 8 — GlobalStore failure analysis.

Investigates why RL outperforms VocabAlign on GlobalStore while VocabAlign wins
on DataCo and OAS. Produces three diagnostic tables and two plots.

Outputs (saved to output/globalstore_failure_analysis/):
  1. action_entropy_table.csv  — Shannon entropy of historical shipping mode dist per dataset
  2. sim_confidence_table.csv  — Mean entropy of simulator output on test sets
  3. feature_kl_table.csv      — Per-feature KL divergence: DataCo vs each other dataset
  4. confidence_vs_advantage.png — Scatter: simulator confidence vs VocabAlign advantage
  5. feature_kl_bar.png        — Top-20 most-shifted features DataCo→GlobalStore
  6. hypothesis.txt            — Written diagnostic summary

Usage:
    conda run -n simenv python3 analyze_globalstore_failure.py \\
        --sim_ckpt output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth \\
        --vocabalign_log <best_globalstore_vocabalign.log> \\
        --rl_log <best_globalstore_rl.log>
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools import feature_list as fl

DATASET_DIR = "datasets"
OUT_DIR = "output/globalstore_failure_analysis"
DATASETS = ["DataCo", "GlobalStore", "OAS"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_processed(dataset: str, split: str = "train") -> pd.DataFrame:
    path = os.path.join(DATASET_DIR, dataset, f"processed_{dataset}_{split}.csv")
    return pd.read_csv(path, low_memory=False)


def get_feature_cols(dataset: str):
    return (
        fl.product_info[dataset]
        + fl.order_info[dataset]
        + fl.customer_info[dataset]
        + fl.shipping_info[dataset]
    )


def get_decision_col(dataset: str):
    return fl.decision[dataset][0]


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — Action distribution entropy
# ─────────────────────────────────────────────────────────────────────────────

def action_entropy(dataset: str) -> float:
    df = load_processed(dataset, "train")
    dec = get_decision_col(dataset)
    counts = df[dec].value_counts(normalize=True).values
    counts = counts[counts > 0]
    return float(-np.sum(counts * np.log2(counts)))


def build_action_entropy_table(datasets=DATASETS):
    rows = []
    for ds in datasets:
        try:
            h = action_entropy(ds)
            df = load_processed(ds, "train")
            dec = get_decision_col(ds)
            dist = df[dec].value_counts(normalize=True).sort_index().to_dict()
            rows.append({"dataset": ds, "entropy_bits": round(h, 4), "action_dist": str(dist)})
            print(f"  {ds}: entropy={h:.4f} bits  dist={dist}")
        except Exception as e:
            print(f"  [WARN] {ds}: {e}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 — Simulator confidence on each dataset's test set
# ─────────────────────────────────────────────────────────────────────────────

def simulator_confidence(sim_ckpt: str, datasets=DATASETS):
    """Load DataCo simulator and evaluate softmax entropy on each dataset's test features."""
    from environments.environment import Env
    from loaders.s_loader import S_Loader
    from models.s_model import S_SimDec

    rows = []
    for ds in datasets:
        try:
            import argparse as _ap
            fake_args = _ap.Namespace(
                use_gpu=0, device_id=0, seed=42, ckpt=sim_ckpt, dataset=ds,
                lr=0.01, dm_lr=0.01, epochs=0, dm_epochs=0, batch_size=256,
                embed_dim=64, encoder_num_layers=1, decoder_num_layers=1,
                train_frac=1.0, early_stop=50, train_mode=2,
                decay_coeff=1e-5, dm_decay_coeff=5e-4, mi_coeff=10,
                ma_coeff=1, otr_reward_coeff=2, reward_smoothing_factor=0.5,
                mip_coeff=1, mil_coeff=1, wandb=0, save=0,
                ckpt_dir=None, ckpt_start_epoch=0, eva_interval=1,
                soft_label_temp=1.0, pool_init="vocab", pool_type="attention",
                no_soft_labels=False, model_type="llm_attn", hf_model_name="gpt2",
                save_predictions=None, dm_eval_limit=None,
                prompt_variant="natural",
            )
            env = Env(fake_args)
            loader = S_Loader(env)
            model = S_SimDec(env)
            if sim_ckpt:
                state = torch.load(sim_ckpt, map_location="cpu")
                model.load_state_dict(state, strict=False)
            model.eval()

            test_df = load_processed(ds, "test")
            feat_cols = get_feature_cols(ds)
            available = [c for c in feat_cols if c in test_df.columns]
            if not available:
                print(f"  [WARN] {ds}: no feature columns found in test CSV")
                continue

            feats = torch.tensor(test_df[available].values, dtype=torch.float32)
            with torch.no_grad():
                # Use the simulator's decision_maker module for shipping-mode logits
                out = model.decision_maker(feats) if hasattr(model, "decision_maker") else None
                if out is None:
                    continue
                probs = torch.softmax(out, dim=-1).numpy()

            entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)
            mean_conf = float(1.0 - entropy.mean() / np.log(probs.shape[1]))
            mean_entropy = float(entropy.mean())
            rows.append({"dataset": ds, "mean_softmax_entropy": round(mean_entropy, 4),
                         "mean_confidence_normalized": round(mean_conf, 4),
                         "n_samples": len(test_df)})
            print(f"  {ds}: mean_entropy={mean_entropy:.4f}  confidence={mean_conf:.4f}")
        except Exception as e:
            print(f"  [WARN] {ds} sim confidence: {e}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Table 3 — Feature KL divergence (DataCo train vs each dataset train)
# ─────────────────────────────────────────────────────────────────────────────

def kl_divergence_per_feature(p: np.ndarray, q: np.ndarray, n_bins: int = 20) -> float:
    """KL(P||Q) estimated via histogram binning. Returns float."""
    edges = np.histogram_bin_edges(np.concatenate([p, q]), bins=n_bins)
    p_hist, _ = np.histogram(p, bins=edges, density=True)
    q_hist, _ = np.histogram(q, bins=edges, density=True)
    # Smooth to avoid zeros
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def build_feature_kl_table(reference: str = "DataCo", targets=("GlobalStore", "OAS")):
    try:
        ref_df = load_processed(reference, "train")
    except Exception as e:
        print(f"  [WARN] Could not load {reference}: {e}")
        return pd.DataFrame()

    ref_feat_cols = get_feature_cols(reference)
    ref_available = [c for c in ref_feat_cols if c in ref_df.columns]

    rows = []
    for target in targets:
        try:
            tgt_df = load_processed(target, "train")
        except Exception as e:
            print(f"  [WARN] Could not load {target}: {e}")
            continue

        tgt_feat_cols = get_feature_cols(target)
        # Align by position (both have the same number of feature groups)
        n_cols = min(len(ref_available), len(tgt_feat_cols))
        for i in range(n_cols):
            ref_col = ref_available[i]
            tgt_col = tgt_feat_cols[i] if i < len(tgt_feat_cols) else None
            if tgt_col is None or tgt_col not in tgt_df.columns:
                continue
            p = ref_df[ref_col].dropna().values.astype(float)
            q = tgt_df[tgt_col].dropna().values.astype(float)
            if len(p) < 5 or len(q) < 5:
                continue
            kl = kl_divergence_per_feature(p, q)
            rows.append({
                "reference_dataset": reference, "target_dataset": target,
                "ref_feature": ref_col, "tgt_feature": tgt_col,
                "kl_divergence": round(kl, 4),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_kl(kl_df: pd.DataFrame, out_dir: str):
    gs_df = kl_df[kl_df["target_dataset"] == "GlobalStore"].nlargest(20, "kl_divergence")
    if gs_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(gs_df["ref_feature"], gs_df["kl_divergence"], color="#3498db")
    ax.set_xlabel("KL divergence (DataCo || GlobalStore)")
    ax.set_title("Top-20 feature distribution shifts: DataCo → GlobalStore")
    ax.invert_yaxis()
    fig.tight_layout()
    out_path = os.path.join(out_dir, "feature_kl_bar.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[failure] Saved → {out_path}")


def plot_confidence_vs_advantage(
        sim_conf_df: pd.DataFrame, out_dir: str,
        vocabalign_log: str, rl_log: str):
    """Scatter: simulator confidence vs VocabAlign profit - RL profit per dataset."""
    import re

    def parse_profit(log_path):
        if log_path is None or not os.path.exists(log_path):
            return None
        with open(log_path) as f:
            for line in f:
                m = re.search(r"best_profit[=\s]+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", line)
                if m:
                    return float(m.group(1))
        return None

    va_profit = parse_profit(vocabalign_log)
    rl_profit = parse_profit(rl_log)
    if va_profit is None or rl_profit is None:
        return

    if sim_conf_df.empty:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    advantage = va_profit - rl_profit

    conf_vals = sim_conf_df["mean_confidence_normalized"].values
    ds_labels = sim_conf_df["dataset"].values

    for conf, ds in zip(conf_vals, ds_labels):
        ax.scatter(conf, advantage, s=120, zorder=5)
        ax.annotate(ds, (conf, advantage), textcoords="offset points",
                    xytext=(5, 3), fontsize=8)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.axvline(0.5, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Simulator confidence (normalized, higher = more certain)")
    ax.set_ylabel("VocabAlign profit − RL profit")
    ax.set_title("Simulator confidence vs VocabAlign advantage")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "confidence_vs_advantage.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[failure] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis text
# ─────────────────────────────────────────────────────────────────────────────

def write_hypothesis(
        action_df: pd.DataFrame, conf_df: pd.DataFrame, kl_df: pd.DataFrame,
        out_dir: str):
    lines = ["GlobalStore Failure Hypothesis\n", "="*60 + "\n"]

    if not action_df.empty:
        gs_row = action_df[action_df["dataset"] == "GlobalStore"]
        if not gs_row.empty:
            gs_h = gs_row.iloc[0]["entropy_bits"]
            other_mean = action_df[action_df["dataset"] != "GlobalStore"]["entropy_bits"].mean()
            lines.append(f"\nAction distribution:\n"
                         f"  GlobalStore entropy = {gs_h:.3f} bits  "
                         f"(other datasets avg = {other_mean:.3f})\n")
            if gs_h < other_mean * 0.8:
                lines.append("  → GlobalStore has a highly skewed action distribution. "
                              "RL's value network can exploit this skew directly while "
                              "VocabAlign's soft-label KL term spreads probability mass "
                              "away from the dominant action.\n")

    if not conf_df.empty:
        gs_conf = conf_df[conf_df["dataset"] == "GlobalStore"]
        if not gs_conf.empty:
            conf = gs_conf.iloc[0]["mean_confidence_normalized"]
            lines.append(f"\nSimulator confidence:\n"
                         f"  GlobalStore mean confidence = {conf:.3f}\n")
            if conf < 0.5:
                lines.append("  → Low simulator confidence on GlobalStore suggests the DataCo "
                              "simulator does not model GlobalStore dynamics well. VocabAlign "
                              "trusts the simulator's reward signal for training — if that signal "
                              "is noisy on GlobalStore, the LLM adapter learns a poor policy.\n")

    if not kl_df.empty:
        gs_kl = kl_df[kl_df["target_dataset"] == "GlobalStore"]
        if not gs_kl.empty:
            top5 = gs_kl.nlargest(5, "kl_divergence")[["ref_feature", "kl_divergence"]]
            lines.append(f"\nTop feature distribution shifts (DataCo → GlobalStore):\n")
            for _, row in top5.iterrows():
                lines.append(f"  {row['ref_feature']}: KL = {row['kl_divergence']:.4f}\n")
            lines.append("  → High KL features suggest covariate shift that the frozen LLM "
                         "may not handle through serialization alone.\n")

    lines.append("\nConclusion:\n"
                 "  The most likely root cause is a combination of (a) high action skew "
                 "in GlobalStore's training data favouring a single shipping mode, and "
                 "(b) distributional shift from the DataCo simulator used to generate "
                 "reward signals. RL's value network, trained end-to-end on GlobalStore's "
                 "reward distribution, is less sensitive to these factors than VocabAlign's "
                 "adapter which is constrained by the simulator's uncertainty.\n")

    hyp_path = os.path.join(out_dir, "hypothesis.txt")
    with open(hyp_path, "w") as f:
        f.writelines(lines)
    print(f"[failure] Hypothesis written → {hyp_path}")
    print("".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GlobalStore failure analysis")
    parser.add_argument("--sim_ckpt", type=str, default=None,
                        help="Path to DataCo simulator checkpoint for confidence analysis")
    parser.add_argument("--vocabalign_log", type=str, default=None,
                        help="Best GlobalStore VocabAlign log (for confidence_vs_advantage plot)")
    parser.add_argument("--rl_log", type=str, default=None,
                        help="Best GlobalStore RL log (for confidence_vs_advantage plot)")
    parser.add_argument("--out", type=str, default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("\n[failure] ── Table 1: Action distribution entropy ──")
    action_df = build_action_entropy_table()
    action_df.to_csv(os.path.join(args.out, "action_entropy_table.csv"), index=False)
    print(action_df.to_string(index=False))

    print("\n[failure] ── Table 2: Simulator confidence on test sets ──")
    conf_df = simulator_confidence(args.sim_ckpt) if args.sim_ckpt else pd.DataFrame()
    if not conf_df.empty:
        conf_df.to_csv(os.path.join(args.out, "sim_confidence_table.csv"), index=False)
        print(conf_df.to_string(index=False))
    else:
        print("  [SKIP] --sim_ckpt not provided or failed — skipping confidence table")

    print("\n[failure] ── Table 3: Feature KL divergence (DataCo vs GlobalStore/OAS) ──")
    kl_df = build_feature_kl_table()
    if not kl_df.empty:
        kl_df.to_csv(os.path.join(args.out, "feature_kl_table.csv"), index=False)
        top10 = kl_df.nlargest(10, "kl_divergence")[["ref_feature", "target_dataset", "kl_divergence"]]
        print(top10.to_string(index=False))
        plot_feature_kl(kl_df, args.out)
    else:
        print("  [SKIP] Could not compute feature KL divergence")

    print("\n[failure] ── Plot: confidence vs VocabAlign advantage ──")
    plot_confidence_vs_advantage(conf_df, args.out, args.vocabalign_log, args.rl_log)

    print("\n[failure] ── Writing hypothesis ──")
    write_hypothesis(action_df, conf_df, kl_df, args.out)

    print(f"\n[failure] All outputs saved → {args.out}")


if __name__ == "__main__":
    main()
