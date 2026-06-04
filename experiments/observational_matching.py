#!/usr/bin/env python3
"""Observational matching for out-of-simulator validation (Experiment 1).

Addresses the circular evaluation critique: finds test orders where VocabAlign
disagrees with the historically executed action, matches each such order to
training orders that actually used VocabAlign's preferred action (nearest-neighbor
on features), then estimates the Average Treatment Effect (ATE) on realized outcomes.

Realized outcomes come from processed_*_test.csv and processed_*_train.csv which
contain actual columns: Late_delivery_risk, Days for shipping (real), on_time — these
are ground-truth labels never used in the simulator-based training or evaluation.

Workflow:
    # Step 1 — generate per-sample predictions from a trained VocabAlign model:
    python3 main/cb_main_llm.py --dataset DataCo --train_mode 2 --dm_epochs 0 \\
        --ckpt <sim.pth> --value_network_ckpt <adapter.pth> \\
        --save_predictions output/exp1/dataco_predictions.csv

    # Step 2 — run observational matching:
    conda run -n simenv python3 experiments/observational_matching.py \\
        --predictions output/exp1/dataco_predictions.csv \\
        --dataset DataCo
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools import feature_list

DATASET_DIR = "datasets"
ACTION_NAMES = ["Standard Class", "Second Class", "First Class", "Same Day"]


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_splits(dataset: str):
    """Return train and test DataFrames with all CSV columns intact."""
    base = os.path.join(DATASET_DIR, dataset)
    train = pd.read_csv(os.path.join(base, f"processed_{dataset}_train.csv"), low_memory=False)
    test  = pd.read_csv(os.path.join(base, f"processed_{dataset}_test.csv"),  low_memory=False)
    return train, test


def get_col_names(dataset: str):
    """Return (feature_cols, decision_col, on_time_col, days_col) by name from feature_list."""
    feature_cols = (
        feature_list.product_info[dataset]
        + feature_list.order_info[dataset]
        + feature_list.customer_info[dataset]
        + feature_list.shipping_info[dataset]
    )
    decision_col = feature_list.decision[dataset][0]
    lbl_names    = feature_list.label[dataset]
    on_time_col  = next((n for n in lbl_names if "on_time" in n.lower()), None)
    days_col     = next((n for n in lbl_names if "day" in n.lower() and "ship" in n.lower()), None)
    return feature_cols, decision_col, on_time_col, days_col


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(arr, n_boot: int = 2000, alpha: float = 0.05):
    rng = np.random.default_rng(42)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return float(np.mean(arr)), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def standardized_mean_diff(a, b):
    pooled_std = np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2 + 1e-9)
    return (np.mean(a) - np.mean(b)) / pooled_std


# ─────────────────────────────────────────────────────────────────────────────
# Core matching
# ─────────────────────────────────────────────────────────────────────────────

def run_matching(predictions_csv: str, dataset: str, k: int = 5):
    pred_df = pd.read_csv(predictions_csv)
    train_df, test_df = load_splits(dataset)
    feature_cols, decision_col, on_time_col, days_col = get_col_names(dataset)

    if on_time_col is None and days_col is None:
        print(f"[WARN] Could not identify outcome columns in label list: {feature_list.label[dataset]}")

    # All column selection by name — robust to CSV column ordering
    train_feats = train_df[feature_cols].values.astype(np.float32)
    test_feats  = test_df[feature_cols].values.astype(np.float32)
    train_acts  = train_df[decision_col].values.astype(int)

    train_on_time = train_df[on_time_col].values.astype(np.float32) if on_time_col else None
    test_on_time  = test_df[on_time_col].values.astype(np.float32)  if on_time_col else None
    train_days    = train_df[days_col].values.astype(np.float32)    if days_col    else None
    test_days     = test_df[days_col].values.astype(np.float32)     if days_col    else None

    # Align predictions with test set
    if len(pred_df) != len(test_df):
        raise ValueError(
            f"Prediction rows ({len(pred_df)}) != test rows ({len(test_df)}). "
            "Re-generate predictions with --save_predictions."
        )
    pred_actions = pred_df["pred_action"].values.astype(int)
    true_actions = pred_df["true_action"].values.astype(int)

    # ── Disagreement set ──────────────────────────────────────────────────────
    disagree_mask = pred_actions != true_actions
    n_disagree = int(disagree_mask.sum())
    n_total = len(test_df)
    print(f"\n{'='*65}")
    print(f"Disagreement set: {n_disagree}/{n_total} test orders "
          f"({100*n_disagree/n_total:.1f}%) where VocabAlign diverges from history")
    print("VocabAlign preferred actions in disagreement set:")
    unique_acts = np.unique(pred_actions[disagree_mask])
    for a in unique_acts:
        c = int((pred_actions[disagree_mask] == a).sum())
        print(f"  {ACTION_NAMES[a]} (action {a}): {c} ({100*c/n_disagree:.1f}%)")

    if n_disagree == 0:
        print("No disagreement found — policy matches historical actions exactly.")
        return

    # ── Per-action nearest-neighbor matching ──────────────────────────────────
    results = {}
    treat_feats_all = []
    ctrl_feats_all  = []
    n_matched = 0

    for act in unique_acts:
        act_mask   = disagree_mask & (pred_actions == act)
        ctrl_feats = test_feats[act_mask]

        treat_mask  = train_acts == act
        treat_feats = train_feats[treat_mask]

        if treat_feats.shape[0] < k:
            print(f"\n[SKIP] Action {act}: only {treat_feats.shape[0]} training examples — "
                  f"not enough for k={k} matching")
            continue

        nn_model = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        nn_model.fit(treat_feats)
        _, idxs = nn_model.kneighbors(ctrl_feats)  # [n_ctrl, k]

        n_matched += ctrl_feats.shape[0]
        treat_feats_all.append(treat_feats[idxs.flatten()])
        ctrl_feats_all.append(ctrl_feats)

        entry = {}
        if on_time_col is not None:
            matched_ot = train_on_time[treat_mask][idxs.flatten()].reshape(-1, k).mean(axis=1)
            entry["treat_on_time"] = matched_ot
            entry["ctrl_on_time"]  = test_on_time[act_mask]
        if days_col is not None:
            matched_d = train_days[treat_mask][idxs.flatten()].reshape(-1, k).mean(axis=1)
            entry["treat_days"] = matched_d
            entry["ctrl_days"]  = test_days[act_mask]
        results[act] = entry

    if not results:
        print("\nNo actions could be matched — insufficient training coverage.")
        return

    print(f"\nMatched {n_matched}/{n_disagree} disagreement orders "
          f"({100*n_matched/n_disagree:.1f}%)")

    # ── ATE table ─────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("AVERAGE TREATMENT EFFECT (treatment=VocabAlign action, control=historical)")
    print(f"{'='*65}")

    for treat_key, ctrl_key, higher_is_better, label in [
        ("treat_on_time", "ctrl_on_time", True,  f"on_time_rate  [{on_time_col}]"),
        ("treat_days",    "ctrl_days",    False, f"days_shipping [{days_col}]"),
    ]:
        all_treat = np.concatenate([v[treat_key] for v in results.values() if treat_key in v])
        all_ctrl  = np.concatenate([v[ctrl_key]  for v in results.values() if ctrl_key  in v])
        if len(all_treat) == 0:
            continue
        ate_arr = all_treat - all_ctrl
        mu, lo, hi = bootstrap_ci(ate_arr)
        direction = "better" if (higher_is_better and mu > 0) or (not higher_is_better and mu < 0) else "worse/neutral"
        print(f"\n{label}:")
        print(f"  Treatment (VocabAlign action historically executed): {np.mean(all_treat):.4f}")
        print(f"  Control   (historical default action):               {np.mean(all_ctrl):.4f}")
        print(f"  ATE = {mu:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   → {direction}")

        for act, entry in results.items():
            if treat_key not in entry:
                continue
            ate_a = entry[treat_key] - entry[ctrl_key]
            mu_a, lo_a, hi_a = bootstrap_ci(ate_a)
            print(f"    {ACTION_NAMES[act]:<16s} n={len(ate_a):5d}  ATE={mu_a:+.4f}  [{lo_a:+.4f}, {hi_a:+.4f}]")

    # ── Covariate balance ─────────────────────────────────────────────────────
    print(f"\n{'-'*65}")
    print("COVARIATE BALANCE (matched treatment vs control, first 8 features)")
    print(f"  |SMD| < 0.1 is considered balanced")
    print(f"{'-'*65}")
    all_treat_f = np.concatenate(treat_feats_all, axis=0)
    all_ctrl_f  = np.concatenate(ctrl_feats_all,  axis=0)
    for fi, name in enumerate(feature_cols[:8]):
        d = standardized_mean_diff(all_treat_f[:, fi], all_ctrl_f[:, fi])
        status = "OK  " if abs(d) < 0.1 else "WARN"
        print(f"  [{status}] {name:<40s}  SMD={d:+.3f}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Observational matching — out-of-simulator validation for VocabAlign"
    )
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to per-sample predictions CSV (from --save_predictions in cb_main_llm.py)"
    )
    parser.add_argument(
        "--dataset", type=str, default="DataCo",
        choices=["DataCo", "GlobalStore", "OAS", "LSCRW"],
        help="Dataset used for evaluation"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of nearest neighbors for matching (default: 5)"
    )
    args = parser.parse_args()
    run_matching(args.predictions, args.dataset, k=args.k)
