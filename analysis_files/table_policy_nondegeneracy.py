#!/usr/bin/env python3
"""
Policy non-degeneracy analysis.

For each (model, dataset) reads the saved prediction CSV and reports:
  1. Predicted action distribution  P(action=k)
  2. Ground-truth action distribution  Q(action=k)
  3. Comparison metrics:
       - TVD   (Total Variation Distance) = 0.5 * sum|P_k - Q_k|  ∈ [0, 1]
       - KL    KL(pred || gt)
       - H_pred  Shannon entropy of predicted distribution (bits)
       - H_gt    Shannon entropy of ground-truth distribution (bits)
       - H_ratio H_pred / H_gt  (<1 = more concentrated than historical)

A degenerate policy always picks the same action → H_pred ≈ 0, TVD ≈ max.

Outputs:
  policy_nondegeneracy_summary.csv   — one row per (model, dataset)
  policy_nondegeneracy_{dataset}.csv — one row per model, action-wise dist columns

Usage:
  conda run -n simenv python3 table_policy_nondegeneracy.py \\
      --pred_dir output/tables/policy_nondegeneracy/predictions \\
      --out      output/tables/policy_nondegeneracy
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

DS_ALIASES  = {"supplychainshipmentpricing": "scsp"}
DS_DISPLAY  = {"dataco": "DataCo", "globalstore": "GlobalStore",
               "oas": "OAS", "scsp": "SCSP"}


def norm_ds(s):
    s = s.lower()
    return DS_ALIASES.get(s, s)


def entropy_bits(dist: np.ndarray) -> float:
    d = dist[dist > 0]
    return float(-np.sum(d * np.log2(d)))


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-10
    p = p + eps;  p /= p.sum()
    q = q + eps;  q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def action_dist(series: pd.Series, n_actions: int) -> np.ndarray:
    counts = series.value_counts()
    dist = np.zeros(n_actions)
    for k, v in counts.items():
        if 0 <= int(k) < n_actions:
            dist[int(k)] = v
    total = dist.sum()
    return dist / total if total > 0 else dist


def load_predictions(pred_dir: str):
    """Returns dict: (model, dataset) → DataFrame."""
    data = {}
    for csv_path in glob.glob(os.path.join(pred_dir, "*", "*_predictions.csv")):
        parts = csv_path.replace("\\", "/").split("/")
        model   = parts[-2]
        fname   = parts[-1]                          # {dataset}_predictions.csv
        ds      = norm_ds(fname.replace("_predictions.csv", ""))
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [WARN] Could not read {csv_path}: {e}")
            continue
        if "pred_action" not in df.columns or "true_action" not in df.columns:
            print(f"  [WARN] Missing columns in {csv_path}")
            continue
        data[(model, ds)] = df
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="output/tables/policy_nondegeneracy/predictions",
                    help="Dir containing {model}/{dataset}_predictions.csv files")
    ap.add_argument("--out", default="output/tables/policy_nondegeneracy")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[nondeg] Loading predictions from: {args.pred_dir}")
    preds = load_predictions(args.pred_dir)
    print(f"[nondeg] {len(preds)} (model, dataset) pairs loaded")
    if not preds:
        print("[nondeg] No prediction CSVs found — run run_action_distribution.sh first.")
        return

    # Infer number of actions per dataset
    n_actions_per_ds = {}
    for (model, ds), df in preds.items():
        n = max(int(df["true_action"].max()), int(df["pred_action"].max())) + 1
        n_actions_per_ds[ds] = max(n_actions_per_ds.get(ds, 0), n)

    summary_rows = []
    per_dataset   = {}   # ds → list of row dicts (one per model)

    for (model, ds), df in sorted(preds.items()):
        n_act = n_actions_per_ds[ds]
        P = action_dist(df["pred_action"], n_act)   # predicted
        Q = action_dist(df["true_action"], n_act)   # ground-truth

        h_pred  = entropy_bits(P)
        h_gt    = entropy_bits(Q)
        h_ratio = h_pred / h_gt if h_gt > 0 else float("nan")
        tvd_val = tvd(P, Q)
        kl_val  = kl_div(P, Q)

        # Summary row
        summary_rows.append({
            "model":     model,
            "dataset":   DS_DISPLAY.get(ds, ds),
            "n_samples": len(df),
            "TVD":       round(tvd_val, 4),
            "KL(P||Q)":  round(kl_val, 4),
            "H_pred":    round(h_pred, 4),
            "H_gt":      round(h_gt, 4),
            "H_ratio":   round(h_ratio, 4),
            **{f"P(a={k})": round(P[k], 4) for k in range(n_act)},
            **{f"Q(a={k})": round(Q[k], 4) for k in range(n_act)},
        })

        # Per-dataset row (for the per-dataset table)
        row = {"model": model}
        for k in range(n_act):
            row[f"pred_a{k}"] = f"{P[k]:.3f}"
            row[f"gt_a{k}"]   = f"{Q[k]:.3f}"
        row["TVD"]     = f"{tvd_val:.4f}"
        row["KL(P||Q)"] = f"{kl_val:.4f}"
        row["H_pred"]  = f"{h_pred:.4f}"
        row["H_gt"]    = f"{h_gt:.4f}"
        row["H_ratio"] = f"{h_ratio:.4f}"
        per_dataset.setdefault(ds, []).append(row)

    # ── Summary table ─────────────────────────────────────────────────────────
    sum_df = pd.DataFrame(summary_rows)
    sum_path = os.path.join(args.out, "policy_nondegeneracy_summary.csv")
    sum_df.to_csv(sum_path, index=False)
    print(f"\n{'='*90}")
    print("  Policy Non-Degeneracy Summary")
    print(f"{'='*90}")
    # Print compact view (metrics only, not per-action columns)
    metric_cols = ["model", "dataset", "n_samples", "TVD", "KL(P||Q)",
                   "H_pred", "H_gt", "H_ratio"]
    print(sum_df[metric_cols].to_string(index=False))
    print(f"\n  → {sum_path}")

    # ── Per-dataset tables ─────────────────────────────────────────────────────
    for ds, rows in sorted(per_dataset.items()):
        display = DS_DISPLAY.get(ds, ds)
        df = pd.DataFrame(rows)
        print(f"\n{'='*90}")
        print(f"  {display} — predicted vs ground-truth action distributions")
        print(f"{'='*90}")
        print(df.to_string(index=False))
        out_path = os.path.join(args.out, f"policy_nondegeneracy_{ds}.csv")
        df.to_csv(out_path, index=False)
        print(f"  → {out_path}")

    # ── Degeneracy flag ────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  Degeneracy check  (flag if H_ratio < 0.5 or TVD > 0.4)")
    print(f"{'='*90}")
    flagged = sum_df[(sum_df["H_ratio"] < 0.5) | (sum_df["TVD"] > 0.4)]
    if flagged.empty:
        print("  No degenerate policies detected.")
    else:
        print(flagged[metric_cols].to_string(index=False))


if __name__ == "__main__":
    main()
