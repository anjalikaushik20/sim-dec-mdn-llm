"""
Analyze the distribution shift between DataCo_OOD Subset_1 (train) and Subset_2 (test).

Outputs:
  - Console summary table (feature means, stds, KL divergence)
  - /data/akaush39/sim-to-dec/output/latest_output/comparisons/ood_analysis/
      feature_shift.png    — per-feature mean/std comparison bar charts
      shipping_mode.png    — shipping mode class distribution comparison
      profit_distribution.png — profit distribution overlay
      ood_shift_summary.csv — numeric summary for paper tables

Run with:  conda run -n simenv python3 analyze_ood_shift.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SUBSET1 = os.path.join(ROOT, "datasets/DataCo_OOD/Subset_1.csv")
SUBSET2 = os.path.join(ROOT, "datasets/DataCo_OOD/Subset_2.csv")
OUT_DIR = os.path.join(ROOT, "/data/akaush39/sim-to-dec/output/latest_output/comparisons/ood_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Features (from tools/feature_list.py for DataCo) ─────────────────────────
NUMERICAL = [
    "Benefit per order",
    "Sales per customer",
    "Latitude",
    "Longitude",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Product Price",
]
SHIPPING_MODE_COL = "Shipping Mode"
PROFIT_COL = "Order Profit Per Order"
LATE_RISK_COL = "Late_delivery_risk"
DAYS_REAL_COL = "Days for shipping (real)"

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading Subset_1 (train) from {SUBSET1} ...")
df1 = pd.read_csv(SUBSET1, low_memory=False)
print(f"Loading Subset_2 (test/OOD) from {SUBSET2} ...")
df2 = pd.read_csv(SUBSET2, low_memory=False)
print(f"Subset_1 rows: {len(df1):,}   Subset_2 rows: {len(df2):,}")
print()


# ── Helper: KL divergence between two continuous distributions via histogram ──
def kl_div_hist(x1, x2, bins=50):
    """Estimate KL(P1 || P2) where P1=Subset1, P2=Subset2."""
    lo = min(x1.min(), x2.min())
    hi = max(x1.max(), x2.max())
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p1, _ = np.histogram(x1, bins=edges, density=True)
    p2, _ = np.histogram(x2, bins=edges, density=True)
    # Smooth to avoid log(0)
    p1 = p1 + 1e-10
    p2 = p2 + 1e-10
    p1 /= p1.sum()
    p2 /= p2.sum()
    return float(entropy(p1, p2))


# ── 1. Numerical feature shift ─────────────────────────────────────────────────
rows = []
for feat in NUMERICAL:
    if feat not in df1.columns or feat not in df2.columns:
        print(f"  WARNING: '{feat}' not found in one of the subsets, skipping")
        continue
    v1 = df1[feat].dropna().values.astype(float)
    v2 = df2[feat].dropna().values.astype(float)
    kl = kl_div_hist(v1, v2)
    rows.append({
        "Feature": feat,
        "Subset1_mean": v1.mean(),
        "Subset2_mean": v2.mean(),
        "Subset1_std":  v1.std(),
        "Subset2_std":  v2.std(),
        "Mean_shift":   abs(v1.mean() - v2.mean()),
        "KL(S1||S2)":   kl,
    })

feat_df = pd.DataFrame(rows)
print("=" * 80)
print("NUMERICAL FEATURE SHIFT SUMMARY")
print("=" * 80)
print(feat_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print()

# ── 2. Shipping mode distribution ─────────────────────────────────────────────
mode_order = ["Standard Class", "Second Class", "First Class", "Same Day"]

def mode_counts(df):
    c = df[SHIPPING_MODE_COL].value_counts()
    total = c.sum()
    return {m: c.get(m, 0) / total for m in mode_order}

mc1 = mode_counts(df1)
mc2 = mode_counts(df2)
print("=" * 80)
print("SHIPPING MODE DISTRIBUTION (fraction of rows)")
print("=" * 80)
print(f"{'Mode':<20} {'Subset1 (train)':>16} {'Subset2 (OOD test)':>18}")
for m in mode_order:
    print(f"{m:<20} {mc1[m]:>16.4f} {mc2[m]:>18.4f}")

# KL divergence of shipping mode distribution
p1_modes = np.array([mc1[m] for m in mode_order]) + 1e-10
p2_modes = np.array([mc2[m] for m in mode_order]) + 1e-10
p1_modes /= p1_modes.sum()
p2_modes /= p2_modes.sum()
mode_kl = float(entropy(p1_modes, p2_modes))
print(f"\n  KL(Subset1 || Subset2) for Shipping Mode: {mode_kl:.6f}")
print()

# ── 3. Late delivery risk and actual shipping days ─────────────────────────────
print("=" * 80)
print("LABEL DISTRIBUTION")
print("=" * 80)
for col in [LATE_RISK_COL, DAYS_REAL_COL]:
    if col in df1.columns:
        v1 = df1[col].dropna()
        v2 = df2[col].dropna()
        if v1.dtype == object:
            continue
        print(f"  {col}:")
        print(f"    Subset1: mean={v1.mean():.4f}  std={v1.std():.4f}")
        print(f"    Subset2: mean={v2.mean():.4f}  std={v2.std():.4f}")
print()

# ── 4. Save summary CSV ────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "ood_shift_summary.csv")
feat_df.to_csv(csv_path, index=False)
print(f"Saved numeric summary → {csv_path}")

# ── 5. Plots ───────────────────────────────────────────────────────────────────

# 5a. Feature mean comparison (top-K by KL divergence)
top_k = feat_df.nlargest(8, "KL(S1||S2)")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(top_k))
w = 0.35
ax = axes[0]
ax.bar(x - w/2, top_k["Subset1_mean"], w, label="Subset1 (train)", color="#4C72B0")
ax.bar(x + w/2, top_k["Subset2_mean"], w, label="Subset2 (OOD test)", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(top_k["Feature"], rotation=35, ha="right", fontsize=8)
ax.set_title("Top-8 Features by KL Divergence — Mean Comparison")
ax.set_ylabel("Mean value")
ax.legend()

ax2 = axes[1]
ax2.barh(top_k["Feature"], top_k["KL(S1||S2)"], color="#55A868")
ax2.set_xlabel("KL(S1 || S2)")
ax2.set_title("KL Divergence by Feature")
ax2.invert_yaxis()

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "feature_shift.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {fig_path}")

# 5b. Shipping mode distribution
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(mode_order))
w = 0.35
ax.bar(x - w/2, [mc1[m] for m in mode_order], w, label="Subset1 (train)", color="#4C72B0")
ax.bar(x + w/2, [mc2[m] for m in mode_order], w, label="Subset2 (OOD test)", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(mode_order)
ax.set_ylabel("Fraction of orders")
ax.set_title(f"Shipping Mode Distribution Shift  (KL={mode_kl:.4f})")
ax.legend()
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "shipping_mode.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {fig_path}")

# 5c. Profit distribution overlay
fig, ax = plt.subplots(figsize=(8, 4))
p1_vals = df1[PROFIT_COL].dropna().clip(-500, 500)
p2_vals = df2[PROFIT_COL].dropna().clip(-500, 500)
bins = np.linspace(-500, 500, 80)
ax.hist(p1_vals, bins=bins, density=True, alpha=0.6, label="Subset1 (train)", color="#4C72B0")
ax.hist(p2_vals, bins=bins, density=True, alpha=0.6, label="Subset2 (OOD test)", color="#DD8452")
kl_profit = kl_div_hist(p1_vals.values, p2_vals.values)
ax.set_xlabel("Order Profit Per Order")
ax.set_ylabel("Density")
ax.set_title(f"Profit Distribution Shift  (KL={kl_profit:.4f})")
ax.legend()
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "profit_distribution.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {fig_path}")

print()
print("=" * 80)
print(f"All outputs saved to: {OUT_DIR}")
print("=" * 80)
