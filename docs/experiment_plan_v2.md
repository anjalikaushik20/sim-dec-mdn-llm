# VocabAlign Full Experiment Suite — Plan v2

**Date:** 2026-06-04  
**Branch:** llm-training  
**Author:** Anjali Kaushik

---

## Overview

This document specifies the full 8-experiment suite required for the paper submission, including the addition of `microsoft/Phi-4-mini-reasoning` as the 7th backbone. It is organized as:

1. **Gap analysis** — current state vs. target matrix
2. **Prerequisite** — SupplyChainShipmentPricing dataset setup and simulator training
3. **Per-experiment specs** — what exists, what's missing, new files vs. modifications
4. **File inventory** — master list of what to create and what to modify
5. **Compute budget** — estimated job counts per experiment
6. **Verification** — how to sanity-check each experiment before full runs

---

## Gap Analysis

| Dimension | Current | Target |
|-----------|---------|--------|
| Backbones | 6 (gpt2, gpt2-medium, gpt2-large, Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B) | 7 — add `microsoft/Phi-4-mini-reasoning` (~3.8B params, ≈15 GB bfloat16) |
| Fracs (multi-frac scripts) | {0.01, 0.05, 0.10, 0.25, 0.50, 1.00} | Same 6 fracs — frac=0 (zero-shot) is handled separately by existing `run_zeroshot_all_server.sh` |
| Seeds | `--seed 42` only | 5 seeds: {42, 0, 1, 2, 3} |
| Datasets | DataCo, GlobalStore, OAS | + SupplyChainShipmentPricing (SCSP) — requires preprocessing + simulator training (see §Prerequisite) |
| Architecture ablation models | `bert_model.py`, `serialized_mlp_model.py` (implemented) | Already exists |
| Statistical significance | None | New `experiments/bootstrap_significance.py` |
| Prompt variants | None (single mode) | 4 variants in `llm_model.py` + new run script |
| Pareto analysis | None | New `plot_pareto_frontiers.py` |
| GlobalStore failure analysis | None | New `analyze_globalstore_failure.py` |

**Phi-4-mini-reasoning note:** `llm_model.py` loads any model via `AutoModelForCausalLM.from_pretrained()`. Phi-4-mini-reasoning requires `trust_remote_code=True` — verify this flag is in the `from_pretrained` call before running. Memory budget: ~15 GB/job, same tier as Qwen3-1.7B.

---

## Prerequisite: SupplyChainShipmentPricing (SCSP) Dataset Setup

Complete before any experiment that includes SCSP. Full spec in `docs/dataset_integration_plan.md` §Dataset 1.

### Step 1 — Preprocessing script

**New file:** `datasets/SupplyChainShipmentPricing/preprocess_scsp.py`

- Read `SCMS_Delivery_History_Dataset.csv` (9,324 rows, 33 cols)
- Drop rows where `Shipment Mode` is N/A
- Remap: Air→0, Air Charter→1, Ocean→2, Truck→3
- Derive: `on_time` (delivered ≤ scheduled), `days_for_shipping`, `late_risk`, `benefit` (Line Item Value − Freight Cost − Insurance)
- Parse `PO Sent to Vendor Date` and `Scheduled Delivery Date` → year/month/day columns
- Build 4 cost CSVs via groupby `(Product Group, Country, Shipment Mode)` → mean benefit
- 70/15/15 stratified split (stratify on Shipment Mode); save train/val/test CSVs and 4 `processed_cost_*.csv` files

### Step 2 — Feature list and argparse registration

**Edit `tools/feature_list.py`:** Add the 11 dict entries listed in `docs/dataset_integration_plan.md` §Dataset 1 (numerical_features, categorical_features, date_features, product_info, order_info, customer_info, shipping_info, decision, label, profit, retrieva_index).

**Edit three argparse choices lists:**
- `main/cb_main.py:31` — add `SupplyChainShipmentPricing`
- `main/cb_main_llm.py:31` — add `SupplyChainShipmentPricing`
- `main/cb_main_ml.py:50` — add `SupplyChainShipmentPricing`

### Step 3 — Simulator training

```bash
conda activate simenv
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"

# Smoke-test (no GPU, 5 epochs — verifies S_Loader runs without crash)
python3 main/cb_main.py --dataset SupplyChainShipmentPricing \
    --train_mode 1 --use_gpu 0 --epochs 5

# Confirm cost_mrp has all 4 actions
python3 -c "import pandas as pd; df=pd.read_csv('datasets/SupplyChainShipmentPricing/processed_cost_mrp.csv'); print(df.iloc[:,2].value_counts())"

# Full training
python3 main/cb_main.py --dataset SupplyChainShipmentPricing \
    --train_mode 1 --use_gpu 1 --device_id 0 --wandb 1
```

Save best checkpoint to `output/simulator/latest_run/ckpts/scsp/`.

### Step 4 — Calibrate otr_reward_coeff

```bash
python3 main/cb_main_llm.py --dataset SupplyChainShipmentPricing \
    --train_mode 2 --dm_epochs 50 --train_frac 0.10 \
    --ckpt output/simulator/latest_run/ckpts/scsp/<best>.pth \
    --hf_model_name Qwen/Qwen3-1.7B --otr_reward_coeff 2
```

Inspect profit/on_time trade-off in log, then tune `otr_reward_coeff` for SCSP before adding it to sweep scripts.

---

## Experiment 1 — Headline Sample Efficiency

**Matrix:** 7 backbones × 4 datasets × 6 fracs × 5 seeds (multi-frac), plus zero-shot curve point from `run_zeroshot_all_server.sh`  
**Purpose:** Main empirical claim — VocabAlign matches/exceeds full-data RL at low fractions.

### Current state

`run_sample_efficiency_all_vocabalign_parallel.sh` covers 6 backbones × 3 datasets × 6 fracs × seed=42 = **108 jobs done.** Results under `output/decision_maker/all_fracs/`.

`run_sample_efficiency_rl_server.sh` covers RL baseline at 6 fracs × 3 datasets × seed=42 = **18 jobs done.**

### Gaps

1. Missing `microsoft/Phi-4-mini-reasoning` backbone (7th model)
2. Missing 4 additional seeds {0, 1, 2, 3} for all backbones + datasets
3. Missing SupplyChainShipmentPricing (requires §Prerequisite first)

### Changes required

**New file: `run_sample_efficiency_phi4_parallel.sh`**
- Single `run_model_group` call for `microsoft/Phi-4-mini-reasoning`
- Memory: ~15 GB/job → max_par=6 on two 49 GB GPUs
- Includes SCSP case block; uses same semaphore pattern as existing script
- Fracs: `(0.01 0.05 0.10 0.25 0.50 1.00)` — zero-shot handled separately

**Modify `run_sample_efficiency_all_vocabalign_parallel.sh`:**
- Add `SEEDS=(42 0 1 2 3)` outer loop wrapping the model-group calls
- Add `SupplyChainShipmentPricing` case block inside `launch_job()` (once simulator ckpt and otr_reward_coeff are known)
- Log path: include seed in directory name `${BASE_OUT_DIR}/seed${SEED}/${model_tag}/`

**Modify `run_sample_efficiency_rl_server.sh`:**
- Add `SEEDS=(42 0 1 2 3)` outer loop
- Add `SupplyChainShipmentPricing` case block

### Job count

| Component | Jobs |
|-----------|------|
| VocabAlign 6 existing backbones × 4 datasets × 6 fracs × 4 new seeds | 576 |
| VocabAlign Phi-4-mini-reasoning × 4 datasets × 6 fracs × 5 seeds | 120 |
| RL 4 datasets × 6 fracs × 4 new seeds | 96 |
| **Total new** | **792** |

Zero-shot (frac=0) curve point: run `run_zeroshot_all_server.sh` for all 7 backbones × 4 datasets once per seed.

---

## Experiment 2 — Statistical Significance Overlay

**Matrix:** Post-hoc analysis on Exp 1 results — no new training  
**Purpose:** Paired bootstrap (1000 resamples) + Wilcoxon signed-rank for each VocabAlign-vs-RL comparison.

### Current state

No script exists. Bootstrap pattern is in `experiments/observational_matching.py` (~lines 80–120).

### New file: `experiments/bootstrap_significance.py`

- Parse per-seed `best_profit` values from Exp 1 log files (`output/decision_maker/all_fracs/seed{N}/{model}/{dataset}_frac{f}.log` and `output/decision_maker/rl/seed{N}/{dataset}_frac{f}.log`)
- For each `(backbone, dataset, frac)`: collect 5-seed profit vectors for VocabAlign and RL
- Run `scipy.stats.wilcoxon` (paired) and numpy bootstrap (1000 resamples) for 95% CI on the delta
- Output CSV: `(backbone, dataset, frac, mean_vocabalign, mean_rl, delta, p_value, ci_lower, ci_upper)`
- Print LaTeX table fragment for direct paper inclusion

```bash
conda run -n simenv python3 experiments/bootstrap_significance.py \
    --vocabalign_dir output/decision_maker/all_fracs \
    --rl_dir output/decision_maker/rl \
    --out output/significance_table.csv
```

---

## Experiment 3 — Architecture Ablation

**Matrix:** 2 baselines (SerializedMLP, frozen BERT-base) × 4 datasets × 6 fracs × 5 seeds  
**Purpose:** Defends "pretrained decoder-only LLM, not just any text encoder" claim.

### Current state

`run_exp2_arch_ablation_parallel.sh` covers SerializedMLP + BERT × 3 datasets × 6 fracs × seed=42 = **36 jobs done.** Models already fully implemented: `models/serialized_mlp_model.py`, `models/bert_model.py`. **Known bug on line 34:** `GPUS=(1, 2)` must be `GPUS=(1 2)`.

### Modifications to `run_exp2_arch_ablation_parallel.sh`

1. Fix bash bug: `GPUS=(1, 2)` → `GPUS=(1 2)`
2. Add `SEEDS=(42 0 1 2 3)` outer loop
3. Add `SupplyChainShipmentPricing` case block in `launch_job()`
4. Fracs unchanged: `(0.01 0.05 0.10 0.25 0.50 1.00)`

### Job count

2 × 4 × 6 × 5 = **240 total**; subtract 36 done = **~204 additional.**

---

## Experiment 4 — Method Ablations

**Matrix:** 3 backbones × 3 ablation variants × 4 datasets × 5 seeds, frac=1.00 only  
**Purpose:** Shows each VocabAlign component (vocab init, attention pool, soft labels) contributes.  
**Caption note:** "Ablations run on three representative backbones across scales and pretraining families."

### Current state

`run_ablation_vocabalign_server.sh` runs 6 backbones × 3 variants × 6 fracs × 3 datasets × 1 seed = 324 jobs — wrong scope. **Do not modify it.**

### New file: `run_ablation_3backbones_server.sh`

| Dimension | Value |
|-----------|-------|
| Backbones | `gpt2`, `Qwen/Qwen3-1.7B`, `microsoft/Phi-4-mini-reasoning` |
| Variants | `no_vocab_init` (--pool_init random), `mean_pool` (--pool_type mean), `hard_labels_only` (--no_soft_labels) |
| Datasets | DataCo, GlobalStore, OAS, SupplyChainShipmentPricing |
| Seeds | 5 |
| Frac | 1.00 only |

Reuse `launch_job` pattern from `run_ablation_vocabalign_server.sh`. Log path: `output/decision_maker/ablation_3b/{RUN_ID}/{variant}/{model_tag}/{dataset}_seed{N}.log`.

**Total: 3 × 3 × 4 × 5 = 180 jobs.**

---

## Experiment 5 — Out-of-Simulator Validation

**Matrix:** Qwen3-1.7B × DataCo × 5 seeds  
**Purpose:** Rebuttal to circular evaluation concern — validates against realized outcomes in `processed_DataCo_test.csv`.

### Current state

- `run_exp1_observational_matching.sh` — inference + matching for 3 datasets, seed=42
- `experiments/observational_matching.py` — fully implemented (bootstrap CI, balance table, subgroup breakdown)
- Results exist at `output/exp1_observational_matching/20260603_183808/`

### Modification to `run_exp1_observational_matching.sh`

Add `SEEDS=(42 0 1 2 3)` outer loop; output dir becomes `output/exp1_observational_matching/seed${SEED}/`. After all seeds complete, add a final aggregation call:

```bash
conda run -n simenv python3 experiments/observational_matching.py \
    --mode aggregate \
    --seed_dirs "output/exp1_observational_matching/seed*" \
    --out output/exp1_observational_matching/aggregate_ate.csv
```

Add `--mode aggregate` branch to `experiments/observational_matching.py` that reads per-seed ATE CSVs and reports mean ± std across seeds.

**Total: 5 inference runs + 5 matching runs = 10 jobs.**

---

## Experiment 6 — Prompt Ablation

**Matrix:** 4 prompt variants × 2 backbones (gpt2, Qwen3-1.7B) × 2 datasets (DataCo, OAS) × 5 seeds, frac=1.00  
**Purpose:** Tests whether natural-language serialization contributes, or whether the LLM does numeric pattern matching.

### Code changes required

**Modify `models/llm_model.py`:**

Add `prompt_variant: str = "natural"` parameter to `LLMAttnPoolNetwork.__init__()` and store as `self._prompt_variant`. In `serialize_batch()`, branch on `self._prompt_variant`:

| Variant | Description | Implementation |
|---------|-------------|----------------|
| `natural` | Existing behavior | Unchanged |
| `numeric` | Feature values only, no names | `" ".join(str(v) for v in feature_vector)` |
| `shuffled_names` | Values with randomly permuted feature names | Permute the name list with `np.random.default_rng(hash(tuple(raw_state[0].tolist())) & 0xFFFFFFFF).permutation(names)` per batch |
| `names_only` | Feature names without values | `" ".join(feature_names)` — same for every sample |

**Modify `main/cb_main_llm.py`:**

Add to `parse_args()`:
```python
parser.add_argument('--prompt_variant', type=str, default='natural',
    choices=['natural', 'numeric', 'shuffled_names', 'names_only'])
```
Pass to model constructor.

**New file: `run_prompt_ablation_server.sh`**

- Backbones: `gpt2`, `Qwen/Qwen3-1.7B`
- Datasets: DataCo, OAS
- Variants: natural, numeric, shuffled_names, names_only
- Seeds: 5, frac=1.00
- Log path: `output/decision_maker/prompt_ablation/{RUN_ID}/{variant}/{model_tag}/{dataset}_seed{N}.log`
- **Total: 4 × 2 × 2 × 5 = 80 jobs**

---

## Experiment 7 — RF/XGBoost Pareto Analysis

**Matrix:** No new training — uses existing ML baseline results  
**Purpose:** Shows RF/XGBoost achieve high profit but poor on-time; VocabAlign achieves better trade-off.

### Current state

ML results from `run_sample_efficiency_ml_server.sh` under `output/decision_maker/ml/`. Log parsing patterns in `compare_all_methods.py` and `compare_ml_vs_llm.py`.

### New file: `plot_pareto_frontiers.py`

- Parse `best_profit` and `best_on_time` from ML logs (RF, XGBoost, Random, Historical) and VocabAlign logs (all backbones, frac=1.00)
- One subplot per dataset; x-axis = profit, y-axis = on_time_ratio
- Plot Pareto frontier as step function connecting non-dominated points
- Color: VocabAlign backbones in blue gradient, RF/XGBoost in red/orange, RL in green, baselines in grey
- Save PNG to `output/decision_maker/comparisons/pareto/{dataset}_pareto.png`

```bash
conda run -n simenv python3 plot_pareto_frontiers.py \
    --ml_dir output/decision_maker/ml \
    --llm_dir output/decision_maker/all_fracs \
    --rl_dir output/decision_maker/rl \
    --frac 1.00
```

**Key reuse:** `compare_all_methods.py` log-parsing functions.

---

## Experiment 8 — GlobalStore Failure Analysis

**Matrix:** No new training — diagnostic over existing logs and processed CSVs  
**Purpose:** Honest engagement with negative result (RL wins on GlobalStore); raises reviewer trust.

### Current state

`analyze_ood_shift.py` implements KL divergence and distribution shift analysis for DataCo_OOD; patterns are directly reusable.

### New file: `analyze_globalstore_failure.py`

**Outputs** (saved to `output/globalstore_failure_analysis/`):

1. **Action distribution entropy table** — compute Shannon entropy of the historical shipping mode distribution for each dataset's training split. Low entropy = skewed toward one mode.

2. **Simulator confidence table** — for each dataset's test split, pass features through the frozen simulator (DataCo simulator used for cross-dataset comparison), take softmax of output logits, compute mean entropy. Low entropy = high confidence; high entropy = uncertain.

3. **Feature KL divergence table** — per-feature KL divergence between DataCo train distribution and GlobalStore train distribution (discretize continuous features into 20 bins). Rank features by KL to identify the largest distribution gaps.

4. **Correlation plot** — scatter of (simulator confidence on each GlobalStore test sample) vs (VocabAlign profit - RL profit per sample); hypothesis: low simulator confidence → RL advantage.

5. **Hypothesis paragraph** — written as a text block in stdout summarizing the most likely root cause based on the above three tables.

```bash
conda run -n simenv python3 analyze_globalstore_failure.py \
    --sim_ckpt output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth \
    --vocabalign_log <best_dataco_vocabalign.log> \
    --rl_log <best_dataco_rl.log>
```

**Key reuse:** KL divergence / entropy functions from `analyze_ood_shift.py`.

---

## File Inventory

### New files to create

| File | Experiment | Notes |
|------|-----------|-------|
| `datasets/SupplyChainShipmentPricing/preprocess_scsp.py` | Prerequisite | Full spec in `docs/dataset_integration_plan.md` §Dataset 1 |
| `run_sample_efficiency_phi4_parallel.sh` | Exp 1 | Phi-4-mini-reasoning backbone sweep |
| `run_rl_multiseed_server.sh` | Exp 1 | Multi-seed RL companion (or modify existing) |
| `experiments/bootstrap_significance.py` | Exp 2 | Paired bootstrap + Wilcoxon |
| `run_ablation_3backbones_server.sh` | Exp 4 | 3-backbone method ablation |
| `run_prompt_ablation_server.sh` | Exp 6 | 4 prompt variants × 2 backbones |
| `plot_pareto_frontiers.py` | Exp 7 | Profit vs on_time Pareto |
| `analyze_globalstore_failure.py` | Exp 8 | GlobalStore diagnostic |

### Files to modify (not create)

| File | Change |
|------|--------|
| `tools/feature_list.py` | Add 11 SCSP dict entries |
| `main/cb_main.py:31` | Add `SupplyChainShipmentPricing` to choices |
| `main/cb_main_llm.py:31` | Add `SupplyChainShipmentPricing` to choices; add `--prompt_variant` arg |
| `main/cb_main_ml.py:50` | Add `SupplyChainShipmentPricing` to choices |
| `models/llm_model.py` | Add `prompt_variant` param + 3 new modes in `serialize_batch()` |
| `run_sample_efficiency_all_vocabalign_parallel.sh` | Add seed loop + SCSP case block |
| `run_sample_efficiency_rl_server.sh` | Add seed loop + SCSP case block |
| `run_exp2_arch_ablation_parallel.sh` | Fix bash bug (line 34), add seed loop + SCSP case block |
| `run_exp1_observational_matching.sh` | Add seed loop |
| `experiments/observational_matching.py` | Add `--mode aggregate` branch |

### Files that already exist — do NOT recreate

`models/llm_model.py`, `models/bert_model.py`, `models/serialized_mlp_model.py`, `experiments/observational_matching.py`, all `run_sample_efficiency_*.sh`, `run_exp2_arch_ablation_parallel.sh`, `run_ablation_vocabalign_server.sh`, `run_zeroshot_all_server.sh`, all `compare_*.py`, all `plot_*.py`, `analyze_ood_shift.py`.

---

## Compute Budget

| Experiment | New jobs | Notes |
|-----------|---------|-------|
| 1 (VocabAlign multi-seed) | ~696 | 6 existing backbones × 4 datasets × 6 fracs × 4 new seeds |
| 1 (Phi-4) | 120 | × 5 seeds |
| 1 (RL multi-seed) | 96 | 4 datasets × 6 fracs × 4 new seeds |
| 1 (zero-shot) | ~56 | 7 backbones × 4 datasets × 2 per direction via `run_zeroshot_all_server.sh` |
| 2 (significance) | 0 | Post-hoc analysis only |
| 3 (arch ablation) | ~204 | 2 baselines × 4 datasets × 6 fracs × 4 new seeds |
| 4 (method ablation) | 180 | 3 × 3 × 4 × 5 |
| 5 (obs. matching) | 10 | 5 seeds inference + matching |
| 6 (prompt ablation) | 80 | 4 × 2 × 2 × 5 |
| 7 (Pareto) | 0 | Analysis only |
| 8 (GlobalStore diag.) | 0 | Analysis only |
| **Total new** | **~1,442** | |

---

## Verification

### Before each full sweep
```bash
# Test a single job with frac=0.01, dm_epochs=10 first
CUDA_VISIBLE_DEVICES=2 python3 main/cb_main_llm.py \
    --dataset DataCo --train_mode 2 --dm_epochs 10 \
    --hf_model_name microsoft/Phi-4-mini-reasoning \
    --ckpt output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth \
    --train_frac 0.01 --otr_reward_coeff 2 --use_gpu 1 --device_id 0
```

### After each sweep
```bash
# Extract metrics from all logs
grep 'best_profit\|best_on_time' output/decision_maker/all_fracs/**/*.log

# Check for failed runs (empty logs or missing best_profit)
for f in output/decision_maker/all_fracs/**/*.log; do
    grep -q 'best_profit' "$f" || echo "MISSING: $f"
done
```

### Exp 2 (significance)
```bash
python3 experiments/bootstrap_significance.py \
    --vocabalign_dir output/decision_maker/all_fracs \
    --rl_dir output/decision_maker/rl \
    --out output/significance_table.csv
# Expect: p_value < 0.05 for low fracs where VocabAlign > RL; NA where seeds not yet collected
```

### Exp 6 (prompt ablation sanity)
- `numeric` and `names_only` variants should converge slower than `natural` — if they match, the LLM isn't using semantics
- `shuffled_names` should perform between `numeric` and `natural`

### SCSP simulator
```bash
# Confirm all 4 actions present in cost_mrp
python3 -c "
import pandas as pd
df = pd.read_csv('datasets/SupplyChainShipmentPricing/processed_cost_mrp.csv')
print(df.iloc[:,2].value_counts())
assert len(df.iloc[:,2].unique()) == 4, 'Missing actions in cost_mrp!'
"
```
