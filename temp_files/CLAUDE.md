# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All experiments run inside the `simenv` conda environment:
```bash
conda activate simenv
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
```

All `run_*.sh` scripts handle this setup internally. To run one in the background:
```bash
bash run_nohup.sh run_<script>.sh
```

## Running Experiments

There are three entry points in `main/`:

| Script | Purpose |
|---|---|
| `main/cb_main.py` | Simulator (MDN) training and RL decision-maker |
| `main/cb_main_llm.py` | LLM/VocabAlign decision-maker (frozen backbone + attnpool head) |
| `main/cb_main_ml.py` | ML baselines (Random Forest, XGBoost, Random, Historical) |

**Simulator training** (`--train_mode 1`):
```bash
python3 main/cb_main.py --dataset DataCo --train_mode 1 --use_gpu 1 --device_id 0 --wandb 1
```

**RL decision-maker** (`--train_mode 2`, requires simulator checkpoint):
```bash
python3 main/cb_main.py --dataset DataCo --train_mode 2 --ckpt <path>.pth --use_gpu 1 --device_id 0
```

**LLM decision-maker** (VocabAlign):
```bash
python3 main/cb_main_llm.py --dataset DataCo --train_mode 2 --hf_model_name Qwen/Qwen3-1.7B \
    --ckpt <simulator.pth> --dm_epochs 200 --train_frac 0.10 --wandb 1 --save 1 --ckpt_dir <dir>
```

**Cross-dataset eval** (train on DataCo, eval on GlobalStore/OAS — requires a saved adapter):
```bash
python3 main/cb_main_llm.py --dataset GlobalStore --train_mode 2 \
    --ckpt <globalstore_sim.pth> --dm_epochs 0 --value_network_ckpt <adapter.pth>
```

**Extract results from logs:**
```bash
grep 'best_profit\|best_on_time' output/latest_output/**/*.log
```

## Architecture

### Two-stage pipeline

**Stage 1 — Simulator** (`models/s_model.py: S_SimDec`): An MDN that learns the supply chain dynamics from historical data. Takes order features + shipping mode → predicts labels (late delivery risk, days for shipping, on-time). Trained with `--train_mode 1`.

**Stage 2 — Decision-maker**: Uses the frozen simulator as an oracle to evaluate shipping-mode decisions. Three variants:
- **RL** (`sessions/cb_session.py`): `ValueNetwork` (MLP, `models/v_model.py`) trained via policy gradient using the simulator's reward signal.
- **VocabAlign/LLM** (`sessions/cb_session_llm.py`): Frozen HuggingFace backbone + learned `pool_attn` + `cls_head`. Only the 2-layer adapter trains. Saves as `*_attnpool_best.pth`.
- **ML baselines** (`main/cb_main_ml.py`): Scikit-learn models operating directly on processed features.

### Data flow

`S_Loader` (`loaders/s_loader.py`) reads from `datasets/{dataset}/`:
- If `processed_{dataset}.csv` already exists → reads the four processed CSVs directly (fast path).
- Otherwise → reads the raw CSV, runs numerical/categorical/date preprocessing, and writes the processed CSVs.
- **Known inefficiency**: the raw CSV is always read into `self.ori_data` even on the fast path, then discarded.

`DataCo_OOD` is a special case: raw data comes from `Subset_1.csv` + `Subset_2.csv` but shares DataCo's entire feature schema (see `tools/feature_list.py` bottom).

### LLM serialization

`LLMAttnPoolNetwork.serialize_batch` (`models/llm_model.py`) converts processed (integer-encoded) feature vectors into natural-language prompts using feature names from `tools/feature_list.py`. The backbone runs once at startup to cache hidden states (`precompute_hidden_states`); only `forward_from_hidden` runs each training step.

For cross-dataset transfer, `serialize_batch` uses the **eval dataset's** feature names (e.g. GlobalStore's), not DataCo's — so the LLM reads semantically appropriate prompts even when the adapter was trained on a different dataset.

### Feature definitions

All dataset-specific feature lists live in `tools/feature_list.py`. Features are split into four groups used both for data loading and for LLM prompt construction:
- `product_info`, `order_info`, `customer_info`, `shipping_info` → features passed to the model
- `decision` → the shipping mode column
- `label` → prediction targets (late risk, days, on_time)

`retrieva_index` marks the slice boundaries for FAISS-based best-action lookup during LLM training.

### Checkpoint conventions

- **Simulator**: `{suffix}_epoch{N}.pth` — full `S_SimDec` state dict
- **LLM adapter**: `{suffix}_attnpool_best.pth` — contains only `pool_attn` and `cls_head` keys
- `suffix` = wandb run name (if `--wandb 1`) or `{timestamp}_{dataset}`
- `--ckpt_dir` overrides the default `exp_report/{dataset}/ckpt/` save location

### Output structure

```
output/latest_output/
  simulator/latest_run/ckpts/{dataset}/     ← simulator .pth files
  sample_efficiency/{model}/{RUN_ID}/       ← LLM sample-efficiency logs
  cross_dataset/vocabalign/{RUN_ID}/{model}/
    ckpts/frac{N}/                          ← one adapter per training fraction
    train_dataco_frac{N}.log
    eval_globalstore_frac{N}.log
    eval_oas_frac{N}.log
```

## Plotting

All plot scripts read log files and save `.png` alongside the logs:

```bash
conda run -n simenv python3 plot_sample_efficiency.py <run_dir>
conda run -n simenv python3 plot_crossdataset.py <model_run_dir>
conda run -n simenv python3 compare_all_methods.py
conda run -n simenv python3 compare_ablations.py
conda run -n simenv python3 compare_ood_vs_ind.py
```

Metrics to grep for in logs: `best_profit`, `best_on_time`, `best_pmp_1/2/3`.
