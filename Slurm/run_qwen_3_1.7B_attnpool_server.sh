#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/latest_output/training/attnpool/qwen_3_1.7B/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

# Number of SFT epochs — override via: DM_EPOCHS=200 ./run_qwen_3_1.7B_attnpool_server.sh
# or pass as first positional arg: ./run_qwen_3_1.7B_attnpool_server.sh 200
DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"
# OAS is a small dataset — use a lower LR to prevent NaN loss (override via OAS_DM_LR=...)
OAS_DM_LR="${OAS_DM_LR:-0.00003}"
# Per-dataset entropy regularization coefficient (override via OAS_ENTROPY=...)
OAS_ENTROPY="${OAS_ENTROPY:-0.05}"
# Per-dataset otr_reward_coeff overrides (override via OAS_OTR=... etc.)
GS_OTR="${GS_OTR:-10}"
OAS_OTR="${OAS_OTR:-50}"
echo "Running AttnPool head fine-tuning for ${DM_EPOCHS} epochs (OAS dm_lr=${OAS_DM_LR} gs_otr=${GS_OTR} oas_otr=${OAS_OTR})"

# Run datasets in parallel — each on a dedicated GPU (skip GPU 2, used by another job)
# DataCo → GPU 0
python3 main/cb_main_llm_attnpool.py --use_gpu 1 --device_id 0 --dataset DataCo --train_mode 2 --wandb 1 --hf_model_name "Qwen/Qwen3-1.7B" --save 1 --dm_epochs "${DM_EPOCHS}" --otr_reward_coeff 2 --ckpt exp_report/DataCo/ckpt/mythical-commander-352_epoch218.pth > "${BASE_OUT_DIR}/dataco.log" 2>&1 &

# GlobalStore → GPU 1 — raised otr_reward_coeff 5→10 to close on_time gap vs RL (0.5976→0.7582)
python3 main/cb_main_llm_attnpool.py --use_gpu 1 --device_id 1 --dataset GlobalStore --train_mode 2 --wandb 1 --hf_model_name "Qwen/Qwen3-1.7B" --save 1 --dm_epochs "${DM_EPOCHS}" --otr_reward_coeff "${GS_OTR}" --ckpt exp_report/GlobalStore/ckpt/legendary-federation-353_epoch202.pth > "${BASE_OUT_DIR}/globalstore.log" 2>&1 &

# OAS → GPU 3 — otr_reward_coeff raised to 50 to force on_time signal over profit dominance
python3 main/cb_main_llm_attnpool.py --use_gpu 1 --device_id 3 --dataset OAS --train_mode 2 --wandb 1 --hf_model_name "Qwen/Qwen3-1.7B" --save 1 --dm_epochs "${DM_EPOCHS}" --dm_lr "${OAS_DM_LR}" --otr_reward_coeff "${OAS_OTR}" --ckpt exp_report/OAS/ckpt/jedi-carrier-353_epoch19.pth > "${BASE_OUT_DIR}/oas.log" 2>&1 &

wait
echo "All training runs completed. Logs saved to ${BASE_OUT_DIR}"
