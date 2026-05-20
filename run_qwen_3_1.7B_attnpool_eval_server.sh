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
BASE_OUT_DIR="output/latest_output/eval/attnpool/qwen_3_1.7B/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

# Per-dataset otr_reward_coeff overrides — used in label precomputation (override via OAS_OTR=... etc.)
GS_OTR="${GS_OTR:-10}"
OAS_OTR="${OAS_OTR:-50}"
echo "Running AttnPool eval (no adapter training) — zero-init pool_attn = mean pooling (gs_otr=${GS_OTR} oas_otr=${OAS_OTR})"

# Run datasets in parallel — each on a dedicated GPU (skip GPU 2, used by another job)
# DataCo → GPU 0
python3 main/cb_main_llm_attnpool.py --use_gpu 1 --device_id 0 --dataset DataCo --train_mode 2 --wandb 0 --hf_model_name "Qwen/Qwen3-1.7B" --save 0 --dm_epochs 0 --otr_reward_coeff 2 --ckpt exp_report/DataCo/ckpt/mythical-commander-352_epoch218.pth > "${BASE_OUT_DIR}/dataco.log" 2>&1 &

# GlobalStore → GPU 1
python3 main/cb_main_llm_attnpool.py --use_gpu 1 --device_id 1 --dataset GlobalStore --train_mode 2 --wandb 0 --hf_model_name "Qwen/Qwen3-1.7B" --save 0 --dm_epochs 0 --otr_reward_coeff "${GS_OTR}" --ckpt exp_report/GlobalStore/ckpt/legendary-federation-353_epoch202.pth > "${BASE_OUT_DIR}/globalstore.log" 2>&1 &

# OAS → GPU 3
python3 main/cb_main_llm_attnpool.py --use_gpu 1 --device_id 3 --dataset OAS --train_mode 2 --wandb 0 --hf_model_name "Qwen/Qwen3-1.7B" --save 0 --dm_epochs 0 --otr_reward_coeff "${OAS_OTR}" --ckpt exp_report/OAS/ckpt/jedi-carrier-353_epoch19.pth > "${BASE_OUT_DIR}/oas.log" 2>&1 &

wait
echo "All eval runs completed. Logs saved to ${BASE_OUT_DIR}"
