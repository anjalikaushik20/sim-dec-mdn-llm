#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/latest_output/zero_shot/rl_baseline/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

echo "Zero-shot evaluation — RL baseline × {DataCo, GlobalStore, OAS}"
echo "All runs sequential on GPU 2"

python3 main/cb_main.py \
    --use_gpu 1 --device_id 2 --dataset DataCo --train_mode 2 \
    --wandb 1 --save 0 \
    --dm_epochs 0 --train_frac 1.0 --otr_reward_coeff 2 \
    --ckpt exp_report/DataCo/ckpt/mythical-commander-352_epoch218.pth \
    > "${BASE_OUT_DIR}/dataco.log" 2>&1

python3 main/cb_main.py \
    --use_gpu 1 --device_id 2 --dataset GlobalStore --train_mode 2 \
    --wandb 1 --save 0 \
    --dm_epochs 0 --train_frac 1.0 --otr_reward_coeff 10 \
    --ckpt exp_report/GlobalStore/ckpt/legendary-federation-353_epoch202.pth \
    > "${BASE_OUT_DIR}/globalstore.log" 2>&1

python3 main/cb_main.py \
    --use_gpu 1 --device_id 2 --dataset OAS --train_mode 2 \
    --wandb 1 --save 0 \
    --dm_epochs 0 --train_frac 1.0 --otr_reward_coeff 50 \
    --ckpt exp_report/OAS/ckpt/jedi-carrier-353_epoch19.pth \
    > "${BASE_OUT_DIR}/oas.log" 2>&1

echo "All zero-shot runs complete. Logs saved to ${BASE_OUT_DIR}"
echo "Extract results: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*.log"
