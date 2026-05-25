#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/latest_output/sample_efficiency/rl_baseline/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"
echo "Sample efficiency matrix — RL baseline (ValueNetwork), dm_epochs=${DM_EPOCHS}"
echo "Fracs: 0.01 0.05 0.10 0.25 0.50 1.0 × {DataCo, GlobalStore, OAS}"

for FRAC in 0.01 0.05 0.10 0.25 0.50 1.0; do
    echo "--- frac=${FRAC} ---"

    # DataCo → GPU 0
    python3 main/cb_main.py \
        --use_gpu 1 --device_id 1 --dataset DataCo --train_mode 2 \
        --wandb 1 --save 1 --ckpt_dir "${BASE_OUT_DIR}/ckpts" \
        --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" --otr_reward_coeff 2 \
        --ckpt output/latest_output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth \
        > "${BASE_OUT_DIR}/dataco_frac${FRAC}.log" 2>&1 &

    # GlobalStore → GPU 1
    python3 main/cb_main.py \
        --use_gpu 1 --device_id 1 --dataset GlobalStore --train_mode 2 \
        --wandb 1 --save 1 --ckpt_dir "${BASE_OUT_DIR}/ckpts" \
        --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" --otr_reward_coeff 10 \
        --ckpt output/latest_output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth \
        > "${BASE_OUT_DIR}/globalstore_frac${FRAC}.log" 2>&1 &

    # OAS → GPU 3
    python3 main/cb_main.py \
        --use_gpu 1 --device_id 1 --dataset OAS --train_mode 2 \
        --wandb 1 --save 1 --ckpt_dir "${BASE_OUT_DIR}/ckpts" \
        --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" --otr_reward_coeff 50 \
        --ckpt output/latest_output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth \
        > "${BASE_OUT_DIR}/oas_frac${FRAC}.log" 2>&1 &

    wait
    echo "frac=${FRAC} done"
done

echo "All runs complete. Logs saved to ${BASE_OUT_DIR}"
echo "Extract results: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*.log"
