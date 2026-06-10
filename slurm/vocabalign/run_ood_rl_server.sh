#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/latest_output/sample_efficiency/ood_rl_baseline/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

DM_EPOCHS="${1:-${DM_EPOCHS:-6000}}"
echo "OOD evaluation — RL baseline (ValueNetwork) on DataCo_OOD, dm_epochs=${DM_EPOCHS}"
echo "Fracs: 0.01 0.05 0.10 0.25 0.50 1.00, sequential on GPU 0"

DATACO_CKPT="output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"

for FRAC in 0.01 0.05 0.10 0.25 0.50 1.00; do
    echo "--- frac=${FRAC} ---"
    LOG_FILE="${BASE_OUT_DIR}/dataco_ood_frac${FRAC}.log"

    python3 main/cb_main.py \
        --use_gpu 1 --device_id 1 --dataset DataCo_OOD --train_mode 2 \
        --wandb 1 --save 1 --ckpt_dir "${BASE_OUT_DIR}/ckpts" \
        --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" --otr_reward_coeff 2 \
        --ckpt "${DATACO_CKPT}" \
        > "${LOG_FILE}" 2>&1

    echo "frac=${FRAC} done"
done

echo "All OOD RL runs complete. Logs saved to ${BASE_OUT_DIR}"
echo "Extract results: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*.log"
