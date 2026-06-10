#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_OUT_DIR="output/decision_maker/all_fracs/models_ood"

DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"
echo "OOD evaluation — Qwen models (VocabAlign) on DataCo_OOD, dm_epochs=${DM_EPOCHS}"
echo "Running 2 models sequentially on GPU 3"

DATACO_CKPT="output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"

# HF name : model tag : output dir name (matches models_ood directory structure)
MODELS=(
    "Qwen/Qwen3-0.6B:qwen3-0.6B:qwen3-0.6"
    "Qwen/Qwen3-1.7B:qwen3-1.7B:qwen3-1.7"
)

run_model() {
    local HF_NAME="$1"
    local MODEL_TAG="$2"
    local DIR_NAME="$3"
    local GPU=3
    local OUT_DIR="${BASE_OUT_DIR}/${DIR_NAME}"
    mkdir -p "${OUT_DIR}/ckpts"

    local total=6
    local done_count=0
    for FRAC in 0.01 0.05 0.10 0.25 0.50 1.00; do
        done_count=$((done_count + 1))
        local LOG_FILE="${OUT_DIR}/dataco_ood_frac${FRAC}.log"
        echo "[GPU${GPU}] [${done_count}/${total}] ${MODEL_TAG} frac=${FRAC} starting"

        CUDA_VISIBLE_DEVICES="${GPU}" python3 main/cb_main_llm.py \
            --use_gpu 1 --device_id 0 --dataset DataCo_OOD --train_mode 2 \
            --wandb 1 --hf_model_name "${HF_NAME}" --save 1 --ckpt_dir "${OUT_DIR}/ckpts" \
            --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" --otr_reward_coeff 2 \
            --ckpt "${DATACO_CKPT}" \
            > "${LOG_FILE}" 2>&1

        echo "[GPU${GPU}] [${done_count}/${total}] ${MODEL_TAG} frac=${FRAC} done"
    done
    echo "[GPU${GPU}] ${MODEL_TAG} all fracs complete"
}

# Run models sequentially on GPU 3
for ENTRY in "${MODELS[@]}"; do
    IFS=":" read -r HF_NAME MODEL_TAG DIR_NAME <<< "${ENTRY}"
    run_model "${HF_NAME}" "${MODEL_TAG}" "${DIR_NAME}"
done

echo ""
echo "All OOD VocabAlign jobs complete. Logs saved to ${BASE_OUT_DIR}"
echo "Extract results: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*/*.log"
