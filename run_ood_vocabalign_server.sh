#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="/data/akaush39/sim-to-dec/output/latest_output/sample_efficiency/ood_vocabalign/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"
echo "OOD evaluation — all models (VocabAlign) on DataCo_OOD, dm_epochs=${DM_EPOCHS}"
echo "Fracs: 0.01 0.05 0.10 0.25 0.50 1.00 × 6 models, sequential on GPU 0"

DATACO_CKPT="/data/akaush39/sim-to-dec/output/latest_output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"

total=0
done_count=0
for ENTRY in \
    "Qwen/Qwen3-0.6B:qwen3-0.6B" \
    "Qwen/Qwen3-1.7B:qwen3-1.7B" \
    "Qwen/Qwen3-4B:qwen3-4B" \
    "gpt2:gpt2" \
    "gpt2-medium:gpt2-medium" \
    "gpt2-large:gpt2-large"
do
    for FRAC in 0.01 0.05 0.10 0.25 0.50 1.00; do
        total=$((total + 1))
    done
done
echo "Total jobs: ${total}"

for ENTRY in \
    "Qwen/Qwen3-0.6B:qwen3-0.6B" \
    "Qwen/Qwen3-1.7B:qwen3-1.7B" \
    "Qwen/Qwen3-4B:qwen3-4B" \
    "gpt2:gpt2" \
    "gpt2-medium:gpt2-medium" \
    "gpt2-large:gpt2-large"
do
    HF_NAME="${ENTRY%%:*}"
    MODEL_TAG="${ENTRY##*:}"

    OUT_DIR="${BASE_OUT_DIR}/${MODEL_TAG}"
    mkdir -p "${OUT_DIR}/ckpts"

    for FRAC in 0.01 0.05 0.10 0.25 0.50 1.00; do
        done_count=$((done_count + 1))
        LOG_FILE="${OUT_DIR}/dataco_ood_frac${FRAC}.log"
        echo "[${done_count}/${total}] ${MODEL_TAG} DataCo_OOD frac=${FRAC}"

        python3 main/cb_main_llm.py \
            --use_gpu 1 --device_id 1 --dataset DataCo_OOD --train_mode 2 \
            --wandb 1 --hf_model_name "${HF_NAME}" --save 1 --ckpt_dir "${OUT_DIR}/ckpts" \
            --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" --otr_reward_coeff 2 \
            --ckpt "${DATACO_CKPT}" \
            > "${LOG_FILE}" 2>&1

        echo "[${done_count}/${total}] ${MODEL_TAG} DataCo_OOD frac=${FRAC} done"
    done
done

echo "All ${total} OOD VocabAlign jobs complete. Logs saved to ${BASE_OUT_DIR}"
echo "Extract results: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*/*.log"
