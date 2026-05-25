#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/latest_output/zero_shot/ood_vocabalign/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

echo "Zero-shot OOD evaluation (VocabAlign, dm_epochs=0) on DataCo_OOD"
echo "Models: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, GPT-2, GPT-2 Medium, GPT-2 Large"
echo "Sequential on GPU 0"

DATACO_CKPT="output/latest_output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"

MODELS=(
    "Qwen/Qwen3-0.6B:qwen3-0.6B"
    "Qwen/Qwen3-1.7B:qwen3-1.7B"
    "Qwen/Qwen3-4B:qwen3-4B"
    "gpt2:gpt2"
    "gpt2-medium:gpt2-medium"
    "gpt2-large:gpt2-large"
)

for ENTRY in "${MODELS[@]}"; do
    HF_NAME="${ENTRY%%:*}"
    MODEL_TAG="${ENTRY##*:}"
    echo "--- ${MODEL_TAG} DataCo_OOD zero-shot ---"

    python3 main/cb_main_llm.py \
        --use_gpu 1 --device_id 1 --dataset DataCo_OOD --train_mode 2 \
        --wandb 1 --hf_model_name "${HF_NAME}" --save 0 \
        --dm_epochs 0 --train_frac 1.0 --otr_reward_coeff 2 \
        --ckpt "${DATACO_CKPT}" \
        > "${BASE_OUT_DIR}/dataco_ood_${MODEL_TAG}.log" 2>&1

    echo "${MODEL_TAG} done"
done

echo "All zero-shot OOD runs complete. Logs saved to ${BASE_OUT_DIR}"
echo "Extract results: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*.log"
