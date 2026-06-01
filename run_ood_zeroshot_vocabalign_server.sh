#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="/output/latest_output/zero_shot/ood_vocabalign/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

echo "Zero-shot OOD evaluation (VocabAlign, dm_epochs=0) on DataCo_OOD"
echo "Models: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, GPT-2, GPT-2 Medium, GPT-2 Large"
echo "Parallel execution across GPUs 0-3 (round-robin assignment)"

DATACO_CKPT="/output/latest_output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"

MODELS=(
    "Qwen/Qwen3-0.6B:qwen3-0.6B"
    "Qwen/Qwen3-1.7B:qwen3-1.7B"
    "Qwen/Qwen3-4B:qwen3-4B"
    "gpt2:gpt2"
    "gpt2-medium:gpt2-medium"
    "gpt2-large:gpt2-large"
)

# GPU assignment (round-robin across 4 GPUs):
#   GPU 0: Qwen3-0.6B, gpt2-medium
#   GPU 1: Qwen3-1.7B, gpt2-large
#   GPU 2: Qwen3-4B
#   GPU 3: gpt2
NUM_GPUS=4
PIDS=()
GPU_ASSIGNMENTS=()

for i in "${!MODELS[@]}"; do
    ENTRY="${MODELS[$i]}"
    HF_NAME="${ENTRY%%:*}"
    MODEL_TAG="${ENTRY##*:}"
    GPU_ID=$((i % NUM_GPUS))
    GPU_ASSIGNMENTS+=("${GPU_ID}")

    echo "--- Launching ${MODEL_TAG} on GPU ${GPU_ID} ---"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 main/cb_main_llm.py \
        --use_gpu 1 --device_id 0 --dataset DataCo_OOD --train_mode 2 \
        --wandb 1 --hf_model_name "${HF_NAME}" --save 0 \
        --dm_epochs 0 --train_frac 1.0 --otr_reward_coeff 2 \
        --ckpt "${DATACO_CKPT}" \
        > "${BASE_OUT_DIR}/dataco_ood_${MODEL_TAG}.log" 2>&1 &
    PIDS+=($!)
done

echo "Waiting for all ${#PIDS[@]} jobs to complete..."
FAILED=0
for i in "${!PIDS[@]}"; do
    ENTRY="${MODELS[$i]}"
    MODEL_TAG="${ENTRY##*:}"
    GPU_ID="${GPU_ASSIGNMENTS[$i]}"
    if wait "${PIDS[$i]}"; then
        echo "${MODEL_TAG} (GPU ${GPU_ID}) done"
    else
        echo "ERROR: ${MODEL_TAG} (GPU ${GPU_ID}) failed — see ${BASE_OUT_DIR}/dataco_ood_${MODEL_TAG}.log"
        FAILED=$((FAILED + 1))
    fi
done

if [ "${FAILED}" -gt 0 ]; then
    echo "${FAILED} job(s) failed. Check logs in ${BASE_OUT_DIR}"
    exit 1
fi

echo "All zero-shot OOD runs complete. Logs saved to ${BASE_OUT_DIR}"
echo "Extract results: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*.log"
