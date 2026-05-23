#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/latest_output/zero_shot/vocabalign/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

echo "Zero-shot evaluation (vocab-aligned) — all models × {DataCo, GlobalStore, OAS}"
echo "Models: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, GPT-2, GPT-2 Medium, GPT-2 Large"
echo "DataCo sequential on GPU 2 | GlobalStore+OAS sequential on GPU 3 (parallel streams)"

MODELS=(
    "Qwen/Qwen3-0.6B:qwen3-0.6B"
    "gpt2:gpt2"
    "Qwen/Qwen3-1.7B:qwen3-1.7B"
    "gpt2-medium:gpt2-medium"
    "Qwen/Qwen3-4B:qwen3-4B"
    "gpt2-large:gpt2-large"
)

# GPU 2 stream: DataCo for all models
(
for ENTRY in "${MODELS[@]}"; do
    HF_NAME="${ENTRY%%:*}"
    MODEL_TAG="${ENTRY##*:}"
    echo "[GPU2] DataCo ${MODEL_TAG}"
    python3 main/cb_main_llm_attnpool_vocabalign.py \
        --use_gpu 1 --device_id 2 --dataset DataCo --train_mode 2 \
        --wandb 1 --hf_model_name "${HF_NAME}" --save 0 \
        --dm_epochs 0 --train_frac 1.0 --otr_reward_coeff 2 \
        --ckpt exp_report/DataCo/ckpt/mythical-commander-352_epoch218.pth \
        > "${BASE_OUT_DIR}/dataco_${MODEL_TAG}.log" 2>&1
    echo "[GPU2] DataCo ${MODEL_TAG} done"
done
) &

# GPU 3 stream: GlobalStore then OAS for all models
(
for ENTRY in "${MODELS[@]}"; do
    HF_NAME="${ENTRY%%:*}"
    MODEL_TAG="${ENTRY##*:}"
    echo "[GPU3] GlobalStore ${MODEL_TAG}"
    python3 main/cb_main_llm_attnpool_vocabalign.py \
        --use_gpu 1 --device_id 3 --dataset GlobalStore --train_mode 2 \
        --wandb 1 --hf_model_name "${HF_NAME}" --save 0 \
        --dm_epochs 0 --train_frac 1.0 --otr_reward_coeff 10 \
        --ckpt exp_report/GlobalStore/ckpt/legendary-federation-353_epoch202.pth \
        > "${BASE_OUT_DIR}/globalstore_${MODEL_TAG}.log" 2>&1
    echo "[GPU3] GlobalStore ${MODEL_TAG} done"

    echo "[GPU3] OAS ${MODEL_TAG}"
    python3 main/cb_main_llm_attnpool_vocabalign.py \
        --use_gpu 1 --device_id 3 --dataset OAS --train_mode 2 \
        --wandb 1 --hf_model_name "${HF_NAME}" --save 0 \
        --dm_epochs 0 --train_frac 1.0 --dm_lr 0.00003 --otr_reward_coeff 50 \
        --ckpt exp_report/OAS/ckpt/jedi-carrier-353_epoch19.pth \
        > "${BASE_OUT_DIR}/oas_${MODEL_TAG}.log" 2>&1
    echo "[GPU3] OAS ${MODEL_TAG} done"
done
) &

wait

echo "All zero-shot runs complete. Logs saved to ${BASE_OUT_DIR}"
echo "Extract results: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*.log"
