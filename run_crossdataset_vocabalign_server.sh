#!/bin/bash
# Eval-only cross-dataset: use existing DataCo frac=1.00 adapters to evaluate
# zero-shot on GlobalStore and OAS. No training — adapters are loaded directly
# from output/decision_maker/DataCo/ckpts/frac1.00/{model_tag}/.
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/decision_maker/cross_dataset/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

echo "Cross-dataset eval — DataCo frac=1.00 adapters → GlobalStore + OAS"
echo "GPU: 1"

DATACO_CKPT_BASE="output/decision_maker/DataCo/ckpts/frac1.00"
GLOBALSTORE_CKPT="output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
OAS_CKPT="output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"

MODELS=(
    "Qwen/Qwen3-0.6B:qwen3-0.6B"
    "Qwen/Qwen3-1.7B:qwen3-1.7B"
    "Qwen/Qwen3-4B:qwen3-4B"
    "gpt2:gpt2"
    "gpt2-medium:gpt2-medium"
    "gpt2-large:gpt2-large"
)

EVAL_DATASETS=("GlobalStore" "OAS")
EVAL_CKPTS=("${GLOBALSTORE_CKPT}" "${OAS_CKPT}")
EVAL_OTRS=(10 50)
EVAL_LRS=("" "--dm_lr 0.00003")

for ENTRY in "${MODELS[@]}"; do
    HF_NAME="${ENTRY%%:*}"
    MODEL_TAG="${ENTRY##*:}"
    OUT_DIR="${BASE_OUT_DIR}/${MODEL_TAG}"
    mkdir -p "${OUT_DIR}"

    # Find the adapter for this model at frac=1.00
    ADAPTER_DIR="${DATACO_CKPT_BASE}/${MODEL_TAG}"
    if [ ! -d "${ADAPTER_DIR}" ]; then
        echo "  SKIP ${MODEL_TAG}: no checkpoint dir at ${ADAPTER_DIR}"
        continue
    fi

    ADAPTER_PATH=$(find "${ADAPTER_DIR}" -name "*_attnpool_best.pth" | sort | tail -1)
    if [ -z "${ADAPTER_PATH}" ]; then
        echo "  SKIP ${MODEL_TAG}: no *_attnpool_best.pth found in ${ADAPTER_DIR}"
        continue
    fi

    echo "--- ${MODEL_TAG} | adapter: ${ADAPTER_PATH} ---"

    for i in "${!EVAL_DATASETS[@]}"; do
        EVAL_DS="${EVAL_DATASETS[$i]}"
        EVAL_CKPT="${EVAL_CKPTS[$i]}"
        EVAL_OTR="${EVAL_OTRS[$i]}"
        EVAL_LR="${EVAL_LRS[$i]}"
        EVAL_LOG="${OUT_DIR}/eval_${EVAL_DS,,}.log"
        echo "  EVAL on ${EVAL_DS}"

        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=1 python3 main/cb_main_llm.py \
            --use_gpu 1 --device_id 0 \
            --dataset "${EVAL_DS}" --train_mode 2 \
            --wandb 1 --hf_model_name "${HF_NAME}" --save 0 \
            --dm_epochs 0 --train_frac 1.0 --otr_reward_coeff "${EVAL_OTR}" \
            --ckpt "${EVAL_CKPT}" \
            --value_network_ckpt "${ADAPTER_PATH}" \
            ${EVAL_LR} \
            > "${EVAL_LOG}" 2>&1

        echo "  EVAL on ${EVAL_DS} done"
    done

    echo "${MODEL_TAG} complete"
done

echo ""
echo "All cross-dataset evals complete. Logs in ${BASE_OUT_DIR}"
echo "Extract results: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*/*.log"
