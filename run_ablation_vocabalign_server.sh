#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/latest_output/sample_efficiency/ablation/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"
MODEL="Qwen/Qwen3-1.7B"
MODEL_TAG="qwen3-1.7B"

echo "Ablation study — VocabAlign (${MODEL_TAG}) on DataCo × GlobalStore × OAS"
echo "Fracs: 0.01 0.10 1.00 × 4 variants (full, no_vocab_init, mean_pool, hard_labels_only)"
echo "dm_epochs=${DM_EPOCHS}, sequential on GPU 0"

DATACO_CKPT="output/latest_output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"
GS_CKPT="output/latest_output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
OAS_CKPT="output/latest_output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"

total=0
done_count=0
for VARIANT in full no_vocab_init mean_pool hard_labels_only; do
    for FRAC in 0.01 0.10 1.00; do
        for DATASET in DataCo GlobalStore OAS; do
            total=$((total + 1))
        done
    done
done
echo "Total jobs: ${total}"

run_one() {
    local VARIANT="$1" DATASET="$2" FRAC="$3"
    local OUT_DIR="${BASE_OUT_DIR}/${VARIANT}"
    mkdir -p "${OUT_DIR}/ckpts"

    local DS_LOWER
    DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')
    local LOG_FILE="${OUT_DIR}/${DS_LOWER}_frac${FRAC}.log"

    local EXTRA_FLAGS=""
    case "${VARIANT}" in
        no_vocab_init)    EXTRA_FLAGS="--pool_init random" ;;
        mean_pool)        EXTRA_FLAGS="--pool_type mean" ;;
        hard_labels_only) EXTRA_FLAGS="--no_soft_labels" ;;
        full)             EXTRA_FLAGS="" ;;
    esac

    local CKPT OTR_LR
    case "${DATASET}" in
        DataCo)
            CKPT="${DATACO_CKPT}"; OTR=2; LR="" ;;
        GlobalStore)
            CKPT="${GS_CKPT}"; OTR=10; LR="" ;;
        OAS)
            CKPT="${OAS_CKPT}"; OTR=50; LR="--dm_lr 0.00003" ;;
    esac

    # shellcheck disable=SC2086
    python3 main/cb_main_llm.py \
        --use_gpu 1 --device_id 1 --dataset "${DATASET}" --train_mode 2 \
        --wandb 1 --hf_model_name "${MODEL}" --save 0 \
        --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" --otr_reward_coeff "${OTR}" \
        --ckpt "${CKPT}" ${LR} ${EXTRA_FLAGS} \
        > "${LOG_FILE}" 2>&1
}

for VARIANT in full no_vocab_init mean_pool hard_labels_only; do
    for FRAC in 0.01 0.10 1.00; do
        for DATASET in DataCo GlobalStore OAS; do
            done_count=$((done_count + 1))
            echo "[${done_count}/${total}] variant=${VARIANT} ${DATASET} frac=${FRAC}"
            run_one "${VARIANT}" "${DATASET}" "${FRAC}"
            echo "[${done_count}/${total}] done"
        done
    done
done

echo "All ${total} ablation jobs complete. Logs saved to ${BASE_OUT_DIR}"
echo "Extract results: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*/*.log"
