#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/latest_output/sample_efficiency/all_vocabalign/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"
echo "Sample efficiency — all models (vocabalign), dm_epochs=${DM_EPOCHS}"
echo "Fracs: 0.01 0.05 0.10 0.25 0.50 1.00 × {DataCo, GlobalStore, OAS} × 6 models"
echo "4 GPU workers — jobs dispatched dynamically as GPUs free up"

QUEUE_FILE="${BASE_OUT_DIR}/job_queue.txt"
LOCK_FILE="${BASE_OUT_DIR}/queue.lock"
touch "${LOCK_FILE}"

# ── Build job queue ───────────────────────────────────────────────────────────
# Format: HF_NAME MODEL_TAG FRAC DATASET
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
    for FRAC in 0.01 0.05 0.10 0.25 0.50 1.00; do
        for DATASET in DataCo GlobalStore OAS; do
            echo "${HF_NAME} ${MODEL_TAG} ${FRAC} ${DATASET}" >> "${QUEUE_FILE}"
        done
    done
done

TOTAL=$(wc -l < "${QUEUE_FILE}")
echo "Total jobs: ${TOTAL}"

# ── Atomic job pickup ─────────────────────────────────────────────────────────
next_job() {
    (
        flock -x 200
        JOB=$(head -1 "${QUEUE_FILE}" 2>/dev/null)
        if [ -n "${JOB}" ]; then
            sed -i '1d' "${QUEUE_FILE}"
        fi
        echo "${JOB}"
    ) 200>"${LOCK_FILE}"
}

# ── Worker: loops until queue is empty ───────────────────────────────────────
worker() {
    local GPU=$1
    echo "[GPU${GPU}] worker started"

    while true; do
        JOB=$(next_job)
        [ -z "${JOB}" ] && break

        read -r HF_NAME MODEL_TAG FRAC DATASET <<< "${JOB}"
        OUT_DIR="${BASE_OUT_DIR}/${MODEL_TAG}"
        mkdir -p "${OUT_DIR}/ckpts"

        LOG_FILE="${OUT_DIR}/$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')_frac${FRAC}.log"

        echo "[GPU${GPU}] ${MODEL_TAG} ${DATASET} frac=${FRAC}"

        case "${DATASET}" in
            DataCo)
                python3 main/cb_main_llm_attnpool_vocabalign.py \
                    --use_gpu 1 --device_id "${GPU}" --dataset DataCo --train_mode 2 \
                    --wandb 1 --hf_model_name "${HF_NAME}" --save 1 --ckpt_dir "${OUT_DIR}/ckpts" \
                    --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" --otr_reward_coeff 2 \
                    --ckpt exp_report/DataCo/ckpt/mythical-commander-352_epoch218.pth \
                    > "${LOG_FILE}" 2>&1
                ;;
            GlobalStore)
                python3 main/cb_main_llm_attnpool_vocabalign.py \
                    --use_gpu 1 --device_id "${GPU}" --dataset GlobalStore --train_mode 2 \
                    --wandb 1 --hf_model_name "${HF_NAME}" --save 1 --ckpt_dir "${OUT_DIR}/ckpts" \
                    --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" --otr_reward_coeff 10 \
                    --ckpt exp_report/GlobalStore/ckpt/legendary-federation-353_epoch202.pth \
                    > "${LOG_FILE}" 2>&1
                ;;
            OAS)
                python3 main/cb_main_llm_attnpool_vocabalign.py \
                    --use_gpu 1 --device_id "${GPU}" --dataset OAS --train_mode 2 \
                    --wandb 1 --hf_model_name "${HF_NAME}" --save 1 --ckpt_dir "${OUT_DIR}/ckpts" \
                    --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" --dm_lr 0.00003 --otr_reward_coeff 50 \
                    --ckpt exp_report/OAS/ckpt/jedi-carrier-353_epoch19.pth \
                    > "${LOG_FILE}" 2>&1
                ;;
        esac

        echo "[GPU${GPU}] ${MODEL_TAG} ${DATASET} frac=${FRAC} done"
    done

    echo "[GPU${GPU}] worker finished — queue empty"
}

# ── Launch one worker per GPU ─────────────────────────────────────────────────
for GPU in 0 1 2 3; do
    worker "${GPU}" &
done

wait
echo "All ${TOTAL} jobs complete. Logs saved to ${BASE_OUT_DIR}"
