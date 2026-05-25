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
echo "All runs sequential on GPU 0"

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
        for DATASET in DataCo GlobalStore OAS; do
            total=$((total + 1))
        done
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
        for DATASET in DataCo GlobalStore OAS; do
            done_count=$((done_count + 1))
            LOG_FILE="${OUT_DIR}/$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')_frac${FRAC}.log"
            echo "[${done_count}/${total}] ${MODEL_TAG} ${DATASET} frac=${FRAC}"

            case "${DATASET}" in
                DataCo)
                    python3 main/cb_main_llm.py \
                        --use_gpu 1 --device_id 1 --dataset DataCo --train_mode 2 \
                        --wandb 1 --hf_model_name "${HF_NAME}" --save 1 --ckpt_dir "${OUT_DIR}/ckpts" \
                        --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" --otr_reward_coeff 2 \
                        --ckpt output/latest_output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth \
                        > "${LOG_FILE}" 2>&1
                    ;;
                GlobalStore)
                    python3 main/cb_main_llm.py \
                        --use_gpu 1 --device_id 1 --dataset GlobalStore --train_mode 2 \
                        --wandb 1 --hf_model_name "${HF_NAME}" --save 1 --ckpt_dir "${OUT_DIR}/ckpts" \
                        --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" --otr_reward_coeff 10 \
                        --ckpt output/latest_output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth \
                        > "${LOG_FILE}" 2>&1
                    ;;
                OAS)
                    python3 main/cb_main_llm.py \
                        --use_gpu 1 --device_id 1 --dataset OAS --train_mode 2 \
                        --wandb 1 --hf_model_name "${HF_NAME}" --save 1 --ckpt_dir "${OUT_DIR}/ckpts" \
                        --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" --dm_lr 0.00003 --otr_reward_coeff 50 \
                        --ckpt output/latest_output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth \
                        > "${LOG_FILE}" 2>&1
                    ;;
            esac

            echo "[${done_count}/${total}] ${MODEL_TAG} ${DATASET} frac=${FRAC} done"
        done
    done
done

echo "All ${total} jobs complete. Logs saved to ${BASE_OUT_DIR}"
