#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/latest_output/sample_efficiency/multiseed/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"
MODEL="Qwen/Qwen3-1.7B"
MODEL_TAG="qwen3-1.7B"

echo "Multi-seed runs — VocabAlign (${MODEL_TAG}) × 3 seeds × all fracs × all datasets"
echo "Seeds: 42 123 456 | dm_epochs=${DM_EPOCHS} | sequential on GPU 0"

DATACO_CKPT="output/latest_output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"
GS_CKPT="output/latest_output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
OAS_CKPT="output/latest_output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"

total=0
done_count=0
for SEED in 42 123 456; do
    for FRAC in 0.01 0.05 0.10 0.25 0.50 1.00; do
        for DATASET in DataCo GlobalStore OAS; do
            total=$((total + 1))
        done
    done
done
echo "Total jobs: ${total}"

for SEED in 42 123 456; do
    SEED_DIR="${BASE_OUT_DIR}/seed${SEED}"
    mkdir -p "${SEED_DIR}/ckpts"

    for FRAC in 0.01 0.05 0.10 0.25 0.50 1.00; do
        for DATASET in DataCo GlobalStore OAS; do
            done_count=$((done_count + 1))
            DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')
            LOG_FILE="${SEED_DIR}/${DS_LOWER}_frac${FRAC}.log"
            echo "[${done_count}/${total}] seed=${SEED} ${DATASET} frac=${FRAC}"

            case "${DATASET}" in
                DataCo)
                    python3 main/cb_main_llm.py \
                        --use_gpu 1 --device_id 1 --dataset DataCo --train_mode 2 \
                        --wandb 1 --hf_model_name "${MODEL}" --save 0 \
                        --seed "${SEED}" --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" \
                        --otr_reward_coeff 2 --ckpt "${DATACO_CKPT}" \
                        > "${LOG_FILE}" 2>&1
                    ;;
                GlobalStore)
                    python3 main/cb_main_llm.py \
                        --use_gpu 1 --device_id 1 --dataset GlobalStore --train_mode 2 \
                        --wandb 1 --hf_model_name "${MODEL}" --save 0 \
                        --seed "${SEED}" --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" \
                        --otr_reward_coeff 10 --ckpt "${GS_CKPT}" \
                        > "${LOG_FILE}" 2>&1
                    ;;
                OAS)
                    python3 main/cb_main_llm.py \
                        --use_gpu 1 --device_id 1 --dataset OAS --train_mode 2 \
                        --wandb 1 --hf_model_name "${MODEL}" --save 0 \
                        --seed "${SEED}" --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" \
                        --dm_lr 0.00003 --otr_reward_coeff 50 --ckpt "${OAS_CKPT}" \
                        > "${LOG_FILE}" 2>&1
                    ;;
            esac

            echo "[${done_count}/${total}] seed=${SEED} ${DATASET} frac=${FRAC} done"
        done
    done
done

echo "All ${total} multi-seed jobs complete. Logs saved to ${BASE_OUT_DIR}"
echo "Each seed dir has {dataset}_frac{frac}.log files matching comparison script format"
