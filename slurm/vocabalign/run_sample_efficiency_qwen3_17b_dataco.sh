#!/bin/bash
# Sample-efficiency sweep for Qwen/Qwen3-1.7B — DataCo only (GPU 0).
# Part of 3-GPU parallelism split (DataCo | OAS+SCSP | GlobalStore).
#
# 1 dataset × 6 fracs × 5 seeds = 30 jobs

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/akaush39/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/decision_maker/all_fracs/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"
GPU_ID=0
MODEL_TAG="qwen3-1.7B"
HF_NAME="Qwen/Qwen3-1.7B"
SEEDS=(42 131 521 1009 2027)
FRACS=(0.01 0.05 0.10 0.25 0.50 1.00)
DATASETS=(DataCo)

DATACO_CKPT="output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"

total=$(( ${#DATASETS[@]} * ${#FRACS[@]} * ${#SEEDS[@]} ))
echo "=========================================="
echo " Sample efficiency — ${MODEL_TAG} (DataCo)"
echo " dm_epochs=${DM_EPOCHS}  run_id=${RUN_ID}  GPU=${GPU_ID}"
echo " Seeds: ${SEEDS[*]}"
echo " Total jobs: ${total}"
echo "=========================================="

job_num=0
failed=0

echo ""
echo "┌─ ${MODEL_TAG}  (sequential, GPU${GPU_ID})  $(date '+%Y-%m-%d %H:%M:%S') ─────────"

for SEED in "${SEEDS[@]}"; do
    for FRAC in "${FRACS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            job_num=$(( job_num + 1 ))
            DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
            log_dir="${BASE_OUT_DIR}/seed${SEED}/${MODEL_TAG}"
            mkdir -p "${log_dir}"
            log="${log_dir}/${DS_LOWER}_frac${FRAC}.log"
            ckpt_dir="output/decision_maker/all_fracs/${DATASET}/ckpts/frac${FRAC}/${MODEL_TAG}/seed${SEED}"
            mkdir -p "${ckpt_dir}"

            echo "  [${job_num}/${total}] GPU${GPU_ID} → ${DATASET}  frac=${FRAC}  seed=${SEED}  $(date '+%H:%M:%S')"

            CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 main/cb_main_llm.py \
                --use_gpu 1 --device_id 0 \
                --dataset "${DATASET}" \
                --train_mode 2 \
                --wandb 0 \
                --hf_model_name "${HF_NAME}" \
                --save 1 \
                --ckpt_dir "${ckpt_dir}" \
                --dm_epochs "${DM_EPOCHS}" \
                --train_frac "${FRAC}" \
                --seed "${SEED}" \
                --otr_reward_coeff 2 --ckpt "${DATACO_CKPT}" \
                > "${log}" 2>&1
            rc=$?
            [ "${rc}" -ne 0 ] && { echo "  [FAIL rc=${rc}] ${DATASET} frac=${FRAC} seed=${SEED}" >&2; failed=$(( failed + 1 )); }
        done
    done
done

echo "└─ ${MODEL_TAG} DONE (${failed} failures)  $(date '+%H:%M:%S')"
echo ""
echo "=========================================="
echo " 30 jobs complete. Logs → ${BASE_OUT_DIR}"
echo " Extract: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/seed*/${MODEL_TAG}/*.log"
echo "=========================================="
