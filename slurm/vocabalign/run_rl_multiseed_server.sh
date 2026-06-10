#!/bin/bash
# Multi-seed RL baseline sample-efficiency sweep (Experiment 1 companion).
# Matrix: 4 datasets × 6 fracs × 5 seeds = 120 jobs.
# Zero-shot (frac=0) is handled separately by run_zeroshot_all_server.sh.
# ValueNetwork is a small MLP — all jobs run in parallel on 1 GPU.
#
# Env overrides:
#   SEEDS     — space-separated seed list (default: 42 0 1 2 3)
#   SCSP_CKPT — path to SupplyChainShipmentPricing simulator checkpoint
#   SCSP_OTR  — otr_reward_coeff for SupplyChainShipmentPricing (default: 2)
#
# Usage:
#   bash run_rl_multiseed_server.sh [DM_EPOCHS]
#   bash run_nohup.sh run_rl_multiseed_server.sh

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
MODEL_TAG="rl"
LOG_DIR="output/decision_maker/${MODEL_TAG}/${RUN_ID}"
mkdir -p "${LOG_DIR}"

NUM_GPUS=1
GPUS=(1)
DM_EPOCHS="${1:-${DM_EPOCHS:-6000}}"
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-42 131 521 1009 2027}"
SCSP_CKPT="${SCSP_CKPT:-output/simulator/latest_run/ckpts/scsp/best.pth}"
SCSP_OTR="${SCSP_OTR:-2}"

DATACO_CKPT="output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"
GS_CKPT="output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
OAS_CKPT="output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"

FRACS=(0.01 0.05 0.10 0.25 0.50 1.00)
DATASETS=(DataCo GlobalStore OAS SupplyChainShipmentPricing)

total=$(( ${#DATASETS[@]} * ${#FRACS[@]} * ${#SEEDS[@]} ))

echo "=========================================="
echo " Sample efficiency — RL baseline (multi-seed, sequential)"
echo " dm_epochs=${DM_EPOCHS}  run_id=${RUN_ID}"
echo " Seeds: ${SEEDS[*]}"
echo " GPU: ${GPUS[*]}"
echo " Jobs: ${#DATASETS[@]} datasets × ${#FRACS[@]} fracs × ${#SEEDS[@]} seeds = ${total}"
echo " Execution: sequential — one job at a time to avoid OOM"
echo " (Zero-shot frac=0 handled by run_zeroshot_all_server.sh)"
echo "=========================================="

failed=0
job_num=0

for SEED in "${SEEDS[@]}"; do
    for FRAC in "${FRACS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            GPU_ID="${GPUS[$(( job_num % NUM_GPUS ))]}"
            DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
            # Use short tag for output paths only; --dataset arg stays as the full name
            DS_TAG="${DS_LOWER/supplychainshipmentpricing/scsp}"
            SEED_LOG_DIR="${LOG_DIR}/seed${SEED}"
            mkdir -p "${SEED_LOG_DIR}"
            LOG="${SEED_LOG_DIR}/${DS_TAG}_frac${FRAC}.log"
            CKPT_DIR="output/decision_maker/${DS_TAG}/ckpts/frac${FRAC}/${MODEL_TAG}/seed${SEED}"
            mkdir -p "${CKPT_DIR}"

            case "${DATASET}" in
                DataCo)      CKPT="${DATACO_CKPT}"; OTR=2  ;;
                GlobalStore) CKPT="${GS_CKPT}";     OTR=10 ;;
                OAS)         CKPT="${OAS_CKPT}";    OTR=50 ;;
                SupplyChainShipmentPricing) CKPT="${SCSP_CKPT}"; OTR="${SCSP_OTR}" ;;
            esac

            job_num=$(( job_num + 1 ))
            echo "  [${job_num}/${total}] GPU${GPU_ID} → RL  ${DATASET}  frac=${FRAC}  seed=${SEED}  $(date '+%H:%M:%S')"

            CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 main/cb_main.py \
                --use_gpu 1 --device_id 0 \
                --dataset "${DATASET}" --train_mode 2 \
                --wandb 0 --save 1 --ckpt_dir "${CKPT_DIR}" \
                --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" \
                --seed "${SEED}" \
                --otr_reward_coeff "${OTR}" \
                --ckpt "${CKPT}" \
                > "${LOG}" 2>&1
            rc=$?
            if [ "${rc}" -ne 0 ]; then
                echo "  [FAIL rc=${rc}] RL ${DATASET} frac=${FRAC} seed=${SEED}" >&2
                failed=$(( failed + 1 ))
            fi
        done
    done
done

echo ""
if [ "${failed}" -gt 0 ]; then
    echo "Finished with ${failed} failure(s) — check logs in ${LOG_DIR}/" >&2
    exit 1
fi
echo "=========================================="
echo " All ${total} RL jobs complete."
echo " Logs → ${LOG_DIR}/seed*/{dataset}_frac*.log"
echo " Extract: grep 'best_profit\|best_on_time' ${LOG_DIR}/seed*/*.log"
echo "=========================================="
