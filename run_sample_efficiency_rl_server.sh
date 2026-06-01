#!/bin/bash
# Parallel sample-efficiency sweep — RL baseline (ValueNetwork).
# ValueNetwork is a small MLP so all 18 jobs (6 fracs × 3 datasets) run
# fully in parallel, distributed round-robin across 4 GPUs.
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
MODEL_TAG="rl"
LOG_DIR="output/decision_maker/${MODEL_TAG}/${RUN_ID}"
mkdir -p "${LOG_DIR}"

NUM_GPUS=2
GPUS=(1 2)
DM_EPOCHS="${1:-${DM_EPOCHS:-6000}}"

echo "=========================================="
echo " Sample efficiency — RL baseline (ValueNetwork)"
echo " dm_epochs=${DM_EPOCHS}  run_id=${RUN_ID}"
echo " GPUs: 1, 2 (RTX 6000 Ada, round-robin)"
echo " Jobs: 6 fracs × 3 datasets = 18 (all parallel)"
echo "=========================================="

DATACO_CKPT="output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"
GS_CKPT="output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
OAS_CKPT="output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"

PIDS=()
LABELS=()
job_num=0

for FRAC in 0.01 0.05 0.10 0.25 0.50 1.00; do
    for DATASET in DataCo GlobalStore OAS; do
        GPU_ID="${GPUS[$(( job_num % NUM_GPUS ))]}"
        DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')
        LOG="${LOG_DIR}/${DS_LOWER}_frac${FRAC}.log"
        CKPT_DIR="output/decision_maker/${DATASET}/ckpts/frac${FRAC}/${MODEL_TAG}"
        mkdir -p "${CKPT_DIR}"

        case "${DATASET}" in
            DataCo)      CKPT="${DATACO_CKPT}"; OTR=2  ;;
            GlobalStore) CKPT="${GS_CKPT}";     OTR=10 ;;
            OAS)         CKPT="${OAS_CKPT}";    OTR=50 ;;
        esac

        echo "  [$(( job_num + 1 ))/18] GPU${GPU_ID} → RL  ${DATASET}  frac=${FRAC}"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 main/cb_main.py \
            --use_gpu 1 --device_id 0 \
            --dataset "${DATASET}" --train_mode 2 \
            --wandb 1 --save 1 --ckpt_dir "${CKPT_DIR}" \
            --dm_epochs "${DM_EPOCHS}" --train_frac "${FRAC}" --otr_reward_coeff "${OTR}" \
            --ckpt "${CKPT}" \
            > "${LOG}" 2>&1 &

        PIDS+=($!)
        LABELS+=("RL ${DATASET} frac=${FRAC}")
        job_num=$(( job_num + 1 ))
    done
done

echo "Waiting for all ${#PIDS[@]} jobs to complete..."
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  ✓ ${LABELS[$i]}"
    else
        echo "  ✗ ${LABELS[$i]} FAILED — see ${LOG_DIR}"
        FAILED=$(( FAILED + 1 ))
    fi
done

if [ "${FAILED}" -gt 0 ]; then
    echo "${FAILED} job(s) failed. Check logs in ${LOG_DIR}"
    exit 1
fi

echo ""
echo "=========================================="
echo " All 18 RL jobs complete."
echo " Logs        → ${LOG_DIR}"
echo " Checkpoints → output/decision_maker/{DataCo,GlobalStore,OAS}/ckpts/frac*/${MODEL_TAG}/"
echo " Extract:      grep 'best_profit\|best_on_time' ${LOG_DIR}/*.log"
echo "=========================================="
