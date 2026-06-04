#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/decision_maker/ml/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

echo "ML baselines (random, historical, rf, xgb) × {DataCo, GlobalStore, OAS} × fracs"
echo "No GPU required. Logs saved to ${BASE_OUT_DIR}"

DATACO_CKPT="output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"
GS_CKPT="output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
OAS_CKPT="output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"

total=0
done_count=0

for BASELINE in rf xgb random historical; do
    for FRAC in 0.01 0.05 0.10 0.25 0.50 1.00; do
        for DATASET in DataCo GlobalStore OAS; do
            total=$((total + 1))
        done
    done
done
echo "Total jobs: ${total}"

for BASELINE in rf xgb random historical; do
    OUT_DIR="${BASE_OUT_DIR}/${BASELINE}"
    mkdir -p "${OUT_DIR}"

    for FRAC in 0.01 0.05 0.10 0.25 0.50 1.00; do
        for DATASET in DataCo GlobalStore OAS; do
            done_count=$((done_count + 1))

            case "${DATASET}" in
                DataCo)      CKPT="${DATACO_CKPT}" ;;
                GlobalStore) CKPT="${GS_CKPT}" ;;
                OAS)         CKPT="${OAS_CKPT}" ;;
            esac

            DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')
            LOG_FILE="${OUT_DIR}/${DS_LOWER}_frac${FRAC}.log"
            echo "[${done_count}/${total}] ${BASELINE} ${DATASET} frac=${FRAC}"

            python3 main/cb_main_ml.py \
                --baseline "${BASELINE}" \
                --dataset "${DATASET}" \
                --train_frac "${FRAC}" \
                --ckpt "${CKPT}" \
                --out_dir "${OUT_DIR}" \
                --use_gpu 1 --device_id 0 \
                > "${LOG_FILE}" 2>&1

            echo "[${done_count}/${total}] ${BASELINE} ${DATASET} frac=${FRAC} done"
        done
    done
done

echo "All ${total} ML baseline jobs complete. Logs saved to ${BASE_OUT_DIR}"
echo "Extract results: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*/*.log"
