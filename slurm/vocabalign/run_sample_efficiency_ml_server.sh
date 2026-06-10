#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/decision_maker/ml/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

echo "ML baselines (random, historical, rf, xgb) × 4 datasets × fracs"
echo "No GPU required. Logs saved to ${BASE_OUT_DIR}"

DATACO_CKPT="output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"
GS_CKPT="output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
OAS_CKPT="output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"
SCSP_CKPT="${SCSP_CKPT:-output/simulator/latest_run/ckpts/scsp/best.pth}"
SCSP_OTR="${SCSP_OTR:-2}"

# DATASETS=(DataCo GlobalStore OAS SupplyChainShipmentPricing)
DATASETS=(SupplyChainShipmentPricing)
FRACS=(0.01 0.05 0.10 0.25 0.50 1.00)
BASELINES=(rf xgb random historical)

total=$(( ${#BASELINES[@]} * ${#FRACS[@]} * ${#DATASETS[@]} ))
done_count=0
echo "Total jobs: ${total}"

for BASELINE in "${BASELINES[@]}"; do
    OUT_DIR="${BASE_OUT_DIR}/${BASELINE}"
    mkdir -p "${OUT_DIR}"

    for FRAC in "${FRACS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            done_count=$((done_count + 1))

            case "${DATASET}" in
                DataCo)      CKPT="${DATACO_CKPT}"; OTR=2 ;;
                GlobalStore) CKPT="${GS_CKPT}";     OTR=10 ;;
                OAS)         CKPT="${OAS_CKPT}";    OTR=50 ;;
                SupplyChainShipmentPricing) CKPT="${SCSP_CKPT}"; OTR="${SCSP_OTR}" ;;
            esac

            DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
            DS_TAG="${DS_LOWER/supplychainshipmentpricing/scsp}"
            LOG_FILE="${OUT_DIR}/${DS_TAG}_frac${FRAC}.log"
            echo "[${done_count}/${total}] ${BASELINE} ${DATASET} frac=${FRAC}"

            python3 main/cb_main_ml.py \
                --baseline "${BASELINE}" \
                --dataset "${DATASET}" \
                --train_frac "${FRAC}" \
                --ckpt "${CKPT}" \
                --otr_reward_coeff "${OTR}" \
                --out_dir "${OUT_DIR}" \
                --use_gpu 1 --device_id 1 \
                > "${LOG_FILE}" 2>&1

            echo "[${done_count}/${total}] ${BASELINE} ${DATASET} frac=${FRAC} done"
        done
    done
done

echo "All ${total} ML baseline jobs complete. Logs → ${BASE_OUT_DIR}"
echo "Extract results: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*/*.log"
