#!/bin/bash
# Experiment 1 — Observational Matching (circular evaluation rebuttal).
#
# Step 1: Run inference with a trained VocabAlign adapter (dm_epochs=0) and save
#         per-sample test predictions to CSV for each dataset.
# Step 2: Run observational_matching.py to compute ATE ± 95% CI on realized outcomes
#         (ground-truth on_time / days_for_shipping from processed test CSVs).
#
# Backbone: Qwen/Qwen3-1.7B (fixed).
# Adapters loaded from: output/decision_maker/checkpoints/{dataset}/qwen3-1.7B/frac1.00/
# Override adapter paths with env vars if needed:
#   DATACO_ADAPTER, GS_ADAPTER, OAS_ADAPTER, SCSP_ADAPTER
#
# Usage:
#   bash run_exp1_observational_matching.sh
#   bash run_nohup.sh run_exp1_observational_matching.sh

set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/akaush39/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Backbone ──────────────────────────────────────────────────────────────────
MODEL_TAG="qwen3-1.7B"
HF_NAME="Qwen/Qwen3-1.7B"

# ── Fixed simulator checkpoints ───────────────────────────────────────────────
DATACO_SIM="output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"
GS_SIM="output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
OAS_SIM="output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"
SCSP_SIM="${SCSP_SIM:-output/simulator/latest_run/ckpts/scsp/best.pth}"

NUM_GPUS=1
GPUS=(0)
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-42 131 521 1009 2027}"

RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_BASE="output/exp1_observational_matching/${RUN_ID}"
mkdir -p "${OUT_BASE}"

# ── Resolve adapter path for a dataset ───────────────────────────────────────
get_adapter() {
    local dataset="$1"
    local ckpt_ds
    case "${dataset}" in
        SupplyChainShipmentPricing) ckpt_ds="SCSP" ;;
        *) ckpt_ds="${dataset}" ;;
    esac
    find "output/decision_maker/checkpoints/${ckpt_ds}/${MODEL_TAG}/frac1.00" \
        -name "*_attnpool_best.pth" 2>/dev/null | sort | tail -1
}

# ── Lookup adapters (env overrides take priority) ─────────────────────────────
DATACO_ADAPTER="${DATACO_ADAPTER:-$(get_adapter DataCo)}"
GS_ADAPTER="${GS_ADAPTER:-$(get_adapter GlobalStore)}"
OAS_ADAPTER="${OAS_ADAPTER:-$(get_adapter OAS)}"
SCSP_ADAPTER="${SCSP_ADAPTER:-$(get_adapter SupplyChainShipmentPricing)}"

echo "=========================================="
echo " Experiment 1 — Observational Matching (multi-seed)"
echo " Backbone: ${HF_NAME}"
echo " run_id=${RUN_ID}"
echo " Seeds: ${SEEDS[*]}"
echo " Output base → ${OUT_BASE}"
echo "=========================================="
echo ""
echo "── Adapters ──"
echo "  DataCo      : ${DATACO_ADAPTER}"
echo "  GlobalStore : ${GS_ADAPTER}"
echo "  OAS         : ${OAS_ADAPTER}"
echo "  SCSP        : ${SCSP_ADAPTER}"
echo ""

# Abort early if any adapter is missing
for pair in "DataCo:${DATACO_ADAPTER}" "GlobalStore:${GS_ADAPTER}" "OAS:${OAS_ADAPTER}" "SCSP:${SCSP_ADAPTER}"; do
    ds="${pair%%:*}"; ckpt="${pair##*:}"
    if [ -z "${ckpt}" ]; then
        echo "ERROR: no frac1.00 attnpool_best.pth found for ${ds} in checkpoints/${ds}/${MODEL_TAG}/frac1.00/" >&2
        exit 1
    fi
done

declare -A DATASET_SIM=(
    [DataCo]="${DATACO_SIM}"
    [GlobalStore]="${GS_SIM}"
    [OAS]="${OAS_SIM}"
    [SupplyChainShipmentPricing]="${SCSP_SIM}"
)
declare -A DATASET_ADAPTER=(
    [DataCo]="${DATACO_ADAPTER}"
    [GlobalStore]="${GS_ADAPTER}"
    [OAS]="${OAS_ADAPTER}"
    [SupplyChainShipmentPricing]="${SCSP_ADAPTER}"
)
declare -A DATASET_OTR=( [DataCo]=2 [GlobalStore]=10 [OAS]=50 [SupplyChainShipmentPricing]=2 )
declare -A DATASET_LR=( [DataCo]=0.01 [GlobalStore]=0.01 [OAS]=0.00003 [SupplyChainShipmentPricing]=0.01 )

gpu_id="${GPUS[0]}"

for SEED in "${SEEDS[@]}"; do
    OUT_DIR="${OUT_BASE}/seed${SEED}"
    mkdir -p "${OUT_DIR}"
    echo ""
    echo "══ Seed ${SEED} ══"

    # ── Step 1: generate per-sample predictions ───────────────────────────────
    echo "── Step 1: generating per-sample test predictions (seed=${SEED}) ──"

    job_num=0
    PRED_PIDS=()

    for DATASET in DataCo GlobalStore OAS SupplyChainShipmentPricing; do
        job_num=$(( job_num + 1 ))
        DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')
        DS_TAG="${DS_LOWER/supplychainshipmentpricing/scsp}"
        PRED_CSV="${OUT_DIR}/${DS_TAG}_predictions.csv"
        LOG="${OUT_DIR}/${DS_TAG}_inference.log"

        echo "  [${job_num}/4] GPU${gpu_id} → ${DATASET} → ${PRED_CSV}"

        (
            CUDA_VISIBLE_DEVICES="${gpu_id}" python3 main/cb_main_llm.py \
                --use_gpu 1 --device_id 0 \
                --dataset "${DATASET}" \
                --train_mode 2 \
                --wandb 0 \
                --hf_model_name "${HF_NAME}" \
                --save 0 \
                --dm_epochs 0 \
                --seed "${SEED}" \
                --otr_reward_coeff "${DATASET_OTR[${DATASET}]}" \
                --dm_lr "${DATASET_LR[${DATASET}]}" \
                --ckpt "${DATASET_SIM[${DATASET}]}" \
                --value_network_ckpt "${DATASET_ADAPTER[${DATASET}]}" \
                --save_predictions "${PRED_CSV}" \
                > "${LOG}" 2>&1
            rc=$?
            if [ "${rc}" -ne 0 ]; then
                echo "  [FAIL rc=${rc}] ${DATASET} seed=${SEED} — see ${LOG}" >&2
            else
                echo "  [OK] ${DATASET} seed=${SEED} → ${PRED_CSV}"
            fi
        ) &
        PRED_PIDS+=($!)
        sleep 2
    done

    failed=0
    for pid in "${PRED_PIDS[@]}"; do wait "${pid}" || failed=$(( failed + 1 )); done
    [ "${failed}" -gt 0 ] && { echo "Step 1 seed=${SEED}: ${failed} failure(s)" >&2; continue; }
    echo "Step 1 seed=${SEED} complete."

    # ── Step 2: observational matching ───────────────────────────────────────
    echo "── Step 2: observational matching (seed=${SEED}) ──"

    for DATASET in DataCo GlobalStore OAS SupplyChainShipmentPricing; do
        DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')
        DS_TAG="${DS_LOWER/supplychainshipmentpricing/scsp}"
        PRED_CSV="${OUT_DIR}/${DS_TAG}_predictions.csv"
        MATCH_LOG="${OUT_DIR}/${DS_TAG}_matching.log"

        [ ! -f "${PRED_CSV}" ] && { echo "  [SKIP] ${DATASET} seed=${SEED}: no predictions" >&2; continue; }

        echo "  → ${DATASET} seed=${SEED} matching..."
        python3 experiments/observational_matching.py \
            --predictions "${PRED_CSV}" \
            --dataset "${DATASET}" \
            --k 5 \
            2>&1 | tee "${MATCH_LOG}"
    done
done

# ── Step 3: aggregate ATE across seeds ───────────────────────────────────────
echo ""
echo "── Step 3: aggregating ATE across seeds ──"
python3 experiments/observational_matching.py \
    --mode aggregate \
    --seed_dirs "${OUT_BASE}/seed*" \
    --out "${OUT_BASE}/aggregate_ate.csv" \
    2>&1 | tee "${OUT_BASE}/aggregate.log"

echo ""
echo "=========================================="
echo " Experiment 1 complete."
echo " Backbone: ${HF_NAME}"
echo " Per-seed predictions → ${OUT_BASE}/seed*/*_predictions.csv"
echo " Per-seed ATE tables  → ${OUT_BASE}/seed*/*_matching.log"
echo " Aggregate ATE        → ${OUT_BASE}/aggregate_ate.csv"
echo ""
echo " Grep results:"
echo "   grep 'ATE\|on_time\|days_for' ${OUT_BASE}/seed*/*_matching.log"
echo "=========================================="
