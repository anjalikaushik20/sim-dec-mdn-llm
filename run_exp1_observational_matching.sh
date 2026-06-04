#!/bin/bash
# Experiment 1 — Observational Matching (circular evaluation rebuttal).
#
# Step 1: Run inference with a trained VocabAlign adapter (dm_epochs=0) and save
#         per-sample test predictions to CSV for each dataset.
# Step 2: Run observational_matching.py to compute ATE ± 95% CI on realized outcomes
#         (ground-truth on_time / days_for_shipping from processed test CSVs).
#
# Adapter auto-detection: the script finds the best available frac1.00 adapter for
# each dataset by model priority (qwen3-1.7B > qwen3-0.6B > gpt2-large > gpt2-medium > gpt2)
# and infers the matching HF backbone name from the adapter directory name.
# Override with env vars if needed:
#   DATACO_ADAPTER, DATACO_HF_MODEL
#   GS_ADAPTER,     GS_HF_MODEL
#   OAS_ADAPTER,    OAS_HF_MODEL
#
# Usage (no args needed if adapters exist in output/):
#   bash run_exp1_observational_matching.sh
#   bash run_nohup.sh run_exp1_observational_matching.sh

set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Fixed simulator checkpoints ───────────────────────────────────────────────
DATACO_SIM="output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"
GS_SIM="output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
OAS_SIM="output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"

NUM_GPUS=1
GPUS=(0)

RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_DIR="output/exp1_observational_matching/${RUN_ID}"
mkdir -p "${OUT_DIR}"

# ── Model-tag → HuggingFace name ─────────────────────────────────────────────
model_to_hf() {
    case "$1" in
        qwen3-1.7B)  echo "Qwen/Qwen3-1.7B"  ;;
        qwen3-0.6B)  echo "Qwen/Qwen3-0.6B"  ;;
        qwen3-4B)    echo "Qwen/Qwen3-4B"    ;;
        gpt2-large)  echo "gpt2-large"        ;;
        gpt2-medium) echo "gpt2-medium"       ;;
        gpt2)        echo "gpt2"              ;;
        *)           echo ""                  ;;
    esac
}

# ── Auto-detect best frac1.00 adapter for a dataset ──────────────────────────
# Searches output/decision_maker/*/{DATASET}/ckpts/frac1.00/{model}/*.pth
# Returns "<path>|<model_tag>" for the highest-priority model found.
find_adapter() {
    local dataset="$1"
    for model_tag in qwen3-1.7B qwen3-0.6B qwen3-4B gpt2-large gpt2-medium gpt2; do
        local p
        p=$(find output/decision_maker -path "*/frac1.00/${model_tag}/*_attnpool_best.pth" \
                2>/dev/null \
            | grep -i "/${dataset}/" \
            | sort | tail -1)
        if [ -n "$p" ]; then
            echo "${p}|${model_tag}"
            return 0
        fi
    done
    echo ""
}

# ── Resolve adapter + HF model per dataset ───────────────────────────────────
resolve() {
    local dataset="$1" adapter_var="$2" hf_var="$3"
    local adapter="${!adapter_var}"
    local hf="${!hf_var}"

    if [ -z "$adapter" ]; then
        local detected
        detected=$(find_adapter "${dataset}")
        if [ -z "$detected" ]; then
            echo "ERROR: no frac1.00 attnpool_best.pth found for ${dataset} under output/decision_maker/" >&2
            echo "  Train first with run_sample_efficiency_all_vocabalign_parallel.sh, or set ${adapter_var}." >&2
            exit 1
        fi
        adapter="${detected%%|*}"
        local tag="${detected##*|}"
        if [ -z "$hf" ]; then
            hf=$(model_to_hf "${tag}")
        fi
        echo "  [auto] ${dataset}: ${adapter} (${hf})"
    else
        if [ -z "$hf" ]; then
            # Infer HF model from path directory name
            local tag
            tag=$(basename "$(dirname "${adapter}")")
            hf=$(model_to_hf "${tag}")
            [ -z "$hf" ] && hf="Qwen/Qwen3-1.7B"
        fi
        echo "  [override] ${dataset}: ${adapter} (${hf})"
    fi

    # Export back to caller via two global vars
    _RESOLVED_ADAPTER="${adapter}"
    _RESOLVED_HF="${hf}"
}

echo "=========================================="
echo " Experiment 1 — Observational Matching"
echo " run_id=${RUN_ID}"
echo " Output → ${OUT_DIR}"
echo "=========================================="
echo ""
echo "── Resolving adapters ──"

resolve DataCo    DATACO_ADAPTER DATACO_HF_MODEL
DATACO_ADAPTER="${_RESOLVED_ADAPTER}"; DATACO_HF="${_RESOLVED_HF}"

resolve GlobalStore GS_ADAPTER GS_HF_MODEL
GS_ADAPTER="${_RESOLVED_ADAPTER}"; GS_HF="${_RESOLVED_HF}"

resolve OAS OAS_ADAPTER OAS_HF_MODEL
OAS_ADAPTER="${_RESOLVED_ADAPTER}"; OAS_HF="${_RESOLVED_HF}"

echo ""

# ── Step 1: generate per-sample predictions (sequential — 1 GPU) ─────────────
echo "── Step 1: generating per-sample test predictions ──"

declare -A DATASET_SIM=( [DataCo]="${DATACO_SIM}" [GlobalStore]="${GS_SIM}" [OAS]="${OAS_SIM}" )
declare -A DATASET_ADAPTER=( [DataCo]="${DATACO_ADAPTER}" [GlobalStore]="${GS_ADAPTER}" [OAS]="${OAS_ADAPTER}" )
declare -A DATASET_HF=( [DataCo]="${DATACO_HF}" [GlobalStore]="${GS_HF}" [OAS]="${OAS_HF}" )
declare -A DATASET_OTR=( [DataCo]=2 [GlobalStore]=10 [OAS]=50 )
declare -A DATASET_LR=( [DataCo]=0.01 [GlobalStore]=0.01 [OAS]=0.00003 )

gpu_id="${GPUS[0]}"
job_num=0

PRED_PIDS=()

for DATASET in DataCo GlobalStore OAS; do
    job_num=$(( job_num + 1 ))
    DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')
    PRED_CSV="${OUT_DIR}/${DS_LOWER}_predictions.csv"
    LOG="${OUT_DIR}/${DS_LOWER}_inference.log"

    echo "  [${job_num}/3] GPU${gpu_id} → ${DATASET} (${DATASET_HF[${DATASET}]}) → ${PRED_CSV}"

    (
        CUDA_VISIBLE_DEVICES="${gpu_id}" python3 main/cb_main_llm.py \
            --use_gpu 1 --device_id 0 \
            --dataset "${DATASET}" \
            --train_mode 2 \
            --wandb 0 \
            --hf_model_name "${DATASET_HF[${DATASET}]}" \
            --save 0 \
            --dm_epochs 0 \
            --otr_reward_coeff "${DATASET_OTR[${DATASET}]}" \
            --dm_lr "${DATASET_LR[${DATASET}]}" \
            --ckpt "${DATASET_SIM[${DATASET}]}" \
            --value_network_ckpt "${DATASET_ADAPTER[${DATASET}]}" \
            --save_predictions "${PRED_CSV}" \
            > "${LOG}" 2>&1
        rc=$?
        if [ "${rc}" -ne 0 ]; then
            echo "  [FAIL rc=${rc}] ${DATASET} inference — see ${LOG}" >&2
        else
            echo "  [OK] ${DATASET} → ${PRED_CSV}"
        fi
    ) &
    PRED_PIDS+=($!)

    sleep 2  # stagger HuggingFace model load
done

failed=0
for pid in "${PRED_PIDS[@]}"; do
    wait "${pid}" || failed=$(( failed + 1 ))
done

if [ "${failed}" -gt 0 ]; then
    echo "Step 1 finished with ${failed} failure(s) — check logs in ${OUT_DIR}/" >&2
    exit 1
fi
echo "Step 1 complete — all predictions generated."

# ── Step 2: observational matching per dataset ───────────────────────────────
echo ""
echo "── Step 2: observational matching ──"

for DATASET in DataCo GlobalStore OAS; do
    DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')
    PRED_CSV="${OUT_DIR}/${DS_LOWER}_predictions.csv"
    MATCH_LOG="${OUT_DIR}/${DS_LOWER}_matching.log"

    if [ ! -f "${PRED_CSV}" ]; then
        echo "  [SKIP] ${DATASET}: predictions file not found (${PRED_CSV})" >&2
        continue
    fi

    echo "  → ${DATASET} matching..."
    python3 experiments/observational_matching.py \
        --predictions "${PRED_CSV}" \
        --dataset "${DATASET}" \
        --k 5 \
        2>&1 | tee "${MATCH_LOG}"
done

echo ""
echo "=========================================="
echo " Experiment 1 complete."
echo " Predictions → ${OUT_DIR}/*_predictions.csv"
echo " ATE tables  → ${OUT_DIR}/*_matching.log"
echo ""
echo " Grep results:"
echo "   grep 'ATE\|on_time\|days_for' ${OUT_DIR}/*_matching.log"
echo "=========================================="
