#!/bin/bash
# Experiment 6 — Prompt ablation.
#
# Tests whether natural-language serialization actually contributes, or whether
# the LLM is doing numeric pattern matching.
#
# Matrix: 4 prompt variants × 2 backbones × 2 datasets × 5 seeds, frac=1.00.
# Variants:
#   natural        — existing NL serialization (default VocabAlign)
#   numeric        — feature values only, no names
#   shuffled_names — key=value pairs with feature names randomly permuted per row
#   names_only     — feature names without values (same text for all rows)
#
# Backbones: gpt2, Qwen/Qwen3-1.7B  [phi4-mini-reasoning commented out]
# Datasets:  GlobalStore, SupplyChainShipmentPricing
# Total: 4 × 2 × 2 × 5 = 80 jobs  (160 if phi4 uncommented)
#
# Env overrides:
#   SEEDS — space-separated seed list (default: 42 0 1 2 3)
#
# Usage:
#   bash run_prompt_ablation_server.sh [DM_EPOCHS]
#   bash run_nohup.sh run_prompt_ablation_server.sh

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/decision_maker/prompt_ablation/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

NUM_GPUS=2
GPUS=(1 3)
DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-42 131 521 1009 2027}"

GS_CKPT="output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
SCSP_CKPT="${SCSP_CKPT:-output/simulator/latest_run/ckpts/scsp/best.pth}"
SCSP_OTR="${SCSP_OTR:-2}"

VARIANTS=(natural numeric shuffled_names names_only)
DATASETS=(GlobalStore SupplyChainShipmentPricing)

declare -A MODEL_HF=(
    [gpt2]="gpt2"
    [qwen3-1.7B]="Qwen/Qwen3-1.7B"
    [phi4-mini-reasoning]="microsoft/Phi-4-mini-reasoning"
)
MODELS=(gpt2 qwen3-1.7B)
declare -A MODEL_MAXPAR=(
    [gpt2]=18               # 0.5 GB × 9/GPU = 4.5 GB peak
    [qwen3-1.7B]=8          # 7 GB × 4/GPU = 28 GB; cache exists → no build race
    [phi4-mini-reasoning]=2 # 15 GB × 1/GPU = 15 GB; must stay at 2: no cache yet,
                            #   >2 concurrent would race on the same frac1.00 cache file
)

total=$(( ${#VARIANTS[@]} * ${#MODELS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]} ))
echo "=========================================="
echo " Experiment 6 — Prompt Ablation"
echo " Variants: ${VARIANTS[*]}"
echo " Backbones: ${MODELS[*]}"
echo " Datasets: ${DATASETS[*]}"
echo " Seeds: ${SEEDS[*]}"
echo " Frac: 1.00 only"
echo " Total jobs: ${total}"
echo " dm_epochs=${DM_EPOCHS}  run_id=${RUN_ID}"
echo "=========================================="

# ── Semaphore ────────────────────────────────────────────────────────────────
sem_init() {
    mkfifo "/tmp/sem_${$}_$1"
    exec 200<>"/tmp/sem_${$}_$1"
    rm "/tmp/sem_${$}_$1"
    local n=$1; for ((i=0; i<n; i++)); do echo >&200; done
}
sem_wait() { read -u200; }
sem_post() { echo >&200; }
sem_close() { exec 200>&- 2>/dev/null || true; }
trap 'sem_close; wait' EXIT

# ── Single-job launcher ──────────────────────────────────────────────────────
# Args: variant dataset hf_name seed gpu_id log
launch_job() {
    local variant="$1" dataset="$2" hf_name="$3" seed="$4" gpu_id="$5" log="$6"
    local extra_args=""
    case "${dataset}" in
        GlobalStore)
            extra_args="--otr_reward_coeff 10 --ckpt ${GS_CKPT}" ;;
        SupplyChainShipmentPricing)
            extra_args="--otr_reward_coeff ${SCSP_OTR} --ckpt ${SCSP_CKPT}" ;;
    esac
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 main/cb_main_llm.py \
        --use_gpu 1 --device_id 0 \
        --dataset "${dataset}" --train_mode 2 \
        --wandb 0 --hf_model_name "${hf_name}" --save 0 \
        --dm_epochs "${DM_EPOCHS}" --train_frac 1.00 \
        --seed "${seed}" \
        --prompt_variant "${variant}" \
        ${extra_args} \
        > "${log}" 2>&1
}

# ── Model runner ─────────────────────────────────────────────────────────────
run_model() {
    local model_tag="$1" max_par="$2"
    local hf_name="${MODEL_HF[${model_tag}]}"
    local group_total=$(( ${#VARIANTS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]} ))

    echo ""
    echo "┌─ ${model_tag}  (parallelism=${max_par})  $(date '+%Y-%m-%d %H:%M:%S') ─────────"

    sem_init "${max_par}"
    local pids=() job_num=0

    for VARIANT in "${VARIANTS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            for DATASET in "${DATASETS[@]}"; do
                local gpu_id="${GPUS[$(( job_num % NUM_GPUS ))]}"
                local DS_LOWER DS_TAG
                DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
                DS_TAG="${DS_LOWER/supplychainshipmentpricing/scsp}"
                local log_dir="${BASE_OUT_DIR}/${VARIANT}/${model_tag}/seed${SEED}"
                mkdir -p "${log_dir}"
                local log="${log_dir}/${DS_TAG}.log"

                job_num=$(( job_num + 1 ))
                sem_wait
                echo "  [${job_num}/${group_total}] GPU${gpu_id} → ${model_tag}  ${VARIANT}  ${DATASET}  seed=${SEED}  $(date '+%H:%M:%S')"

                (
                    launch_job "${VARIANT}" "${DATASET}" "${hf_name}" "${SEED}" "${gpu_id}" "${log}"
                    local rc=$?
                    [ "${rc}" -ne 0 ] && echo "  [FAIL rc=${rc}] ${model_tag} ${VARIANT} ${DATASET} seed=${SEED} GPU${gpu_id}" >&2
                    sem_post
                ) &
                pids+=($!)
                sleep 1
            done
        done
    done

    local failed=0
    for pid in "${pids[@]}"; do wait "${pid}" || failed=$(( failed + 1 )); done
    sem_close
    [ "${failed}" -gt 0 ] && echo "└─ ${model_tag} DONE with ${failed} failure(s)" || echo "└─ ${model_tag} DONE (all ${group_total} OK)"
}

run_model "gpt2"       "${MODEL_MAXPAR[gpt2]}"
run_model "qwen3-1.7B" "${MODEL_MAXPAR[qwen3-1.7B]}"
# run_model "phi4-mini-reasoning" "${MODEL_MAXPAR[phi4-mini-reasoning]}"

echo ""
echo "=========================================="
echo " All ${total} prompt ablation jobs complete."
echo " Logs → ${BASE_OUT_DIR}/{variant}/{model}/seed{N}/{dataset}.log"
echo " Extract: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*/*/seed*/*.log"
echo "=========================================="
