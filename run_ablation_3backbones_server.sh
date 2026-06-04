#!/bin/bash
# Experiment 4 — Method ablations on 3 representative backbones.
#
# Matrix: 3 backbones × 3 ablation variants × 4 datasets × 5 seeds, frac=1.00 only.
# Backbones: gpt2 (small decoder-only), Qwen/Qwen3-1.7B (medium decoder-only),
#            microsoft/Phi-4-mini-reasoning (reasoning decoder-only).
# One per pretraining family and scale — paper caption: "ablations run on three
# representative backbones across scales and pretraining families."
#
# Variants:
#   no_vocab_init    — --pool_init random  (removes VocabAlign initialization)
#   mean_pool        — --pool_type mean    (removes learned attention pooling)
#   hard_labels_only — --no_soft_labels    (removes KL-div soft-label term)
#
# Total: 3 × 3 × 4 × 5 = 180 jobs at frac=1.00.
#
# Env overrides:
#   SEEDS     — space-separated seed list (default: 42 0 1 2 3)
#   SCSP_CKPT — path to SupplyChainShipmentPricing simulator checkpoint
#   SCSP_OTR  — otr_reward_coeff for SCSP (default: 2)
#
# Usage:
#   bash run_ablation_3backbones_server.sh [DM_EPOCHS]
#   bash run_nohup.sh run_ablation_3backbones_server.sh

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/decision_maker/ablation_3b/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

NUM_GPUS=2
GPUS=(2 3)
DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-42 0 1 2 3}"
SCSP_CKPT="${SCSP_CKPT:-output/simulator/latest_run/ckpts/scsp/best.pth}"
SCSP_OTR="${SCSP_OTR:-2}"

DATACO_CKPT="output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"
GS_CKPT="output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
OAS_CKPT="output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"

VARIANTS=(no_vocab_init mean_pool hard_labels_only)
DATASETS=(DataCo GlobalStore OAS SupplyChainShipmentPricing)

declare -A MODEL_HF=(
    [gpt2]="gpt2"
    [qwen3-1.7B]="Qwen/Qwen3-1.7B"
    [phi4-mini-reasoning]="microsoft/Phi-4-mini-reasoning"
)
MODELS=(gpt2 qwen3-1.7B phi4-mini-reasoning)
# Memory: gpt2 ~3 GB, qwen3-1.7B ~14 GB, phi4-mini-reasoning ~15 GB
declare -A MODEL_MAXPAR=([gpt2]=18 [qwen3-1.7B]=6 [phi4-mini-reasoning]=6)

total=$(( ${#MODELS[@]} * ${#VARIANTS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]} ))
echo "=========================================="
echo " Experiment 4 — Method Ablations (3 backbones)"
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
# Args: variant dataset hf_name seed gpu_id model_tag log
launch_job() {
    local variant="$1" dataset="$2" hf_name="$3" seed="$4" gpu_id="$5" model_tag="$6" log="$7"
    local extra_flags=""
    local extra_args=""

    case "${variant}" in
        no_vocab_init)    extra_flags="--pool_init random" ;;
        mean_pool)        extra_flags="--pool_type mean" ;;
        hard_labels_only) extra_flags="--no_soft_labels" ;;
    esac

    case "${dataset}" in
        DataCo)
            extra_args="--otr_reward_coeff 2 --ckpt ${DATACO_CKPT}" ;;
        GlobalStore)
            extra_args="--otr_reward_coeff 10 --ckpt ${GS_CKPT}" ;;
        OAS)
            extra_args="--dm_lr 0.00003 --otr_reward_coeff 50 --ckpt ${OAS_CKPT}" ;;
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
        ${extra_args} ${extra_flags} \
        > "${log}" 2>&1
}

# ── Model-group runner ───────────────────────────────────────────────────────
# run_model_group MODEL_TAG MAX_PARALLEL
run_model_group() {
    local model_tag="$1" max_par="$2"
    local hf_name="${MODEL_HF[${model_tag}]}"
    local group_total=$(( ${#VARIANTS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]} ))

    echo ""
    echo "┌─ ${model_tag}  (parallelism=${max_par} across ${NUM_GPUS} GPUs)  $(date '+%Y-%m-%d %H:%M:%S') ─────────"

    sem_init "${max_par}"
    local pids=() job_num=0

    for VARIANT in "${VARIANTS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            for DATASET in "${DATASETS[@]}"; do
                local gpu_id="${GPUS[$(( job_num % NUM_GPUS ))]}"
                local DS_LOWER
                DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
                local log_dir="${BASE_OUT_DIR}/${VARIANT}/${model_tag}/seed${SEED}"
                mkdir -p "${log_dir}"
                local log="${log_dir}/${DS_LOWER}.log"

                job_num=$(( job_num + 1 ))
                sem_wait

                echo "  [${job_num}/${group_total}] GPU${gpu_id} → ${model_tag}  ${VARIANT}  ${DATASET}  seed=${SEED}  $(date '+%H:%M:%S')"

                (
                    launch_job "${VARIANT}" "${DATASET}" "${hf_name}" "${SEED}" "${gpu_id}" "${model_tag}" "${log}"
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

    if [ "${failed}" -gt 0 ]; then
        echo "└─ ${model_tag} DONE with ${failed} failure(s)  $(date '+%H:%M:%S')"
    else
        echo "└─ ${model_tag} DONE (all ${group_total} OK)  $(date '+%H:%M:%S')"
    fi
}

# ── Execute model groups sequentially ────────────────────────────────────────
run_model_group "gpt2"                "${MODEL_MAXPAR[gpt2]}"
run_model_group "qwen3-1.7B"          "${MODEL_MAXPAR[qwen3-1.7B]}"
run_model_group "phi4-mini-reasoning" "${MODEL_MAXPAR[phi4-mini-reasoning]}"

echo ""
echo "=========================================="
echo " All ${total} ablation jobs complete."
echo " Logs → ${BASE_OUT_DIR}/{variant}/{model_tag}/seed{N}/{dataset}.log"
echo " Extract: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*/*/seed*/*.log"
echo "=========================================="
