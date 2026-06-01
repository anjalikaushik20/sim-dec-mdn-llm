#!/bin/bash
# Parallel ablation sweep — 3 variants × 6 models × 6 fracs × 3 datasets = 324 jobs.
# Model groups run sequentially (one backbone at a time); within each group all
# variant × frac × dataset jobs run in parallel up to the per-group concurrency limit.
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/decision_maker/ablation/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

NUM_GPUS=2
GPUS=(2 3)
DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"

echo "=========================================="
echo " Ablation sweep — VocabAlign"
echo " Variants: no_vocab_init, mean_pool, hard_labels_only"
echo " Models:   6  |  Fracs: 6  |  Datasets: 3"
echo " Total jobs: 324  |  dm_epochs=${DM_EPOCHS}"
echo " GPUs: 2, 3 (gpt2-large and qwen3-4B run sequentially)"
echo "=========================================="

DATACO_CKPT="output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"
GS_CKPT="output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
OAS_CKPT="output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"

VARIANTS=(no_vocab_init mean_pool hard_labels_only)
FRACS=(0.01 0.05 0.10 0.25 0.50 1.00)
DATASETS=(DataCo GlobalStore OAS)

# ── Semaphore: token-pool via anonymous pipe on fd 200 ──────────────────────
sem_init() {
    mkfifo "/tmp/sem_${$}_$1"
    exec 200<>"/tmp/sem_${$}_$1"
    rm "/tmp/sem_${$}_$1"
    local n=$1
    for ((i = 0; i < n; i++)); do echo >&200; done
}
sem_wait() { read -u200; }
sem_post() { echo >&200; }
sem_close() { exec 200>&- 2>/dev/null || true; }

trap 'sem_close; wait' EXIT

# ── Single-job launcher ──────────────────────────────────────────────────────
# Args: variant dataset hf_name frac gpu_id model_tag log
launch_job() {
    local variant="$1" dataset="$2" hf_name="$3" frac="$4" gpu_id="$5" model_tag="$6" log="$7"
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
    esac

    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 main/cb_main_llm.py \
        --use_gpu 1 --device_id 0 \
        --dataset "${dataset}" --train_mode 2 \
        --wandb 1 --hf_model_name "${hf_name}" --save 0 \
        --dm_epochs "${DM_EPOCHS}" --train_frac "${frac}" \
        ${extra_args} ${extra_flags} \
        > "${log}" 2>&1
}

# ── Model-group runner ───────────────────────────────────────────────────────
# run_model_group MODEL_TAG HF_NAME MAX_PARALLEL
# Runs all 54 (variant × frac × dataset) jobs for one model with MAX_PARALLEL
# concurrency. GPU assigned round-robin across NUM_GPUS.
run_model_group() {
    local model_tag="$1" hf_name="$2" max_par="$3"
    local total_group=$(( ${#VARIANTS[@]} * ${#FRACS[@]} * ${#DATASETS[@]} ))

    echo ""
    echo "┌─ ${model_tag}  (parallelism=${max_par} across ${NUM_GPUS} GPUs)  $(date '+%Y-%m-%d %H:%M:%S') ─────────"

    sem_init "${max_par}"

    local pids=() job_num=0

    for VARIANT in "${VARIANTS[@]}"; do
        for FRAC in "${FRACS[@]}"; do
            for DATASET in "${DATASETS[@]}"; do
                local gpu_id="${GPUS[$(( job_num % NUM_GPUS ))]}"
                local DS_LOWER
                DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')
                local log_dir="${BASE_OUT_DIR}/${VARIANT}/${model_tag}"
                mkdir -p "${log_dir}"
                local log="${log_dir}/${DS_LOWER}_frac${FRAC}.log"

                job_num=$(( job_num + 1 ))
                sem_wait

                echo "  [${job_num}/${total_group}] GPU${gpu_id} → ${model_tag}  ${VARIANT}  ${DATASET}  frac=${FRAC}  $(date '+%H:%M:%S')"

                (
                    launch_job "${VARIANT}" "${DATASET}" "${hf_name}" "${FRAC}" "${gpu_id}" "${model_tag}" "${log}"
                    local rc=$?
                    if [ "${rc}" -ne 0 ]; then
                        echo "  [FAIL rc=${rc}] ${model_tag} ${VARIANT} ${DATASET} frac=${FRAC} GPU${gpu_id}" >&2
                    fi
                    sem_post
                ) &
                pids+=($!)

                sleep 1  # stagger HuggingFace / W&B API calls
            done
        done
    done

    local failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=$(( failed + 1 ))
    done

    sem_close

    if [ "${failed}" -gt 0 ]; then
        echo "└─ ${model_tag} DONE with ${failed} failure(s)  $(date '+%H:%M:%S')"
    else
        echo "└─ ${model_tag} DONE (all ${total_group} OK)  $(date '+%H:%M:%S')"
    fi
}

# ── Execute groups sequentially ──────────────────────────────────────────────
# Parallelism = min(54, floor(49 GB / job_GB) × NUM_GPUS) — 2 GPUs (2 and 3).
# gpt2-large and qwen3-4B run sequentially (max_par=1) to avoid OOM.
#
#   gpt2         ~3 GB/job  → floor(49/3)=16 × 2 = 32
#   gpt2-medium  ~5 GB/job  → floor(49/5)= 9 × 2 = 18
#   gpt2-large   ~7 GB/job  → sequential (1)
#   qwen3-0.6B   ~7 GB/job  → floor(49/7)= 7 × 2 = 14
#   qwen3-1.7B  ~14 GB/job  → floor(49/14)=3 × 2 =  6
#   qwen3-4B    ~22 GB/job  → sequential (1)

run_model_group "gpt2"        "gpt2"             32
run_model_group "gpt2-medium" "gpt2-medium"      18
run_model_group "gpt2-large"  "gpt2-large"        1
run_model_group "qwen3-0.6B"  "Qwen/Qwen3-0.6B" 14
run_model_group "qwen3-1.7B"  "Qwen/Qwen3-1.7B"  6
run_model_group "qwen3-4B"    "Qwen/Qwen3-4B"    1

echo ""
echo "=========================================="
echo " All 324 ablation jobs complete."
echo " Logs → output/decision_maker/ablation/${RUN_ID}/{variant}/{model_tag}/{dataset}_frac{frac}.log"
echo " Extract: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*/*/*.log"
echo "=========================================="
