#!/bin/bash
# Parallel sample-efficiency sweep — all models, vocabalign.
#
# H100 NVL GPU 0  (95.8 GB VRAM, 1 TB RAM, 48 CPU cores)
#
# Resource allocation — memory per job (float32 frozen backbone + CUDA overhead):
#   gpt2         ~3 GB  →  18 parallel  (54 GB)
#   gpt2-medium  ~5 GB  →  16 parallel  (80 GB)
#   gpt2-large   ~7 GB  →  12 parallel  (84 GB)
#   qwen3-0.6B   ~7 GB  →  12 parallel  (84 GB)
#   qwen3-1.7B   ~14 GB →   6 parallel  (84 GB)
#   qwen3-4B     ~22 GB →   4 parallel  (88 GB)
#
# Model groups run sequentially (one at a time) to stay within VRAM budget.
# Within each group jobs run in parallel up to the limit above.
#
# Usage: bash run_sample_efficiency_all_vocabalign_parallel.sh [DM_EPOCHS]

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/decision_maker/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"
NUM_GPUS=4   # RTX 6000 Ada × 4, 49 GB each

echo "=========================================="
echo " Sample efficiency — all models (vocabalign)"
echo " dm_epochs=${DM_EPOCHS}  run_id=${RUN_ID}"
echo " GPUs: ${NUM_GPUS} × RTX 6000 Ada (49 GB each, 196 GB total)"
echo " Jobs: 6 models × 6 fracs × 3 datasets = 108"
echo "=========================================="

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

# ── Single-job launcher (runs in a subshell background) ─────────────────────
# Args: dataset hf_name frac gpu_id model_tag log
launch_job() {
    local dataset="$1" hf_name="$2" frac="$3" gpu_id="$4" model_tag="$5" log="$6"
    local ckpt_dir="output/decision_maker/${dataset}/ckpts/frac${frac}/${model_tag}"
    mkdir -p "${ckpt_dir}"
    local extra_args=""

    case "${dataset}" in
        DataCo)
            extra_args="--otr_reward_coeff 2 \
                --ckpt output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"
            ;;
        GlobalStore)
            extra_args="--otr_reward_coeff 10 \
                --ckpt output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
            ;;
        OAS)
            extra_args="--dm_lr 0.00003 --otr_reward_coeff 50 \
                --ckpt output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"
            ;;
    esac

    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 main/cb_main_llm.py \
        --use_gpu 1 --device_id 0 \
        --dataset "${dataset}" \
        --train_mode 2 \
        --wandb 1 \
        --hf_model_name "${hf_name}" \
        --save 1 \
        --ckpt_dir "${ckpt_dir}" \
        --dm_epochs "${DM_EPOCHS}" \
        --train_frac "${frac}" \
        ${extra_args} \
        > "${log}" 2>&1
}

# ── Model-group runner ───────────────────────────────────────────────────────
# run_model_group MODEL_TAG HF_NAME MAX_PARALLEL
# Runs all 18 (frac × dataset) jobs for one model with MAX_PARALLEL concurrency.
# GPU assigned round-robin across NUM_GPUS so load is spread evenly.
run_model_group() {
    local model_tag="$1" hf_name="$2" max_par="$3"
    local log_dir="${BASE_OUT_DIR}/${model_tag}"
    mkdir -p "${log_dir}"

    echo ""
    echo "┌─ ${model_tag}  (parallelism=${max_par} across ${NUM_GPUS} GPUs)  $(date '+%Y-%m-%d %H:%M:%S') ─────────"

    sem_init "${max_par}"

    local pids=() job_num=0

    for FRAC in 0.01 0.05 0.10 0.25 0.50 1.00; do
        for DATASET in DataCo GlobalStore OAS; do
            local gpu_id=$(( job_num % NUM_GPUS ))
            job_num=$((job_num + 1))
            local log="${log_dir}/$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')_frac${FRAC}.log"

            sem_wait  # blocks until a slot is free

            echo "  [${job_num}/18] GPU${gpu_id} → ${model_tag}  ${DATASET}  frac=${FRAC}  $(date '+%H:%M:%S')"

            (
                launch_job "${DATASET}" "${hf_name}" "${FRAC}" "${gpu_id}" "${model_tag}" "${log}"
                local rc=$?
                if [ "${rc}" -ne 0 ]; then
                    echo "  [FAIL rc=${rc}] ${model_tag} ${DATASET} frac=${FRAC} GPU${gpu_id}" >&2
                fi
                sem_post
            ) &
            pids+=($!)

            sleep 1  # stagger HuggingFace / W&B API calls
        done
    done

    # Wait for every job in this group to finish
    local failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=$((failed + 1))
    done

    sem_close

    if [ "${failed}" -gt 0 ]; then
        echo "└─ ${model_tag} DONE with ${failed} failure(s)  $(date '+%H:%M:%S')"
    else
        echo "└─ ${model_tag} DONE (all 18 OK)  $(date '+%H:%M:%S')"
    fi
}

# ── Execute groups sequentially ──────────────────────────────────────────────
# Groups run one at a time so per-GPU memory stays within 49 GB.
# Parallelism = min(18, floor(49 GB / job_GB) × NUM_GPUS).
#
#   gpt2         ~3 GB/job  → floor(49/3)=16 × 4 = 64  → cap at 18 (all at once)
#   gpt2-medium  ~5 GB/job  → floor(49/5)= 9 × 4 = 36  → cap at 18
#   gpt2-large   ~7 GB/job  → floor(49/7)= 7 × 4 = 28  → cap at 18
#   qwen3-0.6B   ~7 GB/job  → floor(49/7)= 7 × 4 = 28  → cap at 18
#   qwen3-1.7B  ~14 GB/job  → floor(49/14)=3 × 4 = 12
#   qwen3-4B    ~22 GB/job  → floor(49/22)=2 × 4 =  8

run_model_group "gpt2"        "gpt2"             18   #  3 GB × 18, ~5 jobs/GPU
run_model_group "gpt2-medium" "gpt2-medium"      18   #  5 GB × 18, ~5 jobs/GPU
run_model_group "gpt2-large"  "gpt2-large"       18   #  7 GB × 18, ~5 jobs/GPU
run_model_group "qwen3-0.6B"  "Qwen/Qwen3-0.6B" 18   #  7 GB × 18, ~5 jobs/GPU
run_model_group "qwen3-1.7B"  "Qwen/Qwen3-1.7B" 12   # 14 GB × 12, 3 jobs/GPU
run_model_group "qwen3-4B"    "Qwen/Qwen3-4B"    8   # 22 GB ×  8, 2 jobs/GPU

echo ""
echo "=========================================="
echo " All 108 jobs complete."
echo " Logs       → ${BASE_OUT_DIR}"
echo " Checkpoints→ output/decision_maker/{DataCo,GlobalStore,OAS}/ckpts/frac*/{model}/"
echo "=========================================="
