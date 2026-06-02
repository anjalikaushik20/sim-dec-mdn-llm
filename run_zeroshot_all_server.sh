#!/bin/bash
# Parallel zero-shot sweep — all models × {DataCo, GlobalStore, OAS}, dm_epochs=0.
# Model groups run sequentially (one backbone loaded at a time); within each group
# all 3 dataset jobs run in parallel across GPUs 0-3.
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/decision_maker/new_prompt/zeroshot/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

NUM_GPUS=4

echo "=========================================="
echo " Zero-shot evaluation — all models (vocabalign)"
echo " dm_epochs=0  run_id=${RUN_ID}"
echo " GPUs: ${NUM_GPUS} × RTX 6000 Ada (49 GB each)"
echo " Jobs: 6 models × 3 datasets = 18"
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

# ── Single-job launcher ──────────────────────────────────────────────────────
# Args: dataset hf_name gpu_id model_tag log
launch_job() {
    local dataset="$1" hf_name="$2" gpu_id="$3" model_tag="$4" log="$5"
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
        --save 0 \
        --dm_epochs 0 \
        --train_frac 1.0 \
        ${extra_args} \
        > "${log}" 2>&1
}

# ── Model-group runner ───────────────────────────────────────────────────────
# run_model_group MODEL_TAG HF_NAME MAX_PARALLEL
# Runs all 3 dataset jobs for one model with MAX_PARALLEL concurrency.
# GPU assigned round-robin across NUM_GPUS.
run_model_group() {
    local model_tag="$1" hf_name="$2" max_par="$3"
    local log_dir="${BASE_OUT_DIR}/${model_tag}"
    mkdir -p "${log_dir}"

    echo ""
    echo "┌─ ${model_tag}  (parallelism=${max_par} across ${NUM_GPUS} GPUs)  $(date '+%Y-%m-%d %H:%M:%S') ─────────"

    sem_init "${max_par}"

    local pids=() job_num=0

    for DATASET in DataCo GlobalStore OAS; do
        local gpu_id=$(( job_num % NUM_GPUS ))
        job_num=$((job_num + 1))
        local log="${log_dir}/$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]').log"

        sem_wait

        echo "  [${job_num}/3] GPU${gpu_id} → ${model_tag}  ${DATASET}  $(date '+%H:%M:%S')"

        (
            launch_job "${DATASET}" "${hf_name}" "${gpu_id}" "${model_tag}" "${log}"
            local rc=$?
            if [ "${rc}" -ne 0 ]; then
                echo "  [FAIL rc=${rc}] ${model_tag} ${DATASET} GPU${gpu_id}" >&2
            fi
            sem_post
        ) &
        pids+=($!)

        sleep 1  # stagger HuggingFace / W&B API calls
    done

    local failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=$((failed + 1))
    done

    sem_close

    if [ "${failed}" -gt 0 ]; then
        echo "└─ ${model_tag} DONE with ${failed} failure(s)  $(date '+%H:%M:%S')"
    else
        echo "└─ ${model_tag} DONE (all 3 OK)  $(date '+%H:%M:%S')"
    fi
}

# ── Execute groups sequentially ──────────────────────────────────────────────
# Zero-shot = inference only, so memory per job is lower than training.
# All 3 dataset jobs per model run in parallel (each on its own GPU);
# model groups are sequential so only one backbone is loaded at a time.
run_model_group "qwen3-0.6B"  "Qwen/Qwen3-0.6B" 3
run_model_group "qwen3-1.7B"  "Qwen/Qwen3-1.7B" 3
run_model_group "qwen3-4B"    "Qwen/Qwen3-4B"   3
run_model_group "gpt2"        "gpt2"             3
run_model_group "gpt2-medium" "gpt2-medium"      3
run_model_group "gpt2-large"  "gpt2-large"       3

echo ""
echo "=========================================="
echo " All 18 zero-shot jobs complete."
echo " Logs        → ${BASE_OUT_DIR}"
echo " Extract:      grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*/*.log"
echo "=========================================="
