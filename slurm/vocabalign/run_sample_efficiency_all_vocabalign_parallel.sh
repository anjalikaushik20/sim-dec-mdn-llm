#!/bin/bash
# Parallel sample-efficiency sweep — all models, vocabalign.
#
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
# Env overrides:
#   SEEDS        — space-separated list of seeds (default: 42 0 1 2 3)
#   SCSP_CKPT    — path to SupplyChainShipmentPricing simulator checkpoint
#   SCSP_OTR     — otr_reward_coeff for SCSP (calibrate after first run; default: 2)

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/decision_maker/all_fracs/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"
NUM_GPUS=2
GPUS=(2 3)
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-42 0 1 2 3}"
SCSP_CKPT="${SCSP_CKPT:-output/simulator/latest_run/ckpts/scsp/best.pth}"
SCSP_OTR="${SCSP_OTR:-2}"

echo "=========================================="
echo " Sample efficiency — all models (vocabalign)"
echo " dm_epochs=${DM_EPOCHS}  run_id=${RUN_ID}"
echo " Seeds: ${SEEDS[*]}"
echo " GPUs: 2, 3 (RTX 6000 Ada, 49 GB each, 98 GB total)"
echo " Jobs: 6 models × 6 fracs × 4 datasets × ${#SEEDS[@]} seeds"
echo " (Zero-shot frac=0 is handled separately by run_zeroshot_all_server.sh)"
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
# Args: dataset hf_name frac seed gpu_id model_tag log
launch_job() {
    local dataset="$1" hf_name="$2" frac="$3" seed="$4" gpu_id="$5" model_tag="$6" log="$7"
    local ckpt_dir="output/decision_maker/all_fracs/${dataset}/ckpts/frac${frac}/${model_tag}/seed${seed}"
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
        SupplyChainShipmentPricing)
            extra_args="--otr_reward_coeff ${SCSP_OTR} --ckpt ${SCSP_CKPT}"
            ;;
    esac

    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 main/cb_main_llm.py \
        --use_gpu 1 --device_id 0 \
        --dataset "${dataset}" \
        --train_mode 2 \
        --wandb 0 \
        --hf_model_name "${hf_name}" \
        --save 1 \
        --ckpt_dir "${ckpt_dir}" \
        --dm_epochs "${DM_EPOCHS}" \
        --train_frac "${frac}" \
        --seed "${seed}" \
        ${extra_args} \
        > "${log}" 2>&1
}

# ── Model-group runner ───────────────────────────────────────────────────────
# run_model_group MODEL_TAG HF_NAME MAX_PARALLEL
# Runs all (frac × dataset × seed) jobs for one model with MAX_PARALLEL concurrency.
# GPU assigned round-robin across NUM_GPUS so load is spread evenly.
run_model_group() {
    local model_tag="$1" hf_name="$2" max_par="$3"
    local total_group=$(( 6 * 4 * ${#SEEDS[@]} ))  # fracs × datasets × seeds

    echo ""
    echo "┌─ ${model_tag}  (parallelism=${max_par} across ${NUM_GPUS} GPUs)  $(date '+%Y-%m-%d %H:%M:%S') ─────────"

    sem_init "${max_par}"

    local pids=() job_num=0

    for SEED in "${SEEDS[@]}"; do
        for FRAC in 0.01 0.05 0.10 0.25 0.50 1.00; do
            for DATASET in DataCo GlobalStore OAS SupplyChainShipmentPricing; do
                local gpu_id="${GPUS[$(( job_num % NUM_GPUS ))]}"
                job_num=$((job_num + 1))
                local DS_LOWER
                DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
                local log_dir="${BASE_OUT_DIR}/seed${SEED}/${model_tag}"
                mkdir -p "${log_dir}"
                local log="${log_dir}/${DS_LOWER}_frac${FRAC}.log"

                sem_wait  # blocks until a slot is free

                echo "  [${job_num}/${total_group}] GPU${gpu_id} → ${model_tag}  ${DATASET}  frac=${FRAC}  seed=${SEED}  $(date '+%H:%M:%S')"

                (
                    launch_job "${DATASET}" "${hf_name}" "${FRAC}" "${SEED}" "${gpu_id}" "${model_tag}" "${log}"
                    local rc=$?
                    if [ "${rc}" -ne 0 ]; then
                        echo "  [FAIL rc=${rc}] ${model_tag} ${DATASET} frac=${FRAC} seed=${SEED} GPU${gpu_id}" >&2
                    fi
                    sem_post
                ) &
                pids+=($!)

                sleep 1  # stagger HuggingFace / W&B API calls
            done
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
        echo "└─ ${model_tag} DONE (all ${total_group} OK)  $(date '+%H:%M:%S')"
    fi
}

run_model_group "qwen3-0.6B"  "Qwen/Qwen3-0.6B" 1   # sequential
run_model_group "qwen3-1.7B"  "Qwen/Qwen3-1.7B" 1   # sequential
run_model_group "qwen3-4B"    "Qwen/Qwen3-4B"   1   # sequential
run_model_group "phi4-mini"   "microsoft/Phi-4-mini-reasoning" 1   # sequential
run_model_group "gpt2"        "gpt2"             1   # sequential
run_model_group "gpt2-medium" "gpt2-medium"      1   # sequential
run_model_group "gpt2-large"  "gpt2-large"       1   # sequential

echo ""
echo "=========================================="
echo " All jobs complete."
echo " Logs        → ${BASE_OUT_DIR}/seed*/{model}/"
echo " Checkpoints → output/decision_maker/all_fracs/{dataset}/ckpts/frac*/{model}/seed*/"
echo " Extract: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/seed*/*/*.log"
echo "=========================================="
