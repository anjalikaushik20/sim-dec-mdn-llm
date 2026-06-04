#!/bin/bash
# Sample-efficiency sweep for Qwen/Qwen3-0.6B.
# Runs 4 datasets × 6 fracs × 5 seeds = 120 jobs.
# Zero-shot (frac=0) is handled separately by run_zeroshot_all_server.sh.
#
# Memory: Qwen3-0.6B ~7 GB float32 → max_par=12 on two 49 GB GPUs (84 GB).
#
# Usage:
#   bash run_sample_efficiency_qwen3_06b_parallel.sh [DM_EPOCHS]
#   bash run_nohup.sh run_sample_efficiency_qwen3_06b_parallel.sh

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/akaush39/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/decision_maker/all_fracs/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"
NUM_GPUS=1
GPUS=(3)
MODEL_TAG="qwen3-0.6B"
HF_NAME="Qwen/Qwen3-0.6B"
SEEDS=(42 131 521 1009 2027)
FRACS=(0.01 0.05 0.10 0.25 0.50 1.00)
DATASETS=(DataCo GlobalStore OAS SupplyChainShipmentPricing)

DATACO_CKPT="output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"
GS_CKPT="output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
OAS_CKPT="output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"
SCSP_CKPT="output/simulator/latest_run/ckpts/scsp/peach-galaxy-2336_epoch171.pth"
SCSP_OTR="${SCSP_OTR:-2}"

echo "=========================================="
echo " Sample efficiency — ${MODEL_TAG}"
echo " dm_epochs=${DM_EPOCHS}  run_id=${RUN_ID}"
echo " Seeds: ${SEEDS[*]}"
echo " GPUs: ${GPUS[*]}"
echo " Jobs: ${#DATASETS[@]} datasets × ${#FRACS[@]} fracs × ${#SEEDS[@]} seeds = $(( ${#DATASETS[@]} * ${#FRACS[@]} * ${#SEEDS[@]} ))"
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
launch_job() {
    local dataset="$1" frac="$2" seed="$3" gpu_id="$4" log="$5"
    local ckpt_dir="output/decision_maker/all_fracs/${dataset/SupplyChainShipmentPricing/SCSP}/ckpts/frac${frac}/${MODEL_TAG}/seed${seed}"
    mkdir -p "${ckpt_dir}"
    local extra_args=""
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
        --dataset "${dataset}" \
        --train_mode 2 \
        --wandb 0 \
        --hf_model_name "${HF_NAME}" \
        --save 1 \
        --ckpt_dir "${ckpt_dir}" \
        --dm_epochs "${DM_EPOCHS}" \
        --train_frac "${frac}" \
        --seed "${seed}" \
        ${extra_args} \
        > "${log}" 2>&1
}

# ── Run all jobs ─────────────────────────────────────────────────────────────
MAX_PAR=1
sem_init "${MAX_PAR}"

pids=()
job_num=0
total=$(( ${#DATASETS[@]} * ${#FRACS[@]} * ${#SEEDS[@]} ))

echo ""
echo "┌─ ${MODEL_TAG}  (parallelism=${MAX_PAR})  $(date '+%Y-%m-%d %H:%M:%S') ─────────"

for SEED in "${SEEDS[@]}"; do
    for FRAC in "${FRACS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            gpu_id="${GPUS[$(( job_num % NUM_GPUS ))]}"
            job_num=$(( job_num + 1 ))
            DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
            DS_LOWER="${DS_LOWER/supplychainshipmentpricing/scsp}"
            log_dir="${BASE_OUT_DIR}/seed${SEED}/${MODEL_TAG}"
            mkdir -p "${log_dir}"
            log="${log_dir}/${DS_LOWER}_frac${FRAC}.log"

            sem_wait
            echo "  [${job_num}/${total}] GPU${gpu_id} → ${MODEL_TAG}  ${DATASET}  frac=${FRAC}  seed=${SEED}  $(date '+%H:%M:%S')"

            (
                launch_job "${DATASET}" "${FRAC}" "${SEED}" "${gpu_id}" "${log}"
                rc=$?
                [ "${rc}" -ne 0 ] && echo "  [FAIL rc=${rc}] ${MODEL_TAG} ${DATASET} frac=${FRAC} seed=${SEED} GPU${gpu_id}" >&2
                sem_post
            ) &
            pids+=($!)
            sleep 1
        done
    done
done

failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=$(( failed + 1 )); done
sem_close

if [ "${failed}" -gt 0 ]; then
    echo "└─ ${MODEL_TAG} DONE with ${failed} failure(s)  $(date '+%H:%M:%S')"
else
    echo "└─ ${MODEL_TAG} DONE (all ${total} OK)  $(date '+%H:%M:%S')"
fi

echo ""
echo "=========================================="
echo " All ${total} jobs complete."
echo " Logs        → ${BASE_OUT_DIR}/seed*/${MODEL_TAG}/"
echo " Checkpoints → output/decision_maker/all_fracs/{dataset}/ckpts/frac*/${MODEL_TAG}/seed*/"
echo " Extract: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/seed*/${MODEL_TAG}/*.log"
echo "=========================================="
