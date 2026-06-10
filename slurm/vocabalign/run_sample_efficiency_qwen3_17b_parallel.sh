#!/bin/bash
# Sample-efficiency sweep for Qwen/Qwen3-1.7B — all 4 datasets, GPU 0, 1, 3.
#
# Each GPU runs a fully independent sequential queue — no cross-GPU barriers.
# As soon as a job finishes its GPU starts the next one immediately.
#
#   GPU 0 : DataCo                          (30 jobs)
#   GPU 1 : GlobalStore + SCSP interleaved  (60 jobs)
#   GPU 3 : OAS                             (30 jobs)
#
# Resume / skip: any job whose checkpoint (ckpt_dir/*_attnpool_best.pth)
# already exists is silently skipped — safe to kill and restart at any time.
#
# Pre-existing completed logs are copied from output/decision_maker/all_fracs/completed/
# into the new run directory at startup so results are not lost across runs.
#
# 4 datasets × 6 fracs × 5 seeds = 120 jobs total.

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
MODEL_TAG="qwen3-1.7B"
HF_NAME="Qwen/Qwen3-1.7B"
SEEDS=(42 131 521 1009 2027)
FRACS=(0.01 0.05 0.10 0.25 0.50 1.00)

DATACO_CKPT="output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"
GS_CKPT="output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
OAS_CKPT="output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"
SCSP_CKPT="${SCSP_CKPT:-output/simulator/latest_run/ckpts/scsp/best.pth}"
SCSP_OTR="${SCSP_OTR:-2}"

echo "=========================================="
echo " Sample efficiency — ${MODEL_TAG} (4 datasets)"
echo " dm_epochs=${DM_EPOCHS}  run_id=${RUN_ID}"
echo " Seeds: ${SEEDS[*]}"
echo " GPUs: 0→DataCo  1→GlobalStore+SCSP  3→OAS"
echo " Total jobs: 120"
echo "=========================================="

# ── copy any previously saved completed logs into this run's log dir ─────────
COMPLETED_DIR="output/decision_maker/all_fracs/completed"
if [ -d "${COMPLETED_DIR}" ]; then
    for seed_dir in "${COMPLETED_DIR}"/seed*/; do
        [ -d "${seed_dir}" ] || continue
        seed=$(basename "${seed_dir}")
        dst="${BASE_OUT_DIR}/${seed}/${MODEL_TAG}"
        mkdir -p "${dst}"
        cp "${seed_dir}${MODEL_TAG}/"*.log "${dst}/" 2>/dev/null || true
    done
    echo "Pre-populated logs from ${COMPLETED_DIR}"
fi

# ── helper ───────────────────────────────────────────────────────────────────
run_job() {
    local DATASET="$1" FRAC="$2" SEED="$3" GPU="$4"
    local DS_LOWER DS_TAG ckpt_dir extra_args log_dir log

    DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    DS_TAG="${DS_LOWER/supplychainshipmentpricing/scsp}"
    ckpt_dir="output/decision_maker/all_fracs/${DS_TAG}/ckpts/frac${FRAC}/${MODEL_TAG}/seed${SEED}"

    # skip only if a completed log (containing best_profit) exists from a previous run
    local completed_log="output/decision_maker/all_fracs/completed/seed${SEED}/${MODEL_TAG}/${DS_TAG}_frac${FRAC}.log"
    if [ -f "${completed_log}" ] && grep -q "best_pmp_3" "${completed_log}"; then
        echo "  [SKIP] GPU${GPU} ${DATASET} frac=${FRAC} seed=${SEED} (completed log exists)"
        return 0
    fi

    mkdir -p "${ckpt_dir}"
    log_dir="${BASE_OUT_DIR}/seed${SEED}/${MODEL_TAG}"
    mkdir -p "${log_dir}"
    log="${log_dir}/${DS_TAG}_frac${FRAC}.log"

    case "${DATASET}" in
        DataCo)                     extra_args="--otr_reward_coeff 2   --ckpt ${DATACO_CKPT}" ;;
        GlobalStore)                extra_args="--otr_reward_coeff 10  --ckpt ${GS_CKPT}" ;;
        OAS)                        extra_args="--dm_lr 0.00003 --otr_reward_coeff 50 --ckpt ${OAS_CKPT}" ;;
        SupplyChainShipmentPricing) extra_args="--otr_reward_coeff ${SCSP_OTR} --ckpt ${SCSP_CKPT}" ;;
    esac

    echo "  [RUN] GPU${GPU} → ${DATASET} frac=${FRAC} seed=${SEED}  $(date '+%H:%M:%S')"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${GPU}" python3 main/cb_main_llm.py \
        --use_gpu 1 --device_id 0 \
        --dataset "${DATASET}" \
        --train_mode 2 \
        --wandb 0 \
        --hf_model_name "${HF_NAME}" \
        --save 1 \
        --ckpt_dir "${ckpt_dir}" \
        --dm_epochs "${DM_EPOCHS}" \
        --train_frac "${FRAC}" \
        --seed "${SEED}" \
        ${extra_args} \
        > "${log}" 2>&1
    local rc=$?
    [ "${rc}" -ne 0 ] && echo "  [FAIL rc=${rc}] GPU${GPU} ${DATASET} frac=${FRAC} seed=${SEED}" >&2
    return "${rc}"
}

# ── GPU 0: DataCo ─────────────────────────────────────────────────────────────
(
    echo "┌─ GPU0: DataCo  $(date '+%H:%M:%S')"
    for SEED in "${SEEDS[@]}"; do
        for FRAC in "${FRACS[@]}"; do
            run_job DataCo "${FRAC}" "${SEED}" 0
        done
    done
    echo "└─ GPU0: DataCo DONE  $(date '+%H:%M:%S')"
) &

# ── GPU 1: GlobalStore + SCSP interleaved ────────────────────────────────────
(
    echo "┌─ GPU1: GlobalStore + SCSP  $(date '+%H:%M:%S')"
    for SEED in "${SEEDS[@]}"; do
        for FRAC in "${FRACS[@]}"; do
            run_job GlobalStore                "${FRAC}" "${SEED}" 1
            run_job SupplyChainShipmentPricing "${FRAC}" "${SEED}" 1
        done
    done
    echo "└─ GPU1: GlobalStore + SCSP DONE  $(date '+%H:%M:%S')"
) &

# ── GPU 3: OAS ───────────────────────────────────────────────────────────────
(
    echo "┌─ GPU3: OAS  $(date '+%H:%M:%S')"
    for SEED in "${SEEDS[@]}"; do
        for FRAC in "${FRACS[@]}"; do
            run_job OAS "${FRAC}" "${SEED}" 3
        done
    done
    echo "└─ GPU3: OAS DONE  $(date '+%H:%M:%S')"
) &

wait

echo ""
echo "=========================================="
echo " All queues complete. Logs → ${BASE_OUT_DIR}"
echo " Extract: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/seed*/${MODEL_TAG}/*.log"
echo "=========================================="
