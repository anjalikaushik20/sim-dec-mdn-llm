#!/bin/bash
# Sequential re-run of specific model×frac×dataset combinations that hit CUDA OOM
# in the parallel sweep. All 12 jobs run one at a time on a single GPU.
#
# Usage: bash run_sample_efficiency_redo_vocabalign_server.sh [GPU_ID] [DM_EPOCHS]
# Defaults: GPU_ID=0, DM_EPOCHS=200
#
# Jobs (in order):
#   DataCo:      gpt2-medium@0.50, gpt2-large@1.00,
#                qwen3-1.7B@0.10/0.25/1.00, qwen3-4B@0.25/0.50/1.00
#   GlobalStore: qwen3-4B@1.00
#   OAS:         qwen3-4B@0.10/0.25/0.50
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/decision_maker/all_fracs/${RUN_ID}"
GPU_ID="${1:-0}"
DM_EPOCHS="${2:-${DM_EPOCHS:-200}}"

echo "=========================================="
echo " Sequential re-run — CUDA OOM recovery"
echo " dm_epochs=${DM_EPOCHS}  gpu=${GPU_ID}  run_id=${RUN_ID}"
echo " Jobs: 12 (sequential, one at a time)"
echo "=========================================="

DATACO_CKPT="output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"
GS_CKPT="output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
OAS_CKPT="output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"

run_job() {
    local dataset="$1" model_tag="$2" hf_name="$3" frac="$4"
    local ds_lower
    ds_lower=$(echo "${dataset}" | tr '[:upper:]' '[:lower:]')
    local log_dir="${BASE_OUT_DIR}/${model_tag}"
    local ckpt_dir="output/decision_maker/${dataset}/ckpts/frac${frac}/${model_tag}"
    local log="${log_dir}/${ds_lower}_frac${frac}.log"
    mkdir -p "${log_dir}" "${ckpt_dir}"

    local extra_args=""
    case "${dataset}" in
        DataCo)
            extra_args="--otr_reward_coeff 2 --ckpt ${DATACO_CKPT}" ;;
        GlobalStore)
            extra_args="--otr_reward_coeff 10 --ckpt ${GS_CKPT}" ;;
        OAS)
            extra_args="--dm_lr 0.00003 --otr_reward_coeff 50 --ckpt ${OAS_CKPT}" ;;
    esac

    echo "  → ${model_tag}  ${dataset}  frac=${frac}  $(date '+%H:%M:%S')"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 main/cb_main_llm.py \
        --use_gpu 1 --device_id 0 \
        --dataset "${dataset}" --train_mode 2 \
        --wandb 1 --hf_model_name "${hf_name}" \
        --save 1 --ckpt_dir "${ckpt_dir}" \
        --dm_epochs "${DM_EPOCHS}" --train_frac "${frac}" \
        ${extra_args} \
        > "${log}" 2>&1
    echo "    done  $(date '+%H:%M:%S')"
}

# ── DataCo ────────────────────────────────────────────────────────────────────
echo "--- DataCo ---"
run_job DataCo gpt2-medium "gpt2-medium"      0.50
run_job DataCo gpt2-large  "gpt2-large"       1.00
run_job DataCo qwen3-1.7B  "Qwen/Qwen3-1.7B" 0.10
run_job DataCo qwen3-1.7B  "Qwen/Qwen3-1.7B" 0.25
run_job DataCo qwen3-1.7B  "Qwen/Qwen3-1.7B" 1.00
run_job DataCo qwen3-4B    "Qwen/Qwen3-4B"   0.25
run_job DataCo qwen3-4B    "Qwen/Qwen3-4B"   0.50
run_job DataCo qwen3-4B    "Qwen/Qwen3-4B"   1.00

# ── GlobalStore ───────────────────────────────────────────────────────────────
echo "--- GlobalStore ---"
run_job GlobalStore qwen3-4B "Qwen/Qwen3-4B"  1.00

# ── OAS ───────────────────────────────────────────────────────────────────────
echo "--- OAS ---"
run_job OAS qwen3-4B "Qwen/Qwen3-4B"          0.10
run_job OAS qwen3-4B "Qwen/Qwen3-4B"          0.25
run_job OAS qwen3-4B "Qwen/Qwen3-4B"          0.50

echo ""
echo "=========================================="
echo " All 12 jobs complete."
echo " Logs        → ${BASE_OUT_DIR}"
echo " Checkpoints → output/decision_maker/{dataset}/ckpts/frac*/{model}/"
echo " Extract:      grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*/*.log"
echo "=========================================="
