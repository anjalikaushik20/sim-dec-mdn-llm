#!/bin/bash
# Parallel zero-shot sweep — all models × 4 datasets, dm_epochs=0.
# Model groups run sequentially (one backbone loaded at a time); within each group
# all 4 dataset jobs run in parallel across GPUs 0-3.
# Env overrides:
#   SCSP_CKPT — path to SupplyChainShipmentPricing simulator checkpoint
#   SCSP_OTR  — otr_reward_coeff for SupplyChainShipmentPricing (default: 2)
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/decision_maker/zeroshot/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

GPU_ID=1
SCSP_CKPT="${SCSP_CKPT:-output/simulator/latest_run/ckpts/scsp/best.pth}"
SCSP_OTR="${SCSP_OTR:-2}"

echo "=========================================="
echo " Zero-shot evaluation — all models (vocabalign)"
echo " dm_epochs=0  run_id=${RUN_ID}"
echo " GPU: ${GPU_ID}"
echo " Execution: sequential — one job at a time"
echo " Jobs: 7 models × 4 datasets = 28"
echo "=========================================="

MODELS=(
    # "qwen3-0.6B:Qwen/Qwen3-0.6B"
    # "qwen3-1.7B:Qwen/Qwen3-1.7B"
    # "qwen3-4B:Qwen/Qwen3-4B"
    "phi4-mini:microsoft/Phi-4-mini-reasoning"
    # "gpt2:gpt2"
    # "gpt2-medium:gpt2-medium"
    # "gpt2-large:gpt2-large"
)
DATASETS=(DataCo GlobalStore OAS)
# DATASETS=(SupplyChainShipmentPricing)

total=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))
job_num=0
failed=0

for MODEL_ENTRY in "${MODELS[@]}"; do
    model_tag="${MODEL_ENTRY%%:*}"
    hf_name="${MODEL_ENTRY##*:}"
    log_dir="${BASE_OUT_DIR}/${model_tag}"
    mkdir -p "${log_dir}"

    echo ""
    echo "┌─ ${model_tag}  $(date '+%Y-%m-%d %H:%M:%S') ─────────"

    for DATASET in "${DATASETS[@]}"; do
        job_num=$(( job_num + 1 ))
        DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
        DS_TAG="${DS_LOWER/supplychainshipmentpricing/scsp}"
        log="${log_dir}/${DS_TAG}.log"

        local_extra=""
        case "${DATASET}" in
            DataCo)
                local_extra="--otr_reward_coeff 2 \
                    --ckpt output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth" ;;
            GlobalStore)
                local_extra="--otr_reward_coeff 10 \
                    --ckpt output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth" ;;
            OAS)
                local_extra="--dm_lr 0.00003 --otr_reward_coeff 50 \
                    --ckpt output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth" ;;
            SupplyChainShipmentPricing)
                local_extra="--otr_reward_coeff ${SCSP_OTR} --ckpt ${SCSP_CKPT}" ;;
        esac

        echo "  [${job_num}/${total}] GPU${GPU_ID} → ${model_tag}  ${DATASET}  $(date '+%H:%M:%S')"

        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 main/cb_main_llm.py \
            --use_gpu 1 --device_id 0 \
            --dataset "${DATASET}" \
            --train_mode 2 \
            --wandb 0 \
            --hf_model_name "${hf_name}" \
            --save 0 \
            --dm_epochs 0 \
            ${local_extra} \
            > "${log}" 2>&1
        rc=$?
        if [ "${rc}" -ne 0 ]; then
            echo "  [FAIL rc=${rc}] ${model_tag} ${DATASET}" >&2
            failed=$(( failed + 1 ))
        fi
    done

    echo "└─ ${model_tag} DONE  $(date '+%H:%M:%S')"
done

echo ""
echo "=========================================="
if [ "${failed}" -gt 0 ]; then
    echo " Finished with ${failed} failure(s) — check logs in ${BASE_OUT_DIR}/" >&2
fi
echo " All ${total} zero-shot jobs complete."
echo " Logs        → ${BASE_OUT_DIR}"
echo " Extract:      grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*/*.log"
echo "=========================================="
