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
# Total: 3 × 3 × 4 × 5 = 180 jobs at frac=1.00, all on GPU 2.
# Parallelism is per-backbone based on float32 model size (peak VRAM during cache build):
#   gpt2            (~0.5 GB) — 4 datasets in parallel
#   Qwen3-1.7B      (~7 GB)   — 2 datasets in parallel
#   Phi-4-mini      (~15 GB)  — 1 dataset at a time (sequential)
#
# Env overrides:
#   SEEDS     — space-separated seed list (default: 42 131 521 1009 2027)
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

GPU_ID=2
DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-42 131 521 1009 2027}"
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
declare -A MODEL_PAR=(
    [gpt2]=4
    [qwen3-1.7B]=2
    [phi4-mini-reasoning]=1
)
MODELS=(gpt2 qwen3-1.7B phi4-mini-reasoning)

total=$(( ${#MODELS[@]} * ${#VARIANTS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]} ))
echo "=========================================="
echo " Experiment 4 — Method Ablations (3 backbones)"
echo " Variants: ${VARIANTS[*]}"
echo " Backbones: ${MODELS[*]}"
echo " Datasets: ${DATASETS[*]}"
echo " Seeds: ${SEEDS[*]}"
echo " Frac: 1.00 only"
echo " Total jobs: ${total}"
echo " GPU: ${GPU_ID}  dm_epochs=${DM_EPOCHS}  run_id=${RUN_ID}"
echo "=========================================="

failed=0
job_num=0

for model_tag in "${MODELS[@]}"; do
    hf_name="${MODEL_HF[${model_tag}]}"

    echo ""
    max_par="${MODEL_PAR[${model_tag}]}"
    echo "┌─ ${model_tag}  (max_par=${max_par}, GPU${GPU_ID})  $(date '+%Y-%m-%d %H:%M:%S') ─────────"

    for VARIANT in "${VARIANTS[@]}"; do
        for SEED in "${SEEDS[@]}"; do

            batch_pids=()
            for DATASET in "${DATASETS[@]}"; do
                DS_LOWER=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
                DS_TAG="${DS_LOWER/supplychainshipmentpricing/scsp}"
                log_dir="${BASE_OUT_DIR}/${VARIANT}/${model_tag}/seed${SEED}"
                mkdir -p "${log_dir}"
                log="${log_dir}/${DS_TAG}.log"

                job_num=$(( job_num + 1 ))
                echo "  [${job_num}/${total}] GPU${GPU_ID} → ${model_tag}  ${VARIANT}  ${DATASET}  seed=${SEED}  $(date '+%H:%M:%S')"

                extra_flags=""
                extra_args=""
                case "${VARIANT}" in
                    no_vocab_init)    extra_flags="--pool_init random" ;;
                    mean_pool)        extra_flags="--pool_type mean" ;;
                    hard_labels_only) extra_flags="--no_soft_labels" ;;
                esac
                case "${DATASET}" in
                    DataCo)                     extra_args="--otr_reward_coeff 2   --ckpt ${DATACO_CKPT}" ;;
                    GlobalStore)                extra_args="--otr_reward_coeff 10  --ckpt ${GS_CKPT}" ;;
                    OAS)                        extra_args="--dm_lr 0.00003 --otr_reward_coeff 50 --ckpt ${OAS_CKPT}" ;;
                    SupplyChainShipmentPricing) extra_args="--otr_reward_coeff ${SCSP_OTR} --ckpt ${SCSP_CKPT}" ;;
                esac

                # shellcheck disable=SC2086
                CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 main/cb_main_llm.py \
                    --use_gpu 1 --device_id 0 \
                    --dataset "${DATASET}" --train_mode 2 \
                    --wandb 0 --hf_model_name "${hf_name}" --save 0 \
                    --dm_epochs "${DM_EPOCHS}" --train_frac 1.00 \
                    --seed "${SEED}" \
                    ${extra_args} ${extra_flags} \
                    > "${log}" 2>&1 &
                batch_pids+=($!)

                # flush batch when full
                if (( ${#batch_pids[@]} >= max_par )); then
                    for pid in "${batch_pids[@]}"; do
                        wait "${pid}" || { echo "  [FAIL] ${model_tag} ${VARIANT} ${DATASET} seed=${SEED}" >&2; failed=$(( failed + 1 )); }
                    done
                    batch_pids=()
                fi
            done

            # drain any remaining in partial batch
            for pid in "${batch_pids[@]}"; do
                wait "${pid}" || { echo "  [FAIL] ${model_tag} ${VARIANT} seed=${SEED}" >&2; failed=$(( failed + 1 )); }
            done

        done
    done

    echo "└─ ${model_tag} DONE  $(date '+%H:%M:%S')"
done

echo ""
echo "=========================================="
echo " All ${total} ablation jobs complete (${failed} failures)."
echo " Logs → ${BASE_OUT_DIR}/{variant}/{model_tag}/seed{N}/{dataset}.log"
echo " Extract: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*/*/seed*/*.log"
echo "=========================================="
