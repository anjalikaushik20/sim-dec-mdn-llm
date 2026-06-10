#!/bin/bash
# Experiment 2 — Architecture ablation (frozen LLM justification).
#
# Runs Serialized MLP and Frozen BERT baselines across the same 6 fracs × 4 datasets × 5 seeds
# grid used for VocabAlign in run_sample_efficiency_all_vocabalign_parallel.sh.
# Compare results against VocabAlign (Qwen3-1.7B) and ML baselines.
#
# Baselines:
#   serialized_mlp  — TF-IDF(max_features=1000, ngram=(1,2)) + 3-layer MLP
#                     Same serialized text as VocabAlign. No backbone.
#                     ~0.5 GB/job; up to 18 parallel per GPU.
#
#   bert            — Frozen bert-base-uncased (110M) + attention pooling head
#                     + VocabAlign init from BERT's MLM head.
#                     ~3 GB/job; up to 14 parallel across 2 GPUs.
#
# Total new jobs: 2 × 6 fracs × 4 datasets × 5 seeds = 240
# (Zero-shot frac=0 handled by run_zeroshot_all_server.sh)
#
# Env overrides:
#   SEEDS     — space-separated seed list (default: 42 131 521 1009 2027)
#   SCSP_CKPT — path to SupplyChainShipmentPricing simulator checkpoint
#   SCSP_OTR  — otr_reward_coeff for SCSP (default: 2)
#
# Usage: bash run_exp2_arch_ablation_parallel.sh [DM_EPOCHS]
#   DM_EPOCHS defaults to 200.

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"
BERT_GPU=1        # sequential BERT jobs on GPU 1
SMLP_GPU=2        # sequential SerializedMLP jobs on GPU 2 (safe: ~0.5 GB, old BERT uses ~2 GB)

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/exp2_arch_ablation/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

DATACO_CKPT="output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"
GS_CKPT="output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
OAS_CKPT="output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"
SCSP_CKPT="${SCSP_CKPT:-output/simulator/latest_run/ckpts/scsp/best.pth}"
SCSP_OTR="${SCSP_OTR:-2}"

FRACS=(0.01 0.05 0.10 0.25 0.50 1.00)
DATASETS=(DataCo GlobalStore OAS SupplyChainShipmentPricing)
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-42 131 521 1009 2027}"

echo "=========================================="
echo " Experiment 3 — Architecture Ablation"
echo " dm_epochs=${DM_EPOCHS}  run_id=${RUN_ID}"
echo " Seeds: ${SEEDS[*]}"
echo " bert → GPU ${BERT_GPU} (sequential)"
echo " serialized_mlp → GPU ${SMLP_GPU} (sequential, shares with old BERT ~2 GB)"
echo " Baselines: serialized_mlp, bert"
echo " Jobs: 2 × ${#FRACS[@]} fracs × ${#DATASETS[@]} datasets × ${#SEEDS[@]} seeds = $(( 2 * ${#FRACS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]} ))"
echo " Output → ${BASE_OUT_DIR}"
echo "=========================================="

per_baseline=$(( ${#FRACS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]} ))
total=$(( 2 * per_baseline ))

# ── Helper: run one baseline sequentially on a given GPU ────────────────────
run_baseline_sequential() {
    local model_type="$1" gpu_id="$2"
    local log_base="${BASE_OUT_DIR}/${model_type}"
    mkdir -p "${log_base}"
    local hf_arg=""
    [ "${model_type}" = "bert" ] && hf_arg="--hf_model_name google-bert/bert-base-uncased"
    local job_num=0 failed=0

    echo ""
    echo "┌─ ${model_type}  (sequential, GPU${gpu_id})  $(date '+%Y-%m-%d %H:%M:%S') ─────────"

    for seed in "${SEEDS[@]}"; do
        for frac in "${FRACS[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                DS_LOWER=$(echo "${dataset}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
                DS_TAG="${DS_LOWER/supplychainshipmentpricing/scsp}"
                log_dir="${log_base}/seed${seed}/${DS_TAG}"
                ckpt_dir="${BASE_OUT_DIR}/${DS_TAG}/ckpts/frac${frac}/${model_type}/seed${seed}"
                mkdir -p "${log_dir}" "${ckpt_dir}"
                log="${log_dir}/frac${frac}.log"

                extra_args=""
                case "${dataset}" in
                    DataCo)      extra_args="--otr_reward_coeff 2 --ckpt ${DATACO_CKPT}" ;;
                    GlobalStore) extra_args="--otr_reward_coeff 10 --ckpt ${GS_CKPT}" ;;
                    OAS)         extra_args="--dm_lr 0.00003 --otr_reward_coeff 50 --ckpt ${OAS_CKPT}" ;;
                    SupplyChainShipmentPricing) extra_args="--otr_reward_coeff ${SCSP_OTR} --ckpt ${SCSP_CKPT}" ;;
                esac

                job_num=$(( job_num + 1 ))
                echo "  [${model_type} ${job_num}/${per_baseline}] GPU${gpu_id} → ${dataset}  frac=${frac}  seed=${seed}  $(date '+%H:%M:%S')"

                # shellcheck disable=SC2086
                CUDA_VISIBLE_DEVICES="${gpu_id}" python3 main/cb_main_llm.py \
                    --use_gpu 1 --device_id 0 \
                    --dataset "${dataset}" \
                    --train_mode 2 \
                    --model_type "${model_type}" \
                    --wandb 0 \
                    --save 1 \
                    --ckpt_dir "${ckpt_dir}" \
                    --dm_epochs "${DM_EPOCHS}" \
                    --train_frac "${frac}" \
                    --seed "${seed}" \
                    ${hf_arg} \
                    ${extra_args} \
                    > "${log}" 2>&1
                rc=$?
                [ "${rc}" -ne 0 ] && { echo "  [FAIL rc=${rc}] ${model_type} ${dataset} frac=${frac} seed=${seed}" >&2; failed=$(( failed + 1 )); }
            done
        done
    done

    echo "└─ ${model_type} DONE (${failed} failures)  $(date '+%H:%M:%S')"
}

# ── Run both baselines in parallel, each sequential on its own GPU ───────────
run_baseline_sequential "bert"           "${BERT_GPU}" &
run_baseline_sequential "serialized_mlp" "${SMLP_GPU}" &
wait

echo ""
echo "=========================================="
[ "${failed}" -gt 0 ] && echo " Finished with ${failed} failure(s) — check logs in ${BASE_OUT_DIR}/" >&2
echo " Experiment 3 (Arch Ablation) complete."
echo " Logs        → ${BASE_OUT_DIR}/{model_type}/{dataset}/frac{frac}.log"
echo " Checkpoints → ${BASE_OUT_DIR}/{dataset}/ckpts/frac{frac}/{model_type}/"
echo ""
echo " Extract results:"
echo "   grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*/*/*frac*.log"
echo ""
echo " Plot (reuse existing sample-efficiency plotter):"
echo "   conda run -n simenv python3 plot_sample_efficiency.py ${BASE_OUT_DIR}"
echo "=========================================="
