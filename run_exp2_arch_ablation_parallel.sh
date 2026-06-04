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
#   SEEDS     — space-separated seed list (default: 42 0 1 2 3)
#   SCSP_CKPT — path to SupplyChainShipmentPricing simulator checkpoint
#   SCSP_OTR  — otr_reward_coeff for SCSP (default: 2)
#
# Usage: bash run_exp2_arch_ablation_parallel.sh [DM_EPOCHS]
#   DM_EPOCHS defaults to 200.

set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DM_EPOCHS="${1:-${DM_EPOCHS:-200}}"
NUM_GPUS=2
GPUS=(1 2)   # was (1, 2) — comma made element "1," which was invalid

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
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-42 0 1 2 3}"

echo "=========================================="
echo " Experiment 3 — Architecture Ablation"
echo " dm_epochs=${DM_EPOCHS}  run_id=${RUN_ID}"
echo " Seeds: ${SEEDS[*]}"
echo " GPUs: ${GPUS[*]}"
echo " Baselines: serialized_mlp, bert"
echo " Jobs: 2 × ${#FRACS[@]} fracs × ${#DATASETS[@]} datasets × ${#SEEDS[@]} seeds = $(( 2 * ${#FRACS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]} ))"
echo " Output → ${BASE_OUT_DIR}"
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
# Args: model_type dataset frac seed gpu_id log ckpt_dir
launch_job() {
    local model_type="$1" dataset="$2" frac="$3" seed="$4" gpu_id="$5" log="$6" ckpt_dir="$7"
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

    # serialized_mlp needs no --hf_model_name; bert uses bert-base-uncased
    local hf_arg=""
    if [ "${model_type}" = "bert" ]; then
        hf_arg="--hf_model_name bert-base-uncased"
    fi

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
}

# ── Baseline runner ──────────────────────────────────────────────────────────
# run_baseline MODEL_TYPE MAX_PARALLEL
# Runs all (frac × dataset × seed) jobs for one baseline with MAX_PARALLEL concurrency.
run_baseline() {
    local model_type="$1" max_par="$2"
    local total_jobs=$(( ${#FRACS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]} ))
    local log_base="${BASE_OUT_DIR}/${model_type}"
    mkdir -p "${log_base}"

    echo ""
    echo "┌─ ${model_type}  (parallelism=${max_par} across ${NUM_GPUS} GPUs)  $(date '+%Y-%m-%d %H:%M:%S') ─────────"

    sem_init "${max_par}"

    local pids=() job_num=0

    for seed in "${SEEDS[@]}"; do
        for frac in "${FRACS[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                local gpu_id="${GPUS[$(( job_num % NUM_GPUS ))]}"
                local DS_LOWER
                DS_LOWER=$(echo "${dataset}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
                local log_dir="${log_base}/seed${seed}/${DS_LOWER}"
                local ckpt_dir="${BASE_OUT_DIR}/${dataset}/ckpts/frac${frac}/${model_type}/seed${seed}"
                mkdir -p "${log_dir}" "${ckpt_dir}"
                local log="${log_dir}/frac${frac}.log"

                job_num=$(( job_num + 1 ))
                sem_wait

                echo "  [${job_num}/${total_jobs}] GPU${gpu_id} → ${model_type}  ${dataset}  frac=${frac}  seed=${seed}  $(date '+%H:%M:%S')"

                (
                    launch_job "${model_type}" "${dataset}" "${frac}" "${seed}" "${gpu_id}" "${log}" "${ckpt_dir}"
                    local rc=$?
                    if [ "${rc}" -ne 0 ]; then
                        echo "  [FAIL rc=${rc}] ${model_type} ${dataset} frac=${frac} seed=${seed} GPU${gpu_id}" >&2
                    fi
                    sem_post
                ) &
                pids+=($!)

                sleep 1  # stagger W&B / init calls
            done
        done
    done

    local failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=$(( failed + 1 ))
    done

    sem_close

    if [ "${failed}" -gt 0 ]; then
        echo "└─ ${model_type} DONE with ${failed} failure(s)  $(date '+%H:%M:%S')"
    else
        echo "└─ ${model_type} DONE (all ${total_jobs} OK)  $(date '+%H:%M:%S')"
    fi
}

# ── Execute baselines ─────────────────────────────────────────────────────────
# serialized_mlp: no backbone load, ~0.5 GB/job → run all 18 in parallel across 2 GPUs
# bert: ~3 GB/job → floor(49/3)=16 × 2 GPUs = 32; cap at 18
#
# Baselines run sequentially (one at a time) so GPU memory is fully available per baseline.
# GPU 1 has ~31 GB free (17 GB used by other processes).
# serialized_mlp: no backbone, ~0.5 GB/job → 6 parallel is safe
# bert: frozen bert-base-uncased, ~3 GB/job → 3 parallel (9 GB) is safe
run_baseline "serialized_mlp" 6
run_baseline "bert"           3

echo ""
echo "=========================================="
echo " Experiment 2 complete."
echo " Logs        → ${BASE_OUT_DIR}/{model_type}/{dataset}/frac{frac}.log"
echo " Checkpoints → ${BASE_OUT_DIR}/{dataset}/ckpts/frac{frac}/{model_type}/"
echo ""
echo " Extract results:"
echo "   grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/*/*/*frac*.log"
echo ""
echo " Plot (reuse existing sample-efficiency plotter):"
echo "   conda run -n simenv python3 plot_sample_efficiency.py ${BASE_OUT_DIR}"
echo "=========================================="
