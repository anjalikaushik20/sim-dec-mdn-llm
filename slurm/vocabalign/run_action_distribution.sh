#!/bin/bash
# Policy non-degeneracy: run inference (dm_epochs=0) for every model × dataset
# and save per-sample test predictions to CSV.
#
# All 4 datasets for a model run in parallel; concurrency is capped per model
# based on actual VRAM usage (GPU with ~48 GB free):
#   phi4-mini      ~20 GB actual → max 2 concurrent
#   qwen3-1.7B     ~10 GB actual → max 4
#   qwen3-0.6B      ~4 GB actual → max 4
#   gpt2-large      ~3 GB actual → max 4
#   gpt2 / bert / sMLP   < 2 GB → max 4
#
# Skips any (model, dataset) pair whose CSV already exists.
#
# Output: output/tables/policy_nondegeneracy/predictions/{model}/{dataset}_predictions.csv
#
# Usage:
#   bash run_action_distribution.sh
#   bash run_nohup.sh run_action_distribution.sh

set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/akaush39/sim-dec-mdn-llm:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU_ID="${GPU_ID:-0}"

DATACO_SIM="output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth"
GS_SIM="output/simulator/latest_run/ckpts/globalstore/flowing-jazz-888_epoch280.pth"
OAS_SIM="output/simulator/latest_run/ckpts/oas/fiery-sky-888_epoch310.pth"
SCSP_SIM="${SCSP_SIM:-output/simulator/latest_run/ckpts/scsp/best.pth}"

OUT_BASE="output/tables/policy_nondegeneracy/predictions"
mkdir -p "${OUT_BASE}"

DATASETS=(DataCo GlobalStore OAS SupplyChainShipmentPricing)

declare -A DS_SIM=(
    [DataCo]="${DATACO_SIM}"
    [GlobalStore]="${GS_SIM}"
    [OAS]="${OAS_SIM}"
    [SupplyChainShipmentPricing]="${SCSP_SIM}"
)
declare -A DS_OTR=( [DataCo]=2 [GlobalStore]=10 [OAS]=50 [SupplyChainShipmentPricing]=2 )
declare -A DS_LR=(  [DataCo]=0.01 [GlobalStore]=0.01 [OAS]=0.00003 [SupplyChainShipmentPricing]=0.01 )

# ── Semaphore ─────────────────────────────────────────────────────────────────
sem_init()  { mkfifo "/tmp/sem_${$}_$1"; exec 200<>"/tmp/sem_${$}_$1"; rm "/tmp/sem_${$}_$1"; local n=$1; for ((i=0;i<n;i++)); do echo >&200; done; }
sem_wait()  { read -u200; }
sem_post()  { echo >&200; }
sem_close() { exec 200>&- 2>/dev/null || true; }
trap 'sem_close; wait' EXIT

# ── Helpers ───────────────────────────────────────────────────────────────────
find_adapter() {
    local model="$1" dataset="$2"
    local ckpt_ds
    case "${dataset}" in
        SupplyChainShipmentPricing) ckpt_ds="SCSP" ;;
        *) ckpt_ds="${dataset}" ;;
    esac
    # Primary: consolidated checkpoints dir
    local p
    p=$(find "output/decision_maker/checkpoints/${ckpt_ds}/${model}/frac1.00" \
            -name "*_attnpool_best.pth" 2>/dev/null | sort | tail -1)
    [ -n "$p" ] && { echo "$p"; return 0; }
    # Fallback: arch ablation dirs (bert / serialized_mlp)
    p=$(find "output/exp2_arch_ablation" \
            -path "*/${dataset}/ckpts/frac1.00/${model}/*_attnpool_best.pth" \
            2>/dev/null | sort | tail -1)
    [ -n "$p" ] && echo "$p"
}

run_inference() {
    local model="$1" dataset="$2" extra_args="$3"
    local adapter
    adapter=$(find_adapter "${model}" "${dataset}")
    if [ -z "${adapter}" ]; then
        echo "  [SKIP] ${model} × ${dataset}: no frac1.00 adapter"
        return
    fi
    local ds_tag
    ds_tag=$(echo "${dataset}" | tr '[:upper:]' '[:lower:]' | sed 's/supplychainshipmentpricing/scsp/')
    local out_dir="${OUT_BASE}/${model}"
    mkdir -p "${out_dir}"
    local pred_csv="${out_dir}/${ds_tag}_predictions.csv"
    if [ -f "${pred_csv}" ]; then
        echo "  [SKIP] ${model} × ${dataset}: already exists"
        return
    fi
    echo "  [RUN] GPU${GPU_ID} → ${model} × ${dataset}  $(date '+%H:%M:%S')"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 main/cb_main_llm.py \
        --use_gpu 1 --device_id 0 \
        --dataset "${dataset}" --train_mode 2 \
        --wandb 0 --save 0 --dm_epochs 0 --seed 42 \
        --otr_reward_coeff "${DS_OTR[${dataset}]}" \
        --dm_lr "${DS_LR[${dataset}]}" \
        --ckpt "${DS_SIM[${dataset}]}" \
        --value_network_ckpt "${adapter}" \
        --save_predictions "${pred_csv}" \
        ${extra_args} \
        > "${out_dir}/${ds_tag}_inference.log" 2>&1
    local rc=$?
    [ "${rc}" -ne 0 ] \
        && echo "  [FAIL rc=${rc}] ${model} × ${dataset}" >&2 \
        || echo "  [OK]  ${model} × ${dataset} → ${pred_csv}"
}

# Run all 4 datasets for one model in parallel, capped at max_par
run_model_group() {
    local model="$1" max_par="$2" extra_args="$3"
    echo ""
    echo "┌─ ${model}  (max_par=${max_par})  $(date '+%Y-%m-%d %H:%M:%S')"
    sem_init "${max_par}"
    local pids=()
    for dataset in "${DATASETS[@]}"; do
        sem_wait
        (
            run_inference "${model}" "${dataset}" "${extra_args}"
            sem_post
        ) &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "${pid}"; done
    sem_close
    echo "└─ ${model} DONE  $(date '+%H:%M:%S')"
}

echo "=========================================="
echo " Policy non-degeneracy inference"
echo " GPU: ${GPU_ID}   Output: ${OUT_BASE}"
echo "=========================================="

# Heaviest first so VRAM pressure is front-loaded
run_model_group "phi4-mini"      2 "--model_type llm_attn --hf_model_name microsoft/Phi-4-mini-reasoning"
run_model_group "qwen3-1.7B"     4 "--model_type llm_attn --hf_model_name Qwen/Qwen3-1.7B"
run_model_group "qwen3-0.6B"     4 "--model_type llm_attn --hf_model_name Qwen/Qwen3-0.6B"
run_model_group "gpt2-large"     4 "--model_type llm_attn --hf_model_name gpt2-large"
run_model_group "gpt2"           4 "--model_type llm_attn --hf_model_name gpt2"
run_model_group "bert"           4 "--model_type bert"
run_model_group "serialized_mlp" 4 "--model_type serialized_mlp"

echo ""
echo "=========================================="
echo " All done. Predictions → ${OUT_BASE}/{model}/{dataset}_predictions.csv"
echo " Next: conda run -n simenv python3 table_policy_nondegeneracy.py"
echo "=========================================="
