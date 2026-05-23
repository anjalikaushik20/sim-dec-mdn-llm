#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/latest_output/sample_efficiency/gonogo/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

echo "Go/no-go: DataCo @ train_frac=0.05, dm_epochs=200"
echo "Compare best_profit + best_on_time to full-data baseline to decide whether to run the full matrix."

python3 main/cb_main_llm_attnpool.py \
    --use_gpu 1 --device_id 0 --dataset DataCo --train_mode 2 \
    --wandb 1 --hf_model_name "Qwen/Qwen3-0.6B" --save 0 \
    --dm_epochs 200 --train_frac 0.05 --otr_reward_coeff 2 --save 1 --ckpt_dir "${BASE_OUT_DIR}/ckpts" \
    --ckpt output/latest_output/simulator/latest_run/ckpts/dataco/confused-frog-888_epoch378.pth \
    > "${BASE_OUT_DIR}/dataco_frac0.05.log" 2>&1


echo "Go/no-go complete. Log: ${BASE_OUT_DIR}/dataco_frac0.05.log"
echo "Check: grep 'best_profit\|best_on_time' ${BASE_OUT_DIR}/dataco_frac0.05.log"
