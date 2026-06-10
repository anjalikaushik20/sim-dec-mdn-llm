#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export PYTHONPATH="/home/local/ASURITE/Anjali/sim-dec-mdn-llm:$PYTHONPATH"

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/latest_output/training/adaptor/qwen_3_0.6B/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

echo "Run ID: ${RUN_ID}"
echo "Logs saved in: ${BASE_OUT_DIR}"

# DataCo
python3 main/cb_main_llm.py --use_gpu 1 --dataset DataCo --epochs 6000 --train_mode 0 --wandb 1 --hf_model_name "Qwen/Qwen3-0.6B" --save 1 > "${BASE_OUT_DIR}/dataco.log" 2>&1 &

# GlobalStore
python3 main/cb_main_llm.py --use_gpu 1 --dataset GlobalStore --epochs 6000 --train_mode 0 --wandb 1 --hf_model_name "Qwen/Qwen3-0.6B" --save 1 > "${BASE_OUT_DIR}/globalstore.log" 2>&1 &

# OAS
python3 main/cb_main_llm.py --use_gpu 1 --dataset OAS --epochs 6000 --train_mode 0 --wandb 1 --hf_model_name "Qwen/Qwen3-0.6B" --save 1 > "${BASE_OUT_DIR}/oas.log" 2>&1 &

wait