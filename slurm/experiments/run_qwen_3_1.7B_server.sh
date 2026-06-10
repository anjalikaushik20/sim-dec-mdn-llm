#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
wandb login

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/latest_output/training/adaptor/qwen_3_1.7B/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

# DataCo
python3 main/cb_main_llm.py --use_gpu 1 --dataset DataCo --train_mode 2 --wandb 1 --hf_model_name "Qwen/Qwen3-1.7B" --save 1 --ckpt exp_report/DataCo/ckpt/mythical-commander-352_epoch218.pth > "${BASE_OUT_DIR}/dataco.log" 2>&1 &

# GlobalStore
python3 main/cb_main_llm.py --use_gpu 1 --dataset GlobalStore --train_mode 2 --wandb 1 --hf_model_name "Qwen/Qwen3-1.7B" --save 1 --ckpt exp_report/GlobalStore/ckpt/legendary-federation-353_epoch202.pth > "${BASE_OUT_DIR}/globalstore.log" 2>&1 &

# OAS
python3 main/cb_main_llm.py --use_gpu 1 --dataset OAS --train_mode 2 --wandb 1 --hf_model_name "Qwen/Qwen3-1.7B" --save 1 --ckpt exp_report/OAS/ckpt/jedi-carrier-353_epoch19.pth > "${BASE_OUT_DIR}/oas.log" 2>&1 &

wait