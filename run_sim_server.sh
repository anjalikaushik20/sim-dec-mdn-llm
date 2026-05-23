#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/latest_output/simulator/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

# DataCo
python3 -m main.cb_main.py --use_gpu 1 --dataset DataCo --epochs 6000 --train_mode 1 --wandb 1 --save 1 --ckpt_dir "${BASE_OUT_DIR}/ckpts/dataco" > "${BASE_OUT_DIR}/dataco.log" 2>&1 &

# GlobalStore
python3 -m main.cb_main.py --use_gpu 1 --dataset GlobalStore --epochs 6000 --train_mode 1 --wandb 1 --save 1 --ckpt_dir "${BASE_OUT_DIR}/ckpts/globalstore" > "${BASE_OUT_DIR}/globalstore.log" 2>&1 &

# OAS
python3 -m main.cb_main.py --use_gpu 1 --dataset OAS --epochs 6000 --train_mode 1 --wandb 1 --save 1 --ckpt_dir "${BASE_OUT_DIR}/ckpts/oas" > "${BASE_OUT_DIR}/oas.log" 2>&1 &

wait