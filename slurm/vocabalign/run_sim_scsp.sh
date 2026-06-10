#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/simulator/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

# SupplyChainShipmentPricing
python3 main/cb_main.py --use_gpu 1 --dataset SupplyChainShipmentPricing --epochs 6000 --train_mode 1 --wandb 1 --save 1 --ckpt_dir "${BASE_OUT_DIR}/ckpts/scsp" > "${BASE_OUT_DIR}/scsp.log" 2>&1 &

wait
