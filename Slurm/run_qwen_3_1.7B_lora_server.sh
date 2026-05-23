#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate simenv

export WANDB_API_KEY="$WB_LOGIN"
export HF_TOKEN="$HF_LOGIN"
wandb login

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/latest_output/training/lora/qwen_3_1.7B/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

# Number of SFT epochs — override via: DM_EPOCHS=200 ./run_qwen_3_1.7B_lora_server.sh
# or pass as first positional arg: ./run_qwen_3_1.7B_lora_server.sh 200
DM_EPOCHS="${1:-${DM_EPOCHS:-500}}"
# OAS is a small dataset — use a lower LR to prevent NaN loss (override via OAS_DM_LR=...)
OAS_DM_LR="${OAS_DM_LR:-0.00003}"
echo "Running LoRA SFT for ${DM_EPOCHS} epochs (OAS dm_lr=${OAS_DM_LR})"

# DataCo
python3 main/cb_main_llm_lora.py --use_gpu 1 --dataset DataCo --train_mode 2 --wandb 1 --hf_model_name "Qwen/Qwen3-1.7B" --save 1 --dm_epochs "${DM_EPOCHS}" --ckpt exp_report/DataCo/ckpt/mythical-commander-352_epoch218.pth > "${BASE_OUT_DIR}/dataco.log" 2>&1 &

# GlobalStore
python3 main/cb_main_llm_lora.py --use_gpu 1 --dataset GlobalStore --train_mode 2 --wandb 1 --hf_model_name "Qwen/Qwen3-1.7B" --save 1 --dm_epochs "${DM_EPOCHS}" --ckpt exp_report/GlobalStore/ckpt/legendary-federation-353_epoch202.pth > "${BASE_OUT_DIR}/globalstore.log" 2>&1 &

# OAS — lower dm_lr to prevent NaN loss on this smaller dataset
python3 main/cb_main_llm_lora.py --use_gpu 1 --dataset OAS --train_mode 2 --wandb 1 --hf_model_name "Qwen/Qwen3-1.7B" --save 1 --dm_epochs "${DM_EPOCHS}" --dm_lr "${OAS_DM_LR}" --ckpt exp_report/OAS/ckpt/jedi-carrier-353_epoch19.pth > "${BASE_OUT_DIR}/oas.log" 2>&1 &

wait