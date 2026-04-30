#!/bin/bash
#SBATCH -A grp_yanjiefu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-04:00:00
#SBATCH -p htc
#SBATCH -q public
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH -o output/latest_output/no_training/prompted/qwen_3_1.7B/lstm_sim_llm_dec.%j.out
#SBATCH -e output/latest_output/no_training/prompted/qwen_3_1.7B/lstm_sim_llm_dec.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akaush39@asu.edu

module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate simenv

wandb login $WB_LOGIN --relogin

RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE_OUT_DIR="output/latest_output/no_training/prompted/qwen_3_1.7B/${RUN_ID}"
mkdir -p "${BASE_OUT_DIR}"

# DataCo
python3 -m main.cb_main_llm_prompted.py --use_gpu 1 --dataset DataCo --epochs 6000 --train_mode 0 --wandb 1 --hf_model_name "Qwen/Qwen3-1.7B" --save 1 --ckpt exp_report/DataCo/ckpt/gallant-thunder-328_epoch245.pth > "${BASE_OUT_DIR}/dataco.log" 2>&1 &

# GlobalStore
python3 -m main.cb_main_llm_prompted.py --use_gpu 1 --dataset GlobalStore --epochs 6000 --train_mode 0 --wandb 1 --hf_model_name "Qwen/Qwen3-1.7B" --save 1 --ckpt exp_report/GlobalStore/ckpt/deft-gorge-329_epoch109.pth > "${BASE_OUT_DIR}/globalstore.log" 2>&1 &

# OAS
python3 -m main.cb_main_llm_prompted.py --use_gpu 1 --dataset OAS --epochs 6000 --train_mode 0 --wandb 1 --hf_model_name "Qwen/Qwen3-1.7B" --save 1 --ckpt exp_report/OAS/ckpt/wild-salad-315_epoch83.pth > "${BASE_OUT_DIR}/oas.log" 2>&1 &

wait