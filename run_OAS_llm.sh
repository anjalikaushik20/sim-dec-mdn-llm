#!/bin/bash
#SBATCH -A grp_yanjiefu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-01:00:00
#SBATCH -p htc
#SBATCH -q public
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH -o output/OAS/Trainable-LLM/MDN-LLM/mdn_sim_llm_dec.%j.out
#SBATCH -e output/OAS/Trainable-LLM/MDN-LLM/mdn_sim_llm_dec.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akaush39@asu.edu

module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate simenv

wandb login $WB_LOGIN --relogin

python3 main/cb_main_llm.py --use_gpu 1 --dataset OAS --epochs 6000 --train_mode 0 --batch_size 64