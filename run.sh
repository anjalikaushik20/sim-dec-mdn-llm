#!/bin/bash
#SBATCH -A grp_yanjiefu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-01:00:00
#SBATCH -p htc
#SBATCH -q public
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH -o output/sim_dec_mdn_llm.%j.out
#SBATCH -e output/sim_dec_mdn_llm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akaush39@asu.edu

module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate simenv

wandb login ad8cbad10b421ebaaca5e60024696545d4781023 --relogin

python3 main/cb_main.py --use_gpu 1 --dataset DataCo --epochs 350 --train_mode 1 --mdn_components 1 --mdn_mse_coeff 0.5 --mdn_entropy_coeff 0 --mdn_feedback_alpha 0.0 --wandb 1