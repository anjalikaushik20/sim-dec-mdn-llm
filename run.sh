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
#SBATCH --mail-user=pkerhalk@asu.edu

#SBATCH --export=NONE

module load mamba/latest
source activate simenv
python3 main/cb_main.py --use_gpu 1 --dataset DataCo --epochs 300 --train_mode 1