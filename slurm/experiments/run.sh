#!/bin/bash
#SBATCH -A grp_yanjiefu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-04:00:00
#SBATCH -p htc
#SBATCH -q public
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH -o output/job.%j.out
#SBATCH -e output/job.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akaush39@asu.edu

module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate simenv

wandb login $WB_LOGIN --relogin

python3 main/cb_main.py --use_gpu 1 --dataset DataCo --epochs 6000 --train_mode 0 --wandb 1 > output/DataCo/vanila01.out 2> output/DataCo/vanila01.err & \
python3 main/cb_main.py --use_gpu 1 --dataset GlobalStore --epochs 6000 --train_mode 0 --wandb 1 > output/GlobalStore/vanila01.out 2> output/GlobalStore/vanila01.err & \
python3 main/cb_main.py --use_gpu 1 --dataset OAS --epochs 6000 --train_mode 0 --wandb 1 > output/OAS/vanila01.out 2> output/OAS/vanila01.err & \
python3 main/cb_main_llm.py --use_gpu 1 --dataset DataCo --epochs 6000 --train_mode 0 --wandb 1 > output/DataCo/Trainable-LLM/LSTM-LLM/lstm_sim_llm_dec02.out 2> output/DataCo/Trainable-LLM/LSTM-LLM/lstm_sim_llm_dec02.err & \
python3 main/cb_main_llm.py --use_gpu 1 --dataset GlobalStore --epochs 6000 --train_mode 0 --wandb 1 > output/GlobalStore/Trainable-LLM/LSTM-LLM/lstm_sim_llm_dec02.out 2> output/GlobalStore/Trainable-LLM/LSTM-LLM/lstm_sim_llm_dec02.err & \
python3 main/cb_main_llm.py --use_gpu 1 --dataset OAS --epochs 6000 --train_mode 0 --wandb 1 > output/OAS/Trainable-LLM/LSTM-LLM/lstm_sim_llm_dec02.out 2> output/OAS/Trainable-LLM/LSTM-LLM/lstm_sim_llm_dec02.err