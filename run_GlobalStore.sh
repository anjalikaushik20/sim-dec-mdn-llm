#!/bin/bash
#SBATCH -A grp_yanjiefu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-01:00:00
#SBATCH -p htc
#SBATCH -q public
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH -o output_GlobalStore/sim_dec_mdn_llm.%j.out
#SBATCH -e output_GlobalStore/sim_dec_mdn_llm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akaush39@asu.edu

module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate simenv

python3 main/cb_main.py --use_gpu 1 --dataset GlobalStore --epochs 350 --train_mode 1 --save 1 --embed_dim 64  --decoder_num_layers 1 --lr 0.001  --batch_size 1024 --mdn_components 3 --mdn_temperature 1.0 > output_GlobalStore/v1.out 2>&1

python3 main/cb_main.py --use_gpu 1 --dataset GlobalStore --epochs 350 --train_mode 1 --save 1 --embed_dim 64  --decoder_num_layers 1 --lr 0.0005 --batch_size 1024 --mdn_components 3 --mdn_temperature 1.2 > output_GlobalStore/v2.out 2>&1

python3 main/cb_main.py --use_gpu 1 --dataset GlobalStore --epochs 350 --train_mode 1 --save 1 --embed_dim 128 --decoder_num_layers 1 --lr 0.001  --batch_size 1024 --mdn_components 5 --mdn_temperature 1.0 > output_GlobalStore/v3.out 2>&1

python3 main/cb_main.py --use_gpu 1 --dataset GlobalStore --epochs 350 --train_mode 1 --save 1 --embed_dim 128 --decoder_num_layers 1 --lr 0.0005 --batch_size 1024 --mdn_components 5 --mdn_temperature 1.2 > output_GlobalStore/v4.out 2>&1

python3 main/cb_main.py --use_gpu 1 --dataset GlobalStore --epochs 350 --train_mode 1 --save 1 --embed_dim 128 --decoder_num_layers 2 --lr 0.001  --batch_size 1024 --mdn_components 5 --mdn_temperature 1.0 > output_GlobalStore/v5.out 2>&1

python3 main/cb_main.py --use_gpu 1 --dataset GlobalStore --epochs 350 --train_mode 1 --save 1 --embed_dim 128 --decoder_num_layers 2 --lr 0.0005 --batch_size 1024 --mdn_components 8 --mdn_temperature 1.2 > output_GlobalStore/v6.out 2>&1

python3 main/cb_main.py --use_gpu 1 --dataset GlobalStore --epochs 350 --train_mode 1 --save 1 --embed_dim 256 --decoder_num_layers 1 --lr 0.001  --batch_size 2048 --mdn_components 5 --mdn_temperature 1.0 > output_GlobalStore/v7.out 2>&1

python3 main/cb_main.py --use_gpu 1 --dataset GlobalStore --epochs 350 --train_mode 1 --save 1 --embed_dim 256 --decoder_num_layers 1 --lr 0.0005 --batch_size 2048 --mdn_components 8 --mdn_temperature 1.2 > output_GlobalStore/v8.out 2>&1

python3 main/cb_main.py --use_gpu 1 --dataset GlobalStore --epochs 350 --train_mode 1 --save 1 --embed_dim 256 --decoder_num_layers 2 --lr 0.001  --batch_size 2048 --mdn_components 8 --mdn_temperature 1.0 > output_GlobalStore/v9.out 2>&1

python3 main/cb_main.py --use_gpu 1 --dataset GlobalStore --epochs 350 --train_mode 1 --save 1 --embed_dim 256 --decoder_num_layers 2 --lr 0.0005 --batch_size 2048 --mdn_components 10 --mdn_temperature 1.2 > output_GlobalStore/v10.out 2>&1