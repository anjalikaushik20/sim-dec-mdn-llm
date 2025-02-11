import sys
sys.path.append('/home/local/ASURITE/haoyueba/AI4Simulation_SuppluChain')

import argparse
import time
import torch
import wandb
from tools.logger import info
from environments.environment import Env
from loaders.s_loader import S_Loader
from AI4Simulation_SuppluChain.models.s_model_8 import S_SimDec
from sessions.s_session import S_Session



def parse_args():   
    parser = argparse.ArgumentParser(description="AI4Simulation")

    # ----------------------- Device Setting
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    # ------------------------ Training Setting
    # parser.add_argument('--ckpt', type=str, default='/home/local/ASURITE/haoyueba/AI4Simulation_SuppluChain/exp_report/DataCo/ckpt/swept-butterfly-129_epoch9999.pth')
    parser.add_argument('--ckpt', type=str, default=None)
    # parser.add_argument('--ckpt_start_epoch', type=int, default=10000)
    parser.add_argument('--ckpt_start_epoch', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='DataCo')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dm_lr', type=float, default=0.0001)

    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--dm_epochs', type=int, default=6000)
    parser.add_argument('--eva_interval', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--early_stop', type=int, default=50)



    # ------------------------ Model Setting
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--decoder_num_layers', type=int, default=1)
    parser.add_argument('--encoder_num_layers', type=int, default=1)
    # parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)


    

    # ----------------------- Regularizer coefficient

    # ----------------------- logger
    parser.add_argument('--wandb', type=int, default=0)
    parser.add_argument('--save', type=int, default=0)

    return parser.parse_args()



# ----------------------------------- Env Init -----------------------------------------------------------
info('--------------------------------Een Init----------------------------------')
args = parse_args()
my_env = Env(args)


# ----------------------------------- Dataset Init -----------------------------------------------------------
info('--------------------------------Dataset Init------------------------------')
my_loader = S_Loader(my_env)
my_env.feature_classes = my_loader.feature_classes

# ----------------------------------- Model Init -----------------------------------------------------------
info('--------------------------------Model Init--------------------------------')
my_model = S_SimDec(my_env)
if args.ckpt != None:
    my_model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))

# ----------------------------------- Session Init -----------------------------------------------------------
info('--------------------------------Session Init------------------------------')
my_session = S_Session(my_env, my_model, my_loader)

# ---------------------------------------- Main -----------------------------------------------------------
info('------------------------------------ Main --------------------------------')

t = time.time()
my_session.train()
info(f'simulator training stage cost time: {time.time() - t}')

my_session.dm_train()
info(f'decision_maker training stage cost time: {time.time() - t}')


# my_env.close()
