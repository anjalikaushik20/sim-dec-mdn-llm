import sys
sys.path.append('/home/local/ASURITE/haoyueba/AI4Simulation_SuppluChain')

import argparse
import time
import torch
import wandb
from tools.logger import info
from environments.environment import Env
from loaders.s_loader import S_Loader
from models.s_model import S_SimDec
from models.v_model import ValueNetwork
from sessions.cb_session import CB_Session



def parse_args():   
    parser = argparse.ArgumentParser(description="AI4Simulation")

    # ----------------------- Device Setting
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    # ------------------------ Training Setting
    # DataCo
    # parser.add_argument('--ckpt', type=str, default='/home/local/ASURITE/haoyueba/AI4Simulation_SuppluChain/exp_report/DataCo/ckpt/01-17-15_DataCo_epoch817.pth')

    # LSCRW
    # parser.add_argument('--ckpt', type=str, default='/home/local/ASURITE/haoyueba/AI4Simulation_SuppluChain/exp_report/LSCRW/ckpt/01-17-14_LSCRW_epoch57.pth')

    # GlobalStore
    # parser.add_argument('--ckpt', type=str, default='/home/local/ASURITE/haoyueba/AI4Simulation_SuppluChain/exp_report/GlobalStore/ckpt/01-17-14_GlobalStore_epoch476.pth')

    # OAS
    parser.add_argument('--ckpt', type=str, default='/home/local/ASURITE/haoyueba/AI4Simulation_SuppluChain/exp_report/OAS/ckpt/01-17-14_OAS_epoch223.pth')



    # parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt_start_epoch', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='OAS', choices=['LSCRW', 'DataCo','GlobalStore','OAS', 'DataCo_OOD'])
    parser.add_argument('--lr', type=float, default=0.01)

    # parser.add_argument('--mi_lr', type=float, default=0.0001)
    parser.add_argument('--dm_lr', type=float, default=0.01)

    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--dm_epochs', type=int, default=6000)
    parser.add_argument('--eva_interval', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=5096)

    parser.add_argument('--early_stop', type=int, default=50)

    parser.add_argument('--train_mode', type=int, default=2, help='0 means traning both simulator and decision-maker, 1 means training simulator only, 2 means training decision-maker only')


    # ------------------------ Model Setting
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--decoder_num_layers', type=int, default=1)
    parser.add_argument('--encoder_num_layers', type=int, default=1)
    # parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)


    # ----------------------- Regularizer coefficient
    parser.add_argument('--decay_coeff', type=float, default=0.00001)
    parser.add_argument('--dm_decay_coeff', type=float, default=0.0005)

    # parser.add_argument('--gl_coeff', type=float, default=1)

    parser.add_argument('--mi_coeff', type=float, default=10)
    parser.add_argument('--ma_coeff', type=float, default=1)

    parser.add_argument('--otr_reward_coeff', type=float, default=100)

    parser.add_argument('--reward_smoothing_factor', type=float, default=0.5)

    parser.add_argument('--mip_coeff', type=float, default=10)
    parser.add_argument('--mil_coeff', type=float, default=10)


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
v_model = ValueNetwork(my_env)
# ----------------------------------- Session Init -----------------------------------------------------------
info('--------------------------------Session Init------------------------------')
my_session = CB_Session(my_env, my_model, my_loader)
my_session.init_value_network(v_model)
# ---------------------------------------- Main -----------------------------------------------------------
info('------------------------------------ Main --------------------------------')

t = time.time()
if my_env.args.train_mode == 0 or my_env.args.train_mode == 1:
    my_session.train()
    info(f'simulator training stage cost time: {time.time() - t}')
    info(f'best_acc1 {my_session.best_acc1}')
    info(f'best_acc2 {my_session.best_acc2}')
    info(f'best_acc3 {my_session.best_acc3}')
    info(f'best_overall_accuracy {my_session.best_overall_accuracy}')

if my_env.args.train_mode == 0 or my_env.args.train_mode == 2:
    my_session.dm_train()
    info(f'decision_maker training stage cost time: {time.time() - t}')
    info(f'best_dm_accuracy {my_session.best_dm_accuracy}')
    info(f'best_profit {my_session.best_p}')
    info(f'best_on_time {my_session.best_o}')
    info(f'best_pmp_1 {my_session.best_pmp1}')
    info(f'best_pmp_2 {my_session.best_pmp2}') 
    info(f'best_pmp_3 {my_session.best_pmp3}')
# my_env.close()
