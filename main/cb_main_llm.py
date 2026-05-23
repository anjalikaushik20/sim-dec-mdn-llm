import sys
import os
import glob
import argparse
import logging
import time
import torch
import wandb
from tools.logger import info
from environments.environment import Env
from loaders.s_loader import S_Loader
from models.s_model import S_SimDec
from models.llm_model import LLMAttnPoolNetwork
from sessions.cb_session_llm import CB_Session



def parse_args():
    parser = argparse.ArgumentParser(description="AI4Simulation")

    # ----------------------- Device Setting
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    # ------------------------ Training Setting

    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt_start_epoch', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='OAS', choices=['LSCRW', 'DataCo','GlobalStore','OAS', 'DataCo_OOD'])
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--dm_lr', type=float, default=0.01)

    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--dm_epochs', type=int, default=6000)
    parser.add_argument('--eva_interval', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--early_stop', type=int, default=50)

    parser.add_argument('--train_mode', type=int, default=2, help='0 means traning both simulator and decision-maker, 1 means training simulator only, 2 means training decision-maker only')


    # ------------------------ Model Setting
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--decoder_num_layers', type=int, default=1)
    parser.add_argument('--encoder_num_layers', type=int, default=1)
    parser.add_argument('--dm_eval_limit', type=int, default=None)
    parser.add_argument('--train_frac', type=float, default=1.0,
        help='Fraction of training data to use for attnpool head training. 1.0 = full dataset.')
    parser.add_argument("--hf_model_name", type=str, default="google/gemma-3-1b-it", help="HuggingFace model name for LLMAttnPoolNetwork backbone")


    # ----------------------- Regularizer coefficient
    parser.add_argument('--decay_coeff', type=float, default=0.00001)
    parser.add_argument('--dm_decay_coeff', type=float, default=0.0005)

    parser.add_argument('--mi_coeff', type=float, default=10)
    parser.add_argument('--ma_coeff', type=float, default=1)

    parser.add_argument('--otr_reward_coeff', type=float, default=1)

    parser.add_argument('--reward_smoothing_factor', type=float, default=0.5)

    parser.add_argument('--mip_coeff', type=float, default=1)
    parser.add_argument('--mil_coeff', type=float, default=1)

    parser.add_argument('--soft_label_temp', type=float, default=1.0,
        help='Temperature for soft reward targets. Lower = harder labels.')

    # ----------------------- logger
    parser.add_argument('--wandb', type=int, default=0)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--ckpt_dir', type=str, default=None, help='Override checkpoint save directory')

    return parser.parse_args()



# ----------------------------------- Logging Setup -----------------------------------------------------------
os.makedirs("run_logs", exist_ok=True)
_log_ts = time.strftime("%Y%m%d_%H%M%S")

# ----------------------------------- Env Init -----------------------------------------------------------
info('--------------------------------Een Init----------------------------------')
args = parse_args()

_log_path = f"run_logs/train_{args.dataset}_{_log_ts}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    handlers=[
        logging.FileHandler(_log_path),
        logging.StreamHandler(),
    ],
    force=True,
)
info(f"Log file: {_log_path}")

my_env = Env(args)


# ----------------------------------- Dataset Init -----------------------------------------------------------
info('--------------------------------Dataset Init------------------------------')
my_loader = S_Loader(my_env)
my_env.feature_classes = my_loader.feature_classes

# ----------------------------------- Model Init -----------------------------------------------------------
info('--------------------------------Model Init--------------------------------')
info(f"hf_model_name arg: {args.hf_model_name}")
my_model = S_SimDec(my_env)
if args.ckpt is not None:
    ckpt_path = args.ckpt
    if os.path.isdir(ckpt_path):
        pth_files = sorted(glob.glob(os.path.join(ckpt_path, "*.pth")), key=os.path.getmtime)
        if not pth_files:
            raise FileNotFoundError(f"No .pth files found in --ckpt directory: {ckpt_path}")
        ckpt_path = pth_files[-1]
        info(f"Auto-selected latest checkpoint: {ckpt_path}")
    my_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
llm_model = LLMAttnPoolNetwork(my_env, model_name=args.hf_model_name)
info("Training: frozen backbone → learned attention pooling → vocab-aligned cls_head (4 logits) | Inference: forward() + argmax")
# ----------------------------------- Session Init -----------------------------------------------------------
info('--------------------------------Session Init------------------------------')
my_session = CB_Session(my_env, my_model, my_loader)
my_session.init_value_network(llm_model)
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
    prof, on_time, pmp, _, acc_hist, acc_true = my_session.dm_test("test")
    _acc_true_str = f"{acc_true:.4f}" if acc_true is not None else "N/A"
    info(f"[TEST] profit={prof:.4f} on_time={on_time:.4f} acc_hist={acc_hist:.4f} acc_true={_acc_true_str} pmp={pmp}")
    info(f"best_dm_accuracy={my_session.best_dm_accuracy:.4f}")
    info(f"best_dm_accuracy_true={my_session.best_dm_accuracy_true:.4f}")
    info(f"best_profit={my_session.best_p:.4f}")
    info(f"best_on_time={my_session.best_o:.4f}")
    info(f"best_pmp_1={my_session.best_pmp1:.4f}")
    info(f"best_pmp_2={my_session.best_pmp2:.4f}")
    info(f"best_pmp_3={my_session.best_pmp3:.4f}")
    if args.wandb:
        wandb.log({"test/profit": prof, "test/on_time": on_time,
                   "test/acc_hist": acc_hist if acc_hist is not None else 0.0,
                   "test/acc_true": acc_true if acc_true is not None else 0.0,
                   "test/pmp_0.1": pmp[0.1], "test/pmp_0.2": pmp[0.2], "test/pmp_0.3": pmp[0.3]})

my_session.test("test")

my_env.close()
