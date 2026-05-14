# Entry point for the LORA SFT decision-making path.
# Trains LLM backbone (Qwen2.5-0.5B) with LoRA + adapters + cls_head.
# Best-Action Cross-Entropy training signal from FAISS + S_SimDec.
# Run example:
#   python cb_main_llm_lora.py --lora_r 16 --lora_alpha 32 \
#       --dm_lr 0.0001 --dm_epochs 50 --batch_size 64

import sys
import os
import argparse
import time
import torch
import wandb

# Reduce CUDA allocator fragmentation (safe no-op if already set by shell)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from tools.logger import info
from environments.environment import Env
from loaders.s_loader import S_Loader
from models.s_model import S_SimDec
from models.llm_model_lora import LLMLoRAValueNetwork
from sessions.cb_session_llm_lora import CB_Session



def parse_args():
    parser = argparse.ArgumentParser(description="AI4Simulation — LoRA SFT")

    # ----------------------- Device Setting
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    # ------------------------ Training Setting

    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt_start_epoch', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='OAS', choices=['LSCRW', 'DataCo','GlobalStore','OAS', 'DataCo_OOD'])
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--dm_lr', type=float, default=0.0001,
        help='Learning rate for adapters and cls_head; LoRA layers use dm_lr * 0.1 automatically')

    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--dm_epochs', type=int, default=6000)
    parser.add_argument('--eva_interval', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--early_stop', type=int, default=50)

    parser.add_argument('--train_mode', type=int, default=2,
        help='0=train simulator+DM, 1=simulator only, 2=DM only')


    # ------------------------ Model Setting
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--decoder_num_layers', type=int, default=1)
    parser.add_argument('--encoder_num_layers', type=int, default=1)
    parser.add_argument('--dm_eval_limit', type=int, default=None)
    parser.add_argument("--hf_model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name for LLMLoRAValueNetwork backbone")

    # LoRA hyperparameters
    parser.add_argument('--lora_r', type=int, default=16,
        help='LoRA rank — higher = more capacity but more params')
    parser.add_argument('--lora_alpha', type=int, default=32,
        help='LoRA alpha scaling factor — typically set to 2*lora_r')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
        help='Dropout applied inside LoRA layers')
    parser.add_argument('--lora_target_modules', type=str, default=None,
        help='Comma-separated list of module names to apply LoRA to. '
             'If None, auto-detected from model architecture.')

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
       help='Softmax temperature for soft labels. Lower = harder labels.')


    # ----------------------- logger
    parser.add_argument('--wandb', type=int, default=0)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--ckpt_dir', type=str, default=None, help='Override checkpoint save directory')

    return parser.parse_args()



# ----------------------------------- Env Init -----------------------------------------------------------
info('--------------------------------Een Init----------------------------------')
args = parse_args()

# Parse lora_target_modules from comma-separated string to list
if args.lora_target_modules is not None:
    args.lora_target_modules = [m.strip() for m in args.lora_target_modules.split(',')]

my_env = Env(args)


# ----------------------------------- Dataset Init -----------------------------------------------------------
info('--------------------------------Dataset Init------------------------------')
my_loader = S_Loader(my_env)
my_env.feature_classes = my_loader.feature_classes

# ----------------------------------- Model Init -----------------------------------------------------------
info('--------------------------------Model Init--------------------------------')
info(f"hf_model_name arg: {args.hf_model_name}")
info(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, "
     f"target_modules={args.lora_target_modules}")
my_model = S_SimDec(my_env)
if args.ckpt != None:
    my_model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
torch.cuda.empty_cache()
llm_model = LLMLoRAValueNetwork(
    my_env,
    model_name=args.hf_model_name,
    batch_size=args.batch_size,
    lora_r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    lora_target_modules=args.lora_target_modules,
)
torch.cuda.empty_cache()

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
    prof, on_time, pmp, _, test_acc = my_session.dm_test("test")
    print("profit", prof, "on_time", on_time, "pmp", pmp)
    if args.wandb:
        wandb.log({"test/profit": prof, "test/on_time": on_time,
                   "test/pmp_0.1": pmp[0.1], "test/pmp_0.2": pmp[0.2], "test/pmp_0.3": pmp[0.3]})
    print("best_dm_accuracy", my_session.best_dm_accuracy)
    print("best_profit", my_session.best_p)
    print("best_on_time", my_session.best_o)
    print("best_pmp_1", my_session.best_pmp1)
    print("best_pmp_2", my_session.best_pmp2)
    print("best_pmp_3", my_session.best_pmp3)

my_session.test("test")

my_env.close()
