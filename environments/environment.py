import sys
import os
import time
import numpy as np
import torch
import wandb
from transformers import BertTokenizer
from tools.logger import info

class Env(object):
    def __init__(self, args):
        self.args = args
        self.DATA_PATH = '' # data path
        self.ROOT_PATH = '' # code path

        self.DATA_PATH = os.path.join(self.DATA_PATH, self.args.dataset)
        self.BASE_PATH = os.path.join(self.ROOT_PATH, 'exp_report')
        self.BASE_PATH = os.path.join(self.BASE_PATH, self.args.dataset)
        self.CKPT_PATH = os.path.join(self.BASE_PATH, 'ckpt')
        self.TEMP_PATH = os.path.join(self.BASE_PATH, 'temp')
        self.reset(args)
        

    def reset(self, args):
        self.args = args
        self.time_stamp = time.strftime('%m-%d-%H', time.localtime(time.time()))
        self._check_direcoty()
        self._init_device()
        # self._init_tokenizer()
        self._set_seed(self.args.seed)

        if self.args.wandb:
            self._init_wandb()
        
        self.suffix = wandb.run.name if self.args.wandb else f'{self.time_stamp}_{self.args.dataset}'

    def close(self):
        if self.args.wandb:
            wandb.finish()
        else:
            pass

    def _check_direcoty(self):
        if not os.path.exists(self.BASE_PATH):
            os.makedirs(self.BASE_PATH, exist_ok=True)
        if not os.path.exists(self.CKPT_PATH):
            os.makedirs(self.CKPT_PATH, exist_ok=True)


    def _init_tokenizer(self):
        if os.path.exists(os.path.join(self.DATA_PATH, f'tokenizer')):
            self.tokenizer = BertTokenizer.from_pretrained(os.path.join(self.DATA_PATH, f'tokenizer'))
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def _init_device(self):
        if torch.cuda.is_available() and self.args.use_gpu:
            self.device = torch.device(self.args.device_id)
        else:
            self.device = 'cpu'
        info(f'Code will run in {self.device}')

    def _init_wandb(self):
        wandb.init(project="",config=self.args)



    def _set_seed(self, seed):
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
