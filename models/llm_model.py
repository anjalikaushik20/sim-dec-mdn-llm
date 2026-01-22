# llm_model_new.py  (essentials only)

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from tools import feature_list

# used the following models for experiments:
# Qwen/Qwen2.5-1.5B-Instruct, 1.5B params
# meta-llama/Llama-3.2-1B-Instruct, 1B params - no access
# google/gemma-3-4b-it, 4B params
# microsoft/Phi-4-mini-instruct, 4B params
# Qwen/Qwen3-4B-Instruct-2507, 4B params
# Qwen/Qwen3-1.7B, 1.7B params
# Qwen/Qwen3-VL-Embedding-8B, 8B params
# Qwen/Qwen3-30B-A3B-Instruct-2507, 30B params

class LLMValueNetwork(nn.Module):
    def __init__(self, env, model_name="google/gemma-3-1b-it", batch_size=64):
        super().__init__()
        self.env = env
        self.batch_size = batch_size

        dataset = self.env.args.dataset
        self.feature_dim = len(
            feature_list.product_info[dataset]
            + feature_list.order_info[dataset]
            + feature_list.customer_info[dataset]
            + feature_list.shipping_info[dataset]
        )

        dtype = torch.float16 if ("cuda" in str(self.env.device)) else torch.float32
        cfg = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(self.env.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False  # frozen LLM

        hidden = self.backbone.config.hidden_size

        # Keep head in float32 for numerics; project to backbone dtype only at the boundary
        # Only these two layers are trainable
        self.adapter = nn.Linear(self.feature_dim, hidden).to(self.env.device, dtype=torch.float32)
        self.cls_head = nn.Linear(hidden, 4).to(self.env.device, dtype=torch.float32)

    # <-- remove @torch.no_grad() so the head is trainable
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state32 = state.to(self.env.device, dtype=torch.float32)  # head runs in fp32
        proj32 = self.adapter(state32)                             # [B, H] fp32

        # Cast only the token fed into the frozen LLM to its dtype
        inputs_embeds = proj32.to(self.backbone.dtype).unsqueeze(1)  # [B,1,H]

        out = self.backbone(inputs_embeds=inputs_embeds, use_cache=False, return_dict=True)
        last = out.last_hidden_state[:, -1, :].to(torch.float32)     # back to fp32 for the head
        logits32 = self.cls_head(last)                                # [B,4] fp32

        return logits32.to(state.dtype)  # keep your original API
