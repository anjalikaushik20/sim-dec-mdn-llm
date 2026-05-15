# llm_model_new.py  (essentials only)

import torch
import torch.nn as nn
from transformers import AutoModel
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
        # Sizes of each feature group in the order they appear in the state vector
        self.group_dims = [
            len(feature_list.product_info[dataset]),
            len(feature_list.order_info[dataset]),
            len(feature_list.customer_info[dataset]),
            len(feature_list.shipping_info[dataset]),
        ]
        self.feature_dim = sum(self.group_dims)

        dtype = torch.float16 if ("cuda" in str(self.env.device)) else torch.float32
        self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(self.env.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False  # frozen LLM

        hidden = self.backbone.config.hidden_size

        # One adapter per feature group; cls_head reads the last token (attends over all 4)
        # All trainable layers stay in float32 for numerical stability
        self.adapters = nn.ModuleList([
            nn.Linear(dim, hidden).to(self.env.device, dtype=torch.float32)
            for dim in self.group_dims
        ])
        self.cls_head = nn.Linear(hidden, 4).to(self.env.device, dtype=torch.float32)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state32 = state.to(self.env.device, dtype=torch.float32)

        # Split the flat state into per-group slices and project each to hidden dim
        group_embeds = []
        offset = 0
        for adapter, dim in zip(self.adapters, self.group_dims):
            group_slice = state32[:, offset: offset + dim]   # [B, group_dim]
            group_embeds.append(adapter(group_slice))         # [B, H]
            offset += dim

        # Stack into a sequence of 4 tokens so the LLM attends across feature groups
        token_seq = torch.stack(group_embeds, dim=1)          # [B, 4, H]
        inputs_embeds = token_seq.to(self.backbone.dtype)     # match frozen backbone dtype

        out = self.backbone(inputs_embeds=inputs_embeds, use_cache=False, return_dict=True)
        # Mean-pool across all 4 group tokens — every feature group contributes equally
        last = out.last_hidden_state.mean(dim=1).to(torch.float32)  # [B, H]
        logits32 = self.cls_head(last)                              # [B, 4]

        return logits32  # always float32 for numerical stability in loss functions
