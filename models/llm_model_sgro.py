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
        # Sizes of each feature group in the order they appear in the state vector
        self.group_dims = [
            len(feature_list.product_info[dataset]),
            len(feature_list.order_info[dataset]),
            len(feature_list.customer_info[dataset]),
            len(feature_list.shipping_info[dataset]),
        ]
        self.feature_dim = sum(self.group_dims)

        dtype = torch.float16 if ("cuda" in str(self.env.device)) else torch.float32
        cfg = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(self.env.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False  # frozen LLM

        hidden = self.backbone.config.hidden_size

        # One adapter per feature group; all trainable layers stay in float32
        self.adapters = nn.ModuleList([
            nn.Linear(dim, hidden).to(self.env.device, dtype=torch.float32)
            for dim in self.group_dims
        ])
        # Action embedding: one vector per action, same normal init as simulator's embedding
        self.action_embed = nn.Embedding(4, hidden).to(self.env.device, dtype=torch.float32)
        nn.init.normal_(self.action_embed.weight)
        # Scalar Q-head: scores a single (state, action) pair
        self.q_head = nn.Linear(hidden, 1).to(self.env.device, dtype=torch.float32)

    def forward(self, state: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:          [B, D] float tensor of scaled features
            action_indices: [B]    long tensor of action ids in {0,1,2,3}
        Returns:
            [B] float tensor of Q(state, action) scalar scores
        """
        state32 = state.to(self.env.device, dtype=torch.float32)

        # Split the flat state into per-group slices and project each to hidden dim
        group_embeds = []
        offset = 0
        for adapter, dim in zip(self.adapters, self.group_dims):
            group_slice = state32[:, offset: offset + dim]   # [B, group_dim]
            group_embeds.append(adapter(group_slice))         # [B, H]
            offset += dim

        # 4 feature-group tokens
        token_seq = torch.stack(group_embeds, dim=1)          # [B, 4, H]

        # 1 action token appended at the end
        action_emb = self.action_embed(
            action_indices.to(self.env.device, dtype=torch.long)
        )                                                       # [B, H]
        action_token = action_emb.unsqueeze(1)                  # [B, 1, H]

        # Concatenate → [B, 5, H], then cast to backbone dtype
        inputs_embeds = torch.cat([token_seq, action_token], dim=1)  # [B, 5, H]
        inputs_embeds = inputs_embeds.to(self.backbone.dtype)

        out = self.backbone(inputs_embeds=inputs_embeds, use_cache=False, return_dict=True)
        # Last token is the action token; for causal models it has attended over all 4 feature tokens
        last = out.last_hidden_state[:, -1, :].to(torch.float32)   # [B, H]
        q_val = self.q_head(last).squeeze(-1)                       # [B]

        return q_val.to(state.dtype)

    def score_all_actions(self, state: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Q(state, a) for every a in {0,1,2,3}.
        Args:
            state: [B, D]
        Returns:
            [B, 4] Q-value matrix — column a is Q(state, a)
        """
        B = state.shape[0]
        scores = []
        for a in range(4):
            a_idx = torch.full((B,), a, dtype=torch.long, device=self.env.device)
            scores.append(self.forward(state, a_idx))   # [B]
        return torch.stack(scores, dim=1)               # [B, 4]
