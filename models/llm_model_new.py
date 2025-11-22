# models/llm_model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from tools import feature_list

class LLMValueNetwork(nn.Module):
    def __init__(self, env, model_name="google/gemma-3-1b-it", batch_size=64):
        super().__init__()
        self.env = env
        self.batch_size = batch_size

        # ---- compute numeric feature_dim from your feature_list ----
        dataset = self.env.args.dataset
        self.feature_dim = len(
            feature_list.product_info[dataset]
            + feature_list.order_info[dataset]
            + feature_list.customer_info[dataset]
            + feature_list.shipping_info[dataset]
        )

        # ---- load frozen LLM backbone (no tokenizer needed when using inputs_embeds) ----
        dtype = torch.float16 if ("cuda" in str(self.env.device)) else torch.float32
        cfg = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype
        )
        self.backbone.to(self.env.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False  # keep it frozen

        hidden = self.backbone.config.hidden_size

        # ---- light-weight adapter + classifier head ----
        self.adapter = nn.Linear(self.feature_dim, hidden)
        self.cls_head = nn.Linear(hidden, 4)
        
        self.adapter.to(self.env.device, dtype=self.backbone.dtype)
        self.cls_head.to(self.env.device, dtype=self.backbone.dtype)

    @torch.no_grad()
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # make sure state matches backbone’s device/dtype
        state_local = state.to(self.env.device, dtype=self.backbone.dtype)

        # [B, H]
        proj = self.adapter(state_local)
        inputs_embeds = proj.unsqueeze(1)  # [B, 1, H]  (same device/dtype)

        out = self.backbone(inputs_embeds=inputs_embeds, use_cache=False, return_dict=True)
        last = out.last_hidden_state[:, -1, :]            # [B, H], backbone dtype
        logits = self.cls_head(last)                      # [B, 4], backbone dtype

        # return logits in the original state's dtype (what the rest of your code expects)
        return logits.to(state.dtype)
