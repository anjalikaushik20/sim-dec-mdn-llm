# Required installations:
# pip install peft>=0.10.0
# pip install bitsandbytes>=0.43.0   # optional, for 8-bit/4-bit quantization
# pip install transformers>=4.40.0

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from tools import feature_list


class LLMLoRAValueNetwork(nn.Module):
    def __init__(
        self,
        env,
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        batch_size=64,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None,
    ):
        super().__init__()
        self.env = env
        self.batch_size = batch_size

        dataset = self.env.args.dataset
        self.group_dims = [
            len(feature_list.product_info[dataset]),
            len(feature_list.order_info[dataset]),
            len(feature_list.customer_info[dataset]),
            len(feature_list.shipping_info[dataset]),
        ]
        self.feature_dim = sum(self.group_dims)

        dtype = torch.float16 if "cuda" in str(self.env.device) else torch.float32
        # Store for use in forward (avoids PEFT dtype proxy ambiguity)
        self._backbone_dtype = dtype

        # Load base model — AutoModelForCausalLM required for Qwen2.5/Qwen3
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        ).to(self.env.device)

        hidden = self.backbone.config.hidden_size

        # Adapters and cls_head in float32 for numerical stability
        self.adapters = nn.ModuleList([
            nn.Linear(dim, hidden, device=self.env.device, dtype=torch.float32)
            for dim in self.group_dims
        ])
        self.cls_head = nn.Linear(hidden, 4, device=self.env.device, dtype=torch.float32)

        # Apply LoRA — PEFT freezes base weights and marks only LoRA A/B trainable
        target_modules = (
            lora_target_modules
            if lora_target_modules is not None
            else self._get_lora_target_modules(model_name)
        )

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            inference_mode=False,
        )

        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()  # log how many params are trainable
        self.backbone.train()

    def _get_lora_target_modules(self, model_name):
        name = model_name.lower()
        if "qwen" in name:
            # Qwen2/Qwen2.5 attention and FFN projection names
            return ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
        elif "gemma" in name:
            return ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
        elif "llama" in name:
            return ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
        elif "phi" in name:
            return ["q_proj", "k_proj", "v_proj", "dense",
                    "fc1", "fc2"]
        else:
            # Generic fallback — attention projections only
            return ["q_proj", "k_proj", "v_proj", "o_proj"]

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
        inputs_embeds = token_seq.to(dtype=self._backbone_dtype)

        # AutoModelForCausalLM does not expose last_hidden_state directly;
        # request all hidden states and take the final layer's last token.
        out = self.backbone(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        # For causal LLMs the last token has attended over all preceding tokens
        last = out.hidden_states[-1][:, -1, :].to(torch.float32)   # [B, H]
        logits32 = self.cls_head(last)                              # [B, 4]

        return logits32.to(state.dtype)

    def save_lora(self, path):
        """Save only the LoRA weights (small file, not full model)."""
        self.backbone.save_pretrained(path)

    def load_lora(self, path):
        """Load LoRA weights back into the backbone."""
        from peft import PeftModel
        self.backbone = PeftModel.from_pretrained(self.backbone.base_model, path)
