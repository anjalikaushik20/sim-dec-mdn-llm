import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
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
        self.group_dims = [
            len(feature_list.product_info[dataset]),
            len(feature_list.order_info[dataset]),
            len(feature_list.customer_info[dataset]),
            len(feature_list.shipping_info[dataset]),
        ]
        self.feature_dim = sum(self.group_dims)

        # Flat list of feature names and per-group slice boundaries for serialization
        self.feature_names = (
            feature_list.product_info[dataset]
            + feature_list.order_info[dataset]
            + feature_list.customer_info[dataset]
            + feature_list.shipping_info[dataset]
        )
        self._group_labels = ["Product", "Order", "Customer", "Shipping"]
        self._group_slices = []
        offset = 0
        for dim in self.group_dims:
            self._group_slices.append((offset, offset + dim))
            offset += dim

        dtype = torch.float16 if ("cuda" in str(self.env.device)) else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(self.env.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False  # frozen LLM

        hidden = self.backbone.config.hidden_size
        # Only the classification head is trainable — backbone is fully frozen
        self.cls_head = nn.Linear(hidden, 4).to(self.env.device, dtype=torch.float32)

    def serialize_batch(self, raw_state: torch.Tensor) -> list:
        """Convert a batch of raw (unscaled) feature vectors to natural-language strings.

        Each feature group becomes a comma-separated key=value clause so the LLM
        can use its pretrained knowledge of feature names like 'Order Quantity' or
        'Discount %' to reason about the best shipping action.
        """
        raw_np = raw_state.detach().cpu().numpy()
        texts = []
        for row in raw_np:
            parts = []
            for label, (s, e) in zip(self._group_labels, self._group_slices):
                names = self.feature_names[s:e]
                vals = row[s:e]
                kv = ", ".join(f"{n}={v:.3g}" for n, v in zip(names, vals))
                parts.append(f"{label}: {kv}")
            texts.append(". ".join(parts) + ". Predict optimal shipping action.")
        return texts

    def forward(self, raw_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_state: [B, feature_dim] unscaled feature tensor
        Returns:
            [B, 4] float32 logits
        """
        texts = self.serialize_batch(raw_state)
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(self.env.device)
        attention_mask = enc["attention_mask"].to(self.env.device)

        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        # Mean-pool over non-padding token positions so all input tokens contribute equally
        hidden = out.last_hidden_state                                    # [B, T, H]
        mask = attention_mask.unsqueeze(-1).float()                      # [B, T, 1]
        pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # [B, H]
        return self.cls_head(pooled)                                     # [B, 4]
