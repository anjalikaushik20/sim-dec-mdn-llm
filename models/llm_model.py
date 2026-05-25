import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools import feature_list


class LLMAttnPoolNetwork(nn.Module):
    def __init__(self, env, model_name="google/gemma-3-1b-it", batch_size=64):
        super().__init__()
        self.env = env
        self.batch_size = batch_size
        # Ablation flags (read from args with safe defaults so existing code paths still work)
        self._pool_init = getattr(env.args, "pool_init", "vocab")   # "vocab" | "random"
        self._pool_type = getattr(env.args, "pool_type", "attention")  # "attention" | "mean"

        dataset = self.env.args.dataset
        self.group_dims = [
            len(feature_list.product_info[dataset]),
            len(feature_list.order_info[dataset]),
            len(feature_list.customer_info[dataset]),
            len(feature_list.shipping_info[dataset]),
        ]
        self.feature_dim = sum(self.group_dims)

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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-padding keeps the last real token at position -1.
        self.tokenizer.padding_side = "left"

        # Load in float32 — same reasoning as llm_model.py (float16/bfloat16 CUDA errors
        # on this hardware; device_map avoided for the same reason).
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        self.backbone.to(self.env.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False  # entire backbone frozen; only pool_attn + cls_head train

        hidden = self.backbone.config.hidden_size

        # Shared: LM-head rows for action tokens "0"–"3" used by both heads.
        action_token_ids = [
            self.tokenizer.encode(t, add_special_tokens=False)[0]
            for t in ["0", "1", "2", "3"]
        ]
        with torch.no_grad():
            action_rows = self.backbone.lm_head.weight[action_token_ids]  # [4, H]

        # pool_attn: score tokens by similarity to mean action direction so attention
        # concentrates on answer-like positions rather than uniform mean-pooling.
        # (Zero-init → near-zero pooled vector → bias-only output, identical across
        # all models. Action-direction init gives model-specific representations.)
        self.pool_attn = nn.Linear(hidden, 1, bias=False)
        with torch.no_grad():
            if self._pool_init == "random":
                nn.init.xavier_uniform_(self.pool_attn.weight)
            else:  # "vocab" — default VocabAlign
                self.pool_attn.weight.data.copy_(action_rows.mean(0).unsqueeze(0))
        self.pool_attn.to(self.env.device)

        # cls_head: initialized from per-action LM-head rows — the LLM's implicit
        # zero-shot shipping policy read from next-token probabilities for "0"–"3".
        self.cls_head = nn.Linear(hidden, 4)
        with torch.no_grad():
            if self._pool_init == "random":
                nn.init.xavier_uniform_(self.cls_head.weight)
                nn.init.zeros_(self.cls_head.bias)
            else:  # "vocab" — default VocabAlign
                self.cls_head.weight.data.copy_(action_rows)
                self.cls_head.bias.data.zero_()
        self.cls_head.to(self.env.device)

    def serialize_batch(self, raw_state: torch.Tensor) -> list:
        """Convert raw feature vectors to natural-language prompts.

        Identical to LLMValueNetwork.serialize_batch — feature names in the prompt
        activate the LLM's pre-trained semantic knowledge about shipping concepts.
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
            texts.append(
                ". ".join(parts)
                + ". Optimal shipping action"
                  " (0=Standard Class, 1=Second Class, 2=First Class, 3=Same Day):"
            )
        return texts

    def encode_batch(self, raw_state: torch.Tensor):
        """Run only the frozen backbone. Returns (hidden [B,T,H], attention_mask [B,T]).

        Called once per sample during hidden-state caching. After caching, training
        never calls this again — only forward_from_hidden() runs per epoch.
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

        base = getattr(self.backbone, "model", None) or getattr(self.backbone, "transformer", self.backbone)
        with torch.no_grad():
            out = base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        return out.last_hidden_state.float(), attention_mask  # [B,T,H], [B,T]

    def forward_from_hidden(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply pool_attn + cls_head to pre-computed hidden states. Returns [B, 4] logits.

        This is the only path that runs during every training step after caching —
        the backbone is never touched again.
        """
        if self._pool_type == "mean":
            # Ablation: uniform mean over valid tokens, no learned attention weights
            mask_f = attention_mask.float().unsqueeze(-1)          # [B, T, 1]
            pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        else:  # "attention" — default VocabAlign
            scores = self.pool_attn(hidden).squeeze(-1)                      # [B, T]
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
            weights = torch.softmax(scores, dim=-1)                          # [B, T]
            pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)             # [B, H]
        return self.cls_head(pooled)                                         # [B, 4]

    def forward(self, raw_state: torch.Tensor) -> torch.Tensor:
        """Full pipeline: encode then classify. Used when no hidden-state cache is available.

        Args:
            raw_state: [B, feature_dim] unscaled feature tensor
        Returns:
            [B, 4] float32 logits over the 4 shipping actions
        """
        hidden, attention_mask = self.encode_batch(raw_state)
        return self.forward_from_hidden(hidden, attention_mask)
