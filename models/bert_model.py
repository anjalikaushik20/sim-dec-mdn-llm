import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tools import feature_list


class BertAttnPoolNetwork(nn.Module):
    """Frozen bert-base-uncased + attention pooling head.

    Architecture mirrors LLMAttnPoolNetwork exactly — same pool_attn + cls_head with
    VocabAlign init read from BERT's MLM head, same training protocol (backbone frozen,
    only pool_attn + cls_head train) — but uses BERT (110M params) instead of a larger
    causal LM.

    Exposes the same encode_batch / forward_from_hidden / forward interface so
    cb_session_llm.py needs no changes.
    """

    def __init__(self, env, model_name: str = "google-bert/bert-base-uncased", raw_csv_path=None):
        super().__init__()
        self.env = env
        self._pool_init = getattr(env.args, "pool_init", "vocab")
        self._pool_type = getattr(env.args, "pool_type", "attention")

        dataset = env.args.dataset
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
        self.decoders = {}  # numeric-only serialization for ablation clarity

        # BERT tokenizer — pad_token is already defined for BERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"

        # Load BertForMaskedLM: gives us both the encoder (backbone.bert) and the
        # MLM head weight (backbone.cls.predictions.decoder.weight) for vocab-aligned init.
        self.backbone = AutoModelForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        self.backbone.to(env.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        hidden = self.backbone.config.hidden_size  # 768 for bert-base-uncased

        # VocabAlign init: read BERT MLM-head rows for tokens "0"–"3"
        action_token_ids = [
            self.tokenizer.encode(t, add_special_tokens=False)[0]
            for t in ["0", "1", "2", "3"]
        ]
        lm_weight = self.backbone.cls.predictions.decoder.weight  # [vocab, 768]
        with torch.no_grad():
            action_rows = lm_weight[action_token_ids].float()  # [4, 768]

        self.pool_attn = nn.Linear(hidden, 1, bias=False)
        with torch.no_grad():
            if self._pool_init == "random":
                nn.init.xavier_uniform_(self.pool_attn.weight)
            else:
                self.pool_attn.weight.data.copy_(action_rows.mean(0).unsqueeze(0))
        self.pool_attn.to(env.device)

        self.cls_head = nn.Linear(hidden, 4)
        with torch.no_grad():
            if self._pool_init == "random":
                nn.init.xavier_uniform_(self.cls_head.weight)
                nn.init.zeros_(self.cls_head.bias)
            else:
                self.cls_head.weight.data.copy_(action_rows)
                self.cls_head.bias.data.zero_()
        self.cls_head.to(env.device)

    def serialize_batch(self, raw_state: torch.Tensor) -> list:
        """Same text format as LLMAttnPoolNetwork (numeric-only, no categorical decoders)."""
        import numpy as np
        raw_np = raw_state.detach().cpu().numpy()
        texts = []
        for row in raw_np:
            parts = []
            for label, (s, e) in zip(self._group_labels, self._group_slices):
                kv = ", ".join(
                    f"{n}={v:.3g}"
                    for n, v in zip(self.feature_names[s:e], row[s:e])
                )
                parts.append(f"{label}: {kv}")
            texts.append(
                ". ".join(parts)
                + ". Optimal shipping action"
                  " (0=Standard Class, 1=Second Class, 2=First Class, 3=Same Day):"
            )
        return texts

    def _bert_base(self):
        return (
            getattr(self.backbone, "bert", None)
            or getattr(self.backbone, "model", None)
            or getattr(self.backbone, "transformer", self.backbone)
        )

    def encode_batch(self, raw_state: torch.Tensor):
        """Run frozen BERT encoder. Returns (hidden [B,T,H], attention_mask [B,T])."""
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
        with torch.no_grad():
            out = self._bert_base()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return out.last_hidden_state.float(), attention_mask  # [B,T,H], [B,T]

    def encode_batch_for_cache(self, raw_state: torch.Tensor, max_length: int = 128):
        """Encode with fixed-length padding for val hidden-state cache building.

        Uses padding='max_length' so every batch produces the same T, giving a
        uniform [N, max_length, H] shape that can be written to a numpy memmap.
        """
        texts = self.serialize_batch(raw_state)
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(self.env.device)
        attention_mask = enc["attention_mask"].to(self.env.device)
        with torch.no_grad():
            out = self._bert_base()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return out.last_hidden_state.float(), attention_mask  # [B,T,H], [B,T]

    def forward_from_hidden(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply pool_attn + cls_head to BERT hidden states. Returns [B, 4] logits."""
        if self._pool_type == "mean":
            mask_f = attention_mask.float().unsqueeze(-1)
            pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        else:
            scores = self.pool_attn(hidden).squeeze(-1)
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        return self.cls_head(pooled)

    def forward(self, raw_state: torch.Tensor) -> torch.Tensor:
        hidden, attention_mask = self.encode_batch(raw_state)
        return self.forward_from_hidden(hidden, attention_mask)
