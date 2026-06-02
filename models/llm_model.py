import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools import feature_list


class LLMAttnPoolNetwork(nn.Module):
    def __init__(self, env, model_name="google/gemma-3-1b-it", batch_size=64, raw_csv_path=None):
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
        self._group_labels = feature_list.group_labels[dataset]
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

        # Shared: LM-head rows for the first token of each action name.
        # Using action name tokens ("Standard", "Second", "First", "Same") instead of
        # digit tokens ("0"–"3") gives the cls_head semantically meaningful initialization.
        self._action_names = feature_list.action_names[dataset]
        action_token_ids = [
            self.tokenizer.encode(name, add_special_tokens=False)[0]
            for name in self._action_names
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

        # Build categorical decoders if a raw CSV is provided
        self.decoders = {}
        if raw_csv_path is not None and os.path.exists(raw_csv_path):
            self.decoders = self._build_decoders(raw_csv_path)
            processed_path = os.path.join(
                os.path.dirname(raw_csv_path),
                f"processed_{env.args.dataset}_train.csv",
            )
            self._verify_decoders(processed_path, self.decoders)
        elif raw_csv_path is not None:
            print(f"[DECODER] WARNING: raw_csv_path '{raw_csv_path}' not found — "
                  "falling back to numeric serialization.")

    def _build_decoders(self, raw_csv_path: str) -> dict:
        """Build int→string lookup tables for categorical columns from the raw CSV.

        Reproduces the LabelEncoder encoding used by S_Loader.categorical_features_process
        so that processed integer values can be decoded back to human-readable strings.

        General rule: any column that (a) appears in the serialized feature groups and
        (b) has string (object) dtype in the raw CSV gets a LabelEncoder decoder.

        DataCo special cases: Category Id and Product Card Id are numeric in the raw CSV
        but map to human-readable Category Name / Product Name via a join.
        """
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder

        df = pd.read_csv(raw_csv_path, encoding="latin1")
        decoders = {}
        serialized = set(self.feature_names)

        # General: string-typed columns that appear in the serialized feature groups.
        # Use is_string_dtype to handle both legacy object dtype and pandas StringDtype.
        for col in serialized:
            if col in df.columns and pd.api.types.is_string_dtype(df[col]):
                le = LabelEncoder()
                le.fit(df[col].dropna().astype(str).unique())
                decoders[col] = {i: str(cls) for i, cls in enumerate(le.classes_)}

        # DataCo special case: Category Id (numeric) → Category Name
        if ("Category Id" in serialized
                and "Category Id" in df.columns
                and "Category Name" in df.columns
                and not pd.api.types.is_string_dtype(df["Category Id"])):
            le = LabelEncoder()
            le.fit(df["Category Id"].dropna().astype(str).unique())
            cat_map = df[["Category Id", "Category Name"]].drop_duplicates().copy()
            cat_map["key"] = cat_map["Category Id"].astype(str)
            str_to_name = dict(zip(cat_map["key"], cat_map["Category Name"]))
            decoders["Category Id"] = {
                i: str(str_to_name.get(cls, cls)) for i, cls in enumerate(le.classes_)
            }

        # DataCo special case: Product Card Id (numeric) → Product Name
        if ("Product Card Id" in serialized
                and "Product Card Id" in df.columns
                and "Product Name" in df.columns
                and not pd.api.types.is_string_dtype(df["Product Card Id"])):
            le = LabelEncoder()
            le.fit(df["Product Card Id"].dropna().astype(str).unique())
            prod_map = df[["Product Card Id", "Product Name"]].drop_duplicates().copy()
            prod_map["key"] = prod_map["Product Card Id"].astype(str)
            str_to_name = dict(zip(prod_map["key"], prod_map["Product Name"]))
            decoders["Product Card Id"] = {
                i: str(str_to_name.get(cls, cls)) for i, cls in enumerate(le.classes_)
            }

        for col, dec in decoders.items():
            print(f"[DECODER] {col} ({len(dec)} entries): {dec}")

        return decoders

    def _verify_decoders(self, processed_csv_path: str, decoders: dict):
        """Check that every integer in the processed file has a decoder entry."""
        import pandas as pd

        if not os.path.exists(processed_csv_path):
            print(f"[DECODER] WARNING: processed file '{processed_csv_path}' not found — "
                  "skipping verification.")
            return

        df = pd.read_csv(processed_csv_path)
        all_ok = True
        for col, dec in decoders.items():
            if col not in df.columns:
                continue
            for v in df[col].dropna().unique():
                k = int(round(float(v)))
                if k not in dec:
                    print(f"[DECODER WARNING] {col}: integer {k} has no decoder entry")
                    all_ok = False
        if all_ok:
            print("[DECODER] Verification passed — all processed integers have decoder entries.")

    def serialize_batch(self, raw_state: torch.Tensor) -> list:
        """Convert raw feature vectors to instruction-style natural-language prompts.

        Categorical columns listed in self.decoders are decoded to human-readable
        strings; all other columns fall back to numeric formatting.
        Action options are expressed as natural language labels, not numeric codes.
        """
        raw_np = raw_state.detach().cpu().numpy()
        action_opts = ", ".join(self._action_names)
        texts = []
        for row in raw_np:
            parts = []
            for label, (s, e) in zip(self._group_labels, self._group_slices):
                names = self.feature_names[s:e]
                vals = row[s:e]
                kv_parts = []
                for n, v in zip(names, vals):
                    if n in self.decoders:
                        decoded = self.decoders[n].get(int(round(v)), f"{v:.3g}")
                        kv_parts.append(f"{n}={decoded}")
                    else:
                        kv_parts.append(f"{n}={v:.3g}")
                parts.append(f"{label}: {', '.join(kv_parts)}")
            context = "\n".join(parts)
            texts.append(
                "You are a decision-making assistant. "
                "Based on the context below, select the optimal action.\n\n"
                f"Context:\n{context}\n\n"
                f"Available actions: {action_opts}\n"
                "Optimal action:"
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
