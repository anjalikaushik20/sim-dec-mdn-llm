import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tools import feature_list


class SerializedMLPNetwork(nn.Module):
    """TF-IDF on serialized text + MLP classifier.

    Ablation baseline for VocabAlign: exposes the identical encode_batch /
    forward_from_hidden / forward interface so cb_session_llm.py needs no changes.

    Checkpoint compatibility: all trainable weights live under cls_head.*, which
    satisfies the existing "pool_attn" | "cls_head" filter in dm_train(). A dummy
    pool_attn (no grad) is included only to keep the filter from saving an empty dict.
    """

    def __init__(self, env, loader, tfidf_max_features: int = 1000):
        super().__init__()
        self.env = env

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
        self.decoders = {}  # no categorical decoding; numeric fallback used throughout

        # TF-IDF encoder (sklearn, not a nn.Module — not saved in state_dict)
        self.tfidf = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1, 2))
        self._tfidf_max_features = tfidf_max_features

        # Trainable MLP — all layers named cls_head.* for checkpoint compatibility
        self.cls_head = nn.Sequential(
            nn.Linear(tfidf_max_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        self.cls_head.to(env.device)

        # Dummy pool_attn: no gradient, satisfies checkpoint filter, never used in forward
        self.pool_attn = nn.Linear(1, 1, bias=False)
        self.pool_attn.weight.requires_grad_(False)
        self.pool_attn.to(env.device)

        # Fit TF-IDF on training split before any training starts
        self._fit_tfidf(loader)

    def _fit_tfidf(self, loader):
        train_X = loader.train_inputs
        if isinstance(train_X, torch.Tensor):
            raw = train_X[:, :self.feature_dim]
        else:
            raw = torch.tensor(train_X, dtype=torch.float32)[:, :self.feature_dim]
        texts = self.serialize_batch(raw)
        self.tfidf.fit(texts)

    def serialize_batch(self, raw_state) -> list:
        """Same text format as LLMAttnPoolNetwork (numeric-only, no categorical decoders)."""
        if isinstance(raw_state, torch.Tensor):
            raw_np = raw_state.detach().cpu().numpy()
        else:
            raw_np = np.asarray(raw_state, dtype=np.float32)
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

    def encode_batch(self, raw_state: torch.Tensor):
        """TF-IDF encode. Returns (dense [B, max_features], ones_mask [B, 1]).

        The ones_mask is a placeholder so forward_from_hidden has a consistent signature.
        """
        texts = self.serialize_batch(raw_state)
        sparse = self.tfidf.transform(texts)
        dense = torch.tensor(sparse.toarray(), dtype=torch.float32, device=self.env.device)
        mask = torch.ones(dense.shape[0], 1, dtype=torch.float32, device=self.env.device)
        return dense, mask

    def forward_from_hidden(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """hidden is [B, max_features] TF-IDF output; mask is unused. Returns [B, 4]."""
        return self.cls_head(hidden)

    def forward(self, raw_state: torch.Tensor) -> torch.Tensor:
        hidden, mask = self.encode_batch(raw_state)
        return self.forward_from_hidden(hidden, mask)
