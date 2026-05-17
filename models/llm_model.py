import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-padding keeps the last real token at position -1, which is what
        # we read for hidden-state extraction and what generate() expects.
        self.tokenizer.padding_side = "left"

        # Load in float32 then move to device.
        # float16/bfloat16 cause "CUDA driver error: invalid argument" on this
        # hardware during both the forward pass (RMSNorm) and the .to() conversion.
        # device_map is also avoided: it triggers caching_allocator_warmup() inside
        # transformers which unconditionally allocates a float16 tensor and fails
        # with the same error regardless of the model dtype.
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        self.backbone.to(self.env.device)
        self.backbone.eval()
        # Freeze the entire backbone first; lm_head is selectively unfrozen below.
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Token IDs for the digit strings "0"–"3" used as action labels.
        self._action_token_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0]
            for i in range(4)
        ]

        # Detach lm_head weight from embed_tokens weight tying (Qwen3/Llama share the
        # same Python object).  Without clone(), unfreezing lm_head also unfreezes
        # embed_tokens, bloating the gradient and causing CUBLAS OOM.
        lm = self.backbone.lm_head
        lm.weight = nn.Parameter(lm.weight.detach().clone())
        for p in lm.parameters():
            p.requires_grad = True

    def serialize_batch(self, raw_state: torch.Tensor) -> list:
        """Convert a batch of raw (unscaled) feature vectors to natural-language prompt strings.

        Each feature group becomes a comma-separated key=value clause so the LLM
        can use its pretrained knowledge of feature names like 'Order Quantity' or
        'Discount %' to reason about the best shipping action.  The prompt ends with
        an explicit action enumeration so the very next generated token is the answer.
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

    def forward(self, raw_state: torch.Tensor) -> torch.Tensor:
        """
        Frozen transformer body → last-position hidden state → lm_head → [B, 4] logits.

        Only the lm_head is differentiable; the transformer body runs inside
        torch.no_grad().  The 4 logits correspond to the token IDs for "0"–"3".

        Args:
            raw_state: [B, feature_dim] unscaled feature tensor
        Returns:
            [B, 4] float32 logits over the 4 shipping actions
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

        # Run only the transformer body (no lm_head) to get hidden states.
        # backbone.model is the base transformer for all CausalLM architectures
        # (Qwen3ForCausalLM.model, LlamaForCausalLM.model, GemmaForCausalLM.model …).
        base = getattr(self.backbone, "model", self.backbone)
        with torch.no_grad():
            out = base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

        # With left-padding the last column is always a real token.
        last_hidden = out.last_hidden_state[:, -1, :].float()   # [B, hidden]
        full_logits = self.backbone.lm_head(last_hidden)         # [B, vocab_size]
        return full_logits[:, self._action_token_ids]            # [B, 4]

    @torch.no_grad()
    def generate_action(self, raw_state: torch.Tensor) -> torch.Tensor:
        """
        True LLM inference: autoregressively generate the action token and parse it.

        Uses the full backbone (transformer + frozen lm_head) via generate(), so the
        complete sampling / decoding pipeline of the LLM is active.  No gradients
        are computed here — inference only.

        Args:
            raw_state: [B, feature_dim] unscaled feature tensor
        Returns:
            [B] long tensor of action indices in {0, 1, 2, 3}
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

        generated = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = generated[:, input_ids.shape[1]:]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        actions = []
        for text in decoded:
            found = None
            for ch in text.strip():
                if ch in "0123":
                    found = int(ch)
                    break
            actions.append(found if found is not None else 0)

        return torch.tensor(actions, dtype=torch.long, device=self.env.device)
