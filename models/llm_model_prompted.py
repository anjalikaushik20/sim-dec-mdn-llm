# models/llm_model_prompted.py
#
# PROMPTED MODE: The LLM makes decisions via natural language prompting.
# No training is performed. The LLM receives order features as structured
# text and outputs a shipping mode (0-3) directly from its pretrained knowledge.
# No adapters, no cls_head, no requires_grad anywhere.

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools import feature_list

_PROMPT_TEMPLATE = (
    "You are a logistics decision-making assistant.\n"
    "Given the following order details, choose the best shipping mode.\n\n"
    "Order details:\n"
    "{feature_lines}\n\n"
    "Available shipping modes:\n"
    "0 - Standard Shipping\n"
    "1 - Express Shipping\n"
    "2 - Same-Day Delivery\n"
    "3 - Economy Shipping\n\n"
    "Think first step-by-step, then provide ONLY the number (0, 1, 2, or 3) of the best shipping mode.\n"
    "Respond with ONLY the number (0, 1, 2, or 3) of the best shipping mode.\n"
    "Do not explain. Do not add any other text."
)


class LLMPromptedDecisionMaker(nn.Module):
    def __init__(self, env, model_name="google/gemma-3-1b-it", batch_size=64):
        super().__init__()
        self.env = env
        self.batch_size = batch_size
        self.model_name = model_name

        dtype = torch.float16 if ("cuda" in str(self.env.device)) else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.env.device)
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad = False

        # No adapters or cls_head — purely prompted inference.
        self.eval()

    def decide(self, raw_feature_rows, feature_names):
        """
        Args:
            raw_feature_rows: list of dicts mapping feature_name -> float value
            feature_names:    list of feature names in order
        Returns:
            list of int actions (0-3), one per row
        """
        actions = []
        with torch.no_grad():
            for row in raw_feature_rows:
                feature_lines = "\n".join(
                    f"- {name}: {row.get(name, 0.0):.4g}"
                    for name in feature_names
                )
                prompt = _PROMPT_TEMPLATE.format(feature_lines=feature_lines)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.env.device)
                outputs = self.llm.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=5,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                # Decode only the newly generated tokens (not the prompt)
                new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                match = re.search(r"[0-3]", generated_text)
                if match:
                    actions.append(int(match.group()))
                else:
                    from tools.logger import info
                    info(f"[PROMPTED] No valid action (0-3) in LLM output: {repr(generated_text)}. Defaulting to 0.")
                    actions.append(0)
        return actions

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Drop-in replacement for LLMValueNetwork.forward().
        Args:
            state_tensor: [B, feature_dim] float tensor (may be scaled)
        Returns:
            one-hot float tensor of shape [B, 4] — exactly one 1.0 per row
        """
        dataset = self.env.args.dataset
        feature_names = (
            feature_list.product_info[dataset]
            + feature_list.order_info[dataset]
            + feature_list.customer_info[dataset]
            + feature_list.shipping_info[dataset]
        )

        state_np = state_tensor.detach().cpu().float().numpy()
        B = state_np.shape[0]

        raw_feature_rows = [
            {name: float(state_np[i, j]) for j, name in enumerate(feature_names)}
            for i in range(B)
        ]

        with torch.no_grad():
            actions = self.decide(raw_feature_rows, feature_names)
            action_tensor = torch.tensor(actions, dtype=torch.long, device=self.env.device)
            return F.one_hot(action_tensor, num_classes=4).float()
