
import torch.nn as nn
import torch.nn.functional as F
from tools import feature_list
import torch.nn.init as init
from transformers import pipeline
import torch
from huggingface_hub import login

login(token="HF_TOKEN_PLACEHOLDER")

class LLMValueNetwork(nn.Module):
    def __init__(self, env, model_name="google/gemma-3-1b-it"):
        super(LLMValueNetwork, self).__init__()
        self.env = env
        self.device = "cuda"
        self.pipe = pipeline(
            "text-generation", 
            model=model_name, 
            device=0 if self.device == "cuda" else -1, 
            dtype=torch.bfloat16 if self.device == "cuda" else None,
        )

    def forward(self, state):
        outputs = []
        for sample in state.cpu().numpy().tolist():
            prompt = self._make_prompt(sample)
            text = self.pipe(prompt, max_new_tokens=30, return_full_text=False)[0]["generated_text"]
            # map LLM text output to numeric logits
            logits = self._parse_output_to_logits(text)
            outputs.append(logits)
        return torch.tensor(outputs, device=self.env.device, dtype=torch.float32)
 
    def _make_prompt(self, features):
            return (
                "You are selecting a shipping mode for an order.\n"
                "Given the following numerical features from a simulated environment:\n"
                f"{features}\n"
                "Predict the preference scores for 4 possible actions (A0, A1, A2, A3). "
                "Return 4 numbers separated by commas."
            )

    def _parse_output_to_logits(self, text):
        import re
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if len(nums) < 4:
            nums = nums + ["0"] * (4 - len(nums))
        return [float(x) for x in nums[:4]]
 
