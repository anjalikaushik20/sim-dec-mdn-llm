
import torch.nn as nn
import torch.nn.functional as F
from tools import feature_list
import torch.nn.init as init
from transformers import pipeline
import torch
from huggingface_hub import login

login(token="HF_TOKEN")

class LLMValueNetwork(nn.Module):
    def __init__(self, env, model_name="google/gemma-3-1b-it", batch_size=32):
        super().__init__()
        self.env = env
        self.device = "cuda"
        self.batch_size = batch_size
        self.pipe = pipeline(
            "text-generation", 
            model=model_name, 
            device=0 if self.device == "cuda" else -1, 
            dtype=torch.bfloat16 if self.device == "cuda" else None,
        )

    def forward(self, state):
        outputs = []
        with torch.inference_mode():   
            for sample in state.detach().cpu().numpy().tolist():
                prompt = self._make_prompt(sample)
                text = self.pipe(
                    prompt,
                    max_new_tokens=30,
                    return_full_text=False,
                    do_sample=False
                )[0]["generated_text"]
                logits = self._parse_output_to_logits(text)
                outputs.append(logits)
        return torch.tensor(outputs, device=state.device, dtype=torch.float32)
 
    def _make_prompt(self, features):
            return (
                "You are selecting a shipping mode for an order.\n"
                "Given the following numerical features from a simulated environment:\n"
                f"{features}\n"
                "Predict the preference scores for 4 possible actions (A0, A1, A2, A3). "
                "Return 4 numbers separated by commas."
            )

    def _parse_output_to_logits(self, text):
        import re, math
        # minimal: capture signed ints/floats robustly
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)
        vals = [float(x) for x in nums[:4]]
        if len(vals) < 4:
            vals += [0.0] * (4 - len(vals))
        # replace NaN/inf if any
        vals = [0.0 if not math.isfinite(v) else v for v in vals]
        return vals
 
