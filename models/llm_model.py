
import torch.nn as nn
import torch.nn.functional as F
from tools import feature_list
import torch.nn.init as init
from transformers import pipeline
import torch
from huggingface_hub import login

login(token="HF_TOKEN_PLACEHOLDER")

class LLMValueNetwork(nn.Module):
    def __init__(self, env, model_name="google/gemma-3-1b-it", batch_size=64):
        super().__init__()
        self.env = env
        self.batch_size = batch_size

        # Lighter weights + faster decode
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device=0,
            torch_dtype=torch.float16,
            model_kwargs={"low_cpu_mem_usage": True, "temperature": 0.1},
        )


    def forward(self, state):
        # Build prompts for the entire batch
        feats = state.detach().cpu().numpy().tolist()
        prompts = [self._make_prompt(f) for f in feats]

        # Greedy, short output; no sampling flags to avoid pipeline warnings
        out = self.pipe(
            prompts,
            max_new_tokens=12,             # just need "x, y, z, w"
            return_full_text=False,
            do_sample=False,
            batch_size=self.batch_size    # <<< important
        )
        texts = [o[0]["generated_text"] if isinstance(o, list) else o["generated_text"] for o in out]
        rows = [self._parse_output_to_logits(t) for t in texts]
        return torch.tensor(rows, device=state.device, dtype=torch.float32)

    def _make_prompt(self, features):
        return (
            "You are a precise model returning 4 floating-point scores.\n"
            "Given numeric features, output ONLY a valid JSON array of 4 numbers.\n"
            "Do not include text or words.\n"
            f"Features: {features}\n"
            "Output: [x0, x1, x2, x3]"
        )

    def _parse_output_to_logits(self, text):
        import re, math
        # prefer bracketed arrays first
        m = re.search(r"\[\s*([-+eE0-9\.\s,]+)\s*\]", text)
        s = m.group(1) if m else text
        # extract floats
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", s)
        vals = [float(x) for x in nums[:4]]
        # try “A0: v” style if needed
        if len(vals) < 4:
            labels = re.findall(r"A[0-3]\s*[:=]\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)", text)
            while len(vals) < 4 and labels:
                vals.append(float(labels.pop(0)))
        # pad
        while len(vals) < 4:
            vals.append(0.0)
        # sanitize
        vals = [0.0 if not math.isfinite(v) else v for v in vals]
        # avoid all-zeros vector — give a tiny uniform preference
        if all(v == 0.0 for v in vals):
            vals = torch.randn(4).softmax(0).tolist()
        return vals
