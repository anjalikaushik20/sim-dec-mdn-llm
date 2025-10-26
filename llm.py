from transformers import pipeline
import torch
from huggingface_hub import login

# Or, if you want to provide the token directly:
login(token="HF_TOKEN")

pipe = pipeline("text-generation", model="google/gemma-3-1b-it", device="cuda", dtype=torch.bfloat16)

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Write a poem on Mixture Density Networks."},]
        },
    ],
]

output = pipe(messages, max_new_tokens=50, return_full_text=False)
print(output[0][0]['generated_text'])
