"""Exercise 18 — Compare greedy choice with random sampling.

Sampling first converts logits to probabilities, then randomly chooses a
candidate. Change TEMPERATURE and compare several runs. Keep outputs short.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "distilbert/distilgpt2"
PROMPT = "A robot walked into the garden and"
MAX_NEW_TOKENS = 18
TEMPERATURE = 0.8  # Lower is more concentrated; higher is more varied.

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()
input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]

with torch.inference_mode():
    for _ in range(MAX_NEW_TOKENS):
        logits = model(input_ids=input_ids).logits[:, -1, :]
        probabilities = torch.softmax(logits / TEMPERATURE, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
