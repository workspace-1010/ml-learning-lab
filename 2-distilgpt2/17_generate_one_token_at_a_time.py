"""Exercise 17 — Generate text by repeatedly predicting one next token.

This manually implements greedy decoding. Compare its result with changing
the prompt or max_new_tokens. Generation is deliberately short for 4 GB RAM.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "distilbert/distilgpt2"
PROMPT = "Once upon a time, a small computer"
MAX_NEW_TOKENS = 16

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()
input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]

with torch.inference_mode():
    for _ in range(MAX_NEW_TOKENS):
        outputs = model(input_ids=input_ids)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
