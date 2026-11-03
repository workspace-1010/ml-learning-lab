"""Exercise 24 — Print each generated token as soon as it is available.

This is a tiny illustration of streaming: users see the response arrive
incrementally rather than waiting for all generated tokens. It is not a
network server; output is streamed to this terminal.
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "distilbert/distilgpt2"
PROMPT = "A tiny computer can"
MAX_NEW_TOKENS = 12

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()
input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]

print(PROMPT, end="", flush=True)

with torch.inference_mode():
    for _ in range(MAX_NEW_TOKENS):
        logits = model(input_ids=input_ids).logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        piece = tokenizer.decode(next_token[0], skip_special_tokens=True)
        print(piece, end="", flush=True)
        sys.stdout.flush()

        if next_token.item() == tokenizer.eos_token_id:
            break

print()
