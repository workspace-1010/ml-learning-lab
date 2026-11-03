"""Exercise 16 — Turn next-token scores into a token choice.

For the last input position, inspect the highest scoring candidate tokens.
This shows a simple decision rule, not a guarantee of factual correctness.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()

prompt = "The computer learned to"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.inference_mode():
    logits = model(**inputs).logits[0, -1, :]
    top_scores, top_ids = torch.topk(logits, k=8)

print("Prompt:", prompt)
for score, token_id in zip(top_scores.tolist(), top_ids.tolist()):
    piece = tokenizer.decode([token_id])
    print(f"{piece!r:18} score={score:.3f}")

chosen_id = int(torch.argmax(logits))
print("Greedy next token:", repr(tokenizer.decode([chosen_id])))
