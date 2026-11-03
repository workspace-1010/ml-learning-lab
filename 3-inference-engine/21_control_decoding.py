"""Exercise 21 — Make token selection a configurable policy.

Temperature changes how peaked the scores are. top_k restricts the random
choice to the k highest-scoring tokens. These are small versions of decoding
controls exposed by real text-generation systems.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "distilbert/distilgpt2"
PROMPT = "A tiny computer can"
MAX_NEW_TOKENS = 12
TEMPERATURE = 0.8
TOP_K = 20

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()
input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]

with torch.inference_mode():
    for _ in range(MAX_NEW_TOKENS):
        logits = model(input_ids=input_ids).logits[:, -1, :]
        logits = logits / TEMPERATURE

        # Keep only the top K candidate tokens.
        top_values, top_indices = torch.topk(logits, k=TOP_K, dim=-1)
        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(1, top_indices, top_values)

        probabilities = torch.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
