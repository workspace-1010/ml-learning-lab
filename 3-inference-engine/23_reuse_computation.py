"""Exercise 23 — Use the KV cache to avoid recomputing old token states.

Without a cache, each step sends the entire growing sequence through the model.
With use_cache=True, later steps can send only the newest token and reuse
past_key_values. This is a core inference optimization; this demo stays short.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "distilbert/distilgpt2"
PROMPT = "A tiny computer can"
MAX_NEW_TOKENS = 8

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()
input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]

with torch.inference_mode():
    outputs = model(input_ids=input_ids, use_cache=True)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    past_key_values = outputs.past_key_values

    for _ in range(MAX_NEW_TOKENS - 1):
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Only send the new token; the model reuses cached prior key/value states.
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)

all_new_tokens = torch.cat(generated, dim=1)
print("Prompt:", PROMPT)
print("Generated continuation:", tokenizer.decode(all_new_tokens[0], skip_special_tokens=True))
