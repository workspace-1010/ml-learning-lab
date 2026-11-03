"""Exercise 22 — Give generation explicit stopping and context limits.

An engine should not generate forever or feed a sequence beyond the model's
position limit. The short generation cap is especially important on small PCs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "distilbert/distilgpt2"
PROMPT = "Explain a computer in simple words:"
MAX_NEW_TOKENS = 12

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()

input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
context_limit = model.config.n_positions
available_positions = context_limit - input_ids.shape[1]
generation_limit = min(MAX_NEW_TOKENS, max(0, available_positions))

if generation_limit == 0:
    raise ValueError("Prompt already fills the model's context window.")

print("Model context limit:", context_limit, "tokens")
print("Prompt token count:", input_ids.shape[1])
print("New-token budget:", generation_limit)

with torch.inference_mode():
    for _ in range(generation_limit):
        logits = model(input_ids=input_ids).logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            print("Stopped because the model emitted its end-of-sequence token.")
            break
    else:
        print("Stopped because the token budget was used.")

print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
