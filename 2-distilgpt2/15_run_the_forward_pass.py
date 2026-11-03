"""Exercise 15 — Run one forward pass and inspect the raw output.

The last dimension of logits has one score for each token in the vocabulary.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()

text = "The computer learned to"
inputs = tokenizer(text, return_tensors="pt")

with torch.inference_mode():
    outputs = model(**inputs)

print("Input token IDs shape:", tuple(inputs["input_ids"].shape))
print("Logits shape (batch, positions, vocabulary):", tuple(outputs.logits.shape))
print("One score vector per input position; vocabulary size:", outputs.logits.shape[-1])
