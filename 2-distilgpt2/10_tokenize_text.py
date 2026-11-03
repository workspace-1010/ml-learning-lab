"""Exercise 10 — A language model reads token IDs, not raw text.

Try different punctuation, spaces and words. Observe how the token IDs change.
First run downloads the tokenizer's small vocabulary files.
"""

from transformers import AutoTokenizer

MODEL_ID = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

text = "A model learns to predict the next token."
encoded = tokenizer(text, return_tensors="pt")

print("Original text:", text)
print("Token pieces:", tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))
print("Token IDs:", encoded["input_ids"][0].tolist())
print("Decoded again:", tokenizer.decode(encoded["input_ids"][0]))
