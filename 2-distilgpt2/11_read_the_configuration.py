"""Exercise 11 — Read the model's architectural blueprint.

Configuration describes architecture settings; it is not the learned weights.
"""

from transformers import AutoConfig

MODEL_ID = "distilbert/distilgpt2"
config = AutoConfig.from_pretrained(MODEL_ID)

print("Model type:", config.model_type)
print("Architecture class:", config.architectures)
print("Transformer layers:", config.n_layer)
print("Attention heads per layer:", config.n_head)
print("Hidden width:", config.n_embd)
print("Vocabulary size:", config.vocab_size)
print("Maximum positions:", config.n_positions)
