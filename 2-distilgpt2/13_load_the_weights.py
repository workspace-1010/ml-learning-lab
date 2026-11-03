"""Exercise 13 — Build the model object and load pretrained weights.

No pipeline is used. Transformers reads the config, constructs the correct
architecture, then loads the learned tensors from the checkpoint.
"""

from transformers import AutoModelForCausalLM

MODEL_ID = "distilbert/distilgpt2"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()

print("Model class:", type(model).__name__)
print("Total parameters:", sum(parameter.numel() for parameter in model.parameters()))
print("First parameter dtype:", next(model.parameters()).dtype)
print("Training mode:", model.training)
print("Model is ready for inference.")
