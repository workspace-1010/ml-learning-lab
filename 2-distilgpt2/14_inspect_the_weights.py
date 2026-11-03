"""Exercise 14 — Inspect named parameters and their tensor shapes.

Change the number of displayed parameters. Shapes hint at the model's design.
"""

from transformers import AutoModelForCausalLM

MODEL_ID = "distilbert/distilgpt2"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()

total = 0
for name, parameter in model.named_parameters():
    total += parameter.numel()

print("Parameter tensors:")
for index, (name, parameter) in enumerate(model.named_parameters()):
    if index == 12:
        break
    print(f"{name:48} shape={str(tuple(parameter.shape)):20} dtype={parameter.dtype}")

print("Total scalar parameters:", f"{total:,}")
