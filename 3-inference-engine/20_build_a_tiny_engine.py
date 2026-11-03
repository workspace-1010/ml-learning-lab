"""Exercise 20 — Put the repeated steps behind a small engine interface.

The engine owns model loading and generation. Applications only need to
call engine.generate(prompt). This uses direct model APIs, not pipeline().
The first run downloads the model files if they are not already cached.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "distilbert/distilgpt2"


class TinyInferenceEngine:
    def __init__(self, model_id: str = MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()  # Turn off training-specific behavior.
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def generate(self, prompt: str, max_new_tokens: int = 12) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        if not 1 <= max_new_tokens <= 32:
            raise ValueError("Use 1–32 new tokens in this lesson.")

        input_ids = self.tokenizer(
            prompt, return_tensors="pt"
        )["input_ids"].to(self.device)

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                outputs = self.model(input_ids=input_ids)
                next_token = outputs.logits[:, -1, :].argmax(
                    dim=-1, keepdim=True
                )
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    engine = TinyInferenceEngine()
    answer = engine.generate("A tiny computer can", max_new_tokens=12)
    print(answer)
