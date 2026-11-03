"""Exercise 12 — See the parts of a pretrained model repository.

This downloads only the model files this course uses, not every alternate
TensorFlow, Flax, Core ML or LiteRT export in the upstream repository.
The checkpoint itself is about 353 MB.
"""

from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "distilbert/distilgpt2"
COURSE_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
]

model_dir = Path(snapshot_download(repo_id=MODEL_ID, allow_patterns=COURSE_FILES))
print("Model snapshot:", model_dir)
print("Files downloaded for this course:")
for path in sorted(model_dir.iterdir()):
    if path.is_file():
        print(f"  {path.name:28} {path.stat().st_size / (1024 * 1024):7.2f} MiB")
