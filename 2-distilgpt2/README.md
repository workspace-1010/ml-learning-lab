# Part 2 — Meet DistilGPT2

The model for every exercise in this part is **`distilbert/distilgpt2`**. It is an 82-million-parameter, English-language GPT-2-style text-completion model under the Apache 2.0 license. The standard Safetensors checkpoint is about 353 MB. See the [model card](https://huggingface.co/distilbert/distilgpt2) before using it.

This is a base completion model, not a chat/instruction model. Strange continuations are expected and useful to observe. Keep prompts and generations short on a 4 GB RAM machine.

## Before you start: install and download

Do these steps in VS Code from the **repository root** (the folder containing `pyproject.toml`). The full beginner guide is [Setup guide](../docs/setup-guide.md).

1. Open **Terminal → New Terminal** in VS Code and activate the project's `.venv` environment. If you have not created it yet, follow the setup guide.
2. Install the extra Python libraries for this part. These are not part of the basic Part 1 install:

   ```bash
   python -m pip install -e ".[pretrained]"
   ```

   This installs PyTorch and Hugging Face Transformers into your current Python environment.
3. Make sure you have an internet connection for the first model run. The course downloads model files from [the official Hugging Face model page](https://huggingface.co/distilbert/distilgpt2) automatically.
4. To explicitly download and list the selected model files, run:

   ```bash
   python 2-distilgpt2/12_list_the_model_files.py
   ```

   This downloads the configuration (`config.json`, `generation_config.json`), tokenizer assets (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, and `special_tokens_map.json` when present), and the learned checkpoint `model.safetensors` (about **353 MB**). The script prints the cache location.

**You normally do not have to download the files manually in a browser.** The Transformers/Hugging Face code fetches what each exercise needs on its first run and reuses the local cache later. The files live in that cache, not in this repository. Do not download alternate model formats unnecessarily or commit model files to GitHub.

## Sequence

| File | Try to understand |
|---|---|
| [10 — Tokenize text](10_tokenize_text.py) | Text → token pieces → token IDs |
| [11 — Read the configuration](11_read_the_configuration.py) | Layer count, attention heads, hidden width, vocabulary |
| [12 — List the model files](12_list_the_model_files.py) | Configuration, tokenizer files and learned checkpoint |
| [13 — Load the weights](13_load_the_weights.py) | Architecture object plus pretrained weights |
| [14 — Inspect the weights](14_inspect_the_weights.py) | Parameter names, tensor shapes and data types |
| [15 — Run the forward pass](15_run_the_forward_pass.py) | Input tensor and logits dimensions |
| [16 — Choose the next token](16_choose_the_next_token.py) | Highest scoring token candidates |
| [17 — Generate one token at a time](17_generate_one_token_at_a_time.py) | Autoregressive generation loop |
| [18 — Explore sampling](18_explore_sampling.py) | Probabilities, temperature and random selection |

## Model assembly: the mental picture

- **Tokenizer:** converts text into IDs the network can process.
- **Config:** describes the model blueprint (such as layer count and hidden width).
- **Architecture class:** implements computations for that blueprint.
- **Checkpoint:** supplies the learned parameter values.
- **Forward pass:** calculates output logits for each input position.
- **Decoder loop:** chooses a next token and feeds it back as part of the next input.

The exercises call the direct tokenizer and model APIs. They deliberately do not use `pipeline()`, which would hide the steps we want to inspect.

## Running on a 4 GB RAM computer

Use CPU inference; a GPU is not required. The checkpoint itself is about 353 MB, but total RAM use is much higher because Python, PyTorch, Transformers and intermediate tensors also take memory. Close memory-heavy apps, keep prompts short, and generate only a few tokens while learning. Four GB is a constrained target, so performance and success can vary by system.
