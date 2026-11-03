# Part 2 — Meet DistilGPT2

The model for every exercise in this part is **`distilbert/distilgpt2`**. It is an 82-million-parameter, English-language GPT-2-style text-completion model under the Apache 2.0 license. The standard Safetensors checkpoint is about 353 MB. See the [model card](https://huggingface.co/distilbert/distilgpt2) before using it.

This is a base completion model, not a chat/instruction model. Strange continuations are expected and useful to observe. Keep prompts and generations short on a 4 GB RAM machine.

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

## Setup

From the repository root, install the optional dependencies:

```bash
python -m pip install -e ".[pretrained]"
```

Run one file at a time, e.g.:

```bash
python 2-distilgpt2/10_tokenize_text.py
```

The first run downloads model/tokenizer files. CPU is intentional; no GPU or pipeline is required. The model checkpoint is around 353 MB, but total memory use also includes PyTorch, Transformers and intermediate tensors. Close memory-heavy applications and keep the sequence short.

## Model assembly: the mental picture

- **Tokenizer:** converts text into IDs the network can process.
- **Config:** describes the model blueprint (such as layer count and hidden width).
- **Architecture class:** implements computations for that blueprint.
- **Checkpoint:** supplies the learned parameter values.
- **Forward pass:** calculates output logits for each input position.
- **Decoder loop:** chooses a next token and feeds it back as part of the next input.

The exercises call the direct tokenizer and model APIs. They deliberately do not use `pipeline()`, which would hide the steps we want to inspect.
