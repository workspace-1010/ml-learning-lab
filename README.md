# ML Learning Lab

An exercise-first path to **basic machine learning and understanding a pretrained language model from the inside**.

The format is inspired by the structure of beginner-friendly programming courses such as Codédex: numbered topic folders, one small runnable program per exercise, and a single index that helps you follow the sequence. The code and explanations here are original; this is not a copy of another course.

## What you will learn

**Part 1 — First ML programs:** prepare examples, separate features and targets, split data, create and fit a model, make predictions, measure error, classify examples, and notice overfitting.

**Part 2 — Assemble and inspect DistilGPT2:** tokenize text, inspect the model configuration and files, load the actual checkpoint, examine learned parameters, run the raw forward pass, inspect next-token scores, and write a small autoregressive generation loop.

The target is to understand useful APIs and how the pieces fit together—not to memorize every function or train an LLM from scratch.

## Start here: setup in VS Code

For the complete beginner walkthrough, including every required library installation and the model-file download steps, read **[the setup guide](docs/setup-guide.md)** first. This project assumes you already have Python and VS Code installed and have opened the repository folder in VS Code.

Open **Terminal → New Terminal** in VS Code. Create and activate a virtual environment:

**Windows PowerShell**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the libraries for Part 1 (NumPy, scikit-learn, and pytest):

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

When you reach Part 2, install the additional libraries (PyTorch and Transformers):

```bash
python -m pip install -e ".[pretrained]"
```

**Model download:** the first relevant Part 2 exercise downloads DistilGPT2's tokenizer/configuration files and the approximately 353 MB `model.safetensors` checkpoint from [Hugging Face](https://huggingface.co/distilbert/distilgpt2). Normally you do not need to download them manually in a browser. Exercise 12 explicitly downloads and lists the selected files. An internet connection is needed on the first run; files are reused from the local cache afterward. Details and troubleshooting are in the [setup guide](docs/setup-guide.md).

## Part 1 — First ML programs

| Exercise | Program | Idea |
|---:|---|---|
| 01 | [The dataset](1-first-ml/01_the_dataset.py) | Inspect examples before fitting |
| 02 | [Features and targets](1-first-ml/02_features_and_targets.py) | Separate input data from the value to predict |
| 03 | [Split the data](1-first-ml/03_split_the_data.py) | Hold back examples for testing |
| 04 | [Create a model](1-first-ml/04_create_a_model.py) | Model object vs. trained model |
| 05 | [Fit the model](1-first-ml/05_fit_the_model.py) | Learn parameters from data |
| 06 | [Make predictions](1-first-ml/06_make_predictions.py) | Predict values for new inputs |
| 07 | [Measure error](1-first-ml/07_measure_error.py) | Evaluate held-out predictions |
| 08 | [Classify flowers](1-first-ml/08_classify_flowers.py) | Predict categories |
| 09 | [Spot overfitting](1-first-ml/09_spot_overfitting.py) | Compare training and test performance |

## Part 2 — DistilGPT2, without a pipeline

We use one specific open model throughout: **[`distilbert/distilgpt2`](https://huggingface.co/distilbert/distilgpt2)**. Its model card lists 82 million parameters and the Apache 2.0 license. It is a small English text-completion model, not a chat assistant. Read [Part 2's guide](2-distilgpt2/README.md) before running these exercises.

| Exercise | Program | Idea |
|---:|---|---|
| 10 | [Tokenize text](2-distilgpt2/10_tokenize_text.py) | Text becomes token IDs |
| 11 | [Read the configuration](2-distilgpt2/11_read_the_configuration.py) | Inspect the model blueprint |
| 12 | [List model files](2-distilgpt2/12_list_the_model_files.py) | Distinguish weights, config and tokenizer files |
| 13 | [Load the weights](2-distilgpt2/13_load_the_weights.py) | Construct the architecture and load its checkpoint |
| 14 | [Inspect the weights](2-distilgpt2/14_inspect_the_weights.py) | Names, shapes, dtypes and parameter counts |
| 15 | [Run the forward pass](2-distilgpt2/15_run_the_forward_pass.py) | Inspect raw logits and their dimensions |
| 16 | [Choose the next token](2-distilgpt2/16_choose_the_next_token.py) | Read the top candidate scores |
| 17 | [Generate token by token](2-distilgpt2/17_generate_one_token_at_a_time.py) | Implement greedy autoregressive decoding |
| 18 | [Explore sampling](2-distilgpt2/18_explore_sampling.py) | Turn scores into a sampled next token |

Every model exercise uses direct Transformers APIs such as `AutoTokenizer` and `AutoModelForCausalLM`. **No `pipeline()` is used**, because loading, tokenization, logits and generation are what we want to learn.

## Designed for a modest computer

Start with CPU inference, short prompts, and small generation lengths. The checkpoint is about 353 MB, but the full program uses more RAM because Python, PyTorch and intermediate tensors also require memory. Four GB is a constrained target, so close memory-heavy programs. Do not download every alternate model export, and never commit weights or model caches to this repository.

## Course design and study method

- [Course design](docs/course-design.md)
- [How to study](docs/how-to-study.md)
- [Setup guide](docs/setup-guide.md)
- [Glossary](docs/glossary.md)
- [Contribution guidelines](CONTRIBUTING.md)

No RAG or vector-database section is included in this version. The goal is to understand the model itself before adding another system around it.
