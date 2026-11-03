# ML Learning Lab

A practical, example-first course for learning three things: write basic ML code with libraries, load and inspect a pretrained open-source language model, and build a basic retrieval-augmented generation (RAG) pipeline.

## Learning outcomes

1. Prepare data, train a basic scikit-learn model with `fit()`, predict with `predict()`, and evaluate on held-out data.
2. Explain model configuration, tokenizer, architecture and learned weights; load a pretrained model with existing libraries and run inference.
3. Build a basic RAG pipeline: documents → chunks → embeddings → retrieval → context + question → language model → answer.
4. Explain the purpose and inputs/outputs of key functions, then modify working examples with a clear hypothesis.

This is not a course on training a large language model from scratch or implementing transformers from scratch. Learn the mental model and useful APIs, then connect the parts into working programs.

## Learning path

| Stage | Build | Main tools |
|---|---|---|
| 0. Foundations | Reproducible Python environment and Git workflow | Python, venv, pip, Git |
| 1. Basic ML | Train, predict, and evaluate a small model | scikit-learn |
| 2. Model components | Inspect arrays, tensors, parameters and configuration | NumPy; optional PyTorch |
| 3. Pretrained models | Load a small model, inspect tokenizer/config and run inference | Hugging Face Transformers |
| 4. RAG | Retrieve passages and provide them as context to an LLM | embeddings, vector search, an LLM |
| 5. Capstone | A small document Q&A application with source passages | Earlier components |

## Lesson design

Each lesson should have a clear mental model, a complete runnable example, a trace of important inputs and outputs, experiments, understanding questions, and troubleshooting notes. Prefer small examples and explicit data flow over unnecessary abstractions.

## Repository map

- `docs/` — setup, learning map, glossary and Git workflow.
- `lessons/00-foundations/` — environment and reproducibility.
- `lessons/01-basic-ml/` — first model and evaluation.
- `lessons/02-model-components/` — weights, parameters and tensors.
- `lessons/03-pretrained-models/` — configuration, tokenizer, loading and inference.
- `lessons/04-rag/` — document loading, chunking, embeddings, retrieval and generation.
- `lessons/05-capstone/` — final project specification.
- `tests/` — small automated checks where useful.

## Quick start

Use Python 3.10 or newer. Create and activate a virtual environment before installing dependencies.

**Windows PowerShell**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python lessons/01-basic-ml/01-first-model/example.py
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python lessons/01-basic-ml/01-first-model/example.py
```

Pretrained-model and RAG lessons may require extra dependencies and download model files on first use. Their lesson pages should explain disk, memory, internet and license requirements first. Keep examples CPU-friendly where possible.

## Recommended study loop

1. Run the reference example unchanged.
2. Explain what each important function does and returns.
3. Predict what a small change will do.
4. Create an experiment branch, change one thing, run it, and record the result.
5. Answer the understanding questions before moving on.

You do not need to memorize every API. Learn how to find the right function, inspect inputs and outputs, and understand how components connect.

## Git workflow

Fork the repository, clone your fork, and make a branch for an experiment:

```bash
git clone https://github.com/YOUR-USERNAME/ml-learning-lab.git
cd ml-learning-lab
git switch -c experiment/first-model
```

After experimenting:

```bash
git status
git diff
git add .
git commit -m "Experiment with first ML model"
git push -u origin experiment/first-model
```

Do not commit virtual environments, secrets, large model checkpoints, or datasets that should be downloaded separately.

## Scope and responsible use

- Keep examples reproducible, inspectable, and appropriately small.
- Check model and dataset licenses before use or redistribution.
- Treat retrieved documents as untrusted input; retrieval does not guarantee correctness.
- A local RAG demo is a learning project, not a production security design.

## Contributing and license

See `CONTRIBUTING.md` for lesson format and contribution principles. Original code and learning materials are intended to use the MIT License unless otherwise stated. External models, datasets and references retain their own licenses.
