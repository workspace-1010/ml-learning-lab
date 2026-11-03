# Setup guide

## Basic ML exercises

Requires Python 3.10 or newer and Git. Use a virtual environment to keep dependencies isolated.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python 1-first-ml/01_the_dataset.py
```

If activation is blocked by policy, invoke the environment's Python directly: `.venv\Scripts\python.exe`.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python 1-first-ml/01_the_dataset.py
```

## DistilGPT2 exercises

Install the additional dependencies only when you reach Part 2:

```bash
python -m pip install -e ".[pretrained]"
python 2-distilgpt2/10_tokenize_text.py
```

The first model-loading step downloads weights and tokenizer files. The Safetensors checkpoint is approximately 353 MB, and total memory use is higher than the checkpoint size. Use CPU inference and short prompts on 4 GB RAM. Do not download all alternate upstream exports or commit model files to Git.
