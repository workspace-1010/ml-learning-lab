# Setup guide: Python + VS Code

This guide assumes you already have Python, VS Code, and this repository on your computer. You do **not** need to download another editor, a separate AI app, or a GPU. You will install Python libraries from the VS Code terminal, and Part 2 will download the model files when you run the relevant exercise.

## 1. Open the project in VS Code

1. Open VS Code.
2. Choose **File → Open Folder…** and select the `ml-learning-lab` folder you downloaded.
3. Open **Terminal → New Terminal**. Commands below go in this terminal, not inside a Python file.
4. Check that Python is available:

   **Windows**
   ```powershell
   py --version
   ```

   **macOS / Linux**
   ```bash
   python3 --version
   ```

   Python 3.10 or newer is required. If the command is not found, install Python from [python.org/downloads](https://www.python.org/downloads/) and reopen VS Code. On Windows, select **Add Python to PATH** in the installer if that option appears.

## 2. Create a project environment

A virtual environment keeps this course's libraries separate from other Python projects.

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

If Windows blocks activation, you can run the environment's Python directly as `.venv\Scripts\python.exe` instead. You do not have to change your system-wide security policy.

In VS Code, select the environment for running code: press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS), choose **Python: Select Interpreter**, and select the interpreter inside this project's `.venv` folder. If VS Code asks you to install its Microsoft Python extension, install that extension from the VS Code Extensions panel; it helps VS Code work with Python but is not a Python library.

## 3. Install the libraries for Part 1

**Download/install required:** NumPy and scikit-learn are Python libraries. Install them from the VS Code terminal; do not search for and download individual library files manually.

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If your terminal does not recognize `python`, use `py` on Windows or `python3` on macOS/Linux for these commands.

Run the first exercise:

```bash
python 1-first-ml/01_the_dataset.py
```

You can now work through exercises 01–09. Pytest is included in the `[dev]` extra if you want to run the tests.

## 4. Install the libraries for Part 2 (only when you reach it)

**Additional download/install required:** PyTorch and Hugging Face Transformers. They are not included in the basic Part 1 installation. Run this command in the same activated environment:

```bash
python -m pip install -e ".[pretrained]"
```

This installs the libraries that load and run the model. Installation size and time vary by operating system and available Python wheels. Use a stable internet connection.

## 5. Download the DistilGPT2 model files

Part 2 uses one model only: [`distilbert/distilgpt2` on Hugging Face](https://huggingface.co/distilbert/distilgpt2).

**Model files that will be downloaded:** configuration files (such as `config.json` and `generation_config.json`), tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, and possibly `special_tokens_map.json`), and the learned weights checkpoint `model.safetensors` (about **353 MB**).

You normally **do not need to download these files by hand in your browser**. The course code downloads the needed files from Hugging Face automatically the first time an exercise requests them. Exercise 12 explicitly downloads the selected course files and prints their local cache location. Exercises that load the tokenizer, configuration, or model also fetch their required files if they are not already cached. Keep the internet connection on during the first run; later runs can reuse the cache.

To download and inspect the selected files, run from the repository root:

```bash
python 2-distilgpt2/12_list_the_model_files.py
```

To load the actual pretrained weights, run:

```bash
python 2-distilgpt2/13_load_the_weights.py
```

The files are stored in Hugging Face's local cache, not inside your project folder. The program prints the cache path. You do not need to move the files into the repository. Do not download every alternative model format from the model page, and do not upload the weights or cache to GitHub.

If automatic downloads fail, check your internet connection and try the same command again. If you choose to download files manually, use only the files listed above from the official model page and keep them together in one model directory; however, the automatic method is the intended route for this course.

## 6. Run the model lessons on a 4 GB RAM computer

Start with CPU inference; a GPU is not required. Close memory-heavy applications, keep prompts short, and generate only a few tokens while learning. The checkpoint is about 353 MB, but Python, PyTorch, Transformers and intermediate tensors use additional RAM, so **the full program needs much more memory than the checkpoint alone**. Four GB is a constrained target and performance can vary. If the computer runs out of memory, stop other applications and restart with a shorter exercise.

Run one numbered file at a time from the repository root, for example:

```bash
python 2-distilgpt2/10_tokenize_text.py
```

Part 3 reuses the same dependencies and model. If you completed Part 2, there is **no extra package or model download** just for the inference-engine lessons. The first model run in Part 3 can still contact Hugging Face if a required file is missing from your cache. Run one exercise at a time:

```bash
python 3-inference-engine/19_define_a_request.py
python 3-inference-engine/20_build_a_tiny_engine.py
```

Do not use `pipeline()` in this course: the goal is to see the tokenizer, model object, weights, forward pass, and token-selection loop directly.
