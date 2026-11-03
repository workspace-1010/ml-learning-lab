# Setup guide

Requires Python 3.10+, Git, and a code editor. Use a virtual environment so project dependencies stay isolated.

## Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python lessons/01-basic-ml/01-first-model/example.py
```

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python lessons/01-basic-ml/01-first-model/example.py
```

Optional model stages may download large files and use substantial disk/RAM. Install their extras only when you reach them: `python -m pip install -e ".[pretrained]"` or `python -m pip install -e ".[rag]"`. Check model licenses and never commit API keys.
