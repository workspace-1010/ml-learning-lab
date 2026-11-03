from pathlib import Path
import subprocess
import sys


def test_first_model_example_runs():
    script = (
        Path(__file__).parents[1]
        / "lessons"
        / "01-basic-ml"
        / "01-first-model"
        / "example.py"
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Test MAE:" in result.stdout
    assert "Predictions:" in result.stdout
