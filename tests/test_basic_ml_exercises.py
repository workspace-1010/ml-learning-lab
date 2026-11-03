from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]
EXERCISES = sorted((ROOT / "1-first-ml").glob("*.py"))


def test_each_basic_ml_exercise_runs():
    assert len(EXERCISES) == 9
    for script in EXERCISES:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{script.name}\n{result.stderr}"
