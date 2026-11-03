"""Exercise 03 — View test digits and your network's predictions.

Requires Matplotlib. Run Exercise 02 first to train and save the network.
Run from the repository root:
    python 1-first-ml/mnist/03_visualize_predictions.py
"""

from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as error:
    raise SystemExit(
        "This visual exercise needs Matplotlib. Install it with: "
        'python -m pip install "matplotlib>=3.8,<4"'
    ) from error

DATA_FILE = Path("data/mnist/mnist.npz")
MODEL_FILE = Path("artifacts/mnist_numpy.npz")


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit("Dataset missing. Run Exercise 01 first.")
    if not MODEL_FILE.exists():
        raise SystemExit("Trained weights missing. Run Exercise 02 first.")

    with np.load(DATA_FILE) as data:
        images = data["test_images"][:16]
        labels = data["test_labels"][:16]
    with np.load(MODEL_FILE) as saved:
        params = {name: saved[name] for name in ("W1", "b1", "W2", "b2")}

    x = images.reshape(len(images), 784).astype(np.float32) / 255.0
    hidden_raw = x @ params["W1"] + params["b1"]
    hidden = np.maximum(hidden_raw, 0.0)
    probabilities = softmax(hidden @ params["W2"] + params["b2"])
    predictions = probabilities.argmax(axis=1)

    figure, axes = plt.subplots(4, 4, figsize=(9, 9))
    for index, axis in enumerate(axes.flat):
        axis.imshow(images[index], cmap="gray")
        confidence = probabilities[index, predictions[index]]
        color = "green" if predictions[index] == labels[index] else "red"
        axis.set_title(
            f"Pred: {predictions[index]} ({confidence:.0%})\nActual: {labels[index]}",
            color=color,
        )
        axis.axis("off")
    figure.suptitle("MNIST: green = correct, red = incorrect", fontsize=14)
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
