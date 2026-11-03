"""Exercise 02 — Train a tiny neural network on MNIST using NumPy only.

Run from the repository root:
    python 1-first-ml/mnist/02_train_mnist.py
    python 1-first-ml/mnist/02_train_mnist.py --epochs 5

This implements the forward pass, cross-entropy loss, backpropagation, and
mini-batch gradient descent. It does not use TensorFlow or PyTorch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DATA_FILE = Path("data/mnist/mnist.npz")
MODEL_DIR = Path("artifacts")
MODEL_FILE = MODEL_DIR / "mnist_numpy.npz"
INPUTS = 784
HIDDEN = 64
CLASSES = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.1
SEED = 42


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert each row of scores into probabilities that sum to one."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def forward(x: np.ndarray, params: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Calculate hidden activations and output probabilities."""
    hidden_raw = x @ params["W1"] + params["b1"]
    hidden = np.maximum(hidden_raw, 0.0)  # ReLU activation
    probabilities = softmax(hidden @ params["W2"] + params["b2"])
    return hidden, probabilities


def accuracy(x: np.ndarray, labels: np.ndarray, params: dict[str, np.ndarray]) -> float:
    """Measure the fraction of examples whose most likely digit is correct."""
    correct = 0
    for start in range(0, len(x), BATCH_SIZE):
        batch = x[start : start + BATCH_SIZE]
        _, probabilities = forward(batch, params)
        predicted = probabilities.argmax(axis=1)
        correct += int(np.sum(predicted == labels[start : start + BATCH_SIZE]))
    return correct / len(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3, help="training passes over the data (default: 3)")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if not DATA_FILE.exists():
        raise SystemExit(
            f"Dataset not found: {DATA_FILE}\nRun Exercise 01 first: "
            "python 1-first-ml/mnist/01_download_mnist.py"
        )

    with np.load(DATA_FILE) as data:
        x_train = data["train_images"].reshape(-1, INPUTS).astype(np.float32) / 255.0
        y_train = data["train_labels"].astype(np.int64)
        x_test = data["test_images"].reshape(-1, INPUTS).astype(np.float32) / 255.0
        y_test = data["test_labels"].astype(np.int64)

    rng = np.random.default_rng(SEED)
    params = {
        "W1": (rng.standard_normal((INPUTS, HIDDEN)) * np.sqrt(2.0 / INPUTS)).astype(np.float32),
        "b1": np.zeros(HIDDEN, dtype=np.float32),
        "W2": (rng.standard_normal((HIDDEN, CLASSES)) * np.sqrt(2.0 / HIDDEN)).astype(np.float32),
        "b2": np.zeros(CLASSES, dtype=np.float32),
    }

    print("Network: 784 inputs -> 64 hidden neurons (ReLU) -> 10 digit scores")
    print(f"Training examples: {len(x_train):,}; test examples: {len(x_test):,}")
    print(f"Training from random initial weights for {args.epochs} epoch(s).\n")

    for epoch in range(args.epochs):
        order = rng.permutation(len(x_train))
        total_loss = 0.0
        seen = 0

        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            x = x_train[indices]
            labels = y_train[indices]
            batch_size = len(x)

            hidden_raw = x @ params["W1"] + params["b1"]
            hidden = np.maximum(hidden_raw, 0.0)
            probabilities = softmax(hidden @ params["W2"] + params["b2"])

            loss = -np.log(np.clip(probabilities[np.arange(batch_size), labels], 1e-9, 1.0)).mean()
            total_loss += float(loss) * batch_size
            seen += batch_size

            # Backpropagation: calculate how each parameter contributed to the loss.
            d_logits = probabilities.copy()
            d_logits[np.arange(batch_size), labels] -= 1.0
            d_logits /= batch_size

            grad_W2 = hidden.T @ d_logits
            grad_b2 = d_logits.sum(axis=0)
            d_hidden = d_logits @ params["W2"].T
            d_hidden *= hidden_raw > 0.0  # derivative of ReLU
            grad_W1 = x.T @ d_hidden
            grad_b1 = d_hidden.sum(axis=0)

            # Gradient descent: move parameters in the direction that lowers loss.
            for name, gradient in (
                ("W1", grad_W1), ("b1", grad_b1),
                ("W2", grad_W2), ("b2", grad_b2),
            ):
                params[name] -= LEARNING_RATE * gradient

        train_accuracy = accuracy(x_train, y_train, params)
        test_accuracy = accuracy(x_test, y_test, params)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={total_loss / seen:.4f} | "
            f"train accuracy={train_accuracy:.1%} | "
            f"test accuracy={test_accuracy:.1%}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL_FILE, **params)
    print(f"\nSaved learned weights to {MODEL_FILE}")
    print("Next: python 1-first-ml/mnist/03_visualize_predictions.py")


if __name__ == "__main__":
    main()
