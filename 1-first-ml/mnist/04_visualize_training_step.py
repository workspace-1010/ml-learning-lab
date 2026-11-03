"""Exercise 04 — Open one real MNIST training step and inspect its internals.

Run from the repository root, after Exercises 01 and 02:
    python 1-first-ml/mnist/04_visualize_training_step.py

This uses the same NumPy equations as Exercise 02 on one real training example.
It shows the values before and after one gradient-descent update, but does not
change the saved model. Matplotlib is used only to display the calculations.
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
LEARNING_RATE = 0.1


def softmax(logits: np.ndarray) -> np.ndarray:
    """Turn ten output scores into probabilities that sum to one."""
    shifted = logits - logits.max()
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


def inspect_one_step(
    image: np.ndarray, label: int, params: dict[str, np.ndarray]
) -> dict[str, object]:
    """Run forward pass, loss, backpropagation, and one in-memory update."""
    x = image.reshape(1, 784).astype(np.float32) / 255.0

    # 1. Forward pass: pixels -> hidden layer -> ten digit probabilities.
    hidden_raw = x @ params["W1"] + params["b1"]
    hidden = np.maximum(hidden_raw, 0.0)  # ReLU
    logits = (hidden @ params["W2"] + params["b2"])[0]
    probabilities = softmax(logits)

    # 2. Loss: how much probability did the model give the correct digit?
    loss = -np.log(np.clip(probabilities[label], 1e-9, 1.0))

    # 3. Backpropagation: calculate gradients for every parameter.
    d_logits = probabilities.copy()
    d_logits[label] -= 1.0
    grad_W2 = hidden.T @ d_logits.reshape(1, -1)
    grad_b2 = d_logits
    d_hidden = (d_logits @ params["W2"].T) * (hidden_raw[0] > 0.0)
    grad_W1 = x.T @ d_hidden.reshape(1, -1)
    grad_b1 = d_hidden

    gradients = {
        "W1": grad_W1,
        "b1": grad_b1,
        "W2": grad_W2,
        "b2": grad_b2,
    }

    # 4. Apply one gradient-descent update to a copy, never the saved weights.
    updated = {name: value.copy() for name, value in params.items()}
    for name, gradient in gradients.items():
        updated[name] -= LEARNING_RATE * gradient

    updated_hidden = np.maximum(x @ updated["W1"] + updated["b1"], 0.0)
    after_probabilities = softmax(
        (updated_hidden @ updated["W2"] + updated["b2"])[0]
    )
    after_loss = -np.log(np.clip(after_probabilities[label], 1e-9, 1.0))

    return {
        "hidden": hidden[0],
        "probabilities": probabilities,
        "after_probabilities": after_probabilities,
        "loss": float(loss),
        "after_loss": float(after_loss),
        "gradients": gradients,
        "weight_delta": updated["W2"] - params["W2"],
    }


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit("Dataset missing. Run Exercise 01 first.")
    if not MODEL_FILE.exists():
        raise SystemExit("Trained weights missing. Run Exercise 02 first.")

    with np.load(DATA_FILE) as data:
        images = data["train_images"]
        labels = data["train_labels"]
    with np.load(MODEL_FILE) as saved:
        params = {name: saved[name].copy() for name in ("W1", "b1", "W2", "b2")}

    # Choose a real example the trained model has not had to use as a fixed demo.
    index = 0
    image = images[index]
    label = int(labels[index])
    result = inspect_one_step(image, label, params)

    probabilities = result["probabilities"]
    after_probabilities = result["after_probabilities"]
    predicted_before = int(np.argmax(probabilities))
    predicted_after = int(np.argmax(after_probabilities))
    gradients = result["gradients"]

    print("ONE REAL TRAINING STEP — all values below come from this image")
    print(f"Actual label: {label}")
    print(f"Prediction before update: {predicted_before}")
    print(f"Loss before update: {result['loss']:.5f}")
    print(f"Loss after one update:  {result['after_loss']:.5f}")
    print(f"Largest absolute W2 gradient: {np.abs(gradients['W2']).max():.6f}")
    print(f"Largest absolute W2 change:   {np.abs(result['weight_delta']).max():.6f}")
    print("The update is in memory only; artifacts/mnist_numpy.npz is unchanged.")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"1. Input image — correct label: {label}")
    axes[0, 0].axis("off")

    x_positions = np.arange(10)
    width = 0.38
    axes[0, 1].bar(x_positions - width / 2, probabilities, width, label="Before")
    axes[0, 1].bar(x_positions + width / 2, after_probabilities, width, label="After 1 update")
    axes[0, 1].set_xticks(x_positions)
    axes[0, 1].set_xlabel("Digit class")
    axes[0, 1].set_ylabel("Predicted probability")
    axes[0, 1].set_title("2. Output probabilities")
    axes[0, 1].legend()

    axes[1, 0].bar(np.arange(1, len(result["hidden"]) + 1), result["hidden"])
    axes[1, 0].set_xlabel("Hidden neuron (1–64)")
    axes[1, 0].set_ylabel("Activation")
    axes[1, 0].set_title("3. Hidden layer after ReLU")

    gradient_image = gradients["W2"]
    im = axes[1, 1].imshow(gradient_image, aspect="auto", cmap="coolwarm")
    axes[1, 1].set_xlabel("Output digit (0–9)")
    axes[1, 1].set_ylabel("Hidden neuron (1–64)")
    axes[1, 1].set_title("4. Backpropagated gradients for W2")
    figure.colorbar(im, ax=axes[1, 1], label="Gradient value")

    figure.suptitle(
        f"One training step: loss {result['loss']:.4f} → {result['after_loss']:.4f} "
        f"(prediction {predicted_before} → {predicted_after})",
        fontsize=14,
    )
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
