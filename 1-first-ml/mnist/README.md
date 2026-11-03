# Project 1 — Train a neural network to recognize handwritten digits

We will train a small neural network **from scratch using NumPy** to recognize the digits 0–9 in the MNIST dataset. “From scratch” means we write the forward pass, loss calculation, backpropagation, and gradient-descent updates ourselves. NumPy still performs the underlying array arithmetic; we are not implementing matrix multiplication in pure Python.

This is a learning project, not a state-of-the-art digit recognizer.

## What you will build

1. Download and inspect MNIST: 28 × 28 grayscale images and their digit labels.
2. Train a neural network with 784 input values, one hidden layer of 64 neurons, and 10 output scores.
3. See training loss and test accuracy after each epoch.
4. Display test images beside the model's predictions and confidence scores.

## Where does MNIST come from?

MNIST is a well-known public dataset of handwritten digits created by Yann LeCun and collaborators. The script downloads the four original IDX gzip files from a public Google-hosted mirror of the dataset:

- Training images: [train-images-idx3-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz)
- Training labels: [train-labels-idx1-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz)
- Test images: [t10k-images-idx3-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz)
- Test labels: [t10k-labels-idx1-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz)

You do **not** need to download those four files manually. Exercise 01 downloads them and creates a local `mnist.npz` dataset file. You need an internet connection for the first download. The files are stored under `data/mnist/`, which is ignored by Git; do not upload the dataset to the repository.

## Libraries and tools used

| Library/tool | Built into Python? | Why we use it |
|---|---|---|
| Python 3.10+ | No; install Python itself | Runs the lessons |
| `urllib.request`, `gzip`, `struct`, `pathlib` | Yes — Python standard library | Download files, decompress them, read the IDX file headers, and manage paths |
| NumPy | No | Store image arrays and implement the neural network calculations and training updates |
| Matplotlib | No | Display handwritten digits and predictions in Exercise 03 |

No TensorFlow, PyTorch, Transformers, or pretrained model is used for this project. We train the network's weights ourselves.

## Setup

From the repository root, create/activate the course virtual environment and install the basic course dependencies using the [main setup guide](../../docs/setup-guide.md). Then install Matplotlib for the visualizer:

```bash
python -m pip install "matplotlib>=3.8,<4"
```

If you prefer to install all optional visual tools later, you can skip this install until Exercise 03. The download and training exercises only need NumPy, which is already part of the basic setup.

## Run the project

Run commands from the **repository root**, not from this folder:

```bash
python 1-first-ml/mnist/01_download_mnist.py
python 1-first-ml/mnist/02_train_mnist.py
python 1-first-ml/mnist/03_visualize_predictions.py
```

Exercise 02 trains for 3 epochs by default. You can change this, for example:

```bash
python 1-first-ml/mnist/02_train_mnist.py --epochs 5
```

Training uses the CPU and a mini-batch of 128 examples. It may take a while depending on your computer. Start with the default 3 epochs before increasing it. The trained parameters are saved to `artifacts/mnist_numpy.npz` and are ignored by Git.

## What is the network learning?

- **Input:** each 28 × 28 image is flattened into 784 numbers and scaled to the range 0–1.
- **Hidden layer:** 64 neurons combine inputs using weights and biases, then apply ReLU.
- **Output layer:** 10 scores represent digits 0 through 9.
- **Loss:** cross-entropy measures how poorly the scores match the correct digit.
- **Backpropagation:** calculates how the loss changes with each weight and bias.
- **Gradient descent:** adjusts parameters to reduce the loss over many mini-batches.

The test images are held out from training. Test accuracy is a useful estimate of performance on unseen examples, not a guarantee that every handwritten digit will be recognized correctly.

## Troubleshooting

- If download fails, check your internet connection and rerun Exercise 01.
- If training is slow, keep the default 3 epochs and close other heavy applications.
- If Exercise 03 says the model file is missing, run Exercise 02 first.
- If Matplotlib is missing, install it with the command above.
- The data and trained model are local generated files, not source code. They should not be committed to GitHub.
