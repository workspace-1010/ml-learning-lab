# Project 1 — Train a neural network to recognize handwritten digits

We will train a small neural network **from scratch using NumPy** to recognize the digits 0–9 in the MNIST dataset. “From scratch” means we write the forward pass, loss calculation, backpropagation, and gradient-descent updates ourselves. NumPy still performs the underlying array arithmetic; we are not implementing matrix multiplication in pure Python.

This is a learning project, not a state-of-the-art digit recognizer.

## What you will build

1. Download and inspect MNIST: 28 × 28 grayscale images and their digit labels.
2. Train a neural network with 784 input values, one hidden layer of 64 neurons, and 10 output scores.
3. See training loss and test accuracy after each epoch.
4. Display test images beside the model's predictions and confidence scores.
5. Open one real training step and inspect the forward pass, loss, gradients, and weight update visually.

## Where does MNIST come from?

MNIST is a well-known public dataset of handwritten digits created by Yann LeCun and collaborators. The project uses four IDX gzip files from a public Google-hosted mirror:

- Training images: [train-images-idx3-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz)
- Training labels: [train-labels-idx1-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz)
- Test images: [t10k-images-idx3-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz)
- Test labels: [t10k-labels-idx1-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz)

You can get these files in **either of two ways**.

### Option A — Let the Python script download them (recommended for your first run)

Run the downloader from the repository root:

```bash
python 1-first-ml/mnist/01_download_mnist.py
```

It downloads any missing raw files, reads them, and creates `data/mnist/mnist.npz`. You need an internet connection for the initial download.

### Option B — Download the files manually

Choose this if you want to download the dataset yourself or learn how the files are organized.

1. Create this folder inside the repository: `data/mnist/raw/`.
2. Click each of the four links above and save the downloaded file into that folder. Keep the files compressed as `.gz`; **do not unzip or rename them**.
3. Check that the names match these exactly:
   - `train-images-idx3-ubyte.gz`
   - `train-labels-idx1-ubyte.gz`
   - `t10k-images-idx3-ubyte.gz`
   - `t10k-labels-idx1-ubyte.gz`
4. From the repository root, run the same preparation script:

```bash
python 1-first-ml/mnist/01_download_mnist.py
```

The script will reuse the four files already in `data/mnist/raw/` instead of downloading them again, then parse them and create `data/mnist/mnist.npz`. Manual downloading does **not** change the training steps.

If your browser saves the files to your usual Downloads folder, move them into `data/mnist/raw/` before running the script. The `data/mnist/` directory is ignored by Git; do not upload the dataset files to the repository.

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

If you prefer to install optional visual tools later, you can skip this install until Exercise 03. The download and training exercises only need NumPy, which is part of the basic setup.

## Run the project

Once `data/mnist/mnist.npz` has been created using Option A or Option B, run these commands from the **repository root**, not from this folder:

```bash
python 1-first-ml/mnist/02_train_mnist.py
python 1-first-ml/mnist/03_visualize_predictions.py
python 1-first-ml/mnist/04_visualize_training_step.py
```

Exercise 02 trains for 3 epochs by default. Exercise 03 visualizes predictions. Exercise 04 opens one real training example and visualizes the intermediate values and parameter update; it does not overwrite the saved model. You can change this, for example:

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

- If automatic download fails, check your internet connection and rerun Exercise 01.
- If you downloaded files manually, verify all four are in `data/mnist/raw/`, their names match exactly, and they remain `.gz` files.
- If training is slow, keep the default 3 epochs and close other heavy applications.
- If Exercise 03 or 04 says the model file is missing, run Exercise 02 first.
- Exercise 04 uses the trained weights and a real training image to demonstrate one additional update in memory. It does not save that update.
- If Matplotlib is missing, install it with the command above.
- Dataset files and trained model weights are generated local files, not source code. They should not be committed to GitHub.
