"""Exercise 01 — Download and inspect MNIST.

Run from the repository root:
    python 1-first-ml/mnist/01_download_mnist.py

The four gzip files are downloaded from a public mirror of MNIST, then parsed
with Python's standard library. NumPy stores the arrays in a local NPZ file.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/mnist")
RAW_DIR = DATA_DIR / "raw"
OUTPUT_FILE = DATA_DIR / "mnist.npz"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def download_file(filename: str) -> Path:
    """Download one compressed MNIST file once and reuse the local copy."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DIR / filename
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already downloaded: {destination}")
        return destination

    url = BASE_URL + filename
    print(f"Downloading {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "ML-Learning-Lab/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            destination.write_bytes(response.read())
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    print(f"Saved {destination} ({destination.stat().st_size:,} bytes)")
    return destination


def read_images(path: Path) -> np.ndarray:
    """Read an IDX image file: header followed by rows of unsigned bytes."""
    with gzip.open(path, "rb") as file:
        magic, count, rows, columns = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image-file magic number: {magic}")
        data = file.read()
    expected = count * rows * columns
    if len(data) != expected:
        raise ValueError(f"Expected {expected} image bytes, got {len(data)}")
    return np.frombuffer(data, dtype=np.uint8).reshape(count, rows, columns).copy()


def read_labels(path: Path) -> np.ndarray:
    """Read an IDX label file: header followed by one byte per label."""
    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label-file magic number: {magic}")
        data = file.read()
    if len(data) != count:
        raise ValueError(f"Expected {count} label bytes, got {len(data)}")
    return np.frombuffer(data, dtype=np.uint8).copy()


def main() -> None:
    if OUTPUT_FILE.exists():
        print(f"Dataset already exists: {OUTPUT_FILE}")
        print("Delete that file if you want to recreate it from the downloaded files.")
        return

    paths = {name: download_file(filename) for name, filename in FILES.items()}
    train_images = read_images(paths["train_images"])
    train_labels = read_labels(paths["train_labels"])
    test_images = read_images(paths["test_images"])
    test_labels = read_labels(paths["test_labels"])

    if len(train_images) != len(train_labels) or len(test_images) != len(test_labels):
        raise ValueError("Image counts and label counts do not match.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_FILE,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )
    print("\nMNIST is ready.")
    print(f"Training images: {train_images.shape}; labels: {train_labels.shape}")
    print(f"Test images:     {test_images.shape}; labels: {test_labels.shape}")
    print(f"Pixel range: {int(train_images.min())} to {int(train_images.max())}")
    print(f"Saved dataset:   {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
