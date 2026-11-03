"""Exercise 02 — Separate the input from the answer we want to predict.

Change the first value of X and observe which part of the dataset changed.
"""

import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
y = np.array([3, 5, 7, 9, 11, 13, 15, 16], dtype=float)

print("First example's feature:", X[0])
print("First example's target:", y[0])
print("X shape (examples, features):", X.shape)
print("y shape (targets,):", y.shape)
