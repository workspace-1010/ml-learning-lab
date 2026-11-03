"""Exercise 01 — Look at the data before talking about a model.

Run this file and explain what one row means.
"""

import numpy as np

# One example per row. The feature is a made-up number of study hours.
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)

# A made-up score corresponding to each row.
y = np.array([3, 5, 7, 9, 11, 13, 15, 16], dtype=float)

print("Features (X):")
print(X)
print("Targets (y):")
print(y)
print("Number of examples:", len(X))
