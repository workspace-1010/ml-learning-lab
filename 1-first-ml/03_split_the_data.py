"""Exercise 03 — Keep some examples aside for a fair test.

Predict how many rows will be in each split before running the script.
"""

import numpy as np
from sklearn.model_selection import train_test_split

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
y = np.array([3, 5, 7, 9, 11, 13, 15, 16], dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Features held out for testing:", X_test.ravel().tolist())
print("Matching targets:", y_test.tolist())
