"""Exercise 05 — Let the model learn from examples.

Change one target value and compare the learned slope and intercept.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
y = np.array([3, 5, 7, 9, 11, 13, 15, 16], dtype=float)

model = LinearRegression()
model.fit(X, y)

print("Learned slope (coef_):", round(float(model.coef_[0]), 3))
print("Learned intercept:", round(float(model.intercept_), 3))
