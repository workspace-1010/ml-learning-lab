"""Exercise 07 — Evaluate predictions on examples the model did not train on.

Change test_size, then observe how the test examples and error can change.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
y = np.array([3, 5, 7, 9, 11, 13, 15, 16], dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7
)
model = LinearRegression().fit(X_train, y_train)
predictions = model.predict(X_test)

print("Actual:", y_test.tolist())
print("Predicted:", np.round(predictions, 2).tolist())
print("Mean absolute error:", round(float(mean_absolute_error(y_test, predictions)), 3))
