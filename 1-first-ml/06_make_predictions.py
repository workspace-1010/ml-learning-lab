"""Exercise 06 — Ask the trained model to predict for new inputs.

Compare the predictions with the simple relationship in the training data.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
y = np.array([3, 5, 7, 9, 11, 13, 15, 16], dtype=float)

model = LinearRegression()
model.fit(X, y)

new_examples = np.array([[2.5], [5.5], [9.0]])
predictions = model.predict(new_examples)

for feature, prediction in zip(new_examples.ravel(), predictions):
    print(f"Input {feature:g} -> predicted target {prediction:.2f}")
