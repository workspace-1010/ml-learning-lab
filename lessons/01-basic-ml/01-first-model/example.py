"""A tiny example of the standard supervised ML workflow."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def main() -> None:
    # Each row is one example; the single column is the feature.
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
    # One target intentionally deviates slightly from a straight line.
    y = np.array([3, 5, 7, 9, 11, 13, 15, 16], dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Training examples:", len(X_train))
    print("Test features:", X_test.ravel().tolist())
    print("Actual targets:", y_test.tolist())
    print("Predictions:", np.round(predictions, 2).tolist())
    print("Learned slope:", round(float(model.coef_[0]), 3))
    print("Learned intercept:", round(float(model.intercept_), 3))
    print("Test MAE:", round(float(mean_absolute_error(y_test, predictions)), 3))


if __name__ == "__main__":
    main()
