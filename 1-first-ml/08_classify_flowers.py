"""Exercise 08 — Classification predicts a category rather than a number.

This uses scikit-learn's built-in Iris dataset, so no data download is needed.
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=7, stratify=iris.target
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Flower categories:", iris.target_names.tolist())
print("First five predictions:", iris.target_names[predictions[:5]].tolist())
print("First five actual labels:", iris.target_names[y_test[:5]].tolist())
print("Test accuracy:", round(float(accuracy_score(y_test, predictions)), 3))
