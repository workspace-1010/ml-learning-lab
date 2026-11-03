"""Exercise 09 — Compare a simple model with a more flexible one.

A model can fit training examples extremely well and still perform poorly
on unseen examples. Try changing max_depth.
"""

from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=250, noise=0.28, random_state=12)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=4
)

for depth in (2, None):
    model = DecisionTreeClassifier(max_depth=depth, random_state=4)
    model.fit(X_train, y_train)
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))
    print(f"max_depth={depth!s:>4} | train accuracy={train_score:.3f} | test accuracy={test_score:.3f}")
