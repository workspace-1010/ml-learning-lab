"""Exercise 04 — Create a model object, but do not train it yet.

A model object is not the same thing as a trained model. Try inspecting
its attributes, then notice which learned attributes do not exist yet.
"""

from sklearn.linear_model import LinearRegression

model = LinearRegression()

print("Model class:", type(model).__name__)
print("Chosen model setting (fit_intercept):", model.fit_intercept)
print("The object exists. Its learned coefficients appear after fit().")
