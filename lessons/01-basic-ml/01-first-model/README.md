# First ML model: predict a numeric value

## Goal

Understand the supervised-learning loop: create data, split it, create a model, call fit(), call predict(), and measure error.

## Mental model

- X contains features; rows are examples and columns are input variables.
- y contains the target value for each example.
- train_test_split() keeps some examples out of training.
- LinearRegression() creates a model object.
- fit(X_train, y_train) estimates model parameters.
- predict(X_test) produces predictions for held-out inputs.
- mean_absolute_error() summarizes average absolute prediction error.

## Run

```bash
python lessons/01-basic-ml/01-first-model/example.py
```

## Experiment

Change one target value or change test_size. Predict how the learned slope and test error might change, then run the script and compare.

## Check your understanding

1. What is the difference between fit() and predict()?
2. Why evaluate on held-out examples?
3. Does low test error prove a model works on every real-world dataset? Why not?
