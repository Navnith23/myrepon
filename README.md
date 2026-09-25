# Linear Regression with One Variable

## Program

This program implements simple linear regression on the California Housing dataset without using `scikit-learn`.

### Input

* Dataset: `california_housing.csv`
* Feature: `MedInc`
* Target: `MedHouseVal`

### Algorithm

1. Read `california_housing.csv` using `pd.read_csv()` and store it in `data`.
2. Extract the `MedInc` column into `X`.
3. Extract the `MedHouseVal` column into `y`.
4. Set the random seed using `np.random.seed(42)`.
5. Generate shuffled indices using `np.random.permutation(len(X))` and store them in `indices`.
6. Shuffle `X` and `y` using `indices`.
7. Calculate the 80% split position using:
   `split = int(0.8 * len(X))`.
8. Divide `X` into `X_train` and `X_test`.
9. Divide `y` into `y_train` and `y_test`.
10. Calculate `x_mean` using `X_train` and `y_mean` using `y_train`.
11. Calculate `numerator` using:
    `(X_train - x_mean) * (y_train - y_mean)`.
12. Calculate `denominator` using:
    `(X_train - x_mean) ** 2`.
13. Calculate the slope `b1` using `numerator / denominator`.
14. Calculate the intercept `b0` using:
    `y_mean - b1 * x_mean`.
15. Calculate `y_train_pred` using:
    `b0 + b1 * X_train`.
16. Calculate `y_test_pred` using:
    `b0 + b1 * X_test`.
17. Calculate the Test MSE and store it in `mse`.
18. Calculate `ss_total` and `ss_residual`.
19. Calculate the R-squared value and store it in `r2`.
20. Set a new income value in `new_income`.
21. Calculate the predicted house value using `predicted_price`.
22. Display the results and plot `X_test`, `y_test`, and `y_test_pred`.

### Files

```text
california_housing.csv
linear_regression.py
README.md
```

### Libraries Used

```text
pandas
numpy
matplotlib
```

`scikit-learn` is not used.
