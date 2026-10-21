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

# Logistic Regression from Scratch

## Program

This program implements Logistic Regression on the Breast Cancer dataset without using `scikit-learn`.

### Input

- Dataset: `breast-cancer.csv`
- Features: All columns except `diagnosis` (remove `id` if present).
- Target: `diagnosis` (`M` = 1, `B` = 0)

## Algorithm

1. Read `breast-cancer.csv` using `pd.read_csv()` and store it in `df`.
2. Extract all feature columns except `diagnosis` into `X` and convert them to floating-point values.
3. Convert the `diagnosis` column into numerical values using `np.where()`, assigning `1` to malignant (`M`) and `0` to benign (`B`), and store the result in `y`.
4. Set the random seed using `np.random.seed(42)`.
5. Generate shuffled indices using `np.random.permutation(len(X))` and store them in `indices`.
6. Calculate the 80% split position using `split = int(0.8 * len(X))`.
7. Divide the indices into `train` and `test`.
8. Extract the training features and target into `X_train` and `y_train`.
9. Extract the testing features and target into `X_test` and `y_test`.
10. Calculate the mean and standard deviation of each feature using `X_train`.
11. Normalize `X_train` and `X_test` using the training mean and standard deviation. Add `1e-8` to the standard deviation to avoid division by zero.
12. Add a bias column of ones to `X_train` and `X_test` using `np.column_stack()`.
13. Initialize the weights to zeros using `np.zeros()`.
14. Set the learning rate to `0.01` and the number of training epochs to `1000`.
15. Define the sigmoid function to convert the model's output into a probability between 0 and 1.
16. Clip the input of the sigmoid function using `np.clip()` to prevent numerical overflow.
17. Calculate the linear output using `z = X_train @ weights`.
18. Apply the sigmoid function to `z` to calculate the predicted probabilities.
19. Calculate the error by subtracting `y_train` from the predictions.
20. Calculate the gradient using `(X_train.T @ error) / len(y_train)`.
21. Update the weights using `weights = weights - learning_rate * gradient`.
22. Repeat steps 17–21 for `1000` epochs to train the model.
23. Calculate the predicted probabilities for `X_test` using the trained weights and sigmoid function.
24. Convert the probabilities into class predictions using a threshold of `0.5`: probabilities greater than or equal to `0.5` are classified as `1`, and the rest as `0`.
25. Calculate the test accuracy by comparing `y_pred` with `y_test`.
26. Display the accuracy as a percentage.

## Files

```
breast-cancer.csv
logistic_regression.py
README.md
```

## Libraries Used

```
pandas
numpy
```

`scikit-learn` is not used.
