import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_csv('data/california_housing.csv')
X = data['MedInc'].values
y = data['MedHouseVal'].values
np.random.seed(42)
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]
split = int(0.8*len(X))
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]
X_mean = np.mean(X_train)
y_mean = np.mean(y_train)
numerator = np.sum((X_train-X_mean)*(y_train-y_mean))
denominator = np.sum((X_train-X_mean)**2)
b1 = numerator/denominator
b0 = y_mean - b1*X_mean
y_train_pred = b0 + b1*X_train
y_test_pred = b0 + b1*X_test
mse = np.sum((y_test - y_test_pred)**2)
ss_total = np.sum((y_test - np.mean(y_test))**2)
ss_residual = np.sum((y_test - y_test_pred)**2)
r2 = 1 - (ss_residual/ss_total)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
print("MedInc")
print(f'Coefficients: b0 = {b0}, b1 = {b1}')
print(f'Predicted values for test set: {y_test_pred}')
print(f'Actual values for test set: {y_test}')
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.plot(X_train, y_train_pred, color='red', label='Regression line')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Linear Regression: Median Income vs Median House Value')
plt.legend()
plt.show()