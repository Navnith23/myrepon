import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/auto-mpg.csv")
df['horsepower'] = pd.to_numeric(df['horsepower'],errors='coerce')
df = df.dropna(subset = ['horsepower','mpg'])
X = df['horsepower'].values
y = df['mpg'].values
x_poly = np.column_stack((np.ones(len(X)),X,X**2))
theta = np.linalg.inv(x_poly.T@x_poly)@x_poly.T@y
y_pred = x_poly@theta
mse = np.mean((y - y_pred)**2)

print("Coefficients: ",theta)
print("Mean Squared Error: ",mse)

plt.scatter(X,y,label = 'Actual MPG')
idx = np.argsort(X)
plt.plot(X[idx],y_pred[idx],color = 'red',label = 'Polynomial Regression')
plt.xlabel('horsepower')
plt.ylabel('MPG')
plt.legend()
plt.show()