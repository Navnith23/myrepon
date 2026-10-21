import pandas as pd
import numpy as np

df = pd.read_csv("data/breast-cancer.csv")

# Separate features and target
X = df.drop(columns = ["diagnosis"]).values.astype(float)
y = np.where(df["diagnosis"].values == 'M',1,0)

# Split dataset into training and testing data
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8*len(X))
train = indices[:split]
test = indices[split:]

x_test , x_train = X[test] , X[train] 
y_train , y_test = y[train] , y[test]

# Normalize features using training data only
mean = x_train.mean(axis = 0)
std = x_train.std(axis = 0) + 1e-8
x_train = (x_train - mean)/std
x_test = (x_test - mean)/std
# Add bias column
x_train = np.column_stack((np.ones(len(x_train)),x_train))
x_test = np.column_stack((np.ones(len(x_test)),x_test))
# Initialize weights
weights = np.zeros(x_train.shape[1])
learning_rate = 0.01
epoch = 1000

# Sigmoid function
def sigmoid(z):
    z = np.clip(z,-500,500)
    return 1/(1 + np.exp(-z))

# Train using gradient descent
for i in range(epoch):
    z = x_train@weights
    pred = sigmoid(z)
    error = pred - y_train
    gradient = x_train.T@error/len(y_train)
    weights = weights - learning_rate*gradient #gradient descent

# Predict on test data
prob = x_test@weights
test_pred = sigmoid(prob)
y_pred = (test_pred >= 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)*100
print("Accuracy: ",round(accuracy,2),"%")

