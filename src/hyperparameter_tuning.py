import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,log_loss

# Read dataset
df = pd.read_csv('data/WineQT.csv')

# Separate features and target
X = df.drop(columns = ["quality","Id"],errors = 'ignore')
y = df['quality']

# Split dataset into training and testing data
x_temp , x_test , y_temp , y_test = train_test_split(X , y , test_size= 0.2 , random_state= 42 , stratify= y)
x_train , x_val , y_train , y_val = train_test_split(x_temp , y_temp , test_size= 0.25 , random_state= 42 , stratify= y_temp)

# Normalize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# Six models for the three experiment pairs
models = [
("Pair 1 - Model 1", 0.001, 64, (128, 64)),
("Pair 1 - Model 2", 0.01, 64, (128, 64)),
("Pair 2 - Model 1", 0.001, 64, (128,)),
("Pair 2 - Model 2", 0.001, 64, (256, 128, 64, 32)),
("Pair 3 - Model 1", 0.001, 32, (128, 64)),
("Pair 3 - Model 2", 0.001, 128, (128, 64))]

result = []

for name , lr , batch , layers in models:
    #create neural network
    model = MLPClassifier(
    hidden_layer_sizes=layers,
    activation="relu",
    solver="adam",
    learning_rate_init=lr,
    batch_size=batch,
    max_iter=1000,
    random_state=42)

    # Train the model
    model.fit(x_train,y_train)

    #predictions
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    val_pred = model.predict(x_val)

    # Calculate accuracy
    train_acc = accuracy_score(y_train,train_pred)*100
    test_acc = accuracy_score(y_test , test_pred)*100
    val_acc = accuracy_score(y_val , val_pred)*100

    # Store results
    result.append({
        "Model": name,
        "Learning Rate": lr,
        "Batch Size": batch,
        "Hidden Layers": str(layers),
        "Training Accuracy (%)": round(train_acc, 2),
        "Validation Accuracy (%)": round(val_acc, 2),
        "Test Accuracy (%)": round(test_acc, 2)
    })

# Display comparison
results_df = pd.DataFrame(result)
print(results_df.to_string(index=False))

