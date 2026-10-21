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
x_train , x_test , y_train , y_test = train_test_split(X , y , test_size= 0.2 , random_state= 42 , stratify= y)

# Normalize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Activation functions to compare
activ = ["logistic" , "relu" , "tanh"]
result = []

for act in activ:
    #create neural network
    model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation=act,
    solver="adam",
    max_iter=2000,
    early_stopping=True,
    alpha=0.001,
    random_state=42)

    # Train the model
    model.fit(x_train,y_train)

    #predictions
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    # Probabilities for calculating loss
    train_prob = model.predict_proba(x_train)
    test_prob = model.predict_proba(x_test)

    # Calculate accuracy
    train_acc = accuracy_score(y_train,train_pred)*100
    test_acc = accuracy_score(y_test , test_pred)

    #Calculate loss
    train_loss = log_loss(y_train,train_prob , labels = model.classes_)
    test_loss = log_loss(y_test , test_prob , labels = model.classes_)

    # Store results
    result.append({
        "Activation": act,
        "Training Accuracy (%)": round(train_acc, 2),
        "Testing Accuracy (%)": round(test_acc, 2),
        "Training Loss": round(train_loss, 4),
        "Testing Loss": round(test_loss, 4),
        "Iterations": model.n_iter_
    })


# Display comparison
results_df = pd.DataFrame(result)
print(results_df.to_string(index=False))




