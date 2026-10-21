import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report

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

# Create MLP model
model = MLPClassifier(hidden_layer_sizes= (100,50) , activation= 'relu' , solver = 'adam',learning_rate_init= 0.001,alpha= 0.001, max_iter= 2000 , early_stopping= True , random_state= 42)

# Train the model
model.fit(x_train , y_train)
print("Layer 1:", model.predict(x_test[:1]))
print("Layer sizes:", model.hidden_layer_sizes)
# Predict wine quality
y_pred = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test , y_pred)* 100
print("Classification Report: ")
print(classification_report(y_test , y_pred , zero_division= 0))


