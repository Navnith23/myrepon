import pandas as pd
data = pd.read_csv('data.csv')
target = data.columns[-1]
X = data.drop(columns=[target])
y = data[target]
new_data = {
    "Age": ">40",
    "Income": "Low",
    "Student": "Yes",
    "Credit_Rating": "Excellent"
}
probabilities = {}
classes = y.unique()
for h in classes:
    p_h = (y==h).sum()/len(y)
    probability = p_h
    for feature in X.columns:
        count = ((X[feature]==new_data[feature]) & (y==h)).sum()
        class_count = (y==h).sum()
        p_x_h = count/class_count
        probability *= p_x_h
    probabilities[h] = probability
for h in probabilities:
    print(h,":",probabilities[h])
pred = max(probabilities, key=probabilities.get)
print("predicted class:",pred)
