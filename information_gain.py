import pandas as pd
import math
df = pd.read_csv('data.csv')

def entropy(data):
    total = len(data)
    counts = data.value_counts()
    result = 0
    for count in counts:
      p = count / total
      result -= p*math.log2(p)
    return result
target = "Buys_Computer"
total_entropy = entropy(df[target])
print("Total Entropy:", round(total_entropy, 2))
attribute_ig ={}
for attribute in df.columns[:-1]:
   weighted_entropy = 0
   for value in df[attribute].unique():
        subset = df[df[attribute] == value]
        weighted_entropy += (len(subset)/len(df) * entropy(subset[target]))
        ig = total_entropy - weighted_entropy
        attribute_ig[attribute] = ig
print("\nInformation Gain for each attribute:")
for attribute, ig in attribute_ig.items():
    print(f"{attribute}: {round(ig, 2)}")
best = max(attribute_ig,key = attribute_ig.get)
print("\nBest attribute to split on:", best)
print(df[best])



