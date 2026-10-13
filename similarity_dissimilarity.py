import pandas as pd
import numpy as np
df = pd.read_csv('students.csv')
n = len(df)
nom_dis = np.zeros((n,n))
num_dis = np.zeros((n,n))
nom_col = "Department"
num_col = "Marks"
values = df[num_col].astype(float)
value_range = values.max() - values.min()
for i in range(n):
    for j in range(n):
        nom_dis[i][j] = (df[nom_col].iloc[i] != df[nom_col].iloc[j])
        if value_range != 0:
            num_dis[i][j] = abs(values.iloc[i] - values.iloc[j]) / value_range
nom_sim = 1 - nom_dis
num_sim = 1 - num_dis
mixed_dis = (nom_dis + num_dis) / 2
mixed_sim = 1 - mixed_dis
print("Nominal Similarity Matrix:\n", nom_sim)
print("Numerical Similarity Matrix:\n", num_sim.round(2))
print("Nominal Dissimilarity Matrix:\n", nom_dis)
print("Numerical Dissimilarity Matrix:\n", num_dis.round(2))
print("Mixed Similarity Matrix:\n", mixed_sim.round(2))
print("Mixed Dissimilarity Matrix:\n", mixed_dis.round(2))