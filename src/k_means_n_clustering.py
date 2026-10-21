
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------- K-means from scratch --------

k = 5
np.random.seed(42)

# Initialize centroids randomly
centroids = X[np.random.choice(len(X), k, replace=False)]

for iteration in range(100):

    # Calculate distances and assign clusters
    distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
    labels = np.argmin(distances, axis=1)

    # Calculate new centroids
    new_centroids = np.array([
        X[labels == i].mean(axis=0) if np.any(labels == i)
        else centroids[i]
        for i in range(k)
    ])

    # Stop if centroids do not change
    if np.allclose(centroids, new_centroids):
        centroids = new_centroids
        break

    centroids = new_centroids

# Plot K-means clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow")
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker="X", s=200, color="black", label="Centroids")
plt.xlabel("Annual Income (standardized)")
plt.ylabel("Spending Score (standardized)")
plt.title("K-means Clustering")
plt.legend()
plt.show()

df["KMeans_Cluster"] = labels

# -------- Hierarchical clustering --------

model = AgglomerativeClustering(n_clusters=5, linkage="ward")
hier_labels = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=hier_labels, cmap="rainbow")
plt.xlabel("Annual Income (standardized)")
plt.ylabel("Spending Score (standardized)")
plt.title("Hierarchical Clustering")
plt.show()

df["Hierarchical_Cluster"] = hier_labels

# -------- Dendrogram --------

linked = linkage(X, method="ward")

plt.figure(figsize=(12, 5))
dendrogram(linked, truncate_mode="lastp", p=20)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Cluster or sample group")
plt.ylabel("Distance")
plt.show()

# Display results
print(df[["CustomerID", "KMeans_Cluster",
          "Hierarchical_Cluster"]].head(10))
