
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

# Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Try different eps values
eps_values = [5, 7, 9, 11, 13]
min_samples = 5

results = []

for eps in eps_values:

    # Apply DBSCAN
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    # Count clusters, excluding noise (-1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Count noise points
    n_noise = np.sum(labels == -1)

    # Compare clusters with actual digit labels
    ari = adjusted_rand_score(y, labels)

    results.append([
        eps, n_clusters, n_noise, round(ari, 3)
    ])

    # Plot clusters
    plt.figure(figsize=(7, 5))
    plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels, cmap="rainbow", s=10
    )
    plt.title(f"DBSCAN: eps={eps}, clusters={n_clusters}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# Display comparison table
print(pd.DataFrame(
    results,
    columns=["Eps", "Clusters", "Noise Points", "ARI"]
).to_string(index=False))
