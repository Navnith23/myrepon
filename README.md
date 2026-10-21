# Machine Learning Algorithms Lab

A collection of classic machine learning algorithms, implemented either **from scratch with NumPy** or with **scikit-learn**. Each script is self-contained and covers regression, classification, neural networks, and clustering.

---

## Project Structure

```
myrepon-ml/
├── data/
│   ├── california_housing.csv
│   ├── auto-mpg.csv
│   ├── breast-cancer.csv
│   ├── WineQT.csv
│   └── Mall_Customers.csv
└── src/
    ├── exp1linearregression.py
    ├── polynomial_reg.py
    ├── logistic_regression.py
    ├── multilayerperceptron.py
    ├── activation_function.py
    ├── hyperparameter_tuning.py
    ├── k_means_n_clustering.py
    └── dbscan_algo.py
```

| Script | Algorithm | Dataset | Implementation |
|---|---|---|---|
| `exp1linearregression.py` | Simple Linear Regression | California Housing | From scratch (NumPy) |
| `polynomial_reg.py` | Polynomial Regression (degree 2) | Auto MPG | From scratch (NumPy) |
| `logistic_regression.py` | Logistic Regression | Breast Cancer | From scratch (NumPy) |
| `multilayerperceptron.py` | Multi-Layer Perceptron | Wine Quality | scikit-learn |
| `activation_function.py` | Activation function comparison | Wine Quality | scikit-learn |
| `hyperparameter_tuning.py` | Hyperparameter experiments on an MLP | Wine Quality | scikit-learn |
| `k_means_n_clustering.py` | K-Means and Hierarchical (Ward) clustering | Mall Customers | K-Means from scratch, Hierarchical via scikit-learn/SciPy |
| `dbscan_algo.py` | DBSCAN | Digits (built into scikit-learn) | scikit-learn |

## Setup and Usage

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

Scripts read data using relative paths (`data/...`), so run them **from the repository root**:

```bash
python src/exp1linearregression.py
python src/dbscan_algo.py
```

---

## Common Preprocessing

### Standardization (z-score)

Used in every script except the linear and polynomial regression ones:

$$z = \frac{x - \mu}{\sigma}$$

The mean $\mu$ and standard deviation $\sigma$ are computed **on the training set only** and then applied to the test set, which avoids data leakage. The scikit-learn scripts use `StandardScaler`; `logistic_regression.py` does the same by hand and adds $10^{-8}$ to $\sigma$ to avoid division by zero.

### Train/Test Split

- Scratch implementations shuffle with `np.random.permutation` (seed 42) and take the first **80%** for training.
- scikit-learn scripts use `train_test_split(test_size=0.2, random_state=42, stratify=y)`, so class proportions are preserved.

---

## 1. Simple Linear Regression — `exp1linearregression.py`

**Goal:** predict median house value (`MedHouseVal`) from median income (`MedInc`).

### Model

$$\hat{y} = \beta_0 + \beta_1 x$$

### Algorithm: Ordinary Least Squares (closed form)

The parameters minimize the sum of squared residuals:

$$J(\beta_0,\beta_1) = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2$$

Setting the partial derivatives to zero gives:

$$\beta_1 = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^{n}(x_i-\bar{x})^2}, \qquad \beta_0 = \bar{y} - \beta_1\bar{x}$$

### Steps

1. Shuffle the data and split 80/20.
2. Compute $\bar{x}$ and $\bar{y}$ from the training data.
3. Compute $\beta_1$ and $\beta_0$ with the formulas above.
4. Predict on train and test sets.
5. Evaluate on the test set.

### Evaluation Metrics

$$\text{SSE} = \sum_{i}(y_i-\hat{y}_i)^2 \qquad\qquad R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y}_{test})^2}$$

The script prints the summed squared error (labelled "Mean Squared Error" in the output) and $R^2$. It also plots the training data, test data, and fitted line.

---

## 2. Polynomial Regression — `polynomial_reg.py`

**Goal:** model the non-linear relationship between `horsepower` and `mpg`.

### Model (degree 2)

$$\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2$$

### Algorithm: Normal Equation

The feature matrix (design matrix) is built as:

$$\mathbf{X} = \begin{bmatrix} 1 & x_1 & x_1^2 \\ 1 & x_2 & x_2^2 \\ \vdots & \vdots & \vdots \\ 1 & x_n & x_n^2 \end{bmatrix}$$

Minimizing $J(\theta)=\lVert \mathbf{X}\theta - \mathbf{y}\rVert^2$ gives the closed-form solution:

$$\boldsymbol{\theta} = (\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{y}$$

### Steps

1. Convert `horsepower` to numeric. Missing values (`?`) become `NaN` and those rows are dropped.
2. Build the design matrix with a bias column, $x$, and $x^2$.
3. Solve for $\boldsymbol\theta$ with the normal equation.
4. Compute predictions and the mean squared error over the data:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2$$

5. Plot the scatter of actual values with the fitted curve (points sorted by $x$ so the line is smooth).

---

## 3. Logistic Regression — `logistic_regression.py`

**Goal:** binary classification of tumors as malignant (`M` = 1) or benign (`B` = 0).

### Model

A linear score is passed through the sigmoid function to give a probability:

$$z = \mathbf{w}^{T}\mathbf{x}, \qquad \hat{p} = \sigma(z) = \frac{1}{1+e^{-z}}$$

The code clips $z$ to $[-500, 500]$ for numerical stability. A bias column of ones is prepended to $\mathbf{X}$, so $w_0$ is the intercept.

### Loss: Binary Cross-Entropy

$$J(\mathbf{w}) = -\frac{1}{m}\sum_{i=1}^{m}\Big[y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\Big]$$

### Algorithm: Batch Gradient Descent

The gradient of the loss is:

$$\nabla J(\mathbf{w}) = \frac{1}{m}\mathbf{X}^{T}(\hat{\mathbf{p}} - \mathbf{y})$$

and the weights are updated each epoch by:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta\,\nabla J(\mathbf{w})$$

with learning rate $\eta = 0.01$, **1000 epochs**, and weights initialized to zeros.

### Prediction and Evaluation

$$\hat{y} = \begin{cases}1 & \hat{p}\ge 0.5\\ 0 & \text{otherwise}\end{cases} \qquad\qquad \text{Accuracy} = \frac{1}{N}\sum_i \mathbb{1}[\hat{y}_i = y_i]\times 100$$

---

## 4. Multi-Layer Perceptron — `multilayerperceptron.py`

**Goal:** classify wine quality scores (a multi-class problem) from 11 physicochemical features. The `Id` column is dropped.

### Architecture

`MLPClassifier` with two hidden layers of **100** and **50** neurons, ReLU activations, and a softmax output layer.

### Forward Propagation

For layer $l$:

$$\mathbf{a}^{(l)} = f\!\left(\mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right), \qquad \mathbf{a}^{(0)}=\mathbf{x}$$

$$\text{ReLU}(z)=\max(0,z)$$

The output layer uses softmax over the $K$ quality classes:

$$\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

### Loss: Cross-Entropy with L2 Regularization

$$J = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_{ik}\log\hat{y}_{ik} \;+\; \frac{\alpha}{2m}\sum_{l}\lVert\mathbf{W}^{(l)}\rVert_F^2$$

with $\alpha = 0.001$.

### Backpropagation

The output error is $\boldsymbol\delta^{(L)} = \hat{\mathbf{y}} - \mathbf{y}$, propagated backwards:

$$\boldsymbol\delta^{(l)} = \left(\mathbf{W}^{(l+1)T}\boldsymbol\delta^{(l+1)}\right)\odot f'\!\left(\mathbf{z}^{(l)}\right), \qquad \frac{\partial J}{\partial \mathbf{W}^{(l)}} = \boldsymbol\delta^{(l)}\mathbf{a}^{(l-1)T} + \frac{\alpha}{m}\mathbf{W}^{(l)}$$

### Optimizer: Adam

Adam keeps running estimates of the gradient's first and second moments:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \qquad \theta_t = \theta_{t-1} - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

Defaults: $\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$; initial learning rate $\eta=0.001$.

### Early Stopping

10% of the training data is held out internally as a validation set. Training stops when the validation score fails to improve for 10 consecutive epochs, or when `max_iter=2000` is reached.

### Evaluation

The script prints a classification report with per-class metrics:

$$\text{Precision}=\frac{TP}{TP+FP} \quad \text{Recall}=\frac{TP}{TP+FN} \quad F_1 = \frac{2\cdot \text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$$

---

## 5. Activation Function Comparison — `activation_function.py`

**Goal:** measure how the hidden-layer activation function affects the same MLP (architecture `(100, 50)`, Adam, $\alpha=0.001$, early stopping) on the Wine Quality dataset.

### Activation Functions Compared

| Activation | Function $f(z)$ | Derivative $f'(z)$ | Range |
|---|---|---|---|
| **Logistic (sigmoid)** | $\dfrac{1}{1+e^{-z}}$ | $f(z)\,(1-f(z))$ | $(0,1)$ |
| **ReLU** | $\max(0,z)$ | $1$ if $z>0$, else $0$ | $[0,\infty)$ |
| **Tanh** | $\dfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$ | $1-f(z)^2$ | $(-1,1)$ |

- **Sigmoid** and **tanh** saturate for large $|z|$, so their gradients become very small. This is the vanishing gradient problem.
- **ReLU** keeps a gradient of 1 for positive inputs, so training is typically faster and deeper networks are easier to train. Neurons with $z<0$ output zero and can stop learning.

### Metrics Reported (per activation)

- Training and testing **accuracy**
- Training and testing **log loss** (cross-entropy):

$$\text{LogLoss} = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_{ik}\log \hat{p}_{ik}$$

- **Number of iterations** until convergence (`n_iter_`)

A large gap between training and testing loss indicates overfitting.

---

## 6. Hyperparameter Tuning — `hyperparameter_tuning.py`

**Goal:** study how learning rate, network depth/width, and batch size affect MLP performance.

### Data Split (60 / 20 / 20)

$$\text{Train}: 60\% \qquad \text{Validation}: 20\% \qquad \text{Test}: 20\%$$

The data is first split 80/20 into train+validation and test, then the 80% part is split again 75/25 (stratified both times). The validation set is used to compare configurations; the test set is for the final estimate.

### Experimental Design

Each pair changes **one** hyperparameter and keeps the rest fixed (ReLU, Adam, `max_iter=1000`, no early stopping):

| Pair | Variable | Model 1 | Model 2 |
|---|---|---|---|
| 1 | **Learning rate** | 0.001 | 0.01 |
| 2 | **Architecture** | `(128,)` — shallow | `(256,128,64,32)` — deep |
| 3 | **Batch size** | 32 | 128 |

(Pair 1 uses `(128, 64)` with batch 64. Pair 3 uses `(128, 64)` with learning rate 0.001.)

### Effect of Each Hyperparameter

**Learning rate $\eta$** is the step size in the update $\theta \leftarrow \theta - \eta\,\hat m/(\sqrt{\hat v}+\epsilon)$. A large $\eta$ converges faster but can overshoot or oscillate; a small $\eta$ is stable but slow.

**Architecture** controls model capacity. A deeper and wider network can model more complex patterns but is more prone to overfitting on a small dataset. The parameter count is:

$$\#\text{params} = \sum_{l=1}^{L}\left(n_{l-1}\cdot n_l + n_l\right)$$

**Batch size $B$** sets how many samples are used per gradient estimate (mini-batch gradient):

$$g = \frac{1}{B}\sum_{i\in\text{batch}}\nabla_\theta \mathcal{L}_i(\theta)$$

Small batches give noisier gradients, which can act as regularization and need more updates per epoch. Large batches give smoother gradients and faster epochs.

### Output

A table showing training, validation, and test accuracy for all six models. The gap between training and validation accuracy indicates overfitting.

---

## 7. K-Means and Hierarchical Clustering — `k_means_n_clustering.py`

**Goal:** segment mall customers into 5 groups using `Annual Income (k$)` and `Spending Score (1-100)`. Both features are standardized first.

### 7.1 K-Means (from scratch)

**Objective:** minimize the within-cluster sum of squares (inertia):

$$J = \sum_{j=1}^{k}\sum_{\mathbf{x}_i \in C_j}\lVert \mathbf{x}_i - \boldsymbol\mu_j\rVert^2$$

**Algorithm (Lloyd's algorithm):**

1. **Initialize** $k=5$ centroids by randomly picking 5 distinct data points (seed 42).
2. **Assignment step:** assign each point to its nearest centroid using Euclidean distance:

$$c_i = \arg\min_{j}\ \lVert \mathbf{x}_i - \boldsymbol\mu_j\rVert_2 = \arg\min_j \sqrt{\sum_{d}(x_{id}-\mu_{jd})^2}$$

3. **Update step:** move each centroid to the mean of its cluster:

$$\boldsymbol\mu_j = \frac{1}{|C_j|}\sum_{\mathbf{x}_i \in C_j}\mathbf{x}_i$$

   An empty cluster keeps its previous centroid.
4. **Repeat** steps 2–3 until the centroids stop changing (`np.allclose`) or 100 iterations are reached.

Each iteration lowers $J$, so the algorithm always converges, but possibly to a local minimum that depends on initialization.

### 7.2 Hierarchical (Agglomerative) Clustering — Ward linkage

**Algorithm (bottom-up):**

1. Start with every point as its own cluster.
2. Repeatedly merge the pair of clusters whose merge **increases total within-cluster variance the least**.
3. Stop when 5 clusters remain.

**Ward's merging cost** for clusters $A$ and $B$:

$$\Delta(A,B) = \frac{|A|\,|B|}{|A|+|B|}\,\lVert \boldsymbol\mu_A - \boldsymbol\mu_B\rVert^2$$

### 7.3 Dendrogram

`scipy.cluster.hierarchy.linkage` records the full merge history. `dendrogram(..., truncate_mode="lastp", p=20)` shows only the last 20 merges. The vertical axis is the Ward distance at which clusters merged; a large jump suggests a good place to cut the tree.

### Output

Scatter plots for both methods (K-Means includes centroids marked with ✕), the dendrogram, and a table of the first 10 customers with their K-Means and hierarchical labels.

---

## 8. DBSCAN — `dbscan_algo.py`

**Goal:** density-based clustering of the 8×8 handwritten digits dataset (1797 samples, 64 pixel features) and comparison with the true digit labels.

### Preprocessing

1. Standardize all 64 features.
2. **PCA to 2 components for visualization only.** DBSCAN itself runs on the full 64-dimensional standardized data.

PCA finds the directions of maximum variance from the covariance matrix $\mathbf{C} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X}$ (via its eigenvectors $\mathbf{C}\mathbf{v}=\lambda\mathbf{v}$) and projects the data onto the top two.

### DBSCAN Definitions

Two parameters: radius $\varepsilon$ (`eps`) and `min_samples` ($=5$).

- **Neighborhood:** $N_\varepsilon(p) = \{\,q \in D \mid \lVert p-q\rVert_2 \le \varepsilon\,\}$
- **Core point:** $|N_\varepsilon(p)| \ge \text{min\_samples}$
- **Border point:** not a core point, but within $\varepsilon$ of a core point
- **Noise point:** neither core nor border (label $-1$)
- **Density-reachable:** $q$ is density-reachable from $p$ if there is a chain of core points $p=p_1,\dots,p_n=q$, each within $\varepsilon$ of the next

### Algorithm

1. For each unvisited point, find its $\varepsilon$-neighborhood.
2. If it is not a core point, tentatively mark it as noise.
3. If it is a core point, start a new cluster and expand it by repeatedly adding all density-reachable points.
4. Repeat until every point is visited.

DBSCAN does not need the number of clusters in advance, can find arbitrarily shaped clusters, and flags outliers as noise.

### Parameter Sweep

The script tests `eps` $\in \{5, 7, 9, 11, 13\}$. Because the data is 64-dimensional and standardized, typical distances are large, which is why these `eps` values are much bigger than in 2-D examples. Small `eps` produces many clusters and lots of noise; large `eps` merges everything into one cluster.

### Evaluation: Adjusted Rand Index (ARI)

ARI compares the discovered clusters with the true labels, corrected for chance. With contingency table entries $n_{ij}$, row sums $a_i$, column sums $b_j$, and $n$ samples:

$$\text{ARI} = \frac{\displaystyle\sum_{ij}\binom{n_{ij}}{2} - \frac{\left[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\right]}{\binom{n}{2}}}{\displaystyle\frac{1}{2}\left[\sum_i\binom{a_i}{2}+\sum_j\binom{b_j}{2}\right] - \frac{\left[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\right]}{\binom{n}{2}}}$$

ARI is 1 for a perfect match, about 0 for random labelling, and can be negative.

### Output

For each `eps`, a PCA scatter plot colored by cluster, plus a summary table of `eps`, number of clusters, number of noise points, and ARI.

---

## Datasets

| File | Description | Used by |
|---|---|---|
| `california_housing.csv` | Housing blocks with median income, house age, location, and median house value | Linear Regression |
| `auto-mpg.csv` | Car attributes (horsepower, weight, cylinders, ...) and fuel efficiency | Polynomial Regression |
| `breast-cancer.csv` | 30 tumor measurements with diagnosis (M/B) | Logistic Regression |
| `WineQT.csv` | 11 physicochemical wine properties with a quality score | MLP, Activation Functions, Hyperparameter Tuning |
| `Mall_Customers.csv` | Customer age, gender, annual income, and spending score | K-Means, Hierarchical Clustering |

---

## Algorithm Summary

| Task | Algorithm | Core Idea |
|---|---|---|
| Regression | Linear / Polynomial | Minimize squared error (closed-form solution) |
| Classification | Logistic Regression | Sigmoid probability + gradient descent on cross-entropy |
| Classification | MLP | Stacked non-linear layers trained by backpropagation + Adam |
| Clustering | K-Means | Alternate assignment and centroid update to minimize inertia |
| Clustering | Hierarchical (Ward) | Merge clusters with the smallest variance increase |
| Clustering | DBSCAN | Grow clusters from density-connected core points |