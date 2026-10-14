# Data Mining and Machine Learning Algorithms

A collection of Python implementations of fundamental Data Mining and Machine Learning algorithms. This repository demonstrates how common algorithms work internally using simple datasets, mathematical formulas, and basic Python libraries.

## Table of Contents

1. [Requirements](#requirements)
2. [Information Gain](#1-information-gain)
3. [Naive Bayes Classifier](#2-naive-bayes-classifier)
4. [K-Means Clustering](#3-k-means-clustering)
5. [Apriori Algorithm](#4-apriori-algorithm)
6. [Cosine and Jaccard Similarity](#5-cosine-and-jaccard-similarity)
7. [Nominal Attribute Correlation](#6-nominal-attribute-correlation)
8. [Numerical Correlation](#7-numerical-correlation)
9. [Outlier Detection Using IQR](#8-outlier-detection-using-iqr)
10. [Similarity and Dissimilarity for Mixed Data](#9-similarity-and-dissimilarity-for-mixed-data)
11. [Single-Linkage Hierarchical Clustering](#10-single-linkage-hierarchical-clustering)
12. [Datasets](#datasets)
13. [Running the Programs](#running-the-programs)

## Requirements

Python 3.10 or later is recommended.

Install the required libraries:

```bash
pip install pandas numpy scipy
```

The programs use:

- **pandas** — reading CSV files, filtering rows, and processing tabular data.
- **NumPy** — numerical calculations and matrix operations.
- **SciPy** — chi-square distribution and critical values.
- **math** — mathematical functions such as square root and logarithms.
- **itertools** — generating combinations for association rule mining.
- **collections.Counter** — counting word frequencies.

---

## 1. Information Gain

**File:** `information_gain.py`

### Objective

Calculate the information gain of each attribute in a dataset and identify the best attribute for splitting a decision tree.

The program uses the `data.csv` dataset and predicts the target attribute `Buys_Computer`.

### Algorithm: Entropy and Information Gain

Entropy measures the uncertainty or impurity in a dataset. Information gain measures how much that uncertainty decreases after splitting the dataset using an attribute.

### Mathematical formulas

Entropy:

\[
H(S)=-\sum_{i=1}^{c}p_i\log_2(p_i)
\]

Where:

- \(S\) is the dataset.
- \(c\) is the number of target classes.
- \(p_i\) is the proportion of records belonging to class \(i\).

Weighted entropy after splitting on attribute \(A\):

\[
H(S\mid A)=\sum_{v\in Values(A)}
\frac{|S_v|}{|S|}H(S_v)
\]

Information gain:

\[
IG(S,A)=H(S)-H(S\mid A)
\]

### Algorithm steps

1. Read `data.csv` using pandas.
2. Define a function to calculate entropy using class frequencies.
3. Calculate the entropy of the complete target column.
4. Iterate through every input attribute.
5. Create a subset for each unique attribute value.
6. Calculate the entropy of each subset.
7. Compute the weighted entropy of the split.
8. Subtract weighted entropy from total entropy to obtain information gain.
9. Identify the attribute with the maximum information gain.

### Important implementation detail

The entropy function must use `len(data)` rather than `len(df)` so that entropy is calculated using the size of the particular dataset or subset passed to the function.

The information gain calculation should occur after the inner loop has processed every unique value of an attribute.

### Applications

- Decision tree construction.
- Feature selection.
- Identifying attributes that best separate target classes.

---

## 2. Naive Bayes Classifier

**File:** `naive_bayes10.py`

### Objective

Predict whether a person buys a computer using attributes such as age, income, student status, and credit rating.

The program uses `data.csv` and predicts the class for a new record.

### Algorithm: Naive Bayes

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes that input attributes are conditionally independent given the target class.

Bayes' theorem:

\[
P(C\mid X)=\frac{P(X\mid C)P(C)}{P(X)}
\]

For classification, the denominator \(P(X)\) is the same for all classes. Therefore, the class can be predicted using:

\[
\hat C=\arg\max_C P(C)\prod_{i=1}^{n}P(X_i\mid C)
\]

Where:

- \(C\) is a target class.
- \(P(C)\) is the prior probability of the class.
- \(P(X_i\mid C)\) is the conditional probability of an attribute value given the class.

### Algorithm steps

1. Load the dataset.
2. Separate the input attributes from the target column.
3. Define the new record to classify.
4. Calculate the prior probability of each target class.
5. Calculate the conditional probability of each attribute value within that class.
6. Multiply the prior and conditional probabilities.
7. Compare the resulting scores for all classes.
8. Predict the class with the highest score.

### Applications

- Email spam classification.
- Document classification.
- Basic medical decision-support systems.
- Probabilistic prediction on categorical datasets.

### Limitation

The implementation does not use Laplace smoothing. If an attribute value never occurs in a class, its conditional probability becomes zero and the entire class score becomes zero.

---

## 3. K-Means Clustering

**File:** `k_means.py`

### Objective

Group two-dimensional data points into two clusters based on their distance from cluster centroids.

The dataset consists of six predefined points, and the initial centroids are `(1, 1)` and `(2, 3)`.

### Algorithm: K-Means

K-Means is an unsupervised learning algorithm that partitions data into \(K\) clusters by repeatedly assigning points to their nearest centroid and updating the centroids.

Euclidean distance:

\[
d(x,\mu)=\sqrt{\sum_{j=1}^{n}(x_j-\mu_j)^2}
\]

Centroid update:

\[
\mu_k=\frac{1}{|C_k|}\sum_{x\in C_k}x
\]

Where:

- \(x\) is a data point.
- \(\mu_k\) is the centroid of cluster \(k\).
- \(C_k\) is the set of points assigned to cluster \(k\).

### Algorithm steps

1. Define the data points.
2. Select the initial centroids.
3. Calculate each point's distance from both centroids.
4. Assign each point to the nearest centroid.
5. Recalculate each centroid using the mean of its assigned points.
6. Repeat the assignment and update steps until the centroids stop changing.
7. Print the clusters and updated centroids.

### Applications

- Customer segmentation.
- Grouping similar records.
- Image segmentation.
- Exploratory data analysis.

### Limitation

The number of clusters and initial centroids are predefined. The implementation also needs additional handling for empty clusters.

---

## 4. Apriori Algorithm

**File:** `apriori_algorithm.py`

### Objective

Discover frequent itemsets and generate association rules from a collection of transactions.

The program uses a predefined transaction dataset containing items such as `I1`, `I2`, `I3`, `I4`, and `I5`.

Minimum support count is `2`, and minimum confidence is `0.6`.

### Algorithm: Apriori

Apriori is an association rule mining algorithm that identifies frequently occurring item combinations.

It uses the **Apriori property**:

If an itemset is frequent, all its non-empty subsets must also be frequent.

Equivalently, if an itemset is infrequent, every larger itemset containing it must be infrequent.

### Mathematical formulas

Support count:

\[
\sigma(X)=|\{T:X\subseteq T\}|
\]

Support:

\[
Support(X)=\frac{\sigma(X)}{N}
\]

Confidence of a rule \(X\rightarrow Y\):

\[
Confidence(X\rightarrow Y)=
\frac{Support(X\cup Y)}{Support(X)}
\]

### Algorithm steps

1. Define the transaction database.
2. Count the occurrences of individual items.
3. Keep items satisfying the minimum support count.
4. Generate candidate itemsets by combining frequent itemsets.
5. Use the Apriori property to eliminate candidates with infrequent subsets.
6. Count candidate occurrences in the transactions.
7. Retain candidates satisfying minimum support.
8. Repeat until no new frequent itemsets are found.
9. Generate association rules from frequent itemsets.
10. Calculate each rule's confidence and print rules meeting the threshold.

### Applications

- Market basket analysis.
- Product recommendation.
- Identifying items frequently purchased together.
- Discovering relationships in transactional data.

### Limitation

The implementation uses minimum support **count**, not minimum support percentage. Its candidate generation and support counting are suitable for learning the algorithm but may be inefficient for large datasets.

---

## 5. Cosine and Jaccard Similarity

**File:** `cosine_similarity.py`

### Objective

Measure the similarity between two text documents using word frequencies and word-set overlap.

The program compares:

- Document 1: `I love programming in Python`
- Document 2: `Python programming is fun`

### Algorithm 1: Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors. Here, each document is represented as a vector of word counts.

\[
Cosine(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

Where:

- \(A\cdot B\) is the dot product.
- \(\|A\|\) and \(\|B\|\) are the vector magnitudes.

### Algorithm steps

1. Convert both documents to lowercase.
2. Split each document into words.
3. Create a vocabulary containing the unique words from both documents.
4. Count word frequencies using `Counter`.
5. Represent each document as a frequency vector using the shared vocabulary.
6. Calculate the dot product of the vectors.
7. Calculate their magnitudes.
8. Divide the dot product by the product of the magnitudes.

### Algorithm 2: Jaccard Similarity

Jaccard similarity measures the overlap between two sets.

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

Where:

- \(A\cap B\) contains words common to both documents.
- \(A\cup B\) contains all unique words across both documents.

### Algorithm steps

1. Convert the word lists into sets.
2. Find the intersection and union.
3. Divide the size of the intersection by the size of the union.

### Difference between the measures

- **Cosine similarity** uses word frequencies and vector direction.
- **Jaccard similarity** uses unique word overlap and ignores repeated occurrences.

### Applications

- Document similarity.
- Search engines.
- Text clustering.
- Duplicate document detection.
- Recommendation systems.

---

## 6. Nominal Attribute Correlation

**File:** `nominal_correlation.py`

### Objective

Determine whether two categorical attributes are statistically associated.

The program uses `nominal.csv` and compares `Department` with `Preferred_Language`.

### Algorithm: Chi-Square Test of Independence

The chi-square test determines whether the observed frequencies of two categorical variables differ significantly from the frequencies expected if the variables were independent.

### Mathematical formulas

Expected frequency:

\[
E_{ij}=\frac{R_iC_j}{N}
\]

Chi-square statistic:

\[
\chi^2=\sum_i\sum_j
\frac{(O_{ij}-E_{ij})^2}{E_{ij}}
\]

Degrees of freedom:

\[
df=(r-1)(c-1)
\]

Where:

- \(O_{ij}\) is an observed frequency.
- \(E_{ij}\) is an expected frequency.
- \(R_i\) is a row total.
- \(C_j\) is a column total.
- \(N\) is the total number of observations.
- \(r,c\) are the table's row and column counts.

### Algorithm steps

1. Load `nominal.csv`.
2. Construct a contingency table using `pandas.crosstab()`.
3. Calculate row totals, column totals, and the total number of observations.
4. Calculate expected frequencies using the independence assumption.
5. Compute the chi-square statistic.
6. Calculate the degrees of freedom.
7. Find the critical value for significance level \(\alpha=0.05\).
8. Compare the chi-square statistic with the critical value.
9. Reject or fail to reject the null hypothesis.
10. Calculate Cramér's V to measure the strength of association.

### Cramér's V

\[
V=\sqrt{\frac{\chi^2}{N\min(r-1,c-1)}}
\]

Cramér's V ranges from zero to one for a valid non-degenerate contingency table. Higher values indicate stronger association.

### Applications

- Analyzing relationships between categorical variables.
- Survey analysis.
- Comparing departmental preferences.
- Exploring categorical datasets.

### Limitation

The test relies on appropriate expected frequencies and independent observations. Statistical significance does not establish causation.

---

## 7. Numerical Correlation

**File:** `numeric_correlation.py`

### Objective

Measure the relationship between hours of study and marks using numerical data from `numeric.csv`.

### Algorithm: Pearson Correlation Coefficient

Pearson correlation measures the strength and direction of a linear relationship between two numerical variables.

The coefficient is:

\[
r=\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2
\sum_{i=1}^{n}(y_i-\bar{y})^2}}
\]

Where:

- \(x_i\) represents hours of study.
- \(y_i\) represents marks.
- \(\bar{x},\bar{y}\) are their means.
- \(n\) is the number of observations.

### Algorithm steps

1. Read the dataset.
2. Extract `hours_of_study` and `Marks` as NumPy arrays.
3. Calculate their means.
4. Subtract the means from individual values.
5. Calculate the sum of the products of deviations.
6. Calculate the sum of squared deviations for each variable.
7. Divide the cross-product sum by the square root of the product of squared-deviation sums.
8. Print the correlation coefficient.

The program also computes the coefficient using population covariance and population standard deviations. These approaches produce the same Pearson correlation when calculated consistently.

### Interpretation

- \(r\approx +1\): strong positive linear relationship.
- \(r\approx -1\): strong negative linear relationship.
- \(r\approx 0\): little or no linear relationship.

### Applications

- Studying relationships between variables.
- Educational performance analysis.
- Exploratory data analysis.
- Identifying linear associations.

### Limitation

Correlation does not imply causation. Pearson correlation measures linear relationships and may be affected by outliers.

---

## 8. Outlier Detection Using IQR

**File:** `outliers11.py`

### Objective

Detect potential outliers in numerical columns of `data1.csv` using the Interquartile Range method.

### Algorithm: Interquartile Range (IQR)

IQR measures the spread of the middle 50% of a dataset. Values far below or above the central range are flagged as potential outliers.

### Mathematical formulas

Interquartile range:

\[
IQR=Q_3-Q_1
\]

Lower bound:

\[
L=Q_1-1.5(IQR)
\]

Upper bound:

\[
U=Q_3+1.5(IQR)
\]

Where:

- \(Q_1\) is the first quartile (25th percentile).
- \(Q_3\) is the third quartile (75th percentile).

A value is flagged as an outlier when:

\[
x<L\quad\text{or}\quad x>U
\]

### Algorithm steps

1. Load `data1.csv`.
2. Select numerical columns using `select_dtypes()`.
3. Calculate \(Q_1\) and \(Q_3\) for each numerical column.
4. Calculate the IQR.
5. Compute the lower and upper bounds.
6. Filter records containing values outside those bounds.
7. Print quartiles, bounds, and the number of flagged records.

### Applications

- Data cleaning.
- Identifying unusual observations.
- Exploratory data analysis.
- Detecting potential data-entry errors.

### Implementation note

The final print statement should use the current column variable `col`, not the undefined variable `column`. For example:

```python
print(outliers[col].values)
```

The current implementation counts rows containing an outlier in a given column. It does not necessarily count the total number of individual outlier values across all columns.

---

## 9. Similarity and Dissimilarity for Mixed Data

**File:** `similarity_dissimilarity.py`

### Objective

Calculate pairwise similarity and dissimilarity matrices for students using one nominal attribute (`Department`) and one numerical attribute (`Marks`).

The program uses `students.csv`.

### Algorithm

The program computes three types of comparison:

1. Nominal similarity and dissimilarity.
2. Numerical similarity and dissimilarity.
3. Combined similarity and dissimilarity for mixed attributes.

### A. Nominal dissimilarity

For two records \(i,j\):

\[
d_{nom}(i,j)=
\begin{cases}
0,&x_i=x_j\\
1,&x_i\ne x_j
\end{cases}
\]

Nominal similarity:

\[
s_{nom}(i,j)=1-d_{nom}(i,j)
\]

Two students from the same department have similarity 1 and dissimilarity 0 for that attribute.

### B. Numerical dissimilarity

The numerical dissimilarity is normalized by the range of the numerical attribute:

\[
d_{num}(i,j)=\frac{|x_i-x_j|}{x_{max}-x_{min}}
\]

Numerical similarity:

\[
s_{num}(i,j)=1-d_{num}(i,j)
\]

This normalization places numerical dissimilarities between zero and one when the attribute has a non-zero range.

### C. Mixed dissimilarity

The implementation gives equal weight to the nominal and numerical dissimilarities:

\[
d_{mixed}(i,j)=
\frac{d_{nom}(i,j)+d_{num}(i,j)}{2}
\]

Mixed similarity:

\[
s_{mixed}(i,j)=1-d_{mixed}(i,j)
\]

### Algorithm steps

1. Load `students.csv`.
2. Identify the nominal and numerical columns.
3. Create empty pairwise matrices.
4. Compare every pair of students.
5. Assign nominal dissimilarity according to whether their departments match.
6. Calculate normalized numerical dissimilarity using the difference in marks.
7. Convert individual dissimilarities into similarities.
8. Average the two dissimilarities to obtain mixed dissimilarity.
9. Print all similarity and dissimilarity matrices.

### Applications

- Comparing mixed-type records.
- Student or customer similarity analysis.
- Clustering datasets containing categorical and numerical features.
- Basic recommendation and retrieval systems.

### Limitation

The mixed measure assigns equal importance to both attributes. If the numerical range is zero, the implementation leaves numerical dissimilarities at zero. More sophisticated mixed-data measures may be needed for larger or more diverse datasets.

---

## 10. Single-Linkage Hierarchical Clustering

**File:** `single_linkage_clustering.py`

### Objective

Perform agglomerative hierarchical clustering on a set of one-dimensional numerical observations.

The data points are:

`18, 22, 25, 42, 27, 43`

### Algorithm: Single-Linkage Clustering

Single-linkage clustering is a bottom-up hierarchical clustering algorithm. It starts with every observation in its own cluster and repeatedly merges the two clusters with the smallest minimum pairwise distance.

Distance between two individual observations:

\[
d(a,b)=|a-b|
\]

Single-linkage distance between clusters \(A\) and \(B\):

\[
D(A,B)=\min_{a\in A,b\in B}|a-b|
\]

### Algorithm steps

1. Treat every data point as a separate cluster.
2. Initialize the minimum distance to infinity.
3. Examine every pair of clusters.
4. Calculate the distance between every pair of points belonging to those clusters.
5. Select the pair with the minimum distance.
6. Merge the two clusters.
7. Print the merged cluster and its merge distance.
8. Repeat until only one cluster remains.

### Applications

- Hierarchical data analysis.
- Grouping similar numerical observations.
- Building dendrograms.
- Exploratory clustering.

### Limitation

Single linkage can produce a chaining effect, where clusters grow through successive nearby points. This implementation prints merge operations but does not draw a dendrogram.

---

## Datasets

| File | Purpose |
|---|---|
| `data.csv` | Categorical computer-purchase dataset used for information gain and Naive Bayes |
| `data1.csv` | Dataset used for IQR-based outlier detection |
| `nominal.csv` | Categorical data used for chi-square testing and Cramér's V |
| `numeric.csv` | Numerical data used to calculate Pearson correlation |
| `students.csv` | Student records used for mixed-data similarity and dissimilarity |

Ensure that each CSV file is in the same directory as the Python script that reads it.

## Running the Programs

Clone the repository:

```bash
git clone -b da https://github.com/Navnith23/myrepon.git
cd myrepon
```

Install dependencies:

```bash
pip install pandas numpy scipy
```

Run an individual program:

```bash
python information_gain.py
```

Replace `information_gain.py` with any other Python filename to run a different algorithm.

**Note:** The programs are independent demonstrations. Each script should be executed separately.

## Learning Outcomes

This repository demonstrates the implementation and working principles of:

- Entropy and information gain for feature selection.
- Probabilistic classification using Naive Bayes.
- Centroid-based clustering using K-Means.
- Frequent itemset mining and association rule generation using Apriori.
- Cosine and Jaccard similarity for text.
- Chi-square testing and Cramér's V for categorical attributes.
- Pearson correlation for numerical variables.
- Statistical outlier detection using IQR.
- Similarity and dissimilarity measures for mixed data.
- Agglomerative clustering using the single-linkage criterion.

The focus is on understanding algorithmic logic and mathematical foundations through straightforward Python implementations rather than relying entirely on pre-built machine learning libraries.
