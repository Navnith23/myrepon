import pandas as pd
import numpy as np
from scipy.stats import chi2

df = pd.read_csv('nominal.csv')
col1 = 'Department'
col2 = 'Preferred_Language'
observed = pd.crosstab(df[col1], df[col2])

print("Observed Frequencies:\n", observed)
n = observed.to_numpy().sum()

print("Total Observations:", n)
r_total = observed.sum(axis = 1).to_numpy()
c_total = observed.sum(axis = 0).to_numpy()
expected = np.outer(r_total,c_total)/n

print("Expected Frequencies:\n")
print(pd.DataFrame(expected,index = observed.index , columns = observed.columns).round(2))
O = observed.to_numpy()
chi_sq = np.sum((O - expected)**2 / expected)

print("Chi-Square Statistic:", round(chi_sq, 2))
alpha = 0.05
rows, cols = observed.shape
dof = (rows - 1) * (cols - 1)

print("Degrees of Freedom:", dof)

critical_value = chi2.ppf(1-alpha,dof)
print("Critical Value:", round(critical_value, 2))

cramers_v = np.sqrt(chi_sq / n * min(rows - 1,cols - 1))
print("Cramer's V:", round(cramers_v, 4))

if chi_sq > critical_value:
    print("Reject the null hypothesis: There is a significant association between", col1, "and", col2)
else:
    print("Fail to reject the null hypothesis: There is no significant association between", col1, "and", col2)

