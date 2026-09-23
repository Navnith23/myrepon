import pandas as pd
data = pd.read_csv('data1.csv')
columns = data.select_dtypes(include = 'number').columns

for col in columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1   
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[col]<lower)|(data[col]>upper)]
    print("\nColumn:", col)
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Bound:", lower)
    print("Upper Bound:", upper)
    print("Number of outliers:", len(outliers))

    print(outliers[column].values)
