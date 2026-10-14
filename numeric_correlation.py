import pandas as pd
import numpy as np
df = pd.read_csv('numeric.csv')
hours = np.array(df.hours_of_study)
marks = np.array(df.Marks)
h_mean = np.mean(hours)
m_mean = np.mean(marks)
m_hours = hours - h_mean
m_marks = marks - m_mean
shxm = np.sum(m_hours * m_marks)
cf = shxm/len(marks)
print("Correlation Coefficient:", round(cf, 2))
den = np.sqrt(sum(m_hours**2) * sum(m_marks**2))
print("Correlation:", round(shxm/den, 2))
std_hours = np.std(hours)
std_marks = np.std(marks)
print("Correlation (using std):", round(cf/(std_hours*std_marks),2))
