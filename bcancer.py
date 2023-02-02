import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import optimizerfn as op

df = pd.read_csv("dat/Breast_cancer_data.csv")
print(df.info())
print(df.head(100))
dataset = df.to_numpy()
x = dataset[:, 0:5]
m_tot = x.shape[0]
y = dataset[:, 5].reshape(m_tot, 1)
x = np.insert(x, 0, np.ones(1, m_tot), axis=1)
