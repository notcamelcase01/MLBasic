import numpy as np
import pandas as pd

df = pd.read_csv("real_estate.csv")

data_set = df.to_numpy()
x = data_set[:, 1:7]  # 1 to 6th column #features are stored here
y = data_set[:, 7]
print(x.shape)
print(y.shape)
'''
Currently Y is (n,) which is neither a row or column vector, we need to reshape
'''
y = y.reshape(y.shape[0], 1)
'''
axis=1 means insert as row
'''
x = np.insert(x, 0, np.ones(y.transpose().shape), axis=1)
print(x)
print(y.shape)