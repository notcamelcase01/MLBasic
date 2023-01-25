import numpy as np
import pandas as pd
import optimizerfn as op

df = pd.read_csv("real_estate.csv")

data_set = df.to_numpy()
x = data_set[:, 1:7]  # 1 to 6th column #features are stored here
y = data_set[:, 7]
'''
Currently Y is (n,) which is neither a row or column vector, we need to reshape
'''
y = y.reshape(y.shape[0], 1)
'''
axis=1 means insert as row, we inserted x_0 column
'''
x = np.insert(x, 0, np.ones(y.transpose().shape), axis=1)
theta = np.zeros((7, 1))
# print(op.cost(x, y, theta))
# print(op.gradient(x, y, theta))
print("Before:", theta)
N = 300
alpha = 1e-8
prev_cost = op.cost(x, y, theta)
for i in range(N):
    print(i, prev_cost)
    theta = theta - alpha * op.gradient(x, y, theta)
    cs = op.cost(x, y, theta)
    if abs(prev_cost - cs) < 1e-4:
        print(i)
        break
    prev_cost = cs

print("After: ", theta)
theta_correct = np.linalg.inv(x.T @ x) @ x.T @ y
print(theta_correct)
print("Correct cost:", op.cost(x, y, theta_correct))
