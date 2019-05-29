import os
import subprocess
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

os.chdir("/home/gabor/PycharmProjects/linear_regression/")

subprocess.call(["ls", "-l"])


data = pd.read_csv("./x06.txt", index_col=0)
print(data.head())

X = data.iloc[:, 0:2].values
y = data.iloc[:, -1].values
print(y, np.mean(y), np.sum(y - np.mean(y)))
lmbda = 0
xtx = np.dot(X.T, X)

# beta = np.dot(np.matmul(np.linalg.inv(xtx + lmbda*np.eye(xtx.shape[0])), X.T), y)
#
# TSS = np.sum(y - np.mean(y))
# RSS = np.dot((y - np.dot(X, beta)).T, (y - np.dot(X, beta)))
# R_squared = TSS / RSS
#
# print(TSS, RSS, R_squared)