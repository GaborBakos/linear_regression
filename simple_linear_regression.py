import os
import subprocess
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

os.chdir("/home/gabor/PycharmProjects/linear_regression/")

subprocess.call(["ls", "-l"])


data = pd.read_csv("./x01.txt", index_col=0)
print(data.head())

y_train = data.iloc[:-10, 0]
x_train = data.iloc[:-10, 1]
y_test = data.iloc[-10:, 0]
x_test = data.iloc[-10:, 1]


# y_train = beta_0 + beta_1 * x_train + error
beta_1 = np.sum((x_train - np.mean(x_train)) * (y_train - np.mean(y_train))) / np.sum((x_train - np.mean(x_train))**2)
beta_0 = np.mean(y_train) - beta_1 * np.mean(x_train)

print(beta_0, beta_1)


fig, ax = plt.subplots(1, 2)
ax[0].scatter(x_train, y_train)
x_vals = np.array(ax[0].get_xlim())
y_vals = beta_0 + beta_1 * x_vals
ax[0].plot(x_vals, y_vals)

y = data.iloc[:, 0]
x = data.iloc[:, 1]

ax[1].scatter(x, y)
x_vals = np.array(ax[1].get_xlim())
y_vals = beta_0 + beta_1 * x_vals
ax[1].plot(x_vals, y_vals)

plt.show()
