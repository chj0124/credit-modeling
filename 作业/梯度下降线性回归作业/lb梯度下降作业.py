# 1. 用梯度下降法，求y = x**2 的极小值的数值解
import numpy as np

theta_intial = 10.0
epsilon = 1e-8
eta = 0.01
theta = theta_intial
while True:

    gradient = 2 * theta
    theta_last = theta
    theta = theta - gradient * eta

    if (abs((theta_last) ** 2 - (theta) ** 2)) < epsilon:
        break
print(theta)
print(theta ** 2)

# 2 Python: 写一个线性回归对象,求参数a,b

# 创建一个线性回归方程 y = ax + b
np.random.seed(666)
x_1 = 2 * np.random.random(size=100)
y_1 = 2 * x_1 + 3. + np.random.random(size=100)
import matplotlib.pyplot as plt

X_1 = x_1.reshape(-1, 1)
plt.scatter(X_1, y_1)
plt.show()


def J(theta, x_b, y):
    return np.sum((y - x_b.dot(theta)) ** 2) / len(y)


def dJ(theta, x_b, y):
    return x_b.T.dot(x_b.dot(theta) - y) * 2. / len(x_b)


def gradient_descent(x_b, y, theta_intial, eta, epsilon, n_iters):
    theta = theta_intial
    i_iter = 0
    cost = []
    while i_iter < n_iters:
        gradient = dJ(theta, x_b, y)
        theta_last = theta
        theta = theta - eta * gradient

        if (abs(J(theta, x_b, y) - J(theta_last, x_b, y))) < epsilon:
            break
        i_iter += 1
        cost.append(J(theta, x_b, y))

    return theta, cost


x_b_1 = np.hstack([np.ones((len(X_1), 1)), X_1])
x_b_1.shape
y_1.shape
theta_intial = np.zeros(x_b_1.shape[1])
eta = 0.1
epsilon = 1E-8
n_iters = 10000
a1, b1 = gradient_descent(x_b_1, y_1, theta_intial, eta, epsilon, n_iters)
print(a1)

# 3 波士顿房价数据集预测
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

boston = datasets.load_boston()

X = pd.DataFrame(data=boston.data, columns=boston.feature_names)
y = boston.target
X = X[y < 50]
y = y[y < 50]
print(boston.DESCR)

from sklearn.preprocessing import StandardScaler

Standar = StandardScaler()
Standar.fit(X)
X_standar = Standar.transform(X)

x_b = np.hstack([np.ones((len(X), 1)), X_standar])
x_b.shape
y.shape
theta_intial = np.zeros(x_b.shape[1])
eta = 0.1
epsilon = 1E-8
n_iters = 10000
a, b = gradient_descent(x_b, y, theta_intial, eta, epsilon, n_iters)
# 误差曲线绘制
plt.plot(np.arange(len(b)), b, 'r')

# 求导
