# -*- coding: utf-8 -*-
"""
用梯度下降的方法，重新陷入线性回归噩梦。
"""

# %% 函数f(x) = x**2，初始点P为(10,100)，用梯度下降的方式使P点下降到函数最小值处
import numpy as np
import matplotlib.pyplot as plt

plot_x = np.linspace(-10, 10, 141)  # 在-1到6之间等距的生成141个数
plot_y = plot_x ** 2

plt.plot(plot_x, plot_y)
plt.show()


# 定义一个求二次函数导数的函数dJ
def dJ(x):
    return 2 * x


# 定义一个求函数值的函数J
def J(x):
    try:
        return x ** 2
    except:
        return float('inf')


x = 10  # 初始点P
eta = 0.1  # 学习率
epsilon = 1e-8  # 用来判断是否到达二次函数的最小值点的条件
history_x = [x]  # 用来记录使用梯度下降法走过的点的X坐标

while True:
    gradient = dJ(x)  # 梯度
    last_x = x
    x = x - eta * gradient
    history_x.append(x)
    if (abs(J(last_x) - J(x)) < epsilon):  # 用来判断是否逼近最低点
        break

plt.plot(plot_x, plot_y)
plt.plot(np.array(history_x), J(np.array(history_x)), color='r', marker='*')  # 绘制x的历史轨迹

print("{0} epoch, error = {1}".format(len(history_x), history_x[-1:]))

# %% 波士顿房价数据集 - 线性回归（梯度下降）
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn import preprocessing

# 这里得到的是 scikit-learn 中的 Bunch 对象
boston = load_boston()

# 对这个数据集的文字描述
print(boston.DESCR)

# 特征集与目标变量
x = pd.DataFrame(data=boston.data, columns=boston.feature_names)
y = boston.target
x = pd.DataFrame(preprocessing.scale(x))
x['bias'] = 1

x.shape
m = 506

alpha = 0.01  # learning_rate赋值
theta = np.random.uniform(low=-1, high=1, size=14)
history_theta = []
history_error = []


# 定义误差函数
def error_function(theta, x, y):
    diff = np.dot(x, theta) - y
    return (1 / (2 * m)) * np.dot(np.transpose(diff), diff)


# 定义梯度
def gradient_function(theta, x, y):
    diff = np.dot(x, theta) - y
    return (1 / m) * np.dot(np.transpose(x), diff)


# 梯度下降
def gradient_descent(x, y, alpha):
    theta = np.random.uniform(low=-1.0, high=1.0, size=14)
    gradient = gradient_function(theta, x, y)
    error = error_function(theta, x, y)
    history_theta.append(theta)
    history_error.append(error)
    while not (np.sqrt(np.dot(np.transpose(gradient), gradient)) <= 1e-4):
        theta = theta - alpha * gradient
        history_theta.append(theta)
        history_error.append(error)
        gradient = gradient_function(theta, x, y)
        error = error_function(theta, x, y)
    return theta


optimal = gradient_descent(x, y, alpha)
error_function(optimal, x, y)
