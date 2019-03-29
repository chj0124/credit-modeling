# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:15:52 2019

@author: sy784
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# data = pd.read_excel(r'.\data.xlsx')


def per_gradient(X, Y, lrate):
    w = np.matrix([0.8, -0.5])
    b = 0
    length = len(X)
    for i in range(length):
        if (Y[i] * (w * np.matrix(X[i]).T + b) <= 0):
            w = w + lrate * Y[i] * np.matrix(X[i])
            b = b + lrate * Y[i]
        else:
            pass
    return w, b


X = [(0.3, 0.7), (-0.6, 0.3), (-0.1, -0.8), (0.1, -0.45)]
Y = [1, -1, -1, 1]
lrate = 0.7
# 使用梯度下降求解划分平面参数w,b
w, b = per_gradient(X, Y, lrate)
print(w, b)
# 绘制划分超平面
linex = np.linspace(-1, 1, 10)
liney = -(w[0, 0] / w[0, 1]) * linex - (1 / w[0, 1]) * b
plt.plot(linex, liney, color='r')
# 绘制数据点
X = np.array(X)
Y = np.array(Y)
for i in range(0, len(X)):
    if Y[i] == 1:
        # 正例用圆点表示
        plt.scatter(X[i][0], X[i][1], marker="o", alpha=0.8, s=50)
    else:
        # 负例用×表示
        plt.scatter(X[i][0], X[i][1], marker="x", alpha=0.8, s=50)
# 显示图片
plt.show()
