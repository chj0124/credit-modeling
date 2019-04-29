# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:04:59 2019

@author: Administrator
"""
import pandas as pd
import numpy as np

# 数据标准化
def transf_standard(dataframe):
    """
    对数据框格式的数据进行标准化

    参数：
    dataframe 需标准化的数据框
    ---------------------------
    example:
    >>> X_standard = transf_standard(X)
    """

    df = dataframe.copy()
    for ColName in df.columns:
        MinValue = min(df[ColName])
        MaxValue = max(df[ColName])
        df[ColName] = df[ColName].apply(lambda x: (x - MinValue)/(MaxValue-MinValue))
    return df

# 激活函数
def sigmoid_active_fun(x, w):
    sigmoid = 1/(1 + np.exp(-np.dot(x, w)))
    return sigmoid

def linear_active_fun(x, w):
    active = np.dot(x, w)
    return active

# 损失函数
def sigmoid_loss(y, active):
    loss = -np.dot(y.T, np.log(active+1e-5)) - np.dot((1-y).T, np.log(1-active+1e-5))
    return loss

def mse_loss(m, active, y):
    loss = 1 / (2 * m) * np.dot((active - y).T, (active - y))
    return loss

# 求梯度
def sigmoid_gradient(x, y, active):
    gradient = np.dot(x.T, (active - y))
    return gradient

def linear_gradient(x, y, active, m):
    gradient = (1/m)*np.dot(x.T, active - y)
    return gradient

# 梯度下降求解逻辑回归
def logistic_gradient_regression(a, w, x, y):
    n = 0
    h = sigmoid_active_fun(x, w)
    g = sigmoid_gradient(x, y, h)
    #未知损失函数最小值可以达到多少，因此因限制斜率接近于平稳
    while np.all(np.absolute(g) >= 1e-5):
        w = w - a*g
        h = sigmoid_active_fun(x, w)
        g = sigmoid_gradient(x, y, h)
        j = sigmoid_loss(y, h)
        n = n + 1
    print("损失函数：" + str(j))
    print("迭代次数：" + str(n))
#    print("参数：" + str(p))
#    print("梯度：" + str(g))

# 梯度下降求解线性回归
def linear_gradient_regression(a, w, x, y, m):
    active = linear_active_fun(x, w)
    g = linear_gradient(x, y, active, m)
    n = 0
    while np.all(np.absolute(g) >= 1e-5):
        w = w - a * g
        active = linear_active_fun(x, w)
        g = linear_gradient(x, y, active, m)
        mse = mse_loss(m, active, y)
        n = n + 1
    else:
            print("参数：" + str(w))
            print("梯度：" + str(g))
            print("均方误差：" + str(mse))
            print("迭代次数：" + str(n))
    return

# 梯度下降求解逻辑回归
def logistic_gradient_regression(a, w, x, y):
    n = 0
    h = sigmoid_active_fun(x, w)
    g = sigmoid_gradient(x, y, h)
    #未知损失函数最小值可以达到多少，因此因限制斜率接近于平稳
    while np.all(np.absolute(g) >= 1e-5):
        w = w - a*g
        h = sigmoid_active_fun(x, w)
        g = sigmoid_gradient(x, y, h)
        j = sigmoid_loss(y, h)
        n = n + 1
    print("损失函数：" + str(j))
    print("迭代次数：" + str(n))
#    print("参数：" + str(p))
#    print("梯度：" + str(g))
