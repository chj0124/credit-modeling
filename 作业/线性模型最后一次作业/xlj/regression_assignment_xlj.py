# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:37:53 2019

@author: Administrator
"""
import pandas as pd
import numpy as np

import os
os.chdir('E:\\理论资料代码\\线性回归')

import regression as rg


#%%
############################### 线性回归 波士顿房价
from sklearn.datasets import load_boston

### 这里得到的是 scikit-learn 中的 Bunch 对象
boston = load_boston()

### 特征集与目标变量
X = pd.DataFrame(data=boston.data, columns=boston.feature_names)
Y = boston.target

### 自变量标准化
X_standard = rg.transf_standard(X)
m = len(X_standard.index)
sample = np.array(X_standard)
b = np.ones((m, 1))  # 设置值全为1的矩阵
x = np.hstack((b, sample))  # 矩阵拼接：行拼接vstack, 列拼接hstack

y = np.array(Y).reshape(m, 1)

a = 0.01

p = np.ones((len(X_standard.columns)+1, 1))

rg.linear_gradient_regression(a, p, x, y, m)


#%%
############################## 逻辑回归 肺癌
from sklearn.datasets import load_breast_cancer

# cancer 是个类
cancer = load_breast_cancer()

# 取 xy
x = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
y = cancer.target

# x 标准化并转换成矩阵
x_standard = rg.transf_standard(x)
sample = np.array(x_standard)

m = len(sample)

# y 转换成矩阵
y = y.reshape(m, 1)

b = np.ones((m, 1))  # 设置值全为 1 的矩阵
X = np.hstack((b, sample))  # 矩阵拼接：行拼接 vstack, 列拼接 hstack

# 系数初始值设定（求的就是这个）
p = np.ones((len(x.columns)+1, 1))

rg.logistic_gradient_regression(0.1, p, X, y)
