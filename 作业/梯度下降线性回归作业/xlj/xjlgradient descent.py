# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:23:56 2019

@author: xulingjie

梯度下降法求线性回归（采用均方误差）
"""

import numpy as np 

# 线性回归方程： y = b0 +b1*x1+b2*x2+...+bn*xn

# 梯度下降
def gradient_descent(p, m, sample, y, a):
    """
    使用梯度下降法求均方误差最小值，拟合线性回归函数
    
    参数：
    p--参数矩阵n*1(n-1元);
    m--样本数量;
    sample--样本矩阵m*(n-1);
    y--目标矩阵m*1;
    a--步长/学习率
    ---------------------------
    example:
    >>> gradient_descent(p, m, sample, y, a)
    """
    
    b = np.ones((m, 1)) # 设置值全为1的矩阵
    x = np.hstack((b, sample)) #矩阵拼接：行拼接vstack, 列拼接hstack
    h = (np.dot(x, p)-y)
    # 求梯度
    gradient = np.dot(x.T, h)
    # 求均方误差
    mse = 1/(2*m)*np.dot(h.T, h)
    n = 0
    #未知MSE最小值可以达到多少，因此因限制斜率接近于平稳
    while np.all(np.absolute(gradient) >= 1e-7):       
        p = p - a*gradient
        h = (np.dot(x, p)-y)
        gradient = (1/m)*np.dot(x.T, h)
        mse = 1/(2*m)*np.dot(h.T, h)
        n = n + 1
    else:
            print("参数：" + str(p))
            print("梯度：" + str(gradient))
            print("均方误差：" + str(mse))
            print("迭代次数：" + str(n))
    return 


########################### 一元散点线性回归
# 输入未知数的初始值
p = np.array([[1, 1]]).T
# 输入样本数
m = 11
# 输入样本x
sample = np.array([[0.3, -0.6, -0.1, 0.1, 0.2, 0.8, -0.3, -0.7, -0.4, 0.4, 0.6]]).T
# 输入y
y = np.array([[0.46, -0.9, -0.21, -0.2, 0.1, 0.6, -0.2, -0.82, -0.34, 0.2, 0.9]]).T
# 步长
a = 0.5

gradient_descent(p, m, sample, y, a)


########################### 波士顿房价
import pandas as pd
from sklearn.datasets import load_boston

# 这里得到的是 scikit-learn 中的 Bunch 对象
boston = load_boston()

# 特征集与目标变量
X = pd.DataFrame(data=boston.data, columns=boston.feature_names)
Y = boston.target

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

X_standard = transf_standard(X)

# 对这个数据集的文字描述
#print(boston.DESCR)         

# 设置参数
# 初始值 几元+截距项
p = np.ones((len(X_standard.columns)+1, 1))
m = len(X_standard.index)
sample = np.array(X_standard)
# sample.shape
y = np.array(Y).reshape(m, 1)
a = 0.01

gradient_descent(p, m, sample, y, a)
