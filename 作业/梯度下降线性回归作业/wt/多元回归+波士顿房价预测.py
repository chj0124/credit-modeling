# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:32:55 2019

@author: a4496
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_boston

# 这里得到的是 scikit-learn 中的 Bunch 对象
boston = load_boston()

# 特征集与目标变量
x = pd.DataFrame(data=boston.data, columns=boston.feature_names)
y = boston.target

# 对这个数据集的文字描述
print(boston.DESCR)

# 标准化
def bzh(x):
    x_norm = (x-x.min())/(x.max()-x.min())
    return x_norm

# 多元线性回归
def xxhg(data_x, data_y, learn, num):
    W = np.ones(shape=(1,x.shape[1]))
    b = np.array([[1]])
    loss_history = []
    for i in range(num):
        f = np.dot(data_x, W.T) + b
        loss = np.dot((data_y - f).T,data_y - f)/(data_y.shape[0]*2)
        w_change = -(1/data_x.shape[0])*np.dot((data_y - f).T,x)
        b_change = -1*np.dot((data_y - f).T,np.ones(shape=[data_x.shape[0],1]))/data_x.shape[0]
        W = W-learn*w_change
        b = b-learn*b_change
        loss_history.append(max(max(loss)))
        if num%50==0:
            print(max(max(loss)))
    return W, b, loss, loss_history

if __name__== "__main__":
    x_1 = bzh(x)
    x_2 = np.array(x_1)
    data_x = x_2.reshape(-1,x.shape[1])
    data_y = y.reshape(-1,1)
    result = xxhg(data_x, data_y, learn = 0.001,num = 10000)
    print(result[0],result[1],result[2],result[3])
    plt.plot(result[3])
    
