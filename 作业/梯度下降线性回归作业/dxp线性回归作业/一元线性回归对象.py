# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:58:46 2019

@author: Administrator
"""

#一员线性回归
import numpy as np 
import pandas as pd
from sklearn.datasets import load_boston


#模型
def model(a,b,x):
    return a*x+b

a=0
b=0
c=0
x=np.array([1,2,3,4,5,6,7])
y=np.array([3,5,7,9,11,13,15])
while True:
    c +=1
    n=len(x)
    alpha = 0.01
    y_hat = model(a,b,x)
    da = 1/n*((y_hat-y)*x).sum()    #da,db为损失函数的变化量，为a,b的斜率
    db = 1/n*((y_hat-y).sum())
    a = a - alpha*da
    b = b - alpha*db
    epsilon = 1e-1
    if np.sqrt(np.square(da) + np.square(db)) < epsilon :   
        break 
    
print("第 %s 迭代后，一元线性函数为 y = %s * x + %s" % (c,a,b))

#参考文献：https://blog.csdn.net/juwikuang/article/details/78420337








