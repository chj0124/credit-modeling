# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:01:41 2019

@author: Administrator
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn import preprocessing

def liner_Regression(data_x,data_y,learningRate,Loopnum):
    Weight = np.ones([1,data_x.shape[1]])
    baise = np.array([[1]])
    epsilon=1e-2
    
    for num in range(Loopnum):
        WXPlusB = np.dot(data_x,Weight.T) + baise
        
        loss=np.dot((data_y-WXPlusB).T,data_y-WXPlusB)/data_y.shape[0]
        
        w_gradient = -(2/data_x.shape[0])*np.dot((data_y-WXPlusB).T,data_x)
        baise_gradient = -(2/data_x.shape[0])*np.dot((data_y-WXPlusB).T,np.ones(shape=[data_x.shape[0],1]))
        
        Weight=Weight-learningRate*w_gradient     #更新权重
        baise=baise-learningRate*baise_gradient
        
        if np.sqrt((np.dot(w_gradient,w_gradient.T) + np.square(baise_gradient))[0,0]) < epsilon:
            break
    return (Weight,baise,num)

#Boston房价结果
boston = load_boston()
x = pd.DataFrame(data=boston.data, columns=boston.feature_names)
data_x = preprocessing.scale(x) #对data_x作了一个标准化处理
y = boston.target
data_y = y.reshape(506,1)  #把data_y变成2维
learningRate == 0.001
Loopnum == 10000
liner_Regression(data_x,data_y,learningRate,Loopnum)

#误差
theta = T[0]
baise = T[1]
WXPlusB = np.dot(data_x,theta.T) + baise   
(0.5)*np.dot((data_y-WXPlusB).T,data_y-WXPlusB)/data_y.shape[0] #0.5*MSE
