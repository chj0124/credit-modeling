# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:38:44 2019

@author: Administrator
"""
#整体思路
import numpy as np

#随意的数据
data_x = np.array([[6.3,2.2,7.4],
                    [9.7,-1.7,12.4],
                    [13.7,14.7,4.9]])
data_y = np.array([[4],
                  [-2],
                  [10]])
#起始权重和偏置值
Weight = np.ones([1,data_x.shape[1]])
baise = np.array([[1]])

#多元线性函数
WXPlusB = np.dot(data_x,Weight.T) + baise   #np.dot（A,B）就是矩阵A与B相乘

#损失函数
loss=np.dot((data_y-WXPlusB).T,data_y-WXPlusB)/data_y.shape[0] #shape[0]为行数，shape[1]为列数

#优化函数
w_gradient = -(2/data_x.shape[0])*np.dot((data_y-WXPlusB).T,data_x)
baise_gradient = -(2/data_x.shape[0])*np.dot((data_y-WXPlusB).T,np.ones(shape=[data_x.shape[0],1]))
Weight=Weight-learningRate*w_gradient     #更新权重
baise=baise-learningRate*baise_gradient   #更新偏置值


#整体代码
import numpy as np

def liner_Regression(data_x,data_y,learningRate,Loopnum):
    Weight = np.ones([1,data_x.shape[1]])
    baise = np.array([[1]])
    
    for num in range(Loopnum):
        WXPlusB = np.dot(data_x,Weight.T) + baise
        
        loss=np.dot((data_y-WXPlusB).T,data_y-WXPlusB)/data_y.shape[0]
        
        w_gradient = -(2/data_x.shape[0])*np.dot((data_y-WXPlusB).T,data_x)
        baise_gradient = -(2/data_x.shape[0])*np.dot((data_y-WXPlusB).T,np.ones(shape=[data_x.shape[0],1]))
        
        Weight=Weight-learningRate*w_gradient     #更新权重
        baise=baise-learningRate*baise_gradient
        
        if num%50 == 0:
            print(loss) 
    return (Weight,baise)
        
data_x = np.array([[1,3],
                   [2,4]])
data_y = np.array([[14],
                   [19]])     
learningRate = 0.01
Loopnum = 100000
liner_Regression(data_x,data_y,learningRate,Loopnum)






