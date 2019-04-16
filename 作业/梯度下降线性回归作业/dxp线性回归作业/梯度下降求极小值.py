# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:20:21 2019

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
plot_x=np.linspace(-10,10,100)   #在-5到5之间等距的生成100个数
plot_y=(plot_x)**2	   # 同时根据plot_x来生成plot_y
plt.plot(plot_x,plot_y)
plt.show()

###定义一个求二次函数导数的函数dJ
def dJ(x):
    return 2*x

###定义一个求函数值的函数J
def J(x):
    try:
        return x**2
    except:
        return float('inf')

x=10
eta=0.1
epsilon=1e-8
history_x=[x]
while True:
    gradient=dJ(x)
    last_x=x
    x=x-eta*gradient
    history_x.append(x)
    if (abs(J(last_x)-J(x)) <epsilon):
        break

print(history_x)
plt.plot(plot_x,plot_y)
plt.plot(np.array(history_x),J(np.array(history_x)),color='r',marker='*')
plt.show()


#参考文献：https://blog.csdn.net/qq_39577552/article/details/82918479









