# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:08:07 2019

@author: Administrator
"""

# In[1] 画图
import matplotlib.pyplot as plt
import numpy as np
x = range(-10,11,1)
y = [val**2 for val in x]
plt.plot(x,y)
# In[2]
import matplotlib.pyplot as plt
import numpy as np
# 定义函数f(x)=x^2
def f(x):
    return x**2
# 定义导数 h(x)=2*x
def h(x):
    return 2*x
# 设置初始点
x = 10 
# 设置步长
step = 0.01
#记录迭代次数
count = 0
# 定义函数变化
f_1 = f(x)
f_2 = f(x)
# 设置停止条件
while f_1 > 1e-10:
    x = x-step*h(x) #更新x
    change = f(x)
    f_1 = f_2 - change
    f_2 = change
    count = count+1
print(count,'\n',x,'\n',f_2)
