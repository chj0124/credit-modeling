# _*_ coding: utf-8 _*_

"""
线性模型对象
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

# 波士顿房价数据集
boston = load_boston()
boston_x = pd.DataFrame(data=boston.data, columns=boston.feature_names)
boston_y = boston.target

# 感知机使用数据
data = np.array([[1,  0.3,  0.7],
                 [0, -0.6,  0.3],
                 [0, -0.1, -0.8],
                 [1,  0.1, -0.45]])


def mse(y, t):
    """
    MSE 损失函数

    Param
    -----
    y: array-like, 模型输出
    t: array-like，真实值
    return: 总误差值
    """
    y = np.array(y)
    t = np.array(t)
    return ((y-t)**2).mean()


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


class SGDClassifier(object):
    def __init__(self, beta=0.5, iteration=1000):
        # 偏置值恒为 1，对线性组合的结果的影响用偏置值对应的权值来调整
        self.x = None
        self.t = None
        self.b = 1
        self.weight = None
        self.u = None
        self.y = None
        self.af = None
        self.af_name = None
        self.lf = None
        self.lf_name = None
        self.beta = beta
        self.iteration = iteration

    def set_activation_function(self, name: str):
        # 可用的激活函数列表
        af_list = ['heaviside', 'linear', 'sigmoid']

        # 激活函数名字
        self.af_name = name

        # 单位阶跃函数
        if name == 'heaviside':
            self.af = lambda x: (x >= 0) * 1

        # 线性函数
        if name == 'linear':
            self.af = lambda x: x

        # 逻辑函数
        if name == 'sigmoid':
            self.af = lambda x: sigmoid(x)

        # 错误
        if self.af is None:
            raise UserWarning("此对象内没有内置叫" + name + "的激活函数哦\n",
                              "可用的激活函数有：" + str(af_list))

    def set_loss_function(self, name: str):
        # 可用的损失函数列表
        lf_list = ['simple_minus', 'mse']

        # 损失函数名字
        self.lf_name = name

        # 简单相减
        if name == 'simple_minus':
            self.lf = lambda t, y: t - y

        # mse
        if name == 'mse':
            self.lf = mse

        # 错误
        if self.af is None:
            raise UserWarning("此对象内没有内置叫" + name + "的损失函数哦\n",
                              "可用的损失函数有：" + str(lf_list))

    def get_data(self, x, t):
        self.x = x
        self.t = t
        # 随机初始化权值
        self.weight = np.random.rand(x.shape[1])

    def linear_combine(self):
        self.u = np.dot(self.x, self.weight)
        return self.u

    def output(self):
        self.y = self.af(np.dot(self.x, self.weight))
        return self.y

    def train(self):
        if (self.lf is None) or (self.af is None):
            raise UserWarning("未设定损失函数或激活函数！")
        if (self.x is None) or (self.t is None):
            raise UserWarning("模型未获取数据！")


class Perceptron(SGDClassifier):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.set_activation_function('heaviside')

    # 全样本进行一轮训练
    def train_one_gen(self):
        for i in range(self.x.__len__()):
            e = self.lf(self.t[i], self.output()[i])
            self.weight += self.beta * e * self.x[i]

    def train(self):
        if (self.lf is None) or (self.af is None):
            raise UserWarning("未设定损失函数或激活函数！")
        if (self.x is None) or (self.t is None):
            raise UserWarning("模型未获取数据！")


# 选择要运行的代码块
model_type = None

if model_type == "perceptron":
    # 设定参数
    lm = Perceptron()
    lm.get_data(x=data[:, 1:3], t=data[:, 0])
    lm.set_activation_function(name='heaviside')
    lm.set_loss_function(name='simple_minus')

    # 各个节点的输出值
    lm.linear_combine()
    lm.output()
    lm.lf(lm.t, lm.output())

    # 做一代训练
    lm.train_one_gen()
