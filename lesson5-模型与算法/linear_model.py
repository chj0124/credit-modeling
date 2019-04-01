# _*_ coding: utf-8 _*_

"""
线性模型对象
"""

import numpy as np
from numpy import ndarray

data = np.array([[1,  0.3,  0.7],
                 [0, -0.6,  0.3],
                 [0, -0.1, -0.8],
                 [1,  0.1, -0.45]])


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def mbe(y: ndarray, t: ndarray):
    return (t - y) / len(t)


class SGDClassifier(object):
    def __init__(self, af=None, lf=None):
        # 偏置值恒为 1，对线性组合的结果的影响用偏置值对应的权值来调整
        self.b = 1
        self.weight = None
        self._af = af
        self.af_name = None
        self._lf = lf
        self.lf_name = None

    def set_activation_function(self, name: str):
        # 可用的激活函数列表
        af_list = ['heaviside', 'linear', 'sigmoid']

        # 激活函数名字
        self.af_name = name

        # 单位阶跃函数
        if name == 'heaviside':
            self._af = lambda x: (x >= 0) * 1

        # 线性函数
        if name == 'linear':
            self._af = lambda x: x

        # 逻辑函数
        if name == 'sigmoid':
            self._af = lambda x: sigmoid(x)

        # 错误
        if self._af is None:
            raise UserWarning("此对象内没有内置叫" + name + "的激活函数哦\n",
                              "可用的激活函数有：" + str(af_list))

    def set_loss_function(self, name: str):
        # 可用的损失函数列表
        lf_list = ['mbe']

        # 平均偏差误差 mean bias error
        if name == 'mbe':
            self._lf = mbe

        # 错误
        if self._af is None:
            raise UserWarning("此对象内没有内置叫" + name + "的损失函数哦\n",
                              "可用的损失函数有：" + str(lf_list))

    def get_data(self, x, y):
        pass

    def linear_combine(self):
        pass

    def train(self):
        if (self._lf is None) or (self._af is None):
            raise UserWarning("先设定损失函数与激活函数！")


class Perceptron(SGDClassifier):
    def __init__(self):
        super().__init__()
        self.set_activation_function('heaviside')

    def set_loss_function(self, name='simple minus'):
        self.lf_name = name
        self._lf = lambda t, y: t - y
