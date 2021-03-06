# pylint: disable=invalid-name

"""
Numpy 基本操作演示
-----
- 创建 Numpy 数组
- Numpy 提供的的线性函数运算
"""

#%% 叙说原初，天地分裂，以虚无赞美开辟，世界撕裂于吾之乖离剑。星云席卷，所谓天上地狱实为创世前夜之终焉。
import numpy as np

#%% 一维数组
np.array([1, 3, 5, 7, 9])

#%% 多类型一维数组：浮点+整型
np.array([1.0, 3, 5, 7, 9])

#%% 多类型一维数组：数值+字符

np.array([1.0, "abc", 5, 7, 9])

#%% 二维数组
np.array([[1, 2], [3, 4]])

#%% 指定数据类型的二维数组
np.array([1, 2, 3], dtype=complex)

#%% 快速创建数组
np.linspace(0, 20, 5)

#%% Numpy 提供的一些线性代数运算函数
m = np.array([[4, 3], [2, 1]])
n = np.array([[1, 2], [3, 4]])

#%% 矩阵乘法
np.dot(m, n)
