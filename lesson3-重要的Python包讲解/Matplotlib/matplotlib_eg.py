"""
matplotlib
-----
"""

#%% 叙说原初，天地分裂，以虚无赞美开辟，世界撕裂于吾之乖离剑。星云席卷，所谓天上地狱实为创世前夜之终焉。
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#%% 折线图
x = np.linspace(0, 10, 1000)
y = np.sin(x)
plt.plot(x, y)

#%% 柱状图
x = np.linspace(1, 10, 10)
y = x * x  # 从这里可以看出，一维 ndarray 用 * 是对应元素相乘，计算点积需要用函数
plt.bar(x, y)

#%% 3D
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

# X, Y value
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)    # x-y 平面的网格
R = np.sqrt(X ** 2 + Y ** 2)
# height value
Z = R

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

#%% 设置标题
fig = plt.figure(figsize=[10, 10])
plt.title("这是标题", fontsize=30, color='red')
plt.ylabel("这是 y 轴标签", fontsize=20)
plt.xlabel("这是 x 轴标签", fontsize=15, color='green')
plt.plot(x, y, 'r*-')

#%% 子图
# make up some data in the interval ]0, 1[
x = np.linspace(0, 10, 1000)
y = np.sin(x)

# plot with various axes scales
plt.figure(figsize=[10, 10])

plt.subplot(221)
plt.plot(x, y)

plt.subplot(222)
plt.plot(y, x, 'g*')

#%% 标注
fig = plt.figure()
fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('axes title')

ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')

ax.text(3, 8, 'boxed italics text in data coords', style='italic',
        bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)

ax.text(3, 2, u'unicode: Institut f\374r Festk\366rperphysik')

ax.text(0.95, 0.01, 'colored text in axes coords',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)


ax.plot([2], [1], 'o')
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.axis([0, 10, 0, 10])

plt.show()