# %%
#1 编写一个逻辑回归对象
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from play_ML import LogisticRegression # 自写机器学习包
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
import sklearn.preprocessing as pre

x = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
y = cancer.target
min_max_scaler = pre.MinMaxScaler()
X = min_max_scaler.fit_transform(x)
# print(cancer.DESCR)
# 样本分层
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2)
clf =  LogisticRegression.LogisticRegression()
clf.fit(X_train,y_train)
clf.coef_# 系数
clf.intercept_# 间距
clf.score(X_test, y_test) # 分类准确率

#2 激活函数拟合波士顿房价
# %%
import pandas as pd
from sklearn.datasets import load_boston
from play_ML import LinerRegression # 自写机器学习包
from sklearn.model_selection import train_test_split
boston = load_boston()
import sklearn.preprocessing as pre

x = pd.DataFrame(data=boston.data, columns=boston.feature_names)
y = boston.target
min_max_scaler = pre.MinMaxScaler()
X = min_max_scaler.fit_transform(x)
# print(boston.DESCR)
# 样本分层
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2)
clf =  LinerRegression.LinerRegression()
clf.fit(X_train,y_train)
clf.coef_# 系数
clf.intercept_# 间距
clf.score(X_test, y_test) # RMSE
