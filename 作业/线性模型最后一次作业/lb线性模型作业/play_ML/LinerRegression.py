import numpy as np
from math import sqrt
class LinerRegression:

    def __init__(self):
        """初始化LinerRegression模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def _sigmoid(self, t):
        return t # 更改

    def fit(self, X_train, y_train, eta=0.001, n_iters=1e4):
        """ 使用梯度下降法训练LinerRegression模型"""

        def J(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return np.sum((y_hat - y)**2)/len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """预测值"""

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def root_mean_squared_error(self,y_true, y_predict):
        """计算y_true和y_predict之间的MSE"""

        m = np.sum((y_true - y_predict)**2) / len(y_true)
        return sqrt(m)

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的RMSE"""
        y_predict = self.predict(X_test)
        return self.root_mean_squared_error(y_test, y_predict)

    def __repr__(self):
        return "LinerRegression()"
