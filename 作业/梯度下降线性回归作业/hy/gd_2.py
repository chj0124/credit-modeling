import numpy as np
from perceptron import Perceptron 
import matplotlib.pylab as plt


class LinearRegression(Perceptron):
	def __init__(self, learning_rate, n_iter):
		super().__init__()
		self.loss = []

	def loss_func(self, x, y):
		delta = np.dot(x, self.weights) - y
		return 0.5 * np.dot(delta.T, delta) / x.shape[0]

	def fit(self, x, y):
		def gradient(weights, x, y):
			return 1.0 / x.shape[0] * np.dot(x.T, (np.dot(x, weights) - y))

		def distance(v):
			return np.sqrt(np.dot(v.T, v))
		x = np.hstack((x, np.ones((x.shape[0], 1))))
		self.weights = np.random.rand(x.shape[1])

		while distance(gradient(self.weights, x, y)) > 0.01 and \
			  (len(self.loss) < self.iterations):
			self.weights = self.weights - self.learning_rate * gradient(self.weights, x, y)
			# print(self.weights)
			# print(gradient(self.weights, x, y))
			self.loss.append(self.loss_func(x, y))
		return self

	def predict(self, x):
		x = np.hstack((x, np.ones((x.shape[0], 1))))
		return np.dot(x, self.weights).round(2)


if __name__ == '__main__':
	import pandas as pd
	from sklearn.datasets import load_boston

	# 这里得到的是 scikit-learn 中的 Bunch 对象
	boston = load_boston()

	# 特征集与目标变量
	x = pd.DataFrame(data=boston.data, columns=boston.feature_names)
	y = boston.target
	from sklearn.preprocessing import Normalizer
	scaler = Normalizer()
	x = scaler.fit_transform(x)
	# 对这个数据集的文字描述
	# print(boston.DESCR)
	p = LinearRegression(learning_rate=0.4, n_iter=1000)
	p.fit(x, y)
	y_hat = p.predict(x)
	print('weights are:' + str(p.weights))
	print(np.sqrt(sum((y_hat - y)**2)))
	plt.plot(p.loss)
	plt.show()


