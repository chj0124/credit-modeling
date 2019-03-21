from sklearn import datasets
import numpy as np

from perceptron import Perceptron

iris = datasets.load_iris()
# x.columns = iris.feature_names
# y.columns = iris.target_names
x = iris.data
y = iris.target
target = np.zeros([y.shape[0], len(np.unique(y))])
for i in range(len(np.unique(y))):
	target[:, i] = 1 * (y == i)

# train three perceptrons
target_hat = np.zeros(target.shape)
for i in range(target.shape[1]):
	tmp = Perceptron()
	tmp.fit(x, target[:, i])
	target_hat[:, i] = tmp.predict(x)

y_hat = np.argmax(target_hat, axis=1)
print("{:.2f}".format((y == y_hat).sum() / y.shape[0]))  # 0.67
