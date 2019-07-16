import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):
    """
    感知机对象
    """
    def __init__(self, learning_rate=0.7, n_iter=100):
        self.learning_rate = learning_rate
        self.iterations = n_iter
        self.weights = np.nan

    def fit(self, x, y):
        self.weights = np.random.rand(x.shape[1])
        for i in range(self.iterations):
            y_hat = 1 * (np.dot(x, self.weights) >= 0)
            self.weights -= self.learning_rate * np.dot(x.T, (y_hat - y).T)
        return self

    def predict(self, x):
        return 1 * (np.dot(x, self.weights) >= 0)


if __name__ == '__main__':
    x = np.array([[0.3, 0.7],
                  [-0.6, 0.3],
                  [-0.1, -0.8],
                  [0.1, -0.45]])
    y = np.array([1, 0, 0, 1])
    p = Perceptron()
    p.fit(x, y)
    print(p.predict(x))
    print(p.weights)
    plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], marker='o')
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker='^')
    plt.plot(np.linspace(x[:, 0].min(), x[:, 0].max(), 100),
             -p.weights[0] / p.weights[1] * np.linspace(x[:, 0].min(), x[:, 0].max(), 100))
    plt.show()
