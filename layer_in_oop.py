import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

np.random.seed(0)

X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(500, 3)
x = X[:, 1]

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros([1, n_neurons], float)
    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases
    
class ActivationReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
        
class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

plt.style.use('ggplot')
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o')
plt.show()