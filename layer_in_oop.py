import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

np.random.seed(0)

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
        self.outputs = probabilities
        
class Loss:
    def calculate(self, outputs, y):
            sample_losses = self.forward(outputs, y)
            data_loss = np.mean(sample_losses)
            return data_loss
        
class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        sample_len = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if(len(np.shape(y_true)) == 1):
            losses = -np.log(y_pred_clipped[range(sample_len), y_true])
        elif(len(np.shape(y_true)) == 2):
            losses = -np.log(np.sum((y_pred_clipped * y_true), axis=1))
        
        return losses
    
class SquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_len = len(y_pred)
        
        if(len(np.shape(y_true)) == 1):
            losses = (y_true - y_pred[range(sample_len), y_true])**2
        if(len(np.shape(y_true)) == 2):
            losses = (y_true - np.sum((y_pred * y_true), axis=1))**2
        
        return losses
            
            
X, y = spiral_data(100, 3)

dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()

dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

dense1.forward(X)
activation1.forward(dense1.outputs)

dense2.forward(activation1.outputs)
activation2.forward(dense2.outputs)
print(activation2.outputs[:5])

loss_function1 = CategoricalCrossEntropy()
loss1 = loss_function1.calculate(activation2.outputs, y)

loss_function2 = SquaredError()
loss2 = loss_function2.calculate(activation2.outputs, y)

print(
    'Categorical Cross Entropy loss: ' + str(loss1) + '\n' +
    'Mean Squared Error loss: ' + str(loss2)
)