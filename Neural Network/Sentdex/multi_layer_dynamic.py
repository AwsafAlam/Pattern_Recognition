import numpy as np

np.random.seed(0)

## We take in a batch of inputs
X = [[1,2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5,2.7, 3.3, -0.8]]

class Layer_dense:
  """
  docstring
  """
  def __init__(self, n_inputs, n_neurons):
    """
    docstring
    """
    # Gausian distribution bounded around 0
    self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))

  
  def forward(self, inputs):
    """
    docstring
    """
    self.output = np.dot(inputs, self.weights) + self.biases

# 4 features, and 5 neurons
layer1 = Layer_dense(4, 5)

# 5 putputs of l1, and 2 neurons
layer2 = Layer_dense(5, 2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
