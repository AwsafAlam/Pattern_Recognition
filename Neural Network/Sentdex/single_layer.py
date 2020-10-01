import numpy as np

inputs = [1,2,3, 2.5]
weights1 = [0.2,0.8,-0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27,0.17, 0.87]

bias1 = 2.0
bias2 = 3.0
bias3 = 0.5

single_neuron = np.dot(inputs, weights1) + bias1

layer = [ np.dot(inputs, weights1) + bias1,
          np.dot(inputs, weights2) + bias2,
          np.dot(inputs, weights3) + bias3
        ]

## inputs cannot change, but we can tweak the weights and bias.
# the way we tweak the W & B, using back propagation allows us to have a very good neural net.

print(layer)